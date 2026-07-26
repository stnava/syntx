import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .syn import (
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    separable_gaussian_filter,
    HierarchicalAffine,
    grid_to_physical_affine_torch,
    local_ncc_loss_nd as lncc_loss_nd,
    grid_sample_nd
)

class TVFModel(nn.Module):
    """
    Time-Varying Velocity Field (TVF) Registration Model.
    """
    def __init__(
        self,
        dim,
        image_shape,
        velocity_shape,
        n_time_steps=4,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=1.0,
        transform_type='Affine',
        solver='rk4',
        integration_steps_per_interval=4
    ):
        super().__init__()
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps
        
        self.spacing = spacing if spacing is not None else [1.0] * dim
        self.origin = origin if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction
        else:
            self.direction = np.eye(dim).tolist()
            
        self.fluid_sigma = fluid_sigma
        self.solver = solver
        self.integration_steps_per_interval = integration_steps_per_interval
        
        # Velocity field parameter: (T, 1, *velocity_shape, dim)
        self.velocity = nn.Parameter(torch.zeros(n_time_steps, 1, *self.velocity_shape, self.dim))
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)

    def _get_metadata_tensors(self, device, dtype):
        """Helper to get spatial metadata as tensors for cached normalized coordinates."""
        spacing_rev = tuple(reversed(self.spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
        
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(self.image_shape), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        return shape_t, spacing_t, origin_t, direction_t

    def interpolate_velocity(self, t, velocity_cf):
        """
        Linearly interpolate between discrete velocity keyframes.
        
        Args:
            t: Continuous time in [0, 1]
            velocity_cf: Velocity in channels-first format (T, 1, dim, *velocity_shape)
            
        Returns:
            Velocity at time t (1, dim, *velocity_shape)
        """
        T = self.n_time_steps
        if T == 1:
            return velocity_cf[0]
            
        t_scaled = t * (T - 1)
        idx_lower = math.floor(t_scaled)
        idx_upper = math.ceil(t_scaled)
        
        if idx_lower == idx_upper:
            idx_lower = max(0, min(T - 1, idx_lower))
            return velocity_cf[idx_lower]
            
        idx_lower = max(0, min(T - 1, idx_lower))
        idx_upper = max(0, min(T - 1, idx_upper))
        
        weight_upper = t_scaled - idx_lower
        weight_lower = 1.0 - weight_upper
        
        return weight_lower * velocity_cf[idx_lower] + weight_upper * velocity_cf[idx_upper]

    def upsample_velocity(self, v_coarse_cf, target_shape):
        """
        Spatially upsample velocity from coarse to fine resolution using trilinear/bilinear interpolation.
        Since displacements are in physical normalized space, no scaling of values is needed.
        
        Args:
            v_coarse_cf: Velocity in channels-first format (1, dim, *velocity_shape)
            target_shape: Spatial shape of the target
            
        Returns:
            Upsampled velocity (1, dim, *target_shape)
        """
        if tuple(v_coarse_cf.shape[2:]) == tuple(target_shape):
            return v_coarse_cf
            
        mode = 'trilinear' if self.dim == 3 else 'bilinear'
        
        v_fine_cf = F.interpolate(
            v_coarse_cf,
            size=target_shape,
            mode=mode,
            align_corners=True
        )
        return v_fine_cf

    def integrate(self, t_start, t_end, n_steps=None, image_shape=None):
        """
        Integrates the velocity field ODE from t_start to t_end.
        
        Args:
            t_start: Start time in [0, 1]
            t_end: End time in [0, 1]
            n_steps: Number of integration steps
            image_shape: Optional spatial shape override for multi-resolution levels
            
        Returns:
            Displacement field in physical space (1, *image_shape, dim)
        """
        device = self.velocity.device
        dtype = self.velocity.dtype
        
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        
        if n_steps is None:
            n_steps = self.n_time_steps * self.integration_steps_per_interval
            
        dt = (t_end - t_start) / max(1, n_steps)
        
        # Calculate spacing for current shape
        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]
        
        # Create initial identity grid in physical space
        phys_grid = get_physical_grid_torch(
            target_shape, curr_spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )  # Already (1, *target_shape, dim)
        
        phi_t = phys_grid.clone()
        
        # Convert velocity to channels-first: (T, 1, dim, *velocity_shape)
        if self.dim == 2:
            velocity_cf = self.velocity.permute(0, 1, 4, 2, 3)
        else:
            velocity_cf = self.velocity.permute(0, 1, 5, 2, 3, 4)
            
        # Metadata tensors for normalized coordinate cached lookup
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
        
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
            
        for step in range(n_steps):
            t_current = t_start + step * dt
            
            if self.solver == 'euler':
                v_cf = self.interpolate_velocity(t_current, velocity_cf)
                v_fine_cf = self.upsample_velocity(v_cf, target_shape)
                
                phi_norm = physical_to_normalized_torch_cached(
                    phi_t, shape_t, spacing_t, origin_t, direction_t
                )
                
                v_sampled_cf = grid_sample_nd(v_fine_cf, phi_norm, mode='bilinear')
                if self.dim == 2:
                    v_sampled = v_sampled_cf.permute(0, 2, 3, 1)
                else:
                    v_sampled = v_sampled_cf.permute(0, 2, 3, 4, 1)
                phi_t = phi_t + v_sampled * dt
                
            elif self.solver == 'rk4':
                def eval_v(t, current_phi):
                    v_cf_t = self.interpolate_velocity(t, velocity_cf)
                    v_fine_cf_t = self.upsample_velocity(v_cf_t, target_shape)
                    
                    phi_norm_t = physical_to_normalized_torch_cached(
                        current_phi, shape_t, spacing_t, origin_t, direction_t
                    )
                    v_sampled_cf = grid_sample_nd(v_fine_cf_t, phi_norm_t, mode='bilinear')
                    if self.dim == 2:
                        return v_sampled_cf.permute(0, 2, 3, 1)
                    else:
                        return v_sampled_cf.permute(0, 2, 3, 4, 1)
                    
                k1 = eval_v(t_current, phi_t)
                k2 = eval_v(t_current + 0.5 * dt, phi_t + 0.5 * dt * k1)
                k3 = eval_v(t_current + 0.5 * dt, phi_t + 0.5 * dt * k2)
                k4 = eval_v(t_current + dt, phi_t + dt * k3)
                
                phi_t = phi_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")
                
        return phi_t - phys_grid

    def forward(self, fixed_image, moving_image):
        """
        Midpoint-symmetric registration forward pass.
        Returns LNCC loss.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        target_shape = tuple(fixed_image.shape[2:])
        
        # Integrate backward and forward from midpoint at target resolution
        phi_mid_to_fixed = self.integrate(0.5, 0.0, image_shape=target_shape)
        phi_mid_to_moving = self.integrate(0.5, 1.0, image_shape=target_shape)
        
        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]
        
        phys_grid = get_physical_grid_torch(
            target_shape, curr_spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )  # Already (1, *target_shape, dim)
        
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
        
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        # Warp fixed to midpoint (Single Interpolation Policy)
        phi_fixed = phys_grid + phi_mid_to_fixed
        phi_fixed_norm = physical_to_normalized_torch_cached(
            phi_fixed, shape_t, spacing_t, origin_t, direction_t
        )
        fixed_warped = grid_sample_nd(fixed_image, phi_fixed_norm, mode='bilinear')
        
        # Warp moving to midpoint (applying affine and displacement together)
        phi_moving = phys_grid + phi_mid_to_moving
        
        T_grid = self.affine.get_matrix()
        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, target_shape, curr_spacing, self.origin, self.direction,
            target_shape, curr_spacing, self.origin, self.direction
        )
        
        coord_perm = list(range(self.dim - 1, -1, -1))
        perm_idx = torch.tensor(coord_perm, device=device)
        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
        t_phys_zyx = t_phys[perm_idx]
        
        phi_moving_affine = phi_moving @ M_phys_zyx.t() + t_phys_zyx
        
        phi_moving_norm = physical_to_normalized_torch_cached(
            phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = grid_sample_nd(moving_image, phi_moving_norm, mode='bilinear')
        
        # LNCC with variance floor and Cauchy-Schwarz clamping (built into local_ncc_loss_nd)
        loss = lncc_loss_nd(fixed_warped, moving_warped, window_size=9)
        return loss

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[100, 100, 50],
        affine_epochs=100,
        similarity_metric='lncc',
        lncc_radius=4,
        lr=0.1,
        reg_weight=0.005,
        verbose=False,
        fixed_spacing=None,
        fixed_origin=None,
        fixed_direction=None,
        moving_spacing=None,
        moving_origin=None,
        moving_direction=None,
        **kwargs
    ):
        """
        Multi-resolution optimization.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        
        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction
            
        # Optimize affine pre-alignment first
        if affine_epochs > 0:
            if verbose: print("Optimizing affine pre-alignment...")
            optimizer_aff = torch.optim.Adam(self.affine.parameters(), lr=1e-3)
            
            for epoch in range(affine_epochs):
                optimizer_aff.zero_grad()
                
                phys_grid = get_physical_grid_torch(
                    self.image_shape, self.spacing, self.origin, self.direction,
                    device=device, dtype=dtype
                )
                
                T_grid = self.affine.get_matrix()
                M_phys, t_phys = grid_to_physical_affine_torch(
                    T_grid, self.image_shape, self.spacing, self.origin, self.direction,
                    self.image_shape, self.spacing, self.origin, self.direction
                )
                
                coord_perm = list(range(self.dim - 1, -1, -1))
                perm_idx = torch.tensor(coord_perm, device=device)
                M_phys_zyx = M_phys[perm_idx][:, perm_idx]
                t_phys_zyx = t_phys[perm_idx]
                
                phi_moving_affine = phys_grid @ M_phys_zyx.t() + t_phys_zyx
                
                shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(device, dtype)
                phi_moving_norm = physical_to_normalized_torch_cached(
                    phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = grid_sample_nd(moving_image, phi_moving_norm, mode='bilinear')
                
                loss = lncc_loss_nd(fixed_image, moving_warped, window_size=2*lncc_radius+1)
                loss.backward()
                optimizer_aff.step()
                self.affine.clamp_parameters()
                
        # Optimize velocity field across pyramid levels
        if verbose: print("Optimizing TVF...")
        optimizer = torch.optim.Adam([self.velocity], lr=lr)
        sigma_voxel = math.sqrt(self.fluid_sigma)
        
        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        
        for level, epochs in zip(levels, epochs_per_level):
            if epochs <= 0:
                continue
            if verbose: print(f"Level {level}: {epochs} epochs")
            
            if level > 1:
                down_shape = [max(8, s // level) for s in self.image_shape]
                curr_fixed = F.interpolate(fixed_image, size=down_shape, mode=interp_mode, align_corners=True)
                curr_moving = F.interpolate(moving_image, size=down_shape, mode=interp_mode, align_corners=True)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                sim_loss = self.forward(curr_fixed, curr_moving)
                kinetic = torch.mean(self.velocity ** 2)
                total_loss = sim_loss + reg_weight * kinetic
                total_loss.backward()
                
                # Fluid regularization (smoothing velocity gradients)
                with torch.no_grad():
                    if self.velocity.grad is not None:
                        grad = self.velocity.grad.clone()
                        if self.dim == 2:
                            grad_cf = grad.permute(0, 1, 4, 2, 3) # (T, 1, 2, H, W)
                        else:
                            grad_cf = grad.permute(0, 1, 5, 2, 3, 4) # (T, 1, 3, D, H, W)
                            
                        # Smooth over spatial dims
                        for t in range(self.n_time_steps):
                            smoothed_grad = separable_gaussian_filter(
                                grad_cf[t], sigma=sigma_voxel, spacing=None, sigma_mode='voxel'
                            )
                            grad_cf[t] = smoothed_grad
                            
                        if self.dim == 2:
                            self.velocity.grad = grad_cf.permute(0, 1, 3, 4, 2)
                        else:
                            self.velocity.grad = grad_cf.permute(0, 1, 3, 4, 5, 2)
                            
                optimizer.step()

    def get_forward_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=0 to t=1 in physical space.
        """
        return self.integrate(0.0, 1.0, image_shape=image_shape)
        
    def get_inverse_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=1 to t=0 in physical space.
        """
        return self.integrate(1.0, 0.0, image_shape=image_shape)

