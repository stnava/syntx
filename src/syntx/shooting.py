import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .syn import (
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    separable_gaussian_filter,
    local_ncc_loss_nd as lncc_loss_nd,
    grid_sample_nd,
    _spatial_jacobian_nd,
    HierarchicalAffine,
    grid_to_physical_affine_torch
)
from .tvf import extract_image_metadata, normalize_tensor


def epdiff_advection_nd(p, v):
    """
    Computes the coadjoint action ad_v^* p for EPDiff (Euler-Poincaré Differential Equation).
    
    ad_v^* p = (Dp) v + (Dv)^T p + p (div v)
    
    Args:
        p: Momentum field tensor (1, *spatial, dim)
        v: Velocity field tensor (1, *spatial, dim)
        
    Returns:
        ad_v_star_p: Coadjoint action tensor (1, *spatial, dim)
    """
    dim = p.shape[-1]
    
    # Compute spatial Jacobians: Dp and Dv have shape (1, *spatial, dim, dim)
    # (Dp)[..., i, j] = d(p_i) / d(x_j)
    Dp = _spatial_jacobian_nd(p).squeeze(-2) if p.ndim == dim + 3 else _spatial_jacobian_nd(p)
    Dv = _spatial_jacobian_nd(v).squeeze(-2) if v.ndim == dim + 3 else _spatial_jacobian_nd(v)
    
    # 1. Advection term: (Dp) v -> sum_j (d p_i / d x_j) * v_j
    term_advection = torch.einsum('...ij,...j->...i', Dp, v)
    
    # 2. Stretching term: (Dv)^T p -> sum_j (d v_j / d x_i) * p_j
    term_stretching = torch.einsum('...ji,...j->...i', Dv, p)
    
    # 3. Expansion term: p * div(v) where div(v) = sum_j (d v_j / d x_j)
    div_v = torch.diagonal(Dv, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    term_expansion = p * div_v
    
    ad_v_star_p = term_advection + term_stretching + term_expansion
    return ad_v_star_p


class GeodesicShootingModel(nn.Module):
    """
    Euler-Poincaré Differential Equation (EPDiff) Geodesic Shooting Registration Model.
    
    Parameterizes ONLY ONE initial momentum vector field p0(x) at t=0.
    Shoots p0 forward in time via EPDiff:
      dp/dt = -ad_v^* p
      v(t) = separable_gaussian_filter(p(t), fluid_sigma)
      
    Guarantees strict geodesic paths in Diff(Omega).
    """
    def __init__(
        self,
        dim=2,
        image_shape=None,
        n_time_steps=8,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=1.0,
        fixed_image=None
    ):
        super().__init__()
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image, dim=dim)
        if image_shape is None:
            image_shape = f_shape if f_shape is not None else (128,) * dim
        if spacing is None and f_spacing is not None:
            spacing = f_spacing
        if origin is None and f_origin is not None:
            origin = f_origin
        if direction is None and f_direction is not None:
            direction = f_direction
            
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.n_time_steps = n_time_steps
        self.spacing = spacing if spacing is not None else [1.0] * dim
        self.origin = origin if origin is not None else [0.0] * dim
        self.direction = direction if direction is not None else np.eye(dim).tolist()
        self.fluid_sigma = fluid_sigma
        
        # Hierarchical Affine Pre-Alignment
        self.affine = HierarchicalAffine(dim=dim, transform_type='Affine')
        
        # Single initial momentum parameter at t=0: shape (1, *image_shape, dim)
        self.p0 = nn.Parameter(torch.zeros(1, *self.image_shape, self.dim))

    def shoot(self, p0=None, n_steps=None, fluid_sigma=None, image_shape=None, sigma_mode='voxel'):
        """
        Integrates initial momentum p0 forward from t=0 to t=1 using EPDiff.
        """
        if p0 is None:
            p0 = self.p0
        if n_steps is None:
            n_steps = self.n_time_steps
        if fluid_sigma is None:
            fluid_sigma = self.fluid_sigma
            
        device = p0.device
        dtype = p0.dtype
        dt = 1.0 / float(n_steps)
        
        target_shape = tuple(image_shape) if image_shape is not None else tuple(p0.shape[1:-1])
        
        phys_grid = get_physical_grid_torch(
            target_shape, self.spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(list(reversed(self.spacing)), device=device, dtype=dtype)
        origin_t = torch.tensor(list(reversed(self.origin)), device=device, dtype=dtype)
        direction_t = torch.tensor(np.asarray(self.direction)[::-1, ::-1].copy(), device=device, dtype=dtype)
        
        p_t = p0.clone()
        phi_t = phys_grid.clone()
        
        p_history = [p_t]
        v_history = []
        
        p_spacing = [sp * (float(orig_s) / float(curr_s)) for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)] if sigma_mode == 'physical' else None

        for step in range(n_steps):
            fluid_sigma_val = math.sqrt(fluid_sigma) if fluid_sigma > 0 else 0.0
            v_t = separable_gaussian_filter(p_t, sigma=fluid_sigma_val, spacing=p_spacing, sigma_mode=sigma_mode)
            v_history.append(v_t)
            
            dp_dt = -epdiff_advection_nd(p_t, v_t)
            
            dp_norm = torch.sqrt(torch.sum(dp_dt ** 2, dim=-1, keepdim=True) + 1e-12)
            max_dp = 2.0
            dp_dt = torch.where(dp_norm > max_dp, dp_dt * (max_dp / (dp_norm + 1e-8)), dp_dt)
            
            p_t = p_t + dt * dp_dt
            p_history.append(p_t)
            
            phi_norm = physical_to_normalized_torch_cached(
                phi_t, shape_t, spacing_t, origin_t, direction_t
            )
            v_sampled_cf = grid_sample_nd(v_t.movedim(-1, 1), phi_norm, mode='bilinear', padding_mode='border')
            v_sampled = v_sampled_cf.movedim(1, -1)
            phi_t = phi_t + dt * v_sampled
            
        disp_fwd = phi_t - phys_grid
        return disp_fwd, {"p_history": p_history, "v_history": v_history}

    def forward(self, fixed_image, moving_image, p0=None, fluid_sigma=None, lncc_window_size=5, affine_params=None, reg_weight=0.0, sigma_mode='voxel'):
        """
        Forward pass computing standard LNCC loss at t=1 with optional Affine pre-alignment.
        """
        target_shape = tuple(fixed_image.shape[2:])
        disp_fwd, _ = self.shoot(p0=p0, fluid_sigma=fluid_sigma, image_shape=target_shape, sigma_mode=sigma_mode)
        
        device = fixed_image.device
        dtype = fixed_image.dtype
        
        phys_grid = get_physical_grid_torch(
            target_shape, self.spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(list(reversed(self.spacing)), device=device, dtype=dtype)
        origin_t = torch.tensor(list(reversed(self.origin)), device=device, dtype=dtype)
        direction_t = torch.tensor(np.asarray(self.direction)[::-1, ::-1].copy(), device=device, dtype=dtype)
        
        if affine_params is not None:
            T_grid = affine_params
        else:
            T_grid = self.affine.get_matrix()
            
        from .syn import grid_to_physical_affine_torch
        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, target_shape, self.spacing, self.origin, self.direction,
            target_shape, self.spacing, self.origin, self.direction
        )
        M_phys = M_phys.to(device=device, dtype=dtype)
        t_phys = t_phys.to(device=device, dtype=dtype)
        
        phi_moving_affine = (phys_grid + disp_fwd) @ M_phys.t() + t_phys
        phi_norm = physical_to_normalized_torch_cached(
            phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = grid_sample_nd(moving_image, phi_norm, mode='bilinear', padding_mode='zeros')
        
        fl_sig = self.fluid_sigma if fluid_sigma is None else fluid_sigma
        fl_sig_val = math.sqrt(fl_sig) if fl_sig > 0 else 0.0
        p_spacing = [sp * (float(orig_s) / float(curr_s)) for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)] if sigma_mode == 'physical' else None
        v0 = separable_gaussian_filter(self.p0 if p0 is None else p0, sigma=fl_sig_val, spacing=p_spacing, sigma_mode=sigma_mode)
        energy_reg = 0.5 * torch.mean((self.p0 if p0 is None else p0) * v0)
        
        lncc_val = lncc_loss_nd(fixed_image, moving_warped, window_size=lncc_window_size)
        return lncc_val + reg_weight * energy_reg

    def _resize_p0(self, new_shape, device=None, dtype=None):
        """
        Resize the initial momentum parameter p0 to a new spatial shape using trilinear/bilinear
        interpolation. Preserves learned momentum field when transitioning between
        pyramid-proportional grid resolutions.
        """
        new_shape = tuple(new_shape)
        old_shape = tuple(self.p0.shape[1:-1])
        if new_shape == old_shape:
            return

        with torch.no_grad():
            old_p0 = self.p0.data
            if self.dim == 3:
                old_cf = old_p0.permute(0, 4, 1, 2, 3)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='trilinear', align_corners=False)
                new_p0 = new_cf.permute(0, 2, 3, 4, 1)
            else:
                old_cf = old_p0.permute(0, 3, 1, 2)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='bilinear', align_corners=False)
                new_p0 = new_cf.permute(0, 2, 3, 1)

            if device is not None:
                new_p0 = new_p0.to(device=device)
            if dtype is not None:
                new_p0 = new_p0.to(dtype=dtype)

            self.p0 = nn.Parameter(new_p0.contiguous())

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[100, 100, 50],
        affine_epochs=100,
        lr=2.0,
        fluid_sigma=1.0,
        lncc_window_size=5,
        reg_weight=0.0,
        verbose=False,
        **kwargs
    ):
        """
        Multi-resolution EPDiff Geodesic Shooting optimization with Affine pre-alignment.
        """
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image, dim=self.dim)
        if f_spacing is not None: self.spacing = f_spacing
        if f_origin is not None: self.origin = f_origin
        if f_direction is not None: self.direction = f_direction
        if f_shape is not None and self.image_shape != f_shape:
            self.image_shape = f_shape
            device = self.p0.device if hasattr(self, 'p0') else torch.device('cpu')
            self.p0 = nn.Parameter(torch.zeros(1, *self.image_shape, self.dim, device=device))

        if hasattr(fixed_image, 'numpy'):
            fixed_tensor = normalize_tensor(torch.from_numpy(fixed_image.numpy()).float())
        elif isinstance(fixed_image, torch.Tensor):
            fixed_tensor = fixed_image
        else:
            fixed_tensor = torch.from_numpy(np.array(fixed_image)).float()

        while fixed_tensor.dim() < self.dim + 2:
            fixed_tensor = fixed_tensor.unsqueeze(0)

        if hasattr(moving_image, 'numpy'):
            moving_tensor = normalize_tensor(torch.from_numpy(moving_image.numpy()).float())
        elif isinstance(moving_image, torch.Tensor):
            moving_tensor = moving_image
        else:
            moving_tensor = torch.from_numpy(np.array(moving_image)).float()

        while moving_tensor.dim() < self.dim + 2:
            moving_tensor = moving_tensor.unsqueeze(0)

        device = self.p0.device
        fixed_tensor = fixed_tensor.to(device=device)
        moving_tensor = moving_tensor.to(device=device)

        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'

        fluid_sigma_input = kwargs.get('flow_sigma', kwargs.get('fluid_sigma', fluid_sigma))
        sigma_mode = kwargs.get('sigma_mode', 'voxel')

        # 1. Affine Pre-Alignment Stage
        if affine_epochs > 0:
            fixed_spacing_t = torch.tensor(self.spacing, device=device, dtype=torch.float32)
            fixed_origin_t = torch.tensor(self.origin, device=device, dtype=torch.float32)
            fixed_direction_t = torch.tensor(self.direction, device=device, dtype=torch.float32)

            dim = self.dim
            Nx_t = torch.tensor(fixed_tensor.shape[2:], device=device, dtype=torch.float32)
            Ny_t = torch.tensor(moving_tensor.shape[2:], device=device, dtype=torch.float32)
            
            Dx_t, Sx_t, Ox_t = fixed_direction_t, fixed_spacing_t, fixed_origin_t
            Dy_t, Sy_t, Oy_t = fixed_direction_t, fixed_spacing_t, fixed_origin_t

            com_fixed_fov = Ox_t + Dx_t @ (Sx_t * ((Nx_t - 1) / 2.0))
            com_moving_fov = Oy_t + Dy_t @ (Sy_t * ((Ny_t - 1) / 2.0))

            t_fov = com_moving_fov - com_fixed_fov

            H_x = torch.eye(dim + 1, device=device, dtype=torch.float32)
            H_x[:dim, :dim] = Dx_t @ torch.diag(Sx_t) @ torch.diag((Nx_t - 1) / 2.0)
            H_x[:dim, dim] = com_fixed_fov

            H_y = torch.eye(dim + 1, device=device, dtype=torch.float32)
            H_y[:dim, :dim] = Dy_t @ torch.diag(Sy_t) @ torch.diag((Ny_t - 1) / 2.0)
            H_y[:dim, dim] = com_moving_fov

            T_phys = torch.eye(dim + 1, device=device, dtype=torch.float32)
            T_phys[:dim, dim] = t_fov

            T_init = torch.inverse(H_y) @ T_phys @ H_x
            self.affine.T_init = T_init

            if verbose: print("[GeodesicShooting] Optimizing Affine Pre-Alignment...")
            opt_aff = torch.optim.Adam(self.affine.parameters(), lr=1e-2)
            for ep in range(affine_epochs):
                opt_aff.zero_grad()
                loss_aff = self.forward(fixed_tensor, moving_tensor, p0=torch.zeros_like(self.p0), fluid_sigma=fluid_sigma_input, lncc_window_size=lncc_window_size, reg_weight=reg_weight, sigma_mode=sigma_mode)
                loss_aff.backward()
                opt_aff.step()
                self.affine.clamp_parameters()

        # 2. Deformable Geodesic Shooting Stage
        max_p0_shape = self.image_shape

        for level, epochs in zip(levels, epochs_per_level):
            if epochs <= 0:
                continue

            curr_p0_shape = tuple(max(8, s // level) for s in max_p0_shape)
            prev_p0_shape = tuple(self.p0.shape[1:-1])

            if curr_p0_shape != prev_p0_shape:
                self._resize_p0(curr_p0_shape, device, dtype=torch.float32)

            opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'adam')).lower()
            trust_coeff = float(kwargs.get('trust_coefficient', kwargs.get('trust', 0.05)))
            
            if opt_type == 'lars':
                from .tvf import LARS
                optimizer = LARS([self.p0], lr=lr, trust_coefficient=trust_coeff)
            elif opt_type == 'sgd':
                optimizer = torch.optim.SGD([self.p0], lr=lr)
            elif opt_type == 'cfl':
                optimizer = None
            else:
                optimizer = torch.optim.Adam([self.p0], lr=lr)

            if level > 1:
                down_shape = [max(8, s // level) for s in self.image_shape]
                curr_fixed = F.interpolate(fixed_tensor, size=down_shape, mode=interp_mode, align_corners=False)
                curr_moving = F.interpolate(moving_tensor, size=down_shape, mode=interp_mode, align_corners=False)
            else:
                curr_fixed = fixed_tensor
                curr_moving = moving_tensor

            for epoch in range(epochs):
                if optimizer is not None:
                    optimizer.zero_grad()
                loss = self.forward(curr_fixed, curr_moving, p0=self.p0, fluid_sigma=fluid_sigma_input, lncc_window_size=lncc_window_size, reg_weight=reg_weight, sigma_mode=sigma_mode)
                if torch.isnan(loss) or torch.isinf(loss):
                    if verbose: print(f"[GeodesicShooting] Level {level} NaN loss detected at epoch {epoch}, stopping level.")
                    break
                if optimizer is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([self.p0], max_norm=5.0)
                    optimizer.step()
                elif opt_type == 'cfl':
                    self.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        cfl_step = float(kwargs.get('cfl_voxels', kwargs.get('cfl', kwargs.get('step', lr))))
                        grad_p0 = self.p0.grad
                        max_norm = torch.max(torch.sqrt(torch.sum(grad_p0 ** 2, dim=-1)))
                        if max_norm > 1e-12:
                            curr_spacing_level = [
                                sp * (float(orig_s) / float(curr_s))
                                for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, curr_fixed.shape[2:])
                            ]
                            spacing_rev = tuple(reversed(curr_spacing_level))
                            sp_tensor = torch.tensor(spacing_rev, device=device, dtype=torch.float32)
                            step_update = (cfl_step / max_norm) * grad_p0 * sp_tensor
                            self.p0.data.sub_(step_update)

        final_p0_shape = tuple(self.p0.shape[1:-1])
        if final_p0_shape != max_p0_shape:
            self._resize_p0(max_p0_shape, device, dtype=torch.float32)

        return self

    @torch.no_grad()
    def get_forward_warp(self, image_shape=None):
        """
        Returns displacement field integrating initial momentum p0 from t=0 to t=1 in physical space.
        """
        disp_fwd, _ = self.shoot(image_shape=image_shape)
        return disp_fwd

    @torch.no_grad()
    def get_warped_image(self, moving_image):
        """
        Applies forward displacement warp to moving_image and returns the warped image tensor.
        """
        phi_fwd = self.get_forward_warp()
        device = moving_image.device
        dtype = moving_image.dtype
        target_shape = tuple(moving_image.shape[2:])
        phys_grid = get_physical_grid_torch(
            target_shape, self.spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )
        phi_phys = phys_grid + phi_fwd
        phi_norm = physical_to_normalized_torch_cached(
            phi_phys, target_shape, self.spacing, self.origin, self.direction
        )
        return grid_sample_nd(moving_image, phi_norm, mode='bilinear', padding_mode='zeros')

