import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .syn import (
    get_physical_grid_torch,
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

    def shoot(self, p0=None, n_steps=None, fluid_sigma=None, image_shape=None):
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
        
        for step in range(n_steps):
            v_t = separable_gaussian_filter(p_t, sigma=fluid_sigma, spacing=None, sigma_mode='voxel')
            v_history.append(v_t)
            
            dp_dt = -epdiff_advection_nd(p_t, v_t)
            
            dp_norm = torch.norm(dp_dt, dim=-1, keepdim=True)
            max_dp = 2.0
            dp_dt = torch.where(dp_norm > max_dp, dp_dt * (max_dp / (dp_norm + 1e-8)), dp_dt)
            
            p_t = p_t + dt * dp_dt
            p_history.append(p_t)
            
            phi_norm = physical_to_normalized_torch_cached(
                phi_t, shape_t, spacing_t, origin_t, direction_t
            )
            v_sampled_cf = grid_sample_nd(v_t.movedim(-1, 1), phi_norm, mode='bilinear')
            v_sampled = v_sampled_cf.movedim(1, -1)
            phi_t = phi_t + dt * v_sampled
            
        disp_fwd = phi_t - phys_grid
        return disp_fwd, {"p_history": p_history, "v_history": v_history}

    def forward(self, fixed_image, moving_image, p0=None, fluid_sigma=None, lncc_window_size=5, affine_params=None):
        """
        Forward pass computing standard LNCC loss at t=1 with optional Affine pre-alignment.
        """
        target_shape = tuple(fixed_image.shape[2:])
        disp_fwd, _ = self.shoot(p0=p0, fluid_sigma=fluid_sigma, image_shape=target_shape)
        
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
        
        v0 = separable_gaussian_filter(self.p0 if p0 is None else p0, sigma=self.fluid_sigma if fluid_sigma is None else fluid_sigma)
        energy_reg = 0.5 * torch.mean((self.p0 if p0 is None else p0) * v0)
        
        lncc_val = lncc_loss_nd(fixed_image, moving_warped, window_size=lncc_window_size)
        return lncc_val + 0.001 * energy_reg

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
        verbose=False
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
                loss_aff = self.forward(fixed_tensor, moving_tensor, p0=torch.zeros_like(self.p0), fluid_sigma=fluid_sigma, lncc_window_size=lncc_window_size)
                loss_aff.backward()
                opt_aff.step()
                self.affine.clamp_parameters()

        # 2. Deformable Geodesic Shooting Stage
        optimizer = torch.optim.Adam([self.p0], lr=lr)

        for level, epochs in zip(levels, epochs_per_level):
            if epochs <= 0:
                continue

            if level > 1:
                down_shape = [max(8, s // level) for s in self.image_shape]
                curr_fixed = F.interpolate(fixed_tensor, size=down_shape, mode=interp_mode, align_corners=True)
                curr_moving = F.interpolate(moving_tensor, size=down_shape, mode=interp_mode, align_corners=True)
            else:
                curr_fixed = fixed_tensor
                curr_moving = moving_tensor

            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if level > 1:
                    p0_cf = self.p0.movedim(-1, 1)
                    p0_down_cf = F.interpolate(p0_cf, size=curr_fixed.shape[2:], mode=interp_mode, align_corners=True)
                    p0_curr = p0_down_cf.movedim(1, -1)
                else:
                    p0_curr = self.p0

                loss = self.forward(curr_fixed, curr_moving, p0=p0_curr, fluid_sigma=fluid_sigma, lncc_window_size=lncc_window_size)
                if torch.isnan(loss) or torch.isinf(loss):
                    if verbose: print(f"[GeodesicShooting] Level {level} NaN loss detected at epoch {epoch}, stopping level.")
                    break
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_([self.p0], max_norm=5.0)
                
                optimizer.step()

        return self
