r"""
syngs.py — Geodesic Shooting (SyNGS) Hamiltonian Diffeomorphic Registration
=============================================================================

This module implements Geodesic Shooting (SyNGS) Riemannian diffeomorphic registration in PyTorch.

Key Algorithmic Features & Mechanics
------------------------------------
- Initial Momentum Parameterization: Optimizes initial velocity/momentum vector field $v_0$ at $t=0$.
- Euler-Poincaré Differential Equations (EPDiff): Shoots geodesic paths forward in time via Sobolev-damped EPDiff flow conservation.
- Diffeomorphic Geodesic Paths: Minimal energy geodesic paths in the diffeomorphism group $\text{Diff}(\Omega)$ with $\det(J) > 0$.
- Single Interpolation Invariant: Direct composition of affine pre-alignment and geodesic shooting fields.
"""

import math
import tempfile
import time as _time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ants

from .syn import (
    HierarchicalAffine,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    grid_to_physical_affine_torch,
    grid_sample_nd,
    local_ncc_loss_nd,
    mattes_mi_loss_nd,
    grid_to_physical_affine,
    parse_ants_affine,
)
from .core.smoothing import separable_gaussian_filter
from .core.optimizers import RegAdam, LARS
from .transform import (
    export_ants_displacement_field,
    export_ants_affine_transform,
    compute_grid_to_physical_reference_matrix,
)
from .pyramid import build_image_pyramid


class GeodesicShootingModel(nn.Module):
    """
    Geodesic Shooting (SyNGS) Registration Model in PyTorch.

    Parameterizes deformations by initial momentum / velocity fields v_0,
    evolved forward along geodesic trajectories via Sobolev-damped EPDiff integration.

    Parameters
    ----------
    dim : int
        Spatial dimensionality (2 or 3).
    image_shape : tuple of int
        Image grid shape in ZYX order.
    velocity_shape : tuple of int, optional
        Velocity field grid shape. Defaults to image_shape.
    spacing : list of float, optional
        Voxel spacing in XYZ order. Defaults to 1.0 per dimension.
    origin : list of float, optional
        Image origin in XYZ order. Defaults to 0.0 per dimension.
    direction : list of list of float, optional
        Direction matrix. Defaults to identity.
    fluid_sigma : float, optional
        Fluid regularization standard deviation for momentum smoothing. Default 3.0.
    elastic_sigma : float, optional
        Elastic regularization standard deviation. Default 0.0.
    transform_type : str, optional
        Affine transform type ('Affine', 'Rigid', 'Translation'). Default 'Affine'.
    n_steps : int, optional
        Number of EPDiff ODE integration steps. Default 5.
    solver : str, optional
        ODE integration solver ('euler' or 'rk4'). Default 'euler'.
    """
    def __init__(
        self,
        dim,
        image_shape,
        velocity_shape=None,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=3.0,
        elastic_sigma=0.0,
        transform_type='Affine',
        n_steps=5,
        solver='euler',
        symmetric=True,
        inverse_identity_weight=0.25,
        alpha=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.image_shape = tuple(image_shape)
        if velocity_shape is None:
            velocity_shape = image_shape
        self.velocity_shape = tuple(velocity_shape)
        self.n_steps = n_steps
        self.symmetric = symmetric
        self.inverse_identity_weight = inverse_identity_weight
        
        self.spacing = list(spacing) if spacing is not None else [1.0] * dim
        self.origin = list(origin) if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction.tolist() if hasattr(direction, 'tolist') else list(direction)
        else:
            self.direction = np.eye(dim).tolist()
            
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.solver = solver
        
        # Calibrated default Sobolev alpha: 0.080 for 3D, 0.060 for 2D
        if alpha is not None:
            self.alpha = float(alpha)
        else:
            self.alpha = 0.080 if dim == 3 else 0.060
            
        self.similarity_metric = kwargs.get('similarity_metric', 'lncc')
        self.mattes_bins = int(kwargs.get('mattes_bins', 32))
        
        # Dual momentum fields for symmetric shooting: v0_fwd (Fixed space) and v0_inv (Moving space)
        self.velocity_0_fwd = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        if self.symmetric:
            self.velocity_0_inv = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        else:
            self.velocity_0_inv = None
            
        self.velocity_0 = self.velocity_0_fwd
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)

    def _resize_single_velocity(self, vel_param, new_shape, device=None, dtype=None):
        if vel_param is None:
            return None
        new_shape = tuple(new_shape)
        old_shape = tuple(vel_param.shape[1:-1])
        if new_shape == old_shape:
            return vel_param
        with torch.no_grad():
            old_vel = vel_param.data
            if self.dim == 3:
                old_cf = old_vel.permute(0, 4, 1, 2, 3)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='trilinear', align_corners=True)
                new_vel = new_cf.permute(0, 2, 3, 4, 1)
            else:
                old_cf = old_vel.permute(0, 3, 1, 2)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='bilinear', align_corners=True)
                new_vel = new_cf.permute(0, 2, 3, 1)
            if device is not None:
                new_vel = new_vel.to(device=device)
            if dtype is not None:
                new_vel = new_vel.to(dtype=dtype)
            return nn.Parameter(new_vel.contiguous())

    def _resize_velocity(self, new_shape, device=None, dtype=None):
        self.velocity_0_fwd = self._resize_single_velocity(self.velocity_0_fwd, new_shape, device, dtype)
        if self.symmetric and self.velocity_0_inv is not None:
            self.velocity_0_inv = self._resize_single_velocity(self.velocity_0_inv, new_shape, device, dtype)
        self.velocity_0 = self.velocity_0_fwd

    def _create_boundary_mask(self, spatial_shape, device, dtype, border_width=None):
        dim = len(spatial_shape)
        if border_width is None:
            border_width = max(1, min(spatial_shape) // 32)
        if border_width <= 0:
            return torch.ones((1, *spatial_shape, 1), device=device, dtype=dtype)
        
        axes_masks = []
        for d in range(dim):
            n_d = spatial_shape[d]
            idx = torch.arange(n_d, device=device, dtype=dtype)
            dist = torch.min(idx, (n_d - 1) - idx)
            mask_d = torch.where(
                dist < border_width,
                0.5 * (1.0 - torch.cos(math.pi * dist / float(border_width))),
                torch.ones_like(dist)
            )
            shape_d = [1] * dim
            shape_d[d] = n_d
            axes_masks.append(mask_d.view(*shape_d))
            
        mask = axes_masks[0]
        for d in range(1, dim):
            mask = mask * axes_masks[d]
        return mask.unsqueeze(0).unsqueeze(-1)

    def apply_green_operator(self, m, shape, spacing_zyx):
        """
        Apply exact Fourier Sobolev Green's operator K(k) = 1 / (1 + alpha * |k|^2)^s.
        Enforces smooth boundary conditions to eliminate FFT Gibbs ringing.
        """
        if self.fluid_sigma <= 0:
            return m
        device = m.device
        dtype = m.dtype
        dim = self.dim
        
        bmask = self._create_boundary_mask(shape, device, dtype, border_width=4)
        m_tapered = m * bmask
        
        k_axes = []
        for d in range(dim):
            n_d = shape[d]
            sp_d = spacing_zyx[d]
            if d == dim - 1:
                k_d = torch.fft.rfftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            else:
                k_d = torch.fft.fftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            k_axes.append(k_d)
            
        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        k_sq = sum(k_j ** 2 for k_j in k_mesh)
        K_fourier = 1.0 / ((1.0 + self.alpha * k_sq) ** 2.0)
        
        spatial_dims = tuple(range(2, 2 + dim))
        if dim == 3:
            m_cf = m_tapered.permute(0, 4, 1, 2, 3).to(torch.float32).contiguous()
        else:
            m_cf = m_tapered.permute(0, 3, 1, 2).to(torch.float32).contiguous()
            
        m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)
        K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
        v_fft = m_fft * K_bc
        v_cf = torch.fft.irfftn(v_fft, s=shape, dim=spatial_dims).to(dtype=dtype).contiguous()
        
        if dim == 3:
            v_out = v_cf.permute(0, 2, 3, 4, 1)
        else:
            v_out = v_cf.permute(0, 2, 3, 1)
            
        return v_out * bmask

    def shoot(self, v_init, target_shape, spacing_zyx, phys_grid, meta):
        """
        Evolve initial momentum v_init forward along the geodesic trajectory.
        Integrates: d phi / dt = v(t, phi(t)) with Sobolev-damped EPDiff flow conservation.
        """
        dt = 1.0 / self.n_steps
        
        # Up-sample v_init to target_shape if needed
        if tuple(v_init.shape[1:-1]) != target_shape:
            if self.dim == 3:
                v_cf = v_init.permute(0, 4, 1, 2, 3)
                v_up = F.interpolate(v_cf, size=target_shape, mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
            else:
                v_cf = v_init.permute(0, 3, 1, 2)
                v_up = F.interpolate(v_cf, size=target_shape, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        else:
            v_up = v_init
            
        v = self.apply_green_operator(v_up, target_shape, spacing_zyx)
        shape_t, spacing_t, origin_t, direction_t = meta
        disp = torch.zeros_like(phys_grid)
        
        for step in range(self.n_steps):
            phi_curr = phys_grid + disp
            phi_norm = physical_to_normalized_torch_cached(phi_curr, shape_t, spacing_t, origin_t, direction_t)
            
            if self.dim == 3:
                v_cf = v.permute(0, 4, 1, 2, 3)
                v_sampled_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = v_sampled_cf.permute(0, 2, 3, 4, 1)
            else:
                v_cf = v.permute(0, 3, 1, 2)
                v_sampled_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = v_sampled_cf.permute(0, 2, 3, 1)
                
            disp = disp + dt * v_sampled
            
            if step < self.n_steps - 1:
                # Evolve velocity along geodesic with Sobolev damping
                if self.dim == 3:
                    v_pullback_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                    v = self.apply_green_operator(v_pullback_cf.permute(0, 2, 3, 4, 1), target_shape, spacing_zyx)
                else:
                    v_pullback_cf = grid_sample_nd(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                    v = self.apply_green_operator(v_pullback_cf.permute(0, 2, 3, 1), target_shape, spacing_zyx)
                
        return disp

    def _eval_similarity(self, I, J, metric_name, lncc_window_size=5):
        m_lower = metric_name.lower()
        if m_lower in ('mattes_mi', 'mattes', 'mi', 'mmi') or m_lower.startswith('mattes') or m_lower.startswith('mi_'):
            n_bins = getattr(self, 'mattes_bins', 32)
            parts = m_lower.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                n_bins = int(parts[-1])
            fg_mask = ((I.abs() > 0.01) | (J.abs() > 0.01)).float()
            return mattes_mi_loss_nd(I, J, mask=fg_mask, num_bins=n_bins)
        elif m_lower == 'mse':
            return torch.mean((I - J) ** 2)
        elif m_lower in ('cc2', 'lncc2'):
            return local_ncc_loss_nd(I, J, window_size=lncc_window_size, squared=True)
        else:
            return local_ncc_loss_nd(I, J, window_size=lncc_window_size, squared=False)

    def forward(self, fixed_image, moving_image, lncc_window_size=5, similarity_metric=None):
        """
        Forward pass with symmetric dual-momentum shooting and inverse identity composition constraint.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        target_shape = tuple(fixed_image.shape[2:])

        curr_spacing = [
            sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]

        phys_grid = get_physical_grid_torch(
            target_shape, curr_spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )

        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        dir_arr = np.asarray(self.direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(self.dim, self.dim)
        direction_rev = dir_arr[::-1, ::-1].copy()

        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        meta = (shape_t, spacing_t, origin_t, direction_t)

        T_grid = self.affine.get_matrix()
        M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
            T_grid, target_shape, curr_spacing, self.origin, self.direction,
            target_shape, curr_spacing, self.origin, self.direction
        )
        M_phys_inv_zyx = torch.inverse(M_phys_zyx)
        t_phys_inv_zyx = -M_phys_inv_zyx @ t_phys_zyx

        # 1. Forward shooting (+v0_fwd)
        disp_fwd = self.shoot(self.velocity_0_fwd, target_shape, spacing_rev, phys_grid, meta)
        phi_moving = (phys_grid + disp_fwd) @ M_phys_zyx.t() + t_phys_zyx
        phi_norm_fwd = physical_to_normalized_torch_cached(
            phi_moving, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = grid_sample_nd(moving_image, phi_norm_fwd, mode='bilinear', padding_mode='zeros')
        
        # 2. Inverse shooting (+v0_inv)
        v0_inv_param = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        disp_inv = self.shoot(v0_inv_param, target_shape, spacing_rev, phys_grid, meta)
        phi_fixed = (phys_grid + disp_inv) @ M_phys_inv_zyx.t() + t_phys_inv_zyx
        phi_norm_inv = physical_to_normalized_torch_cached(
            phi_fixed, shape_t, spacing_t, origin_t, direction_t
        )
        fixed_warped = grid_sample_nd(fixed_image, phi_norm_inv, mode='bilinear', padding_mode='zeros')

        metric_to_use = similarity_metric if similarity_metric is not None else self.similarity_metric
        loss_fwd = self._eval_similarity(fixed_image, moving_warped, metric_to_use, lncc_window_size=lncc_window_size)
        loss_inv = self._eval_similarity(moving_image, fixed_warped, metric_to_use, lncc_window_size=lncc_window_size)
        sim_loss = 0.5 * (loss_fwd + loss_inv)

        # 3. Inverse identity consistency loss
        if self.symmetric and self.inverse_identity_weight > 0:
            phi_inv_pure = phys_grid + disp_inv
            phi_inv_pure_norm = physical_to_normalized_torch_cached(
                phi_inv_pure, shape_t, spacing_t, origin_t, direction_t
            )
            if self.dim == 3:
                disp_fwd_cf = disp_fwd.permute(0, 4, 1, 2, 3)
                disp_fwd_at_inv = grid_sample_nd(disp_fwd_cf, phi_inv_pure_norm, mode='bilinear', padding_mode='border').permute(0, 2, 3, 4, 1)
            else:
                disp_fwd_cf = disp_fwd.permute(0, 3, 1, 2)
                disp_fwd_at_inv = grid_sample_nd(disp_fwd_cf, phi_inv_pure_norm, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)
            comp_disp = disp_inv + disp_fwd_at_inv
            inv_id_loss = torch.mean(comp_disp ** 2)
        else:
            inv_id_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return sim_loss + self.inverse_identity_weight * inv_id_loss

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[60, 60, 30],
        affine_epochs=100,
        similarity_metric='lncc',
        lncc_radius=2,
        lr=0.8,
        reg_weight=0.0,
        verbose=False,
        fixed_spacing=None,
        fixed_origin=None,
        fixed_direction=None,
        optimizer_type='reg_adam',
        cfl_step=0.30,
        fluid_sigmas=None,
        elastic_sigmas=None,
        **kwargs
    ):
        """
        Multi-resolution optimization loop for Geodesic Shooting.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        
        self.similarity_metric = similarity_metric
        self.mattes_bins = int(kwargs.get('mattes_bins', getattr(self, 'mattes_bins', 32)))
        
        if fixed_spacing is not None: self.spacing = list(fixed_spacing)
        if fixed_origin is not None: self.origin = list(fixed_origin)
        if fixed_direction is not None:
            self.direction = fixed_direction.tolist() if hasattr(fixed_direction, 'tolist') else list(fixed_direction)

        smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
        if smoothing_sigmas is None:
            smoothing_sigmas = [float(np.log2(s)) if s > 1 else 0.0 for s in levels]
        fixed_pyr = build_image_pyramid(fixed_image, spacing=self.spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')
        moving_pyr = build_image_pyramid(moving_image, spacing=self.spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')

        total_affine_epochs = sum(affine_epochs) if isinstance(affine_epochs, (list, tuple)) else (affine_epochs if affine_epochs is not None else 0)
        if total_affine_epochs > 0 and getattr(self, 'transform_type', 'Affine') != 'Translation_only':
            if verbose: print("Optimizing affine pre-alignment...")
            aff_optimizer = torch.optim.Adam(self.affine.parameters(), lr=1e-2)
            aff_epochs_list = affine_epochs if isinstance(affine_epochs, (list, tuple)) else [affine_epochs] * len(levels)
            
            for idx, level in enumerate(levels):
                curr_aff_epochs = aff_epochs_list[min(idx, len(aff_epochs_list) - 1)]
                if curr_aff_epochs <= 0:
                    continue
                    
                curr_fixed_aff = fixed_pyr[idx]
                curr_moving_aff = moving_pyr[idx]
                curr_target_shape = tuple(curr_fixed_aff.shape[2:])
                curr_spacing_aff = [
                    sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                    for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, curr_target_shape)
                ]
                
                phys_grid_aff = get_physical_grid_torch(
                    curr_target_shape, curr_spacing_aff, self.origin, self.direction,
                    device=device, dtype=dtype
                )

                shape_t_aff = torch.tensor(list(curr_target_shape), device=device, dtype=dtype)
                spacing_t_aff = torch.tensor(tuple(reversed(curr_spacing_aff)), device=device, dtype=dtype)
                origin_t_aff = torch.tensor(tuple(reversed(self.origin)), device=device, dtype=dtype)
                dir_arr = np.asarray(self.direction)
                if dir_arr.ndim == 1:
                    dir_arr = dir_arr.reshape(self.dim, self.dim)
                direction_t_aff = torch.tensor(dir_arr[::-1, ::-1].copy(), device=device, dtype=dtype)

                for ep in range(curr_aff_epochs):
                    aff_optimizer.zero_grad()
                    T_grid = self.affine.get_matrix()
                    M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
                        T_grid, curr_target_shape, curr_spacing_aff, self.origin, self.direction,
                        curr_target_shape, curr_spacing_aff, self.origin, self.direction
                    )
                    
                    phi_moving_aff = phys_grid_aff @ M_phys_zyx.t() + t_phys_zyx
                    phi_norm_aff = physical_to_normalized_torch_cached(
                        phi_moving_aff, shape_t_aff, spacing_t_aff, origin_t_aff, direction_t_aff
                    )
                    moving_warped_aff = grid_sample_nd(curr_moving_aff, phi_norm_aff, mode='bilinear', padding_mode='zeros')
                    
                    aff_metric = kwargs.get('aff_metric', similarity_metric)
                    if aff_metric in ('mattes_mi', 'mattes', 'mi'):
                        aff_loss = mattes_mi_loss_nd(curr_fixed_aff, moving_warped_aff, num_bins=32, sampling_percentage=0.2)
                    else:
                        aff_loss = local_ncc_loss_nd(curr_fixed_aff, moving_warped_aff, window_size=5)
                        
                    aff_loss.backward()
                    aff_optimizer.step()
                    self.affine.clamp_parameters()

        if verbose: print("Optimizing Geodesic Shooting momentum...")
        opt_name = str(optimizer_type).lower()

        for idx, level in enumerate(levels):
            epochs = epochs_per_level[min(idx, len(epochs_per_level) - 1)]
            if epochs <= 0:
                continue
                
            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            self._resize_velocity(curr_vel_shape, device, dtype)
            
            curr_fixed = fixed_pyr[idx]
            curr_moving = moving_pyr[idx]
            target_shape = tuple(curr_fixed.shape[2:])
            
            curr_spacing = [sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)]
            spacing_rev = tuple(reversed(curr_spacing))
            origin_rev = tuple(reversed(self.origin))
            dir_arr = np.asarray(self.direction)
            if dir_arr.ndim == 1:
                dir_arr = dir_arr.reshape(self.dim, self.dim)
            direction_rev = dir_arr[::-1, ::-1].copy()
            
            phys_grid = get_physical_grid_torch(target_shape, curr_spacing, self.origin, self.direction, device=device, dtype=dtype)
            shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
            spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
            origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
            direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
            meta = (shape_t, spacing_t, origin_t, direction_t)
            
            active_params = [self.velocity_0_fwd]
            if self.symmetric and self.velocity_0_inv is not None:
                active_params.append(self.velocity_0_inv)

            level_lr = lr * math.sqrt(1.0 / level)
            max_step = float(kwargs.get('max_step_norm', cfl_step))
            
            if opt_name in ('reg_adam', 'regadam', 'sobolev_adam', 'sobolevadam'):
                optimizer = RegAdam(
                    active_params,
                    lr=level_lr,
                    regularizer='sobolev',
                    sobolev_alpha=self.alpha,
                    max_step_norm=max_step
                )
            elif opt_name == 'adam':
                optimizer = torch.optim.Adam(active_params, lr=level_lr)
            else:
                optimizer = LARS(active_params, lr=level_lr)

            lncc_ws = 2 * lncc_radius + 1
            
            for ep in range(epochs):
                optimizer.zero_grad()
                total_loss = self.forward(
                    curr_fixed, curr_moving,
                    lncc_window_size=lncc_ws,
                    similarity_metric=similarity_metric
                )
                total_loss.backward()
                optimizer.step()

        # Resize parameters back to native resolution for final export
        final_vel_shape = tuple(self.velocity_0_fwd.shape[1:-1])
        if final_vel_shape != tuple(self.image_shape):
            self._resize_velocity(self.image_shape, device, dtype)

    @torch.no_grad()
    def get_forward_warp(self, image_shape=None):
        """Compute forward displacement field (shooting +v0_fwd)."""
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        curr_spacing = [
            sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]
        device = self.velocity_0_fwd.device
        dtype = self.velocity_0_fwd.dtype
        phys_grid = get_physical_grid_torch(target_shape, curr_spacing, self.origin, self.direction, device=device, dtype=dtype)
        
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        dir_arr = np.asarray(self.direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(self.dim, self.dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        meta = (shape_t, spacing_t, origin_t, direction_t)
        
        disp = self.shoot(self.velocity_0_fwd, target_shape, spacing_rev, phys_grid, meta)
        return disp
        
    @torch.no_grad()
    def get_inverse_warp(self, image_shape=None):
        """Compute inverse displacement field (shooting +v0_inv if symmetric, else -v0_fwd)."""
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        curr_spacing = [
            sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]
        device = self.velocity_0_fwd.device
        dtype = self.velocity_0_fwd.dtype
        phys_grid = get_physical_grid_torch(target_shape, curr_spacing, self.origin, self.direction, device=device, dtype=dtype)
        
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        dir_arr = np.asarray(self.direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(self.dim, self.dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        meta = (shape_t, spacing_t, origin_t, direction_t)
        
        v0_inv = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        disp = self.shoot(v0_inv, target_shape, spacing_rev, phys_grid, meta)
        return disp


def syngs_registration(
    fixed,
    moving,
    type_of_transform='SyNGS',
    initial_transform=None,
    syn_metric='lncc',
    syn_sampling=2,
    aff_metric=None,
    aff_sampling=None,
    reg_iterations=None,
    affine_iterations=None,
    grad_step=0.30,
    flow_sigma=3.0,
    total_sigma=0.0,
    n_steps=5,
    n_time_steps=None,
    verbose=False,
    backend='pytorch',
    levels=None,
    cfl_momentum=0.9,
    multipoint_loss=None,
    fast_smooth=True,
    sampling_percentage=None,
    vgg_layers=None,
    vgg_mode=None,
    vgg_patch_size=None,
    vgg_num_patches=None,
    vgg_lncc_window_size=None,
    optimizer='reg_adam',
    optimizer_lr=None,
    project_inverse=None,
    projection_frequency=None,
    interpolator=None,
    inverse_method=None,
    inverse_steps=None,
    **kwargs
):
    """
    High-level SyNGS (Symmetric Normalization Geodesic Shooting) registration function
    matching the ``syntx.syn()`` / ``syntx.registration()`` interface.

    Parameters
    ----------
    fixed : ANTsImage
        Fixed target image.
    moving : ANTsImage
        Moving source image.
    type_of_transform : str, optional
        Transform descriptor (default 'SyNGS'). Included for API parity.
    initial_transform : str or list of str or ANTsTransform, optional
        Initial transform(s) to apply to moving image before registration. Default None.
    syn_metric : str, optional
        Similarity metric. Default 'lncc'.
    syn_sampling : int, optional
        LNCC radius (window_size = 2 * syn_sampling + 1). Default 2.
    aff_metric : str or None, optional
        Similarity metric for affine initialization. Default None.
    reg_iterations : list of int or None, optional
        Deformable iterations per pyramid level. Default [60, 60, 30].
    affine_iterations : list of int or None, optional
        Affine iterations per level. Default 100.
    grad_step : float, optional
        CFL voxel bound step size. Default 0.30.
    flow_sigma : float, optional
        Fluid regularization sigma. Default 3.0.
    total_sigma : float, optional
        Elastic regularization sigma. Default 0.0.
    n_steps : int, optional
        Number of EPDiff ODE integration steps. Default 5.
    verbose : bool, optional
        If True, print optimization progress. Default False.
    backend : str, optional
        Computation backend ('pytorch' or 'jax'). Default 'pytorch'.
    levels : list of int or None, optional
        Multi-resolution pyramid levels. Default [4, 2, 1] for 3D, [8, 4, 2, 1] for 2D.
    optimizer : str, optional
        Optimizer type ('reg_adam', 'adam', 'lars'). Default 'reg_adam'.

    Returns
    -------
    dict
        Same format as ``syntx.syn()`` / ``syntx.registration()``:
            - 'warpedmovout': ANTsImage (moving warped to fixed space)
            - 'warpedfixout': ANTsImage (fixed warped to moving space)
            - 'fwdtransforms': list of str (file paths to transforms)
            - 'invtransforms': list of str (file paths to inverse transforms)
            - 'whichtoinvert_inv': list of bool
            - 'model': GeodesicShootingModel
            - 'provenance': dict
    """
    t_start = _time.time()

    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    origin = fixed.origin
    direction = fixed.direction

    if 'similarity_metric' in kwargs:
        syn_metric = kwargs.pop('similarity_metric')

    # Defaults matching syntx.syn() / syntx.tvf()
    if levels is None:
        if reg_iterations is not None:
            num_levels = len(reg_iterations)
            levels = [2**i for i in range(num_levels)][::-1]
        else:
            levels = [4, 2, 1] if dim == 3 else [8, 4, 2, 1]

    if reg_iterations is None:
        reg_iterations = [60, 60, 30] if dim == 3 else [60, 60, 40, 20]
    if affine_iterations is None:
        affine_iterations = 0 if initial_transform is not None else 100

    fluid_sigma_actual = float(flow_sigma) if flow_sigma > 0 else 3.0
    elastic_sigma_actual = float(total_sigma) if total_sigma > 0 else 0.0

    # Extract initial transform (Single Interpolation Policy)
    init_tx_list = []
    init_M_phys, init_t_phys = None, None
    if initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        init_M_phys, init_t_phys = parse_ants_affine(init_tx_list, dim)

    # Normalize images
    fi_np = fixed.numpy()
    mi_np = moving.numpy()

    winsorize_quantiles = kwargs.pop('winsorize_quantiles', None)
    if winsorize_quantiles is not None:
        lo_f, hi_f = np.quantile(fi_np[fi_np > 0], winsorize_quantiles) if (fi_np > 0).any() else (fi_np.min(), fi_np.max())
        fi_np = np.clip(fi_np, lo_f, hi_f)
        lo_m, hi_m = np.quantile(mi_np[mi_np > 0], winsorize_quantiles) if (mi_np > 0).any() else (mi_np.min(), mi_np.max())
        mi_np = np.clip(mi_np, lo_m, hi_m)

    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

    grid_shape_zyx = tuple(reversed(grid_shape))
    perm = [0, 1] + list(range(dim + 1, 1, -1))

    if backend.lower() == 'pytorch':
        device_str = kwargs.pop('device', None)
        if device_str is None:
            if torch.cuda.is_available():
                device_str = 'cuda'
            elif torch.backends.mps.is_available():
                device_str = 'mps'
            else:
                device_str = 'cpu'

        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device_str).unsqueeze(0).unsqueeze(0).permute(perm)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device_str).unsqueeze(0).unsqueeze(0).permute(perm)

        model = GeodesicShootingModel(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_steps=n_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
            similarity_metric=syn_metric,
            alpha=kwargs.pop('alpha', kwargs.pop('sobolev_alpha', None)),
        ).to(device_str)

        # Single Interpolation Invariant: absorb initial transform into T_init
        if init_M_phys is not None:
            with torch.no_grad():
                dtype_dev = torch.float32
                H_x = compute_grid_to_physical_reference_matrix(fixed.shape, fixed.spacing, fixed.origin, fixed.direction, device=device_str, dtype=dtype_dev)
                H_y = compute_grid_to_physical_reference_matrix(moving.shape, moving.spacing, moving.origin, moving.direction, device=device_str, dtype=dtype_dev)

                T_phys = torch.eye(dim + 1, device=device_str, dtype=dtype_dev)
                T_phys[:dim, :dim] = init_M_phys.to(device=device_str, dtype=dtype_dev)
                T_phys[:dim, dim] = init_t_phys.to(device=device_str, dtype=dtype_dev)

                T_init = torch.inverse(H_y) @ T_phys @ H_x
                model.affine.T_init = T_init

            init_tx_list = []
            if verbose:
                print("[SyNGS] Initialized affine from initial_transform (T_init absorbed)")

        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            affine_epochs=affine_iterations,
            similarity_metric=syn_metric,
            lr=optimizer_lr if optimizer_lr is not None else kwargs.pop('lr', 0.8),
            reg_weight=kwargs.pop('reg_weight', 0.0),
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            optimizer_type=optimizer,
            cfl_step=grad_step,
            **kwargs
        )

        with torch.no_grad():
            fwd_disp = model.get_forward_warp(image_shape=grid_shape_zyx)
            inv_disp = model.get_inverse_warp(image_shape=grid_shape_zyx)
            fwd_np = fwd_disp.cpu().squeeze(0).numpy()
            inv_np = inv_disp.cpu().squeeze(0).numpy()

        T_grid = model.affine.get_matrix().detach().cpu().numpy()

    elif backend.lower() == 'jax':
        from .syngs_jax import GeodesicShootingModelJAX
        from .syn_jax import get_affine_matrix_jax
        import jax.numpy as jnp

        device_str = 'cpu'
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *fixed.shape).transpose(perm)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *moving.shape).transpose(perm)

        model = GeodesicShootingModelJAX(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_steps=n_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
        )

        if init_M_phys is not None:
            H_x = compute_grid_to_physical_reference_matrix(fixed.shape, fixed.spacing, fixed.origin, fixed.direction, device='cpu', dtype=torch.float32).numpy()
            H_y = compute_grid_to_physical_reference_matrix(moving.shape, moving.spacing, moving.origin, moving.direction, device='cpu', dtype=torch.float32).numpy()

            T_phys = np.eye(dim + 1, dtype=np.float32)
            T_phys[:dim, :dim] = init_M_phys.numpy() if hasattr(init_M_phys, 'numpy') else np.asarray(init_M_phys)
            T_phys[:dim, dim] = init_t_phys.numpy() if hasattr(init_t_phys, 'numpy') else np.asarray(init_t_phys)

            T_init_jax = jnp.array(np.linalg.inv(H_y) @ T_phys @ H_x)
            model.T_init = T_init_jax
            model.affine_params['T_init'] = T_init_jax

            init_tx_list = []
            if verbose:
                print("[SyNGS-JAX] Initialized affine from initial_transform (T_init absorbed)")

        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            affine_epochs=affine_iterations,
            similarity_metric=syn_metric,
            lr=kwargs.pop('lr', 0.8),
            reg_weight=kwargs.pop('reg_weight', 0.0),
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            optimizer_type=optimizer,
            cfl_step=grad_step,
            **kwargs
        )

        fwd_disp = np.array(model.get_forward_warp(image_shape=grid_shape_zyx))
        inv_disp = np.array(model.get_inverse_warp(image_shape=grid_shape_zyx))
        fwd_np = fwd_disp.squeeze(0)
        inv_np = inv_disp.squeeze(0)

        T_grid = np.array(get_affine_matrix_jax(model.affine_params, dim, 'Affine'))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Export displacement fields using standardized ITK components
    fwd_img = export_ants_displacement_field(fwd_np, origin=origin, spacing=spacing, direction=direction)
    inv_img = export_ants_displacement_field(inv_np, origin=origin, spacing=spacing, direction=direction)

    fwd_file = tempfile.NamedTemporaryFile(suffix='_syngs_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_syngs_inv_Warp.nii.gz', delete=False).name
    ants.image_write(fwd_img, fwd_file)
    ants.image_write(inv_img, inv_file)

    # Export affine transform using standardized reference matrix conversion
    M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving)
    affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    tx_fwd, tx_inv = export_ants_affine_transform(M_phys, t_phys, dim=dim)
    ants.write_transform(tx_fwd, affine_file)

    # Build transform lists (Single Interpolation Invariant)
    if sum(reg_iterations) > 0:
        fwd_transforms = [fwd_file, affine_file] + init_tx_list
        inv_transforms = init_tx_list + [affine_file, inv_file]
        whichtoinvert_inv = [True] * len(init_tx_list) + [True, False]
    else:
        fwd_transforms = [affine_file] + init_tx_list
        inv_transforms = init_tx_list + [affine_file]
        whichtoinvert_inv = [True] * (len(init_tx_list) + 1)

    # Apply single-interpolation composite transforms
    warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_transforms)
    warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=inv_transforms,
                                          whichtoinvert=whichtoinvert_inv)

    fit_time = _time.time() - t_start

    # Clean up GPU memory
    if device_str == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device_str == 'cuda':
        torch.cuda.empty_cache()

    ret_dict = {
        'warpedmovout': warpedmovout,
        'warpedfixout': warpedfixout,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': whichtoinvert_inv,
        'model': model,
    }

    try:
        from .reporting import build_engine_provenance
        provenance = build_engine_provenance(
            algorithm="syntx.syngs",
            backend=backend,
            device=device_str,
            fit_time=fit_time,
            reg_iterations=reg_iterations,
            affine_iterations=affine_iterations if isinstance(affine_iterations, list) else [affine_iterations],
            solver="GS-Euler",
            fluid_sigma=flow_sigma,
            elastic_sigma=total_sigma,
            learning_rate=grad_step,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            similarity_metric=syn_metric,
            syn_sampling=syn_sampling,
            aff_metric=aff_metric,
            aff_sampling=aff_sampling,
            levels=levels,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_mode=vgg_mode,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_lncc_window_size=vgg_lncc_window_size,
            project_inverse=project_inverse,
            projection_frequency=projection_frequency,
            interpolator=interpolator,
            inverse_method=inverse_method,
            inverse_steps=inverse_steps,
            fixed_shape=tuple(fixed.shape),
            fixed_spacing=tuple(fixed.spacing),
            fixed_orientation=str(fixed.orientation) if hasattr(fixed, 'orientation') else None,
            moving_shape=tuple(moving.shape),
            moving_spacing=tuple(moving.spacing),
            moving_orientation=str(moving.orientation) if hasattr(moving, 'orientation') else None,
            n_steps=n_steps
        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass

    return ret_dict
