r"""
syngs.py — Geodesic Shooting (SyNGS) Hamiltonian Diffeomorphic Registration
=============================================================================

This module implements Geodesic Shooting (SyNGS) Riemannian diffeomorphic registration in PyTorch.

Key Algorithmic Features & Mechanics
------------------------------------
- Initial Momentum Parameterization: Optimizes initial velocity/momentum vector field $v_0$ at $t=0$.
- Euler-Poincaré Differential Equations (EPDiff): Shoots geodesic paths forward in time via EPDiff flow conservation.
- Diffeomorphic Geodesic Paths: Guarantees minimal energy geodesic paths in the diffeomorphism group $\text{Diff}(\Omega)$.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .syn import (
    HierarchicalAffine,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    grid_to_physical_affine_torch,
    grid_sample_nd,
    local_ncc_loss_nd as lncc_loss_nd,
    mattes_mi_loss_nd,
    separable_gaussian_filter,
    grid_to_physical_affine,
    parse_ants_affine,
    _spatial_jacobian_nd
)


class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) Optimizer for Initial Velocity Parameters.

    Rescales initial velocity momentum updates using trust ratio scaling:
    $$\\text{trust\\_ratio} = \\eta \\cdot \\frac{\\max(\\|p\\|_2, 1.0)}{\\|g\\|_2 + \\epsilon}$$

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or parameter group dicts.
    lr : float, default=0.80
        Base learning rate.
    trust_coefficient : float, default=0.05
        Trust ratio scaling factor $\\eta$.
    eps : float, default=1e-8
        Numerical stability epsilon denominator.
    """
    def __init__(self, params, lr=0.80, trust_coefficient=0.05, eps=1e-8):
        defaults = dict(lr=lr, trust_coefficient=trust_coefficient, eps=eps)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            trust_coeff = group['trust_coefficient']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                p_norm = torch.norm(p)
                g_norm = torch.norm(g)
                p_norm_effective = torch.clamp(p_norm, min=1.0)

                if g_norm > 0:
                    trust_ratio = trust_coeff * p_norm_effective / (g_norm + eps)
                else:
                    trust_ratio = 1.0

                local_lr = lr * trust_ratio
                p.sub_(g * local_lr)
        return loss


class GeodesicShootingModel(nn.Module):
    """
    Geodesic Shooting (SyNGS) Registration Model in PyTorch.

    Parameterizes deformations by initial momentum / velocity field v_0,
    evolved forward along geodesic trajectories via Euler integration of the EPDiff equation.

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
        Fluid regularization standard deviation for momentum smoothing. Default 1.0.
    elastic_sigma : float, optional
        Elastic regularization standard deviation. Default 0.0.
    transform_type : str, optional
        Affine transform type ('Affine', 'Rigid', 'Translation'). Default 'Affine'.
    n_steps : int, optional
        Number of EPDiff ODE integration steps. Default 5.
    solver : str, optional
        ODE integration solver ('euler'). Default 'euler'.
    """
    def __init__(
        self,
        dim,
        image_shape,
        velocity_shape=None,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=1.0,
        elastic_sigma=0.0,
        transform_type='Affine',
        n_steps=5,
        solver='euler',
        symmetric=True,
        inverse_identity_weight=1.0,
        image_grad_clip=6.0,
        velocity_clamp=50.0,
        cfl_max=None
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
        self.image_grad_clip = image_grad_clip
        self.velocity_clamp = velocity_clamp
        self.cfl_max = cfl_max
        
        self.spacing = spacing if spacing is not None else [1.0] * dim
        self.origin = origin if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction
        else:
            self.direction = np.eye(dim).tolist()
            
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.solver = solver
        
        # Dual momentum fields for symmetric shooting: v0_fwd (Fixed space) and v0_inv (Moving space)
        self.velocity_0_fwd = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        if self.symmetric:
            self.velocity_0_inv = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        else:
            self.velocity_0_inv = None
            
        # Alias for backward compatibility
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

    def init_velocities_from_image_gradients(self, fixed_image, moving_image):
        """
        Initialize velocity_0_fwd and velocity_0_inv from the spatial gradients of
        fixed_image and moving_image respectively.
        """
        with torch.no_grad():
            device = fixed_image.device
            dtype = fixed_image.dtype
            spacing_rev = tuple(reversed(self.spacing))
            
            from .syn import _spatial_jacobian_nd
            grad_f = _spatial_jacobian_nd(fixed_image.movedim(1, -1), physical_spacing=spacing_rev).squeeze(-2)
            grad_m = _spatial_jacobian_nd(moving_image.movedim(1, -1), physical_spacing=spacing_rev).squeeze(-2)
            
            vel_shape_f = tuple(self.velocity_0_fwd.shape[1:-1])
            if tuple(grad_f.shape[1:-1]) != vel_shape_f:
                if self.dim == 3:
                    grad_f = F.interpolate(grad_f.permute(0, 4, 1, 2, 3), size=vel_shape_f, mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
                else:
                    grad_f = F.interpolate(grad_f.permute(0, 3, 1, 2), size=vel_shape_f, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            
            if self.symmetric and self.velocity_0_inv is not None:
                vel_shape_m = tuple(self.velocity_0_inv.shape[1:-1])
                if tuple(grad_m.shape[1:-1]) != vel_shape_m:
                    if self.dim == 3:
                        grad_m = F.interpolate(grad_m.permute(0, 4, 1, 2, 3), size=vel_shape_m, mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
                    else:
                        grad_m = F.interpolate(grad_m.permute(0, 3, 1, 2), size=vel_shape_m, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            
            norm_f = torch.sqrt(torch.sum(grad_f**2, dim=-1, keepdim=True)) + 1e-8
            norm_m = torch.sqrt(torch.sum(grad_m**2, dim=-1, keepdim=True)) + 1e-8
            
            v0_f = 5e-3 * (grad_f / (norm_f.max() + 1e-8))
            v0_m = 5e-3 * (grad_m / (norm_m.max() + 1e-8))
            
            self.velocity_0_fwd.data.copy_(v0_f)
            if self.symmetric and self.velocity_0_inv is not None:
                self.velocity_0_inv.data.copy_(v0_m)

    def _get_metadata_tensors(self, device, dtype):
        spacing_rev = tuple(reversed(self.spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
        
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(self.image_shape), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        return shape_t, spacing_t, origin_t, direction_t

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

    def apply_green_operator(self, m, vel_shape, spacing_zyx, alpha=None, s=2.0, border_width=0):
        """
        Apply exact Fourier Sobolev Green's operator K(k) = 1 / (1 + alpha * |k|^2)^s.
        Enforces smooth Dirichlet boundary conditions to eliminate FFT Gibbs ringing.
        """
        if self.fluid_sigma <= 0:
            return m
        device = m.device
        dtype = m.dtype
        dim = self.dim
        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(self.fluid_sigma / 2.0)
        s_val = float(s)
        
        bmask = self._create_boundary_mask(vel_shape, device, dtype, border_width=border_width)
        m_tapered = m * bmask
        
        k_axes = []
        for d in range(dim):
            n_d = vel_shape[d]
            sp_d = spacing_zyx[d]
            if d == dim - 1:
                k_d = torch.fft.rfftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            else:
                k_d = torch.fft.fftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            k_axes.append(k_d)
            
        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        k_sq = sum(k_j ** 2 for k_j in k_mesh)
        K_fourier = 1.0 / ((1.0 + alpha_val * k_sq) ** s_val)
        
        spatial_dims = tuple(range(2, 2 + dim))
        if dim == 3:
            m_cf = m_tapered.permute(0, 4, 1, 2, 3).to(torch.float32).contiguous()
        else:
            m_cf = m_tapered.permute(0, 3, 1, 2).to(torch.float32).contiguous()
            
        m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)
        K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
        v_fft = m_fft * K_bc
        v_cf = torch.fft.irfftn(v_fft, s=vel_shape, dim=spatial_dims).to(dtype=dtype).contiguous()
        
        if dim == 3:
            v_out = v_cf.permute(0, 2, 3, 4, 1)
        else:
            v_out = v_cf.permute(0, 2, 3, 1)
            
        return v_out * bmask

    def _apply_sobolev_green_operator(self, m, fluid_sigma=2.0, alpha=None, spacing=None, s=2.0, border_width=0):
        if fluid_sigma <= 0:
            return m
        orig_shape = m.shape
        dim = self.dim
        spatial_shape = orig_shape[-(dim + 1):-1]
        sp = spacing if spacing is not None else getattr(self, 'spacing', [1.0] * dim)
        if sp is None or len(sp) != dim:
            sp = [1.0] * dim
        sp_zyx = list(reversed(sp))
        m_flat = m.reshape(-1, *spatial_shape, dim)
        
        old_fs = self.fluid_sigma
        self.fluid_sigma = fluid_sigma
        out = self.apply_green_operator(m_flat, spatial_shape, sp_zyx, alpha=alpha, s=s, border_width=border_width)
        self.fluid_sigma = old_fs
        return out.reshape(orig_shape)

    def spectral_jacobian(self, v, vel_shape, spacing_zyx):
        """
        Compute spatial Jacobian Dv where Dv[..., i, j] = d v_i / d x_j using FFT.
        """
        device = v.device
        dtype = v.dtype
        dim = self.dim
        
        k_axes = []
        for d in range(dim):
            n_d = vel_shape[d]
            sp_d = spacing_zyx[d]
            if d == dim - 1:
                k_d = torch.fft.rfftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            else:
                k_d = torch.fft.fftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            k_axes.append(k_d)
            
        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        
        spatial_dims = tuple(range(2, 2 + dim))
        if dim == 3:
            v_cf = v.permute(0, 4, 1, 2, 3)
        else:
            v_cf = v.permute(0, 3, 1, 2)
            
        v_fft = torch.fft.rfftn(v_cf.to(torch.float32), dim=spatial_dims)
        
        Dv_list = []
        for d in range(dim):
            k_d = k_mesh[d].unsqueeze(0).unsqueeze(0).to(torch.float32)
            dv_d_fft = 1j * k_d * v_fft
            dv_d_cf = torch.fft.irfftn(dv_d_fft, s=vel_shape, dim=spatial_dims).to(dtype=dtype)
            Dv_list.append(dv_d_cf)
            
        if dim == 3:
            Dv_stacked = torch.stack(Dv_list, dim=-1).permute(0, 2, 3, 4, 1, 5)
        else:
            Dv_stacked = torch.stack(Dv_list, dim=-1).permute(0, 2, 3, 1, 4)
            
        return Dv_stacked

    def _compute_jacobian(self, v, spacing_zyx):
        # v: (1, *spatial, dim) in ZYX order
        # Returns: (1, *spatial, dim, dim) where [i,j] = dv_i/dx_j
        # Uses proper boundary handling (replicate padding) instead of torch.roll
        # wrap-around, which corrupts boundary voxels by mixing opposite-side values.
        dim = v.shape[-1]
        Dv = torch.zeros(*v.shape, dim, device=v.device, dtype=v.dtype)
        for d in range(dim):
            n = v.shape[d + 1]
            h = spacing_zyx[d]
            s = [slice(None)] * v.ndim
            
            s_center = list(s); s_center[d+1] = slice(1, n-1)
            s_fwd = list(s); s_fwd[d+1] = slice(2, n)
            s_bwd = list(s); s_bwd[d+1] = slice(0, n-2)
            Dv[tuple(s_center)][..., :, d] = (v[tuple(s_fwd)] - v[tuple(s_bwd)]) / (2.0 * h)
            
            s_0 = list(s); s_0[d+1] = slice(0, 1)
            s_1 = list(s); s_1[d+1] = slice(1, 2)
            Dv[tuple(s_0)][..., :, d] = (v[tuple(s_1)] - v[tuple(s_0)]) / h
            
            s_last = list(s); s_last[d+1] = slice(n-1, n)
            s_prev = list(s); s_prev[d+1] = slice(n-2, n-1)
            Dv[tuple(s_last)][..., :, d] = (v[tuple(s_last)] - v[tuple(s_prev)]) / h
        return Dv

    def epdiff_rhs(self, v, spacing_zyx):
        """
        Evaluate Euler-Poincaré Differential (EPDiff) equation right-hand side.

        RHS = -((Dv)^T v + Dv · v + v * div(v))

        Parameters
        ----------
        v : Tensor
            Velocity field tensor of shape (1, *spatial, dim).
        spacing_zyx : list of float
            Grid spacing in ZYX order.

        Returns
        -------
        Tensor
            EPDiff derivative field of shape (1, *spatial, dim).
        """
        vel_shape = tuple(v.shape[1:-1])
        if getattr(self, 'solver', 'spectral_rk4') in ('spectral', 'spectral_rk4'):
            Dv = self.spectral_jacobian(v, vel_shape, spacing_zyx)
        else:
            Dv = self._compute_jacobian(v, spacing_zyx)
            
        v_in = v.unsqueeze(-1)
        term1 = torch.matmul(Dv.transpose(-1, -2), v_in).squeeze(-1)
        term2 = torch.matmul(Dv, v_in).squeeze(-1)
        div_v = torch.diagonal(Dv, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        term3 = v * div_v
        ad_v = term1 + term2 + term3
        
        if getattr(self, 'solver', 'spectral_rk4') in ('spectral', 'spectral_rk4'):
            return -self.apply_green_operator(ad_v, vel_shape, spacing_zyx)
        else:
            return -ad_v

    def _smooth_field(self, field, sigma, spacing=None):
        if sigma <= 0:
            return field
        if spacing is None:
            spacing = list(reversed(self.spacing))
            
        vel_shape = field.shape[1:-1]
        curr_spacing = [sp * (orig_s / curr_s) for sp, orig_s, curr_s in zip(spacing, self.image_shape, vel_shape)]
        return separable_gaussian_filter(field, sigma=sigma, spacing=curr_spacing, sigma_mode='physical')

    def shoot(self, v0, n_steps, image_shape, spacing_zyx=None, _cached_phys_grid=None, _cached_meta=None):
        """
        Evolve initial velocity field v0 forward along geodesic trajectory.
        """
        device = v0.device
        dtype = v0.dtype
        dt = 1.0 / n_steps
        v = v0.clone()
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        disp = torch.zeros(1, *target_shape, self.dim, device=device, dtype=dtype)
        
        if _cached_phys_grid is not None and _cached_meta is not None:
            phys_grid = _cached_phys_grid
            shape_t, spacing_t, origin_t, direction_t = _cached_meta
        else:
            curr_spacing = [
                sp * (float(orig_s) / float(curr_s))
                for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
            ]
            phys_grid = get_physical_grid_torch(
                target_shape, curr_spacing, self.origin, self.direction,
                device=device, dtype=dtype
            )
            spacing_rev = tuple(reversed(curr_spacing))
            origin_rev = tuple(reversed(self.origin))
            direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
            shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
            spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
            origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
            direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        if spacing_zyx is None:
            spacing_zyx = spacing_t.tolist()
            
        max_cfl_disp = 0.5 * min(spacing_zyx)
        max_v_phys = max_cfl_disp / dt
        sigma_step = self.fluid_sigma / math.sqrt(n_steps) if self.fluid_sigma > 0 else 0.0
        bmask = self._create_boundary_mask(tuple(v.shape[1:-1]), device, dtype, border_width=4)
        
        for step in range(n_steps):
            if getattr(self, 'solver', 'spectral_rk4') in ('spectral_rk4', 'rk4'):
                k1 = self.epdiff_rhs(v, spacing_zyx)
                k2 = self.epdiff_rhs(v + 0.5 * dt * k1, spacing_zyx)
                k3 = self.epdiff_rhs(v + 0.5 * dt * k2, spacing_zyx)
                k4 = self.epdiff_rhs(v + dt * k3, spacing_zyx)
                v = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                rhs = self.epdiff_rhs(v, spacing_zyx)
                rhs_smooth = self._smooth_field(rhs, sigma=sigma_step)
                v = v + dt * rhs_smooth
                
            v_mag = torch.norm(v, dim=-1, keepdim=True)
            v = torch.where(v_mag > max_v_phys, v * (max_v_phys / (v_mag + 1e-8)), v)
            v = v * bmask
            
            # Upsample v to target_shape for advection if sizes mismatch
            if tuple(v.shape[1:-1]) != target_shape:
                if self.dim == 3:
                    v_cf = v.permute(0, 4, 1, 2, 3)
                    v_up_cf = F.interpolate(v_cf, size=target_shape, mode='trilinear', align_corners=True)
                    v_for_advect = v_up_cf.permute(0, 2, 3, 4, 1)
                else:
                    v_cf = v.permute(0, 3, 1, 2)
                    v_up_cf = F.interpolate(v_cf, size=target_shape, mode='bilinear', align_corners=True)
                    v_for_advect = v_up_cf.permute(0, 2, 3, 1)
            else:
                v_for_advect = v
                
            phi_current = phys_grid + disp
            phi_norm = physical_to_normalized_torch_cached(
                phi_current, shape_t, spacing_t, origin_t, direction_t
            )
            
            if self.dim == 3:
                v_advect_cf = v_for_advect.permute(0, 4, 1, 2, 3)
                v_sampled_cf = grid_sample_nd(v_advect_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = v_sampled_cf.permute(0, 2, 3, 4, 1)
            else:
                v_advect_cf = v_for_advect.permute(0, 3, 1, 2)
                v_sampled_cf = grid_sample_nd(v_advect_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = v_sampled_cf.permute(0, 2, 3, 1)
                
            disp = disp + dt * v_sampled
            
        return disp, v

    def forward(self, fixed_image, moving_image, multipoint_loss=None, lncc_window_size=5):
        """
        Forward pass with symmetric dual-momentum shooting, LNCC similarity,
        and inverse identity composition constraint.

        Parameters
        ----------
        fixed_image : Tensor
            Fixed target image tensor (1, 1, *spatial).
        moving_image : Tensor
            Moving source image tensor (1, 1, *spatial).
        multipoint_loss : list of float, optional
            Evaluation timepoints. Default [0.0, 1.0].
        lncc_window_size : int, optional
            LNCC window size. Default 5.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        target_shape = tuple(fixed_image.shape[2:])

        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]

        phys_grid = get_physical_grid_torch(
            target_shape, curr_spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )

        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()

        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        _cached_meta = (shape_t, spacing_t, origin_t, direction_t)

        T_grid = self.affine.get_matrix()

        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, target_shape, curr_spacing, self.origin, self.direction,
            target_shape, curr_spacing, self.origin, self.direction
        )

        coord_perm = list(range(self.dim - 1, -1, -1))
        perm_idx = torch.tensor(coord_perm, device=device)
        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
        t_phys_zyx = t_phys[perm_idx]
        
        M_phys_inv_zyx = torch.inverse(M_phys_zyx)
        t_phys_inv_zyx = -M_phys_inv_zyx @ t_phys_zyx

        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        if tuple(moving_image.shape[2:]) != target_shape:
            moving_matched = F.interpolate(moving_image, size=list(target_shape),
                                           mode=interp_mode, align_corners=True)
        else:
            moving_matched = moving_image

        # 1. Forward shooting (+v0_fwd) -> warp moving to fixed space
        disp_fwd, _ = self.shoot(self.velocity_0_fwd, self.n_steps, target_shape,
                                 spacing_zyx=spacing_rev,
                                 _cached_phys_grid=phys_grid,
                                 _cached_meta=_cached_meta)
        phi_moving = (phys_grid + disp_fwd) @ M_phys_zyx.t() + t_phys_zyx
        phi_norm_fwd = physical_to_normalized_torch_cached(
            phi_moving, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = grid_sample_nd(moving_image, phi_norm_fwd, mode='bilinear', padding_mode='zeros')
        loss_fwd = lncc_loss_nd(fixed_image, moving_warped, window_size=lncc_window_size)

        # 2. Inverse shooting (+v0_inv if symmetric, -v0_fwd if asymmetric) -> warp fixed to moving space
        v0_inv_param = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        disp_inv, _ = self.shoot(v0_inv_param, self.n_steps, target_shape,
                                 spacing_zyx=spacing_rev,
                                 _cached_phys_grid=phys_grid,
                                 _cached_meta=_cached_meta)
        phi_fixed = (phys_grid + disp_inv) @ M_phys_inv_zyx.t() + t_phys_inv_zyx
        phi_norm_inv = physical_to_normalized_torch_cached(
            phi_fixed, shape_t, spacing_t, origin_t, direction_t
        )
        fixed_warped = grid_sample_nd(fixed_image, phi_norm_inv, mode='bilinear', padding_mode='zeros')
        loss_inv = lncc_loss_nd(moving_matched, fixed_warped, window_size=lncc_window_size)

        sim_loss = 0.5 * (loss_fwd + loss_inv)

        # 3. Inverse identity loss || Identity - phi_fwd(phi_inv(x)) ||^2 + || Identity - phi_inv(phi_fwd(x)) ||^2
        if self.symmetric and self.inverse_identity_weight > 0:
            phi_inv_pure = phys_grid + disp_inv
            phi_inv_pure_norm = physical_to_normalized_torch_cached(
                phi_inv_pure, shape_t, spacing_t, origin_t, direction_t
            )
            if self.dim == 3:
                disp_fwd_cf = disp_fwd.permute(0, 4, 1, 2, 3)
                disp_fwd_at_inv_cf = grid_sample_nd(disp_fwd_cf, phi_inv_pure_norm, mode='bilinear', padding_mode='border')
                disp_fwd_at_inv = disp_fwd_at_inv_cf.permute(0, 2, 3, 4, 1)
            else:
                disp_fwd_cf = disp_fwd.permute(0, 3, 1, 2)
                disp_fwd_at_inv_cf = grid_sample_nd(disp_fwd_cf, phi_inv_pure_norm, mode='bilinear', padding_mode='border')
                disp_fwd_at_inv = disp_fwd_at_inv_cf.permute(0, 2, 3, 1)
            comp_disp_1 = disp_inv + disp_fwd_at_inv

            phi_fwd_pure = phys_grid + disp_fwd
            phi_fwd_pure_norm = physical_to_normalized_torch_cached(
                phi_fwd_pure, shape_t, spacing_t, origin_t, direction_t
            )
            if self.dim == 3:
                disp_inv_cf = disp_inv.permute(0, 4, 1, 2, 3)
                disp_inv_at_fwd_cf = grid_sample_nd(disp_inv_cf, phi_fwd_pure_norm, mode='bilinear', padding_mode='border')
                disp_inv_at_fwd = disp_inv_at_fwd_cf.permute(0, 2, 3, 4, 1)
            else:
                disp_inv_cf = disp_inv.permute(0, 3, 1, 2)
                disp_inv_at_fwd_cf = grid_sample_nd(disp_inv_cf, phi_fwd_pure_norm, mode='bilinear', padding_mode='border')
                disp_inv_at_fwd = disp_inv_at_fwd_cf.permute(0, 2, 3, 1)
            comp_disp_2 = disp_fwd + disp_inv_at_fwd

            inv_id_loss = 0.5 * (torch.mean(comp_disp_1 ** 2) + torch.mean(comp_disp_2 ** 2))
        else:
            inv_id_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return sim_loss + self.inverse_identity_weight * inv_id_loss

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
        optimizer_type='cfl',
        cfl_step=0.12,
        fluid_sigmas=None,
        elastic_sigmas=None,
        smooth_every_n=1,
        sigma_mode='voxel',
        **kwargs
    ):
        """
        Multi-resolution optimization loop for Geodesic Shooting.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        
        # Override spatial metadata if provided
        if fixed_spacing is not None: self.spacing = list(fixed_spacing)
        if fixed_origin is not None: self.origin = list(fixed_origin)
        if fixed_direction is not None:
            self.direction = fixed_direction.tolist() if hasattr(fixed_direction, 'tolist') else list(fixed_direction)

        if fluid_sigmas is None:
            fluid_sigmas_input = self.fluid_sigma
        else:
            fluid_sigmas_input = fluid_sigmas

        if elastic_sigmas is None:
            elastic_sigmas_input = self.elastic_sigma
        else:
            elastic_sigmas_input = elastic_sigmas

        total_affine_epochs = sum(affine_epochs) if isinstance(affine_epochs, (list, tuple)) else (affine_epochs if affine_epochs is not None else 0)
        if total_affine_epochs > 0 and getattr(self, 'transform_type', 'Affine') != 'Translation_only':
            if verbose: print("Optimizing affine pre-alignment...")
            aff_optimizer = torch.optim.Adam(self.affine.parameters(), lr=1e-2)
            
            aff_epochs_list = affine_epochs if isinstance(affine_epochs, (list, tuple)) else [affine_epochs] * len(levels)
            
            for idx, level in enumerate(levels):
                curr_aff_epochs = aff_epochs_list[min(idx, len(aff_epochs_list) - 1)]
                if curr_aff_epochs <= 0:
                    continue
                    
                if level > 1:
                    down_shape = [max(8, s // level) for s in self.image_shape]
                    curr_fixed_aff = F.interpolate(fixed_image, size=down_shape, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=True)
                    curr_moving_aff = F.interpolate(moving_image, size=down_shape, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=True)
                else:
                    curr_fixed_aff = fixed_image
                    curr_moving_aff = moving_image

                curr_target_shape = tuple(curr_fixed_aff.shape[2:])
                curr_spacing_aff = [
                    sp * (float(orig_s) / float(curr_s))
                    for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, curr_target_shape)
                ]
                
                phys_grid_aff = get_physical_grid_torch(
                    curr_target_shape, curr_spacing_aff, self.origin, self.direction,
                    device=device, dtype=dtype
                )

                shape_t_aff = torch.tensor(list(curr_target_shape), device=device, dtype=dtype)
                spacing_rev_aff = tuple(reversed(curr_spacing_aff))
                spacing_t_aff = torch.tensor(spacing_rev_aff, device=device, dtype=dtype)
                origin_t_aff = torch.tensor(tuple(reversed(self.origin)), device=device, dtype=dtype)
                direction_t_aff = torch.tensor(np.asarray(self.direction)[::-1, ::-1].copy(), device=device, dtype=dtype)

                for ep in range(curr_aff_epochs):
                    aff_optimizer.zero_grad()
                    T_grid = self.affine.get_matrix()
                    M_phys, t_phys = grid_to_physical_affine_torch(
                        T_grid, curr_target_shape, curr_spacing_aff, self.origin, self.direction,
                        curr_target_shape, curr_spacing_aff, self.origin, self.direction
                    )
                    coord_perm = list(range(self.dim - 1, -1, -1))
                    perm_idx = torch.tensor(coord_perm, device=device)
                    M_phys_zyx = M_phys[perm_idx][:, perm_idx]
                    t_phys_zyx = t_phys[perm_idx]
                    
                    phi_moving_aff = phys_grid_aff @ M_phys_zyx.t() + t_phys_zyx
                    phi_norm_aff = physical_to_normalized_torch_cached(
                        phi_moving_aff, shape_t_aff, spacing_t_aff, origin_t_aff, direction_t_aff
                    )
                    moving_warped_aff = grid_sample_nd(curr_moving_aff, phi_norm_aff, mode='bilinear', padding_mode='zeros')
                    
                    if similarity_metric == 'mattes_mi' or similarity_metric == 'mattes':
                        from .syn import mattes_mi_loss_nd
                        aff_loss = mattes_mi_loss_nd(curr_fixed_aff, moving_warped_aff, num_bins=32, sampling_percentage=0.2)
                    else:
                        aff_loss = lncc_loss_nd(curr_fixed_aff, moving_warped_aff, window_size=5)
                        
                    aff_loss.backward()
                    aff_optimizer.step()
                    self.affine.clamp_parameters()
                
        # Optimize velocity field across pyramid levels
        if verbose: print("Optimizing geodesic shooting...")
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl')).lower()
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        fast_smooth = kwargs.get('fast_smooth', True)
        multipoint_loss = kwargs.get('multipoint_loss', [0.0, 1.0])
        
        # Initialize velocity parameters from image gradients
        self.init_velocities_from_image_gradients(fixed_image, moving_image)

        for idx, level in enumerate(levels):
            epochs = epochs_per_level[min(idx, len(epochs_per_level) - 1)]
            if epochs <= 0:
                continue
                
            curr_vel_shape = [max(8, s // level) for s in self.image_shape]
            self._resize_velocity(curr_vel_shape, device, dtype)
            
            # Setup optimizer parameters
            active_params = [self.velocity_0_fwd]
            if self.symmetric and self.velocity_0_inv is not None:
                active_params.append(self.velocity_0_inv)
                
            if opt_type == 'adam':
                optimizer = torch.optim.Adam(active_params, lr=lr)
            else:
                optimizer = LARS(active_params, lr=lr)
            
            momentum_buffer_fwd = torch.zeros_like(self.velocity_0_fwd.data) if (cfl_momentum > 0 and opt_type == 'cfl') else None
            momentum_buffer_inv = torch.zeros_like(self.velocity_0_inv.data) if (self.symmetric and cfl_momentum > 0 and opt_type == 'cfl') else None
            
            vel_spacing = [sp * (img_dim / vel_dim) for sp, img_dim, vel_dim in zip(self.spacing, self.image_shape, curr_vel_shape)] if sigma_mode == 'physical' else None
            
            curr_fluid_sig = fluid_sigmas_input[min(idx, len(fluid_sigmas_input) - 1)] if isinstance(fluid_sigmas_input, (list, tuple)) else fluid_sigmas_input
            curr_elastic_sig = elastic_sigmas_input[min(idx, len(elastic_sigmas_input) - 1)] if isinstance(elastic_sigmas_input, (list, tuple)) else elastic_sigmas_input
                
            sigma_val = float(curr_fluid_sig) if curr_fluid_sig > 0 else 0.0
            elastic_sigma_val = float(curr_elastic_sig) if curr_elastic_sig > 0 else 0.0
            
            if verbose:
                print(f"Level {level}: {epochs} max epochs, vel_grid={list(curr_vel_shape)} (fluid_sigma={curr_fluid_sig:.2f}, elastic_sigma={curr_elastic_sig:.2f}, mode={sigma_mode})")
            
            if level > 1:
                down_shape = [max(8, s // level) for s in self.image_shape]
                curr_fixed = F.interpolate(fixed_image, size=down_shape, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=True)
                curr_moving = F.interpolate(moving_image, size=down_shape, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=True)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image
            
            curr_target_shape = tuple(curr_fixed.shape[2:])
            curr_spacing = [sp * (float(orig_s) / float(curr_s)) for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, curr_target_shape)]
            
            recent_losses = []
            lncc_ws = 2 * lncc_radius + 1
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                sim_loss = self.forward(curr_fixed, curr_moving, multipoint_loss=multipoint_loss, lncc_window_size=lncc_ws)
                kinetic = torch.mean(self.velocity_0_fwd ** 2)
                if self.symmetric and self.velocity_0_inv is not None:
                    kinetic = 0.5 * (kinetic + torch.mean(self.velocity_0_inv ** 2))
                total_loss = sim_loss + reg_weight * kinetic
                total_loss.backward()
                
                with torch.no_grad():
                    # Sobolev Green's operator frequency preconditioning on parameter gradients
                    for vel_p in active_params:
                        if vel_p.grad is not None and (smooth_every_n <= 1 or epoch % smooth_every_n == 0):
                            sp_vel = vel_spacing if vel_spacing is not None else curr_spacing
                            grad_smoothed = self.apply_green_operator(vel_p.grad, curr_vel_shape, sp_vel)
                            vel_p.grad.copy_(grad_smoothed)
                            
                if opt_type == 'cfl':
                    with torch.no_grad():
                        cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                        effective_cfl = float(cfl_step_val)
                        sp_vel = vel_spacing if vel_spacing is not None else curr_spacing
                        sp_t = torch.tensor(sp_vel, device=device, dtype=dtype)
                        
                        param_mbuf_pairs = [(self.velocity_0_fwd, momentum_buffer_fwd)]
                        if self.symmetric and self.velocity_0_inv is not None:
                            param_mbuf_pairs.append((self.velocity_0_inv, momentum_buffer_inv))
                            
                        for vel_p, m_buf in param_mbuf_pairs:
                            if vel_p.grad is not None:
                                grad = vel_p.grad
                                grad_voxel = grad / sp_t
                                max_g_voxel = torch.sqrt(torch.sum(grad_voxel**2, dim=-1)).max()
                                if max_g_voxel > 1e-8:
                                    update = (effective_cfl / max_g_voxel) * grad
                                    if cfl_momentum > 0 and m_buf is not None:
                                        m_buf.mul_(cfl_momentum).add_(update)
                                        vel_p.data.sub_(m_buf)
                                    else:
                                        vel_p.data.sub_(update)
                else:
                    optimizer.step()

                # Elastic Regularization & Clamping
                with torch.no_grad():
                    vel_clamp_val = float(kwargs.get('velocity_clamp', kwargs.get('clamp', 50.0)))
                    for vel_p in active_params:
                        if elastic_sigma_val > 0:
                            vel_smoothed = separable_gaussian_filter(vel_p, sigma=elastic_sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode)
                            vel_p.copy_(vel_smoothed)
                        vel_p.clamp_(min=-vel_clamp_val, max=vel_clamp_val)
                
                loss_val = float(sim_loss.item())
                recent_losses.append(loss_val)
                if len(recent_losses) > 10: recent_losses.pop(0)

        final_vel_shape = tuple(self.velocity_0_fwd.shape[1:-1])
        if final_vel_shape != tuple(self.image_shape):
            self._resize_velocity(self.image_shape, device, dtype)

    @torch.no_grad()
    def get_forward_warp(self, image_shape=None):
        """Compute forward displacement field (shooting +v0_fwd)."""
        disp, _ = self.shoot(self.velocity_0_fwd, self.n_steps, image_shape)
        return disp
        
    @torch.no_grad()
    def get_inverse_warp(self, image_shape=None):
        """Compute inverse displacement field (shooting +v0_inv if symmetric, else -v0_fwd)."""
        v0_inv = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        disp, _ = self.shoot(v0_inv, self.n_steps, image_shape)
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
    grad_step=0.12,
    flow_sigma=1.0,
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
    optimizer=None,
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
        Present for API consistency with syntx.syn(). Not natively used by SyNGS.
    aff_sampling : int or None, optional
        Present for API consistency with syntx.syn(). Not natively used by SyNGS.
    reg_iterations : list of int or None, optional
        Deformable iterations per pyramid level. Default [150, 150, 0].
    affine_iterations : list of int or None, optional
        Affine iterations per level. Default 100.
    grad_step : float, optional
        CFL voxel bound step size. Default 0.20.
    flow_sigma : float, optional
        Fluid regularization sigma in ITK variance convention (σ² = flow_sigma). Default 3.0.
    total_sigma : float, optional
        Elastic regularization sigma in ITK variance convention. Default 0.0.
    n_steps : int, optional
        Number of EPDiff ODE integration steps. Default 5.
    n_time_steps : int or None, optional
        Present for API consistency with syntx.tvf(). Not used by SyNGS (see n_steps).
    verbose : bool, optional
        If True, print optimization progress. Default False.
    backend : str, optional
        Computation backend ('pytorch' or 'jax'). Default 'pytorch'.
    levels : list of int or None, optional
        Multi-resolution pyramid levels. Default [4, 2, 1].
    cfl_momentum : float, optional
        SGD-style momentum for CFL velocity updates. Default 0.9.
    multipoint_loss : list of float or None, optional
        Evaluation timepoints for loss. Default [0.0, 1.0].
    fast_smooth : bool, optional
        If True, smooth gradients at half resolution. Default True.
    sampling_percentage : float or None, optional
        Present for API consistency with syntx.syn(). Not natively used by SyNGS.
    vgg_layers, vgg_mode, vgg_patch_size, vgg_num_patches, vgg_lncc_window_size : optional
        Present for API consistency with syntx.syn().
    optimizer, optimizer_lr, project_inverse, projection_frequency, interpolator, inverse_method, inverse_steps : optional
        Present for API consistency with syntx.syn().
    **kwargs
        Additional arguments passed to GeodesicShootingModel.fit().

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

    Examples
    --------
    >>> import syntx
    >>> reg = syntx.syngs(fixed=fi, moving=mi)
    >>> warped = reg['warpedmovout']
    >>> transforms = reg['fwdtransforms']
    """
    import tempfile
    import time as _time
    import ants

    t_start = _time.time()

    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    origin = fixed.origin
    direction = fixed.direction

    # --- Defaults matching syntx.syn() ---
    if levels is None:
        if reg_iterations is not None:
            num_levels = len(reg_iterations)
            levels = [2**i for i in range(num_levels)][::-1]
        else:
            levels = [4, 2, 1] if dim == 3 else [8, 4, 2, 1]

    levels_len = len(levels)
    if reg_iterations is None:
        reg_iterations = [150, 150, 0] if dim == 3 else [150, 150, 150, 0]
    if affine_iterations is None:
        affine_iterations = 100

    if multipoint_loss is None:
        multipoint_loss = [0.0, 1.0]

    # --- ANTs flow_sigma is standard deviation, not variance ---
    fluid_sigma_actual = float(flow_sigma) if flow_sigma > 0 else 0.0
    elastic_sigma_actual = float(total_sigma) if total_sigma > 0 else 0.0

    # --- Extract native space moving image (Single Interpolation Policy: NO pre-warping) ---
    init_tx_list = []
    init_M_phys, init_t_phys = None, None
    if initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        # Try to parse the initial transform as a single ANTs affine .mat file
        from .syn import parse_ants_affine
        init_M_phys, init_t_phys = parse_ants_affine(init_tx_list, dim)

    # --- Normalize images (same as registration()) ---
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

    # --- Convert to tensors (ZYX convention, channels-first) ---
    grid_shape_zyx = tuple(reversed(grid_shape))
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    
    from .syn import grid_to_physical_affine

    if backend.lower() == 'pytorch':
        # --- Device selection ---
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

        # --- Initialize model ---
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
                    ).to(device_str)

        # --- Initialize affine from initial_transform (Single Interpolation Policy) ---
        # Maps the ANTs physical affine into grid coordinates via T_init,
        # matching SyN's approach (syn.py lines 1954-1971).
        if init_M_phys is not None:
            with torch.no_grad():
                dtype_dev = torch.float32
                Nx_t = torch.tensor(list(reversed(fixed.shape)), device=device_str, dtype=dtype_dev)
                Sx_t = torch.tensor(list(fixed.spacing), device=device_str, dtype=dtype_dev)
                Ox_t = torch.tensor(list(fixed.origin), device=device_str, dtype=dtype_dev)
                Dx_t = torch.tensor(np.asarray(fixed.direction), device=device_str, dtype=dtype_dev)
                com_fixed_fov = Dx_t @ (Sx_t * (Nx_t - 1) / 2.0) + Ox_t

                Ny_t = torch.tensor(list(reversed(moving.shape)), device=device_str, dtype=dtype_dev)
                Sy_t = torch.tensor(list(moving.spacing), device=device_str, dtype=dtype_dev)
                Oy_t = torch.tensor(list(moving.origin), device=device_str, dtype=dtype_dev)
                Dy_t = torch.tensor(np.asarray(moving.direction), device=device_str, dtype=dtype_dev)
                com_moving_fov = Dy_t @ (Sy_t * (Ny_t - 1) / 2.0) + Oy_t

                H_x = torch.eye(dim + 1, device=device_str, dtype=dtype_dev)
                H_x[:dim, :dim] = Dx_t @ torch.diag(Sx_t) @ torch.diag((Nx_t - 1) / 2.0)
                H_x[:dim, dim] = com_fixed_fov

                H_y = torch.eye(dim + 1, device=device_str, dtype=dtype_dev)
                H_y[:dim, :dim] = Dy_t @ torch.diag(Sy_t) @ torch.diag((Ny_t - 1) / 2.0)
                H_y[:dim, dim] = com_moving_fov

                T_phys = torch.eye(dim + 1, device=device_str, dtype=dtype_dev)
                T_phys[:dim, :dim] = init_M_phys.to(device=device_str, dtype=dtype_dev)
                T_phys[:dim, dim] = init_t_phys.to(device=device_str, dtype=dtype_dev)

                T_init = torch.inverse(H_y) @ T_phys @ H_x
                model.affine.T_init = T_init

            # Affine absorbed into model parameters; do not append to final transform list
            init_tx_list = []
            if verbose:
                print(f"[GeodesicShooting] Initialized affine from initial_transform (T_init absorbed)")

        # --- Fit ---
        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            affine_epochs=affine_iterations,
            lr=kwargs.pop('lr', 0.1),
            reg_weight=kwargs.pop('reg_weight', 0.0),
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            optimizer_type=kwargs.pop('optimizer_type', kwargs.pop('optimizer', 'cfl')),
            cfl_step=grad_step,
            cfl_momentum=cfl_momentum,
            multipoint_loss=multipoint_loss,
            fast_smooth=fast_smooth,
            smooth_pyramid=kwargs.pop('smooth_pyramid', True),
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
            solver=kwargs.pop('solver', 'spectral_rk4'),
        )

        # --- Initialize affine from initial_transform (JAX parity with PyTorch) ---
        if init_M_phys is not None:
            Nx = np.array(list(reversed(fixed.shape)), dtype=np.float32)
            Sx = np.array(list(fixed.spacing), dtype=np.float32)
            Ox = np.array(list(fixed.origin), dtype=np.float32)
            Dx = np.asarray(fixed.direction, dtype=np.float32)
            com_fixed_fov = Dx @ (Sx * (Nx - 1) / 2.0) + Ox

            Ny = np.array(list(reversed(moving.shape)), dtype=np.float32)
            Sy = np.array(list(moving.spacing), dtype=np.float32)
            Oy = np.array(list(moving.origin), dtype=np.float32)
            Dy = np.asarray(moving.direction, dtype=np.float32)
            com_moving_fov = Dy @ (Sy * (Ny - 1) / 2.0) + Oy

            H_x = np.eye(dim + 1, dtype=np.float32)
            H_x[:dim, :dim] = Dx @ np.diag(Sx) @ np.diag((Nx - 1) / 2.0)
            H_x[:dim, dim] = com_fixed_fov

            H_y = np.eye(dim + 1, dtype=np.float32)
            H_y[:dim, :dim] = Dy @ np.diag(Sy) @ np.diag((Ny - 1) / 2.0)
            H_y[:dim, dim] = com_moving_fov

            T_phys = np.eye(dim + 1, dtype=np.float32)
            T_phys[:dim, :dim] = init_M_phys.numpy() if hasattr(init_M_phys, 'numpy') else np.asarray(init_M_phys)
            T_phys[:dim, dim] = init_t_phys.numpy() if hasattr(init_t_phys, 'numpy') else np.asarray(init_t_phys)

            T_init_jax = jnp.array(np.linalg.inv(H_y) @ T_phys @ H_x)
            model.T_init = T_init_jax
            model.affine_params['T_init'] = T_init_jax

            init_tx_list = []
            if verbose:
                print(f"[SyNGS-JAX] Initialized affine from initial_transform (T_init absorbed)")

        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            affine_epochs=affine_iterations,
            lr=kwargs.pop('lr', 0.1),
            reg_weight=kwargs.pop('reg_weight', 0.0),
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            optimizer_type=kwargs.pop('optimizer_type', kwargs.pop('optimizer', 'cfl')),
            cfl_step=grad_step,
            cfl_momentum=cfl_momentum,
            multipoint_loss=multipoint_loss,
            fast_smooth=fast_smooth,
            smooth_pyramid=kwargs.pop('smooth_pyramid', True),
            **kwargs
        )

        fwd_disp = np.array(model.get_forward_warp(image_shape=grid_shape_zyx))
        inv_disp = np.array(model.get_inverse_warp(image_shape=grid_shape_zyx))
        fwd_np = fwd_disp.squeeze(0)
        inv_np = inv_disp.squeeze(0)

        T_grid = np.array(get_affine_matrix_jax(model.affine_params, dim, 'Affine'))
    else:
        raise ValueError(f"Unknown backend: {backend}")
    fwd_ants_np = fwd_np[..., ::-1].copy()
    inv_ants_np = inv_np[..., ::-1].copy()

    # Transpose spatial dims back to ANTs native order
    dim_order = list(range(dim - 1, -1, -1)) + [dim]
    fwd_ants_np = np.ascontiguousarray(fwd_ants_np.transpose(dim_order))
    inv_ants_np = np.ascontiguousarray(inv_ants_np.transpose(dim_order))

    fwd_img = ants.from_numpy(fwd_ants_np, origin=origin, spacing=spacing,
                               direction=direction, has_components=True)
    inv_img = ants.from_numpy(inv_ants_np, origin=origin, spacing=spacing,
                               direction=direction, has_components=True)

    fwd_file = tempfile.NamedTemporaryFile(suffix='_syngs_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_syngs_inv_Warp.nii.gz', delete=False).name
    ants.image_write(fwd_img, fwd_file)
    ants.image_write(inv_img, inv_file)

    # Export affine transform
    M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving)

    affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    tx_fwd = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx_fwd.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
    tx_fwd.set_fixed_parameters(np.zeros(dim))
    ants.write_transform(tx_fwd, affine_file)

    # Inverse affine
    affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    M_phys_inv = np.linalg.inv(M_phys)
    t_phys_inv = -M_phys_inv @ t_phys
    tx_inv = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx_inv.set_parameters(np.concatenate([M_phys_inv.T.ravel(), t_phys_inv]))
    tx_inv.set_fixed_parameters(np.zeros(dim))
    ants.write_transform(tx_inv, affine_inv_file)

    # Build transform lists (same order as registration())
    if sum(reg_iterations) > 0:
        fwd_transforms = [fwd_file, affine_file] + init_tx_list
        inv_transforms = init_tx_list + [affine_file, inv_file]
        whichtoinvert_inv = [True] * len(init_tx_list) + [True, False]
    else:
        fwd_transforms = [affine_file] + init_tx_list
        inv_transforms = init_tx_list + [affine_file]
        whichtoinvert_inv = [True] * (len(init_tx_list) + 1)

    # Generate warped output images (same as registration())
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
            optimizer_type="CFL" if optimizer is None else optimizer,
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
            cfl_momentum=cfl_momentum,
            multipoint_loss=multipoint_loss,
            fast_smooth=fast_smooth,
            n_time_steps=n_time_steps,
            n_steps=n_steps
        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass

    return ret_dict
