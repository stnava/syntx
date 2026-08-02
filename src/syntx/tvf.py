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
    mattes_mi_loss_nd,
    grid_sample_nd,
    _spatial_jacobian_nd
)

class LARS(torch.optim.Optimizer):
    """PyTorch implementation of LARS (Layer-wise Adaptive Rate Scaling)."""
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

class TVFModel(nn.Module):
    """
    Time-Varying Velocity Field (TVF) Registration Model.

    Parameters
    ----------
    dim : int
        Spatial dimensionality (2 or 3).
    image_shape : tuple of int
        Image grid shape in ZYX order.
    velocity_shape : tuple of int
        Velocity field grid shape in ZYX order.
    n_time_steps : int, optional
        Number of time keyframes T. Default 4.
    spacing : list of float, optional
        Voxel spacing in XYZ order. Default 1.0 per dimension.
    origin : list of float, optional
        Image origin in XYZ order. Default 0.0 per dimension.
    direction : list of list of float, optional
        Direction matrix. Default identity.
    fluid_sigma : float, optional
        Fluid regularization standard deviation. Default 1.0.
    elastic_sigma : float, optional
        Elastic regularization standard deviation. Default 0.0.
    transform_type : str, optional
        Affine transform type ('Affine', 'Rigid', 'Translation'). Default 'Affine'.
    solver : str, optional
        ODE solver ('euler' or 'rk4'). Default 'euler'.
    integration_steps_per_interval : int, optional
        Sub-steps per time interval. Default 1.
    antisymmetric : bool, optional
        Enforce anti-symmetry v(t_k) = -v(t_{K-1-k}). Default False.
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
        elastic_sigma=0.0,
        transform_type='Affine',
        solver='euler',
        integration_steps_per_interval=1,
        antisymmetric=True
    ):
        super().__init__()
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps
        self.antisymmetric = antisymmetric
        
        self.spacing = spacing if spacing is not None else [1.0] * dim
        self.origin = origin if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction
        else:
            self.direction = np.eye(dim).tolist()
            
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.solver = solver
        self.integration_steps_per_interval = integration_steps_per_interval
        
        # Velocity field parameter: (T, 1, *velocity_shape, dim)
        self.velocity = nn.Parameter(torch.zeros(n_time_steps, 1, *self.velocity_shape, self.dim))
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)

    def project_antisymmetric(self):
        """
        Project keyframe velocity fields onto the temporally anti-symmetric subspace:
        v(t_k) <- 0.5 * (v(t_k) - v(t_{K-1-k}))
        Ensures exact geodesic symmetry across time: v(x, 1-t) = -v(x, t).
        """
        with torch.no_grad():
            v_flipped = torch.flip(self.velocity.data, dims=[0])
            self.velocity.data = 0.5 * (self.velocity.data - v_flipped)

    def _resize_velocity(self, new_shape, device=None, dtype=None):
        """
        Resize the velocity parameter to a new spatial shape using trilinear/bilinear
        interpolation. Preserves learned deformations when transitioning between
        pyramid-proportional velocity grid resolutions.
        
        Args:
            new_shape: Target spatial shape tuple, e.g. (48, 48, 48)
            device: Target device
            dtype: Target dtype
        """
        new_shape = tuple(new_shape)
        old_shape = tuple(self.velocity.shape[2:-1])  # (T, 1, *spatial, dim)
        
        if new_shape == old_shape:
            return
            
        with torch.no_grad():
            old_vel = self.velocity.data  # (T, 1, *spatial, dim)
            T = old_vel.shape[0]
            
            if self.dim == 3:
                # (T, 1, D, H, W, 3) → (T, 3, D, H, W) for F.interpolate
                old_cf = old_vel.squeeze(1).permute(0, 4, 1, 2, 3)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='trilinear', align_corners=True)
                # (T, 3, D', H', W') → (T, 1, D', H', W', 3)
                new_vel = new_cf.permute(0, 2, 3, 4, 1).unsqueeze(1)
            else:
                # (T, 1, H, W, 2) → (T, 2, H, W) for F.interpolate
                old_cf = old_vel.squeeze(1).permute(0, 3, 1, 2)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='bilinear', align_corners=True)
                # (T, 2, H', W') → (T, 1, H', W', 2)
                new_vel = new_cf.permute(0, 2, 3, 1).unsqueeze(1)
            
            if device is not None:
                new_vel = new_vel.to(device=device)
            if dtype is not None:
                new_vel = new_vel.to(dtype=dtype)
                
            self.velocity = nn.Parameter(new_vel.contiguous())

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
        Cubic B-spline (Catmull-Rom) temporal interpolation between discrete velocity keyframes.
        
        Args:
            t: Continuous time in [0, 1]
            velocity_cf: Velocity in channels-first format (T, 1, dim, *velocity_shape)
            
        Returns:
            Velocity at time t (1, dim, *velocity_shape)
        """
        T = self.n_time_steps
        if T == 1:
            return velocity_cf[0]
        if T == 2:
            t_scaled = t
            return (1.0 - t_scaled) * velocity_cf[0] + t_scaled * velocity_cf[1]
            
        t_scaled = t * (T - 1)
        i = math.floor(t_scaled)
        s = t_scaled - i
        if i >= T - 1:
            i = T - 2
            s = 1.0
            
        i0 = max(0, min(T - 1, i - 1))
        i1 = max(0, min(T - 1, i))
        i2 = max(0, min(T - 1, i + 1))
        i3 = max(0, min(T - 1, i + 2))
        
        s2 = s * s
        s3 = s2 * s
        
        c0 = 0.5 * (-s3 + 2.0 * s2 - s)
        c1 = 0.5 * (3.0 * s3 - 5.0 * s2 + 2.0)
        c2 = 0.5 * (-3.0 * s3 + 4.0 * s2 + s)
        c3 = 0.5 * (s3 - s2)
        
        return c0 * velocity_cf[i0] + c1 * velocity_cf[i1] + c2 * velocity_cf[i2] + c3 * velocity_cf[i3]

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

    def _apply_sobolev_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        if fluid_sigma <= 0:
            return m
        device = m.device
        dtype = m.dtype
        dim = self.dim
        # Auto-scale alpha by dimension to prevent 3D frequency over-smoothing
        if alpha is not None:
            alpha_val = float(alpha) / float(dim)
        else:
            alpha_val = float(fluid_sigma) / (2.0 * float(dim))
        s = 2.0
        
        spatial_shape = m.shape[1:-1]
        k_axes = []
        for d in range(dim):
            n_d = spatial_shape[d]
            if d == dim - 1:
                k_d = torch.fft.rfftfreq(n_d, device=device) * (2.0 * math.pi)
            else:
                k_d = torch.fft.fftfreq(n_d, device=device) * (2.0 * math.pi)
            k_axes.append(k_d)
            
        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        k_sq = sum(k_j ** 2 for k_j in k_mesh)
        K_fourier = 1.0 / ((1.0 + alpha_val * k_sq) ** s)
        
        spatial_dims = tuple(range(2, 2 + dim))
        m_cf = m.movedim(-1, 1)
        m_fft = torch.fft.rfftn(m_cf.to(torch.float32), dim=spatial_dims)
        K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
        v_fft = m_fft * K_bc
        v_cf = torch.fft.irfftn(v_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
        
        return v_cf.movedim(1, -1)

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

    def _interpolate_velocity_fine(self, t, velocity_fine_cf):
        """
        Cubic B-spline (Catmull-Rom) temporal interpolation between pre-upsampled velocity keyframes.
        
        Args:
            t: Continuous time in [0, 1]
            velocity_fine_cf: List of T pre-upsampled velocity tensors (1, dim, *target_shape)
            
        Returns:
            Velocity at time t (1, dim, *target_shape)
        """
        T = len(velocity_fine_cf)
        if T == 1:
            return velocity_fine_cf[0]
        if T == 2:
            t_scaled = t
            return (1.0 - t_scaled) * velocity_fine_cf[0] + t_scaled * velocity_fine_cf[1]
            
        t_scaled = t * (T - 1)
        i = math.floor(t_scaled)
        s = t_scaled - i
        if i >= T - 1:
            i = T - 2
            s = 1.0
            
        i0 = max(0, min(T - 1, i - 1))
        i1 = max(0, min(T - 1, i))
        i2 = max(0, min(T - 1, i + 1))
        i3 = max(0, min(T - 1, i + 2))
        
        s2 = s * s
        s3 = s2 * s
        
        c0 = 0.5 * (-s3 + 2.0 * s2 - s)
        c1 = 0.5 * (3.0 * s3 - 5.0 * s2 + 2.0)
        c2 = 0.5 * (-3.0 * s3 + 4.0 * s2 + s)
        c3 = 0.5 * (s3 - s2)
        
        return c0 * velocity_fine_cf[i0] + c1 * velocity_fine_cf[i1] + c2 * velocity_fine_cf[i2] + c3 * velocity_fine_cf[i3]

    def integrate(self, t_start, t_end, velocity=None, n_steps=None, image_shape=None,
                  _cached_phys_grid=None, _cached_meta=None):
        """
        Integrates the velocity field ODE from t_start to t_end.
        
        Performance: velocity keyframes are upsampled to target_shape ONCE before
        the integration step loop, eliminating redundant F.interpolate calls from
        the inner RK4/Euler loop.
        """
        if velocity is None:
            velocity = self.velocity

        device = velocity.device
        dtype = velocity.dtype
        
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        
        if n_steps is None:
            n_steps = self.n_time_steps * self.integration_steps_per_interval
            
        dt = (t_end - t_start) / max(1, n_steps)
        
        # Use cached grid and metadata if provided, otherwise compute
        if _cached_phys_grid is not None and _cached_meta is not None:
            phys_grid = _cached_phys_grid
            shape_t, spacing_t, origin_t, direction_t = _cached_meta
        else:
            # Calculate spacing for current shape
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
        
        phi_t = phys_grid.clone()
        
        # Convert velocity to channels-first: (T, 1, dim, *velocity_shape)
        if self.dim == 2:
            velocity_cf = velocity.permute(0, 1, 4, 2, 3)
        else:
            velocity_cf = velocity.permute(0, 1, 5, 2, 3, 4)
        
        # === CRITICAL OPTIMIZATION ===
        # Pre-upsample ALL velocity keyframes to target_shape ONCE,
        # instead of calling F.interpolate inside every integration step.
        # With RK4 (T=8, steps=16, 4 stages), this eliminates ~128 F.interpolate
        # calls per integrate() and replaces them with T=8 upfront calls.
        velocity_fine_cf = []
        for t_idx in range(self.n_time_steps):
            # velocity_cf[t_idx] has shape (1, dim, *vel_shape) — correct for F.interpolate
            v_fine = self.upsample_velocity(velocity_cf[t_idx], target_shape)
            velocity_fine_cf.append(v_fine)
            
        for step in range(n_steps):
            t_current = t_start + step * dt
            
            if self.solver == 'euler':
                v_fine_cf = self._interpolate_velocity_fine(t_current, velocity_fine_cf)
                
                phi_norm = physical_to_normalized_torch_cached(
                    phi_t, shape_t, spacing_t, origin_t, direction_t
                )
                
                v_sampled_cf = grid_sample_nd(v_fine_cf, phi_norm, mode='bilinear', padding_mode='border')
                if self.dim == 2:
                    v_sampled = v_sampled_cf.permute(0, 2, 3, 1)
                else:
                    v_sampled = v_sampled_cf.permute(0, 2, 3, 4, 1)
                phi_t = phi_t + v_sampled * dt
                
            elif self.solver == 'rk4':
                def eval_v(t, current_phi):
                    v_fine_cf_t = self._interpolate_velocity_fine(t, velocity_fine_cf)
                    
                    phi_norm_t = physical_to_normalized_torch_cached(
                        current_phi, shape_t, spacing_t, origin_t, direction_t
                    )
                    v_sampled_cf = grid_sample_nd(v_fine_cf_t, phi_norm_t, mode='bilinear', padding_mode='border')
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

    def forward(self, fixed_image, moving_image, velocity=None, affine_params=None, multipoint_loss=[0.0, 1.0], lncc_window_size=5):
        """
        Registration forward pass supporting arbitrary multi-point LNCC evaluation timepoints t in [0, 1].
        Default: multipoint_loss = [0.5] (SyNTVF geodesic midpoint evaluation).
        Triplet: multipoint_loss = [0.0, 0.5, 1.0] (anchors fixed t=0, midpoint t=0.5, and moving t=1 space).
        
        Args:
            lncc_window_size: LNCC window size (default 5, matching SyN's syn_sampling=2).
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        target_shape = tuple(fixed_image.shape[2:])

        if isinstance(multipoint_loss, bool):
            eval_points = [0.0, 0.5, 1.0] if multipoint_loss else [0.5]
        elif isinstance(multipoint_loss, (list, tuple)):
            eval_points = list(multipoint_loss)
        else:
            eval_points = [float(multipoint_loss)]

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
        
        # Cache metadata for passing into integrate() to avoid redundant tensor creation
        _cached_meta = (shape_t, spacing_t, origin_t, direction_t)

        if affine_params is not None:
            T_grid = affine_params
        else:
            T_grid = self.affine.get_matrix()

        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, target_shape, curr_spacing, self.origin, self.direction,
            target_shape, curr_spacing, self.origin, self.direction
        )

        coord_perm = list(range(self.dim - 1, -1, -1))
        perm_idx = torch.tensor(coord_perm, device=device)
        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
        t_phys_zyx = t_phys[perm_idx]

        losses = []
        for t_k in eval_points:
            t_k = float(t_k)
            if abs(t_k - 0.0) < 1e-5:
                # Calculate inverse identity composition penalty if both forward and inverse integrations are computed
                phi_0_to_1 = self.integrate(0.0, 1.0, velocity=velocity, image_shape=target_shape,
                                            _cached_phys_grid=phys_grid, _cached_meta=_cached_meta)
                phi_1_to_0 = self.integrate(1.0, 0.0, velocity=velocity, image_shape=target_shape,
                                            _cached_phys_grid=phys_grid, _cached_meta=_cached_meta)

                # Compute bidirectional composition penalties:
                # Direction 1: u_inv + u_fwd(x + u_inv)
                phi_1_to_0_norm = physical_to_normalized_torch_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                # Direction 2: u_fwd + u_inv(x + u_fwd)
                phi_0_to_1_norm = physical_to_normalized_torch_cached(
                    phys_grid + phi_0_to_1, shape_t, spacing_t, origin_t, direction_t
                )

                if self.dim == 3:
                    u_fwd_cf = phi_0_to_1.permute(0, 4, 1, 2, 3)
                    u_fwd_at_inv_cf = grid_sample_nd(u_fwd_cf, phi_1_to_0_norm, mode='bilinear', padding_mode='border')
                    u_fwd_at_inv = u_fwd_at_inv_cf.permute(0, 2, 3, 4, 1)

                    u_inv_cf = phi_1_to_0.permute(0, 4, 1, 2, 3)
                    u_inv_at_fwd_cf = grid_sample_nd(u_inv_cf, phi_0_to_1_norm, mode='bilinear', padding_mode='border')
                    u_inv_at_fwd = u_inv_at_fwd_cf.permute(0, 2, 3, 4, 1)
                else:
                    u_fwd_cf = phi_0_to_1.permute(0, 3, 1, 2)
                    u_fwd_at_inv_cf = grid_sample_nd(u_fwd_cf, phi_1_to_0_norm, mode='bilinear', padding_mode='border')
                    u_fwd_at_inv = u_fwd_at_inv_cf.permute(0, 2, 3, 1)

                    u_inv_cf = phi_1_to_0.permute(0, 3, 1, 2)
                    u_inv_at_fwd_cf = grid_sample_nd(u_inv_cf, phi_0_to_1_norm, mode='bilinear', padding_mode='border')
                    u_inv_at_fwd = u_inv_at_fwd_cf.permute(0, 2, 3, 1)

                inv_id_err_1 = phi_1_to_0 + u_fwd_at_inv
                inv_id_err_2 = phi_0_to_1 + u_inv_at_fwd
                inv_id_loss = 0.5 * (torch.mean(inv_id_err_1 ** 2) + torch.mean(inv_id_err_2 ** 2))

                # Forward warping
                phi_moving_affine_end = (phys_grid + phi_0_to_1) @ M_phys_zyx.t() + t_phys_zyx
                phi_norm_end = physical_to_normalized_torch_cached(
                    phi_moving_affine_end, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = grid_sample_nd(moving_image, phi_norm_end, mode='bilinear', padding_mode='zeros')
                loss_fwd = lncc_loss_nd(fixed_image, moving_warped, window_size=lncc_window_size)

                # Inverse warping
                phi_fixed_norm_end = physical_to_normalized_torch_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped = grid_sample_nd(fixed_image, phi_fixed_norm_end, mode='bilinear', padding_mode='zeros')
                phi_moving_identity = phys_grid @ M_phys_zyx.t() + t_phys_zyx
                phi_moving_identity_norm = physical_to_normalized_torch_cached(
                    phi_moving_identity, shape_t, spacing_t, origin_t, direction_t
                )
                moving_affine = grid_sample_nd(moving_image, phi_moving_identity_norm, mode='bilinear', padding_mode='zeros')
                loss_inv = lncc_loss_nd(fixed_warped, moving_affine, window_size=lncc_window_size)

                inv_id_weight = float(getattr(self, 'inverse_identity_weight', 0.05))
                return 0.5 * (loss_fwd + loss_inv) + inv_id_weight * inv_id_loss
            else:
                # Midpoint or Intermediate Space t_k
                phi_tk_to_fixed = self.integrate(t_k, 0.0, velocity=velocity, image_shape=target_shape,
                                                 _cached_phys_grid=phys_grid, _cached_meta=_cached_meta)
                phi_tk_to_moving = self.integrate(t_k, 1.0, velocity=velocity, image_shape=target_shape,
                                                  _cached_phys_grid=phys_grid, _cached_meta=_cached_meta)

                phi_fixed_norm_tk = physical_to_normalized_torch_cached(
                    phys_grid + phi_tk_to_fixed, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped_tk = grid_sample_nd(fixed_image, phi_fixed_norm_tk, mode='bilinear', padding_mode='zeros')

                phi_moving_affine_tk = (phys_grid + phi_tk_to_moving) @ M_phys_zyx.t() + t_phys_zyx
                phi_moving_norm_tk = physical_to_normalized_torch_cached(
                    phi_moving_affine_tk, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped_tk = grid_sample_nd(moving_image, phi_moving_norm_tk, mode='bilinear', padding_mode='zeros')
                losses.append(lncc_loss_nd(fixed_warped_tk, moving_warped_tk, window_size=lncc_window_size))

        return torch.stack(losses).mean()

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[100, 100, 50],
        affine_epochs=100,
        similarity_metric='lncc',
        lncc_radius=4,
        lr=0.15,
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
            
        if isinstance(affine_epochs, (list, tuple)):
            affine_epochs = sum(affine_epochs)
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
                moving_warped = grid_sample_nd(moving_image, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                
                aff_metric = kwargs.get('aff_metric', 'mattes_mi')
                if aff_metric.lower() in ('mattes_mi', 'mattes', 'mi'):
                    mattes_bins = int(kwargs.get('mattes_bins', kwargs.get('num_bins', 32)))
                    sampling_pct = float(kwargs.get('sampling_percentage', 0.2))
                    loss = mattes_mi_loss_nd(fixed_image, moving_warped, num_bins=mattes_bins, sampling_percentage=sampling_pct)
                else:
                    loss = lncc_loss_nd(fixed_image, moving_warped, window_size=2*lncc_radius+1)
                loss.backward()
                optimizer_aff.step()
                self.affine.clamp_parameters()
                
        # Optimize velocity field across pyramid levels
        if verbose: print("Optimizing TVF...")
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl')).lower()
        trust_coeff = kwargs.get('trust_coefficient', kwargs.get('trust', 0.05))
        
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)
        multipoint_loss = kwargs.get('multipoint_loss', [0.0, 1.0])
        
        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        
        sigma_mode = kwargs.get('sigma_mode', 'voxel')
        use_analytical_gradients = kwargs.get('use_analytical_gradients', False)
        
        # CFL momentum for faster convergence (default 0.9, set 0.0 to disable)
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        momentum_buffer = None  # Initialized per-level
        
        # Gradient smoothing frequency: smooth every N epochs (1=every epoch, default)
        # Higher values reduce the dominant smoothing bottleneck at cost of noise
        smooth_every_n = int(kwargs.get('smooth_every_n', 1))
        
        # Fast smooth: downsample gradients to half resolution before smoothing (9.4x faster)
        # Approximate but sufficient for gradient direction estimation
        fast_smooth = bool(kwargs.get('fast_smooth', True))

        # Compute pyramid-proportional velocity shapes for each level
        # velocity_shape is the MAX (finest) grid; coarser levels use proportionally smaller grids
        max_vel_shape = self.velocity_shape  # e.g., (96, 96, 96)
        
        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue
            
            # --- Velocity grid matches the downsampled image shape at current level ---
            # This ensures the velocity field has the same spatial resolution as the
            # working image, avoiding upsampling artifacts during integration.
            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            prev_vel_shape = tuple(self.velocity.shape[2:-1])
            
            if curr_vel_shape != prev_vel_shape:
                self._resize_velocity(curr_vel_shape, device, dtype)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")
            
            curr_spacing = [sp * level for sp in self.spacing]
            
            # Create optimizer fresh for this level (velocity parameter may have changed)
            if opt_type == 'lars':
                optimizer = LARS([self.velocity], lr=lr, trust_coefficient=trust_coeff)
            else:
                optimizer = torch.optim.Adam([self.velocity], lr=lr)
            
            # Reset momentum buffer for this level
            if cfl_momentum > 0 and opt_type == 'cfl':
                momentum_buffer = torch.zeros_like(self.velocity.data)
            
            # Compute vel_spacing for physical-mode smoothing at current velocity resolution
            vel_spacing = [sp * (img_dim / vel_dim) for sp, img_dim, vel_dim in zip(self.spacing, self.image_shape, curr_vel_shape)] if sigma_mode == 'physical' else None
            
            if isinstance(fluid_sigmas_input, (list, tuple)):
                curr_fluid_sig = fluid_sigmas_input[min(idx, len(fluid_sigmas_input) - 1)]
            else:
                curr_fluid_sig = fluid_sigmas_input

            if isinstance(elastic_sigmas_input, (list, tuple)):
                curr_elastic_sig = elastic_sigmas_input[min(idx, len(elastic_sigmas_input) - 1)]
            else:
                curr_elastic_sig = elastic_sigmas_input
                
            sigma_val = math.sqrt(curr_fluid_sig) if curr_fluid_sig > 0 else 0.0
            elastic_sigma_val = math.sqrt(curr_elastic_sig) if curr_elastic_sig > 0 else 0.0
            
            if verbose:
                print(f"Level {level}: {epochs} max epochs, vel_grid={list(curr_vel_shape)} (fluid_sigma={curr_fluid_sig:.2f}, elastic_sigma={curr_elastic_sig:.2f}, mode={sigma_mode})")
            
            if level > 1:
                down_shape = [max(8, s // level) for s in self.image_shape]
                smooth_pyr = kwargs.get('smooth_pyramid', kwargs.get('pre_smooth', False))
                if smooth_pyr:
                    aa_sigma = float(kwargs.get('aa_sigma', math.log2(level)))
                    from syntx.syn import get_cached_gaussian_kernel_1d
                    k1d = get_cached_gaussian_kernel_1d(aa_sigma, device, dtype).squeeze(0)
                    pad_size = k1d.shape[-1] // 2
                    if self.dim == 3:
                        kz = k1d.view(1, 1, -1, 1, 1)
                        ky = k1d.view(1, 1, 1, -1, 1)
                        kx = k1d.view(1, 1, 1, 1, -1)
                        def smooth_3d(img):
                            x = F.pad(img, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
                            x = F.conv3d(x, kz, groups=1)
                            x = F.pad(x, (0, 0, pad_size, pad_size, 0, 0), mode='replicate')
                            x = F.conv3d(x, ky, groups=1)
                            x = F.pad(x, (pad_size, pad_size, 0, 0, 0, 0), mode='replicate')
                            x = F.conv3d(x, kx, groups=1)
                            return x
                        fixed_smooth = smooth_3d(fixed_image)
                        moving_smooth = smooth_3d(moving_image)
                    else:
                        ky = k1d.view(1, 1, -1, 1)
                        kx = k1d.view(1, 1, 1, -1)
                        def smooth_2d(img):
                            x = F.pad(img, (0, 0, pad_size, pad_size), mode='replicate')
                            x = F.conv2d(x, ky, groups=1)
                            x = F.pad(x, (pad_size, pad_size, 0, 0), mode='replicate')
                            x = F.conv2d(x, kx, groups=1)
                            return x
                        fixed_smooth = smooth_2d(fixed_image)
                        moving_smooth = smooth_2d(moving_image)
                else:
                    fixed_smooth = fixed_image
                    moving_smooth = moving_image
                curr_fixed = F.interpolate(fixed_smooth, size=down_shape, mode=interp_mode, align_corners=True)
                curr_moving = F.interpolate(moving_smooth, size=down_shape, mode=interp_mode, align_corners=True)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image
            
            recent_losses = []
            lncc_ws = 2 * lncc_radius + 1
            
            # Pre-compute image spatial Jacobians for analytical gradient mode
            if use_analytical_gradients:
                grad_I_curr = _spatial_jacobian_nd(
                    curr_fixed.movedim(1, -1),
                    physical_spacing=tuple(reversed(curr_spacing))
                ).squeeze(-2)
                grad_J_curr = _spatial_jacobian_nd(
                    curr_moving.movedim(1, -1),
                    physical_spacing=tuple(reversed(curr_spacing))
                ).squeeze(-2)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if use_analytical_gradients:
                    # === Analytical gradient mode ===
                    # Step 1: Forward pass under no_grad to get warped images
                    with torch.no_grad():
                        target_shape = tuple(curr_fixed.shape[2:])
                        curr_spacing_list = [
                            sp * (float(orig_s) / float(curr_s))
                            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
                        ]
                        phys_grid = get_physical_grid_torch(
                            target_shape, curr_spacing_list, self.origin, self.direction,
                            device=device, dtype=dtype
                        )
                        spacing_rev = tuple(reversed(curr_spacing_list))
                        origin_rev = tuple(reversed(self.origin))
                        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
                        shape_t_ag = torch.tensor(list(target_shape), device=device, dtype=dtype)
                        spacing_t_ag = torch.tensor(spacing_rev, device=device, dtype=dtype)
                        origin_t_ag = torch.tensor(origin_rev, device=device, dtype=dtype)
                        direction_t_ag = torch.tensor(direction_rev, device=device, dtype=dtype)
                        _cached_meta_ag = (shape_t_ag, spacing_t_ag, origin_t_ag, direction_t_ag)
                        
                        affine_params = self.affine.get_matrix()
                        M_phys, t_phys_a = grid_to_physical_affine_torch(
                            affine_params, target_shape, curr_spacing_list, self.origin, self.direction,
                            target_shape, curr_spacing_list, self.origin, self.direction
                        )
                        coord_perm = list(range(self.dim - 1, -1, -1))
                        perm_idx = torch.tensor(coord_perm, device=device)
                        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
                        t_phys_zyx = t_phys_a[perm_idx]
                        
                        # Warp fixed and moving to midpoint (t=0.5)
                        phi_05_to_0 = self.integrate(0.5, 0.0, image_shape=target_shape,
                                                     _cached_phys_grid=phys_grid, _cached_meta=_cached_meta_ag)
                        phi_05_to_1 = self.integrate(0.5, 1.0, image_shape=target_shape,
                                                     _cached_phys_grid=phys_grid, _cached_meta=_cached_meta_ag)
                        
                        # Warp fixed to midpoint
                        phi_fixed_norm = physical_to_normalized_torch_cached(
                            phys_grid + phi_05_to_0, shape_t_ag, spacing_t_ag, origin_t_ag, direction_t_ag
                        )
                        I_mid = grid_sample_nd(curr_fixed, phi_fixed_norm, mode='bilinear', padding_mode='zeros')
                        
                        # Warp moving to midpoint (with affine)
                        phi_moving_affine = (phys_grid + phi_05_to_1) @ M_phys_zyx.t() + t_phys_zyx
                        phi_moving_norm = physical_to_normalized_torch_cached(
                            phi_moving_affine, shape_t_ag, spacing_t_ag, origin_t_ag, direction_t_ag
                        )
                        J_mid = grid_sample_nd(curr_moving, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                        
                        # Sample image spatial gradients at warped positions
                        grad_I_mid = grid_sample_nd(
                            grad_I_curr.movedim(-1, 1), phi_fixed_norm,
                            mode='bilinear', padding_mode='zeros'
                        ).movedim(1, -1).contiguous()
                        direction_t_mat = torch.tensor(
                            np.asarray(self.direction)[::-1, ::-1].copy(),
                            device=device, dtype=dtype
                        )
                        grad_I_mid = torch.matmul(grad_I_mid, direction_t_mat.t())
                        
                        grad_J_mid = grid_sample_nd(
                            grad_J_curr.movedim(-1, 1), phi_moving_norm,
                            mode='bilinear', padding_mode='zeros'
                        ).movedim(1, -1).contiguous()
                        grad_J_mid = torch.matmul(grad_J_mid, direction_t_mat.t())
                        grad_J_mid = torch.matmul(grad_J_mid, M_phys)
                    
                    # Step 2: Compute loss with grad tracking on detached midpoint images
                    I_mid_det = I_mid.detach().requires_grad_(True)
                    J_mid_det = J_mid.detach().requires_grad_(True)
                    
                    sim_loss = lncc_loss_nd(I_mid_det, J_mid_det, window_size=lncc_ws)
                    kinetic = torch.mean(self.velocity ** 2)
                    total_loss = sim_loss + reg_weight * kinetic
                    total_loss.backward()
                    
                    # Step 3: Compute analytical velocity gradient via chain rule
                    with torch.no_grad():
                        g_im = I_mid_det.grad if I_mid_det.grad is not None else torch.zeros_like(I_mid_det)
                        g_jm = J_mid_det.grad if J_mid_det.grad is not None else torch.zeros_like(J_mid_det)
                        
                        # Spatial chain rule: dL/dphi = dL/dI * dI/dphi
                        grad_wrt_phi_fixed = (g_im.movedim(1, -1) * grad_I_mid).contiguous()
                        grad_wrt_phi_moving = (g_jm.movedim(1, -1) * grad_J_mid).contiguous()
                        
                        # Combined gradient for velocity (both directions contribute)
                        combined_grad = grad_wrt_phi_fixed + grad_wrt_phi_moving
                        
                        # Assign gradient to velocity parameter (broadcast across time steps)
                        # The velocity gradient comes from how velocity changes the displacement
                        # At the midpoint, velocity directly scales displacement, so grad is proportional
                        if self.velocity.grad is None:
                            self.velocity.grad = torch.zeros_like(self.velocity)
                        
                        # Resize combined gradient to velocity grid shape if different
                        vel_spatial = tuple(self.velocity.shape[2:-1])
                        grad_spatial = tuple(combined_grad.shape[1:-1])
                        if vel_spatial != grad_spatial:
                            if self.dim == 3:
                                cg_cf = combined_grad.squeeze(0).permute(3, 0, 1, 2).unsqueeze(0)
                                cg_cf = F.interpolate(cg_cf, size=vel_spatial, mode='trilinear', align_corners=True)
                                combined_grad = cg_cf.squeeze(0).permute(1, 2, 3, 0).unsqueeze(0)
                            else:
                                cg_cf = combined_grad.squeeze(0).permute(2, 0, 1).unsqueeze(0)
                                cg_cf = F.interpolate(cg_cf, size=vel_spatial, mode='bilinear', align_corners=True)
                                combined_grad = cg_cf.squeeze(0).permute(1, 2, 0).unsqueeze(0)
                        
                        # Distribute gradient across all time steps
                        for t in range(self.n_time_steps):
                            self.velocity.grad[t, 0] = combined_grad[0]
                else:
                    # === Standard autograd mode ===
                    sim_loss = self.forward(curr_fixed, curr_moving, multipoint_loss=multipoint_loss, lncc_window_size=lncc_ws)
                    kinetic = torch.mean(self.velocity ** 2)
                    total_loss = sim_loss + reg_weight * kinetic
                    total_loss.backward()
                
                # Fluid regularization (smoothing velocity gradients)
                # Batched across all T time steps to minimize conv3d kernel launches
                # Smoothing is the dominant bottleneck (~91% of per-epoch time).
                # smooth_every_n > 1 reduces this cost at the expense of gradient noise.
                with torch.no_grad():
                    should_smooth = (sigma_val > 0 and self.velocity.grad is not None
                                     and (smooth_every_n <= 1 or epoch % smooth_every_n == 0))
                    if should_smooth:
                        T = self.n_time_steps
                        # Reshape (T, 1, *spatial, dim) -> (T, dim, *spatial) for batched filtering
                        grad_shape = self.velocity.grad.shape
                        if self.dim == 3:
                            # (T, 1, D, H, W, 3) -> squeeze batch -> (T, D, H, W, 3)
                            grad_batch = self.velocity.grad.squeeze(1)
                        else:
                            grad_batch = self.velocity.grad.squeeze(1)
                        # separable_gaussian_filter expects (B, *spatial, dim) channel-last
                        spatial_shape = list(grad_batch.shape[1:-1])
                        min_spatial = min(spatial_shape)
                        
                        regularizer_mode = kwargs.get('regularizer', 'gaussian')
                        if regularizer_mode == 'sobolev':
                            alpha_sob = float(kwargs.get('sobolev_alpha', kwargs.get('alpha', sigma_val / 2.0)))
                            grad_smoothed = self._apply_sobolev_green_operator(grad_batch, fluid_sigma=sigma_val, alpha=alpha_sob)
                        elif fast_smooth and min_spatial >= 32:
                            # Move to channels-first for F.interpolate
                            interp_3d = 'trilinear' if self.dim == 3 else 'bilinear'
                            g_cf = torch.movedim(grad_batch, -1, 1)  # (T, dim, *spatial)
                            down_shape = [max(8, s // 2) for s in spatial_shape]
                            g_down = F.interpolate(g_cf, size=down_shape, mode=interp_3d, align_corners=True)
                            g_down_cl = torch.movedim(g_down, 1, -1)  # (T, *down, dim)
                            g_smooth = separable_gaussian_filter(
                                g_down_cl, sigma=sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                            )
                            g_smooth_cf = torch.movedim(g_smooth, -1, 1)
                            g_up = F.interpolate(g_smooth_cf, size=spatial_shape, mode=interp_3d, align_corners=True)
                            grad_smoothed = torch.movedim(g_up, 1, -1).contiguous()
                        else:
                            grad_smoothed = separable_gaussian_filter(
                                grad_batch, sigma=sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                            )
                        # Apply smooth Dirichlet Cosine boundary taper mask to velocity gradients
                        bmask = self._create_boundary_mask(spatial_shape, device, dtype, border_width=4)
                        grad_smoothed_tapered = grad_smoothed * bmask
                        self.velocity.grad.copy_(grad_smoothed_tapered.unsqueeze(1))
                            
                if opt_type == 'cfl':
                    with torch.no_grad():
                        if self.velocity.grad is not None:
                            grad = self.velocity.grad
                            # ITK-style CFL: compute max norm in VOXEL space (divide by spacing)
                            # This matches ITK's ScaleUpdateField() exactly:
                            #   localNorm += sqr(vector[d] / spacing[d])
                            #   scale = learningRate / maxNorm
                            sp_t = torch.tensor(curr_spacing, device=device, dtype=dtype)
                            grad_voxel = grad / sp_t  # convert to voxel units
                            max_g_voxel = torch.sqrt(torch.sum(grad_voxel**2, dim=-1)).max()
                            if max_g_voxel > 1e-8:
                                cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                                effective_cfl = min(cfl_step_val, 0.25)
                                # Compute CFL update: scaledUpdate = (learningRate / maxNorm) * gradient
                                update = (effective_cfl / max_g_voxel) * grad
                                
                                # Apply momentum for faster convergence
                                if cfl_momentum > 0 and momentum_buffer is not None:
                                    momentum_buffer.mul_(cfl_momentum).add_(update)
                                    self.velocity.data.sub_(momentum_buffer)
                                else:
                                    self.velocity.data.sub_(update)
                else:
                    optimizer.step()

                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                with torch.no_grad():
                    if elastic_sigma_val > 0:
                        T = self.n_time_steps
                        vel_batch = self.velocity.squeeze(1)
                        vel_smoothed = separable_gaussian_filter(
                            vel_batch, sigma=elastic_sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                        )
                        self.velocity.copy_(vel_smoothed.unsqueeze(1))
                    
                    vel_clamp_val = float(kwargs.get('velocity_clamp', kwargs.get('clamp', 50.0)))
                    self.velocity.clamp_(min=-vel_clamp_val, max=vel_clamp_val)
                    cfl_max_val = kwargs.get('cfl_max', None)
                    if cfl_max_val is not None and float(cfl_max_val) > 0:
                        sp_t = torch.tensor(self.spacing, device=device, dtype=dtype)
                        vel_vox = self.velocity / sp_t
                        max_vox = torch.norm(vel_vox, dim=-1).max()
                        if max_vox > float(cfl_max_val):
                            self.velocity.mul_(float(cfl_max_val) / (max_vox + 1e-8))
                    if kwargs.get('antisymmetric', kwargs.get('antisymmetry', self.antisymmetric)):
                        self.project_antisymmetric()

                # Convergence checking (every 5 epochs to reduce GPU-CPU sync barriers)
                if epoch % 5 == 0 or epoch == epochs - 1:
                    loss_val = sim_loss.item()
                    recent_losses.append(loss_val)
                    if len(recent_losses) >= convergence_window:
                        y = np.array(recent_losses[-convergence_window:])
                        x = np.arange(convergence_window)
                        x_mean = x.mean()
                        y_mean = y.mean()
                        denom = np.sum((x - x_mean) ** 2)
                        if denom > 1e-8 and convergence_threshold is not None and float(convergence_threshold) > 0:
                            slope = np.sum((x - x_mean) * (y - y_mean)) / denom
                            if slope >= -float(convergence_threshold) and loss_val < 0.0 and epoch >= 10:
                                if verbose:
                                    print(f"  Level {level} converged at epoch {epoch+1} (slope = {slope:.2e} >= -{float(convergence_threshold):.2e}). Early stopping level.")
                                break

            # GPU memory management and garbage collection at level transitions
            if device.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Ensure velocity is at full image resolution after fit completes
        final_vel_shape = tuple(self.velocity.shape[2:-1])
        if final_vel_shape != tuple(self.image_shape):
            self._resize_velocity(self.image_shape, device, dtype)
            if verbose:
                print(f"  Final velocity upsample: {list(final_vel_shape)} → {list(self.image_shape)}")

    @torch.no_grad()
    def get_forward_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=0 to t=1 in physical space.
        """
        return self.integrate(0.0, 1.0, image_shape=image_shape)
        
    @torch.no_grad()
    def get_inverse_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=1 to t=0 in physical space.
        """
        return self.integrate(1.0, 0.0, image_shape=image_shape)
def tvf_registration(
    fixed,
    moving,
    type_of_transform='TVF',
    initial_transform=None,
    syn_metric='lncc',
    syn_sampling=2,
    aff_metric=None,
    aff_sampling=None,
    reg_iterations=None,
    affine_iterations=None,
    grad_step=0.15,
    flow_sigma=1.0,
    total_sigma=0.0,
    n_time_steps=4,
    n_steps=None,
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
    High-level TVF (Time-Varying Velocity Field) registration function matching
    the ``syntx.syn()`` / ``syntx.registration()`` interface.

    Usage is identical to ``syntx.syn()``::

        import syntx
        reg = syntx.tvf(fixed=fi, moving=mi)
        warped = reg['warpedmovout']
        transforms = reg['fwdtransforms']

    Parameters
    ----------
    fixed : ANTsImage
        Fixed target image.
    moving : ANTsImage
        Moving source image.
    type_of_transform : str, optional
        Transform descriptor (default 'TVF'). Included for API parity.
    initial_transform : str or list of str or ANTsTransform, optional
        Initial transform(s) to apply to moving image before registration. Default None.
    syn_metric : str, optional
        Similarity metric. Default 'lncc'.
    syn_sampling : int, optional
        LNCC radius (window_size = 2 * syn_sampling + 1). Default 2.
    aff_metric : str or None, optional
        Present for API consistency with syntx.syn(). Not natively used by TVF.
    aff_sampling : int or None, optional
        Present for API consistency with syntx.syn(). Not natively used by TVF.
    reg_iterations : list of int or None, optional
        Number of deformable iterations per level. Default [150, 150, 0].
    affine_iterations : list of int or int or None, optional
        Number of affine iterations. Default 100.
    grad_step : float, optional
        CFL voxel bound step size. Default 0.20.
    flow_sigma : float, optional
        Fluid regularization sigma in ITK variance convention (σ² = flow_sigma).
        Default 3.0 (actual σ = √3 ≈ 1.73).
    total_sigma : float, optional
        Elastic regularization sigma in ITK variance convention. Default 0.0.
    n_time_steps : int, optional
        Number of TVF time keyframes. Default 4.
    n_steps : int or None, optional
        Present for API consistency with syntx.syngs(). Not used by TVF (see n_time_steps).
    verbose : bool, optional
        If True, print optimization progress. Default False.
    backend : str, optional
        Computation backend ('pytorch' or 'jax'). Default 'pytorch'.
    levels : list of int or None, optional
        Multi-resolution pyramid levels. Default [4, 2, 1].
    cfl_momentum : float, optional
        SGD-style momentum for CFL updates. Default 0.9. Set 0.0 to disable.
    multipoint_loss : list of float or None, optional
        ODE evaluation timepoints for loss. Default [0.0, 1.0] (direct-space).
        Use [0.5] for geodesic midpoint, [0.0, 0.5, 1.0] for triplet.
    fast_smooth : bool, optional
        If True, smooth gradients at half resolution (9x faster). Default True.
    sampling_percentage : float or None, optional
        Present for API consistency with syntx.syn(). Not natively used by TVF.
    vgg_layers, vgg_mode, vgg_patch_size, vgg_num_patches, vgg_lncc_window_size : optional
        Present for API consistency with syntx.syn().
    optimizer, optimizer_lr, project_inverse, projection_frequency, interpolator, inverse_method, inverse_steps : optional
        Present for API consistency with syntx.syn().
    **kwargs
        Additional parameters passed to TVFModel.fit().

    Returns
    -------
    dict
        Same format as ``syntx.syn()`` / ``syntx.registration()``::

            {
                'warpedmovout': ANTsImage,      # moving warped to fixed space
                'warpedfixout': ANTsImage,      # fixed warped to moving space
                'fwdtransforms': [str],         # [warp_path, affine_path]
                'invtransforms': [str],         # [affine_path, inv_warp_path]
                'whichtoinvert_inv': [bool],    # [True, False]
                'model': TVFModel,              # fitted model
            }
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

    # --- Convert ITK variance convention to actual sigma (same as registration()) ---
    fluid_sigma_actual = math.sqrt(flow_sigma) if flow_sigma > 0 else 0.0
    elastic_sigma_actual = math.sqrt(total_sigma) if total_sigma > 0 else 0.0

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
        model = TVFModel(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_time_steps=n_time_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
            integration_steps_per_interval=kwargs.pop('integration_steps_per_interval', 1),
            antisymmetric=kwargs.pop('antisymmetric', False),
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
                print(f"[TVF] Initialized affine from initial_transform (T_init absorbed)")

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
        from .tvf_jax import TVFModelJAX
        from .syn_jax import get_affine_matrix_jax
        import jax.numpy as jnp

        device_str = 'cpu'
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *fixed.shape).transpose(perm)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *moving.shape).transpose(perm)

        model = TVFModelJAX(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_time_steps=n_time_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
            integration_steps_per_interval=kwargs.pop('integration_steps_per_interval', 1),
            antisymmetric=kwargs.pop('antisymmetric', False),
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
            # Inject into affine_params for get_affine_matrix_jax composition
            model.affine_params['T_init'] = T_init_jax

            # Affine absorbed into model parameters; do not append to final transform list
            init_tx_list = []
            if verbose:
                print(f"[TVF-JAX] Initialized affine from initial_transform (T_init absorbed)")

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

        fwd_disp = np.array(model.integrate(0.0, 1.0, image_shape=grid_shape_zyx))
        inv_disp = np.array(model.integrate(1.0, 0.0, image_shape=grid_shape_zyx))
        fwd_np = fwd_disp.squeeze(0)
        inv_np = inv_disp.squeeze(0)

        # Export affine including T_init composition (get_affine_matrix_jax composes T_init if present)
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

    fwd_file = tempfile.NamedTemporaryFile(suffix='_tvf_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_tvf_inv_Warp.nii.gz', delete=False).name
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
            algorithm="syntx.tvf",
            backend=backend,
            device=device_str,
            fit_time=fit_time,
            reg_iterations=reg_iterations,
            affine_iterations=affine_iterations if isinstance(affine_iterations, list) else [affine_iterations],
            solver="TVF-Euler",
            fluid_sigma=flow_sigma,
            elastic_sigma=total_sigma,
            learning_rate=grad_step,
            optimizer_type="CFL",
            similarity_metric=syn_metric,
            fixed_shape=tuple(fixed.shape),
            fixed_spacing=tuple(fixed.spacing),
            fixed_orientation=str(fixed.orientation) if hasattr(fixed, 'orientation') else None,
            moving_shape=tuple(moving.shape),
            moving_spacing=tuple(moving.spacing),
            moving_orientation=str(moving.orientation) if hasattr(moving, 'orientation') else None,
        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass

    return ret_dict
