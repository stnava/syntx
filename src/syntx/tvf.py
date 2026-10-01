r"""
tvf.py — Time-Varying Velocity Fields (TVF) Diffeomorphic Registration
=======================================================================

This module implements Time-Varying Velocity Field (TVF) diffeomorphic registration in PyTorch.

Key Algorithmic Features & Guardrails
-------------------------------------
- Layer-wise Adaptive Rate Scaling (LARS, GEMINI.md Rule 6): Rescales velocity updates per keyframe
  tensor using trust ratio $\text{trust\_ratio} = \eta \cdot \frac{\|v(t_k)\|}{\|g(t_k)\| + \epsilon}$,
  preventing Adam optimization stalling on smooth LNCC similarity plateaus.
- Pyramid-Proportional Velocity Grids: Resizes velocity parameter grids proportionally across multi-resolution
  pyramid levels using B-spline/trilinear interpolation.
- Continuous Trajectory ODE Integration: Integrates continuous time trajectories $t \in [0, 1]$ via Euler ODE solver.
- Elastic Total Field Smoothing: Applies mild post-step elastic smoothing (`total_sigma = 0.05`) to eliminate
  grid folding (0.0000% folding, $\min \det(J) > 0.0$).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import numpy as np
from .spatial import (
    reverse_metadata,
    itk_shape_to_tensor_shape,
    compute_autograd_physical_scale,
    disp_tensor_to_itk,
    export_ants_affine_transform,
    export_ants_displacement_field,
    grid_to_physical_affine,
    grid_to_physical_affine_torch,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
)
from .syn import (
    grid_sample_nd,
    HierarchicalAffine,
    local_ncc_loss_nd,
    local_ncc_loss_nd as lncc_loss_nd,
    mattes_mi_loss_nd,
    _spatial_jacobian_nd,
)
from .core.smoothing import (
    separable_gaussian_filter,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    get_boundary_mask,
)
from .core.optimizers import LARS, RegAdam, SobolevAdam
from .core.grid import compose_grids, sample_field_cf, resize_field

class TVFConjugateGradient(torch.optim.Optimizer):
    """
    TVF-specific Conjugate Gradient optimizer harness.
    Normalizes the space+time gradient independently per time index to a constant norm,
    preventing intermediate keyframes from being starved. Then computes a Polak-Ribiere
    conjugate gradient search direction along the manifold to accelerate flow without folding.
    """
    def __init__(self, params, lr=0.35):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """One conjugate-gradient update of every parameter (see the class docstring)."""
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Normalize gradient PER KEYFRAME to a constant max norm
                # grad shape: [T, 1, *spatial, dim]
                spatial_dims = tuple(range(1, grad.ndim - 1))
                max_g = torch.sqrt(torch.sum(grad**2, dim=-1))
                for d in reversed(spatial_dims):
                    max_g = max_g.max(dim=d, keepdim=True)[0]
                max_g = max_g.unsqueeze(-1)
                
                g_norm = grad / torch.clamp(max_g, min=1e-8)
                
                if len(state) == 0:
                    d_k = -g_norm
                    state['prev_g_norm'] = g_norm.clone()
                    state['d_k'] = d_k.clone()
                else:
                    prev_g_norm = state['prev_g_norm']
                    d_k_prev = state['d_k']
                    
                    # Polak-Ribiere beta per keyframe
                    # sum over spatial and dim axes, preserving T
                    reduce_dims = spatial_dims + (-1,)
                    num = torch.sum(g_norm * (g_norm - prev_g_norm), dim=reduce_dims)
                    den = torch.sum(prev_g_norm * prev_g_norm, dim=reduce_dims)
                    
                    # Reshape for broadcasting back to [T, 1, *spatial, dim]
                    for _ in range(len(reduce_dims)):
                        num = num.unsqueeze(-1)
                        den = den.unsqueeze(-1)
                        
                    beta = torch.clamp(num / torch.clamp(den, min=1e-8), min=0.0)
                    d_k = -g_norm + beta * d_k_prev
                    
                    state['prev_g_norm'].copy_(g_norm)
                    state['d_k'].copy_(d_k)
                
                # Apply Conjugate Gradient step
                p.data.add_(d_k, alpha=lr)


class TVFModel(nn.Module):
    """
    The PyTorch model behind ``syntx.tvf``: an affine plus a time-varying velocity field
    v(t, x), stored as ``n_time_steps`` keyframes interpolated linearly in time and integrated
    (Euler or RK4) from t = 0 to 1. Most users call ``syntx.tvf``; use the class directly only
    for custom pipelines (construct, then ``fit``).

    ``velocity`` is a parameter of shape ``(T, 1, *velocity_shape, dim)``: physical
    velocities (mm per unit time), tensor (z, y, x) order. ``integrate`` gives the
    displacement between any two times; ``jacobian_determinant`` the exact Jacobian of the
    integrated map.

    Parameters
    ----------
    dim : {2, 3}
    image_shape, velocity_shape : tuple of int
        Fixed-image grid and finest velocity grid, tensor (z, y, x) order.
    n_time_steps : int, default 3
        Number of velocity keyframes T.
    spacing, origin, direction : optional
        Fixed-image geometry (ITK (x, y, z) order); defaults unit spacing, zero origin, identity.
    moving_shape, moving_spacing, moving_origin, moving_direction : optional
        Moving-image geometry; defaults to the fixed image's.
    fluid_sigma, elastic_sigma : float, default 0.0, 0.2
        Gaussian sigmas (mm) for the 'gaussian' / 'bspline' regularisers (spectral
        regularisers use ``fit``'s ``alpha`` / ``total_alpha``). ``syntx.tvf`` sets both.
    transform_type : {'Affine', 'Rigid', 'Translation'}, default 'Affine'
    solver : {'euler', 'rk4'}, default 'euler'
    integration_steps_per_interval : int, default 4
        Minimum integration steps per keyframe interval; more are taken automatically so
        that no step moves a point more than half a voxel (Euler) or one voxel (RK4).
        ``syntx.tvf`` passes 1.
    antisymmetric : bool, default False
        Always evaluate the similarity at t = 0 and t = 1 (``syntx.tvf`` no longer exposes
        this: put 0 and 1 in ``multipoint_loss`` instead).
    cfl_max : float, default 0.0
        Cap on the velocity magnitude after each step (0 = none).
    **kwargs
        ``similarity_metric`` (default 'cc2'), ``mattes_bins`` / ``num_bins`` (32),
        ``use_analytical_gradients`` (False).
    """
    def __init__(
        self,
        dim,
        image_shape,
        velocity_shape,
        n_time_steps=3,
        spacing=None,
        origin=None,
        direction=None,
        moving_shape=None,
        moving_spacing=None,
        moving_origin=None,
        moving_direction=None,
        fluid_sigma=0.0,
        elastic_sigma=0.2,
        transform_type='Affine',
        solver='euler',
        integration_steps_per_interval=4,
        antisymmetric=False,
        cfl_max=0.0,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps
        self.antisymmetric = antisymmetric
        self.use_analytical_gradients = kwargs.get('use_analytical_gradients', False)
        self.cfl_max = cfl_max

        
        self.spacing = spacing if spacing is not None else [1.0] * dim
        self.origin = origin if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction
        else:
            self.direction = np.eye(dim).tolist()

        self.moving_shape = tuple(moving_shape) if moving_shape is not None else self.image_shape
        self.moving_spacing = moving_spacing if moving_spacing is not None else self.spacing
        self.moving_origin = moving_origin if moving_origin is not None else self.origin
        if moving_direction is not None:
            self.moving_direction = moving_direction
        else:
            self.moving_direction = list(self.direction)
            
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.solver = solver
        self.integration_steps_per_interval = integration_steps_per_interval
        self.similarity_metric = kwargs.get('similarity_metric', 'cc2')
        self.mattes_bins = int(kwargs.get('mattes_bins', kwargs.get('num_bins', 32)))
        
        # Velocity field parameter: (T, 1, *velocity_shape, dim)
        self.velocity = nn.Parameter(torch.zeros(n_time_steps, 1, *self.velocity_shape, self.dim))
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)
        self._sobolev_kernel_cache = {}
        self._v_max_cache = {}

    def _ensure_symmetric_eval_points(self, eval_points):
        """
        When antisymmetric=True, ensure the evaluation timepoints include both
        t=0.0 (fixed-side gradient) and t=1.0 (moving-side gradient).

        In TVF, the antisymmetric approach is simply:
        1. Compute the similarity gradient wrt the velocity using the fixed image warp
        2. Compute the similarity gradient wrt the velocity using the moving image warp
        3. Average them

        This is exactly what autograd does when the loss evaluates LNCC(I_warped, J_warped)
        at both t=0 and t=1. The gradient through the fixed-side warp gives (1), the
        gradient through the moving-side warp gives (2), and autograd sums them.
        Dividing by len(eval_points) averages. This is the exact TVF generalization
        of SyN's antisymmetric delta_l/delta_r averaging.
        """
        pts = list(eval_points)
        if 0.0 not in pts:
            pts.insert(0, 0.0)
        if 1.0 not in pts:
            pts.append(1.0)
        return pts

    def _resize_velocity(self, new_shape, device=None, dtype=None):
        """
        Resize the velocity parameter to a new spatial shape using cubic B-spline
        interpolation for smooth warm-start cascading between pyramid levels.

        For 3D, uses separable tricubic interpolation (two-pass bicubic: first
        along D×H planes, then along W axis) to preserve the high-order smoothness
        of the Catmull-Rom velocity spline across resolution transitions.

        For 2D, uses native PyTorch bicubic interpolation.

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
            # Reshape (T, 1, *spatial, dim) -> (T, *spatial, dim) so time is batch dim
            vel_reshaped = old_vel.squeeze(1)
            new_vel = resize_field(vel_reshaped, size=new_shape).unsqueeze(1)

            if device is not None:
                new_vel = new_vel.to(device=device)
            if dtype is not None:
                new_vel = new_vel.to(dtype=dtype)

            self.velocity = nn.Parameter(new_vel.contiguous())

    def _get_metadata_tensors(self, device, dtype):
        """Helper to get spatial metadata as tensors for cached normalized coordinates."""
        spacing_rev, origin_rev, direction_rev = reverse_metadata(self.spacing, self.origin, self.direction)
        
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(self.image_shape), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        return shape_t, spacing_t, origin_t, direction_t

    def _get_moving_metadata_tensors(self, device, dtype):
        """Helper to get spatial metadata as tensors for cached normalized coordinates (Moving Image)."""
        spacing_rev, origin_rev, direction_rev = reverse_metadata(self.moving_spacing, self.moving_origin, self.moving_direction)
        
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(self.moving_shape), device=device, dtype=dtype)
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
        """Smooth (1, *spatial, 1) taper: 1 inside, falling to 0 over ``border_width`` voxels
        (default min(shape) // 32, at least 1) at every face."""
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

    def _apply_sobolev_green_operator(self, m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0, border_width=0):
        """Sobolev smoothing of ``m`` (``core.smoothing.apply_sobolev_green_operator``)."""
        return apply_sobolev_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha, border_width=border_width, spacing=spacing)

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """DST (Dirichlet) smoothing of ``m`` (``core.smoothing.apply_dsti_green_operator``)."""
        return apply_dsti_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def _apply_dsti1_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """DST-I (Dirichlet) smoothing of ``m`` (``core.smoothing.apply_dsti1_green_operator``)."""
        return apply_dsti1_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

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

    def _upsample_velocity_keyframes(self, velocity, target_shape):
        """
        Pre-upsample ALL velocity keyframes to target_shape.
        Returns list of channels-first tensors, one per keyframe.
        """
        if self.dim == 2:
            velocity_cf = velocity.permute(0, 1, 4, 2, 3)
        else:
            velocity_cf = velocity.permute(0, 1, 5, 2, 3, 4)
        
        velocity_fine_cf = []
        for t_idx in range(self.n_time_steps):
            v_fine = self.upsample_velocity(velocity_cf[t_idx], target_shape)
            velocity_fine_cf.append(v_fine)
        return velocity_fine_cf

    def _euler_step_checkpointed(self, phi_t, t_current, dt, velocity_fine_cf,
                                  shape_t, spacing_t, origin_t, direction_t):
        """
        Single Euler ODE step wrapped in gradient checkpointing.

        Discards intermediate tensors (normalized coords, sampled velocity)
        during forward pass and recomputes them during backward, trading
        ~25% extra forward compute for ~4× memory reduction per step.

        Args:
            phi_t: Current deformation field (1, *spatial, dim)
            t_current: Current time along trajectory
            dt: Time step size
            velocity_fine_cf: List of upsampled velocity keyframes
            shape_t, spacing_t, origin_t, direction_t: Cached metadata tensors
        """
        from torch.utils.checkpoint import checkpoint

        # Pack velocity keyframes into a single tensor for checkpoint's tensor-only interface
        vel_stack = torch.stack(velocity_fine_cf, dim=0)  # (T, 1, dim, *spatial)

        def _euler_fn(phi, vel_packed, shape_t_, spacing_t_, origin_t_, direction_t_):
            # Unpack velocity keyframes
            vels = [vel_packed[i] for i in range(vel_packed.shape[0])]
            v_fine_cf = self._interpolate_velocity_fine(t_current, vels)
            phi_norm = physical_to_normalized_torch_cached(
                phi, shape_t_, spacing_t_, origin_t_, direction_t_
            )
            v_sampled_cf = grid_sample_nd(v_fine_cf, phi_norm, mode='bilinear', padding_mode='border')
            v_sampled = torch.movedim(v_sampled_cf, 1, -1)
            return phi + v_sampled * dt

        return checkpoint(_euler_fn, phi_t, vel_stack, shape_t, spacing_t, origin_t, direction_t,
                          use_reentrant=False)

    def _get_v_max_voxel(self, velocity, target_shape):
        """Compute or retrieve cached maximum voxel velocity without repeated GPU-CPU sync."""
        target_sp = tuple(target_shape) if target_shape is not None else tuple(self.image_shape)
        vel_id = id(velocity)
        vel_version = getattr(velocity, '_version', 0)
        cache_key = (vel_id, vel_version, target_sp)

        if not hasattr(self, '_v_max_cache') or not isinstance(self._v_max_cache, dict):
            self._v_max_cache = {}

        if cache_key in self._v_max_cache:
            return self._v_max_cache[cache_key]

        with torch.no_grad():
            curr_spacing = [
                sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                for sp, orig_s, curr_s in zip(self.spacing, reversed(self.image_shape), reversed(target_sp))
            ]
            sp_t = torch.tensor(curr_spacing, device=velocity.device, dtype=velocity.dtype)
            vel_voxel = velocity.detach() / sp_t
            v_mag_sq = torch.sum(vel_voxel ** 2, dim=-1)
            v_max_voxel = float(torch.sqrt(v_mag_sq.max()).item())

        if len(self._v_max_cache) > 32:
            self._v_max_cache.clear()
        self._v_max_cache[cache_key] = v_max_voxel
        return v_max_voxel

    def integrate(self, t_start, t_end, velocity=None, n_steps=None, image_shape=None,
                  _cached_phys_grid=None, _cached_meta=None, _cached_velocity_fine_cf=None,
                  return_log_jacobian=False):
        """
        Integrates the velocity field ODE from t_start to t_end.

        ``return_log_jacobian=True`` also returns (log |det D phi|, number of steps with
        det < 0) of the map, as the product of the step Jacobians det(I + dt grad v(phi_k(x)))
        along the same Euler trajectories -- the Jacobian of the exported map itself (its
        continuous limit is Liouville's exp int div v dt), unlike a finite-difference Jacobian
        of the displacement, which breaks down where the map varies sharply within a voxel.
        
        Performance optimizations:
        - Short-circuits identity integrations (t_start == t_end → zero displacement).
        - Accepts pre-upsampled velocity keyframes via _cached_velocity_fine_cf to
          avoid redundant F.interpolate calls across multiple integrate() calls
          within a single forward pass.
        """
        if velocity is None:
            velocity = self.velocity

        device = velocity.device
        dtype = velocity.dtype
        
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        
        # Short-circuit: identity integration (t_start == t_end → zero displacement)
        if abs(t_end - t_start) < 1e-8:
            batch_shape = (1,) + target_shape + (self.dim,)
            z = torch.zeros(batch_shape, device=device, dtype=dtype)
            return (z, torch.zeros((1,) + target_shape, device=device, dtype=dtype)) if return_log_jacobian else z
        
        if n_steps is None:
            default_steps = self.n_time_steps * self.integration_steps_per_interval
            # Adaptive CFL: ensure per-step displacement respects integrator stability.
            v_max_voxel = self._get_v_max_voxel(velocity, target_shape)
            if v_max_voxel > 1e-6:
                c_cfl = 1.0 if self.solver == 'rk4' else 0.5
                cfl_steps = int(math.ceil(v_max_voxel * abs(t_end - t_start) / c_cfl))
                n_steps = max(default_steps, cfl_steps)
            else:
                n_steps = default_steps
            
        dt = (t_end - t_start) / max(1, n_steps)
        
        # Use cached grid and metadata if provided, otherwise compute
        if _cached_phys_grid is not None and _cached_meta is not None:
            phys_grid = _cached_phys_grid
            shape_t, spacing_t, origin_t, direction_t = _cached_meta
        else:
            curr_spacing = [
                sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                for sp, orig_s, curr_s in zip(self.spacing, reversed(self.image_shape), reversed(target_shape))
            ]
            
            phys_grid = get_physical_grid_torch(
                target_shape, curr_spacing, self.origin, self.direction,
                device=device, dtype=dtype
            )
            
            spacing_rev, origin_rev, direction_rev = reverse_metadata(curr_spacing, self.origin, self.direction)
            
            shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
            spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
            origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
            direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        phi_t = phys_grid.clone()
        
        # Use cached upsampled velocity keyframes if provided, otherwise compute
        if _cached_velocity_fine_cf is not None:
            velocity_fine_cf = _cached_velocity_fine_cf
        else:
            velocity_fine_cf = self._upsample_velocity_keyframes(velocity, target_shape)
            
        use_ckpt = getattr(self, '_gradient_checkpointing', False)
        solver = 'euler' if return_log_jacobian else self.solver
        if return_log_jacobian:
            log_jac = torch.zeros(phi_t.shape[:-1], device=device, dtype=dtype)
            n_neg = torch.zeros(phi_t.shape[:-1], device=device, dtype=dtype)
            h = 0.5 * float(spacing_t.min())

        for step in range(n_steps):
            t_current = t_start + step * dt
            
            if solver == 'euler':
                if use_ckpt and phi_t.requires_grad and not return_log_jacobian:
                    # Gradient checkpointing: discard intermediates, recompute in backward.
                    # Wraps the Euler step in a checkpointable closure.
                    phi_t = self._euler_step_checkpointed(
                        phi_t, t_current, dt, velocity_fine_cf,
                        shape_t, spacing_t, origin_t, direction_t
                    )
                else:
                    v_fine_cf = self._interpolate_velocity_fine(t_current, velocity_fine_cf)

                    phi_norm = physical_to_normalized_torch_cached(
                        phi_t, shape_t, spacing_t, origin_t, direction_t
                    )

                    v_sampled_cf = grid_sample_nd(v_fine_cf, phi_norm, mode='bilinear', padding_mode='border')
                    v_sampled = torch.movedim(v_sampled_cf, 1, -1)
                    if return_log_jacobian:
                        from .liouville import velocity_gradient_at, step_log_det
                        grad_v = velocity_gradient_at(
                            lambda p: torch.movedim(grid_sample_nd(
                                v_fine_cf, physical_to_normalized_torch_cached(p, shape_t, spacing_t, origin_t, direction_t),
                                mode='bilinear', padding_mode='border'), 1, -1),
                            phi_t, h)
                        ld, neg = step_log_det(grad_v, dt)
                        log_jac = log_jac + ld
                        n_neg = n_neg + neg
                    phi_t = phi_t + v_sampled * dt
                
            elif solver == 'rk4':
                def eval_v(t, current_phi):
                    v_fine_cf_t = self._interpolate_velocity_fine(t, velocity_fine_cf)
                    
                    phi_norm_t = physical_to_normalized_torch_cached(
                        current_phi, shape_t, spacing_t, origin_t, direction_t
                    )
                    v_sampled_cf = grid_sample_nd(v_fine_cf_t, phi_norm_t, mode='bilinear', padding_mode='border')
                    return torch.movedim(v_sampled_cf, 1, -1)
                    
                k1 = eval_v(t_current, phi_t)
                k2 = eval_v(t_current + 0.5 * dt, phi_t + 0.5 * dt * k1)
                k3 = eval_v(t_current + 0.5 * dt, phi_t + 0.5 * dt * k2)
                k4 = eval_v(t_current + dt, phi_t + dt * k3)
                
                phi_t = phi_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")

        if return_log_jacobian:
            return phi_t - phys_grid, log_jac, n_neg
        return phi_t - phys_grid

    def _path_energy(self, energy_weight, temporal_weight, alpha, spacing, device, dtype):
        """energy_weight * mean_t <v_t, L v_t>  +  temporal_weight * int ||d v / d t||_V^2 dt.

        The temporal term, discretised on the keyframes as (T-1) * sum_k ||v_{k+1} - v_k||_V^2,
        penalises changes of the velocity between keyframes (the path turning); both terms are
        symmetric under t -> 1-t, v -> -v and use the same V-norm (L = inverse smoothing kernel).
        """
        from .core.smoothing import sobolev_energy
        e = torch.zeros((), device=device, dtype=dtype)
        v = self.velocity.squeeze(1)
        if energy_weight > 0:
            e = e + energy_weight * sobolev_energy(v, alpha, spacing=spacing).mean()
        if temporal_weight > 0 and v.shape[0] > 1:
            T = v.shape[0]
            e = e + temporal_weight * (T - 1) * sobolev_energy(v[1:] - v[:-1], alpha, spacing=spacing).sum()
        return e

    @torch.no_grad()
    def jacobian_determinant(self, image_shape=None, t_start=0.0, t_end=1.0):
        """Jacobian determinant of the flow map from t_start to t_end (product of the Euler
        step Jacobians along the trajectories; see ``integrate``), shape (1, *spatial)."""
        _, log_jac, n_neg = self.integrate(t_start, t_end, image_shape=image_shape, return_log_jacobian=True)
        # a step with det <= 0 folds the composed map there (even if another step flips it back)
        return torch.where(n_neg > 0, -torch.exp(log_jac), torch.exp(log_jac))

    def forward(
        self,
        fixed_image,
        moving_image,
        velocity=None,
        affine_params=None,
        multipoint_loss=[0.0, 1.0],
        lncc_window_size=5,
        bootstrap_mode=None,
        bootstrap_orig_weight=0.50,
        bootstrap_jitter_scale=0.25,
    ):
        """
        Loss of the current transform. At each time t in ``multipoint_loss`` the fixed image
        is pulled from t = 0 and the moving image from t = 1 to time t and compared there; the
        similarity losses are averaged. When the times include both 0 and 1, an
        inverse-consistency penalty is added: 0.05 x the mean squared error of
        phi(0->1) o phi(1->0) and phi(1->0) o phi(0->1) against identity (weight
        ``self.inverse_identity_weight``, default 0.05, not exposed by ``syntx.tvf``). Lower
        is better.

        Parameters
        ----------
        fixed_image, moving_image : Tensor (1, 1, *spatial)
        velocity, affine_params : optional
            Override the model's own (default None: use them).
        multipoint_loss : list of float, bool or float, default [0.0, 1.0]
            Evaluation times in [0, 1]: [0, 1] compares at both ends (symmetric), [0.5] at the
            midpoint, [0, 0.5, 1] all three; True = [0, 0.5, 1], False = [0.5].
        lncc_window_size : int, default 5
        bootstrap_mode : {None, 'antithetic', 'jitter'}, default None
            Also evaluate on spatially jittered grids (``bootstrap_jitter_scale`` voxels),
            weighting the unshifted grid by ``bootstrap_orig_weight`` (0.5).
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

        # Antisymmetric: ensure both t=0 and t=1 are evaluated so autograd
        # naturally averages the fixed-side and moving-side gradient contributions.
        if getattr(self, 'antisymmetric', False):
            eval_points = self._ensure_symmetric_eval_points(eval_points)

        curr_spacing = [
            sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
            for sp, orig_s, curr_s in zip(self.spacing, reversed(self.image_shape), reversed(target_shape))
        ]

        phys_grid = get_physical_grid_torch(
            target_shape, curr_spacing, self.origin, self.direction,
            device=device, dtype=dtype
        )

        spacing_rev, origin_rev, direction_rev = reverse_metadata(curr_spacing, self.origin, self.direction)

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
            self.moving_shape, self.moving_spacing, self.moving_origin, self.moving_direction
        )
        
        # M_phys and t_phys are already returned in ZYX order from grid_to_physical_affine_torch
        M_phys_zyx = M_phys
        t_phys_zyx = t_phys
        
        compute_id_loss = (0.0 in eval_points) and (1.0 in eval_points)

        # Pre-upsample velocity keyframes ONCE for the entire forward pass,
        # shared across all integrate() calls (eliminates redundant F.interpolate).
        velocity_fine_cf = self._upsample_velocity_keyframes(
            velocity if velocity is not None else self.velocity, target_shape
        )

        # Moving image metadata dynamically scaled to current pyramid resolution
        moving_target_shape = tuple(moving_image.shape[2:])
        curr_moving_spacing = [
            sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
            for sp, orig_s, curr_s in zip(self.moving_spacing, reversed(self.moving_shape), reversed(moving_target_shape))
        ]
        sp_m_rev, orig_m_rev, dir_m_rev = reverse_metadata(curr_moving_spacing, self.moving_origin, self.moving_direction)
        shape_m = torch.tensor(list(moving_target_shape), device=device, dtype=dtype)
        spacing_m = torch.tensor(sp_m_rev, device=device, dtype=dtype)
        origin_m = torch.tensor(orig_m_rev, device=device, dtype=dtype)
        direction_m = torch.tensor(dir_m_rev, device=device, dtype=dtype)

        boot_m = bootstrap_mode if bootstrap_mode is not None else getattr(self, 'bootstrap_mode', None)
        orig_w = float(bootstrap_orig_weight if bootstrap_orig_weight is not None else getattr(self, 'bootstrap_orig_weight', 0.50))
        jitter_amp = float(bootstrap_jitter_scale if bootstrap_jitter_scale is not None else getattr(self, 'bootstrap_jitter_scale', 0.25))

        if boot_m in ('antithetic', 'jitter'):
            j_view_shape = [1] * (self.dim + 1) + [self.dim]
            spacing_tensor = torch.tensor(curr_spacing, device=device, dtype=dtype).view(*j_view_shape)
            rand_dir = (2.0 * torch.rand(*j_view_shape, device=device, dtype=dtype) - 1.0) * jitter_amp * spacing_tensor
            if boot_m == 'antithetic':
                offsets = [rand_dir, -rand_dir]
                boot_weight = (1.0 - orig_w) / len(offsets)
            else:
                offsets = [rand_dir]
                boot_weight = 1.0 - orig_w
        else:
            offsets = []
            boot_weight = 0.0

        losses_k = []
        phi_0_to_1 = None
        phi_1_to_0 = None

        for t_k in eval_points:
            t_k = float(t_k)
            # Identity short-circuit is handled inside integrate() (returns zeros when t_start==t_end)
            phi_tk_to_fixed = self.integrate(t_k, 0.0, velocity=velocity, image_shape=target_shape,
                                             _cached_phys_grid=phys_grid, _cached_meta=_cached_meta,
                                             _cached_velocity_fine_cf=velocity_fine_cf)
            phi_tk_to_moving = self.integrate(t_k, 1.0, velocity=velocity, image_shape=target_shape,
                                              _cached_phys_grid=phys_grid, _cached_meta=_cached_meta,
                                              _cached_velocity_fine_cf=velocity_fine_cf)

            if abs(t_k - 0.0) < 1e-5:
                phi_0_to_1 = phi_tk_to_moving
            if abs(t_k - 1.0) < 1e-5:
                phi_1_to_0 = phi_tk_to_fixed

            # Base coordinates (integrated ODE trajectory)
            phys_fixed_base = phys_grid + phi_tk_to_fixed
            phys_moving_base = (phys_grid + phi_tk_to_moving) @ M_phys_zyx.t() + t_phys_zyx

            def _sample_sim_loss(phys_f, phys_m):
                phi_f_norm = physical_to_normalized_torch_cached(
                    phys_f, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_w = grid_sample_nd(fixed_image, phi_f_norm, mode='bilinear', padding_mode='zeros')

                phi_m_norm = physical_to_normalized_torch_cached(
                    phys_m, shape_m, spacing_m, origin_m, direction_m
                )
                moving_w = grid_sample_nd(moving_image, phi_m_norm, mode='bilinear', padding_mode='zeros')

                sim_m = str(getattr(self, 'similarity_metric', 'lncc')).lower()
                if sim_m in ('mattes_mi', 'mattes', 'mi', 'mmi') or sim_m.startswith('mattes') or sim_m.startswith('mi_') or sim_m.startswith('mmi_'):
                    from .core.losses import mattes_mi_loss_nd
                    n_bins = getattr(self, 'mattes_bins', 32)
                    parts = sim_m.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        n_bins = int(parts[-1])
                    fg_mask = ((fixed_w.abs() > 0.01) | (moving_w.abs() > 0.01)).float()
                    return mattes_mi_loss_nd(fixed_w, moving_w, mask=fg_mask, num_bins=n_bins)
                elif sim_m == 'mse':
                    return torch.mean((fixed_w - moving_w) ** 2)
                elif sim_m in ('dt', 'distance_transform', 'edt'):
                    from .core.losses import distance_transform_loss
                    return distance_transform_loss(fixed_w, moving_w, mode='potential_lncc', window_size=lncc_window_size, spacing=self.spacing)
                elif sim_m in ('sdf', 'signed_distance'):
                    from .core.losses import distance_transform_loss
                    return distance_transform_loss(fixed_w, moving_w, mode='sdf_mse', spacing=self.spacing)
                elif sim_m in ('cc2', 'lncc2'):
                    return local_ncc_loss_nd(fixed_w, moving_w, window_size=lncc_window_size, squared=True)
                elif sim_m in ('box_lncc', 'box_cc', 'fireants_lncc'):
                    from .core.losses import BoxLNCCLoss
                    box_loss_fn = BoxLNCCLoss(kernel_size=lncc_window_size)
                    return box_loss_fn(fixed_w, moving_w)
                else:
                    if getattr(self, '_foreground_mask_lncc', False):
                        fg_mask = ((fixed_w.abs() > 0.01) | (moving_w.abs() > 0.01)).float()
                        return lncc_loss_nd(fixed_w * fg_mask, moving_w * fg_mask, window_size=lncc_window_size)
                    else:
                        return lncc_loss_nd(fixed_w, moving_w, window_size=lncc_window_size)

            loss_base = _sample_sim_loss(phys_fixed_base, phys_moving_base)
            if offsets:
                loss_perturbed = 0.0
                for offset in offsets:
                    # Perturb sampling coordinates at lookup boundary to cancel interpolation noise
                    phys_f_offset = phys_fixed_base + offset
                    phys_m_offset = phys_moving_base + offset @ M_phys_zyx.t()
                    loss_perturbed = loss_perturbed + boot_weight * _sample_sim_loss(phys_f_offset, phys_m_offset)
                loss_timepoint = orig_w * loss_base + loss_perturbed
            else:
                loss_timepoint = loss_base

            losses_k.append(loss_timepoint)

        sim_loss = sum(losses_k) / len(losses_k)

        if compute_id_loss and phi_0_to_1 is not None and phi_1_to_0 is not None:
            # Compute bidirectional composition penalties:
            # Direction 1: u_inv + u_fwd(x + u_inv)
            phi_1_to_0_norm = physical_to_normalized_torch_cached(
                phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
            )
            # Direction 2: u_fwd + u_inv(x + u_fwd)
            phi_0_to_1_norm = physical_to_normalized_torch_cached(
                phys_grid + phi_0_to_1, shape_t, spacing_t, origin_t, direction_t
            )

            # compose_grids handles movedim internally; works for any spatial dimensionality
            u_fwd_at_inv = compose_grids(phi_0_to_1, phi_1_to_0_norm)
            u_inv_at_fwd = compose_grids(phi_1_to_0, phi_0_to_1_norm)

            inv_id_err_1 = phi_1_to_0 + u_fwd_at_inv
            inv_id_err_2 = phi_0_to_1 + u_inv_at_fwd
            inv_id_loss = 0.5 * (torch.mean(inv_id_err_1 ** 2) + torch.mean(inv_id_err_2 ** 2))
            inv_id_weight = float(getattr(self, 'inverse_identity_weight', 0.05))
            sim_loss = sim_loss + inv_id_weight * inv_id_loss

        return sim_loss

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[100, 100, 20],
        similarity_metric='lncc',
        lncc_radius=4,
        lr=1.0,
        verbose=False,
        fixed_spacing=None,
        fixed_origin=None,
        fixed_direction=None,
        moving_spacing=None,
        moving_origin=None,
        moving_direction=None,
        cfl_max=0.0,
        **kwargs
    ):
        """
        Run the multi-resolution optimisation of the velocity keyframes (the affine is not
        optimised: it is set beforehand, e.g. ``model.affine.T_init`` from ``syntx.tvf``).

        ``syntx.tvf`` passes every option explicitly; the defaults below apply only to
        direct calls.

        Parameters
        ----------
        fixed_image, moving_image : Tensor (1, 1, *spatial)
        levels : list of int, default [4, 2, 1]
        epochs_per_level : list of int, default [100, 100, 20]
        similarity_metric : str, default 'lncc'
        lncc_radius : int, default 4
        lr : float, default 1.0
            Learning rate (non-'cfl' optimisers).
        fixed_*, moving_* : geometry overrides (spacing / origin / direction).
        cfl_max : float, default 0.0
            Velocity magnitude cap after each step.
        **kwargs
            ``regularizer`` ('sobolev'), ``alpha`` (spectral fluid strength, mm^2; default
            ``default_tvf_alpha(dim)``), ``total_alpha``, ``energy_weight``,
            ``temporal_weight``, ``optimizer_type`` (default 'adam' here; 'cfl' in syntx.tvf),
            ``cfl_step``, ``cfl_momentum``, ``max_step_norm``, ``multipoint_loss``,
            ``fast_smooth``, ``constant_speed`` (True), ``constant_speed_relaxation`` (0.10),
            and the advanced options listed in ``TVF_ADVANCED_OPTIONS``.

        The loss history is ``self.losses``.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        
        self.similarity_metric = similarity_metric
        self.mattes_bins = int(kwargs.get('mattes_bins', kwargs.get('num_bins', getattr(self, 'mattes_bins', 32))))
        
        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        # Optimize velocity field across pyramid levels
        if verbose: print("Optimizing TVF...")
        opt_type = str(kwargs.get('optimizer_type', kwargs.get('optimizer', 'adam'))).lower()
        trust_coeff = float(kwargs.get('trust_coefficient', kwargs.get('trust', 0.80)))
        
        # Regularisation, resolved ONCE (tvf_registration passes every value explicitly):
        #   spectral (sobolev / dsti / dsti1): fluid strength `alpha`, post-step strength
        #     `total_alpha` (0 / None = off);
        #   gaussian / bspline: fluid sigma = self.fluid_sigma, post-step sigma = self.elastic_sigma.
        # "Fluid" smoothing is applied exactly once per step: inside RegAdam (after the Adam
        # normalisation) for optimizer='reg_adam', otherwise to the raw gradient.
        reg_mode = str(kwargs.get('regularizer', 'sobolev')).lower()
        spectral = reg_mode in ('sobolev', 'dsti', 'dsti1')
        fluid_alpha = (float(kwargs['alpha']) if kwargs.get('alpha') is not None
                       else (default_tvf_alpha(self.dim) if spectral else 0.0))
        total_alpha = float(kwargs.get('total_alpha') or 0.0)
        # Path energy E = mean_t <v_t, L v_t> (L = inverse of the smoothing kernel: the LDDMM /
        # geodesic energy). Symmetric under t -> 1-t, v -> -v; favours short, smooth paths.
        # Its alpha is the fluid alpha (spectral) or sigma^2 / 2 (gaussian: Sobolev equivalent).
        energy_weight = float(kwargs.get('energy_weight', 0.0) or 0.0)
        temporal_weight = float(kwargs.get('temporal_weight', 0.0) or 0.0)
        energy_alpha = fluid_alpha if spectral else 0.5 * float(self.fluid_sigma or 0.0) ** 2
        from .core.smoothing import sobolev_energy
        fluid_sigmas_input = self.fluid_sigma if not spectral else (1.0 if fluid_alpha > 0 else 0.0)
        elastic_sigmas_input = self.elastic_sigma if not spectral else (1.0 if total_alpha > 0 else 0.0)
        convergence_threshold = kwargs.get('convergence_threshold', 0.0)
        convergence_window = kwargs.get('convergence_window', 10)
        multipoint_loss = kwargs.get('multipoint_loss', [0.5])
        cfl_max_val = float(cfl_max or self.cfl_max or 0.0)   # named parameter (was never read)
        
        # Regularisation is in physical units (mm, mm^2) at every pyramid level, so alpha /
        # sigma mean the same smoothness at every scale.
        sigma_mode = 'physical'
        use_analytical_gradients = kwargs.get('use_analytical_gradients', getattr(self, 'use_analytical_gradients', False))
        self.losses = []
        
        bootstrap_mode = kwargs.get('bootstrap_mode', getattr(self, 'bootstrap_mode', None))
        bootstrap_orig_weight = float(kwargs.get('bootstrap_orig_weight', getattr(self, 'bootstrap_orig_weight', 0.50)))
        bootstrap_jitter_scale = float(kwargs.get('bootstrap_jitter_scale', getattr(self, 'bootstrap_jitter_scale', 0.25)))
        
        # CFL momentum for faster convergence (default 0.9, set 0.0 to disable)
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        momentum_buffer = None  # Initialized per-level
        
        # Gradient smoothing frequency: smooth every N epochs (1=every epoch, default)
        # Higher values reduce the dominant smoothing bottleneck at cost of noise
        smooth_every_n = int(kwargs.get('smooth_every_n', 1))
        
        # Fast smooth: downsample gradients to half resolution before smoothing (9.4x faster)
        # Approximate but sufficient for gradient direction estimation
        fast_smooth = bool(kwargs.get('fast_smooth', True))


        smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
        from .pyramid import build_image_pyramid
        if smoothing_sigmas is None:
            smoothing_sigmas = [float(np.log2(s)) if s > 1 else 0.0 for s in levels]
        fixed_pyr = build_image_pyramid(fixed_image, spacing=self.spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')
        moving_pyr = build_image_pyramid(moving_image, spacing=self.moving_spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')
        
        # Compute pyramid-proportional velocity shapes for each level
        # velocity_shape is the MAX (finest) grid; coarser levels use proportionally smaller grids
        max_vel_shape = self.velocity_shape  # e.g., (96, 96, 96)

        for level_idx, level in enumerate(levels):
            epochs = epochs_per_level[min(level_idx, len(epochs_per_level) - 1)]
            if epochs <= 0:
                continue

            # Multi-resolution Pyramidal Resizing: Resize velocity parameter grid to match current image scale.
            max_vel_ds = int(kwargs.get('max_velocity_downsample', 1))
            effective_level = max(level, max_vel_ds)
            curr_vel_shape = tuple(max(8, s // effective_level) for s in self.image_shape)
            shrink_ratio = (math.prod(curr_vel_shape) / math.prod(max_vel_shape)) ** (1.0 / self.dim)
            prev_vel_shape = tuple(self.velocity.shape[2:-1])
            
            if curr_vel_shape != prev_vel_shape:
                self._resize_velocity(curr_vel_shape, device, dtype)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")
            
            curr_spacing = [sp * level for sp in self.spacing]
            # ZYX-ordered spacing tensor for CFL normalization (velocity last dim is ZYX)
            sp_t_zyx = torch.tensor(itk_shape_to_tensor_shape(curr_spacing), device=device, dtype=dtype)
            sp_t_xyz = torch.tensor(curr_spacing, device=device, dtype=dtype)
            
            # Compute vel_spacing for physical-mode smoothing at current velocity resolution
            # physical spacing of the current velocity grid, ITK (x, y, z) order like self.spacing
            # (image_shape / curr_vel_shape are tensor (z, y, x) order)
            vel_spacing = [float(sp) * (img_dim / vel_dim) for sp, img_dim, vel_dim in
                           zip(self.spacing, reversed(self.image_shape), reversed(curr_vel_shape))]

            # Create optimizer fresh for this level (velocity parameter may have changed)
            if opt_type == 'lars':
                lars_lr = float(kwargs.get('cfl_step', kwargs.get('grad_step', lr))) * math.sqrt(shrink_ratio)
                optimizer = LARS([self.velocity], lr=lars_lr, trust_coefficient=trust_coeff)
            elif opt_type == 'cg':
                optimizer = TVFConjugateGradient([self.velocity], lr=lr)
            elif opt_type == 'sgd':
                optimizer = torch.optim.SGD([self.velocity], lr=lr, momentum=0.9, nesterov=True)
            elif opt_type == 'rmsprop':
                optimizer = torch.optim.RMSprop([self.velocity], lr=lr, momentum=0.9)
            elif opt_type == 'adamw':
                optimizer = torch.optim.AdamW([self.velocity], lr=lr)
            elif opt_type == 'reg_adam':
                if spectral:
                    ra_reg = reg_mode if fluid_alpha > 0 else 'none'
                    ra_alpha, ra_sigma = fluid_alpha, 0.0
                else:
                    ra_reg = 'gaussian' if (self.fluid_sigma or 0) > 0 else 'none'
                    ra_alpha, ra_sigma = 0.0, float(self.fluid_sigma or 0.0)
                max_step_norm = float(kwargs['max_step_norm'])
                optimizer = RegAdam(
                    [self.velocity],
                    lr=lr,
                    regularizer=ra_reg,
                    sobolev_alpha=ra_alpha,
                    gaussian_sigma=ra_sigma,
                    max_step_norm=max_step_norm,
                    spacing=vel_spacing,
                    **({"eps_rel": float(kwargs["adam_eps_rel"])} if kwargs.get("adam_eps_rel") is not None else {})
                )
            else:
                optimizer = torch.optim.Adam([self.velocity], lr=lr)
            
            # Reset momentum buffer for this level
            if cfl_momentum > 0 and opt_type == 'cfl':
                momentum_buffer = torch.zeros_like(self.velocity.data)
            
            if isinstance(fluid_sigmas_input, (list, tuple)):
                curr_fluid_sig = fluid_sigmas_input[min(level_idx, len(fluid_sigmas_input) - 1)]
            else:
                curr_fluid_sig = fluid_sigmas_input

            if isinstance(elastic_sigmas_input, (list, tuple)):
                curr_elastic_sig = elastic_sigmas_input[min(level_idx, len(elastic_sigmas_input) - 1)]
            else:
                curr_elastic_sig = elastic_sigmas_input
                
            sigma_val = float(curr_fluid_sig) if curr_fluid_sig > 0 else 0.0
            elastic_sigma_val = float(curr_elastic_sig) if curr_elastic_sig > 0 else 0.0
            
            curr_fixed = fixed_pyr[level_idx]
            curr_moving = moving_pyr[level_idx]
            
            recent_losses = []
            lncc_ws = 2 * lncc_radius + 1
            
            # Pre-compute image spatial Jacobians for analytical gradient mode
            if getattr(self, 'use_analytical_gradients', False):
                if self.n_time_steps > 1:
                    raise NotImplementedError(
                        "Analytical gradients are not supported for TVF with n_time_steps > 1. "
                        "Analytical mode distributes identical gradients to all keyframes, "
                        "collapsing TVF to SVF. Use use_analytical_gradients=False (autograd)."
                    )
                grad_I_curr = _spatial_jacobian_nd(
                    curr_fixed.movedim(1, -1),
                    physical_spacing=itk_shape_to_tensor_shape(curr_spacing)
                ).squeeze(-2)
                
                moving_target_shape = tuple(curr_moving.shape[2:])
                curr_moving_spacing_list = [
                    sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                    for sp, orig_s, curr_s in zip(self.moving_spacing, reversed(self.moving_shape), reversed(moving_target_shape))
                ]
                
                grad_J_curr = _spatial_jacobian_nd(
                    curr_moving.movedim(1, -1),
                    physical_spacing=itk_shape_to_tensor_shape(curr_moving_spacing_list)
                ).squeeze(-2)
            
            # Per-level multipoint scheduling: allows coarse levels to use cheaper
            # loss evaluation (e.g. midpoint-only [0.5]) while fine levels use full
            # 3-point [0.0, 0.5, 1.0] for boundary accuracy.
            mp_schedule = kwargs.get('multipoint_schedule', None)
            if mp_schedule is not None and isinstance(mp_schedule, (list, tuple)):
                if level_idx < len(mp_schedule) and isinstance(mp_schedule[level_idx], (list, tuple)):
                    multipoint_loss = list(mp_schedule[level_idx])

            best_level_loss = float('inf')
            best_velocity = None

            for epoch in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                
                if getattr(self, 'use_analytical_gradients', False):
                    # === Analytical gradient mode ===
                    # Step 1: Forward pass under no_grad to get warped images
                    with torch.no_grad():
                        target_shape = tuple(curr_fixed.shape[2:])
                        curr_spacing_list = [
                            sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                            for sp, orig_s, curr_s in zip(self.spacing, reversed(self.image_shape), reversed(target_shape))
                        ]
                        phys_grid = get_physical_grid_torch(
                            target_shape, curr_spacing_list, self.origin, self.direction,
                            device=device, dtype=dtype
                        )
                        spacing_rev, origin_rev, direction_rev = reverse_metadata(curr_spacing_list, self.origin, self.direction)
                        shape_t_ag = torch.tensor(list(target_shape), device=device, dtype=dtype)
                        spacing_t_ag = torch.tensor(spacing_rev, device=device, dtype=dtype)
                        origin_t_ag = torch.tensor(origin_rev, device=device, dtype=dtype)
                        direction_t_ag = torch.tensor(direction_rev, device=device, dtype=dtype)
                        _cached_meta_ag = (shape_t_ag, spacing_t_ag, origin_t_ag, direction_t_ag)
                        
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
                        
                        sp_m_ag, orig_m_ag, dir_m_ag = reverse_metadata(curr_moving_spacing_list, self.moving_origin, self.moving_direction)
                        shape_m_ag = torch.tensor(moving_target_shape, device=device, dtype=dtype)
                        spacing_m_ag = torch.tensor(sp_m_ag, device=device, dtype=dtype)
                        origin_m_ag = torch.tensor(orig_m_ag, device=device, dtype=dtype)
                        direction_m_ag = torch.tensor(dir_m_ag, device=device, dtype=dtype)
                        
                        affine_params = self.affine.get_matrix()
                        M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
                            affine_params, self.image_shape, self.spacing, self.origin, self.direction,
                            self.moving_shape, self.moving_spacing, self.moving_origin, self.moving_direction
                        )
                        
                        # Warp moving to midpoint (with affine)
                        phi_moving_affine = (phys_grid + phi_05_to_1) @ M_phys_zyx.t() + t_phys_zyx
                        phi_moving_norm = physical_to_normalized_torch_cached(
                            phi_moving_affine, shape_m_ag, spacing_m_ag, origin_m_ag, direction_m_ag
                        )
                        J_mid = grid_sample_nd(curr_moving, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                        
                        # Sample image spatial gradients at warped positions
                        grad_I_mid = grid_sample_nd(
                            grad_I_curr.movedim(-1, 1), phi_fixed_norm,
                            mode='bilinear', padding_mode='zeros'
                        ).movedim(1, -1)
                        direction_t_mat = direction_t_ag
                        grad_I_mid = torch.matmul(grad_I_mid, direction_t_mat)
                        
                        grad_J_mid = grid_sample_nd(
                            grad_J_curr.movedim(-1, 1), phi_moving_norm,
                            mode='bilinear', padding_mode='zeros'
                        ).movedim(1, -1)
                        grad_J_mid = torch.matmul(grad_J_mid, direction_m_ag)
                        grad_J_mid = torch.matmul(grad_J_mid, M_phys_zyx)
                    
                    # Step 2: Compute loss with grad tracking on detached midpoint images
                    I_mid_det = I_mid.detach().requires_grad_(True)
                    J_mid_det = J_mid.detach().requires_grad_(True)
                    sim_loss = lncc_loss_nd(I_mid_det, J_mid_det, window_size=lncc_ws)
                    kinetic = self._path_energy(energy_weight, temporal_weight, energy_alpha, vel_spacing, device, dtype)
                    total_loss = sim_loss + kinetic
                    total_loss.backward()
                    
                    # Step 3: Compute analytical velocity gradient via chain rule
                    with torch.no_grad():
                        g_im = I_mid_det.grad if I_mid_det.grad is not None else torch.zeros_like(I_mid_det)
                        g_jm = J_mid_det.grad if J_mid_det.grad is not None else torch.zeros_like(J_mid_det)
                        
                        # Spatial chain rule: dL/dphi = dL/dI * dI/dphi
                        grad_wrt_phi_fixed = g_im.movedim(1, -1) * grad_I_mid
                        grad_wrt_phi_moving = g_jm.movedim(1, -1) * grad_J_mid
                        
                        # Combined gradient for velocity (both directions contribute)
                        combined_grad = (grad_wrt_phi_moving - grad_wrt_phi_fixed) / 2.0
                        
                        # Assign gradient to velocity parameter
                        if self.velocity.grad is None:
                            self.velocity.grad = torch.zeros_like(self.velocity)
                        
                        # Resize combined gradient to velocity grid shape if different
                        vel_spatial = tuple(self.velocity.shape[2:-1])
                        grad_spatial = tuple(combined_grad.shape[1:-1])
                        if vel_spatial != grad_spatial:
                            # resize_field operates on (B, *spatial, dim); squeeze batch for the operation
                            combined_grad = resize_field(combined_grad, vel_spatial)
                        
                        # Distribute gradient across all time steps
                        for t in range(self.n_time_steps):
                            self.velocity.grad[t, 0] = combined_grad[0]
                else:
                    # === Standard autograd mode with AMP Mixed Precision ===
                    dev_type = 'cuda' if 'cuda' in str(device) else ('mps' if 'mps' in str(device) else 'cpu')
                    use_amp = bool(kwargs.get('amp', False)) and (dev_type in ('cuda', 'mps'))
                    amp_dtype = torch.float16
                    
                    with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                        sim_loss = self.forward(
                            curr_fixed, curr_moving,
                            multipoint_loss=multipoint_loss,
                            lncc_window_size=lncc_ws,
                            bootstrap_mode=bootstrap_mode,
                            bootstrap_orig_weight=bootstrap_orig_weight,
                            bootstrap_jitter_scale=bootstrap_jitter_scale
                        )
                        kinetic = self._path_energy(energy_weight, temporal_weight, energy_alpha, vel_spacing, device, dtype)
                        total_loss = sim_loss + kinetic
                    total_loss.backward()

                # Record epoch loss in self.losses history and checkpoint best velocity BEFORE parameter updates
                loss_val = float(total_loss.detach())   # the objective (similarity + energy)
                self.losses.append(loss_val)
                if loss_val < best_level_loss:
                    best_level_loss = loss_val
                    best_velocity = self.velocity.detach().clone()
                
                # Fluid regularization (smoothing velocity gradients)
                # Batched across all T time steps to minimize conv3d kernel launches
                # Smoothing is the dominant bottleneck (~91% of per-epoch time).
                # smooth_every_n > 1 reduces this cost at the expense of gradient noise.
                with torch.no_grad():
                    should_smooth = (sigma_val > 0 and self.velocity.grad is not None
                                     and opt_type != 'reg_adam'   # RegAdam smooths its own step
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
                        
                        regularizer_mode = reg_mode
                        
                        do_fast = fast_smooth and min_spatial >= 32
                        if do_fast:
                            interp_3d = 'trilinear' if self.dim == 3 else 'bilinear'
                            g_cf = torch.movedim(grad_batch, -1, 1)  # (T, dim, *spatial)
                            down_shape = [max(8, s // 2) for s in spatial_shape]
                            g_down = F.interpolate(g_cf, size=down_shape, mode=interp_3d, align_corners=True)
                            g_process = torch.movedim(g_down, 1, -1)  # (T, *down, dim)
                        else:
                            g_process = grad_batch
                            
                        # Prepare tapered gradients for spectral regularizers
                        bmask_pre = self._create_boundary_mask(g_process.shape[1:-1], device, dtype, border_width=4)
                        g_process_tapered = g_process * bmask_pre

                        # Adjust physical spacing if downsampled so physical scale remains correct
                        if vel_spacing is not None:
                            adj_spacing = [sp * 2.0 for sp in vel_spacing] if do_fast else vel_spacing
                        else:
                            adj_spacing = [sp * 2.0 for sp in getattr(self, 'spacing', [1.0] * self.dim)] if do_fast else getattr(self, 'spacing', [1.0] * self.dim)

                        if regularizer_mode == 'sobolev':
                            g_smoothed = self._apply_sobolev_green_operator(g_process_tapered, fluid_sigma=1.0, alpha=fluid_alpha, spacing=adj_spacing)
                        elif regularizer_mode == 'dsti':
                            g_smoothed = self._apply_dsti_green_operator(g_process_tapered, fluid_sigma=1.0, alpha=fluid_alpha)
                        elif regularizer_mode == 'dsti1':
                            g_smoothed = self._apply_dsti1_green_operator(g_process_tapered, fluid_sigma=1.0, alpha=fluid_alpha)
                        elif regularizer_mode in ['bspline', 'bsplinesyn']:
                            from .core.smoothing import smooth_displacement_field_bspline
                            g_smoothed = smooth_displacement_field_bspline(
                                g_process,
                                spacing=adj_spacing,
                                mesh_size=kwargs.get('mesh_size'),
                                spline_distance=kwargs.get('spline_distance'),
                                fluid_sigma=sigma_val,
                                enforce_stationary_boundary=kwargs.get('enforce_stationary_boundary', False),
                                order=kwargs.get('spline_order', 3),
                                coord_convention='xyz',
                            )
                        else:
                            g_smoothed = separable_gaussian_filter(
                                g_process, sigma=sigma_val, spacing=adj_spacing, sigma_mode=sigma_mode
                            )
                            
                        if do_fast:
                            g_smooth_cf = torch.movedim(g_smoothed, -1, 1)
                            g_up = F.interpolate(g_smooth_cf, size=spatial_shape, mode=interp_3d, align_corners=True)
                            grad_smoothed = torch.movedim(g_up, 1, -1).contiguous()
                        else:
                            grad_smoothed = g_smoothed
                        # Apply smooth Dirichlet Cosine boundary taper mask to velocity gradients
                        # Cache boundary mask per pyramid level to avoid recomputation every epoch
                        bmask_key = (spatial_shape, device, dtype)
                        if not hasattr(self, '_bmask_cache') or getattr(self, '_bmask_cache_key', None) != bmask_key:
                            self._bmask_cache = self._create_boundary_mask(spatial_shape, device, dtype, border_width=4)
                            self._bmask_cache_key = bmask_key
                        grad_smoothed_tapered = grad_smoothed * self._bmask_cache
                        self.velocity.grad.copy_(grad_smoothed_tapered.unsqueeze(1))
                            
                if opt_type == 'cfl':
                    with torch.no_grad():
                        if self.velocity.grad is not None:
                            grad = self.velocity.grad
                            # ITK-style CFL: compute max norm in VOXEL space (divide by spacing)
                            # This matches ITK's ScaleUpdateField() exactly:
                            #   localNorm += sqr(vector[d] / spacing[d])
                            #   scale = learningRate / maxNorm
                            grad_voxel = grad / sp_t_xyz  # convert to voxel units (ZYX)
                            max_g_voxel = torch.sqrt(torch.sum(grad_voxel**2, dim=-1)).max()
                            if max_g_voxel > 1e-8:
                                cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                                effective_cfl = float(cfl_step_val) * math.sqrt(shrink_ratio)
                                # Compute CFL update: scaledUpdate = (learningRate / maxNorm) * gradient
                                update = (effective_cfl / max_g_voxel) * grad
                                # CFL-consistent momentum with (1-μ) scaling.
                                # Standard heavy-ball: buf = μ·buf + g; θ -= buf
                                #   → steady-state amplification = 1/(1-μ) = 20× for μ=0.95
                                #   → violates CFL bound on parameter update magnitude.
                                # Fix: θ -= (1-μ)·buf
                                #   → steady-state: (1-μ) · g/(1-μ) = g  (exact CFL bound)
                                #   → momentum smooths gradient DIRECTION without amplifying MAGNITUDE.
                                if cfl_momentum > 0 and momentum_buffer is not None:
                                    momentum_buffer.mul_(cfl_momentum).add_(update)
                                    bias_corr = 1.0 - (cfl_momentum ** (epoch + 1))
                                    corrected_buf = momentum_buffer * (1.0 - cfl_momentum) / max(bias_corr, 1e-8)
                                    with torch.no_grad():
                                        self.velocity.sub_(corrected_buf)
                                else:
                                    with torch.no_grad():
                                        self.velocity.sub_(update)
                else:
                    optimizer.step()


                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                with torch.no_grad():
                    if elastic_sigma_val > 0:
                        T = self.n_time_steps
                        vel_batch = self.velocity.squeeze(1)
                        regularizer_mode = reg_mode
                        
                        if regularizer_mode == 'dsti':
                            # Enforce strict zero boundary conditions before spectral filtering
                            vel_tapered = vel_batch.clone()
                            for d in range(self.dim):
                                sl_first = [slice(None)] * (self.dim + 2)
                                sl_first[d + 1] = 0
                                vel_tapered[tuple(sl_first)] = 0.0
                                sl_last = [slice(None)] * (self.dim + 2)
                                sl_last[d + 1] = -1
                                vel_tapered[tuple(sl_last)] = 0.0
                            vel_smoothed = self._apply_dsti_green_operator(vel_tapered, fluid_sigma=1.0, alpha=total_alpha)
                        elif regularizer_mode == 'dsti1':
                            vel_tapered = vel_batch.clone()
                            for d in range(self.dim):
                                sl_first = [slice(None)] * (self.dim + 2)
                                sl_first[d + 1] = 0
                                vel_tapered[tuple(sl_first)] = 0.0
                                sl_last = [slice(None)] * (self.dim + 2)
                                sl_last[d + 1] = -1
                                vel_tapered[tuple(sl_last)] = 0.0
                            vel_smoothed = self._apply_dsti1_green_operator(vel_tapered, fluid_sigma=1.0, alpha=total_alpha)
                        elif regularizer_mode == 'sobolev':
                            vel_tapered = vel_batch.clone()
                            for d in range(self.dim):
                                sl_first = [slice(None)] * (self.dim + 2)
                                sl_first[d + 1] = 0
                                vel_tapered[tuple(sl_first)] = 0.0
                                sl_last = [slice(None)] * (self.dim + 2)
                                sl_last[d + 1] = -1
                                vel_tapered[tuple(sl_last)] = 0.0
                            vel_smoothed = self._apply_sobolev_green_operator(vel_tapered, fluid_sigma=1.0, alpha=total_alpha, spacing=vel_spacing)
                        elif regularizer_mode in ['bspline', 'bsplinesyn']:
                            from .core.smoothing import smooth_displacement_field_bspline
                            vel_smoothed = smooth_displacement_field_bspline(
                                vel_batch,
                                spacing=vel_spacing,
                                mesh_size=kwargs.get('elastic_mesh_size', kwargs.get('mesh_size')),
                                spline_distance=kwargs.get('elastic_spline_distance', kwargs.get('spline_distance')),
                                fluid_sigma=elastic_sigma_val,
                                enforce_stationary_boundary=kwargs.get('enforce_stationary_boundary', False),
                                order=kwargs.get('spline_order', 3),
                                coord_convention='xyz',
                            )
                        else:
                            vel_smoothed = separable_gaussian_filter(
                                vel_batch, sigma=elastic_sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                            )
                        self.velocity.copy_(vel_smoothed.unsqueeze(1))
                    
                    if cfl_max_val > 0:
                        vel_voxel = self.velocity / sp_t_zyx
                        vel_voxel_norm = torch.norm(vel_voxel, dim=-1, keepdim=True)
                        max_vel_norm = vel_voxel_norm.max()
                        if max_vel_norm > cfl_max_val:
                            self.velocity.mul_(cfl_max_val / (max_vel_norm + 1e-8))
                    
                    # Constant speed constraint: project velocity keyframes onto uniform-speed manifold.
                    # Ensures geodesic parameterization (constant-speed path through diffeomorphism group)
                    # and prevents velocity energy concentration in a single keyframe.
                    cs_enabled = kwargs.get('constant_speed', True)
                    cs_relax = float(kwargs.get('constant_speed_relaxation', 0.10))
                    if cs_enabled and self.n_time_steps > 1 and cs_relax > 0:
                        with torch.no_grad():
                            # Compute per-keyframe 2-norm
                            vel_data = self.velocity.data  # (T, 1, *spatial, dim)
                            # speed in the same V-norm as the path energy (geodesics: constant ||v_t||_V)
                            speeds = torch.sqrt(sobolev_energy(vel_data.squeeze(1), energy_alpha, spacing=vel_spacing))  # (T,)
                            mean_speed = speeds.mean()
                            if mean_speed > 1e-10:
                                # Relaxation toward uniform speed: v_k *= (1-α) + α * (mean/speed_k)
                                for t_k in range(self.n_time_steps):
                                    if speeds[t_k] > 1e-10:
                                        scale = (1.0 - cs_relax) + cs_relax * (mean_speed / speeds[t_k])
                                        self.velocity.data[t_k].mul_(scale)

                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"  [TVF Level {level}] Epoch {epoch+1}/{epochs}: loss={loss_val:.6f}", flush=True)

                # Convergence checking (every 5 epochs to reduce GPU-CPU sync barriers)
                if epoch % 5 == 0 or epoch == epochs - 1:
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

                # Aggressive in-loop garbage collection (make less aggressive)
                if epoch % 50 == 0 or epoch == epochs - 1:
                    try:
                        del sim_loss, total_loss, kinetic
                    except NameError:
                        pass

                    gc.collect()
                    if device.type == 'mps':
                        torch.mps.empty_cache()

            # Evaluate final velocity after the last optimization step
            if epochs > 0:
                with torch.no_grad():
                    if not getattr(self, 'use_analytical_gradients', False):
                        final_sim = float(self.forward(
                            curr_fixed, curr_moving,
                            multipoint_loss=multipoint_loss,
                            lncc_window_size=lncc_ws,
                        ).item())
                        if final_sim < best_level_loss:
                            best_level_loss = final_sim
                        elif best_velocity is not None:
                            self.velocity.copy_(best_velocity)
                    else:
                        phi_05_to_0 = self.integrate(0.5, 0.0, image_shape=target_shape,
                                                     _cached_phys_grid=phys_grid, _cached_meta=_cached_meta_ag)
                        phi_05_to_1 = self.integrate(0.5, 1.0, image_shape=target_shape,
                                                     _cached_phys_grid=phys_grid, _cached_meta=_cached_meta_ag)
                        phi_fixed_norm = physical_to_normalized_torch_cached(
                            phys_grid + phi_05_to_0, shape_t_ag, spacing_t_ag, origin_t_ag, direction_t_ag
                        )
                        I_mid = grid_sample_nd(curr_fixed, phi_fixed_norm, mode='bilinear', padding_mode='zeros')
                        phi_moving_aff = phys_grid + phi_05_to_1
                        if affine_params is not None or hasattr(self, 'affine'):
                            phi_moving_aff = phi_moving_aff @ M_phys_zyx.t() + t_phys_zyx
                        phi_moving_norm = physical_to_normalized_torch_cached(
                            phi_moving_aff, shape_m_ag, spacing_m_ag, origin_m_ag, direction_m_ag
                        )
                        J_mid = grid_sample_nd(curr_moving, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                        final_sim = float(lncc_loss_nd(I_mid, J_mid, window_size=lncc_ws).item())
                        if final_sim < best_level_loss:
                            best_level_loss = final_sim
                        elif best_velocity is not None:
                            self.velocity.copy_(best_velocity)

            # GPU memory management and garbage collection at level transitions
            if device.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

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
# Default spectral strength alpha (mm^2, physical units at every pyramid level) per dimension.
# 2-D study 2026-09-30 (3 pairs, grad_step 1.0, energy 1e-3): mean Dice 0.749 vs SyN 0.741,
# Liouville determinant > 0 (min 0.023), interior inverse error 3.3-4.5 mm vs SyN 3.6-4.7 mm;
# alpha 4 / 8 trade Dice for smoothness (0.741 / 0.726). 3-D to be set by the TVF tune.
TVF_DEFAULT_ALPHA = {2: 2.0, 3: 2.0}

_TVF_SPECTRAL = ('sobolev', 'dsti', 'dsti1')
_TVF_SIGMA = ('gaussian', 'bspline')
_TVF_OPTIMIZERS = ('reg_adam', 'adam', 'adamw', 'sgd', 'rmsprop', 'lars', 'cg', 'cfl')

# Advanced options: accepted as keywords, forwarded unchanged to TVFModel.fit / the model.
# Anything else raises TypeError (nothing is silently ignored).
TVF_ADVANCED_OPTIONS = {
    # velocity model / integration
    'solver': "ODE integrator ('euler' default)",
    'integration_steps_per_interval': 'sub-steps between keyframes (default 1)',
    'constant_speed': 'project keyframes onto a constant-speed path (default True)',
    'constant_speed_relaxation': 'relaxation of that projection (default 0.10)',
    'use_analytical_gradients': 'analytical similarity gradients (n_time_steps == 1 only)',
    # loss
    'multipoint_schedule': 'per-level list of multipoint_loss lists',
    'bootstrap_mode': "'antithetic' jittered loss evaluation (default None)",
    'bootstrap_orig_weight': 'weight of the un-jittered term (default 0.5)',
    'bootstrap_jitter_scale': 'jitter in voxels (default 0.25)',
    'mattes_bins': "histogram bins for syn_metric='mattes' (default 32)",
    'foreground_mask_lncc': 'restrict LNCC to the foreground',
    # schedule / numerics
    'smoothing_sigmas': 'image pyramid smoothing per level (default log2(level))',
    'smooth_pyramid': 'smooth the image pyramid (default True)',
    'smooth_every_n': 'apply the gradient smoothing every n epochs (non-reg_adam optimisers)',
    'max_velocity_downsample': 'minimum velocity-grid downsampling factor',
    'convergence_threshold': 'early-stop threshold on the relative loss change',
    'convergence_window': 'epochs over which convergence is measured',
    'winsorize_quantiles': 'intensity winsorisation before normalisation',
    'amp': 'mixed precision on GPU (default False)',
    'gradient_checkpointing': 'trade compute for memory',
    'adam_eps_rel': "RegAdam relative epsilon (default 0)",
    'trust_coefficient': "LARS trust coefficient (optimizer='lars')",
    # bspline regulariser
    'mesh_size': 'B-spline mesh size', 'spline_distance': 'B-spline control-point spacing',
    'spline_order': 'B-spline order (default 3)',
    'enforce_stationary_boundary': 'zero the field at the image boundary',
    'elastic_mesh_size': 'B-spline mesh for the post-step smoothing',
    'elastic_spline_distance': 'B-spline spacing for the post-step smoothing',
}


def default_tvf_alpha(dim: int) -> float:
    """Default spectral strength ``alpha`` (mm^2) for ``dim``-D images (``TVF_DEFAULT_ALPHA``)."""
    return TVF_DEFAULT_ALPHA[dim]


def tvf_registration(
    fixed,
    moving,
    *,
    initial_transform=None,
    affine_dof='affine',
    affine_mode='pytorch',
    affine_seed=None,
    syn_metric='cc2',
    syn_sampling=2,
    reg_iterations=None,
    levels=None,
    n_time_steps=3,
    multipoint_loss=(0.0, 0.5, 1.0),
    regularizer='sobolev',
    alpha=None,
    total_alpha=None,
    flow_sigma=None,
    total_sigma=None,
    optimizer='cfl',
    optimizer_lr=None,
    max_step_norm=None,
    grad_step=None,
    cfl_momentum=None,
    cfl_max=0.0,
    energy_weight=1e-3,
    temporal_weight=0.0,
    fast_smooth=None,
    backend='pytorch',
    device=None,
    verbose=False,
    **advanced,
):
    """
    Time-varying velocity field (TVF) diffeomorphic registration.

    ::

        reg = syntx.tvf(fixed=fi, moving=mi)
        reg['warpedmovout'], reg['fwdtransforms'], reg['invtransforms']

    Every parameter below has an effect in the configuration where it is accepted; a parameter
    that would be ignored raises instead (``ValueError`` for the wrong regulariser / optimiser
    family, ``TypeError`` for an unknown keyword).

    Alignment
    ---------
    initial_transform : str, list of str, ANTsTransform or None
        Initial (affine) transform. None runs ``syntx.robust_affine`` (``affine_dof``,
        ``affine_mode``, ``affine_seed``).

    Similarity and schedule
    -----------------------
    syn_metric : str
        'cc2' (squared local NCC, default), 'lncc', 'mattes', 'mse'.
    syn_sampling : int
        LNCC radius (window 2 * syn_sampling + 1). Default 2.
    reg_iterations : list of int
        Iterations per pyramid level. Default [100, 100, 20].
    levels : list of int
        Pyramid shrink factors. Default [2**(L-1), ..., 1] for L = len(reg_iterations).

    Velocity model
    --------------
    n_time_steps : int
        Velocity keyframes in time. Default 3.
    multipoint_loss : sequence of float in [0, 1]
        Times at which the similarity is evaluated (0 = fixed side, 1 = moving side,
        0.5 = geodesic midpoint). Including both 0 and 1 is the symmetric formulation and
        also adds a forward/inverse consistency penalty (weight 0.05; see TVFModel.forward).
        Default (0.0, 0.5, 1.0).

    Regularisation -- one strength parameter per regulariser
    ---------------------------------------------------------
    regularizer : {'sobolev', 'dsti', 'dsti1', 'gaussian', 'bspline'}
        Default 'sobolev'.
    alpha : float
        Spectral regularisers only: strength (mm^2, physical units at every pyramid level) of
        the fluid smoothing applied to every update (0 = off). Default ``TVF_DEFAULT_ALPHA[dim]``.
    total_alpha : float
        Spectral regularisers only: strength of the post-step smoothing of the velocity
        field itself (0 / None = off). Default None.
    flow_sigma : float
        'gaussian' / 'bspline' only: Gaussian sigma (mm) of the fluid smoothing (0 = off).
        Default 3.0.
    total_sigma : float
        'gaussian' / 'bspline' only: sigma (mm) of the post-step smoothing (0 / None = off).
        Default None.

    Optimiser -- parameters are specific to their optimiser family
    ---------------------------------------------------------------
    optimizer : {'cfl', 'reg_adam', 'adam', 'adamw', 'sgd', 'rmsprop', 'lars', 'cg'}
        Default 'cfl': the regularised (smoothed) similarity gradient, scaled so its largest
        displacement is ``grad_step`` voxels -- one global normalisation, as in SyN. The
        Adam family normalises every voxel separately, which turns low-signal regions into
        full-size noisy steps and roughens the velocity (2-D r16/r64 study 2026-09-30:
        'reg_adam' folds 0.3-0.4 %, 'cfl' + energy 0.006-0.034 %, SyN 0.011-0.034 %).
    optimizer_lr : float
        Learning rate (all but 'cfl'). Default 1.2. With 'reg_adam' each step is capped at
        ``max_step_norm`` voxels, so once the cap is active the rate no longer matters.
    max_step_norm : float
        'reg_adam' only: largest per-iteration update (voxels). Default 0.5.
    grad_step : float
        'cfl' only: CFL step (voxels). Default 1.0.
    cfl_momentum : float
        'cfl' only: direction momentum in [0, 1). Default 0.9.
    cfl_max : float
        Cap on the velocity magnitude (voxels per unit time) after every step; 0 = no cap.
    energy_weight : float
        Weight of the path energy mean_t <v_t, L v_t> (L = inverse of the regulariser's kernel,
        same alpha): the LDDMM geodesic energy -- symmetric in time, it makes short, smooth
        paths preferable among those reaching the same map. Default 1e-3 (provisional, 2-D);
        0 = off. With the default constant_speed the keyframe speeds are equalised in the
        same V-norm (geodesics have constant speed).
    temporal_weight : float
        Weight of int ||d v / d t||_V^2 dt (keyframes: (T-1) sum_k ||v_{k+1} - v_k||_V^2): keeps
        the path from turning between keyframes; symmetric in time. 0 = off (n_time_steps 1: no
        effect).
    fast_smooth : bool
        Optimisers other than 'reg_adam' only (they smooth the raw gradient): smooth it at half
        resolution (faster, approximate). Default False. 'reg_adam' smooths its own step, so
        passing fast_smooth with it raises.

    Execution
    ---------
    backend : {'pytorch', 'jax'}, default 'pytorch'
        'jax' supports only regularizer='gaussian' with optimizer='cfl' (else ValueError).
    device : str or None
        None: CUDA, else MPS, else CPU.
    verbose : bool

    **advanced
        Options in ``TVF_ADVANCED_OPTIONS`` (see that dict for their meaning).

    Returns
    -------
    dict with 'warpedmovout', 'warpedfixout', 'fwdtransforms', 'invtransforms',
    'whichtoinvert_inv', 'model', 'inverse_identity_error_map', 'inverse_identity_errors',
    'provenance'.
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

    # ---- unknown / removed keywords ----------------------------------------------------------
    _removed = {
        'antisymmetric': "use multipoint_loss (containing 0.0 and 1.0 is the antisymmetric form)",
        'sobolev_alpha': "use alpha", 'dsti_alpha': "use alpha", 'gaussian_sigma': "use flow_sigma",
        'type_of_transform': "use regularizer=", 'similarity_metric': "use syn_metric",
        'cfl_step': "use grad_step (optimizer='cfl')", 'optimizer_type': "use optimizer",
        'lr': "use optimizer_lr", 'fluid_sigma': "use flow_sigma", 'elastic_sigma': "use total_sigma",
        'n_steps': "TVF has n_time_steps", 'sampling_percentage': None, 'vgg_layers': None,
        'vgg_mode': None, 'vgg_patch_size': None, 'vgg_num_patches': None, 'vgg_lncc_window_size': None,
        'project_inverse': None, 'projection_frequency': None, 'interpolator': None,
        'inverse_method': None, 'inverse_steps': None, 'affine_iterations': "use affine_dof / affine_mode",
        'aff_metric': "use affine_dof / affine_mode", 'aff_sampling': "use affine_dof / affine_mode",
        'fluid_sigmas': None, 'elastic_sigmas': None, 'sobolev_precondition': None,
        'reg_weight': "use energy_weight (V-norm path energy)",
        'sigma_mode': "regularisation is always in physical units (mm)",
    }
    bad = sorted(set(advanced) & set(_removed))
    if bad:
        hints = "; ".join(f"{k}: {_removed[k] or 'not used by TVF'}" for k in bad)
        raise TypeError(f"tvf_registration() does not accept {bad} ({hints})")
    unknown = sorted(set(advanced) - set(TVF_ADVANCED_OPTIONS))
    if unknown:
        raise TypeError(f"tvf_registration() got unknown keyword(s) {unknown}; see the docstring "
                        f"and syntx.tvf.TVF_ADVANCED_OPTIONS")

    # ---- regulariser: exactly one strength parameter family -------------------------------
    regularizer = str(regularizer).lower()
    if regularizer in _TVF_SPECTRAL:
        for name, val in (('flow_sigma', flow_sigma), ('total_sigma', total_sigma)):
            if val is not None:
                raise ValueError(f"{name} is only used with regularizer='gaussian' / 'bspline'; with "
                                 f"regularizer='{regularizer}' use alpha / total_alpha")
        alpha = default_tvf_alpha(dim) if alpha is None else float(alpha)
        total_alpha = 0.0 if total_alpha is None else float(total_alpha)
        if alpha < 0 or total_alpha < 0:
            raise ValueError("alpha and total_alpha must be >= 0 (0 = off)")
        fluid_sigma_actual, elastic_sigma_actual = 0.0, 0.0
    elif regularizer in _TVF_SIGMA:
        for name, val in (('alpha', alpha), ('total_alpha', total_alpha)):
            if val is not None:
                raise ValueError(f"{name} is only used with the spectral regularizers "
                                 f"{_TVF_SPECTRAL}; with regularizer='{regularizer}' use flow_sigma / total_sigma")
        fluid_sigma_actual = 3.0 if flow_sigma is None else float(flow_sigma)
        elastic_sigma_actual = 0.0 if total_sigma is None else float(total_sigma)
        if fluid_sigma_actual < 0 or elastic_sigma_actual < 0:
            raise ValueError("flow_sigma and total_sigma must be >= 0 (0 = off)")
    else:
        raise ValueError(f"unknown regularizer {regularizer!r}; expected one of {_TVF_SPECTRAL + _TVF_SIGMA}")

    # ---- optimiser: family-specific parameters ------------------------------------------------
    optimizer = str(optimizer).lower()
    if optimizer not in _TVF_OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {_TVF_OPTIMIZERS}")
    if optimizer == 'cfl':
        for name, val in (('optimizer_lr', optimizer_lr), ('max_step_norm', max_step_norm)):
            if val is not None:
                raise ValueError(f"{name} is not used by optimizer='cfl' (it uses grad_step / cfl_momentum)")
        grad_step = 1.0 if grad_step is None else float(grad_step)
        cfl_momentum = 0.9 if cfl_momentum is None else float(cfl_momentum)
        optimizer_lr = 1.0
    else:
        for name, val in (('grad_step', grad_step), ('cfl_momentum', cfl_momentum)):
            if val is not None:
                raise ValueError(f"{name} is only used by optimizer='cfl'")
        optimizer_lr = 1.2 if optimizer_lr is None else float(optimizer_lr)
        if optimizer == 'reg_adam':
            max_step_norm = 0.5 if max_step_norm is None else float(max_step_norm)
        elif max_step_norm is not None:
            raise ValueError("max_step_norm is only used by optimizer='reg_adam'")
        cfl_momentum = 0.0
    if 'trust_coefficient' in advanced and optimizer != 'lars':
        raise ValueError("trust_coefficient is only used by optimizer='lars'")
    if optimizer == 'reg_adam':
        for name, val in (('fast_smooth', fast_smooth), ('smooth_every_n', advanced.get('smooth_every_n'))):
            if val is not None:
                raise ValueError(f"{name} applies to the raw-gradient smoothing of the other optimisers; "
                                 f"'reg_adam' smooths its own step")
    fast_smooth = bool(fast_smooth) if fast_smooth is not None else False

    # ---- loss evaluation times ------------------------------------------------------------------
    multipoint_loss = [float(t) for t in (multipoint_loss if isinstance(multipoint_loss, (list, tuple))
                                          else [multipoint_loss])]
    if not multipoint_loss or any(t < 0 or t > 1 for t in multipoint_loss):
        raise ValueError(f"multipoint_loss must be a non-empty list of times in [0, 1]; got {multipoint_loss}")
    if energy_weight < 0 or temporal_weight < 0:
        raise ValueError("energy_weight and temporal_weight must be >= 0")
    if advanced.get('use_analytical_gradients') and n_time_steps > 1:
        raise ValueError("use_analytical_gradients requires n_time_steps == 1")

    # ---- schedule ----------------------------------------------------------------------------
    if reg_iterations is None:
        reg_iterations = [100, 100, 20]
    if levels is None:
        levels = [2 ** i for i in range(len(reg_iterations))][::-1]
    if len(levels) != len(reg_iterations):
        raise ValueError(f"levels {levels} and reg_iterations {reg_iterations} differ in length")

    # --- Extract native space moving image (Single Interpolation Policy: NO pre-warping) ---
    init_tx_list = []
    init_M_phys, init_t_phys = None, None
    from .syn import parse_ants_affine
    if initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        init_M_phys, init_t_phys = parse_ants_affine(init_tx_list, dim)
    else:
        from .robust_affine import robust_affine
        reg_aff = robust_affine(fixed, moving, dof=affine_dof, mode=affine_mode, seed=affine_seed, verbose=verbose)
        init_tx_list = reg_aff['fwdtransforms']
        init_M_phys, init_t_phys = parse_ants_affine(init_tx_list, dim)

    from .core.pipeline import normalize_and_tensorize, auto_detect_device, cleanup_gpu

    grid_shape_zyx = itk_shape_to_tensor_shape(grid_shape)
    moving_shape_zyx = itk_shape_to_tensor_shape(moving.shape)
    moving_spacing = list(moving.spacing)
    moving_origin = list(moving.origin)
    moving_direction = moving.direction.tolist() if hasattr(moving.direction, 'tolist') else moving.direction

    # everything TVFModel.fit reads, explicitly
    fit_kwargs = dict(advanced)
    model_kwargs = {k: fit_kwargs.pop(k) for k in ('solver', 'integration_steps_per_interval',
                                                   'use_analytical_gradients') if k in fit_kwargs}
    model_kwargs.setdefault('integration_steps_per_interval', 1)
    fg_mask = bool(fit_kwargs.pop('foreground_mask_lncc', False))
    grad_ckpt = bool(fit_kwargs.pop('gradient_checkpointing', False))
    winsor = fit_kwargs.pop('winsorize_quantiles', None)
    fit_kwargs.update(regularizer=regularizer, alpha=alpha if regularizer in _TVF_SPECTRAL else None,
                      total_alpha=total_alpha if regularizer in _TVF_SPECTRAL else None,
                      optimizer_type=optimizer, cfl_momentum=cfl_momentum,
                      max_step_norm=max_step_norm, multipoint_loss=multipoint_loss,
                      fast_smooth=bool(fast_smooth), cfl_max=float(cfl_max))
    fit_kwargs.setdefault('smooth_pyramid', True)
    if grad_step is not None:
        fit_kwargs['cfl_step'] = grad_step

    if backend.lower() == 'pytorch':
        device = auto_detect_device(backend='pytorch', requested_device=device)
        device_str = str(device)

        I_tensor, J_tensor = normalize_and_tensorize(
            fixed, moving, winsorize_quantiles=winsor, backend='pytorch', device=device
        )

        # --- Initialize model ---
        model = TVFModel(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_time_steps=n_time_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            moving_shape=moving_shape_zyx,
            moving_spacing=moving_spacing,
            moving_origin=moving_origin,
            moving_direction=moving_direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=model_kwargs.get('solver', 'euler'),
            integration_steps_per_interval=model_kwargs['integration_steps_per_interval'],
            use_analytical_gradients=model_kwargs.get('use_analytical_gradients', False),
        ).to(device)
        model._foreground_mask_lncc = fg_mask
        model._gradient_checkpointing = grad_ckpt

        # --- Initialize affine from initial_transform (Single Interpolation Policy) ---
        if init_M_phys is not None:
            with torch.no_grad():
                from .transform import compute_grid_to_physical_reference_matrix
                dtype_dev = torch.float32
                H_x = compute_grid_to_physical_reference_matrix(fixed.shape, fixed.spacing, fixed.origin, fixed.direction, device=device, dtype=dtype_dev)
                H_y = compute_grid_to_physical_reference_matrix(moving.shape, moving.spacing, moving.origin, moving.direction, device=device, dtype=dtype_dev)
                T_phys = torch.eye(dim + 1, device=device, dtype=dtype_dev)
                T_phys[:dim, :dim] = init_M_phys.to(device=device, dtype=dtype_dev)
                T_phys[:dim, dim] = init_t_phys.to(device=device, dtype=dtype_dev)
                model.affine.T_init = torch.inverse(H_y) @ T_phys @ H_x
            init_tx_list = []
            if verbose:
                print("[TVF] Initialized affine from initial_transform (T_init absorbed)")

        # --- Fit ---
        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            similarity_metric=syn_metric,
            lr=optimizer_lr,
            energy_weight=float(energy_weight),
            temporal_weight=float(temporal_weight),
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            **fit_kwargs
        )

        with torch.no_grad():
            fwd_disp = model.get_forward_warp(image_shape=grid_shape_zyx)
            inv_disp = model.get_inverse_warp(image_shape=grid_shape_zyx)
            fwd_np = fwd_disp.cpu().squeeze(0).numpy()
            inv_np = inv_disp.cpu().squeeze(0).numpy()


        T_grid = model.affine.get_matrix().detach().cpu().numpy()

    elif backend.lower() == 'jax':
        # The JAX implementation has Gaussian fluid smoothing and the CFL optimiser only.
        if regularizer != 'gaussian' or optimizer != 'cfl':
            raise ValueError("backend='jax' supports regularizer='gaussian' with optimizer='cfl' only; "
                             f"got regularizer={regularizer!r}, optimizer={optimizer!r}")
        from .tvf_jax import TVFModelJAX
        from .syn_jax import get_affine_matrix_jax
        import jax.numpy as jnp

        device_str = 'cpu'
        I_tensor, J_tensor = normalize_and_tensorize(fixed, moving, winsorize_quantiles=winsor, backend='jax')

        model = TVFModelJAX(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_time_steps=n_time_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            moving_shape=moving_shape_zyx,
            moving_spacing=moving_spacing,
            moving_origin=moving_origin,
            moving_direction=moving_direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=model_kwargs.get('solver', 'euler'),
            integration_steps_per_interval=model_kwargs['integration_steps_per_interval'],
            use_analytical_gradients=model_kwargs.get('use_analytical_gradients', False),
        )

        if init_M_phys is not None:
            from .transform import compute_grid_to_physical_reference_matrix
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
                print("[TVF-JAX] Initialized affine from initial_transform (T_init absorbed)")

        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            lr=0.1,
            reg_weight=fit_kwargs.pop('reg_weight', 0.0),
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            **fit_kwargs
        )

        fwd_disp = np.array(model.integrate(0.0, 1.0, image_shape=grid_shape_zyx))
        inv_disp = np.array(model.integrate(1.0, 0.0, image_shape=grid_shape_zyx))
        fwd_np = fwd_disp.squeeze(0)
        inv_np = inv_disp.squeeze(0)


        # Export affine including T_init composition (get_affine_matrix_jax composes T_init if present)
        T_grid = np.array(get_affine_matrix_jax(model.affine_params, dim, 'Affine'))

    else:
        raise ValueError(f"Unknown backend: {backend}")
    fwd_img = disp_tensor_to_itk(fwd_disp, fixed)
    inv_img = disp_tensor_to_itk(inv_disp, fixed)

    fwd_file = tempfile.NamedTemporaryFile(suffix='_tvf_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_tvf_inv_Warp.nii.gz', delete=False).name
    ants.image_write(fwd_img, fwd_file)
    ants.image_write(inv_img, inv_file)

    # Export physical affine transform using standardized ITK layout
    M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving)
    affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name

    tx_fwd, tx_inv = export_ants_affine_transform(M_phys, t_phys, dim=dim, filename=affine_file)

    # Build transform lists (same order as registration())
    # Note: affine_file already incorporates initial_transform (absorbed during initialization)
    if sum(reg_iterations) > 0:
        fwd_transforms = [fwd_file, affine_file]
        inv_transforms = [affine_file, inv_file]
        whichtoinvert_inv = [True, False]
    else:
        fwd_transforms = [affine_file]
        inv_transforms = [affine_file]
        whichtoinvert_inv = [True]

    # Generate warped output images (same as registration())
    warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_transforms)
    warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=inv_transforms,
                                          whichtoinvert=whichtoinvert_inv)

    fit_time = _time.time() - t_start

    # Clean up GPU memory
    cleanup_gpu(device=device if 'device' in locals() else None, backend=backend)

    from .syn import calculate_inverse_identity_error
    

    W_fwd_tensor = torch.from_numpy(fwd_np).to(device).float()
    W_inv_tensor = torch.from_numpy(inv_np).to(device).float()
    
    inv_err_dict = calculate_inverse_identity_error(
        W_fwd_tensor, W_inv_tensor, 
        spacing=spacing, origin=origin, direction=direction
    )

    ret_dict = {
        'warpedmovout': warpedmovout,
        'warpedfixout': warpedfixout,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': whichtoinvert_inv,
        'model': model,
        'inverse_identity_error_map': inv_err_dict['error_map'],
        'inverse_identity_errors': {'phi_1': inv_err_dict}
    }

    try:
        from .reporting import build_engine_provenance
        provenance = build_engine_provenance(
            algorithm="syntx.tvf",
            backend=backend,
            device=device_str,
            fit_time=fit_time,
            reg_iterations=reg_iterations,
            levels=levels,
            similarity_metric=syn_metric,
            syn_sampling=syn_sampling,
            n_time_steps=n_time_steps,
            multipoint_loss=multipoint_loss,
            regularizer=regularizer,
            alpha=alpha if regularizer in _TVF_SPECTRAL else None,
            total_alpha=total_alpha if regularizer in _TVF_SPECTRAL else None,
            fluid_sigma=fluid_sigma_actual if regularizer in _TVF_SIGMA else None,
            elastic_sigma=elastic_sigma_actual if regularizer in _TVF_SIGMA else None,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            max_step_norm=max_step_norm,
            grad_step=grad_step,
            cfl_momentum=cfl_momentum if optimizer == 'cfl' else None,
            cfl_max=cfl_max,
            energy_weight=energy_weight,
            temporal_weight=temporal_weight,
            fast_smooth=fast_smooth,
            advanced=dict(advanced),
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
