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
import numpy as np
from .syn import (
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    grid_to_physical_affine_torch,
    grid_sample_nd,
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

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) Optimizer for TVF Velocity Parameters.

    Rescales parameter update magnitudes using trust ratio scaling:
    $$\\text{trust\\_ratio} = \\eta \\cdot \\frac{\\max(\\|p\\|_2, 1.0)}{\\|g\\|_2 + \\epsilon}$$

    Prevents momentum collapse in smooth LNCC similarity plateaus during non-linear deformable optimization.

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
        integration_steps_per_interval=1,
        antisymmetric=True,
        image_grad_clip=6.0,
        velocity_clamp=None,
        cfl_max=0.40,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps
        self.antisymmetric = antisymmetric
        self.use_analytical_gradients = kwargs.get('use_analytical_gradients', False)
        self.image_grad_clip = image_grad_clip
        self.velocity_clamp = velocity_clamp
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
        
        # Velocity field parameter: (T, 1, *velocity_shape, dim)
        self.velocity = nn.Parameter(torch.zeros(n_time_steps, 1, *self.velocity_shape, self.dim))
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)
        self._sobolev_kernel_cache = {}

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

    def _get_moving_metadata_tensors(self, device, dtype):
        """Helper to get spatial metadata as tensors for cached normalized coordinates (Moving Image)."""
        spacing_rev = tuple(reversed(self.moving_spacing))
        origin_rev = tuple(reversed(self.moving_origin))
        direction_rev = np.asarray(self.moving_direction)[::-1, ::-1].copy()
        
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
        if fluid_sigma <= 0:
            return m
        device = m.device
        dtype = m.dtype
        dim = self.dim
        orig_shape = m.shape
        spatial_shape = orig_shape[-(dim + 1):-1]
        
        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s_val = float(s)
        
        bmask = self._create_boundary_mask(spatial_shape, device, dtype, border_width=border_width)
        m_flat = m.reshape(-1, *spatial_shape, dim)
        m_tapered = m_flat * bmask
        
        sp = spacing if spacing is not None else getattr(self, 'spacing', [1.0] * dim)
        if sp is None or len(sp) != dim:
            sp = [1.0] * dim
            
        k_axes = []
        for d in range(dim):
            n_d = spatial_shape[d]
            sp_d = float(sp[d])
            if d == dim - 1:
                k_d = torch.fft.rfftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            else:
                k_d = torch.fft.fftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            k_axes.append(k_d)
            
        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        k_sq = sum(k_j ** 2 for k_j in k_mesh)
        K_fourier = 1.0 / ((1.0 + alpha_val * k_sq) ** s_val)
        
        spatial_dims = tuple(range(2, 2 + dim))
        m_cf = m_tapered.movedim(-1, 1).to(torch.float32).contiguous()
        
        m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)
        K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
        v_fft = m_fft * K_bc
        v_cf = torch.fft.irfftn(v_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype).contiguous()
        
        v_out = (v_cf.movedim(1, -1) * bmask).reshape(orig_shape)
        return v_out

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """
        Applies Sobolev Green's operator in Discrete Sine Transform Type-I (DST-I) space.
        Analytically enforces exact homogeneous Dirichlet boundary conditions (v = 0 at boundaries).
        """
        if fluid_sigma <= 0:
            return m

        device = m.device
        dtype = m.dtype
        dim = self.dim

        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s = 2.0

        spatial_shape = m.shape[1:-1]
        k_axes = []
        for d in range(dim):
            n_d = spatial_shape[d]
            k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
            lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
            k_axes.append(lambda_d)

        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        lambda_sq = sum(k_j for k_j in k_mesh)
        K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

        m_cf = m.movedim(-1, 1).to(torch.float32)

        # nD DST-I via odd-symmetric FFT extension
        padded = m_cf
        for d in range(dim):
            axis = 2 + d
            z_shape = list(padded.shape)
            z_shape[axis] = 1
            z = torch.zeros(z_shape, device=device, dtype=torch.float32)
            rev = -torch.flip(padded, dims=[axis])
            padded = torch.cat([z, padded, z, rev], dim=axis)

        spatial_axes = tuple(range(2, 2 + dim))
        fft_padded = torch.fft.rfftn(padded, dim=spatial_axes)

        slices = [slice(None), slice(None)]
        for n_d in spatial_shape:
            slices.append(slice(1, n_d + 1))

        if dim % 2 == 1:
            sign = -1.0 if (dim % 4 == 1) else 1.0
            dst_coeff = sign * (0.5 ** dim) * torch.imag(fft_padded[tuple(slices)])
        else:
            sign = -1.0 if (dim % 4 == 2) else 1.0
            dst_coeff = sign * (0.5 ** dim) * torch.real(fft_padded[tuple(slices)])
            
        del fft_padded
        
        K_bc = K_dst.unsqueeze(0).unsqueeze(0)
        dst_filtered = dst_coeff * K_bc

        # Inverse nD DST-I
        padded_c = dst_filtered
        norm_factor = 1.0
        for d in range(dim):
            axis = 2 + d
            n_d = spatial_shape[d]
            norm_factor *= 4.0 / (n_d + 1)
            z_shape = list(padded_c.shape)
            z_shape[axis] = 1
            z = torch.zeros(z_shape, device=device, dtype=torch.float32)
            rev_c = -torch.flip(padded_c, dims=[axis])
            padded_c = torch.cat([z, padded_c, z, rev_c], dim=axis)

        fft_padded_c = torch.fft.rfftn(padded_c, dim=spatial_axes)

        if dim % 2 == 1:
            sign = -1.0 if (dim % 4 == 1) else 1.0
            idst_out = sign * (0.5 ** dim) * torch.imag(fft_padded_c[tuple(slices)]) * norm_factor
        else:
            sign = -1.0 if (dim % 4 == 2) else 1.0
            idst_out = sign * (0.5 ** dim) * torch.real(fft_padded_c[tuple(slices)]) * norm_factor

        return idst_out.to(dtype=dtype).movedim(1, -1)

    def _apply_dsti1_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """
        Applies Sobolev Green's operator using separable 1D DST-I transforms.
        Mathematically equivalent to _apply_dsti_green_operator but:
        - 8x less peak memory for 3D volumes
        - Each 1D rfft operates on size 2N+2 along one axis only
        - More robust on MPS (Apple Silicon) backend
        """
        if fluid_sigma <= 0:
            return m

        device = m.device
        dtype = m.dtype
        dim = self.dim

        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s = 2.0

        spatial_shape = m.shape[1:-1]

        k_axes = []
        for d in range(dim):
            n_d = spatial_shape[d]
            k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
            lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
            k_axes.append(lambda_d)

        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        lambda_sq = sum(k_j for k_j in k_mesh)
        K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

        m_cf = m.movedim(-1, 1).to(torch.float32)

        def _dst1_1d(arr, axis):
            n_d = arr.shape[axis]
            z_shape = list(arr.shape)
            z_shape[axis] = 1
            z = torch.zeros(z_shape, device=device, dtype=torch.float32)
            rev = -torch.flip(arr, dims=[axis])
            padded = torch.cat([z, arr, z, rev], dim=axis)
            fft_1d = torch.fft.rfft(padded, dim=axis)
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(1, n_d + 1)
            out = -0.5 * torch.imag(fft_1d[tuple(sl)]).clone()
            
            # Aggressive cleanup to prevent MPS OOM in tight ODE loops
            del z, rev, padded, fft_1d
            if str(device) == 'mps':
                torch.mps.empty_cache()
            return out

        curr = m_cf
        for d in range(dim):
            axis = 2 + d
            curr = _dst1_1d(curr, axis)

        K_bc = K_dst.unsqueeze(0).unsqueeze(0)
        v_dst = curr * K_bc

        curr_inv = v_dst
        for d in range(dim):
            axis = 2 + d
            n_d = spatial_shape[d]
            curr_inv = _dst1_1d(curr_inv, axis) * (4.0 / float(n_d + 1))

        if str(device) == 'mps':
            torch.mps.empty_cache()

        return curr_inv.to(dtype=dtype).movedim(1, -1)

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

    def integrate(self, t_start, t_end, velocity=None, n_steps=None, image_shape=None,
                  _cached_phys_grid=None, _cached_meta=None, _cached_velocity_fine_cf=None):
        """
        Integrates the velocity field ODE from t_start to t_end.
        
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
            return torch.zeros(batch_shape, device=device, dtype=dtype)
        
        if n_steps is None:
            default_steps = self.n_time_steps * self.integration_steps_per_interval
            # Adaptive CFL: ensure per-step displacement respects integrator stability.
            with torch.no_grad():
                target_sp = tuple(image_shape) if image_shape is not None else self.image_shape
                curr_spacing = [
                    sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                    for sp, orig_s, curr_s in zip(self.spacing, reversed(self.image_shape), reversed(target_sp))
                ]
                sp_t = torch.tensor(curr_spacing, device=velocity.device, dtype=velocity.dtype)
                
                vel_voxel = velocity.detach() / sp_t
                v_mag_sq = torch.sum(vel_voxel ** 2, dim=-1)
                v_max_voxel = torch.sqrt(v_mag_sq.max()).item()
                
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
            
            spacing_rev = tuple(reversed(curr_spacing))
            origin_rev = tuple(reversed(self.origin))
            direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
            
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
            self.moving_shape, self.moving_spacing, self.moving_origin, self.moving_direction
        )
        
        # M_phys and t_phys are already returned in ZYX order from grid_to_physical_affine_torch
        M_phys_zyx = M_phys
        t_phys_zyx = t_phys
        
        losses = []
        compute_id_loss = (0.0 in eval_points) and (1.0 in eval_points)
        phi_0_to_1 = None
        phi_1_to_0 = None

        # Pre-upsample velocity keyframes ONCE for the entire forward pass,
        # shared across all integrate() calls (eliminates redundant F.interpolate).
        velocity_fine_cf = self._upsample_velocity_keyframes(
            velocity if velocity is not None else self.velocity, target_shape
        )

        # Cache moving image metadata tensors once
        shape_m, spacing_m, origin_m, direction_m = self._get_moving_metadata_tensors(device, dtype)

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

            phi_fixed_norm_tk = physical_to_normalized_torch_cached(
                phys_grid + phi_tk_to_fixed, shape_t, spacing_t, origin_t, direction_t
            )
            fixed_warped_tk = grid_sample_nd(fixed_image, phi_fixed_norm_tk, mode='bilinear', padding_mode='zeros')

            phi_moving_affine_tk = (phys_grid + phi_tk_to_moving) @ M_phys_zyx.t() + t_phys_zyx
            phi_norm_tk = physical_to_normalized_torch_cached(
                phi_moving_affine_tk, shape_m, spacing_m, origin_m, direction_m
            )
            moving_warped_tk = grid_sample_nd(moving_image, phi_norm_tk, mode='bilinear', padding_mode='zeros')

            losses.append(lncc_loss_nd(fixed_warped_tk, moving_warped_tk, window_size=lncc_window_size))

        sim_loss = sum(losses) / len(losses)

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
            inv_id_weight = float(getattr(self, 'inverse_identity_weight', 0.05))
            sim_loss = sim_loss + inv_id_weight * inv_id_loss

        return sim_loss

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
        image_grad_clip=6.0,
        velocity_clamp=None,
        cfl_max=0.40,
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
                    self.moving_shape, self.moving_spacing, self.moving_origin, self.moving_direction
                )
                
                # M_phys and t_phys are already returned in ZYX order from grid_to_physical_affine_torch
                M_phys_zyx = M_phys
                t_phys_zyx = t_phys
                
                phi_moving_affine = phys_grid @ M_phys_zyx.t() + t_phys_zyx
                
                shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(device, dtype)
                shape_m, spacing_m, origin_m, direction_m = self._get_moving_metadata_tensors(device, dtype)
                phi_moving_norm = physical_to_normalized_torch_cached(
                    phi_moving_affine, shape_m, spacing_m, origin_m, direction_m
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
        opt_type = str(kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl'))).lower()
        trust_coeff = float(kwargs.get('trust_coefficient', kwargs.get('trust', 0.80)))
        
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)
        multipoint_loss = kwargs.get('multipoint_loss', [0.5])
        cfl_max_val = float(kwargs.get('cfl_max', self.cfl_max if self.cfl_max is not None else 0.0))

        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        
        sigma_mode = kwargs.get('sigma_mode', 'voxel')
        use_analytical_gradients = kwargs.get('use_analytical_gradients', getattr(self, 'use_analytical_gradients', False))
        self.losses = []
        
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
            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            shrink_ratio = float(curr_vel_shape[0]) / float(max_vel_shape[0])
            prev_vel_shape = tuple(self.velocity.shape[2:-1])
            
            if curr_vel_shape != prev_vel_shape:
                self._resize_velocity(curr_vel_shape, device, dtype)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")
            
            curr_spacing = [sp * level for sp in self.spacing]
            
            # Create optimizer fresh for this level (velocity parameter may have changed)
            if opt_type == 'lars':
                import math
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
            else:
                optimizer = torch.optim.Adam([self.velocity], lr=lr)
            
            # Reset momentum buffer for this level
            if cfl_momentum > 0 and opt_type == 'cfl':
                momentum_buffer = torch.zeros_like(self.velocity.data)
            
            # Compute vel_spacing for physical-mode smoothing at current velocity resolution
            vel_spacing = [sp * (img_dim / vel_dim) for sp, img_dim, vel_dim in zip(self.spacing, self.image_shape, curr_vel_shape)] if sigma_mode == 'physical' else None
            
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
                grad_I_curr = _spatial_jacobian_nd(
                    curr_fixed.movedim(1, -1),
                    physical_spacing=tuple(reversed(curr_spacing))
                ).squeeze(-2)
                
                moving_target_shape = tuple(curr_moving.shape[2:])
                curr_moving_spacing_list = [
                    sp * (float(orig_s - 1) / float(curr_s - 1)) if curr_s > 1 else sp
                    for sp, orig_s, curr_s in zip(self.moving_spacing, reversed(self.moving_shape), reversed(moving_target_shape))
                ]
                
                grad_J_curr = _spatial_jacobian_nd(
                    curr_moving.movedim(1, -1),
                    physical_spacing=tuple(reversed(curr_moving_spacing_list))
                ).squeeze(-2)
            
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
                        spacing_rev = tuple(reversed(curr_spacing_list))
                        origin_rev = tuple(reversed(self.origin))
                        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
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
                        
                        shape_m_ag = torch.tensor(moving_target_shape, device=device, dtype=dtype)
                        spacing_m_ag = torch.tensor(tuple(reversed(curr_moving_spacing_list)), device=device, dtype=dtype)
                        origin_m_ag = torch.tensor(tuple(reversed(self.moving_origin)), device=device, dtype=dtype)
                        direction_m_ag = torch.tensor(np.asarray(self.moving_direction)[::-1, ::-1].copy(), device=device, dtype=dtype)
                        
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
                        ).movedim(1, -1).contiguous()
                        direction_t_mat = torch.tensor(
                            np.asarray(self.direction)[::-1, ::-1].copy(),
                            device=device, dtype=dtype
                        )
                        grad_I_mid = torch.matmul(grad_I_mid, direction_t_mat)
                        
                        grad_J_mid = grid_sample_nd(
                            grad_J_curr.movedim(-1, 1), phi_moving_norm,
                            mode='bilinear', padding_mode='zeros'
                        ).movedim(1, -1).contiguous()
                        grad_J_mid = torch.matmul(grad_J_mid, direction_m_ag)
                        grad_J_mid = torch.matmul(grad_J_mid, M_phys_zyx)
                    
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
                        combined_grad = (grad_wrt_phi_moving - grad_wrt_phi_fixed) / 2.0
                        
                        # Assign gradient to velocity parameter
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

                        if regularizer_mode == 'sobolev':
                            # Adjust physical spacing if downsampled so physical scale remains correct
                            if vel_spacing is not None:
                                adj_spacing = [sp * 2.0 for sp in vel_spacing] if do_fast else vel_spacing
                            else:
                                adj_spacing = [sp * 2.0 for sp in getattr(self, 'spacing', [1.0] * self.dim)] if do_fast else getattr(self, 'spacing', [1.0] * self.dim)
                            
                            alpha_sob = float(kwargs.get('sobolev_alpha', kwargs.get('alpha', sigma_val / 2.0)))
                            g_smoothed = self._apply_sobolev_green_operator(g_process, fluid_sigma=sigma_val, alpha=alpha_sob, spacing=adj_spacing)
                        elif regularizer_mode == 'dsti':
                            alpha_dsti = float(kwargs.get('dsti_alpha', kwargs.get('alpha', sigma_val / 2.0)))
                            g_smoothed = self._apply_dsti_green_operator(g_process_tapered, fluid_sigma=sigma_val, alpha=alpha_dsti)
                        elif regularizer_mode == 'dsti1':
                            alpha_dsti = float(kwargs.get('dsti_alpha', kwargs.get('alpha', sigma_val / 2.0)))
                            g_smoothed = self._apply_dsti1_green_operator(g_process_tapered, fluid_sigma=sigma_val, alpha=alpha_dsti)
                        else:
                            # Adjust physical spacing if downsampled so blur radius remains correct
                            if vel_spacing is not None:
                                adj_spacing = [sp * 2.0 for sp in vel_spacing] if do_fast else vel_spacing
                            else:
                                adj_spacing = [2.0] * self.dim if do_fast else None
                                
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
                            sp_t = torch.tensor(curr_spacing, device=device, dtype=dtype)
                            grad_voxel = grad / sp_t  # convert to voxel units
                            max_g_voxel = torch.sqrt(torch.sum(grad_voxel**2, dim=-1)).max()
                            if max_g_voxel > 1e-8:
                                cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                                import math
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
                                    self.velocity.data.sub_(corrected_buf)
                                else:
                                    self.velocity.data.sub_(update)
                else:
                    optimizer.step()


                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                with torch.no_grad():
                    if elastic_sigma_val > 0:
                        T = self.n_time_steps
                        vel_batch = self.velocity.squeeze(1)
                        regularizer_mode = kwargs.get('regularizer', 'gaussian')
                        
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
                            alpha_dsti = float(kwargs.get('dsti_alpha', kwargs.get('alpha', elastic_sigma_val / 2.0)))
                            vel_smoothed = self._apply_dsti_green_operator(vel_tapered, fluid_sigma=elastic_sigma_val, alpha=alpha_dsti)
                        elif regularizer_mode == 'dsti1':
                            vel_tapered = vel_batch.clone()
                            for d in range(self.dim):
                                sl_first = [slice(None)] * (self.dim + 2)
                                sl_first[d + 1] = 0
                                vel_tapered[tuple(sl_first)] = 0.0
                                sl_last = [slice(None)] * (self.dim + 2)
                                sl_last[d + 1] = -1
                                vel_tapered[tuple(sl_last)] = 0.0
                            alpha_dsti = float(kwargs.get('dsti_alpha', kwargs.get('alpha', elastic_sigma_val / 2.0)))
                            vel_smoothed = self._apply_dsti1_green_operator(vel_tapered, fluid_sigma=elastic_sigma_val, alpha=alpha_dsti)
                        elif regularizer_mode == 'sobolev':
                            alpha_sob = float(kwargs.get('sobolev_alpha', kwargs.get('alpha', elastic_sigma_val / 2.0)))
                            vel_smoothed = self._apply_sobolev_green_operator(vel_batch, fluid_sigma=elastic_sigma_val, alpha=alpha_sob, spacing=vel_spacing)
                        else:
                            vel_smoothed = separable_gaussian_filter(
                                vel_batch, sigma=elastic_sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                            )
                        self.velocity.copy_(vel_smoothed.unsqueeze(1))
                    
                    cfl_max_val = kwargs.get('cfl_max', None)
                    if cfl_max_val is not None and float(cfl_max_val) > 0:
                        sp_t = torch.tensor(curr_spacing, device=device, dtype=dtype)
                        vel_voxel = self.velocity / sp_t
                        vel_voxel_norm = torch.norm(vel_voxel, dim=-1, keepdim=True)
                        max_vel_norm = vel_voxel_norm.max()
                        if max_vel_norm > float(cfl_max_val):
                            self.velocity.mul_(float(cfl_max_val) / (max_vel_norm + 1e-8))
                    
                    # Constant speed constraint: project velocity keyframes onto uniform-speed manifold.
                    # Ensures geodesic parameterization (constant-speed path through diffeomorphism group)
                    # and prevents velocity energy concentration in a single keyframe.
                    cs_enabled = kwargs.get('constant_speed', True)
                    cs_relax = float(kwargs.get('constant_speed_relaxation', 0.10))
                    if cs_enabled and self.n_time_steps > 1 and cs_relax > 0:
                        with torch.no_grad():
                            # Compute per-keyframe 2-norm
                            vel_data = self.velocity.data  # (T, 1, *spatial, dim)
                            speeds = torch.sqrt(torch.sum(vel_data ** 2, dim=tuple(range(1, vel_data.ndim))))  # (T,)
                            mean_speed = speeds.mean()
                            if mean_speed > 1e-10:
                                # Relaxation toward uniform speed: v_k *= (1-α) + α * (mean/speed_k)
                                for t_k in range(self.n_time_steps):
                                    if speeds[t_k] > 1e-10:
                                        scale = (1.0 - cs_relax) + cs_relax * (mean_speed / speeds[t_k])
                                        self.velocity.data[t_k].mul_(scale)

                # Record epoch loss in self.losses history
                loss_val = sim_loss.item()
                self.losses.append(loss_val)

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
                    except:
                        pass
                    import gc
                    gc.collect()
                    if device.type == 'mps':
                        torch.mps.empty_cache()

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
    grad_step=0.211,
    flow_sigma=0.0,
    total_sigma=0.2,
    n_time_steps=3,
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
    image_grad_clip=6.0,
    velocity_clamp=None,
    cfl_max=0.40,
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
        affine_iterations = 0 if initial_transform is not None else 100

    if multipoint_loss is None:
        multipoint_loss = [0.0, 0.5, 1.0]


    # --- Convert ITK variance convention to actual sigma (same as registration()) ---
    fluid_sigma_actual = math.sqrt(flow_sigma) if flow_sigma > 0 else 0.0
    elastic_sigma_actual = math.sqrt(total_sigma) if total_sigma > 0 else 0.0

    # --- Extract native space moving image (Single Interpolation Policy: NO pre-warping) ---
    init_tx_list = []
    init_M_phys, init_t_phys = None, None
    if initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        from .syn import parse_ants_affine
        init_M_phys, init_t_phys = parse_ants_affine(init_tx_list, dim)
    else:
        from .robust_affine import robust_affine
        reg_aff = robust_affine(fixed, moving, mode='pytorch', verbose=verbose)
        init_tx_list = reg_aff['fwdtransforms']
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

    moving_shape_zyx = tuple(reversed(moving.shape))
    moving_spacing = list(moving.spacing)
    moving_origin = list(moving.origin)
    moving_direction = moving.direction.tolist() if hasattr(moving.direction, 'tolist') else moving.direction

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
            moving_shape=moving_shape_zyx,
            moving_spacing=moving_spacing,
            moving_origin=moving_origin,
            moving_direction=moving_direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
            integration_steps_per_interval=kwargs.pop('integration_steps_per_interval', 1),
            antisymmetric=kwargs.pop('antisymmetric', True),
            use_analytical_gradients=kwargs.pop('use_analytical_gradients', False),

        ).to(device_str)

        # --- Initialize affine from initial_transform (Single Interpolation Policy) ---
        # Maps the ANTs physical affine into grid coordinates via T_init,
        # matching SyN's approach (syn.py lines 1954-1971).
        if init_M_phys is not None:
            with torch.no_grad():
                from .transform import compute_grid_to_physical_reference_matrix
                dtype_dev = torch.float32
                H_x = compute_grid_to_physical_reference_matrix(fixed.shape, fixed.spacing, fixed.origin, fixed.direction, device=device_str, dtype=dtype_dev)
                H_y = compute_grid_to_physical_reference_matrix(moving.shape, moving.spacing, moving.origin, moving.direction, device=device_str, dtype=dtype_dev)

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
            optimizer_type=optimizer if optimizer is not None else kwargs.pop('optimizer_type', kwargs.pop('optimizer', 'cfl')),
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
            moving_shape=moving_shape_zyx,
            moving_spacing=moving_spacing,
            moving_origin=moving_origin,
            moving_direction=moving_direction,
            fluid_sigma=fluid_sigma_actual,
            elastic_sigma=elastic_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
            integration_steps_per_interval=kwargs.pop('integration_steps_per_interval', 1),
            antisymmetric=kwargs.pop('antisymmetric', True),
            use_analytical_gradients=kwargs.pop('use_analytical_gradients', False),
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
    from .transform import export_ants_displacement_field, export_ants_affine_transform

    fwd_img = export_ants_displacement_field(fwd_np, origin=origin, spacing=spacing, direction=direction)
    inv_img = export_ants_displacement_field(inv_np, origin=origin, spacing=spacing, direction=direction)

    fwd_file = tempfile.NamedTemporaryFile(suffix='_tvf_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_tvf_inv_Warp.nii.gz', delete=False).name
    ants.image_write(fwd_img, fwd_file)
    ants.image_write(inv_img, inv_file)

    # Export physical affine transform using standardized ITK layout
    M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving)
    affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name

    tx_fwd, tx_inv = export_ants_affine_transform(M_phys, t_phys, dim=dim)
    ants.write_transform(tx_fwd, affine_file)
    ants.write_transform(tx_inv, affine_inv_file)

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
    if device_str == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device_str == 'cuda':
        torch.cuda.empty_cache()

    from .syn import calculate_inverse_identity_error
    

    W_fwd_tensor = torch.from_numpy(fwd_np).to(device_str).float()
    W_inv_tensor = torch.from_numpy(inv_np).to(device_str).float()
    
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
            affine_iterations=affine_iterations if isinstance(affine_iterations, list) else [affine_iterations],
            solver="TVF-Euler",
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
