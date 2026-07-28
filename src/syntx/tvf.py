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
    grid_sample_nd
)

def normalize_tensor(tensor):
    """Min-max normalizes tensor to [0, 1] range."""
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max > t_min:
        return (tensor - t_min) / (t_max - t_min)
    return tensor

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

def extract_image_metadata(img):
    """
    Extract physical space metadata (shape, spacing, origin, direction)
    from an ANTsImage, PyTorch Tensor, or NumPy array.
    """
    shape, spacing, origin, direction = None, None, None, None
    if img is None:
        return shape, spacing, origin, direction
        
    if hasattr(img, 'spacing'):
        spacing = list(img.spacing)
    if hasattr(img, 'origin'):
        origin = list(img.origin)
    if hasattr(img, 'direction'):
        direction = img.direction.tolist() if hasattr(img.direction, 'tolist') else list(img.direction)
        
    if hasattr(img, 'shape'):
        shape = tuple(img.shape)
    elif hasattr(img, 'numpy'):
        shape = tuple(img.numpy().shape)
        
    return shape, spacing, origin, direction

class TVFModel(nn.Module):
    """
    Time-Varying Velocity Field (TVF) Registration Model.
    Automatically extracts physical space metadata (spacing, origin, direction)
    directly from image objects (ANTsImage, Tensors, Arrays).
    """
    def __init__(
        self,
        dim=3,
        image_shape=None,
        velocity_shape=(96, 96, 96),
        n_time_steps=4,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=2.0,
        elastic_sigma=0.05,
        transform_type='Affine',
        solver='euler',
        integration_steps_per_interval=1,
        fixed_image=None
    ):
        super().__init__()
        
        # Auto-extract physical metadata if fixed_image object is passed
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image)
        if image_shape is None:
            image_shape = f_shape if f_shape is not None else (128, 128, 128)
        if spacing is None and f_spacing is not None:
            spacing = f_spacing
        if origin is None and f_origin is not None:
            origin = f_origin
        if direction is None and f_direction is not None:
            direction = f_direction
            
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
        self.elastic_sigma = elastic_sigma
        self.solver = solver
        self.integration_steps_per_interval = integration_steps_per_interval
        
        # Velocity field parameter: (T, 1, *velocity_shape, dim)
        self.velocity = nn.Parameter(torch.zeros(n_time_steps, 1, *self.velocity_shape, self.dim))
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)

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

    def _interpolate_velocity_fine(self, t, velocity_fine_cf):
        """
        Linearly interpolate between pre-upsampled velocity keyframes.
        
        Args:
            t: Continuous time in [0, 1]
            velocity_fine_cf: List of T pre-upsampled velocity tensors (1, dim, *target_shape)
            
        Returns:
            Velocity at time t (1, dim, *target_shape)
        """
        T = len(velocity_fine_cf)
        if T == 1:
            return velocity_fine_cf[0]
            
        t_scaled = t * (T - 1)
        idx_lower = math.floor(t_scaled)
        idx_upper = math.ceil(t_scaled)
        
        if idx_lower == idx_upper:
            idx_lower = max(0, min(T - 1, idx_lower))
            return velocity_fine_cf[idx_lower]
            
        idx_lower = max(0, min(T - 1, idx_lower))
        idx_upper = max(0, min(T - 1, idx_upper))
        
        weight_upper = t_scaled - idx_lower
        weight_lower = 1.0 - weight_upper
        
        return weight_lower * velocity_fine_cf[idx_lower] + weight_upper * velocity_fine_cf[idx_upper]

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
                
                v_sampled_cf = grid_sample_nd(v_fine_cf, phi_norm, mode='bilinear')
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

    def forward(self, fixed_image, moving_image, velocity=None, affine_params=None, multipoint_loss=[0.5], lncc_window_size=5):
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
        M_phys = M_phys.to(device=device, dtype=dtype)
        t_phys = t_phys.to(device=device, dtype=dtype)

        coord_perm = list(range(self.dim - 1, -1, -1))
        perm_idx = torch.tensor(coord_perm, device=M_phys.device)
        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
        t_phys_zyx = t_phys[perm_idx]

        losses = []
        for t_k in eval_points:
            t_k = float(t_k)
            if abs(t_k - 0.0) < 1e-5:
                # Fixed Space (t=0.0: warp moving to fixed)
                phi_0_to_1 = self.integrate(0.0, 1.0, velocity=velocity, image_shape=target_shape,
                                            _cached_phys_grid=phys_grid, _cached_meta=_cached_meta)
                phi_moving_affine_end = (phys_grid + phi_0_to_1) @ M_phys_zyx.t() + t_phys_zyx
                phi_norm_end = physical_to_normalized_torch_cached(
                    phi_moving_affine_end, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = grid_sample_nd(moving_image, phi_norm_end, mode='bilinear', padding_mode='zeros')
                losses.append(lncc_loss_nd(fixed_image, moving_warped, window_size=lncc_window_size))
            elif abs(t_k - 1.0) < 1e-5:
                # Moving Space (t=1.0: warp fixed to moving)
                phi_1_to_0 = self.integrate(1.0, 0.0, velocity=velocity, image_shape=target_shape,
                                            _cached_phys_grid=phys_grid, _cached_meta=_cached_meta)
                phi_fixed_norm_end = physical_to_normalized_torch_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped = grid_sample_nd(fixed_image, phi_fixed_norm_end, mode='bilinear', padding_mode='zeros')

                phi_moving_identity = phys_grid @ M_phys_zyx.t() + t_phys_zyx
                phi_moving_identity_norm = physical_to_normalized_torch_cached(
                    phi_moving_identity, shape_t, spacing_t, origin_t, direction_t
                )
                moving_affine = grid_sample_nd(moving_image, phi_moving_identity_norm, mode='bilinear', padding_mode='zeros')
                losses.append(lncc_loss_nd(fixed_warped, moving_affine, window_size=lncc_window_size))
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
        Automatically extracts physical space metadata directly from fixed_image
        and converts ANTsImage/NumPy inputs to PyTorch tensors.
        """
        # Auto-extract physical space metadata directly from fixed_image
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image)
        if f_spacing is not None and fixed_spacing is None: self.spacing = f_spacing
        if f_origin is not None and fixed_origin is None: self.origin = f_origin
        if f_direction is not None and fixed_direction is None: self.direction = f_direction
        if f_shape is not None and self.image_shape != f_shape: self.image_shape = f_shape

        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        # Automatic conversion of ANTsImage or NumPy array to PyTorch 5D Tensor (B, C, Z, Y, X)
        if hasattr(fixed_image, 'numpy'):
            fixed_tensor = normalize_tensor(torch.from_numpy(fixed_image.numpy()).float()).unsqueeze(0).unsqueeze(0)
        elif isinstance(fixed_image, torch.Tensor):
            fixed_tensor = fixed_image
        else:
            fixed_tensor = torch.from_numpy(np.array(fixed_image)).float().unsqueeze(0).unsqueeze(0)

        if hasattr(moving_image, 'numpy'):
            moving_tensor = normalize_tensor(torch.from_numpy(moving_image.numpy()).float()).unsqueeze(0).unsqueeze(0)
        elif isinstance(moving_image, torch.Tensor):
            moving_tensor = moving_image
        else:
            moving_tensor = torch.from_numpy(np.array(moving_image)).float().unsqueeze(0).unsqueeze(0)

        device = fixed_tensor.device if isinstance(fixed_tensor, torch.Tensor) and (fixed_tensor.is_cuda or fixed_tensor.is_mps) else (
            torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        )
        dtype = torch.float32

        fixed_image = fixed_tensor.to(device=device, dtype=dtype)
        moving_image = moving_tensor.to(device=device, dtype=dtype)

        # Initial alignment via dynamic FOV & Foreground CoM evaluation (matching syntx.syn)
        fixed_spacing_t = torch.tensor(self.spacing, device=device, dtype=dtype)
        fixed_origin_t = torch.tensor(self.origin, device=device, dtype=dtype)
        fixed_direction_t = torch.tensor(self.direction, device=device, dtype=dtype)

        moving_spacing_t = torch.tensor(moving_spacing if moving_spacing is not None else self.spacing, device=device, dtype=dtype)
        moving_origin_t = torch.tensor(moving_origin if moving_origin is not None else self.origin, device=device, dtype=dtype)
        moving_direction_t = torch.tensor(moving_direction if moving_direction is not None else self.direction, device=device, dtype=dtype)

        dim = self.dim
        Nx_t = torch.tensor(fixed_image.shape[2:], device=device, dtype=dtype)
        Ny_t = torch.tensor(moving_image.shape[2:], device=device, dtype=dtype)
        
        Dx_t, Sx_t, Ox_t = fixed_direction_t, fixed_spacing_t, fixed_origin_t
        Dy_t, Sy_t, Oy_t = moving_direction_t, moving_spacing_t, moving_origin_t

        com_fixed_fov = Ox_t + Dx_t @ (Sx_t * ((Nx_t - 1) / 2.0))
        com_moving_fov = Oy_t + Dy_t @ (Sy_t * ((Ny_t - 1) / 2.0))

        # Intensity-weighted foreground CoM
        grid_idx_x = torch.stack(torch.meshgrid([torch.arange(n, device=device, dtype=dtype) for n in fixed_image.shape[2:]], indexing='ij'), dim=-1)
        grid_phys_x = Ox_t + (grid_idx_x * Sx_t) @ Dx_t.t()
        weights_x = torch.clamp(fixed_image.squeeze(0).squeeze(0), min=0.0)
        sum_w_x = torch.sum(weights_x)
        com_fixed_fg = torch.sum(grid_phys_x * weights_x.unsqueeze(-1), dim=tuple(range(dim))) / sum_w_x if sum_w_x > 0 else com_fixed_fov

        grid_idx_y = torch.stack(torch.meshgrid([torch.arange(n, device=device, dtype=dtype) for n in moving_image.shape[2:]], indexing='ij'), dim=-1)
        grid_phys_y = Oy_t + (grid_idx_y * Sy_t) @ Dy_t.t()
        weights_y = torch.clamp(moving_image.squeeze(0).squeeze(0), min=0.0)
        sum_w_y = torch.sum(weights_y)
        com_moving_fg = torch.sum(grid_phys_y * weights_y.unsqueeze(-1), dim=tuple(range(dim))) / sum_w_y if sum_w_y > 0 else com_moving_fov

        t_fov = com_moving_fov - com_fixed_fov
        t_fg = com_moving_fg - com_fixed_fg

        # Downsample for fast Mattes MI evaluation
        down_shape = [max(16, int(s // 4)) for s in self.image_shape]
        down_spacing = [(s * orig) / d for s, orig, d in zip(self.spacing, self.image_shape, down_shape)]
        I_down = F.interpolate(fixed_image, size=down_shape, mode='trilinear' if dim==3 else 'bilinear', align_corners=True)
        J_down = F.interpolate(moving_image, size=down_shape, mode='trilinear' if dim==3 else 'bilinear', align_corners=True)
        X_down = get_physical_grid_torch(down_shape, down_spacing, self.origin, self.direction, device=device, dtype=dtype)

        def eval_translation(t_candidate):
            t_candidate_zyx = torch.flip(t_candidate, dims=[-1])
            y_phys = X_down + t_candidate_zyx
            shape_mt = torch.tensor(moving_image.shape[2:], device=device, dtype=dtype)
            y_norm = physical_to_normalized_torch_cached(y_phys, shape_mt, moving_spacing_t, moving_origin_t, moving_direction_t)
            J_warped = grid_sample_nd(J_down, y_norm, padding_mode='border', align_corners=True)
            return mattes_mi_loss_nd(J_warped, I_down, num_bins=16).item()

        loss_fov = eval_translation(t_fov)
        loss_fg = eval_translation(t_fg)
        best_t = t_fov if loss_fov < loss_fg else t_fg

        H_x = torch.eye(dim + 1, device=device, dtype=dtype)
        H_x[:dim, :dim] = Dx_t @ torch.diag(Sx_t) @ torch.diag((Nx_t - 1) / 2.0)
        H_x[:dim, dim] = com_fixed_fov

        H_y = torch.eye(dim + 1, device=device, dtype=dtype)
        H_y[:dim, :dim] = Dy_t @ torch.diag(Sy_t) @ torch.diag((Ny_t - 1) / 2.0)
        H_y[:dim, dim] = com_moving_fov

        T_phys = torch.eye(dim + 1, device=device, dtype=dtype)
        T_phys[:dim, dim] = best_t

        T_init = torch.inverse(H_y) @ T_phys @ H_x
        self.affine.T_init = T_init
            
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
                M_phys = M_phys.to(device=device, dtype=dtype)
                t_phys = t_phys.to(device=device, dtype=dtype)
                
                coord_perm = list(range(self.dim - 1, -1, -1))
                perm_idx = torch.tensor(coord_perm, device=M_phys.device)
                M_phys_zyx = M_phys[perm_idx][:, perm_idx]
                t_phys_zyx = t_phys[perm_idx]
                
                phi_moving_affine = phys_grid @ M_phys_zyx.t() + t_phys_zyx
                
                shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(device, dtype)
                phi_moving_norm = physical_to_normalized_torch_cached(
                    phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = grid_sample_nd(moving_image, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                
                loss = lncc_loss_nd(fixed_image, moving_warped, window_size=2*lncc_radius+1)
                loss.backward()
                optimizer_aff.step()
                self.affine.clamp_parameters()
                
        # Optimize velocity field across pyramid levels
        if verbose: print("Optimizing TVF...")
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'adam')).lower()
        trust_coeff = kwargs.get('trust_coefficient', kwargs.get('trust', 0.05))
        
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)
        multipoint_loss = kwargs.get('multipoint_loss', [0.5])
        
        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        
        sigma_mode = kwargs.get('sigma_mode', 'voxel')

        # Compute pyramid-proportional velocity shapes for each level
        # velocity_shape is the MAX (finest) grid; coarser levels use proportionally smaller grids
        max_vel_shape = self.velocity_shape  # e.g., (96, 96, 96)
        
        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue
            
            # --- Pyramid-proportional velocity grid ---
            # Scale velocity grid inversely with pyramid level, clamped to min 8 per axis
            curr_vel_shape = tuple(max(8, v // level) for v in max_vel_shape)
            prev_vel_shape = tuple(self.velocity.shape[2:-1])
            
            if curr_vel_shape != prev_vel_shape:
                self._resize_velocity(curr_vel_shape, device, dtype)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")
            
            # Create optimizer fresh for this level (velocity parameter may have changed)
            if opt_type == 'lars':
                optimizer = LARS([self.velocity], lr=lr, trust_coefficient=trust_coeff)
            else:
                optimizer = torch.optim.Adam([self.velocity], lr=lr)
            
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
                print(f"Level {level}: {epochs} max epochs, vel_grid={list(curr_vel_shape)} (fluid_sigma={curr_fluid_sig:.3f}, elastic_sigma={curr_elastic_sig:.3f}, mode={sigma_mode})")
            
            if level > 1:
                down_shape = [max(8, s // level) for s in self.image_shape]
                # Anti-aliasing: Gaussian smooth before downsample to prevent 
                # high-frequency spatial aliasing (matching SyN pyramid behavior)
                aa_sigma = math.log2(level)
                if aa_sigma > 0:
                    from syntx.syn import get_cached_gaussian_kernel_1d
                    k1d = get_cached_gaussian_kernel_1d(aa_sigma, device, dtype).squeeze(0)
                    pad_size = k1d.shape[-1] // 2
                    if self.dim == 3:
                        # Separable 3D Gaussian smoothing on (N,C,D,H,W) images
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
            for epoch in range(epochs):
                optimizer.zero_grad()
                sim_loss = self.forward(curr_fixed, curr_moving, multipoint_loss=multipoint_loss, lncc_window_size=lncc_ws)
                kinetic = torch.mean(self.velocity ** 2)
                total_loss = sim_loss + reg_weight * kinetic
                total_loss.backward()
                
                # Fluid regularization (smoothing velocity gradients)
                # Batched across all T time steps to minimize conv3d kernel launches
                with torch.no_grad():
                    if sigma_val > 0 and self.velocity.grad is not None:
                        T = self.n_time_steps
                        # Reshape (T, 1, *spatial, dim) -> (T, dim, *spatial) for batched filtering
                        grad_shape = self.velocity.grad.shape
                        if self.dim == 3:
                            # (T, 1, D, H, W, 3) -> squeeze batch -> (T, D, H, W, 3)
                            grad_batch = self.velocity.grad.squeeze(1)
                        else:
                            grad_batch = self.velocity.grad.squeeze(1)
                        # separable_gaussian_filter expects (B, *spatial, dim) channel-last
                        grad_smoothed = separable_gaussian_filter(
                            grad_batch, sigma=sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                        )
                        self.velocity.grad.copy_(grad_smoothed.unsqueeze(1))
                            
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
                        if denom > 1e-8:
                            slope = np.sum((x - x_mean) * (y - y_mean)) / denom
                            if slope >= -convergence_threshold:
                                if verbose:
                                    print(f"  Level {level} converged at epoch {epoch+1} (slope = {slope:.2e} >= -{convergence_threshold:.2e}). Early stopping level.")
                                break

            # MPS memory management at level transitions only (not per-epoch)
            if device.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()

        # Ensure velocity is at full (max) resolution after fit completes
        final_vel_shape = tuple(self.velocity.shape[2:-1])
        if final_vel_shape != max_vel_shape:
            self._resize_velocity(max_vel_shape, device, dtype)
            if verbose:
                print(f"  Final velocity upsample: {list(final_vel_shape)} → {list(max_vel_shape)}")

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

    def save_registration(self, prefix, save_warp=True, format="nii.gz", verbose=True):
        """
        Saves current TVFModel registration results to disk via syntx.write_registration.
        """
        from .io import write_registration
        if not hasattr(self, 'reg_results') or self.reg_results is None:
            raise ValueError("TVFModel has not been fitted yet. Call model.fit() first.")
        return write_registration(self.reg_results, prefix=prefix, save_warp=save_warp, format=format, verbose=verbose)

