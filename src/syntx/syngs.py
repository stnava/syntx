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
                p_norm_effective = torch.clamp(p_norm, min=1.0)

                if g_norm > 0:
                    trust_ratio = trust_coeff * p_norm_effective / (g_norm + eps)
                else:
                    trust_ratio = 1.0

                local_lr = lr * trust_ratio
                p.sub_(g * local_lr)
        return loss


class GeodesicShootingModel(nn.Module):
    def __init__(self, dim, image_shape, velocity_shape=None, spacing=None, origin=None, direction=None, fluid_sigma=1.0, elastic_sigma=0.0, transform_type='Affine', n_steps=5, solver='euler'):
        super().__init__()
        self.dim = dim
        self.image_shape = tuple(image_shape)
        if velocity_shape is None:
            velocity_shape = image_shape
        self.velocity_shape = tuple(velocity_shape)
        self.n_steps = n_steps
        
        self.spacing = spacing if spacing is not None else [1.0] * dim
        self.origin = origin if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction
        else:
            self.direction = np.eye(dim).tolist()
            
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.solver = solver
        
        # velocity_0 parameter: (1, 1, *velocity_shape, dim) to easily reuse PyTorch channel logic,
        # but prompt specifies (1, *velocity_shape, dim). Let's use (1, *velocity_shape, dim)
        self.velocity_0 = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)

    def _resize_velocity(self, new_shape, device=None, dtype=None):
        new_shape = tuple(new_shape)
        old_shape = tuple(self.velocity_0.shape[1:-1])  # (1, *spatial, dim)
        
        if new_shape == old_shape:
            return
            
        with torch.no_grad():
            old_vel = self.velocity_0.data  # (1, *spatial, dim)
            
            if self.dim == 3:
                # (1, D, H, W, 3) → (1, 3, D, H, W) for F.interpolate
                old_cf = old_vel.permute(0, 4, 1, 2, 3)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='trilinear', align_corners=True)
                # (1, 3, D', H', W') → (1, D', H', W', 3)
                new_vel = new_cf.permute(0, 2, 3, 4, 1)
            else:
                # (1, H, W, 2) → (1, 2, H, W) for F.interpolate
                old_cf = old_vel.permute(0, 3, 1, 2)
                new_cf = F.interpolate(old_cf, size=new_shape, mode='bilinear', align_corners=True)
                # (1, 2, H', W') → (1, H', W', 2)
                new_vel = new_cf.permute(0, 2, 3, 1)
            
            if device is not None:
                new_vel = new_vel.to(device=device)
            if dtype is not None:
                new_vel = new_vel.to(dtype=dtype)
                
            self.velocity_0 = nn.Parameter(new_vel.contiguous())

    def _get_metadata_tensors(self, device, dtype):
        spacing_rev = tuple(reversed(self.spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = np.asarray(self.direction)[::-1, ::-1].copy()
        
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(self.image_shape), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        return shape_t, spacing_t, origin_t, direction_t

    def _compute_jacobian(self, v, spacing_zyx):
        # v: (1, *spatial, dim) in ZYX order
        # Returns: (1, *spatial, dim, dim) where [i,j] = dv_i/dx_j
        # Uses proper boundary handling (replicate padding) instead of torch.roll
        # wrap-around, which corrupts boundary voxels by mixing opposite-side values.
        dim = v.shape[-1]
        Dv = torch.zeros(*v.shape, dim, device=v.device, dtype=v.dtype)
        for d in range(dim):
            # Central difference along spatial axis d (dim d+1 in tensor)
            # Interior voxels: (v[i+1] - v[i-1]) / (2*h)
            # Boundary voxels: forward/backward difference (v[i+1] - v[i]) / h
            n = v.shape[d + 1]
            h = spacing_zyx[d]
            
            # Build index slices for this axis
            s = [slice(None)] * v.ndim
            
            # Interior: central differences
            s_center = list(s); s_center[d+1] = slice(1, n-1)
            s_fwd = list(s); s_fwd[d+1] = slice(2, n)
            s_bwd = list(s); s_bwd[d+1] = slice(0, n-2)
            Dv[tuple(s_center)][..., :, d] = (v[tuple(s_fwd)] - v[tuple(s_bwd)]) / (2.0 * h)
            
            # Left boundary: forward difference
            s_0 = list(s); s_0[d+1] = slice(0, 1)
            s_1 = list(s); s_1[d+1] = slice(1, 2)
            Dv[tuple(s_0)][..., :, d] = (v[tuple(s_1)] - v[tuple(s_0)]) / h
            
            # Right boundary: backward difference
            s_last = list(s); s_last[d+1] = slice(n-1, n)
            s_prev = list(s); s_prev[d+1] = slice(n-2, n-1)
            Dv[tuple(s_last)][..., :, d] = (v[tuple(s_last)] - v[tuple(s_prev)]) / h
        return Dv

    def epdiff_rhs(self, v, spacing_zyx):
        Dv = self._compute_jacobian(v, spacing_zyx)
        # (Dv)^T v
        term1 = torch.einsum('...ji,...j->...i', Dv, v)
        # Dv · v  
        term2 = torch.einsum('...ij,...j->...i', Dv, v)
        # v * div(v)
        div_v = sum(Dv[..., d, d] for d in range(v.shape[-1]))
        term3 = v * div_v.unsqueeze(-1)
        return -(term1 + term2 + term3)

    def _smooth_field(self, field, sigma, spacing=None):
        if sigma <= 0:
            return field
        if spacing is None:
            # self.spacing is in XYZ order, but image_shape and field are in ZYX order.
            # Reverse spacing to match ZYX convention.
            spacing = list(reversed(self.spacing))
            
        vel_shape = field.shape[1:-1]
        # Compute spacing for smoothing based on grid size (both in ZYX order)
        curr_spacing = [sp * (orig_s / curr_s) for sp, orig_s, curr_s in zip(spacing, self.image_shape, vel_shape)]
        
        # separable_gaussian_filter expects (B, *spatial, dim)
        return separable_gaussian_filter(field, sigma=sigma, spacing=curr_spacing, sigma_mode='physical')

    def shoot(self, v0, n_steps, image_shape, spacing_zyx=None, _cached_phys_grid=None, _cached_meta=None):
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
            
        # Compute max allowed velocity magnitude for stability clamping.
        # EPDiff is cubic in v, so unclamped velocities can explode. Limit to
        # 2× the physical spacing per integration step to keep the deformation
        # smooth and diffeomorphic.
        max_v_phys = 2.0 * max(spacing_zyx) if isinstance(spacing_zyx, (list, tuple)) else 2.0 * spacing_zyx.max().item()
        
        for step in range(n_steps):
            rhs = self.epdiff_rhs(v, spacing_zyx)
            rhs_smooth = self._smooth_field(rhs, sigma=self.fluid_sigma)
            v = v + dt * rhs_smooth
            
            # Clamp velocity magnitude to prevent EPDiff velocity explosion.
            # The cubic nonlinearity in EPDiff can cause v to grow unboundedly;
            # this per-step clamp keeps the deformation diffeomorphic.
            v_mag = torch.norm(v, dim=-1, keepdim=True)
            v = torch.where(v_mag > max_v_phys, v * (max_v_phys / (v_mag + 1e-8)), v)
            
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
                v_sampled_cf = grid_sample_nd(v_advect_cf, phi_norm, mode='bilinear')
                v_sampled = v_sampled_cf.permute(0, 2, 3, 4, 1)
            else:
                v_advect_cf = v_for_advect.permute(0, 3, 1, 2)
                v_sampled_cf = grid_sample_nd(v_advect_cf, phi_norm, mode='bilinear')
                v_sampled = v_sampled_cf.permute(0, 2, 3, 1)
                
            disp = disp + dt * v_sampled
            
        return disp, v

    def forward(self, fixed_image, moving_image, multipoint_loss=None, lncc_window_size=5):
        """Forward pass with bidirectional multipoint loss evaluation.
        
        For geodesic shooting, shooting v0 gives the forward warp (moving→fixed)
        and shooting -v0 gives the inverse warp (fixed→moving). Evaluating LNCC
        at both endpoints provides balanced gradient signal from both directions.
        
        Parameters
        ----------
        multipoint_loss : list of float, optional
            Evaluation timepoints. t >= 0.5 warps moving→fixed, t < 0.5 warps
            fixed→moving. Default [0.0, 1.0] for bidirectional.
        """
        if multipoint_loss is None:
            multipoint_loss = [0.0, 1.0]
        
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
        
        # Compute inverse affine for fixed→moving direction
        M_phys_inv_zyx = torch.inverse(M_phys_zyx)
        t_phys_inv_zyx = -M_phys_inv_zyx @ t_phys_zyx

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        n_eval = 0
        
        for t_eval in multipoint_loss:
            if t_eval >= 0.5:
                # Forward direction: shoot v0, warp moving→fixed
                disp_fwd, _ = self.shoot(self.velocity_0, self.n_steps, target_shape,
                                         spacing_zyx=spacing_rev,
                                         _cached_phys_grid=phys_grid,
                                         _cached_meta=_cached_meta)
                phi_moving = (phys_grid + disp_fwd) @ M_phys_zyx.t() + t_phys_zyx
                phi_norm = physical_to_normalized_torch_cached(
                    phi_moving, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = grid_sample_nd(moving_image, phi_norm, mode='bilinear', padding_mode='zeros')
                total_loss = total_loss + lncc_loss_nd(fixed_image, moving_warped, window_size=lncc_window_size)
            else:
                # Inverse direction: shoot -v0, warp fixed→moving
                disp_inv, _ = self.shoot(-self.velocity_0, self.n_steps, target_shape,
                                         spacing_zyx=spacing_rev,
                                         _cached_phys_grid=phys_grid,
                                         _cached_meta=_cached_meta)
                phi_fixed = (phys_grid + disp_inv) @ M_phys_inv_zyx.t() + t_phys_inv_zyx
                phi_norm = physical_to_normalized_torch_cached(
                    phi_fixed, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped = grid_sample_nd(fixed_image, phi_norm, mode='bilinear', padding_mode='zeros')
                total_loss = total_loss + lncc_loss_nd(moving_image, fixed_warped, window_size=lncc_window_size)
            n_eval += 1
        
        return total_loss / max(n_eval, 1)

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
        if verbose: print("Optimizing geodesic shooting...")
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl')).lower()
        trust_coeff = kwargs.get('trust_coefficient', kwargs.get('trust', 0.05))
        
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)
        multipoint_loss = kwargs.get('multipoint_loss', [0.0, 1.0])
        
        interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
        
        sigma_mode = kwargs.get('sigma_mode', 'voxel')
        
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
            prev_vel_shape = tuple(self.velocity_0.shape[1:-1])
            
            if curr_vel_shape != prev_vel_shape:
                self._resize_velocity(curr_vel_shape, device, dtype)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")
            
            curr_spacing = [sp * level for sp in self.spacing]
            
            # Create optimizer fresh for this level (velocity parameter may have changed)
            if opt_type == 'lars':
                optimizer = LARS([self.velocity_0], lr=lr, trust_coefficient=trust_coeff)
            else:
                optimizer = torch.optim.Adam([self.velocity_0], lr=lr)
            
            # Reset momentum buffer for this level
            if cfl_momentum > 0 and opt_type == 'cfl':
                momentum_buffer = torch.zeros_like(self.velocity_0.data)
            
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
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Standard autograd mode with bidirectional multipoint loss
                sim_loss = self.forward(curr_fixed, curr_moving, multipoint_loss=multipoint_loss, lncc_window_size=lncc_ws)
                kinetic = torch.mean(self.velocity_0 ** 2)
                total_loss = sim_loss + reg_weight * kinetic
                total_loss.backward()
                
                # Fluid regularization (smoothing velocity gradients)
                # Batched across all T time steps to minimize conv3d kernel launches
                # Smoothing is the dominant bottleneck (~91% of per-epoch time).
                # smooth_every_n > 1 reduces this cost at the expense of gradient noise.
                with torch.no_grad():
                    should_smooth = (sigma_val > 0 and self.velocity_0.grad is not None
                                     and (smooth_every_n <= 1 or epoch % smooth_every_n == 0))
                    if should_smooth:
                        # Reshape (T, 1, *spatial, dim) -> (T, dim, *spatial) for batched filtering
                        grad_shape = self.velocity_0.grad.shape
                        if self.dim == 3:
                            # (T, 1, D, H, W, 3) -> squeeze batch -> (T, D, H, W, 3)
                            grad_batch = self.velocity_0.grad
                        else:
                            grad_batch = self.velocity_0.grad
                        # separable_gaussian_filter expects (B, *spatial, dim) channel-last
                        spatial_shape = list(grad_batch.shape[1:-1])
                        min_spatial = min(spatial_shape)
                        
                        # Fast smooth: downsample → smooth → upsample (9.4x speedup)
                        # Only worthwhile when spatial dims are large enough
                        if fast_smooth and min_spatial >= 32:
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
                        self.velocity_0.grad.copy_(grad_smoothed)
                            
                if opt_type == 'cfl':
                    with torch.no_grad():
                        if self.velocity_0.grad is not None:
                            grad = self.velocity_0.grad
                            # ITK-style CFL: compute max norm in VOXEL space (divide by spacing)
                            # This matches ITK's ScaleUpdateField() exactly:
                            #   localNorm += sqr(vector[d] / spacing[d])
                            #   scale = learningRate / maxNorm
                            sp_t = torch.tensor(curr_spacing, device=device, dtype=dtype)
                            grad_voxel = grad / sp_t  # convert to voxel units
                            max_g_voxel = torch.sqrt(torch.sum(grad_voxel**2, dim=-1)).max()
                            if max_g_voxel > 1e-8:
                                cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                                # Cap effective CFL at 0.10 for EPDiff stability.
                                # EPDiff's RHS is cubic in v (Dv*v terms), so the same
                                # CFL step that's safe for linear SyN/TVF advection can
                                # amplify EPDiff velocities nonlinearly. 0.10 provides
                                # stable convergence without folding.
                                effective_cfl = min(cfl_step_val, 0.10)
                                # Compute CFL update: scaledUpdate = (learningRate / maxNorm) * gradient
                                update = (effective_cfl / max_g_voxel) * grad
                                
                                # Apply momentum for faster convergence
                                if cfl_momentum > 0 and momentum_buffer is not None:
                                    momentum_buffer.mul_(cfl_momentum).add_(update)
                                    self.velocity_0.data.sub_(momentum_buffer)
                                else:
                                    self.velocity_0.data.sub_(update)
                else:
                    optimizer.step()

                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                with torch.no_grad():
                    if elastic_sigma_val > 0:
                        vel_batch = self.velocity_0
                        vel_smoothed = separable_gaussian_filter(
                            vel_batch, sigma=elastic_sigma_val, spacing=vel_spacing, sigma_mode=sigma_mode
                        )
                        self.velocity_0.copy_(vel_smoothed)
                    vel_clamp_val = float(kwargs.get('velocity_clamp', kwargs.get('clamp', 50.0)))
                    self.velocity_0.clamp_(min=-vel_clamp_val, max=vel_clamp_val)
                    cfl_max_val = kwargs.get('cfl_max', None)
                    if cfl_max_val is not None and float(cfl_max_val) > 0:
                        sp_t = torch.tensor(self.spacing, device=device, dtype=dtype)
                        vel_vox = self.velocity_0 / sp_t
                        max_vox = torch.norm(vel_vox, dim=-1).max()
                        if max_vox > float(cfl_max_val):
                            self.velocity_0.mul_(float(cfl_max_val) / (max_vox + 1e-8))
                    

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

            # MPS memory management at level transitions only (not per-epoch)
            if device.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()

        # Ensure velocity is at full image resolution after fit completes
        final_vel_shape = tuple(self.velocity_0.shape[1:-1])
        if final_vel_shape != tuple(self.image_shape):
            self._resize_velocity(self.image_shape, device, dtype)
            if verbose:
                print(f"  Final velocity upsample: {list(final_vel_shape)} → {list(self.image_shape)}")



    @torch.no_grad()
    def get_forward_warp(self, image_shape=None):
        disp, _ = self.shoot(self.velocity_0, self.n_steps, image_shape)
        return disp
        
    @torch.no_grad()
    def get_inverse_warp(self, image_shape=None):
        disp, _ = self.shoot(-self.velocity_0, self.n_steps, image_shape)
        return disp

def syngs_registration(
    fixed,
    moving,
    type_of_transform='SyNGS',
    initial_transform=None,
    syn_metric='lncc',
    syn_sampling=2,
    reg_iterations=None,
    affine_iterations=None,
    grad_step=0.20,
    flow_sigma=3.0,
    total_sigma=0.0,
    n_steps=5,
    verbose=False,
    backend='pytorch',
    levels=None,
    cfl_momentum=0.9,
    multipoint_loss=None,
    fast_smooth=True,
    **kwargs
):
    """
    High-level TVF (Time-Varying Velocity Field) registration function matching
    the ``syntx.syn()`` / ``syntx.registration()`` interface.

    Usage is identical to ``syntx.syn()``::

        import syntx
        reg = syntx.syngs(fixed=fi, moving=mi)
        warped = reg['warpedmovout']
        transforms = reg['fwdtransforms']

    Parameters
    ----------
    fixed : ANTsImage
        Fixed target image.
    moving : ANTsImage
        Moving source image.
    type_of_transform : str
        Ignored (included for API parity with registration()).
    syn_metric : str
        Similarity metric. Currently only 'lncc' is supported.
    syn_sampling : int
        LNCC radius (window_size = 2 * syn_sampling + 1). Default 2.
    reg_iterations : list of int or None
        Number of deformable iterations per level. Default [150, 150, 0].
    affine_iterations : list of int or int or None
        Number of affine iterations. Default 100.
    grad_step : float
        CFL voxel bound step size. Default 0.20.
    flow_sigma : float
        Fluid regularization sigma in ITK variance convention (σ² = flow_sigma).
        Default 3.0 (actual σ = √3 ≈ 1.73).
    total_sigma : float
        Elastic (total field) regularization sigma in ITK variance convention.
        Default 0.0 (disabled).
    n_steps : int
        Number of TVF time keyframes. Default 4.
    verbose : bool
        If True, print optimization progress.
    levels : list of int or None
        Multi-resolution pyramid levels. Default [4, 2, 1].
    cfl_momentum : float
        SGD-style momentum for CFL updates. Default 0.9. Set 0.0 to disable.
    multipoint_loss : list of float or None
        ODE evaluation timepoints for loss. Default [0.0, 1.0] (direct-space).
        Use [0.5] for geodesic midpoint, [0.0, 0.5, 1.0] for triplet.
    fast_smooth : bool
        If True, smooth gradients at half resolution (9x faster). Default True.
    **kwargs
        Additional parameters passed to GeodesicShootingModel.fit().

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
                'model': GeodesicShootingModel,              # fitted model
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
            solver=kwargs.pop('solver', 'euler'),
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

        T_grid_learned = np.array(get_affine_matrix_jax(model.affine_params, dim, 'Affine'))
        if hasattr(model, 'T_init') and model.T_init is not None:
            T_grid = T_grid_learned @ np.array(model.T_init)
        else:
            T_grid = T_grid_learned
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
