import math
import numpy as np
import jax
import jax.numpy as jnp
from .syn_jax import (
    get_physical_grid_jax,
    physical_to_normalized_jax_cached,
    separable_gaussian_filter_jax,
    get_affine_matrix_jax,
    grid_to_physical_affine_jax,
    local_ncc_loss_nd_jax,
    mattes_mi_loss_nd_jax,
    jax_grid_sample,
    interpolate_jax,
)
from .tvf_jax import adam_step, adam_step_dict, clamp_affine_params_jax


class GeodesicShootingModelJAX:
    """
    Geodesic Shooting Registration Model in JAX.
    Symmetrically mirrors PyTorch GeodesicShootingModel.
    """
    def __init__(
        self,
        dim,
        image_shape,
        velocity_shape,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=1.0,
        elastic_sigma=0.0,
        transform_type='Affine',
        solver='euler',
        n_steps=10
    ):
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)

        self.spacing = list(spacing) if spacing is not None else [1.0] * dim
        self.origin = list(origin) if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = np.array(direction, dtype=np.float32).tolist()
        else:
            self.direction = np.eye(dim, dtype=np.float32).tolist()

        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.transform_type = transform_type
        self.solver = solver
        self.n_steps = n_steps

        # Single initial velocity field parameter: (1, *velocity_shape, dim)
        self.velocity_0 = jnp.zeros((1, *self.velocity_shape, self.dim), dtype=jnp.float32)

        num_rot = dim * (dim - 1) // 2
        self.affine_params = {
            'translation': jnp.zeros(dim, dtype=jnp.float32),
            'omega': jnp.zeros(num_rot, dtype=jnp.float32),
            'scale': jnp.ones(1, dtype=jnp.float32),
            'anisotropic_scale': jnp.ones(dim, dtype=jnp.float32),
            'shear': jnp.zeros(num_rot, dtype=jnp.float32)
        }

        # Optional initial affine transform
        self.T_init = None

    def _get_metadata_tensors(self, target_shape, curr_spacing):
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(self.direction)[::-1, ::-1])

        shape_t = jnp.array(target_shape, dtype=jnp.float32)
        spacing_t = jnp.array(spacing_rev, dtype=jnp.float32)
        origin_t = jnp.array(origin_rev, dtype=jnp.float32)
        direction_t = jnp.array(direction_rev, dtype=jnp.float32)

        return shape_t, spacing_t, origin_t, direction_t

    def _compute_jacobian(self, v, spacing_zyx):
        """
        Computes the Jacobian matrix of v w.r.t spatial dimensions using central differences.
        v shape: (*spatial, dim)
        Returns: J shape (*spatial, dim, dim) where J[..., i, j] = dv_i / dx_j
        """
        J = []
        for i in range(self.dim):
            vi = v[..., i]
            grad_vi = []
            for j in range(self.dim):
                # Central difference along axis j
                diff = (jnp.roll(vi, -1, axis=j) - jnp.roll(vi, 1, axis=j)) / (2.0 * spacing_zyx[j])
                grad_vi.append(diff)
            J.append(jnp.stack(grad_vi, axis=-1))
        return jnp.stack(J, axis=-2)

    def epdiff_rhs(self, v, spacing_zyx):
        r"""
        Computes the RHS of the EPDiff equation (L2 metric):
        dv/dt = - (v \cdot \nabla) v - v (\nabla \cdot v) - (\nabla v)^T v
        """
        J = self._compute_jacobian(v, spacing_zyx)
        
        # 1. (v \cdot \nabla) v
        term1 = jnp.einsum('...ij,...j->...i', J, v)
        
        # 2. v (\nabla \cdot v)
        div_v = jnp.trace(J, axis1=-2, axis2=-1)
        term2 = v * div_v[..., None]
        
        # 3. (\nabla v)^T v
        term3 = jnp.einsum('...ji,...j->...i', J, v)
        
        return -term1 - term2 - term3

    def shoot(self, v0, n_steps, image_shape=None):
        """
        Integrate EPDiff and advect phi forward in time using Euler integration.
        """
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        dt = 1.0 / max(1, n_steps)

        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]
        
        phys_grid = get_physical_grid_jax(
            target_shape, curr_spacing, self.origin, self.direction
        )
        phi = phys_grid
        
        shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(target_shape, curr_spacing)
        
        # Ensure v0 matches target_shape
        v = v0
        if tuple(v.shape[1:-1]) != target_shape:
            if self.dim == 3:
                v_cf = jnp.transpose(v, (0, 4, 1, 2, 3))
                v = jnp.stack([
                    jax.image.resize(v_cf[0, c], target_shape, method='trilinear' if hasattr(jax.image, 'resize') else 'linear')
                    for c in range(self.dim)
                ], axis=-1).reshape(1, *target_shape, self.dim)
            else:
                v = jax.image.resize(
                    v.squeeze(0), (*target_shape, self.dim), method='bilinear'
                ).reshape(1, *target_shape, self.dim)
        
        v_spatial = v[0]
        sigma_val = math.sqrt(self.fluid_sigma) if self.fluid_sigma > 0 else 0.0
        
        for step in range(n_steps):
            # 1. Update velocity via EPDiff with Green's kernel smoothing
            dv = self.epdiff_rhs(v_spatial, spacing_t)
            if sigma_val > 0:
                dv = separable_gaussian_filter_jax(dv, sigma=sigma_val, spacing=None, sigma_mode='voxel')
            v_spatial = v_spatial + dv * dt
            
            # 2. Advect coordinates
            phi_norm = physical_to_normalized_jax_cached(
                phi, shape_t, spacing_t, origin_t, direction_t
            )
            
            if self.dim == 2:
                v_cf = jnp.transpose(v_spatial[None, ...], (0, 3, 1, 2))
                v_sampled_cf = jax_grid_sample(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = jnp.transpose(v_sampled_cf, (0, 2, 3, 1))[0]
            else:
                v_cf = jnp.transpose(v_spatial[None, ...], (0, 4, 1, 2, 3))
                v_sampled_cf = jax_grid_sample(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = jnp.transpose(v_sampled_cf, (0, 2, 3, 4, 1))[0]
                
            phi = phi + v_sampled * dt
            
        return phi - phys_grid

    def forward(self, fixed_image, moving_image, velocity_0=None, affine_params=None, multipoint_loss=None, lncc_window_size=5):
        """
        Registration forward pass computing LNCC loss.
        """
        if velocity_0 is None:
            velocity_0 = self.velocity_0
        if affine_params is None:
            affine_params = self.affine_params

        fixed_image = jnp.array(fixed_image)
        moving_image = jnp.array(moving_image)
        target_shape = tuple(fixed_image.shape[2:])

        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]

        phys_grid = get_physical_grid_jax(
            target_shape, curr_spacing, self.origin, self.direction
        )
        shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(target_shape, curr_spacing)

        T_grid = get_affine_matrix_jax(affine_params, self.dim, self.transform_type)
        M_phys, t_phys = grid_to_physical_affine_jax(
            T_grid, target_shape, curr_spacing, self.origin, self.direction,
            target_shape, curr_spacing, self.origin, self.direction
        )

        coord_perm = list(range(self.dim - 1, -1, -1))
        perm_idx = jnp.array(coord_perm, dtype=jnp.int32)
        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
        t_phys_zyx = t_phys[perm_idx]

        phi_0_to_1 = self.shoot(velocity_0, n_steps=self.n_steps, image_shape=target_shape)
        
        phi_moving_affine_end = (phys_grid + phi_0_to_1) @ M_phys_zyx.T + t_phys_zyx
        phi_norm_end = physical_to_normalized_jax_cached(
            phi_moving_affine_end, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = jax_grid_sample(moving_image, phi_norm_end, mode='bilinear', padding_mode='zeros')
        
        return local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=lncc_window_size)

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
        Multi-resolution optimization for Geodesic Shooting in JAX.
        """
        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        fixed_image = jnp.array(fixed_image)
        moving_image = jnp.array(moving_image)

        if isinstance(affine_epochs, (list, tuple)):
            affine_epochs = sum(affine_epochs)
        if affine_epochs > 0:
            if verbose: print("Optimizing affine pre-alignment in JAX...")
            m_aff = {k: jnp.zeros_like(v) for k, v in self.affine_params.items()}
            v_aff = {k: jnp.zeros_like(v) for k, v in self.affine_params.items()}
            t_aff = 0

            def affine_loss_fn(params_aff):
                phys_grid = get_physical_grid_jax(
                    self.image_shape, self.spacing, self.origin, self.direction
                )
                T_grid = get_affine_matrix_jax(params_aff, self.dim, self.transform_type)
                M_phys, t_phys = grid_to_physical_affine_jax(
                    T_grid, self.image_shape, self.spacing, self.origin, self.direction,
                    self.image_shape, self.spacing, self.origin, self.direction
                )
                coord_perm = list(range(self.dim - 1, -1, -1))
                perm_idx = jnp.array(coord_perm, dtype=jnp.int32)
                M_phys_zyx = M_phys[perm_idx][:, perm_idx]
                t_phys_zyx = t_phys[perm_idx]

                phi_moving_affine = phys_grid @ M_phys_zyx.T + t_phys_zyx
                shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(self.image_shape, self.spacing)

                phi_moving_norm = physical_to_normalized_jax_cached(
                    phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = jax_grid_sample(moving_image, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                aff_metric = kwargs.get('aff_metric', 'mattes_mi')
                if aff_metric.lower() in ('mattes_mi', 'mattes', 'mi'):
                    mattes_bins = int(kwargs.get('mattes_bins', kwargs.get('num_bins', 32)))
                    return mattes_mi_loss_nd_jax(fixed_image, moving_warped, num_bins=mattes_bins)
                else:
                    return local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=2*lncc_radius+1)

            grad_aff_fn = jax.grad(affine_loss_fn)

            for epoch in range(affine_epochs):
                grads_aff = grad_aff_fn(self.affine_params)
                self.affine_params, m_aff, v_aff, t_aff = adam_step_dict(
                    self.affine_params, grads_aff, m_aff, v_aff, t_aff, lr=1e-3
                )
                self.affine_params = clamp_affine_params_jax(self.affine_params)

        # 2. Optimize initial velocity field
        if verbose: print("Optimizing Geodesic Shooting in JAX...")
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)

        m_vel = jnp.zeros_like(self.velocity_0)
        v_vel = jnp.zeros_like(self.velocity_0)
        t_vel = 0

        multipoint_loss = kwargs.get('multipoint_loss', [1.0])
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl')).lower()
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        momentum_buffer = None
        smooth_pyramid = kwargs.get('smooth_pyramid', kwargs.get('pre_smooth', False))
        fast_smooth = kwargs.get('fast_smooth', True)

        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue

            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            prev_vel_shape = tuple(self.velocity_0.shape[1:-1])
            if curr_vel_shape != prev_vel_shape:
                if self.dim == 3:
                    v_cf = jnp.transpose(self.velocity_0, (0, 4, 1, 2, 3))
                    v_resized = jnp.stack([
                        jax.image.resize(v_cf[0, c], curr_vel_shape, method='trilinear' if hasattr(jax.image, 'resize') else 'linear')
                        for c in range(self.dim)
                    ], axis=-1).reshape(1, *curr_vel_shape, self.dim)
                else:
                    v_resized = jax.image.resize(
                        self.velocity_0.squeeze(0), (*curr_vel_shape, self.dim), method='bilinear'
                    ).reshape(1, *curr_vel_shape, self.dim)
                self.velocity_0 = v_resized
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")

            momentum_buffer = None

            if isinstance(fluid_sigmas_input, (list, tuple)):
                curr_fluid_sig = fluid_sigmas_input[min(idx, len(fluid_sigmas_input) - 1)]
            else:
                curr_fluid_sig = fluid_sigmas_input

            if isinstance(elastic_sigmas_input, (list, tuple)):
                curr_elastic_sig = elastic_sigmas_input[min(idx, len(elastic_sigmas_input) - 1)]
            else:
                curr_elastic_sig = elastic_sigmas_input

            sigma_voxel = math.sqrt(curr_fluid_sig) if curr_fluid_sig > 0 else 0.0
            elastic_sigma_voxel = math.sqrt(curr_elastic_sig) if curr_elastic_sig > 0 else 0.0

            curr_spacing = [sp * level for sp in self.spacing]

            if verbose:
                print(f"Level {level}: {epochs} max epochs, vel_grid={list(curr_vel_shape)} (fluid_sigma={curr_fluid_sig:.2f}, elastic_sigma={curr_elastic_sig:.2f})")

            if level > 1:
                down_shape = tuple([max(8, s // level) for s in self.image_shape])
                if smooth_pyramid:
                    aa_sigma = float(kwargs.get('aa_sigma', math.log2(level)))
                    fixed_smooth = separable_gaussian_filter_jax(
                        fixed_image.squeeze(0).squeeze(0), sigma=aa_sigma, spacing=None, sigma_mode='voxel'
                    )
                    moving_smooth = separable_gaussian_filter_jax(
                        moving_image.squeeze(0).squeeze(0), sigma=aa_sigma, spacing=None, sigma_mode='voxel'
                    )
                    if self.dim == 3:
                        fixed_smooth = fixed_smooth.reshape(1, 1, *fixed_smooth.shape[:3])
                        moving_smooth = moving_smooth.reshape(1, 1, *moving_smooth.shape[:3])
                    else:
                        fixed_smooth = fixed_smooth.reshape(1, 1, *fixed_smooth.shape[:2])
                        moving_smooth = moving_smooth.reshape(1, 1, *moving_smooth.shape[:2])
                    curr_fixed = interpolate_jax(fixed_smooth, down_shape, self.dim)
                    curr_moving = interpolate_jax(moving_smooth, down_shape, self.dim)
                else:
                    curr_fixed = interpolate_jax(fixed_image, down_shape, self.dim)
                    curr_moving = interpolate_jax(moving_image, down_shape, self.dim)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image

            def gs_loss_fn(vel0):
                sim_loss = self.forward(curr_fixed, curr_moving, velocity_0=vel0, multipoint_loss=multipoint_loss, lncc_window_size=2*lncc_radius+1)
                kinetic = jnp.mean(vel0 ** 2)
                return sim_loss + reg_weight * kinetic

            grad_gs_fn = jax.grad(gs_loss_fn)
            recent_losses = []

            for epoch in range(epochs):
                grad_raw = grad_gs_fn(self.velocity_0)

                if sigma_voxel > 0:
                    spatial_shape = list(grad_raw.shape[1:-1])
                    min_spatial = min(spatial_shape)
                    if fast_smooth and min_spatial >= 32:
                        down_shape_sm = [max(8, s // 2) for s in spatial_shape]
                        gt = grad_raw[0]
                        gt_down = jax.image.resize(gt, (*down_shape_sm, self.dim), method='bilinear')
                        gt_sm = separable_gaussian_filter_jax(gt_down, sigma=sigma_voxel, spacing=None, sigma_mode='voxel')
                        gt_up = jax.image.resize(gt_sm, (*spatial_shape, self.dim), method='bilinear')
                        grad_smoothed = gt_up[None]
                    else:
                        gt = grad_raw[0]
                        gt_sm = separable_gaussian_filter_jax(gt, sigma=sigma_voxel, spacing=None, sigma_mode='voxel')
                        grad_smoothed = gt_sm[None]
                else:
                    grad_smoothed = grad_raw

                if opt_type == 'cfl':
                    sp_j = jnp.array(curr_spacing)
                    grad_voxel = grad_smoothed / sp_j
                    max_g_voxel = jnp.max(jnp.sqrt(jnp.sum(grad_voxel**2, axis=-1)))
                    cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                    effective_cfl = min(cfl_step_val, 0.20)

                    if max_g_voxel > 1e-8:
                        update = (effective_cfl / max_g_voxel) * grad_smoothed
                        if cfl_momentum > 0:
                            if momentum_buffer is None:
                                momentum_buffer = update
                            else:
                                momentum_buffer = cfl_momentum * momentum_buffer + update
                            self.velocity_0 = self.velocity_0 - momentum_buffer
                        else:
                            self.velocity_0 = self.velocity_0 - update
                else:
                    self.velocity_0, m_vel, v_vel, t_vel = adam_step(
                        self.velocity_0, grad_smoothed, m_vel, v_vel, t_vel, lr=lr
                    )

                if elastic_sigma_voxel > 0:
                    vt = self.velocity_0[0]
                    vt_sm = separable_gaussian_filter_jax(vt, sigma=elastic_sigma_voxel, spacing=None, sigma_mode='voxel')
                    self.velocity_0 = vt_sm[None]

                vel_clamp_val = float(kwargs.get('velocity_clamp', kwargs.get('clamp', 50.0)))
                self.velocity_0 = jnp.clip(self.velocity_0, -vel_clamp_val, vel_clamp_val)

                if epoch % 5 == 0 or epoch == epochs - 1:
                    loss_val = float(self.forward(curr_fixed, curr_moving, velocity_0=self.velocity_0))
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

        final_vel_shape = tuple(self.velocity_0.shape[1:-1])
        if final_vel_shape != tuple(self.image_shape):
            if self.dim == 3:
                v_cf = jnp.transpose(self.velocity_0, (0, 4, 1, 2, 3))
                v_resized = jnp.stack([
                    jax.image.resize(v_cf[0, c], self.image_shape, method='trilinear' if hasattr(jax.image, 'resize') else 'linear')
                    for c in range(self.dim)
                ], axis=-1).reshape(1, *self.image_shape, self.dim)
            else:
                v_resized = jax.image.resize(
                    self.velocity_0.squeeze(0), (*self.image_shape, self.dim), method='bilinear'
                ).reshape(1, *self.image_shape, self.dim)
            self.velocity_0 = v_resized
            if verbose:
                print(f"  Final velocity upsample: {list(final_vel_shape)} → {list(self.image_shape)}")

    def get_forward_warp(self, image_shape=None):
        return self.shoot(self.velocity_0, self.n_steps, image_shape)

    def get_inverse_warp(self, image_shape=None):
        return self.shoot(-self.velocity_0, self.n_steps, image_shape)
