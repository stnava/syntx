"""
Geodesic Shooting Registration in JAX (`syntx.syngs` JAX backend).
================================================================
Symmetrically mirrors PyTorch `GeodesicShootingModel` in `src/syntx/syngs.py`
with dual momentum parameters (velocity_0_fwd and velocity_0_inv),
image gradient initialization, and full-trajectory inverse identity composition loss.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp

from .syn_jax import (
    get_affine_matrix_jax, get_physical_grid_jax,
    physical_to_normalized_jax_cached, jax_grid_sample,
    local_ncc_loss_nd_jax, mattes_mi_loss_nd_jax, grid_to_physical_affine_jax,
    separable_gaussian_filter_jax, interpolate_jax
)
from .tvf_jax import clamp_affine_params_jax, adam_step_dict


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
        n_steps=5,
        symmetric=True,
        inverse_identity_weight=1.0
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
        self.symmetric = symmetric
        self.inverse_identity_weight = inverse_identity_weight

        # Dual momentum fields for symmetric shooting: v0_fwd (Fixed space) and v0_inv (Moving space)
        self.velocity_0_fwd = jnp.zeros((1, *self.velocity_shape, self.dim), dtype=jnp.float32)
        if self.symmetric:
            self.velocity_0_inv = jnp.zeros((1, *self.velocity_shape, self.dim), dtype=jnp.float32)
        else:
            self.velocity_0_inv = None

        self.velocity_0 = self.velocity_0_fwd

        num_rot = dim * (dim - 1) // 2
        self.affine_params = {
            'translation': jnp.zeros(dim, dtype=jnp.float32),
            'omega': jnp.zeros(num_rot, dtype=jnp.float32),
            'scale': jnp.ones(1, dtype=jnp.float32),
            'anisotropic_scale': jnp.ones(dim, dtype=jnp.float32),
            'shear': jnp.zeros(num_rot, dtype=jnp.float32)
        }

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
        if v.ndim == self.dim + 1:
            v = v[None]
        J = []
        for i in range(self.dim):
            vi = v[..., i]
            grad_vi = []
            for j in range(self.dim):
                n = vi.shape[j + 1]
                
                slice_left_0 = [slice(None)] * vi.ndim
                slice_left_1 = [slice(None)] * vi.ndim
                slice_left_0[j + 1] = slice(0, 1)
                slice_left_1[j + 1] = slice(1, 2)
                left_bound = (vi[tuple(slice_left_1)] - vi[tuple(slice_left_0)]) / spacing_zyx[j]
                
                slice_mid_fwd = [slice(None)] * vi.ndim
                slice_mid_bwd = [slice(None)] * vi.ndim
                slice_mid_fwd[j + 1] = slice(2, n)
                slice_mid_bwd[j + 1] = slice(0, n - 2)
                interior = (vi[tuple(slice_mid_fwd)] - vi[tuple(slice_mid_bwd)]) / (2.0 * spacing_zyx[j])
                
                slice_right_1 = [slice(None)] * vi.ndim
                slice_right_2 = [slice(None)] * vi.ndim
                slice_right_1[j + 1] = slice(n - 1, n)
                slice_right_2[j + 1] = slice(n - 2, n - 1)
                right_bound = (vi[tuple(slice_right_1)] - vi[tuple(slice_right_2)]) / spacing_zyx[j]
                
                diff = jnp.concatenate([left_bound, interior, right_bound], axis=j + 1)
                grad_vi.append(diff)
            J.append(jnp.stack(grad_vi, axis=-1))
        return jnp.stack(J, axis=-2)

    def _create_boundary_mask(self, vel_shape, border_width=4):
        dim = self.dim
        axes_masks = []
        for d in range(dim):
            n_d = vel_shape[d]
            coords = jnp.arange(n_d, dtype=jnp.float32)
            dist = jnp.minimum(coords, float(n_d - 1) - coords)
            mask_d = jnp.where(
                dist < float(border_width),
                0.5 * (1.0 - jnp.cos(math.pi * dist / float(border_width))),
                jnp.ones_like(dist)
            )
            shape_d = [1] * dim
            shape_d[d] = n_d
            axes_masks.append(mask_d.reshape(*shape_d))

        mask = axes_masks[0]
        for d in range(1, dim):
            mask = mask * axes_masks[d]
        return mask[None, ..., None]

    def apply_green_operator(self, m, vel_shape, spacing_zyx):
        if self.fluid_sigma <= 0:
            return m
        dim = self.dim
        alpha = float(self.fluid_sigma / 2.0)
        s = 2.0

        bmask = self._create_boundary_mask(vel_shape, border_width=4)
        m_tapered = m * bmask

        k_axes = []
        for d in range(dim):
            n_d = vel_shape[d]
            sp_d = spacing_zyx[d]
            if d == dim - 1:
                k_d = jnp.fft.rfftfreq(n_d, d=sp_d) * (2.0 * math.pi)
            else:
                k_d = jnp.fft.fftfreq(n_d, d=sp_d) * (2.0 * math.pi)
            k_axes.append(k_d)

        k_mesh = []
        for d in range(dim):
            shape_k = [1] * dim
            shape_k[d] = len(k_axes[d])
            k_mesh.append(k_axes[d].reshape(*shape_k))

        k_sq = sum(k_j ** 2 for k_j in k_mesh)
        K_fourier = 1.0 / ((1.0 + alpha * k_sq) ** s)

        spatial_dims = tuple(range(2, 2 + dim))
        if dim == 3:
            m_cf = jnp.transpose(m_tapered, (0, 4, 1, 2, 3))
        else:
            m_cf = jnp.transpose(m_tapered, (0, 3, 1, 2))

        m_fft = jnp.fft.rfftn(m_cf.astype(jnp.float32), axes=spatial_dims)
        K_bc = K_fourier[None, None, ...]
        v_fft = m_fft * K_bc
        v_cf = jnp.fft.irfftn(v_fft, s=vel_shape, axes=spatial_dims)

        if dim == 3:
            v_out = jnp.transpose(v_cf, (0, 2, 3, 4, 1))
        else:
            v_out = jnp.transpose(v_cf, (0, 2, 3, 1))

        return v_out * bmask

    def spectral_jacobian(self, v, vel_shape, spacing_zyx):
        dim = self.dim
        k_axes = []
        for d in range(dim):
            n_d = vel_shape[d]
            sp_d = spacing_zyx[d]
            if d == dim - 1:
                k_d = jnp.fft.rfftfreq(n_d, d=sp_d) * (2.0 * math.pi)
            else:
                k_d = jnp.fft.fftfreq(n_d, d=sp_d) * (2.0 * math.pi)
            k_axes.append(k_d)

        k_mesh = []
        for d in range(dim):
            shape_k = [1] * dim
            shape_k[d] = len(k_axes[d])
            k_mesh.append(k_axes[d].reshape(*shape_k))

        spatial_dims = tuple(range(2, 2 + dim))

        if dim == 3:
            v_cf = jnp.transpose(v, (0, 4, 1, 2, 3))
        else:
            v_cf = jnp.transpose(v, (0, 3, 1, 2))

        v_fft = jnp.fft.rfftn(v_cf.astype(jnp.float32), axes=spatial_dims)

        Dv_list = []
        for i in range(dim):
            v_i_fft = v_fft[:, i:i+1, ...]
            dv_i_components = []
            for j in range(dim):
                k_j = k_mesh[j][None, None, ...]
                dv_ij_fft = 1j * k_j * v_i_fft
                dv_ij = jnp.fft.irfftn(dv_ij_fft, s=vel_shape, axes=spatial_dims)
                dv_i_components.append(dv_ij)
            if dim == 3:
                dv_i = jnp.transpose(jnp.concatenate(dv_i_components, axis=1), (0, 2, 3, 4, 1))
            else:
                dv_i = jnp.transpose(jnp.concatenate(dv_i_components, axis=1), (0, 2, 3, 1))
            Dv_list.append(dv_i)
        return jnp.stack(Dv_list, axis=-2)

    def epdiff_rhs(self, v, spacing_zyx):
        vel_shape = tuple(v.shape[1:-1])
        if getattr(self, 'solver', 'spectral_rk4') in ('spectral', 'spectral_rk4'):
            Dv = self.spectral_jacobian(v, vel_shape, spacing_zyx)
        else:
            Dv = self._compute_jacobian(v, spacing_zyx)

        v_in = v[..., None]
        term1 = jnp.squeeze(jnp.matmul(jnp.swapaxes(Dv, -2, -1), v_in), axis=-1)
        term2 = jnp.squeeze(jnp.matmul(Dv, v_in), axis=-1)
        div_v = jnp.trace(Dv, axis1=-2, axis2=-1)[..., None]
        term3 = v * div_v
        ad_v = term1 + term2 + term3

        if getattr(self, 'solver', 'spectral_rk4') in ('spectral', 'spectral_rk4'):
            return -self.apply_green_operator(ad_v, vel_shape, spacing_zyx)
        else:
            return -ad_v

    def shoot(self, v0, n_steps, image_shape=None):
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        dt = 1.0 / n_steps
        v = jnp.array(v0)
        disp = jnp.zeros((1, *target_shape, self.dim), dtype=jnp.float32)

        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]
        phys_grid = get_physical_grid_jax(
            target_shape, curr_spacing, self.origin, self.direction
        )
        shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(target_shape, curr_spacing)
        spacing_zyx = spacing_t.tolist()

        max_v_phys = 50.0
        for step in range(n_steps):
            if getattr(self, 'solver', 'spectral_rk4') in ('spectral_rk4', 'rk4'):
                k1 = self.epdiff_rhs(v, spacing_zyx)
                k2 = self.epdiff_rhs(v + 0.5 * dt * k1, spacing_zyx)
                k3 = self.epdiff_rhs(v + 0.5 * dt * k2, spacing_zyx)
                k4 = self.epdiff_rhs(v + dt * k3, spacing_zyx)
                v = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                rhs = self.epdiff_rhs(v, spacing_zyx)
                v = v + rhs * dt

            v_mag = jnp.linalg.norm(v, axis=-1, keepdims=True)
            v = jnp.where(v_mag > max_v_phys, v * (max_v_phys / (v_mag + 1e-8)), v)

            if tuple(v.shape[1:-1]) != target_shape:
                if self.dim == 3:
                    v_cf = jnp.transpose(v, (0, 4, 1, 2, 3))
                    v_up_cf = interpolate_jax(v_cf, target_shape, self.dim)
                    v_for_advect = jnp.transpose(v_up_cf, (0, 2, 3, 4, 1))
                else:
                    v_cf = jnp.transpose(v, (0, 3, 1, 2))
                    v_up_cf = interpolate_jax(v_cf, target_shape, self.dim)
                    v_for_advect = jnp.transpose(v_up_cf, (0, 2, 3, 1))
            else:
                v_for_advect = v

            phi_current = phys_grid + disp
            phi_norm = physical_to_normalized_jax_cached(
                phi_current, shape_t, spacing_t, origin_t, direction_t
            )
            if self.dim == 3:
                v_cf = jnp.transpose(v_for_advect, (0, 4, 1, 2, 3))
                v_sampled_cf = jax_grid_sample(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = jnp.transpose(v_sampled_cf, (0, 2, 3, 4, 1))
            else:
                v_cf = jnp.transpose(v_for_advect, (0, 3, 1, 2))
                v_sampled_cf = jax_grid_sample(v_cf, phi_norm, mode='bilinear', padding_mode='border')
                v_sampled = jnp.transpose(v_sampled_cf, (0, 2, 3, 1))

            disp = disp + v_sampled * dt

        return disp

    def init_velocities_from_image_gradients(self, fixed_image, moving_image):
        spacing_rev = tuple(reversed(self.spacing))
        grad_f = jnp.stack(jnp.gradient(fixed_image.squeeze(0).squeeze(0), *spacing_rev), axis=-1)[None]
        grad_m = jnp.stack(jnp.gradient(moving_image.squeeze(0).squeeze(0), *spacing_rev), axis=-1)[None]

        vel_shape_f = tuple(self.velocity_0_fwd.shape[1:-1])
        if tuple(grad_f.shape[1:-1]) != vel_shape_f:
            grad_f = interpolate_jax(grad_f, vel_shape_f, self.dim)

        if self.symmetric and self.velocity_0_inv is not None:
            vel_shape_m = tuple(self.velocity_0_inv.shape[1:-1])
            if tuple(grad_m.shape[1:-1]) != vel_shape_m:
                grad_m = interpolate_jax(grad_m, vel_shape_m, self.dim)

        norm_f = jnp.sqrt(jnp.sum(grad_f**2, axis=-1, keepdims=True)) + 1e-8
        norm_m = jnp.sqrt(jnp.sum(grad_m**2, axis=-1, keepdims=True)) + 1e-8

        v0_f = 5e-3 * (grad_f / (jnp.max(norm_f) + 1e-8))
        v0_m = 5e-3 * (grad_m / (jnp.max(norm_m) + 1e-8))

        self.velocity_0_fwd = v0_f
        if self.symmetric and self.velocity_0_inv is not None:
            self.velocity_0_inv = v0_m
        self.velocity_0 = self.velocity_0_fwd

    def _resize_single_velocity(self, vel, new_shape):
        if vel is None:
            return None
        new_shape = tuple(new_shape)
        old_shape = tuple(vel.shape[1:-1])
        if new_shape == old_shape:
            return vel
        if self.dim == 3:
            vel_cf = jnp.transpose(vel, (0, 4, 1, 2, 3))
            vel_resized_cf = interpolate_jax(vel_cf, new_shape, self.dim)
            return jnp.transpose(vel_resized_cf, (0, 2, 3, 4, 1))
        else:
            vel_cf = jnp.transpose(vel, (0, 3, 1, 2))
            vel_resized_cf = interpolate_jax(vel_cf, new_shape, self.dim)
            return jnp.transpose(vel_resized_cf, (0, 2, 3, 1))

    def _resize_velocity(self, new_shape):
        self.velocity_0_fwd = self._resize_single_velocity(self.velocity_0_fwd, new_shape)
        if self.symmetric and self.velocity_0_inv is not None:
            self.velocity_0_inv = self._resize_single_velocity(self.velocity_0_inv, new_shape)
        self.velocity_0 = self.velocity_0_fwd

    def forward(self, fixed_image, moving_image, velocity_0_fwd=None, velocity_0_inv=None, affine_params=None, multipoint_loss=None, lncc_window_size=5):
        if velocity_0_fwd is None:
            velocity_0_fwd = self.velocity_0_fwd
        if velocity_0_inv is None:
            velocity_0_inv = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -velocity_0_fwd
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

        M_phys_inv_zyx = jnp.linalg.inv(M_phys_zyx)
        t_phys_inv_zyx = -(M_phys_inv_zyx @ t_phys_zyx)

        # 1. Forward shooting -> warp moving to fixed
        disp_fwd = self.shoot(velocity_0_fwd, n_steps=self.n_steps, image_shape=target_shape)
        phi_moving_affine = (phys_grid + disp_fwd) @ M_phys_zyx.T + t_phys_zyx
        phi_norm_fwd = physical_to_normalized_jax_cached(
            phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = jax_grid_sample(moving_image, phi_norm_fwd, mode='bilinear', padding_mode='zeros')
        loss_fwd = local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=lncc_window_size)

        # 2. Inverse shooting -> warp fixed to moving
        disp_inv = self.shoot(velocity_0_inv, n_steps=self.n_steps, image_shape=target_shape)
        phi_fixed_affine = (phys_grid + disp_inv) @ M_phys_inv_zyx.T + t_phys_inv_zyx
        phi_norm_inv = physical_to_normalized_jax_cached(
            phi_fixed_affine, shape_t, spacing_t, origin_t, direction_t
        )
        fixed_warped = jax_grid_sample(fixed_image, phi_norm_inv, mode='bilinear', padding_mode='zeros')
        loss_inv = local_ncc_loss_nd_jax(moving_image, fixed_warped, window_size=lncc_window_size)

        sim_loss = 0.5 * (loss_fwd + loss_inv)

        # 3. Inverse identity loss
        if self.symmetric and self.inverse_identity_weight > 0:
            phi_inv_pure = phys_grid + disp_inv
            phi_inv_pure_norm = physical_to_normalized_jax_cached(
                phi_inv_pure, shape_t, spacing_t, origin_t, direction_t
            )
            if self.dim == 3:
                disp_fwd_cf = jnp.transpose(disp_fwd, (0, 4, 1, 2, 3))
                disp_fwd_at_inv_cf = jax_grid_sample(disp_fwd_cf, phi_inv_pure_norm, mode='bilinear', padding_mode='border')
                disp_fwd_at_inv = jnp.transpose(disp_fwd_at_inv_cf, (0, 2, 3, 4, 1))
            else:
                disp_fwd_cf = jnp.transpose(disp_fwd, (0, 3, 1, 2))
                disp_fwd_at_inv_cf = jax_grid_sample(disp_fwd_cf, phi_inv_pure_norm, mode='bilinear', padding_mode='border')
                disp_fwd_at_inv = jnp.transpose(disp_fwd_at_inv_cf, (0, 2, 3, 1))
            comp_disp_1 = disp_inv + disp_fwd_at_inv

            phi_fwd_pure = phys_grid + disp_fwd
            phi_fwd_pure_norm = physical_to_normalized_jax_cached(
                phi_fwd_pure, shape_t, spacing_t, origin_t, direction_t
            )
            if self.dim == 3:
                disp_inv_cf = jnp.transpose(disp_inv, (0, 4, 1, 2, 3))
                disp_inv_at_fwd_cf = jax_grid_sample(disp_inv_cf, phi_fwd_pure_norm, mode='bilinear', padding_mode='border')
                disp_inv_at_fwd = jnp.transpose(disp_inv_at_fwd_cf, (0, 2, 3, 4, 1))
            else:
                disp_inv_cf = jnp.transpose(disp_inv, (0, 3, 1, 2))
                disp_inv_at_fwd_cf = jax_grid_sample(disp_inv_cf, phi_fwd_pure_norm, mode='bilinear', padding_mode='border')
                disp_inv_at_fwd = jnp.transpose(disp_inv_at_fwd_cf, (0, 2, 3, 1))
            comp_disp_2 = disp_fwd + disp_inv_at_fwd

            inv_id_loss = 0.5 * (jnp.mean(comp_disp_1 ** 2) + jnp.mean(comp_disp_2 ** 2))
        else:
            inv_id_loss = 0.0

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
        **kwargs
    ):
        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        fixed_image = jnp.array(fixed_image.numpy() if hasattr(fixed_image, 'numpy') else fixed_image)
        moving_image = jnp.array(moving_image.numpy() if hasattr(moving_image, 'numpy') else moving_image)
        if fixed_image.ndim == self.dim:
            fixed_image = fixed_image[None, None]
        if moving_image.ndim == self.dim:
            moving_image = moving_image[None, None]

        initial_transform = kwargs.get('initial_transform', None)
        if initial_transform is not None:
            from .syn import parse_ants_affine
            tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
            parsed_M, parsed_t = parse_ants_affine(tx_list, self.dim)
            if parsed_M is not None:
                T_mat = np.eye(self.dim + 1, dtype=np.float32)
                T_mat[:self.dim, :self.dim] = parsed_M
                T_mat[:self.dim, self.dim] = parsed_t
                self.T_init = jnp.array(T_mat)

        if self.T_init is not None:
            self.affine_params['T_init'] = self.T_init

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
                return local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=2*lncc_radius+1)

            grad_aff_fn = jax.grad(affine_loss_fn)

            for epoch in range(affine_epochs):
                grads_aff = grad_aff_fn(self.affine_params)
                self.affine_params, m_aff, v_aff, t_aff = adam_step_dict(
                    self.affine_params, grads_aff, m_aff, v_aff, t_aff, lr=1e-3
                )
                self.affine_params = clamp_affine_params_jax(self.affine_params)

        # Optimize velocity field across pyramid levels
        if verbose: print("Optimizing geodesic shooting in JAX...")
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))

        multipoint_loss = kwargs.get('multipoint_loss', [0.0, 1.0])
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl')).lower()
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        fast_smooth = kwargs.get('fast_smooth', True)
        smooth_pyramid = kwargs.get('smooth_pyramid', kwargs.get('pre_smooth', False))

        self.init_velocities_from_image_gradients(fixed_image, moving_image)

        m_fwd = None
        m_inv = None

        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue

            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            self._resize_velocity(curr_vel_shape)

            if isinstance(fluid_sigmas_input, (list, tuple)):
                curr_fluid_sig = fluid_sigmas_input[min(idx, len(fluid_sigmas_input) - 1)]
            else:
                curr_fluid_sig = fluid_sigmas_input

            self.fluid_sigma = curr_fluid_sig
            sigma_voxel = math.sqrt(curr_fluid_sig) if curr_fluid_sig > 0 else 0.0
            curr_spacing = [sp * level for sp in self.spacing]

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
                    curr_fixed = interpolate_jax(fixed_smooth[None, None], down_shape, self.dim)
                    curr_moving = interpolate_jax(moving_smooth[None, None], down_shape, self.dim)
                else:
                    curr_fixed = interpolate_jax(fixed_image, down_shape, self.dim)
                    curr_moving = interpolate_jax(moving_image, down_shape, self.dim)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image

            def gs_loss_fn(v_fwd, v_inv):
                sim_loss = self.forward(curr_fixed, curr_moving, velocity_0_fwd=v_fwd, velocity_0_inv=v_inv, multipoint_loss=multipoint_loss, lncc_window_size=2*lncc_radius+1)
                kinetic = jnp.mean(v_fwd ** 2)
                if self.symmetric and v_inv is not None:
                    kinetic = 0.5 * (kinetic + jnp.mean(v_inv ** 2))
                return sim_loss + reg_weight * kinetic

            grad_gs_fn = jax.grad(gs_loss_fn, argnums=(0, 1))

            for epoch in range(epochs):
                v_inv_in = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else self.velocity_0_fwd
                grad_fwd, grad_inv = grad_gs_fn(self.velocity_0_fwd, v_inv_in)

                vel_spacing = kwargs.get('vel_spacing', None)
                sp_vel = vel_spacing if vel_spacing is not None else curr_spacing
                grad_smoothed_fwd = self.apply_green_operator(grad_fwd, curr_vel_shape, sp_vel)
                if self.symmetric and self.velocity_0_inv is not None:
                    grad_smoothed_inv = self.apply_green_operator(grad_inv, curr_vel_shape, sp_vel)
                else:
                    grad_smoothed_inv = grad_inv

                if opt_type == 'cfl':
                    vel_spacing = kwargs.get('vel_spacing', None)
                    sp_vel = vel_spacing if vel_spacing is not None else curr_spacing
                    sp_j = jnp.array(sp_vel)
                    cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.25)))
                    effective_cfl = float(cfl_step_val)

                    # Forward velocity update
                    grad_voxel_fwd = grad_smoothed_fwd / sp_j
                    max_g_voxel_fwd = jnp.max(jnp.sqrt(jnp.sum(grad_voxel_fwd**2, axis=-1)))
                    if max_g_voxel_fwd > 1e-8:
                        up_fwd = (effective_cfl / max_g_voxel_fwd) * grad_smoothed_fwd
                        if cfl_momentum > 0:
                            m_fwd = up_fwd if m_fwd is None else (cfl_momentum * m_fwd + up_fwd)
                            self.velocity_0_fwd = self.velocity_0_fwd - m_fwd
                        else:
                            self.velocity_0_fwd = self.velocity_0_fwd - up_fwd

                    # Inverse velocity update
                    if self.symmetric and self.velocity_0_inv is not None:
                        grad_voxel_inv = grad_smoothed_inv / sp_j
                        max_g_voxel_inv = jnp.max(jnp.sqrt(jnp.sum(grad_voxel_inv**2, axis=-1)))
                        if max_g_voxel_inv > 1e-8:
                            up_inv = (effective_cfl / max_g_voxel_inv) * grad_smoothed_inv
                            if cfl_momentum > 0:
                                m_inv = up_inv if m_inv is None else (cfl_momentum * m_inv + up_inv)
                                self.velocity_0_inv = self.velocity_0_inv - m_inv
                            else:
                                self.velocity_0_inv = self.velocity_0_inv - up_inv
                else:
                    self.velocity_0_fwd = self.velocity_0_fwd - lr * grad_smoothed_fwd
                    if self.symmetric and self.velocity_0_inv is not None:
                        self.velocity_0_inv = self.velocity_0_inv - lr * grad_smoothed_inv

        self.velocity_0 = self.velocity_0_fwd

    def get_forward_warp(self, image_shape=None):
        return np.array(self.shoot(self.velocity_0_fwd, n_steps=self.n_steps, image_shape=image_shape))

    def get_inverse_warp(self, image_shape=None):
        v0_inv = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        return np.array(self.shoot(v0_inv, n_steps=self.n_steps, image_shape=image_shape))
