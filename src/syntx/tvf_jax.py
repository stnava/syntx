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


def clamp_affine_params_jax(params):
    params = dict(params)
    if 'scale' in params:
        params['scale'] = jnp.clip(params['scale'], 0.05, 20.0)
    if 'anisotropic_scale' in params:
        params['anisotropic_scale'] = jnp.clip(params['anisotropic_scale'], 0.05, 20.0)
    if 'shear' in params:
        params['shear'] = jnp.clip(params['shear'], -5.0, 5.0)
    if 'omega' in params:
        params['omega'] = jnp.clip(params['omega'], -np.pi, np.pi)
    return params


def adam_step(param, grad, m, v, t, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    t_next = t + 1
    m_next = beta1 * m + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (grad ** 2)
    m_hat = m_next / (1.0 - (beta1 ** t_next))
    v_hat = v_next / (1.0 - (beta2 ** t_next))
    param_next = param - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return param_next, m_next, v_next, t_next


def adam_step_dict(params, grads, m_dict, v_dict, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    t_next = t + 1
    new_params = {}
    new_m = {}
    new_v = {}
    for k in params:
        if k == 'T_init':
            new_params[k] = params[k]
            new_m[k] = m_dict[k]
            new_v[k] = v_dict[k]
            continue
        g = grads[k]
        m_next = beta1 * m_dict[k] + (1.0 - beta1) * g
        v_next = beta2 * v_dict[k] + (1.0 - beta2) * (g ** 2)
        m_hat = m_next / (1.0 - (beta1 ** t_next))
        v_hat = v_next / (1.0 - (beta2 ** t_next))
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = m_next
        new_v[k] = v_next
    return new_params, new_m, new_v, t_next


class TVFModelJAX:
    """
    Time-Varying Velocity Field (TVF) Registration Model in JAX.
    Symmetrically mirrors PyTorch TVFModel.
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
        moving_shape=None,
        moving_spacing=None,
        moving_origin=None,
        moving_direction=None,
        fluid_sigma=1.0,
        elastic_sigma=0.0,
        transform_type='Affine',
        solver='rk4',
        integration_steps_per_interval=4,
        antisymmetric=False,
        image_grad_clip=6.0,
        velocity_clamp=None,
        cfl_max=0.40
    ):
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps
        self.antisymmetric = antisymmetric
        self.image_grad_clip = image_grad_clip
        self.velocity_clamp = velocity_clamp
        self.cfl_max = cfl_max

        self.spacing = list(spacing) if spacing is not None else [1.0] * dim
        self.origin = list(origin) if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = np.array(direction, dtype=np.float32).tolist()
        else:
            self.direction = np.eye(dim, dtype=np.float32).tolist()

        self.moving_shape = tuple(moving_shape) if moving_shape is not None else self.image_shape
        self.moving_spacing = list(moving_spacing) if moving_spacing is not None else self.spacing
        self.moving_origin = list(moving_origin) if moving_origin is not None else self.origin
        if moving_direction is not None:
            self.moving_direction = np.array(moving_direction, dtype=np.float32).tolist()
        else:
            self.moving_direction = list(self.direction)

        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.transform_type = transform_type
        self.solver = solver
        self.integration_steps_per_interval = integration_steps_per_interval

        # Velocity parameter: (T, 1, *velocity_shape, dim)
        self.velocity = jnp.zeros((n_time_steps, 1, *self.velocity_shape, self.dim), dtype=jnp.float32)

        num_rot = dim * (dim - 1) // 2
        self.affine_params = {
            'translation': jnp.zeros(dim, dtype=jnp.float32),
            'omega': jnp.zeros(num_rot, dtype=jnp.float32),
            'scale': jnp.ones(1, dtype=jnp.float32),
            'anisotropic_scale': jnp.ones(dim, dtype=jnp.float32),
            'shear': jnp.zeros(num_rot, dtype=jnp.float32)
        }

        # Optional initial affine transform (set externally via tvf_registration)
        self.T_init = None

    def _create_boundary_mask(self, spatial_shape, border_width=None):
        dim = len(spatial_shape)
        if border_width is None:
            border_width = max(1, min(spatial_shape) // 32)
        if border_width <= 0:
            return jnp.ones((1, *spatial_shape, 1), dtype=jnp.float32)

        axes_masks = []
        for d in range(dim):
            n_d = spatial_shape[d]
            idx = jnp.arange(n_d, dtype=jnp.float32)
            dist = jnp.minimum(idx, (n_d - 1) - idx)
            mask_d = jnp.where(
                dist < border_width,
                0.5 * (1.0 - jnp.cos(np.pi * dist / float(border_width))),
                jnp.ones_like(dist)
            )
            shape_d = [1] * dim
            shape_d[d] = n_d
            axes_masks.append(mask_d.reshape(*shape_d))

        mask = axes_masks[0]
        for d in range(1, dim):
            mask = mask * axes_masks[d]
        return mask[None, ..., None]

    def _apply_sobolev_green_operator(self, m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0, border_width=0):
        if fluid_sigma <= 0:
            return m
        dim = self.dim
        orig_shape = m.shape
        spatial_shape = orig_shape[-(dim + 1):-1]
        dtype = m.dtype

        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s_val = float(s)

        bmask = self._create_boundary_mask(spatial_shape, border_width=border_width)
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
        K_fourier = 1.0 / ((1.0 + alpha_val * k_sq) ** s_val)

        spatial_dims = tuple(range(2, 2 + dim))
        m_cf = jnp.moveaxis(m_tapered, -1, 1)

        m_fft = jnp.fft.rfftn(m_cf.astype(jnp.float32), axes=spatial_dims)
        K_bc = K_fourier[None, None, ...]
        v_fft = m_fft * K_bc
        v_cf = jnp.fft.irfftn(v_fft, s=spatial_shape, axes=spatial_dims).astype(dtype)

        v_out = (jnp.moveaxis(v_cf, 1, -1) * bmask).reshape(orig_shape)
        return v_out

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None, spacing=None):
        if fluid_sigma <= 0:
            return m
        dim = self.dim
        orig_shape = m.shape
        spatial_shape = orig_shape[-(dim + 1):-1]
        dtype = m.dtype

        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s = 2.0

        k_axes = []
        for d in range(dim):
            n_d = spatial_shape[d]
            k_vec = jnp.arange(1, n_d + 1, dtype=jnp.float32)
            lambda_d = 4.0 * (jnp.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
            k_axes.append(lambda_d)

        k_mesh = []
        for d in range(dim):
            shape_k = [1] * dim
            shape_k[d] = len(k_axes[d])
            k_mesh.append(k_axes[d].reshape(*shape_k))

        lambda_sq = sum(k_j for k_j in k_mesh)
        K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

        m_flat = m.reshape(-1, *spatial_shape, dim)
        m_cf = jnp.moveaxis(m_flat, -1, 1).astype(jnp.float32)

        def _dst1_1d(arr, axis):
            n_d = arr.shape[axis]
            z_shape = list(arr.shape)
            z_shape[axis] = 1
            z = jnp.zeros(z_shape, dtype=arr.dtype)
            rev = -jnp.flip(arr, axis=axis)
            padded = jnp.concatenate([z, arr, z, rev], axis=axis)
            fft_1d = jnp.fft.rfft(padded, axis=axis)
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(1, n_d + 1)
            return -0.5 * jnp.imag(fft_1d[tuple(sl)])

        curr = m_cf
        for d in range(dim):
            axis = 2 + d
            curr = _dst1_1d(curr, axis)

        K_bc = K_dst[None, None, ...]
        v_dst = curr * K_bc

        curr_inv = v_dst
        for d in range(dim):
            axis = 2 + d
            n_d = spatial_shape[d]
            curr_inv = _dst1_1d(curr_inv, axis) * (2.0 / float(n_d + 1))

        v_out = jnp.moveaxis(curr_inv, 1, -1).astype(dtype).reshape(orig_shape)
        return v_out

    def project_antisymmetric(self, vel=None):
        """
        Project keyframe velocity fields onto the temporally anti-symmetric subspace:
        v(t_k) <- 0.5 * (v(t_k) - v(t_{K-1-k}))
        Ensures exact geodesic symmetry across time: v(x, 1-t) = -v(x, t).
        """
        if vel is None:
            vel = self.velocity
        v_flipped = jnp.flip(vel, axis=0)
        return 0.5 * (vel - v_flipped)

    def _get_metadata_tensors(self, target_shape, curr_spacing):
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(self.direction)[::-1, ::-1])

        shape_t = jnp.array(target_shape, dtype=jnp.float32)
        spacing_t = jnp.array(spacing_rev, dtype=jnp.float32)
        origin_t = jnp.array(origin_rev, dtype=jnp.float32)
        direction_t = jnp.array(direction_rev, dtype=jnp.float32)

        return shape_t, spacing_t, origin_t, direction_t

    def _get_moving_metadata_tensors(self):
        """Helper to get spatial metadata as tensors for cached normalized coordinates (Moving Image)."""
        spacing_rev = tuple(reversed(self.moving_spacing))
        origin_rev = tuple(reversed(self.moving_origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(self.moving_direction)[::-1, ::-1])

        shape_t = jnp.array(self.moving_shape, dtype=jnp.float32)
        spacing_t = jnp.array(spacing_rev, dtype=jnp.float32)
        origin_t = jnp.array(origin_rev, dtype=jnp.float32)
        direction_t = jnp.array(direction_rev, dtype=jnp.float32)

        return shape_t, spacing_t, origin_t, direction_t

    def integrate(self, t_start, t_end, velocity=None, n_steps=None, image_shape=None):
        """
        Integrates the time-varying velocity field ODE from t_start to t_end in JAX.
        """
        if velocity is None:
            velocity = self.velocity

        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape

        if n_steps is None:
            n_steps = self.n_time_steps * self.integration_steps_per_interval

        dt = (t_end - t_start) / max(1, n_steps)

        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)
        ]

        phys_grid = get_physical_grid_jax(
            target_shape, curr_spacing, self.origin, self.direction
        )
        phi_t = phys_grid

        shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(target_shape, curr_spacing)

        if self.dim == 2:
            velocity_cf = jnp.transpose(velocity, (0, 1, 4, 2, 3))
        else:
            velocity_cf = jnp.transpose(velocity, (0, 1, 5, 2, 3, 4))

        def interpolate_velocity(t):
            T = self.n_time_steps
            if T == 1:
                return velocity_cf[0]
            if T == 2:
                t_scaled = t
                return (1.0 - t_scaled) * velocity_cf[0] + t_scaled * velocity_cf[1]

            t_scaled = t * (T - 1)
            i = jnp.clip(jnp.floor(t_scaled).astype(jnp.int32), 0, T - 2)
            s = t_scaled - i

            i0 = jnp.clip(i - 1, 0, T - 1)
            i1 = jnp.clip(i, 0, T - 1)
            i2 = jnp.clip(i + 1, 0, T - 1)
            i3 = jnp.clip(i + 2, 0, T - 1)

            s2 = s * s
            s3 = s2 * s

            c0 = 0.5 * (-s3 + 2.0 * s2 - s)
            c1 = 0.5 * (3.0 * s3 - 5.0 * s2 + 2.0)
            c2 = 0.5 * (-3.0 * s3 + 4.0 * s2 + s)
            c3 = 0.5 * (s3 - s2)

            return c0 * velocity_cf[i0] + c1 * velocity_cf[i1] + c2 * velocity_cf[i2] + c3 * velocity_cf[i3]

        def upsample_velocity(v_coarse_cf):
            if tuple(v_coarse_cf.shape[2:]) == target_shape:
                return v_coarse_cf
            return interpolate_jax(v_coarse_cf, target_shape, self.dim)

        def eval_v(t, current_phi):
            v_cf = interpolate_velocity(t)
            v_fine_cf = upsample_velocity(v_cf)
            phi_norm = physical_to_normalized_jax_cached(
                current_phi, shape_t, spacing_t, origin_t, direction_t
            )
            v_sampled_cf = jax_grid_sample(v_fine_cf, phi_norm, mode='bilinear', padding_mode='border')
            if self.dim == 2:
                return jnp.transpose(v_sampled_cf, (0, 2, 3, 1))
            else:
                return jnp.transpose(v_sampled_cf, (0, 2, 3, 4, 1))

        if self.solver == 'euler':
            for step in range(n_steps):
                t_current = t_start + step * dt
                phi_t = phi_t + eval_v(t_current, phi_t) * dt
        elif self.solver == 'rk4':
            for step in range(n_steps):
                t_current = t_start + step * dt
                k1 = eval_v(t_current, phi_t)
                k2 = eval_v(t_current + 0.5 * dt, phi_t + 0.5 * dt * k1)
                k3 = eval_v(t_current + 0.5 * dt, phi_t + 0.5 * dt * k2)
                k4 = eval_v(t_current + dt, phi_t + dt * k3)
                phi_t = phi_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return phi_t - phys_grid

    def forward(self, fixed_image, moving_image, velocity=None, affine_params=None, multipoint_loss=[0.0, 0.5, 1.0], lncc_window_size=5):
        """
        Registration forward pass supporting arbitrary multi-point LNCC evaluation timepoints t in [0, 1] in JAX.
        Default: multipoint_loss = [0.0, 0.5, 1.0] (anchors fixed t=0, midpoint t=0.5, and moving t=1 space).
        """
        if velocity is None:
            velocity = self.velocity
        if affine_params is None:
            affine_params = self.affine_params

        fixed_image = jnp.array(fixed_image)
        moving_image = jnp.array(moving_image)
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

        phys_grid = get_physical_grid_jax(
            target_shape, curr_spacing, self.origin, self.direction
        )
        shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(target_shape, curr_spacing)

        T_grid = get_affine_matrix_jax(affine_params, self.dim, self.transform_type)
        M_phys, t_phys = grid_to_physical_affine_jax(
            T_grid, target_shape, curr_spacing, self.origin, self.direction,
            self.moving_shape, self.moving_spacing, self.moving_origin, self.moving_direction
        )

        # M_phys and t_phys are already returned in ZYX order from grid_to_physical_affine_jax
        M_phys_zyx = M_phys
        t_phys_zyx = t_phys

        losses = []
        for t_k in eval_points:
            t_k = float(t_k)
            if abs(t_k - 0.0) < 1e-5:
                # Fixed Space (t=0.0: bidirectional warping + inverse identity penalty)
                phi_0_to_1 = self.integrate(0.0, 1.0, velocity=velocity, image_shape=target_shape)
                phi_1_to_0 = self.integrate(1.0, 0.0, velocity=velocity, image_shape=target_shape)

                # Direction 1: u_inv + u_fwd(x + u_inv)
                phi_1_to_0_norm = physical_to_normalized_jax_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                # Direction 2: u_fwd + u_inv(x + u_fwd)
                phi_0_to_1_norm = physical_to_normalized_jax_cached(
                    phys_grid + phi_0_to_1, shape_t, spacing_t, origin_t, direction_t
                )

                if self.dim == 3:
                    u_fwd_cf = jnp.transpose(phi_0_to_1, (0, 4, 1, 2, 3))
                    u_fwd_at_inv_cf = jax_grid_sample(u_fwd_cf, phi_1_to_0_norm, mode='bilinear', padding_mode='border')
                    u_fwd_at_inv = jnp.transpose(u_fwd_at_inv_cf, (0, 2, 3, 4, 1))

                    u_inv_cf = jnp.transpose(phi_1_to_0, (0, 4, 1, 2, 3))
                    u_inv_at_fwd_cf = jax_grid_sample(u_inv_cf, phi_0_to_1_norm, mode='bilinear', padding_mode='border')
                    u_inv_at_fwd = jnp.transpose(u_inv_at_fwd_cf, (0, 2, 3, 4, 1))
                else:
                    u_fwd_cf = jnp.transpose(phi_0_to_1, (0, 3, 1, 2))
                    u_fwd_at_inv_cf = jax_grid_sample(u_fwd_cf, phi_1_to_0_norm, mode='bilinear', padding_mode='border')
                    u_fwd_at_inv = jnp.transpose(u_fwd_at_inv_cf, (0, 2, 3, 1))

                    u_inv_cf = jnp.transpose(phi_1_to_0, (0, 3, 1, 2))
                    u_inv_at_fwd_cf = jax_grid_sample(u_inv_cf, phi_0_to_1_norm, mode='bilinear', padding_mode='border')
                    u_inv_at_fwd = jnp.transpose(u_inv_at_fwd_cf, (0, 2, 3, 1))

                inv_id_err_1 = phi_1_to_0 + u_fwd_at_inv
                inv_id_err_2 = phi_0_to_1 + u_inv_at_fwd
                inv_id_loss = 0.5 * (jnp.mean(inv_id_err_1 ** 2) + jnp.mean(inv_id_err_2 ** 2))

                # Forward warping
                shape_m, spacing_m, origin_m, direction_m = self._get_moving_metadata_tensors()
                phi_moving_affine_end = (phys_grid + phi_0_to_1) @ M_phys_zyx.T + t_phys_zyx
                phi_norm_end = physical_to_normalized_jax_cached(
                    phi_moving_affine_end, shape_m, spacing_m, origin_m, direction_m
                )
                moving_warped = jax_grid_sample(moving_image, phi_norm_end, mode='bilinear', padding_mode='zeros')
                loss_fwd = local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=lncc_window_size)

                # Inverse warping
                phi_fixed_norm_end = physical_to_normalized_jax_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped = jax_grid_sample(fixed_image, phi_fixed_norm_end, mode='bilinear', padding_mode='zeros')
                phi_moving_identity = phys_grid @ M_phys_zyx.T + t_phys_zyx
                phi_moving_identity_norm = physical_to_normalized_jax_cached(
                    phi_moving_identity, shape_m, spacing_m, origin_m, direction_m
                )
                moving_affine = jax_grid_sample(moving_image, phi_moving_identity_norm, mode='bilinear', padding_mode='zeros')
                loss_inv = local_ncc_loss_nd_jax(fixed_warped, moving_affine, window_size=lncc_window_size)

                inv_id_weight = float(getattr(self, 'inverse_identity_weight', 0.05))
                return 0.5 * (loss_fwd + loss_inv) + inv_id_weight * inv_id_loss
            else:
                # Midpoint or Intermediate Space t_k
                phi_tk_to_fixed = self.integrate(t_k, 0.0, velocity=velocity, image_shape=target_shape)
                phi_tk_to_moving = self.integrate(t_k, 1.0, velocity=velocity, image_shape=target_shape)

                phi_fixed_norm_tk = physical_to_normalized_jax_cached(
                    phys_grid + phi_tk_to_fixed, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped_tk = jax_grid_sample(fixed_image, phi_fixed_norm_tk, mode='bilinear', padding_mode='zeros')

                shape_m, spacing_m, origin_m, direction_m = self._get_moving_metadata_tensors()
                phi_moving_affine_tk = (phys_grid + phi_tk_to_moving) @ M_phys_zyx.T + t_phys_zyx
                phi_moving_norm_tk = physical_to_normalized_jax_cached(
                    phi_moving_affine_tk, shape_m, spacing_m, origin_m, direction_m
                )
                moving_warped_tk = jax_grid_sample(moving_image, phi_moving_norm_tk, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_warped_tk, moving_warped_tk, window_size=lncc_window_size))

        return jnp.mean(jnp.stack(losses))

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
        Multi-resolution optimization in JAX.
        """
        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        fixed_image = jnp.array(fixed_image.numpy() if hasattr(fixed_image, 'numpy') else fixed_image)
        moving_image = jnp.array(moving_image.numpy() if hasattr(moving_image, 'numpy') else moving_image)
        if fixed_image.ndim == self.dim:
            fixed_image = fixed_image[None, None]
        if moving_image.ndim == self.dim:
            moving_image = moving_image[None, None]

        # Identity registration guard: short-circuit if fixed and moving images are identical
        if fixed_image.shape == moving_image.shape and jnp.allclose(fixed_image, moving_image, atol=1e-5):
            if verbose:
                print("[TVF-JAX] Identity image pair detected in fit(). Setting velocity to zero.")
            self.velocity = jnp.zeros_like(self.velocity)
            return

        initial_transform = kwargs.get('initial_transform', None)
        if initial_transform is not None:
            from .syn import parse_ants_affine
            tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
            parsed_M, parsed_t = parse_ants_affine(tx_list, self.dim)
            if parsed_M is not None:
                # Convert parsed_M and parsed_t (XYZ physical) to T_init homogeneous grid matrix
                T_mat = np.eye(self.dim + 1, dtype=np.float32)
                T_mat[:self.dim, :self.dim] = parsed_M
                T_mat[:self.dim, self.dim] = parsed_t
                self.T_init = jnp.array(T_mat)

        if self.T_init is not None:
            self.affine_params['T_init'] = self.T_init

        if isinstance(affine_epochs, (list, tuple)):
            affine_epochs = sum(affine_epochs)
        # 1. Optimize affine pre-alignment first
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
                    self.moving_shape, self.moving_spacing, self.moving_origin, self.moving_direction
                )
                # M_phys and t_phys are already returned in ZYX order from grid_to_physical_affine_jax
                M_phys_zyx = M_phys
                t_phys_zyx = t_phys

                phi_moving_affine = phys_grid @ M_phys_zyx.T + t_phys_zyx
                shape_m, spacing_m, origin_m, direction_m = self._get_moving_metadata_tensors()

                phi_moving_norm = physical_to_normalized_jax_cached(
                    phi_moving_affine, shape_m, spacing_m, origin_m, direction_m
                )
                moving_warped = jax_grid_sample(moving_image, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                aff_metric = kwargs.get('aff_metric', 'mattes_mi')
                if aff_metric.lower() in ('mattes_mi', 'mattes', 'mi'):
                    mattes_bins = int(kwargs.get('mattes_bins', kwargs.get('num_bins', 32)))
                    sampling_pct = float(kwargs.get('sampling_percentage', 0.2))
                    return mattes_mi_loss_nd_jax(fixed_image, moving_warped, num_bins=mattes_bins, sampling_percentage=sampling_pct)
                else:
                    return local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=2*lncc_radius+1)

            grad_aff_fn = jax.grad(affine_loss_fn)

            for epoch in range(affine_epochs):
                grads_aff = grad_aff_fn(self.affine_params)
                self.affine_params, m_aff, v_aff, t_aff = adam_step_dict(
                    self.affine_params, grads_aff, m_aff, v_aff, t_aff, lr=1e-3
                )
                self.affine_params = clamp_affine_params_jax(self.affine_params)

        # 2. Optimize velocity field across pyramid levels
        if verbose: print("Optimizing TVF in JAX...")
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', self.fluid_sigma))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)

        m_vel = jnp.zeros_like(self.velocity)
        v_vel = jnp.zeros_like(self.velocity)
        t_vel = 0

        multipoint_loss = kwargs.get('multipoint_loss', [0.0, 1.0])
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'cfl')).lower()
        cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.35)))
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        momentum_buffer = None
        smooth_pyramid = kwargs.get('smooth_pyramid', True)
        fast_smooth = kwargs.get('fast_smooth', True)

        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue

            # --- Pyramid-proportional velocity grid resizing (PyTorch parity) ---
            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            prev_vel_shape = tuple(self.velocity.shape[2:-1])
            if curr_vel_shape != prev_vel_shape:
                # Resize velocity via trilinear/bilinear interpolation
                resized_keyframes = []
                for t_k in range(self.n_time_steps):
                    vk = self.velocity[t_k]  # (1, *prev_spatial, dim)
                    # Move dim components to leading axis for resize
                    if self.dim == 3:
                        vk_cf = jnp.transpose(vk, (0, 4, 1, 2, 3))  # (1, 3, D, H, W)
                        vk_resized = jnp.stack([
                            jax.image.resize(vk_cf[0, c], curr_vel_shape, method='trilinear' if hasattr(jax.image, 'resize') else 'linear')
                            for c in range(self.dim)
                        ], axis=-1).reshape(1, *curr_vel_shape, self.dim)
                    else:
                        vk_resized = jax.image.resize(
                            vk.squeeze(0), (*curr_vel_shape, self.dim), method='bilinear'
                        ).reshape(1, *curr_vel_shape, self.dim)
                    resized_keyframes.append(vk_resized)
                self.velocity = jnp.stack(resized_keyframes, axis=0)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")

            # Reset momentum buffer for each level
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
                # Anti-aliasing pyramid smoothing (PyTorch parity)
                if smooth_pyramid:
                    aa_sigma = float(kwargs.get('aa_sigma', math.log2(level)))
                    fixed_smooth = separable_gaussian_filter_jax(
                        fixed_image.squeeze(0).squeeze(0), sigma=aa_sigma, spacing=None, sigma_mode='voxel'
                    )
                    moving_smooth = separable_gaussian_filter_jax(
                        moving_image.squeeze(0).squeeze(0), sigma=aa_sigma, spacing=None, sigma_mode='voxel'
                    )
                    # Reshape back and interpolate
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

            def tvf_loss_fn(vel):
                sim_loss = self.forward(curr_fixed, curr_moving, velocity=vel, multipoint_loss=multipoint_loss, lncc_window_size=2*lncc_radius+1)
                kinetic = jnp.mean(vel ** 2)
                return sim_loss + reg_weight * kinetic

            grad_tvf_fn = jax.grad(tvf_loss_fn)
            recent_losses = []

            for epoch in range(epochs):
                grad_raw = grad_tvf_fn(self.velocity)

                # Fluid regularization (smoothing velocity gradients)
                regularizer_mode = kwargs.get('regularizer_mode', kwargs.get('regularizer', 'sobolev'))
                alpha_sob = float(kwargs.get('sobolev_alpha', kwargs.get('alpha', sigma_voxel / 2.0)))
                if regularizer_mode == 'sobolev':
                    grad_smoothed = self._apply_sobolev_green_operator(grad_raw, fluid_sigma=sigma_voxel, alpha=alpha_sob, spacing=curr_spacing)
                elif regularizer_mode in ['dsti', 'dst1', 'dst_i']:
                    spatial_shape = list(grad_raw.shape[2:-1])
                    bmask_pre = self._create_boundary_mask(spatial_shape, border_width=4)
                    grad_tapered = grad_raw * bmask_pre
                    grad_smoothed = self._apply_dsti_green_operator(grad_tapered, fluid_sigma=sigma_voxel, alpha=alpha_sob, spacing=curr_spacing)
                elif sigma_voxel > 0:
                    spatial_shape = list(grad_raw.shape[2:-1])
                    min_spatial = min(spatial_shape)
                    if fast_smooth and min_spatial >= 32:
                        down_shape_sm = [max(8, s // 2) for s in spatial_shape]
                        smoothed_grads = []
                        for t in range(self.n_time_steps):
                            gt = grad_raw[t, 0]  # (*spatial, dim)
                            gt_down = jax.image.resize(gt, (*down_shape_sm, self.dim), method='bilinear')
                            gt_sm = separable_gaussian_filter_jax(gt_down, sigma=sigma_voxel, spacing=None, sigma_mode='voxel')
                            gt_up = jax.image.resize(gt_sm, (*spatial_shape, self.dim), method='bilinear')
                            smoothed_grads.append(gt_up[None])
                        grad_smoothed = jnp.stack(smoothed_grads, axis=0)
                    else:
                        smoothed_grads = []
                        for t in range(self.n_time_steps):
                            gt = grad_raw[t, 0]
                            gt_sm = separable_gaussian_filter_jax(gt, sigma=sigma_voxel, spacing=None, sigma_mode='voxel')
                            smoothed_grads.append(gt_sm[None])
                        grad_smoothed = jnp.stack(smoothed_grads, axis=0)
                else:
                    grad_smoothed = grad_raw

                # Apply boundary mask taper to velocity gradients (PyTorch parity)
                spatial_shape = list(grad_raw.shape[2:-1])
                bmask = self._create_boundary_mask(spatial_shape, border_width=4)
                grad_smoothed = grad_smoothed * bmask

                if opt_type == 'cfl':
                    # ITK-style CFL: normalize in voxel space (matching PyTorch exactly)
                    sp_j = jnp.array(curr_spacing)
                    grad_voxel = grad_smoothed / sp_j  # convert to voxel units
                    max_g_voxel = jnp.max(jnp.sqrt(jnp.sum(grad_voxel**2, axis=-1)))

                    if max_g_voxel > 1e-8:
                        update = (cfl_step_val / max_g_voxel) * grad_smoothed
                        if cfl_momentum > 0:
                            if momentum_buffer is None:
                                momentum_buffer = update
                            else:
                                momentum_buffer = cfl_momentum * momentum_buffer + update
                            
                            bias_corr = 1.0 - (cfl_momentum ** (epoch + 1))
                            corrected_buf = momentum_buffer / jnp.maximum(bias_corr, 1e-8)
                            self.velocity = self.velocity - (corrected_buf * (1.0 - cfl_momentum))
                        else:
                            self.velocity = self.velocity - update
                else:
                    self.velocity, m_vel, v_vel, t_vel = adam_step(
                        self.velocity, grad_smoothed, m_vel, v_vel, t_vel, lr=lr
                    )

                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                if elastic_sigma_voxel > 0:
                    smoothed_vel = []
                    for t in range(self.n_time_steps):
                        vt = self.velocity[t, 0]
                        vt_sm = separable_gaussian_filter_jax(vt, sigma=elastic_sigma_voxel, spacing=None, sigma_mode='voxel')
                        smoothed_vel.append(vt_sm[None])
                    self.velocity = jnp.stack(smoothed_vel, axis=0)

                cfl_max_val = float(kwargs.get('cfl_max', 0.40))
                if cfl_max_val > 0:
                    sp_j = jnp.array(curr_spacing)
                    vel_vox = self.velocity / sp_j
                    max_vox = jnp.max(jnp.sqrt(jnp.sum(vel_vox**2, axis=-1)))
                    if max_vox > cfl_max_val:
                        self.velocity = self.velocity * (cfl_max_val / (max_vox + 1e-8))

                if kwargs.get('antisymmetric', kwargs.get('antisymmetry', self.antisymmetric)):
                    self.velocity = self.project_antisymmetric(self.velocity)

                # Convergence checking (every 5 epochs to match PyTorch)
                if epoch % 5 == 0 or epoch == epochs - 1:
                    loss_val = float(self.forward(curr_fixed, curr_moving, velocity=self.velocity))
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

        # Ensure velocity is at full image resolution after fit completes
        final_vel_shape = tuple(self.velocity.shape[2:-1])
        if final_vel_shape != tuple(self.image_shape):
            resized_keyframes = []
            for t_k in range(self.n_time_steps):
                vk = self.velocity[t_k]
                if self.dim == 3:
                    vk_resized = jnp.stack([
                        jax.image.resize(vk[0, :, :, :, c], self.image_shape, method='trilinear' if hasattr(jax.image, 'resize') else 'linear')
                        for c in range(self.dim)
                    ], axis=-1).reshape(1, *self.image_shape, self.dim)
                else:
                    vk_resized = jax.image.resize(
                        vk.squeeze(0), (*self.image_shape, self.dim), method='bilinear'
                    ).reshape(1, *self.image_shape, self.dim)
                resized_keyframes.append(vk_resized)
            self.velocity = jnp.stack(resized_keyframes, axis=0)
            if verbose:
                print(f"  Final velocity upsample: {list(final_vel_shape)} → {list(self.image_shape)}")

    def get_forward_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=0 to t=1 in physical space.
        """
        return self.integrate(0.0, 1.0, image_shape=image_shape)

    def get_inverse_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=1 to t=0 in physical space.
        """
        return self.integrate(1.0, 0.0, image_shape=image_shape)
