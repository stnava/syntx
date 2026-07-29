import math
import numpy as np
import jax
import jax.numpy as jnp
from .syn_jax import (
    get_physical_grid_jax,
    physical_to_normalized_jax,
    physical_to_normalized_jax_cached,
    separable_gaussian_filter_jax,
    get_affine_matrix_jax,
    grid_to_physical_affine_jax,
    local_ncc_loss_nd_jax,
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


def lars_step_jax(param, grad, lr=0.1, trust_coefficient=0.05, eps=1e-8):
    param_norm = jnp.linalg.norm(param)
    grad_norm = jnp.linalg.norm(grad)
    param_norm_effective = jnp.maximum(param_norm, 1.0)
    trust_ratio = jnp.where(
        grad_norm > 0.0,
        trust_coefficient * param_norm_effective / (grad_norm + eps),
        1.0
    )
    param_next = param - lr * trust_ratio * grad
    return param_next


def adam_step_dict(params, grads, m_dict, v_dict, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    t_next = t + 1
    new_params = {}
    new_m = {}
    new_v = {}
    for k in params:
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
        fluid_sigma=0.5,
        elastic_sigma=0.0,
        transform_type='Affine',
        solver='euler',
        integration_steps_per_interval=1
    ):
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps

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

    def _get_metadata_tensors(self, target_shape, curr_spacing):
        spacing_rev = tuple(reversed(curr_spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(self.direction)[::-1, ::-1])

        shape_t = jnp.array(target_shape, dtype=jnp.float32)
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

        # Pre-upsample ALL velocity keyframes to target_shape ONCE upfront,
        # eliminating redundant interpolate_jax calls inside the integration loop.
        if tuple(velocity_cf.shape[3:]) == target_shape:
            velocity_fine_cf = velocity_cf
        else:
            v_fine_list = []
            for t_idx in range(self.n_time_steps):
                v_fine = interpolate_jax(velocity_cf[t_idx], target_shape, self.dim)
                v_fine_list.append(v_fine)
            velocity_fine_cf = jnp.stack(v_fine_list, axis=0)

        def interpolate_velocity_fine(t):
            T = self.n_time_steps
            if T == 1:
                return velocity_fine_cf[0]
            t_scaled = t * (T - 1)
            idx_lower = jnp.clip(jnp.floor(t_scaled).astype(jnp.int32), 0, T - 1)
            idx_upper = jnp.clip(jnp.ceil(t_scaled).astype(jnp.int32), 0, T - 1)

            weight_upper = t_scaled - idx_lower
            weight_lower = 1.0 - weight_upper

            return weight_lower * velocity_fine_cf[idx_lower] + weight_upper * velocity_fine_cf[idx_upper]

        def eval_v(t, current_phi):
            v_fine_cf_t = interpolate_velocity_fine(t)
            phi_norm = physical_to_normalized_jax_cached(
                current_phi, shape_t, spacing_t, origin_t, direction_t
            )
            v_sampled_cf = jax_grid_sample(v_fine_cf_t, phi_norm, mode='bilinear', padding_mode='border')
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

    def _resize_velocity(self, new_shape):
        """
        Resize velocity parameter tensor in JAX across pyramid levels using interpolate_jax.
        """
        new_shape = tuple(new_shape)
        is_6d = (self.velocity.ndim == self.dim + 3)
        old_shape = tuple(self.velocity.shape[2:-1]) if is_6d else tuple(self.velocity.shape[2:])
        if new_shape == old_shape:
            return

        vel_squeezed = self.velocity.squeeze(1) if is_6d else self.velocity
        vel_cf = jnp.moveaxis(vel_squeezed, -1, 1)
        new_cf = interpolate_jax(vel_cf, new_shape, self.dim)
        new_cl = jnp.moveaxis(new_cf, 1, -1)
        self.velocity = new_cl[:, None, ...] if is_6d else new_cl

    def forward(self, fixed_image, moving_image, velocity=None, affine_params=None, multipoint_loss=[0.5], lncc_window_size=5):
        """
        Registration forward pass supporting arbitrary multi-point LNCC evaluation timepoints t in [0, 1] in JAX.
        Default: multipoint_loss = [0.5] (SyNTVF geodesic midpoint evaluation).
        Triplet: multipoint_loss = [0.0, 0.5, 1.0] (anchors fixed t=0, midpoint t=0.5, and moving t=1 space).
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
            target_shape, curr_spacing, self.origin, self.direction
        )

        losses = []
        for t_k in eval_points:
            t_k = float(t_k)
            if abs(t_k - 0.0) < 1e-5:
                # Fixed Space (t=0.0: warp moving to fixed)
                phi_0_to_1 = self.integrate(0.0, 1.0, velocity=velocity, image_shape=target_shape)
                phi_moving_affine_end = (phys_grid + phi_0_to_1) @ M_phys.T + t_phys
                phi_norm_end = physical_to_normalized_jax_cached(
                    phi_moving_affine_end, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = jax_grid_sample(moving_image, phi_norm_end, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=lncc_window_size))
            elif abs(t_k - 1.0) < 1e-5:
                # Moving Space (t=1.0: warp fixed to moving)
                phi_1_to_0 = self.integrate(1.0, 0.0, velocity=velocity, image_shape=target_shape)
                phi_fixed_norm_end = physical_to_normalized_jax_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped = jax_grid_sample(fixed_image, phi_fixed_norm_end, mode='bilinear', padding_mode='zeros')

                phi_moving_identity = phys_grid @ M_phys.T + t_phys
                phi_moving_identity_norm = physical_to_normalized_jax_cached(
                    phi_moving_identity, shape_t, spacing_t, origin_t, direction_t
                )
                moving_affine = jax_grid_sample(moving_image, phi_moving_identity_norm, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_warped, moving_affine, window_size=lncc_window_size))
            else:
                # Midpoint or Intermediate Space t_k
                phi_tk_to_fixed = self.integrate(t_k, 0.0, velocity=velocity, image_shape=target_shape)
                phi_tk_to_moving = self.integrate(t_k, 1.0, velocity=velocity, image_shape=target_shape)

                phi_fixed_norm_tk = physical_to_normalized_jax_cached(
                    phys_grid + phi_tk_to_fixed, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped_tk = jax_grid_sample(fixed_image, phi_fixed_norm_tk, mode='bilinear', padding_mode='zeros')

                phi_moving_affine_tk = (phys_grid + phi_tk_to_moving) @ M_phys.T + t_phys
                phi_moving_norm_tk = physical_to_normalized_jax_cached(
                    phi_moving_affine_tk, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped_tk = jax_grid_sample(moving_image, phi_moving_norm_tk, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_warped_tk, moving_warped_tk, window_size=lncc_window_size))

        loss_val = jnp.mean(jnp.stack(losses))
        return loss_val

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[100, 100, 50],
        affine_epochs=100,
        similarity_metric='lncc',
        lncc_radius=2,
        lr=1.0,
        reg_weight=0.0,
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
        # Auto-extract physical space metadata directly from fixed_image
        from .tvf import extract_image_metadata
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image, dim=self.dim)
        if f_spacing is not None and fixed_spacing is None: self.spacing = f_spacing
        if f_origin is not None and fixed_origin is None: self.origin = f_origin
        if f_direction is not None and fixed_direction is None: self.direction = f_direction
        if f_shape is not None and self.image_shape != f_shape: self.image_shape = f_shape

        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        if hasattr(fixed_image, 'numpy'):
            fixed_image = fixed_image.numpy()
        elif hasattr(fixed_image, 'detach'):
            fixed_image = fixed_image.detach().cpu().numpy()
        fixed_image = jnp.array(fixed_image, dtype=jnp.float32)

        if hasattr(moving_image, 'numpy'):
            moving_image = moving_image.numpy()
        elif hasattr(moving_image, 'detach'):
            moving_image = moving_image.detach().cpu().numpy()
        moving_image = jnp.array(moving_image, dtype=jnp.float32)

        f_min, f_max = jnp.min(fixed_image), jnp.max(fixed_image)
        if f_max > f_min:
            fixed_image = (fixed_image - f_min) / (f_max - f_min + 1e-8)

        m_min, m_max = jnp.min(moving_image), jnp.max(moving_image)
        if m_max > m_min:
            moving_image = (moving_image - m_min) / (m_max - m_min + 1e-8)

        while fixed_image.ndim < self.dim + 2:
            fixed_image = fixed_image[None, ...]
        while moving_image.ndim < self.dim + 2:
            moving_image = moving_image[None, ...]

        # 1. Optimize affine pre-alignment first
        if affine_epochs > 0:
            # Initial alignment via dynamic FOV & Foreground CoM evaluation (matching PyTorch / syntx.syn)
            dim = self.dim
            fixed_spacing_t = jnp.array(self.spacing, dtype=jnp.float32)
            fixed_origin_t = jnp.array(self.origin, dtype=jnp.float32)
            fixed_direction_t = jnp.array(self.direction, dtype=jnp.float32)

            moving_spacing_t = jnp.array(moving_spacing if moving_spacing is not None else self.spacing, dtype=jnp.float32)
            moving_origin_t = jnp.array(moving_origin if moving_origin is not None else self.origin, dtype=jnp.float32)
            moving_direction_t = jnp.array(moving_direction if moving_direction is not None else self.direction, dtype=jnp.float32)

            Nx_t = jnp.array(fixed_image.shape[2:], dtype=jnp.float32)
            Ny_t = jnp.array(moving_image.shape[2:], dtype=jnp.float32)

            Dx_t, Sx_t, Ox_t = fixed_direction_t, fixed_spacing_t, fixed_origin_t
            Dy_t, Sy_t, Oy_t = moving_direction_t, moving_spacing_t, moving_origin_t

            com_fixed_fov = Ox_t + Dx_t @ (Sx_t * ((Nx_t - 1) / 2.0))
            com_moving_fov = Oy_t + Dy_t @ (Sy_t * ((Ny_t - 1) / 2.0))

            grid_idx_x = jnp.stack(jnp.meshgrid(*[jnp.arange(n, dtype=jnp.float32) for n in fixed_image.shape[2:]], indexing='ij'), axis=-1)
            grid_phys_x = Ox_t + (grid_idx_x * Sx_t) @ Dx_t.T
            weights_x = jnp.maximum(fixed_image.squeeze(0).squeeze(0), 0.0)
            sum_w_x = jnp.sum(weights_x)
            com_fixed_fg = jnp.sum(grid_phys_x * weights_x[..., None], axis=tuple(range(dim))) / sum_w_x if sum_w_x > 0 else com_fixed_fov

            grid_idx_y = jnp.stack(jnp.meshgrid(*[jnp.arange(n, dtype=jnp.float32) for n in moving_image.shape[2:]], indexing='ij'), axis=-1)
            grid_phys_y = Oy_t + (grid_idx_y * Sy_t) @ Dy_t.T
            weights_y = jnp.maximum(moving_image.squeeze(0).squeeze(0), 0.0)
            sum_w_y = jnp.sum(weights_y)
            com_moving_fg = jnp.sum(grid_phys_y * weights_y[..., None], axis=tuple(range(dim))) / sum_w_y if sum_w_y > 0 else com_moving_fov

            t_fov = com_moving_fov - com_fixed_fov
            t_fg = com_moving_fg - com_fixed_fg

            down_shape = tuple([max(16, int(s // 4)) for s in self.image_shape])
            down_spacing = [(s * orig) / d for s, orig, d in zip(self.spacing, self.image_shape, down_shape)]
            I_down = interpolate_jax(fixed_image, down_shape, dim)
            J_down = interpolate_jax(moving_image, down_shape, dim)
            X_down = get_physical_grid_jax(down_shape, down_spacing, self.origin, self.direction)

            def eval_translation(t_candidate):
                t_candidate_zyx = jnp.flip(t_candidate, axis=-1)
                y_phys = X_down + t_candidate_zyx
                shape_mt = jnp.array(moving_image.shape[2:], dtype=jnp.float32)
                y_norm = physical_to_normalized_jax_cached(y_phys, shape_mt, moving_spacing_t, moving_origin_t, moving_direction_t)
                J_warped = jax_grid_sample(J_down, y_norm, padding_mode='border')
                return float(local_ncc_loss_nd_jax(I_down, J_warped))

            loss_fov = eval_translation(t_fov)
            loss_fg = eval_translation(t_fg)
            best_t = t_fov if loss_fov < loss_fg else t_fg

            H_x = jnp.eye(dim + 1)
            H_x = H_x.at[:dim, :dim].set(Dx_t @ jnp.diag(Sx_t) @ jnp.diag((Nx_t - 1) / 2.0))
            H_x = H_x.at[:dim, dim].set(com_fixed_fov)

            H_y = jnp.eye(dim + 1)
            H_y = H_y.at[:dim, :dim].set(Dy_t @ jnp.diag(Sy_t) @ jnp.diag((Ny_t - 1) / 2.0))
            H_y = H_y.at[:dim, dim].set(com_moving_fov)

            T_phys = jnp.eye(dim + 1)
            T_phys = T_phys.at[:dim, dim].set(best_t)

            T_init = jnp.linalg.inv(H_y) @ T_phys @ H_x
            self.affine_params['T_init'] = T_init

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
                phi_moving_affine = phys_grid @ M_phys.T + t_phys
                shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(self.image_shape, self.spacing)

                phi_moving_norm = physical_to_normalized_jax_cached(
                    phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = jax_grid_sample(moving_image, phi_moving_norm, mode='bilinear', padding_mode='zeros')
                return local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=2*lncc_radius+1)

            grad_aff_fn = jax.grad(affine_loss_fn)

            affine_lr = float(kwargs.get('affine_lr', 1e-2))
            for epoch in range(affine_epochs):
                grads_aff = grad_aff_fn(self.affine_params)
                self.affine_params, m_aff, v_aff, t_aff = adam_step_dict(
                    self.affine_params, grads_aff, m_aff, v_aff, t_aff, lr=affine_lr
                )
                self.affine_params = clamp_affine_params_jax(self.affine_params)

        # 2. Optimize velocity field across pyramid levels
        if verbose: print("Optimizing TVF in JAX...")
        opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'adam')).lower()
        trust_coeff = kwargs.get('trust_coefficient', kwargs.get('trust', 0.05))
        fluid_sigmas_input = kwargs.get('fluid_sigmas', kwargs.get('fluid_sigma', kwargs.get('flow_sigma', self.fluid_sigma)))
        elastic_sigmas_input = kwargs.get('elastic_sigmas', kwargs.get('elastic_sigma', kwargs.get('total_sigma', self.elastic_sigma)))
        convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        convergence_window = kwargs.get('convergence_window', 10)
        lncc_ws = kwargs.get('lncc_window_size', 2 * lncc_radius + 1)
        multipoint_loss = kwargs.get('multipoint_loss', [0.5])
        sigma_mode = kwargs.get('sigma_mode', 'voxel')

        max_vel_shape = self.velocity_shape
        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue

            # --- Pyramid-proportional velocity grid ---
            curr_vel_shape = tuple(max(8, int(v // level)) for v in max_vel_shape)
            is_6d = (self.velocity.ndim == self.dim + 3)
            prev_vel_shape = tuple(self.velocity.shape[2:-1]) if is_6d else tuple(self.velocity.shape[2:])

            if curr_vel_shape != prev_vel_shape:
                self._resize_velocity(curr_vel_shape)
                if verbose:
                    print(f"  Velocity grid: {list(prev_vel_shape)} → {list(curr_vel_shape)}")

            # Initialize fresh optimizer state for this level matching PyTorch
            m_vel = jnp.zeros_like(self.velocity)
            v_vel = jnp.zeros_like(self.velocity)
            t_vel = 0

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

            sigma_voxel = math.sqrt(curr_fluid_sig) if curr_fluid_sig > 0 else 0.0
            elastic_sigma_voxel = math.sqrt(curr_elastic_sig) if curr_elastic_sig > 0 else 0.0

            if verbose:
                print(f"Level {level}: {epochs} max epochs (fluid_sigma={curr_fluid_sig:.2f}, elastic_sigma={curr_elastic_sig:.2f})")

            if level > 1:
                down_shape = tuple([max(8, s // level) for s in self.image_shape])
                aa_sigma = math.log2(level)
                if aa_sigma > 0:
                    fixed_cl = jnp.moveaxis(fixed_image, 1, -1)
                    moving_cl = jnp.moveaxis(moving_image, 1, -1)
                    fixed_smooth_cl = separable_gaussian_filter_jax(fixed_cl, sigma=aa_sigma, spacing=None, sigma_mode='voxel')
                    moving_smooth_cl = separable_gaussian_filter_jax(moving_cl, sigma=aa_sigma, spacing=None, sigma_mode='voxel')
                    fixed_smooth = jnp.moveaxis(fixed_smooth_cl, -1, 1)
                    moving_smooth = jnp.moveaxis(moving_smooth_cl, -1, 1)
                else:
                    fixed_smooth = fixed_image
                    moving_smooth = moving_image
                curr_fixed = interpolate_jax(fixed_smooth, down_shape, self.dim)
                curr_moving = interpolate_jax(moving_smooth, down_shape, self.dim)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image

            def tvf_loss_fn(vel):
                sim_loss = self.forward(curr_fixed, curr_moving, velocity=vel, multipoint_loss=multipoint_loss, lncc_window_size=lncc_ws)
                kinetic = jnp.mean(vel ** 2)
                return sim_loss + reg_weight * kinetic, sim_loss

            val_and_grad_fn = jax.value_and_grad(tvf_loss_fn, has_aux=True)
            recent_losses = []

            for epoch in range(epochs):
                (total_loss, sim_loss), grad_raw = val_and_grad_fn(self.velocity)

                # Fluid regularization & Antisymmetric Geodesic Projection (Rule 11)
                grad_cl = grad_raw.squeeze(1) if grad_raw.ndim == (self.dim + 3) else grad_raw
                # 1. Antisymmetric Geodesic Projection on raw gradients
                grad_flip = jnp.flip(grad_cl, axis=0)
                grad_proj = 0.5 * (grad_cl - grad_flip)
                
                # 2. Smooth projected antisymmetric gradient ONCE
                if sigma_voxel > 0:
                    grad_smooth_cl = separable_gaussian_filter_jax(
                        grad_proj, sigma=sigma_voxel, spacing=vel_spacing, sigma_mode=sigma_mode
                    )
                else:
                    grad_smooth_cl = grad_proj

                grad_smoothed = grad_smooth_cl[:, None, ...] if grad_raw.ndim == (self.dim + 3) else grad_smooth_cl

                if opt_type == 'lars':
                    self.velocity = lars_step_jax(
                        self.velocity, grad_smoothed, lr=lr, trust_coefficient=trust_coeff
                    )
                elif opt_type == 'cfl':
                    cfl_step = float(kwargs.get('cfl_voxels', kwargs.get('cfl', kwargs.get('step', lr))))
                    grad_cl = grad_smoothed.squeeze(1) if grad_smoothed.ndim == (self.dim + 3) else grad_smoothed
                    max_norm = jnp.max(jnp.sqrt(jnp.sum(grad_cl ** 2, axis=-1)))
                    curr_spacing_level = [
                        sp * (float(orig_s) / float(curr_s))
                        for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, curr_fixed.shape[2:])
                    ]
                    spacing_rev = tuple(reversed(curr_spacing_level))
                    sp_tensor = jnp.array(spacing_rev, dtype=jnp.float32)
                    step_update_cl = jnp.where(
                        max_norm > 1e-12,
                        (cfl_step / jnp.maximum(max_norm, 1e-8)) * grad_cl * sp_tensor,
                        jnp.zeros_like(grad_cl)
                    )
                    step_update = step_update_cl[:, None, ...] if self.velocity.ndim == (self.dim + 3) else step_update_cl
                    self.velocity = self.velocity - step_update
                elif opt_type == 'sgd':
                    self.velocity = self.velocity - lr * grad_smoothed
                else:
                    self.velocity, m_vel, v_vel, t_vel = adam_step(
                        self.velocity, grad_smoothed, m_vel, v_vel, t_vel, lr=lr
                    )

                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                if elastic_sigma_voxel > 0:
                    vel_cl = self.velocity.squeeze(1) if self.velocity.ndim == (self.dim + 3) else self.velocity
                    vel_smooth_cl = separable_gaussian_filter_jax(
                        vel_cl, sigma=elastic_sigma_voxel, spacing=vel_spacing, sigma_mode=sigma_mode
                    )
                    self.velocity = vel_smooth_cl[:, None, ...] if self.velocity.ndim == (self.dim + 3) else vel_smooth_cl
                    self.velocity = vel_smooth_cl[:, None, ...] if self.velocity.ndim == (self.dim + 3) else vel_smooth_cl

                vel_clamp_val = float(kwargs.get('velocity_clamp', kwargs.get('clamp', 50.0)))
                self.velocity = jnp.clip(self.velocity, -vel_clamp_val, vel_clamp_val)

                # Convergence checking (every 5 epochs matching PyTorch)
                if epoch % 5 == 0 or epoch == epochs - 1:
                    loss_val = float(sim_loss)
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

        # Ensure velocity is at full (max) resolution after fit completes
        is_6d = (self.velocity.ndim == self.dim + 3)
        final_vel_shape = tuple(self.velocity.shape[2:-1]) if is_6d else tuple(self.velocity.shape[2:])
        if final_vel_shape != max_vel_shape:
            self._resize_velocity(max_vel_shape)
            if verbose:
                print(f"  Final velocity upsample: {list(final_vel_shape)} → {list(max_vel_shape)}")

    def get_forward_warp(self, image_shape=None):
        """
        Returns displacement field integrating from t=0 to t=1 in physical space.
        """
        return self.integrate(0.0, 1.0, image_shape=image_shape)

    def get_warped_image(self, moving_image):
        """
        Applies forward displacement warp to moving_image and returns the warped image array.
        """
        phi_fwd = self.get_forward_warp()
        target_shape = tuple(moving_image.shape[2:])
        phys_grid = get_physical_grid_jax(
            target_shape, self.spacing, self.origin, self.direction
        )
        phi_phys = phys_grid + phi_fwd
        phi_norm = physical_to_normalized_jax(
            phi_phys, target_shape, self.spacing, self.origin, self.direction
        )
        return jax_grid_sample(moving_image, phi_norm, mode='bilinear', padding_mode='zeros')

    def get_inverse_warp(self, image_shape=None):

        """
        Returns displacement field integrating from t=1 to t=0 in physical space.
        """
        return self.integrate(1.0, 0.0, image_shape=image_shape)
