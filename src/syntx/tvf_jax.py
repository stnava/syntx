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
        fluid_sigma=1.0,
        elastic_sigma=0.0,
        transform_type='Affine',
        solver='euler',
        integration_steps_per_interval=1,
        antisymmetric=False
    ):
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.velocity_shape = tuple(velocity_shape)
        self.n_time_steps = n_time_steps
        self.antisymmetric = antisymmetric

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

    def forward(self, fixed_image, moving_image, velocity=None, affine_params=None, multipoint_loss=[0.5]):
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

        coord_perm = list(range(self.dim - 1, -1, -1))
        perm_idx = jnp.array(coord_perm, dtype=jnp.int32)
        M_phys_zyx = M_phys[perm_idx][:, perm_idx]
        t_phys_zyx = t_phys[perm_idx]

        losses = []
        for t_k in eval_points:
            t_k = float(t_k)
            if abs(t_k - 0.0) < 1e-5:
                # Fixed Space (t=0.0: warp moving to fixed)
                phi_0_to_1 = self.integrate(0.0, 1.0, velocity=velocity, image_shape=target_shape)
                phi_moving_affine_end = (phys_grid + phi_0_to_1) @ M_phys_zyx.T + t_phys_zyx
                phi_norm_end = physical_to_normalized_jax_cached(
                    phi_moving_affine_end, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped = jax_grid_sample(moving_image, phi_norm_end, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=9))
            elif abs(t_k - 1.0) < 1e-5:
                # Moving Space (t=1.0: warp fixed to moving)
                phi_1_to_0 = self.integrate(1.0, 0.0, velocity=velocity, image_shape=target_shape)
                phi_fixed_norm_end = physical_to_normalized_jax_cached(
                    phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped = jax_grid_sample(fixed_image, phi_fixed_norm_end, mode='bilinear', padding_mode='zeros')

                phi_moving_identity = phys_grid @ M_phys_zyx.T + t_phys_zyx
                phi_moving_identity_norm = physical_to_normalized_jax_cached(
                    phi_moving_identity, shape_t, spacing_t, origin_t, direction_t
                )
                moving_affine = jax_grid_sample(moving_image, phi_moving_identity_norm, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_warped, moving_affine, window_size=9))
            else:
                # Midpoint or Intermediate Space t_k
                phi_tk_to_fixed = self.integrate(t_k, 0.0, velocity=velocity, image_shape=target_shape)
                phi_tk_to_moving = self.integrate(t_k, 1.0, velocity=velocity, image_shape=target_shape)

                phi_fixed_norm_tk = physical_to_normalized_jax_cached(
                    phys_grid + phi_tk_to_fixed, shape_t, spacing_t, origin_t, direction_t
                )
                fixed_warped_tk = jax_grid_sample(fixed_image, phi_fixed_norm_tk, mode='bilinear', padding_mode='zeros')

                phi_moving_affine_tk = (phys_grid + phi_tk_to_moving) @ M_phys_zyx.T + t_phys_zyx
                phi_moving_norm_tk = physical_to_normalized_jax_cached(
                    phi_moving_affine_tk, shape_t, spacing_t, origin_t, direction_t
                )
                moving_warped_tk = jax_grid_sample(moving_image, phi_moving_norm_tk, mode='bilinear', padding_mode='zeros')
                losses.append(local_ncc_loss_nd_jax(fixed_warped_tk, moving_warped_tk, window_size=9))

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
        Multi-resolution optimization in JAX.
        """
        if fixed_spacing is not None: self.spacing = fixed_spacing
        if fixed_origin is not None: self.origin = fixed_origin
        if fixed_direction is not None: self.direction = fixed_direction

        fixed_image = jnp.array(fixed_image)
        moving_image = jnp.array(moving_image)

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
        cfl_momentum = float(kwargs.get('cfl_momentum', 0.9))
        momentum_buffer = None

        for idx, (level, epochs) in enumerate(zip(levels, epochs_per_level)):
            if epochs <= 0:
                continue

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
                curr_fixed = interpolate_jax(fixed_image, down_shape, self.dim)
                curr_moving = interpolate_jax(moving_image, down_shape, self.dim)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image

            def tvf_loss_fn(vel):
                sim_loss = self.forward(curr_fixed, curr_moving, velocity=vel, multipoint_loss=multipoint_loss)
                kinetic = jnp.mean(vel ** 2)
                return sim_loss + reg_weight * kinetic

            grad_tvf_fn = jax.grad(tvf_loss_fn)
            recent_losses = []

            for epoch in range(epochs):
                grad_raw = grad_tvf_fn(self.velocity)

                # Fluid regularization (smoothing velocity gradients)
                fast_smooth = kwargs.get('fast_smooth', True)
                if sigma_voxel > 0:
                    spatial_shape = list(grad_raw.shape[1:-1])
                    min_spatial = min(spatial_shape)
                    if fast_smooth and min_spatial >= 32:
                        down_shape = [max(8, s // 2) for s in spatial_shape]
                        smoothed_grads = []
                        for t in range(self.n_time_steps):
                            gt = grad_raw[t]
                            gt_down = jax.image.resize(gt, (*down_shape, self.dim), method='bilinear')
                            gt_sm = separable_gaussian_filter_jax(gt_down, sigma=sigma_voxel, spacing=None, sigma_mode='voxel')
                            gt_up = jax.image.resize(gt_sm, (*spatial_shape, self.dim), method='bilinear')
                            smoothed_grads.append(gt_up)
                        grad_smoothed = jnp.stack(smoothed_grads, axis=0)
                    else:
                        smoothed_grads = [
                            separable_gaussian_filter_jax(grad_raw[t], sigma=sigma_voxel, spacing=None, sigma_mode='voxel')
                            for t in range(self.n_time_steps)
                        ]
                        grad_smoothed = jnp.stack(smoothed_grads, axis=0)
                else:
                    grad_smoothed = grad_raw

                if opt_type == 'cfl':
                    max_g = jnp.max(jnp.linalg.norm(grad_smoothed, axis=-1))
                    cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', 0.5)))
                    sp_j = jnp.array(self.spacing)
                    step_mm = cfl_step_val * sp_j
                    norm_grad = jnp.where(max_g > 1e-8, grad_smoothed / max_g, jnp.zeros_like(grad_smoothed))
                    update = norm_grad * step_mm
                    if cfl_momentum > 0:
                        if momentum_buffer is None:
                            momentum_buffer = update
                        else:
                            momentum_buffer = cfl_momentum * momentum_buffer + update
                        self.velocity = self.velocity - momentum_buffer
                    else:
                        self.velocity = self.velocity - update
                else:
                    self.velocity, m_vel, v_vel, t_vel = adam_step(
                        self.velocity, grad_smoothed, m_vel, v_vel, t_vel, lr=lr
                    )

                # Elastic / Total Field Regularization (smoothing velocity field parameters post-step)
                if elastic_sigma_voxel > 0:
                    smoothed_vel = [
                        separable_gaussian_filter_jax(self.velocity[t], sigma=elastic_sigma_voxel, spacing=None, sigma_mode='voxel')
                        for t in range(self.n_time_steps)
                    ]
                    self.velocity = jnp.stack(smoothed_vel, axis=0)

                if kwargs.get('antisymmetric', kwargs.get('antisymmetry', self.antisymmetric)):
                    self.velocity = self.project_antisymmetric(self.velocity)

                # Convergence checking
                loss_val = float(self.forward(curr_fixed, curr_moving, velocity=self.velocity))
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
