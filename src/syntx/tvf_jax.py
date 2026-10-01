"""
JAX implementation of time-varying velocity field (TVF) registration: ``TVFModelJAX`` and the
stand-alone wrapper ``tvf_registration_jax``.

``syntx.tvf(..., backend='jax')`` (in ``syntx.tvf``) builds a ``TVFModelJAX`` and calls its
``fit``; that path accepts only ``regularizer='gaussian'`` with ``optimizer='cfl'``. This module
is a partial port of ``syntx.tvf.TVFModel``, not a mirror of it: defaults and several
behaviours differ (see the class docstring). Velocities and displacements are physical (mm),
tensor (z, y, x) order; geometry arguments (spacing / origin / direction) are ANTs (x, y, z).
Runs wherever JAX runs (CPU unless a JAX GPU backend is installed).
"""
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
    jax_grid_sample_image,
    interpolate_jax,
)


def clamp_affine_params_jax(params):
    """
    Return a copy of an affine-parameter dict with entries clipped to a sane range:
    ``scale`` and ``anisotropic_scale`` to [0.05, 20], ``shear`` to [-5, 5], ``omega`` to
    [-pi, pi]. Missing keys and any other keys (e.g. ``'T_init'``) are passed through.
    """
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
    """
    One bias-corrected Adam update of a single array (functional; nothing is mutated).

    Returns
    -------
    (param_next, m_next, v_next, t_next)
        Updated parameter, first / second moment estimates and step count ``t + 1``.
    """
    t_next = t + 1
    m_next = beta1 * m + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (grad ** 2)
    m_hat = m_next / (1.0 - (beta1 ** t_next))
    v_hat = v_next / (1.0 - (beta2 ** t_next))
    param_next = param - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return param_next, m_next, v_next, t_next


def adam_step_dict(params, grads, m_dict, v_dict, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    ``adam_step`` applied to every entry of a parameter dict sharing one step count ``t``.
    The ``'T_init'`` entry (a fixed initial matrix) is copied unchanged, with its moments.

    Returns
    -------
    (new_params, new_m, new_v, t + 1)
    """
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


def _resolve_alpha(kwargs, default):
    """Spectral regulariser strength from ``sobolev_alpha`` or ``alpha`` (first one that is
    given and not None), else ``default``."""
    for k in ('sobolev_alpha', 'alpha'):
        if kwargs.get(k) is not None:
            return float(kwargs[k])
    return float(default)


class TVFModelJAX:
    """
    JAX TVF model: an affine (``affine_params``) plus a time-varying velocity field stored as
    ``n_time_steps`` keyframes, interpolated in time with Catmull-Rom splines (linear for 2
    keyframes) and integrated with Euler or RK4.

    ``velocity`` is a plain ``jnp`` array of shape (T, 1, *velocity_shape, dim): physical
    velocities (mm per unit time), tensor (z, y, x) component order. ``affine_params`` is a
    dict with 'translation' (dim,), 'omega' (dim*(dim-1)/2,), 'scale' (1,),
    'anisotropic_scale' (dim,), 'shear' (dim*(dim-1)/2,) and optionally 'T_init'.

    Parameters
    ----------
    dim : {2, 3}
    image_shape, velocity_shape : tuple of int
        Fixed-image grid and initial velocity grid, tensor (z, y, x) order. ``fit`` resizes
        the velocity to ``max(8, s // level)`` per level and back to ``image_shape`` at the end.
    n_time_steps : int, default 3
    spacing, origin, direction : optional
        Fixed geometry, ANTs (x, y, z) order; defaults unit / zero / identity.
    moving_shape, moving_spacing, moving_origin, moving_direction : optional
        Moving geometry (shape in tensor order); default to the fixed values.
    fluid_sigma : float, default 0.0
        Fluid (gradient) smoothing strength. Interpreted by ``fit`` as a variance in voxels^2:
        the Gaussian / Sobolev width used is ``sqrt(fluid_sigma)`` voxels (PyTorch: sigma in mm).
    elastic_sigma : float, default 0.2
        Post-step velocity smoothing, same variance-in-voxels convention (``sqrt`` taken).
    transform_type : {'Affine', other}, default 'Affine'
        'Affine' optimises rotation, isotropic and anisotropic scale and shear; any other value
        (e.g. 'Translation', 'Rigid') gives rotation x isotropic scale + translation
        (``syn_jax.get_affine_matrix_jax``) -- 'Translation' still optimises rotation and scale.
    solver : {'euler', 'rk4'}, default 'euler'
    integration_steps_per_interval : int, default 1
        Integration uses exactly ``n_time_steps * integration_steps_per_interval`` steps over
        any time interval; no adaptive (CFL) step count.
    antisymmetric : bool, default True
        Different meaning from PyTorch: here it makes ``fit`` average each keyframe's gradient
        with the time-mirrored one (``project_symmetric``); it does not add evaluation times.
    image_grad_clip, velocity_clamp, cfl_max, use_analytical_gradients, **kwargs
        Stored or accepted but never read (``fit`` reads ``cfl_max`` from its own kwargs).

    Differences from ``syntx.tvf.TVFModel``
    ---------------------------------------
    PyTorch defaults ``integration_steps_per_interval=4`` (plus adaptive CFL steps),
    ``antisymmetric=False``, ``cfl_max=0.0``; the PyTorch model optimises no affine inside
    ``fit``, while ``TVFModelJAX.fit`` runs ``affine_epochs`` (default 100) Adam steps on the
    affine first. The similarity here is always local NCC (not squared; PyTorch default
    'cc2'), there is no path-energy term (a plain ``reg_weight * mean(v**2)`` instead), no
    ``losses`` history and no Jacobian / log-Jacobian output. Velocity keyframes are resized
    with ``jax.image.resize`` (half-pixel linear) rather than align-corners interpolation.
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
        use_analytical_gradients=False,
        **kwargs
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
        """Smooth (1, *spatial, 1) taper: 1 inside, a raised-cosine fall to 0 over
        ``border_width`` voxels (default ``max(1, min(shape) // 32)``) at every face; all ones
        when ``border_width <= 0``."""
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
        """
        FFT Sobolev smoothing: multiply the spectrum by ``1 / (1 + alpha |k|^2)^s`` (periodic
        boundaries; k in rad per ``spacing`` unit).

        ``m`` is (..., *spatial, dim). Returns ``m`` unchanged when ``fluid_sigma <= 0``, even
        if ``alpha`` is given; ``alpha`` defaults to ``fluid_sigma / 2``. ``spacing`` (default
        ``self.spacing``) is applied per tensor axis in the order given, so an ANTs (x, y, z)
        list is matched to tensor axes (z, y, x). ``border_width`` is passed to
        ``_create_boundary_mask`` (0 = no taper) and the mask is applied before and after.
        """
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
        """
        Sobolev smoothing with Dirichlet (zero) boundaries via a DST-I transform: spectrum
        multiplied by ``1 / (1 + alpha * sum_d lambda_d)^2`` with the discrete-Laplacian
        eigenvalues ``lambda_d = 4 sin^2(pi k / (2 (n_d + 1)))`` (voxel units).

        Returns ``m`` unchanged when ``fluid_sigma <= 0``; ``alpha`` defaults to
        ``fluid_sigma / 2``; ``spacing`` is unused. Same shape as ``m``.
        """
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

    def project_symmetric(self, grad):
        """
        Average each keyframe gradient with its time mirror:
        ``g(t_k) <- 0.5 * (g(t_k) + g(t_{T-1-k}))`` (flip along axis 0).

        Starting from a zero velocity this keeps the keyframes symmetric in time,
        v(t_k) = v(t_{T-1-k}); for T = 3 the first and last keyframes stay equal.
        """
        g_flipped = jnp.flip(grad, axis=0)
        return 0.5 * (grad + g_flipped)

    def _get_metadata_tensors(self, target_shape, curr_spacing):
        """(shape, spacing, origin, direction) as float32 arrays in tensor (z, y, x) order for
        ``physical_to_normalized_jax_cached``; ``curr_spacing`` is ANTs (x, y, z) order."""
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
        Integrate the velocity ODE dphi/dt = v(t, phi) from ``t_start`` to ``t_end``.

        Keyframes are first resized to ``image_shape`` (``jax.image.resize``, linear) and
        sampled with bilinear interpolation and border padding.

        Parameters
        ----------
        t_start, t_end : float
            Times in [0, 1]; ``t_end < t_start`` integrates backwards.
        velocity : array, optional
            (T, 1, *spatial, dim); default ``self.velocity``.
        n_steps : int, optional
            Default ``n_time_steps * integration_steps_per_interval`` for any interval length.
        image_shape : tuple of int, optional
            Output grid, tensor order; default ``self.image_shape``. Its spacing is
            ``spacing * N / n`` per axis (PyTorch: ``spacing * (N - 1) / (n - 1)``).

        Returns
        -------
        jnp.ndarray
            Displacement (1, *image_shape, dim), physical units, tensor (z, y, x) components.
        """
        from .syn_jax import jax_grid_sample, get_physical_grid_jax, physical_to_normalized_jax_cached
        import jax

        if velocity is None:
            velocity = self.velocity

        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape

        if n_steps is None:
            n_steps = self.n_time_steps * self.integration_steps_per_interval

        dt = (t_end - t_start) / max(1, n_steps)

        curr_spacing = [
            sp * (float(orig_s) / float(curr_s))
            for sp, orig_s, curr_s in zip(self.spacing, reversed(self.image_shape), reversed(target_shape))
        ]

        phys_grid = get_physical_grid_jax(
            target_shape, curr_spacing, self.origin, self.direction
        )
        phi_t = phys_grid

        shape_t, spacing_t, origin_t, direction_t = self._get_metadata_tensors(target_shape, curr_spacing)

        if velocity is None:
            velocity = self.velocity

        if self.dim == 2:
            velocity_cf = jnp.transpose(velocity, (0, 1, 4, 2, 3))
        else:
            velocity_cf = jnp.transpose(velocity, (0, 1, 5, 2, 3, 4))

        # === CRITICAL OPTIMIZATION ===
        # Pre-upsample ALL velocity keyframes to target_shape ONCE
        velocity_fine_cf_list = []
        for t_idx in range(self.n_time_steps):
            if tuple(velocity_cf[t_idx].shape[2:]) == tuple(target_shape):
                v_fine = velocity_cf[t_idx]
            else:
                v_fine = jax.image.resize(velocity_cf[t_idx], (1, self.dim, *target_shape), method='linear')
            velocity_fine_cf_list.append(v_fine)
            
        velocity_fine_cf = jnp.stack(velocity_fine_cf_list, axis=0) # (T, 1, dim, *target_shape)

        def _interpolate_velocity_fine_jax(t, vel_fine_cf):
            T = self.n_time_steps
            if T == 1:
                return vel_fine_cf[0]
            if T == 2:
                t_scaled = t
                return (1.0 - t_scaled) * vel_fine_cf[0] + t_scaled * vel_fine_cf[1]

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

            
            return c0 * vel_fine_cf[i0] + c1 * vel_fine_cf[i1] + c2 * vel_fine_cf[i2] + c3 * vel_fine_cf[i3]

        def eval_v(t, current_phi):
            v_fine_cf_t = _interpolate_velocity_fine_jax(t, velocity_fine_cf)
            phi_norm_t = physical_to_normalized_jax_cached(
                current_phi, shape_t, spacing_t, origin_t, direction_t
            )
            v_sampled_cf = jax_grid_sample(v_fine_cf_t, phi_norm_t, mode='bilinear', padding_mode='border')
            if self.dim == 2:
                return jnp.transpose(v_sampled_cf, (0, 2, 3, 1))
            else:
                return jnp.transpose(v_sampled_cf, (0, 2, 3, 4, 1))

        if self.solver == 'euler':
            # Use jax.lax.fori_loop for performance if possible, but python loop is okay for tracing
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
        Loss of the current transform (lower is better).

        At each time t in ``multipoint_loss`` the fixed image is pulled from t = 0 and the
        moving image (through the affine) from t = 1 to time t, and compared with local NCC
        (``local_ncc_loss_nd_jax``, not squared); the losses are averaged. When the times
        include both 0 and 1, ``inverse_identity_weight`` (attribute, default 0.05) times the
        mean squared inverse-consistency error of the 0->1 and 1->0 maps is added.

        Parameters
        ----------
        fixed_image, moving_image : array (1, 1, *spatial)
        velocity, affine_params : optional
            Default the model's own.
        multipoint_loss : list of float, bool or float, default [0.0, 0.5, 1.0]
            True = [0, 0.5, 1], False = [0.5]. PyTorch ``TVFModel.forward`` defaults to
            [0.0, 1.0].
        lncc_window_size : int, default 5

        Returns
        -------
        jnp scalar

        Notes
        -----
        The level spacing here is computed with the fixed spacing (x, y, z) zipped against the
        shapes in tensor order (no reversal, unlike ``integrate``), so for non-cubic grids the
        sampling grid and ``integrate``'s grid differ. The moving geometry is always the
        full-resolution one (``moving_shape``), also at coarse pyramid levels.
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
        compute_id_loss = (0.0 in eval_points) and (1.0 in eval_points)
        phi_0_to_1 = None
        phi_1_to_0 = None

        for t_k in eval_points:
            t_k = float(t_k)
            # Midpoint or Intermediate Space t_k
            phi_tk_to_fixed = self.integrate(t_k, 0.0, velocity=velocity, image_shape=target_shape)
            phi_tk_to_moving = self.integrate(t_k, 1.0, velocity=velocity, image_shape=target_shape)

            if abs(t_k - 0.0) < 1e-5:
                phi_0_to_1 = phi_tk_to_moving
            if abs(t_k - 1.0) < 1e-5:
                phi_1_to_0 = phi_tk_to_fixed

            phi_fixed_norm_tk = physical_to_normalized_jax_cached(
                phys_grid + phi_tk_to_fixed, shape_t, spacing_t, origin_t, direction_t
            )
            fixed_warped_tk = jax_grid_sample_image(fixed_image, phi_fixed_norm_tk, mode='bilinear', padding_mode='zeros')

            shape_m, spacing_m, origin_m, direction_m = self._get_moving_metadata_tensors()
            phi_moving_affine_tk = (phys_grid + phi_tk_to_moving) @ M_phys_zyx.T + t_phys_zyx
            phi_moving_norm_tk = physical_to_normalized_jax_cached(
                phi_moving_affine_tk, shape_m, spacing_m, origin_m, direction_m
            )
            moving_warped_tk = jax_grid_sample_image(moving_image, phi_moving_norm_tk, mode='bilinear', padding_mode='zeros')
            losses.append(local_ncc_loss_nd_jax(fixed_warped_tk, moving_warped_tk, window_size=lncc_window_size))

        sim_loss = sum(losses) / len(losses)

        if compute_id_loss and phi_0_to_1 is not None and phi_1_to_0 is not None:
            # Compute bidirectional composition penalties:
            phi_1_to_0_norm = physical_to_normalized_jax_cached(
                phys_grid + phi_1_to_0, shape_t, spacing_t, origin_t, direction_t
            )
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

            inv_id_weight = float(getattr(self, 'inverse_identity_weight', 0.05))
            return sim_loss + inv_id_weight * inv_id_loss

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
        **kwargs
    ):
        """
        Optimise the affine, then the velocity keyframes over an image pyramid. Updates
        ``self.affine_params`` and ``self.velocity`` in place; returns None.

        If fixed and moving arrays are identical (``allclose``, atol 1e-5) the velocity is set
        to zero and nothing else is done.

        Algorithm: (1) if ``initial_transform`` (kwarg) is given, ``parse_ants_affine``'s
        physical matrix is stored as ``T_init`` without conversion to grid coordinates (the
        ``syntx.tvf`` JAX path instead sets ``T_init`` itself after converting). (2)
        ``affine_epochs`` Adam steps (lr ``affine_lr``) on the affine at full resolution,
        metric ``aff_metric`` (Mattes MI by default, else local NCC). (3) For each level:
        resize the velocity to ``max(8, s // level)``, smooth (sigma ``aa_sigma``) and
        downsample the images, then per epoch take ``jax.grad`` of
        ``forward(...) + reg_weight * mean(v**2)``, smooth the gradient (regulariser), taper
        it at the border (4 voxels), optionally ``project_symmetric``, step (CFL or Adam),
        smooth the velocity (elastic), optionally equalise keyframe speeds and cap the
        magnitude. Every 5 epochs the loss is recomputed (with ``forward``'s default
        ``multipoint_loss`` and window 5, not the training ones) for early stopping.
        (4) Resize the velocity to ``image_shape``.

        Parameters
        ----------
        fixed_image, moving_image : array, torch tensor or ANTsImage-like
            Converted with ``.numpy()`` when available. Arrays of ndim ``dim`` get
            (1, 1) prepended. Must already be in tensor (z, y, x) order.
        levels : list of int, default [4, 2, 1]
        epochs_per_level : list of int, default [100, 100, 50]
            PyTorch ``TVFModel.fit`` default: [100, 100, 20].
        affine_epochs : int or list of int, default 100
            A list is summed. 0 skips the affine stage. PyTorch ``fit`` has no affine stage.
        similarity_metric : str, default 'lncc'
            Unused: the deformable loss is always local NCC.
        lncc_radius : int, default 4
            Window ``2 * lncc_radius + 1`` (deformable stage and the NCC affine metric).
        lr : float, default 0.15
            Adam learning rate when ``optimizer_type`` is not 'cfl' (PyTorch default 1.0).
        reg_weight : float, default 0.005
            Weight of ``mean(velocity**2)``.
        verbose : bool, default False
        fixed_spacing, fixed_origin, fixed_direction : optional
            Replace the model's fixed geometry.
        moving_spacing, moving_origin, moving_direction : optional
            Unused (the constructor's moving geometry is kept).
        **kwargs
            initial_transform; aff_metric ('mattes_mi'); mattes_bins / num_bins (32);
            sampling_percentage (0.2); affine_lr (1e-3); fluid_sigmas / fluid_sigma
            (``self.fluid_sigma``) and elastic_sigmas / elastic_sigma / total_sigma
            (``self.elastic_sigma``), scalars or per-level lists, square-rooted into voxel
            sigmas; regularizer_mode / regularizer ('sobolev'; 'dsti' / 'dst1' / 'dst_i'; any
            other value = Gaussian, at half resolution when fast_smooth and the smallest axis
            is >= 32); sobolev_alpha / alpha (default ``sqrt(fluid_sigma) / 2``; an explicit
            None raises TypeError); optimizer_type / optimizer ('cfl', else Adam); cfl_step /
            grad_step (0.35, voxels); cfl_momentum (0.9); multipoint_loss ([0.0, 1.0]; PyTorch
            fit default [0.5]); smooth_pyramid (True); fast_smooth (True); aa_sigma
            (log2(level)); antisymmetric / antisymmetry (``self.antisymmetric``);
            constant_speed (False; PyTorch default True) and constant_speed_relaxation (1.0);
            cfl_max (None = no cap); convergence_threshold (1e-6) and convergence_window (10).
            Unknown keys are ignored.
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
                moving_warped = jax_grid_sample_image(moving_image, phi_moving_norm, mode='bilinear', padding_mode='zeros')
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
                    self.affine_params, grads_aff, m_aff, v_aff, t_aff, lr=float(kwargs.get('affine_lr', 1e-3))
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
                alpha_sob = _resolve_alpha(kwargs, sigma_voxel / 2.0)
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

                if kwargs.get('antisymmetric', kwargs.get('antisymmetry', self.antisymmetric)):
                    grad_smoothed = self.project_symmetric(grad_smoothed)

                if opt_type == 'cfl':
                    # ITK-style CFL: normalize in voxel space (matching PyTorch exactly)
                    sp_j = jnp.array(curr_spacing)
                    grad_voxel = grad_smoothed / sp_j  # convert to voxel units
                    max_g_voxel = jnp.max(jnp.sqrt(jnp.sum(grad_voxel**2, axis=-1)))

                    if max_g_voxel > 1e-8:
                        update = (cfl_step_val / max_g_voxel) * grad_smoothed
                        if cfl_momentum > 0:
                            if momentum_buffer is None:
                                momentum_buffer = update * (1.0 - cfl_momentum)
                            else:
                                momentum_buffer = cfl_momentum * momentum_buffer + (1.0 - cfl_momentum) * update
                            
                            bias_corr = 1.0 - (cfl_momentum ** (epoch + 1))
                            corrected_buf = momentum_buffer / jnp.maximum(bias_corr, 1e-8)
                            self.velocity = self.velocity - corrected_buf
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

                # Constant speed constraint
                cs_enabled = kwargs.get('constant_speed', False)
                cs_relax = float(kwargs.get('constant_speed_relaxation', 1.0))
                if cs_enabled and self.n_time_steps > 1 and cs_relax > 0:
                    vel_data = self.velocity
                    spatial_axes = tuple(range(1, vel_data.ndim))
                    speeds = jnp.sqrt(jnp.sum(vel_data**2, axis=spatial_axes))  # (T,)
                    mean_speed = jnp.mean(speeds)
                    
                    def scale_fn(v_t, speed_t):
                        # Avoid div by zero
                        valid = speed_t > 1e-10
                        scale = jnp.where(valid, (1.0 - cs_relax) + cs_relax * (mean_speed / jnp.maximum(speed_t, 1e-10)), 1.0)
                        return v_t * scale
                    
                    self.velocity = jax.vmap(scale_fn)(self.velocity, speeds)

                cfl_max_val = kwargs.get('cfl_max', None)
                if cfl_max_val is not None and float(cfl_max_val) > 0:
                    cfl_max_val = float(cfl_max_val)
                    vel_norm = jnp.sqrt(jnp.sum(self.velocity**2, axis=-1))
                    max_vel_norm = jnp.max(vel_norm)
                    if max_vel_norm > cfl_max_val:
                        self.velocity = self.velocity * (cfl_max_val / (max_vel_norm + 1e-8))

                # Convergence checking (every 5 epochs to match PyTorch)
                if epoch % 5 == 0 or epoch == epochs - 1:
                    loss_val = float(self.forward(curr_fixed, curr_moving, velocity=self.velocity))
                    if verbose:
                        print(f"  [TVF Level {level}] Epoch {epoch+1}/{epochs}: loss={loss_val:.6f}")
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
        Forward displacement ``integrate(0, 1)``: (1, *image_shape, dim), physical units,
        tensor (z, y, x) components.
        """
        return self.integrate(0.0, 1.0, image_shape=image_shape)

    def get_inverse_warp(self, image_shape=None):
        """
        Inverse displacement ``integrate(1, 0)``: (1, *image_shape, dim), physical units,
        tensor (z, y, x) components.
        """
        return self.integrate(1.0, 0.0, image_shape=image_shape)


def tvf_registration_jax(
    fixed,
    moving,
    levels=[4, 2, 1],
    reg_iterations=[100, 100, 20],
    affine_iterations=100,
    similarity_metric='lncc',
    lncc_radius=2,
    grad_step=0.35,
    cfl_momentum=0.95,
    flow_sigma=0.4,
    total_sigma=0.5,
    reg_weight=0.005,
    initial_transform=None,
    verbose=False,
    multipoint_loss=[0.5],
    solver='euler',
    n_time_steps=3,
    **kwargs
):
    """
    Stand-alone JAX TVF registration of two ANTs images (separate from
    ``syntx.tvf(backend='jax')``, which has its own, different set-up).

    Both images are min-max normalised to [0, 1], converted with
    ``spatial.image_to_tensor`` (default ``to_zyx=False``, i.e. kept in ANTs (x, y, z) array
    order while the model treats axes as tensor (z, y, x)), a ``TVFModelJAX`` is fitted and the
    forward / inverse displacements are written to temporary NIfTI files
    (``tempfile.mktemp``, not deleted).

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
    levels : list of int, default [4, 2, 1]
    reg_iterations : list of int, default [100, 100, 20]
        Epochs per level.
    affine_iterations : int, default 100
        Adam steps of the affine stage in ``fit``.
    similarity_metric : str, default 'lncc'
        Passed to ``fit``, where it is unused.
    lncc_radius : int, default 2
    grad_step : float, default 0.35
        Used both as the CFL step (voxels) and as the Adam ``lr``.
    cfl_momentum : float, default 0.95
    flow_sigma : float, default 0.4
        Model ``fluid_sigma`` (variance in voxels^2; the default 'sobolev' regulariser then
        uses alpha = sqrt(0.4) / 2).
    total_sigma : float, default 0.5
        Model ``elastic_sigma`` (variance in voxels^2).
    reg_weight : float, default 0.005
    initial_transform : str or list of str, optional
        ANTs affine file(s). Parsed with ``parse_ants_affine`` and used as ``T_init`` without
        conversion to grid coordinates; also sets ``transform_type='Translation'`` (which
        still optimises rotation and scale). The files are appended to ``fwdtransforms``; the
        learned affine itself is not exported.
    verbose : bool, default False
    multipoint_loss : list of float, default [0.5]
    solver : {'euler', 'rk4'}, default 'euler'
    n_time_steps : int, default 3
    **kwargs
        Passed to ``TVFModelJAX.fit``.

    Returns
    -------
    dict
        ``'warpedmovout'`` (original ``moving`` resampled into the normalised fixed image by
        ``fwdtransforms``, linear), ``'fwdtransforms'`` ([forward warp file] + initial
        transform files), ``'invtransforms'`` ([inverse warp file], preceded by inverted
        initial transforms when given), ``'runtime'`` (seconds), ``'model'``.

    Raises
    ------
    ValueError
        If ``initial_transform`` is neither a string nor a list.
    """
    import jax.numpy as jnp
    from syntx.spatial import image_to_tensor
    import time
    import ants
    
    start_time = time.time()
    
    dim = fixed.dimension
    
    # 1. Image checks and normalizations
    # Normalize intensities to [0, 1]
    fi = fixed.clone()
    mi = moving.clone()
    
    # JAX doesn't have an ants wrapper built-in here, so we do it via numpy
    fi_arr = fi.numpy()
    fi_arr = (fi_arr - fi_arr.min()) / (fi_arr.max() - fi_arr.min() + 1e-8)
    fi = ants.from_numpy(fi_arr, origin=fi.origin, spacing=fi.spacing, direction=fi.direction)
    
    mi_arr = mi.numpy()
    mi_arr = (mi_arr - mi_arr.min()) / (mi_arr.max() - mi_arr.min() + 1e-8)
    mi = ants.from_numpy(mi_arr, origin=mi.origin, spacing=mi.spacing, direction=mi.direction)
    
    # 2. Convert to Tensors
    import torch
    fi_tensor = image_to_tensor(fi, device='cpu', dtype=torch.float32)
    mi_tensor = image_to_tensor(mi, device='cpu', dtype=torch.float32)
    
    fi_jax = jnp.array(fi_tensor.numpy())
    mi_jax = jnp.array(mi_tensor.numpy())
    
    # 3. Setup Model
    model = TVFModelJAX(
        dim=dim,
        image_shape=fi.shape,
        velocity_shape=fi.shape,
        n_time_steps=n_time_steps,
        spacing=fi.spacing,
        origin=fi.origin,
        direction=fi.direction,
        moving_shape=mi.shape,
        moving_spacing=mi.spacing,
        moving_origin=mi.origin,
        moving_direction=mi.direction,
        fluid_sigma=flow_sigma,
        elastic_sigma=total_sigma,
        transform_type='Affine' if initial_transform is None else 'Translation',
        solver=solver
    )
    
    # 4. Handle initial transform
    if initial_transform is not None:
        if isinstance(initial_transform, str):
            tx_list = [initial_transform]
        elif isinstance(initial_transform, list):
            tx_list = initial_transform
        else:
            raise ValueError("initial_transform must be a path string or list of paths")
            
        from syntx.syn import parse_ants_affine
        import numpy as np
        parsed_M, parsed_t = parse_ants_affine(tx_list, dim)
        if parsed_M is not None:
            T_mat = np.eye(dim + 1, dtype=np.float32)
            T_mat[:dim, :dim] = parsed_M
            T_mat[:dim, dim] = parsed_t
            model.T_init = jnp.array(T_mat)
    
    # 5. Fit
    model.fit(
        fixed_image=fi_jax,
        moving_image=mi_jax,
        levels=levels,
        epochs_per_level=reg_iterations,
        affine_epochs=affine_iterations,
        similarity_metric=similarity_metric,
        lncc_radius=lncc_radius,
        lr=grad_step,
        reg_weight=reg_weight,
        verbose=verbose,
        cfl_step=grad_step,
        cfl_momentum=cfl_momentum,
        multipoint_loss=multipoint_loss,
        **kwargs
    )
    
    # 6. Extract transforms
    phi_fwd = model.get_forward_warp()
    phi_inv = model.get_inverse_warp()
    
    from syntx.spatial import disp_tensor_to_itk
    import numpy as np
    
    fwd_disp = disp_tensor_to_itk(np.array(phi_fwd), fi)
    inv_disp = disp_tensor_to_itk(np.array(phi_inv), fi)
    
    import tempfile
    fwd_file = tempfile.mktemp(suffix='_fwd.nii.gz')
    inv_file = tempfile.mktemp(suffix='_inv.nii.gz')
    
    ants.image_write(fwd_disp, fwd_file)
    ants.image_write(inv_disp, inv_file)
    
    fwd_transforms = [fwd_file]
    inv_transforms = [inv_file]
    
    if initial_transform is not None:
        fwd_transforms.extend(tx_list)
        inv_tx_list = [ants.invert_ants_transform(t) for t in tx_list]
        inv_transforms = inv_tx_list + inv_transforms
    else:
        if model.T_init is not None:
            pass # TODO: export learned affine to file
            
    warped = ants.apply_transforms(
        fixed=fi,
        moving=moving,
        transformlist=fwd_transforms,
        interpolator='linear'
    )
    
    runtime = time.time() - start_time
    
    return {
        'warpedmovout': warped,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'runtime': runtime,
        'model': model
    }

