import math
import numpy as np
import jax
import jax.numpy as jnp

from .syn_jax import (
    get_physical_grid_jax,
    physical_to_normalized_jax,
    physical_to_normalized_jax_cached,
    separable_gaussian_filter_jax,
    local_ncc_loss_nd_jax,
    jax_grid_sample,
    _spatial_jacobian_nd_jax,
    get_affine_matrix_jax,
    grid_to_physical_affine_jax,
    interpolate_jax,
)
from .tvf import extract_image_metadata


def epdiff_advection_nd_jax(p, v):
    """
    Computes the coadjoint action ad_v^* p for EPDiff in JAX.
    
    ad_v^* p = (Dp) v + (Dv)^T p + p (div v)
    
    Args:
        p: Momentum field tensor (1, *spatial, dim)
        v: Velocity field tensor (1, *spatial, dim)
        
    Returns:
        ad_v_star_p: Coadjoint action tensor (1, *spatial, dim)
    """
    dim = p.shape[-1]
    
    # Compute spatial Jacobians: Dp and Dv have shape (1, *spatial, dim, dim)
    Dp = _spatial_jacobian_nd_jax(p)
    Dv = _spatial_jacobian_nd_jax(v)
    
    # 1. Advection term: (Dp) v -> sum_j (d p_i / d x_j) * v_j
    term_advection = jnp.einsum('...ij,...j->...i', Dp, v)
    
    # 2. Stretching term: (Dv)^T p -> sum_j (d v_j / d x_i) * p_j
    term_stretching = jnp.einsum('...ji,...j->...i', Dv, p)
    
    # 3. Expansion term: p * div(v) where div(v) = sum_j (d v_j / d x_j)
    div_v = jnp.diagonal(Dv, axis1=-2, axis2=-1).sum(axis=-1, keepdims=True)
    term_expansion = p * div_v
    
    ad_v_star_p = term_advection + term_stretching + term_expansion
    return ad_v_star_p


class GeodesicShootingModelJAX:
    """
    Euler-Poincaré Differential Equation (EPDiff) Geodesic Shooting Registration Model in JAX.
    Symmetrically mirrors PyTorch GeodesicShootingModel.
    
    Parameterizes ONLY ONE initial momentum vector field p0(x) at t=0.
    Shoots p0 forward in time via EPDiff:
      dp/dt = -ad_v^* p
      v(t) = separable_gaussian_filter(p(t), fluid_sigma)
      
    Guarantees strict geodesic paths in Diff(Omega).
    """
    def __init__(
        self,
        dim=2,
        image_shape=None,
        n_time_steps=8,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=1.0,
        fixed_image=None
    ):
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image, dim=dim)
        if image_shape is None:
            image_shape = f_shape if f_shape is not None else (128,) * dim
        if spacing is None and f_spacing is not None:
            spacing = f_spacing
        if origin is None and f_origin is not None:
            origin = f_origin
        if direction is None and f_direction is not None:
            direction = f_direction
            
        self.dim = dim
        self.image_shape = tuple(image_shape)
        self.n_time_steps = n_time_steps
        self.spacing = list(spacing) if spacing is not None else [1.0] * dim
        self.origin = list(origin) if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = np.array(direction, dtype=np.float32).tolist()
        else:
            self.direction = np.eye(dim, dtype=np.float32).tolist()
        self.fluid_sigma = fluid_sigma
        
        num_rot = dim * (dim - 1) // 2
        self.affine_params = {
            'translation': jnp.zeros(dim, dtype=jnp.float32),
            'omega': jnp.zeros(num_rot, dtype=jnp.float32),
            'scale': jnp.ones(1, dtype=jnp.float32),
            'anisotropic_scale': jnp.ones(dim, dtype=jnp.float32),
            'shear': jnp.zeros(num_rot, dtype=jnp.float32)
        }
        self.transform_type = 'Affine'
        
        # Single initial momentum parameter at t=0: shape (1, *image_shape, dim)
        self.p0 = jnp.zeros((1, *self.image_shape, self.dim), dtype=jnp.float32)

    def shoot(self, p0=None, n_steps=None, fluid_sigma=None, image_shape=None, sigma_mode='voxel'):
        """
        Integrates initial momentum p0 forward from t=0 to t=1 using EPDiff in JAX.
        """
        if p0 is None:
            p0 = self.p0
        if n_steps is None:
            n_steps = self.n_time_steps
        if fluid_sigma is None:
            fluid_sigma = self.fluid_sigma
            
        dt = 1.0 / float(n_steps)
        target_shape = tuple(image_shape) if image_shape is not None else tuple(p0.shape[1:-1])
        
        phys_grid = get_physical_grid_jax(
            target_shape, self.spacing, self.origin, self.direction
        )
        spacing_rev = tuple(reversed(self.spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(self.direction)[::-1, ::-1])

        shape_t = jnp.array(target_shape, dtype=jnp.float32)
        spacing_t = jnp.array(spacing_rev, dtype=jnp.float32)
        origin_t = jnp.array(origin_rev, dtype=jnp.float32)
        direction_t = jnp.array(direction_rev, dtype=jnp.float32)
        
        p_t = p0
        phi_t = phys_grid
        
        p_history = [p_t]
        v_history = []
        
        p_spacing = [sp * (float(orig_s) / float(curr_s)) for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)] if sigma_mode == 'physical' else None

        for step in range(n_steps):
            fluid_sigma_val = math.sqrt(fluid_sigma) if fluid_sigma > 0 else 0.0
            v_t = separable_gaussian_filter_jax(p_t, sigma=fluid_sigma_val, spacing=p_spacing, sigma_mode=sigma_mode)
            v_history.append(v_t)
            
            dp_dt = -epdiff_advection_nd_jax(p_t, v_t)
            
            dp_norm = jnp.sqrt(jnp.sum(dp_dt ** 2, axis=-1, keepdims=True) + 1e-12)
            max_dp = 2.0
            dp_dt = jnp.where(dp_norm > max_dp, dp_dt * (max_dp / (dp_norm + 1e-8)), dp_dt)
            
            p_t = p_t + dt * dp_dt
            p_history.append(p_t)
            
            phi_norm = physical_to_normalized_jax_cached(
                phi_t, shape_t, spacing_t, origin_t, direction_t
            )
            v_t_cf = jnp.moveaxis(v_t, -1, 1)
            v_sampled_cf = jax_grid_sample(v_t_cf, phi_norm, mode='bilinear', padding_mode='border')
            v_sampled = jnp.moveaxis(v_sampled_cf, 1, -1)
            phi_t = phi_t + dt * v_sampled
            
        disp_fwd = phi_t - phys_grid
        return disp_fwd, {"p_history": p_history, "v_history": v_history}

    def forward(self, fixed_image, moving_image, p0=None, fluid_sigma=None, lncc_window_size=5, affine_params=None, reg_weight=0.0, sigma_mode='voxel'):
        """
        Forward pass computing standard LNCC loss at t=1 with optional Affine pre-alignment in JAX.
        """
        if p0 is None:
            p0 = self.p0
        if fluid_sigma is None:
            fluid_sigma = self.fluid_sigma
        if affine_params is None:
            affine_params = self.affine_params

        fixed_image = jnp.array(fixed_image)
        moving_image = jnp.array(moving_image)
        target_shape = tuple(fixed_image.shape[2:])

        disp_fwd, _ = self.shoot(p0=p0, n_steps=self.n_time_steps, fluid_sigma=fluid_sigma, image_shape=target_shape, sigma_mode=sigma_mode)

        phys_grid = get_physical_grid_jax(
            target_shape, self.spacing, self.origin, self.direction
        )
        spacing_rev = tuple(reversed(self.spacing))
        origin_rev = tuple(reversed(self.origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(self.direction)[::-1, ::-1])

        shape_t = jnp.array(target_shape, dtype=jnp.float32)
        spacing_t = jnp.array(spacing_rev, dtype=jnp.float32)
        origin_t = jnp.array(origin_rev, dtype=jnp.float32)
        direction_t = jnp.array(direction_rev, dtype=jnp.float32)

        T_grid = get_affine_matrix_jax(affine_params, self.dim, self.transform_type)
        M_phys, t_phys = grid_to_physical_affine_jax(
            T_grid, target_shape, self.spacing, self.origin, self.direction,
            target_shape, self.spacing, self.origin, self.direction
        )

        phi_moving_affine = (phys_grid + disp_fwd) @ M_phys.T + t_phys
        phi_norm = physical_to_normalized_jax_cached(
            phi_moving_affine, shape_t, spacing_t, origin_t, direction_t
        )
        moving_warped = jax_grid_sample(moving_image, phi_norm, mode='bilinear', padding_mode='zeros')

        fl_sig = self.fluid_sigma if fluid_sigma is None else fluid_sigma
        fl_sig_val = math.sqrt(fl_sig) if fl_sig > 0 else 0.0
        p_spacing = [sp * (float(orig_s) / float(curr_s)) for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, target_shape)] if sigma_mode == 'physical' else None
        v0 = separable_gaussian_filter_jax(p0, sigma=fl_sig_val, spacing=p_spacing, sigma_mode=sigma_mode)
        energy_reg = 0.5 * jnp.mean(p0 * v0)

        lncc_val = local_ncc_loss_nd_jax(fixed_image, moving_warped, window_size=lncc_window_size)
        return lncc_val + reg_weight * energy_reg

    def _resize_p0(self, new_shape):
        """
        Resize momentum parameter tensor in JAX across pyramid levels using interpolate_jax.
        """
        new_shape = tuple(new_shape)
        old_shape = tuple(self.p0.shape[1:-1])
        if new_shape == old_shape:
            return

        p0_cf = jnp.moveaxis(self.p0, -1, 1)
        new_cf = interpolate_jax(p0_cf, new_shape, self.dim)
        self.p0 = jnp.moveaxis(new_cf, 1, -1)

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[100, 100, 50],
        affine_epochs=100,
        lr=2.0,
        fluid_sigma=1.0,
        lncc_window_size=5,
        reg_weight=0.0,
        verbose=False,
        **kwargs
    ):
        """
        Multi-resolution EPDiff Geodesic Shooting optimization with Affine pre-alignment in JAX.
        """
        f_shape, f_spacing, f_origin, f_direction = extract_image_metadata(fixed_image, dim=self.dim)
        if f_spacing is not None: self.spacing = f_spacing
        if f_origin is not None: self.origin = f_origin
        if f_direction is not None: self.direction = f_direction
        if f_shape is not None and self.image_shape != f_shape:
            self.image_shape = f_shape
            self.p0 = jnp.zeros((1, *self.image_shape, self.dim), dtype=jnp.float32)

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

        dim = self.dim
        fluid_sigma_input = kwargs.get('flow_sigma', kwargs.get('fluid_sigma', fluid_sigma))
        sigma_mode = kwargs.get('sigma_mode', 'voxel')

        # 1. Affine Pre-Alignment Stage
        if affine_epochs > 0:
            fixed_spacing_t = jnp.array(self.spacing, dtype=jnp.float32)
            fixed_origin_t = jnp.array(self.origin, dtype=jnp.float32)
            fixed_direction_t = jnp.array(self.direction, dtype=jnp.float32)

            Nx_t = jnp.array(fixed_image.shape[2:], dtype=jnp.float32)
            Ny_t = jnp.array(moving_image.shape[2:], dtype=jnp.float32)

            Dx_t, Sx_t, Ox_t = fixed_direction_t, fixed_spacing_t, fixed_origin_t
            Dy_t, Sy_t, Oy_t = fixed_direction_t, fixed_spacing_t, fixed_origin_t

            com_fixed_fov = Ox_t + Dx_t @ (Sx_t * ((Nx_t - 1) / 2.0))
            com_moving_fov = Oy_t + Dy_t @ (Sy_t * ((Ny_t - 1) / 2.0))

            t_fov = com_moving_fov - com_fixed_fov

            H_x = jnp.eye(dim + 1)
            H_x = H_x.at[:dim, :dim].set(Dx_t @ jnp.diag(Sx_t) @ jnp.diag((Nx_t - 1) / 2.0))
            H_x = H_x.at[:dim, dim].set(com_fixed_fov)

            H_y = jnp.eye(dim + 1)
            H_y = H_y.at[:dim, :dim].set(Dy_t @ jnp.diag(Sy_t) @ jnp.diag((Ny_t - 1) / 2.0))
            H_y = H_y.at[:dim, dim].set(com_moving_fov)

            T_phys = jnp.eye(dim + 1)
            T_phys = T_phys.at[:dim, dim].set(t_fov)

            T_init = jnp.linalg.inv(H_y) @ T_phys @ H_x
            self.affine_params['T_init'] = T_init

            if verbose: print("[GeodesicShooting] Optimizing Affine Pre-Alignment in JAX...")
            from .tvf_jax import adam_step_dict, clamp_affine_params_jax
            m_aff = {k: jnp.zeros_like(v) for k, v in self.affine_params.items()}
            v_aff = {k: jnp.zeros_like(v) for k, v in self.affine_params.items()}
            t_aff = 0

            def affine_loss_fn(params_aff):
                return self.forward(fixed_image, moving_image, p0=jnp.zeros_like(self.p0), fluid_sigma=fluid_sigma_input, lncc_window_size=lncc_window_size, affine_params=params_aff, reg_weight=reg_weight, sigma_mode=sigma_mode)

            grad_aff_fn = jax.grad(affine_loss_fn)
            for ep in range(affine_epochs):
                grads_aff = grad_aff_fn(self.affine_params)
                self.affine_params, m_aff, v_aff, t_aff = adam_step_dict(
                    self.affine_params, grads_aff, m_aff, v_aff, t_aff, lr=1e-2
                )
                self.affine_params = clamp_affine_params_jax(self.affine_params)

        # 2. Deformable Geodesic Shooting Stage
        from .tvf_jax import adam_step
        max_p0_shape = self.image_shape

        for level, epochs in zip(levels, epochs_per_level):
            if epochs <= 0:
                continue

            curr_p0_shape = tuple(max(8, int(s // level)) for s in max_p0_shape)
            prev_p0_shape = tuple(self.p0.shape[1:-1])

            if curr_p0_shape != prev_p0_shape:
                self._resize_p0(curr_p0_shape)

            m_p0 = jnp.zeros_like(self.p0)
            v_p0 = jnp.zeros_like(self.p0)
            t_p0 = 0

            if level > 1:
                down_shape = tuple([max(8, s // level) for s in self.image_shape])
                curr_fixed = interpolate_jax(fixed_image, down_shape, self.dim)
                curr_moving = interpolate_jax(moving_image, down_shape, self.dim)
            else:
                curr_fixed = fixed_image
                curr_moving = moving_image

            def shooting_loss_fn(p0_param):
                return self.forward(curr_fixed, curr_moving, p0=p0_param, fluid_sigma=fluid_sigma_input, lncc_window_size=lncc_window_size, reg_weight=reg_weight, sigma_mode=sigma_mode)

            val_and_grad_fn = jax.value_and_grad(shooting_loss_fn)
            opt_type = kwargs.get('optimizer_type', kwargs.get('optimizer', 'adam')).lower()
            trust_coeff = float(kwargs.get('trust_coefficient', kwargs.get('trust', 0.05)))

            for epoch in range(epochs):
                loss_val, grad_p0 = val_and_grad_fn(self.p0)
                if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                    if verbose: print(f"[GeodesicShooting] Level {level} NaN loss detected at epoch {epoch}, stopping level.")
                    break

                # Clip gradient norm matching PyTorch clip_grad_norm_(max_norm=5.0)
                grad_norm = jnp.sqrt(jnp.sum(grad_p0 ** 2) + 1e-12)
                grad_p0_clipped = jnp.where(grad_norm > 5.0, grad_p0 * (5.0 / (grad_norm + 1e-8)), grad_p0)

                if opt_type == 'lars':
                    from .tvf_jax import lars_step_jax
                    self.p0 = lars_step_jax(self.p0, grad_p0_clipped, lr=lr, trust_coefficient=trust_coeff)
                elif opt_type == 'cfl':
                    cfl_step = float(kwargs.get('cfl_voxels', kwargs.get('cfl', kwargs.get('step', lr))))
                    max_norm = jnp.max(jnp.sqrt(jnp.sum(grad_p0 ** 2, axis=-1)))
                    curr_spacing_level = [
                        sp * (float(orig_s) / float(curr_s))
                        for sp, orig_s, curr_s in zip(self.spacing, self.image_shape, curr_fixed.shape[2:])
                    ]
                    spacing_rev = tuple(reversed(curr_spacing_level))
                    sp_tensor = jnp.array(spacing_rev, dtype=jnp.float32)
                    step_update = jnp.where(
                        max_norm > 1e-12,
                        (cfl_step / jnp.maximum(max_norm, 1e-8)) * grad_p0 * sp_tensor,
                        jnp.zeros_like(grad_p0)
                    )
                    self.p0 = self.p0 - step_update
                elif opt_type == 'sgd':
                    self.p0 = self.p0 - lr * grad_p0_clipped
                else:
                    self.p0, m_p0, v_p0, t_p0 = adam_step(self.p0, grad_p0_clipped, m_p0, v_p0, t_p0, lr=lr)

        final_p0_shape = tuple(self.p0.shape[1:-1])
        if final_p0_shape != max_p0_shape:
            self._resize_p0(max_p0_shape)

        return self

    def get_forward_warp(self, image_shape=None):
        disp_fwd, _ = self.shoot(image_shape=image_shape)
        return disp_fwd

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

