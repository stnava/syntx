r"""
SyNGS -- "geodesic shooting" registration (``syntx.syngs``).

The deformation is parameterised by an initial velocity field v0 on the fixed grid: it is
smoothed once by the regulariser (Sobolev by default, strength ``alpha``) and integrated with
``n_steps`` Euler steps, phi_{k+1} = phi_k + dt * v0(phi_k). In the default
``transport_mode='transport'`` the same smoothed v0 is used at every step, i.e. the map is the
flow of a *stationary* velocity field (exp(v0)); EPDiff momentum transport is not
implemented. The 'scaled' / 'recursive' modes re-smooth the sampled velocity between steps.

With ``symmetric=True`` (default) a second, independent field v0_inv produces the inverse
warp; the two are tied by an inverse-consistency penalty (``inverse_identity_weight``), not
by construction. There is no energy term on v0 (2026-09-30 study: 2-D defaults reach true
minimum Jacobian determinants of ~3e-4 -- see docs/DOCSTRING_AUDIT.md).
"""

import math
import tempfile
import time as _time
from typing import Union, Sequence, Tuple, List, Optional, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ants

from .syn import (
    HierarchicalAffine,
    grid_sample_nd,
    local_ncc_loss_nd,
    mattes_mi_loss_nd,
    parse_ants_affine,
)
from .core.smoothing import separable_gaussian_filter
from .core.optimizers import RegAdam, LARS
from .core.grid import compose_grids, resize_field, sample_field_cf
from .spatial import (
    reverse_metadata,
    itk_shape_to_tensor_shape,
    image_to_tensor,
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    grid_to_physical_affine,
    grid_to_physical_affine_torch,
    export_ants_affine_transform,
    export_ants_displacement_field,
    compute_grid_to_physical_reference_matrix,
)
from .pyramid import build_image_pyramid


# Default spectral (Sobolev) strength per dimension. 3-D: the benchmark configuration (3-D
# sweep 2026-09-19); 2-D: the calibrated legacy value. Shared by syngs_registration(), the model
# and integrate_momentum() so a saved momentum field reconstructs exactly.
SYNGS_DEFAULT_ALPHA = {2: 0.060, 3: 0.675}


def default_alpha(dim: int) -> float:
    """Default Sobolev strength ``alpha`` for ``dim``-D images (``SYNGS_DEFAULT_ALPHA``; 3-D if missing)."""
    return SYNGS_DEFAULT_ALPHA.get(int(dim), SYNGS_DEFAULT_ALPHA[3])


def _level_spacing(spacing_xyz, orig_shape_zyx, curr_shape_zyx):
    """ITK (x, y, z)-order spacing of a resampled grid that spans the same box:
    ``sp * (n_orig - 1) / (n_curr - 1)`` per axis, pairing the ITK spacing with the tensor
    (z, y, x) shapes axis by axis (the shapes are reversed)."""
    return [sp * (float(o - 1) / float(c - 1)) if c > 1 else sp
            for sp, o, c in zip(spacing_xyz, reversed(tuple(orig_shape_zyx)), reversed(tuple(curr_shape_zyx)))]


class GeodesicShootingModel(nn.Module):
    """
    The PyTorch model behind ``syntx.syngs`` (see the module docstring for the transform).
    Most users call ``syntx.syngs``; use the class directly only for custom pipelines.

    Parameters ``velocity_0_fwd`` (and ``velocity_0_inv`` when symmetric): shape
    ``(1, *velocity_shape, dim)``, physical velocity (mm per unit time), tensor (z, y, x)
    order, on the fixed grid.

    Parameters
    ----------
    dim : {2, 3}
    image_shape : tuple of int
        Fixed-image grid, tensor (z, y, x) order.
    velocity_shape : tuple of int, optional
        Velocity grid (default ``image_shape``).
    spacing, origin, direction : optional
        Fixed-image geometry (ITK (x, y, z) order); defaults unit spacing, zero origin, identity.
    moving_shape, moving_spacing, moving_origin, moving_direction : optional
        Moving-image geometry; defaults to the fixed image's.
    fluid_sigma : float, default 3.0
        Gaussian sigma for the 'gaussian' / 'bspline' regularisers.
    transform_type : {'Affine', 'Rigid', 'Translation'}, default 'Affine'
    n_steps : int, default 6
        Euler steps (``syntx.syngs`` passes 8).
    solver : {'euler', 'midpoint' / 'rk2' / 'heun', 'rk4'}, default 'euler'
    symmetric : bool, default True
        Optimise a separate inverse field v0_inv (else the inverse uses -v0).
    inverse_identity_weight : float, default 0.5
        Weight of the inverse-consistency penalty between the two fields.
    alpha : float, optional
        Spectral strength (default ``default_alpha(dim)``).
    seed : int, default 42
    **kwargs
        ``regularizer`` ('sobolev' default; 'gaussian', 'dsti' / 'dsti1' (both mean DST-I),
        'bspline'; other values raise ValueError), ``transport_mode``
        ('transport' default, 'scaled', 'recursive'), ``similarity_metric`` ('lncc'),
        ``mattes_bins`` (32), ``bootstrap_mode`` ('none'), ``bootstrap_orig_weight`` (0.5),
        ``bootstrap_jitter_scale`` (0.25), ``spline_distance``, ``mesh_size``.
    """
    def __init__(
        self,
        dim,
        image_shape,
        velocity_shape=None,
        spacing=None,
        origin=None,
        direction=None,
        fluid_sigma=3.0,
        transform_type='Affine',
        n_steps=6,
        solver='euler',
        symmetric=True,
        inverse_identity_weight=0.50,
        alpha=None,
        moving_shape=None,
        moving_spacing=None,
        moving_origin=None,
        moving_direction=None,
        seed=42,
        **kwargs
    ):
        super().__init__()
        unknown = sorted(set(kwargs) - {'similarity_metric', 'mattes_bins', 'bootstrap_mode',
                                        'bootstrap_orig_weight', 'bootstrap_jitter_scale',
                                        'regularizer', 'spline_distance', 'mesh_size',
                                        'transport_mode'})
        if unknown:
            raise TypeError(f"GeodesicShootingModel() got unexpected keyword(s) {unknown}")
        self.dim = dim
        self.seed = int(seed) if seed is not None else None
        self._rng = None
        self.image_shape = tuple(image_shape)
        if velocity_shape is None:
            velocity_shape = image_shape
        self.velocity_shape = tuple(velocity_shape)
        self.n_steps = n_steps
        self.symmetric = symmetric
        self.inverse_identity_weight = inverse_identity_weight
        
        self.spacing = list(spacing) if spacing is not None else [1.0] * dim
        self.origin = list(origin) if origin is not None else [0.0] * dim
        if direction is not None:
            self.direction = direction.tolist() if hasattr(direction, 'tolist') else list(direction)
        else:
            self.direction = np.eye(dim).tolist()
            
        self.moving_shape = tuple(moving_shape) if moving_shape is not None else self.image_shape
        self.moving_spacing = list(moving_spacing) if moving_spacing is not None else self.spacing
        self.moving_origin = list(moving_origin) if moving_origin is not None else self.origin
        if moving_direction is not None:
            self.moving_direction = moving_direction.tolist() if hasattr(moving_direction, 'tolist') else list(moving_direction)
        else:
            self.moving_direction = self.direction

        self.fluid_sigma = fluid_sigma
        self.solver = solver
        
        self.alpha = float(alpha) if alpha is not None else default_alpha(dim)
            
        self.similarity_metric = kwargs.get('similarity_metric', 'lncc')
        self.mattes_bins = int(kwargs.get('mattes_bins', 32))
        self.bootstrap_mode = kwargs.get('bootstrap_mode', 'none')
        self.bootstrap_orig_weight = float(kwargs.get('bootstrap_orig_weight', 0.50))
        self.bootstrap_jitter_scale = float(kwargs.get('bootstrap_jitter_scale', 0.25))
        self.regularizer = str(kwargs.get('regularizer', 'sobolev')).lower()
        if self.regularizer in ('gauss', 'gaussian'):
            self.regularizer = 'gaussian'
        elif self.regularizer in ('dsti', 'dsti1', 'dst_i', 'dirichlet'):
            self.regularizer = 'dsti1'
        elif self.regularizer in ('bspline', 'bsplinesyn'):
            self.regularizer = 'bspline'
        elif self.regularizer != 'sobolev':
            raise ValueError(f"unknown regularizer {self.regularizer!r}: use 'sobolev', 'gaussian', "
                             "'dsti' / 'dsti1' or 'bspline'")
        self.spline_distance = kwargs.get('spline_distance', None)
        self.mesh_size = kwargs.get('mesh_size', None)
        self.transport_mode = str(kwargs.get('transport_mode', 'transport')).lower()
        
        # Dual momentum fields for symmetric shooting: v0_fwd (Fixed space) and v0_inv (Moving space)
        self.velocity_0_fwd = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        if self.symmetric:
            self.velocity_0_inv = nn.Parameter(torch.zeros(1, *self.velocity_shape, self.dim))
        else:
            self.velocity_0_inv = None
            
        self.velocity_0 = self.velocity_0_fwd
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)

    def _resize_single_velocity(self, vel_param, new_shape, device=None, dtype=None):
        """Resize one velocity parameter to ``new_shape`` (trilinear; values are mm, unchanged)."""
        if vel_param is None:
            return None
        new_shape = tuple(new_shape)
        old_shape = tuple(vel_param.shape[1:-1])
        if new_shape == old_shape:
            return vel_param
        with torch.no_grad():
            new_vel = resize_field(vel_param.data, new_shape)
            if device is not None:
                new_vel = new_vel.to(device=device)
            if dtype is not None:
                new_vel = new_vel.to(dtype=dtype)
            return nn.Parameter(new_vel.contiguous())

    def _resize_velocity(self, new_shape, device=None, dtype=None):
        """Resize the velocity parameter(s) to ``new_shape`` (pyramid level change)."""
        self.velocity_0_fwd = self._resize_single_velocity(self.velocity_0_fwd, new_shape, device, dtype)
        if self.symmetric and self.velocity_0_inv is not None:
            self.velocity_0_inv = self._resize_single_velocity(self.velocity_0_inv, new_shape, device, dtype)
        self.velocity_0 = self.velocity_0_fwd

    def _create_boundary_mask(self, spatial_shape, device, dtype, border_width=None):
        """Smooth (1, *spatial, 1) taper falling to 0 over ``border_width`` voxels at every face."""
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

    def _apply_sobolev_green_operator(self, m, fluid_sigma=3.0, alpha=None, spacing=None):
        """Sobolev smoothing of ``m`` (``core.smoothing.apply_sobolev_green_operator``)."""
        from .core.smoothing import apply_sobolev_green_operator
        return apply_sobolev_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha, spacing=spacing)

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """DST (Dirichlet) smoothing of ``m`` (``core.smoothing.apply_dsti_green_operator``)."""
        from .core.smoothing import apply_dsti_green_operator
        return apply_dsti_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def _apply_dsti1_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """DST-I (Dirichlet) smoothing of ``m`` (``core.smoothing.apply_dsti1_green_operator``)."""
        from .core.smoothing import apply_dsti1_green_operator
        return apply_dsti1_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def apply_green_operator(self, m, shape, spacing_zyx):
        """
        Apply elected regularizer / Green's operator:
        - 'gaussian': separable Gaussian filter with sigma = fluid_sigma
        - 'dsti1': Discrete Sine Transform Type-I with Dirichlet boundary conditions
        - 'sobolev': exact Fourier Sobolev operator K(k) = 1 / (1 + alpha * |k|^2)^s with boundary cosine tapering
        """
        if self.regularizer in ('gaussian', 'gauss', 'bspline', 'bsplinesyn'):
            if self.fluid_sigma is None or self.fluid_sigma <= 0:     # flow_sigma = 0: no smoothing
                return m
        elif self.alpha is None or self.alpha <= 0:                   # spectral: alpha = 0: no smoothing
            return m
        device = m.device
        dtype = m.dtype
        dim = self.dim
        
        if self.regularizer in ('gaussian', 'gauss'):
            from .core.smoothing import separable_gaussian_filter
            return separable_gaussian_filter(m, sigma=self.fluid_sigma)   # sigma in voxels
            
        if self.regularizer in ('dsti', 'dsti1', 'dst_i', 'dirichlet'):
            from .core.smoothing import apply_dsti1_green_operator
            return apply_dsti1_green_operator(m, fluid_sigma=1.0, alpha=self.alpha)  # fluid_sigma: gate only

        if self.regularizer in ('bspline', 'bsplinesyn'):
            from .core.smoothing import smooth_displacement_field_bspline
            spacing_itk = tuple(reversed(spacing_zyx))
            return smooth_displacement_field_bspline(
                m,
                spacing=spacing_itk,
                mesh_size=self.mesh_size,
                spline_distance=self.spline_distance,
                fluid_sigma=self.fluid_sigma,
                enforce_stationary_boundary=False,
                coord_convention='xyz',
            )

        # Standard Sobolev with boundary cosine tapering
        bmask = self._create_boundary_mask(shape, device, dtype, border_width=4)
        m_tapered = m * bmask
        
        k_axes = []
        for d in range(dim):
            n_d = shape[d]
            sp_d = spacing_zyx[d]
            if d == dim - 1:
                k_d = torch.fft.rfftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            else:
                k_d = torch.fft.fftfreq(n_d, d=sp_d, device=device) * (2.0 * math.pi)
            k_axes.append(k_d)
            
        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        k_sq = sum(k_j ** 2 for k_j in k_mesh)
        K_fourier = 1.0 / ((1.0 + self.alpha * k_sq) ** 2.0)
        
        v_out = self._apply_sobolev_fft(m_tapered, K_fourier, shape, dtype)
        return v_out * bmask

    def _apply_sobolev_fft(
        self,
        m_tapered: torch.Tensor,
        K_fourier: torch.Tensor,
        shape: Union[Sequence[int], Tuple[int, ...]],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Applies frequency-domain Fourier Sobolev Green filter using dimension-agnostic movedim."""
        dim = len(shape)
        spatial_dims = tuple(range(2, 2 + dim))
        m_cf = torch.movedim(m_tapered, -1, 1).to(torch.float32).contiguous()
        m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)
        K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
        v_fft = m_fft * K_bc
        v_cf = torch.fft.irfftn(v_fft, s=shape, dim=spatial_dims).to(dtype=dtype).contiguous()
        return torch.movedim(v_cf, 1, -1)

    def shoot(self, v_init, target_shape, spacing_zyx, phys_grid, meta):
        """
        Evolve initial momentum v_init forward along the geodesic trajectory.
        Integrates: d phi / dt = v(t, phi(t)).

        transport_mode:
        - 'transport' (default): Initial momentum is smoothed once at t=0 (v0 = K(v_init)),
          and transported along the trajectory phi(t) without in-loop re-filtering.
          Prevents compounding exponential damping and runs 2x faster.
        - 'scaled': In-loop re-filtering with step-scaled bandwidth (sigma / sqrt(N_steps)).
        - 'recursive': Traditional in-loop re-filtering at each step.
        """
        dt = 1.0 / self.n_steps
        
        if tuple(v_init.shape[1:-1]) != target_shape:
            v_up = resize_field(v_init, target_shape)
        else:
            v_up = v_init
            
        v0_smooth = self.apply_green_operator(v_up, target_shape, spacing_zyx)
        shape_t, spacing_t, origin_t, direction_t = meta
        disp = torch.zeros_like(phys_grid)

        # Pre-compute affine mapping matrix and bias once, fusing torch.flip
        scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
        M = direction_t * scale_t.unsqueeze(0)
        b = - (origin_t @ M) - 1.0
        M_norm = torch.flip(M, dims=[1])
        b_norm = torch.flip(b, dims=[0])

        # Pre-compute normalized identity grid once
        u_id = (phys_grid.view(-1, self.dim) @ M_norm + b_norm).view(phys_grid.shape)

        def _to_norm(disp_phys):
            return u_id + (disp_phys.view(-1, self.dim) @ M_norm).view(disp_phys.shape)

        if self.transport_mode == 'transport':
            # sample_field_cf(v0_cf, grid) = grid_sample_nd(v0_cf, grid).movedim(1,-1)
            # v0_smooth is in last-channel format (B, *spatial, dim)
            v0_cf = torch.movedim(v0_smooth, -1, 1)   # channel-first for repeated sampling
            sol = str(self.solver).lower()
            if sol in ('midpoint', 'rk2', 'heun'):
                for step in range(self.n_steps):
                    phi_norm_1 = u_id if step == 0 else _to_norm(disp)
                    k1 = sample_field_cf(v0_cf, phi_norm_1)
                    phi_norm_2 = _to_norm(disp + (0.5 * dt) * k1)
                    k2 = sample_field_cf(v0_cf, phi_norm_2)
                    disp = disp + dt * k2
                return disp
            elif sol == 'rk4':
                for step in range(self.n_steps):
                    phi_norm_1 = u_id if step == 0 else _to_norm(disp)
                    k1 = sample_field_cf(v0_cf, phi_norm_1)
                    phi_norm_2 = _to_norm(disp + (0.5 * dt) * k1)
                    k2 = sample_field_cf(v0_cf, phi_norm_2)
                    phi_norm_3 = _to_norm(disp + (0.5 * dt) * k2)
                    k3 = sample_field_cf(v0_cf, phi_norm_3)
                    phi_norm_4 = _to_norm(disp + dt * k3)
                    k4 = sample_field_cf(v0_cf, phi_norm_4)
                    disp = disp + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                return disp
            else:
                for step in range(self.n_steps):
                    phi_norm = u_id if step == 0 else _to_norm(disp)
                    v_sampled = sample_field_cf(v0_cf, phi_norm)
                    disp = disp + dt * v_sampled
                return disp

        v = v0_smooth
        for step in range(self.n_steps):
            phi_norm = u_id if step == 0 else _to_norm(disp)
            v_sampled = sample_field_cf(torch.movedim(v, -1, 1), phi_norm)
            disp = disp + dt * v_sampled

            if step < self.n_steps - 1:
                # Reuse v_sampled directly instead of running grid_sample_nd twice
                v = self.apply_green_operator(v_sampled, target_shape, spacing_zyx)

        return disp


    def _eval_similarity(self, I, J, metric_name, lncc_window_size=5):
        """Similarity loss between ``I`` and ``J`` for ``metric_name`` ('mattes*' / 'mi*', 'mse', 'cc2', 'box_lncc', else local CC); lower is better."""
        m_lower = metric_name.lower()
        if m_lower in ('mattes_mi', 'mattes', 'mi', 'mmi') or m_lower.startswith('mattes') or m_lower.startswith('mi_'):
            n_bins = getattr(self, 'mattes_bins', 32)
            parts = m_lower.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                n_bins = int(parts[-1])
            fg_mask = ((I.abs() > 0.01) | (J.abs() > 0.01)).float()
            return mattes_mi_loss_nd(I, J, mask=fg_mask, num_bins=n_bins)
        elif m_lower == 'mse':
            return torch.mean((I - J) ** 2)
        elif m_lower in ('cc2', 'lncc2'):
            return local_ncc_loss_nd(I, J, window_size=lncc_window_size, squared=True)
        elif m_lower in ('box_lncc', 'box_cc', 'fireants_lncc'):
            from .core.losses import BoxLNCCLoss
            box_loss_fn = BoxLNCCLoss(kernel_size=lncc_window_size)
            return box_loss_fn(I, J)
        else:
            return local_ncc_loss_nd(I, J, window_size=lncc_window_size, squared=False)

    def forward(self, fixed_image, moving_image, lncc_window_size=5, similarity_metric=None):
        """
        Loss of the current fields: shoot v0_fwd (moving -> fixed space) and v0_inv (fixed ->
        moving), compare each warped image with the other image (``similarity_metric``,
        optionally with antithetic jittered evaluations), average the two, and add
        ``inverse_identity_weight`` x the mean squared inverse-consistency error when
        symmetric. ``fixed_image`` / ``moving_image``: tensors (1, 1, *spatial). Lower is better.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        target_shape_f = tuple(fixed_image.shape[2:])
        target_shape_m = tuple(moving_image.shape[2:])

        curr_spacing_f = _level_spacing(self.spacing, self.image_shape, target_shape_f)
        curr_spacing_m = _level_spacing(self.moving_spacing, self.moving_shape, target_shape_m)

        phys_grid_f = get_physical_grid_torch(
            target_shape_f, curr_spacing_f, self.origin, self.direction,
            device=device, dtype=dtype
        )

        spacing_rev_f, origin_rev_f, direction_rev_f = reverse_metadata(curr_spacing_f, self.origin, self.direction)

        shape_t_f = torch.tensor(list(target_shape_f), device=device, dtype=dtype)
        spacing_t_f = torch.tensor(spacing_rev_f, device=device, dtype=dtype)
        origin_t_f = torch.tensor(origin_rev_f, device=device, dtype=dtype)
        direction_t_f = torch.tensor(direction_rev_f, device=device, dtype=dtype)
        meta_f = (shape_t_f, spacing_t_f, origin_t_f, direction_t_f)

        spacing_rev_m, origin_rev_m, direction_rev_m = reverse_metadata(curr_spacing_m, self.moving_origin, self.moving_direction)
        shape_t_m = torch.tensor(list(target_shape_m), device=device, dtype=dtype)
        spacing_t_m = torch.tensor(spacing_rev_m, device=device, dtype=dtype)
        origin_t_m = torch.tensor(origin_rev_m, device=device, dtype=dtype)
        direction_t_m = torch.tensor(direction_rev_m, device=device, dtype=dtype)

        T_grid = self.affine.get_matrix()
        M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
            T_grid, target_shape_f, curr_spacing_f, self.origin, self.direction,
            target_shape_m, curr_spacing_m, self.moving_origin, self.moving_direction
        )

        # 1. Forward shooting (+v0_fwd) -> warp moving to fixed space
        disp_fwd = self.shoot(self.velocity_0_fwd, target_shape_f, spacing_rev_f, phys_grid_f, meta_f)
        phi_moving = (phys_grid_f + disp_fwd) @ M_phys_zyx.t() + t_phys_zyx
        phi_norm_fwd = physical_to_normalized_torch_cached(
            phi_moving, shape_t_m, spacing_t_m, origin_t_m, direction_t_m
        )
        moving_warped = grid_sample_nd(moving_image, phi_norm_fwd, mode='bilinear', padding_mode='zeros')
        
        # 2. Inverse shooting (+v0_inv)
        v0_inv_param = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        disp_inv = self.shoot(v0_inv_param, target_shape_f, spacing_rev_f, phys_grid_f, meta_f)

        # Prealigned moving on reference grid
        phi_m_aff = phys_grid_f @ M_phys_zyx.t() + t_phys_zyx
        phi_m_aff_norm = physical_to_normalized_torch_cached(
            phi_m_aff, shape_t_m, spacing_t_m, origin_t_m, direction_t_m
        )
        moving_aff = grid_sample_nd(moving_image, phi_m_aff_norm, mode='bilinear', padding_mode='zeros')

        # Deformed fixed on reference grid
        phi_fixed = phys_grid_f + disp_inv
        phi_norm_fixed = physical_to_normalized_torch_cached(
            phi_fixed, shape_t_f, spacing_t_f, origin_t_f, direction_t_f
        )
        fixed_warped = grid_sample_nd(fixed_image, phi_norm_fixed, mode='bilinear', padding_mode='zeros')

        metric_to_use = similarity_metric if similarity_metric is not None else self.similarity_metric

        if self.bootstrap_mode == 'antithetic' and self.bootstrap_jitter_scale > 0 and self.training:
            # Unbiased antithetic coordinate triplet
            w0 = self.bootstrap_orig_weight
            s_jitter = self.bootstrap_jitter_scale
            jitter_shape = [1] * (self.dim + 1) + [self.dim]
            if self.seed is not None and (self._rng is None or self._rng.device != device):
                self._rng = torch.Generator(device=device).manual_seed(self.seed)

            if self._rng is not None:
                jitter_vox = (torch.rand(jitter_shape, generator=self._rng, device=device, dtype=dtype) - 0.5) * 2.0 * s_jitter
            else:
                jitter_vox = (torch.rand(jitter_shape, device=device, dtype=dtype) - 0.5) * 2.0 * s_jitter
            jitter_phys = jitter_vox * spacing_t_f

            # Center loss
            loss_fwd_0 = self._eval_similarity(fixed_image, moving_warped, metric_to_use, lncc_window_size=lncc_window_size)
            loss_inv_0 = self._eval_similarity(moving_aff, fixed_warped, metric_to_use, lncc_window_size=lncc_window_size)

            # Plus jitter
            phi_mov_p = (phys_grid_f + disp_fwd + jitter_phys) @ M_phys_zyx.t() + t_phys_zyx
            norm_mov_p = physical_to_normalized_torch_cached(phi_mov_p, shape_t_m, spacing_t_m, origin_t_m, direction_t_m)
            w_mov_p = grid_sample_nd(moving_image, norm_mov_p, mode='bilinear', padding_mode='zeros')
            loss_fwd_p = self._eval_similarity(fixed_image, w_mov_p, metric_to_use, lncc_window_size=lncc_window_size)

            phi_fix_p = phys_grid_f + disp_inv + jitter_phys
            norm_fix_p = physical_to_normalized_torch_cached(phi_fix_p, shape_t_f, spacing_t_f, origin_t_f, direction_t_f)
            w_fix_p = grid_sample_nd(fixed_image, norm_fix_p, mode='bilinear', padding_mode='zeros')
            loss_inv_p = self._eval_similarity(moving_aff, w_fix_p, metric_to_use, lncc_window_size=lncc_window_size)

            # Minus jitter
            phi_mov_m = (phys_grid_f + disp_fwd - jitter_phys) @ M_phys_zyx.t() + t_phys_zyx
            norm_mov_m = physical_to_normalized_torch_cached(phi_mov_m, shape_t_m, spacing_t_m, origin_t_m, direction_t_m)
            w_mov_m = grid_sample_nd(moving_image, norm_mov_m, mode='bilinear', padding_mode='zeros')
            loss_fwd_m = self._eval_similarity(fixed_image, w_mov_m, metric_to_use, lncc_window_size=lncc_window_size)

            phi_fix_m = phys_grid_f + disp_inv - jitter_phys
            norm_fix_m = physical_to_normalized_torch_cached(phi_fix_m, shape_t_f, spacing_t_f, origin_t_f, direction_t_f)
            w_fix_m = grid_sample_nd(fixed_image, norm_fix_m, mode='bilinear', padding_mode='zeros')
            loss_inv_m = self._eval_similarity(moving_aff, w_fix_m, metric_to_use, lncc_window_size=lncc_window_size)

            loss_fwd = w0 * loss_fwd_0 + 0.5 * (1.0 - w0) * (loss_fwd_p + loss_fwd_m)
            loss_inv = w0 * loss_inv_0 + 0.5 * (1.0 - w0) * (loss_inv_p + loss_inv_m)
        else:
            loss_fwd = self._eval_similarity(fixed_image, moving_warped, metric_to_use, lncc_window_size=lncc_window_size)
            loss_inv = self._eval_similarity(moving_aff, fixed_warped, metric_to_use, lncc_window_size=lncc_window_size)

        sim_loss = 0.5 * (loss_fwd + loss_inv)

        # 3. Inverse identity consistency loss in reference fixed space
        if self.symmetric and self.inverse_identity_weight > 0:
            phi_inv_pure = phys_grid_f + disp_inv
            phi_inv_pure_norm = physical_to_normalized_torch_cached(
                phi_inv_pure, shape_t_f, spacing_t_f, origin_t_f, direction_t_f
            )
            # compose_grids handles the movedim internally for any spatial dimensionality
            disp_fwd_at_inv = compose_grids(disp_fwd, phi_inv_pure_norm)
            comp_disp = disp_inv + disp_fwd_at_inv
            inv_id_loss = torch.mean(comp_disp ** 2)
        else:
            inv_id_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return sim_loss + self.inverse_identity_weight * inv_id_loss

    def fit(
        self,
        fixed_image,
        moving_image,
        levels=[4, 2, 1],
        epochs_per_level=[60, 40, 20],
        similarity_metric='lncc',
        lncc_radius=2,
        lr=0.6,
        verbose=False,
        fixed_spacing=None,
        fixed_origin=None,
        fixed_direction=None,
        moving_spacing=None,
        moving_origin=None,
        moving_direction=None,
        optimizer_type='reg_adam',
        cfl_step=0.25,
        **kwargs
    ):
        """
        Multi-resolution optimisation of the initial velocity field(s) (the affine is set
        beforehand, e.g. from ``syntx.syngs``). ``syntx.syngs`` passes every option
        explicitly; these defaults apply only to direct calls.

        Parameters
        ----------
        fixed_image, moving_image : Tensor (1, 1, *spatial)
        levels : list of int, default [4, 2, 1]
        epochs_per_level : list of int, default [60, 40, 20]
        similarity_metric : str, default 'lncc'
        lncc_radius : int, default 2
        lr : float, default 0.6
            Learning rate (scaled by 1/sqrt(level) per level).
        fixed_*, moving_* : geometry overrides.
        optimizer_type : str, default 'reg_adam'
            'reg_adam' (also 'regadam', 'sobolev_adam', 'sobolevadam'), 'adam', 'adamw',
            'sgd', 'lars'; other values raise ValueError.
        cfl_step : float, default 0.25
            Largest update (voxels) when ``max_step_norm`` is not given.
        **kwargs
            ``max_step_norm``, ``adam_eps_rel``, ``weight_decay`` (adamw), ``momentum`` (sgd),
            ``smoothing_sigmas`` (pyramid), ``seed``, ``mattes_bins``; others raise TypeError.

        After each level the best-loss velocity is kept.
        """
        unknown = sorted(set(kwargs) - {'max_step_norm', 'adam_eps_rel', 'weight_decay', 'momentum',
                                        'smoothing_sigmas', 'seed', 'mattes_bins'})
        if unknown:
            raise TypeError(f"GeodesicShootingModel.fit() got unexpected keyword(s) {unknown}")
        device = fixed_image.device
        dtype = fixed_image.dtype

        if 'seed' in kwargs:
            seed_arg = kwargs.pop('seed')
            self.seed = int(seed_arg) if seed_arg is not None else None
        if self.seed is not None:
            self._rng = torch.Generator(device=device).manual_seed(self.seed)
        else:
            self._rng = None
        
        self.similarity_metric = similarity_metric
        self.mattes_bins = int(kwargs.get('mattes_bins', getattr(self, 'mattes_bins', 32)))
        
        if fixed_spacing is not None: self.spacing = list(fixed_spacing)
        if fixed_origin is not None: self.origin = list(fixed_origin)
        if fixed_direction is not None:
            self.direction = fixed_direction.tolist() if hasattr(fixed_direction, 'tolist') else list(fixed_direction)

        if moving_spacing is not None: self.moving_spacing = list(moving_spacing)
        if moving_origin is not None: self.moving_origin = list(moving_origin)
        if moving_direction is not None:
            self.moving_direction = moving_direction.tolist() if hasattr(moving_direction, 'tolist') else list(moving_direction)

        smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
        if smoothing_sigmas is None:
            smoothing_sigmas = [float(np.log2(s)) if s > 1 else 0.0 for s in levels]
        fixed_pyr = build_image_pyramid(fixed_image, spacing=self.spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')
        moving_pyr = build_image_pyramid(moving_image, spacing=self.moving_spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')

        if verbose: print("Optimizing Geodesic Shooting momentum...")
        opt_name = str(optimizer_type).lower()

        for idx, level in enumerate(levels):
            epochs = epochs_per_level[min(idx, len(epochs_per_level) - 1)]
            if epochs <= 0:
                continue
                
            curr_vel_shape = tuple(max(8, s // level) for s in self.image_shape)
            self._resize_velocity(curr_vel_shape, device, dtype)
            
            curr_fixed = fixed_pyr[idx]
            curr_moving = moving_pyr[idx]
            
            active_params = [self.velocity_0_fwd]
            if self.symmetric and self.velocity_0_inv is not None:
                active_params.append(self.velocity_0_inv)

            level_lr = lr * math.sqrt(1.0 / level)
            max_step = float(kwargs.get('max_step_norm', cfl_step))
            
            if opt_name in ('reg_adam', 'regadam', 'sobolev_adam', 'sobolevadam'):
                optimizer = RegAdam(
                    active_params,
                    lr=level_lr,
                    regularizer=self.regularizer,
                    sobolev_alpha=self.alpha,
                    dsti_alpha=self.alpha,
                    gaussian_sigma=self.fluid_sigma,
                    max_step_norm=max_step,
                    **({"eps_rel": float(kwargs["adam_eps_rel"])} if kwargs.get("adam_eps_rel") is not None else {})
                )
            elif opt_name == 'adam':
                optimizer = torch.optim.Adam(active_params, lr=level_lr)
            elif opt_name == 'adamw':
                optimizer = torch.optim.AdamW(active_params, lr=level_lr, weight_decay=float(kwargs.get('weight_decay', 1e-4)))
            elif opt_name == 'sgd':
                optimizer = torch.optim.SGD(active_params, lr=level_lr, momentum=float(kwargs.get('momentum', 0.9)))
            elif opt_name == 'lars':
                optimizer = LARS(active_params, lr=level_lr)
            else:
                raise ValueError(f"unknown optimizer_type {optimizer_type!r}: use 'reg_adam', 'adam', "
                                 "'adamw', 'sgd' or 'lars'")

            lncc_ws = 2 * lncc_radius + 1
            
            best_level_loss = float('inf')
            best_v0_fwd = None
            best_v0_inv = None

            for ep in range(epochs):
                optimizer.zero_grad()
                total_loss = self.forward(
                    curr_fixed, curr_moving,
                    lncc_window_size=lncc_ws,
                    similarity_metric=similarity_metric
                )
                loss_val = float(total_loss.item())
                if loss_val < best_level_loss:
                    best_level_loss = loss_val
                    best_v0_fwd = self.velocity_0_fwd.detach().clone()
                    if self.velocity_0_inv is not None:
                        best_v0_inv = self.velocity_0_inv.detach().clone()

                total_loss.backward()
                optimizer.step()

            if epochs > 0:
                with torch.no_grad():
                    final_loss = float(self.forward(
                        curr_fixed, curr_moving,
                        lncc_window_size=lncc_ws,
                        similarity_metric=similarity_metric
                    ).item())
                if final_loss < best_level_loss:
                    best_level_loss = final_loss
                elif best_v0_fwd is not None:
                    with torch.no_grad():
                        self.velocity_0_fwd.copy_(best_v0_fwd)
                        if self.velocity_0_inv is not None and best_v0_inv is not None:
                            self.velocity_0_inv.copy_(best_v0_inv)

        # Resize parameters back to native resolution for final export
        final_vel_shape = tuple(self.velocity_0_fwd.shape[1:-1])
        if final_vel_shape != tuple(self.image_shape):
            self._resize_velocity(self.image_shape, device, dtype)

    @torch.no_grad()
    def get_forward_warp(self, image_shape=None):
        """Compute forward displacement field (shooting +v0_fwd)."""
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        curr_spacing = _level_spacing(self.spacing, self.image_shape, target_shape)
        device = self.velocity_0_fwd.device
        dtype = self.velocity_0_fwd.dtype
        phys_grid = get_physical_grid_torch(target_shape, curr_spacing, self.origin, self.direction, device=device, dtype=dtype)
        
        spacing_rev, origin_rev, direction_rev = reverse_metadata(curr_spacing, self.origin, self.direction)
        
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        meta = (shape_t, spacing_t, origin_t, direction_t)
        
        disp = self.shoot(self.velocity_0_fwd, target_shape, spacing_rev, phys_grid, meta)
        return disp
        
    @torch.no_grad()
    def get_inverse_warp(self, image_shape=None):
        """Compute inverse displacement field (shooting +v0_inv if symmetric, else -v0_fwd)."""
        target_shape = tuple(image_shape) if image_shape is not None else self.image_shape
        curr_spacing = _level_spacing(self.spacing, self.image_shape, target_shape)
        device = self.velocity_0_fwd.device
        dtype = self.velocity_0_fwd.dtype
        phys_grid = get_physical_grid_torch(target_shape, curr_spacing, self.origin, self.direction, device=device, dtype=dtype)
        
        spacing_rev, origin_rev, direction_rev = reverse_metadata(curr_spacing, self.origin, self.direction)
        
        shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        meta = (shape_t, spacing_t, origin_t, direction_t)
        
        v0_inv = self.velocity_0_inv if (self.symmetric and self.velocity_0_inv is not None) else -self.velocity_0_fwd
        disp = self.shoot(v0_inv, target_shape, spacing_rev, phys_grid, meta)
        return disp


def syngs_registration(
    fixed,
    moving,
    initial_transform=None,
    syn_metric='cc2',
    syn_sampling=2,
    reg_iterations=None,
    affine_dof='affine',
    affine_mode='pytorch',
    affine_seed=None,
    grad_step=0.25,
    flow_sigma=None,
    alpha=None,
    max_step_norm=0.3,
    n_steps=8,
    verbose=False,
    backend='pytorch',
    levels=None,
    optimizer='reg_adam',
    optimizer_lr=1.0,
    bootstrap_mode='antithetic',
    seed=42,
    **kwargs
):
    """
    SyNGS ("geodesic shooting") registration of ``moving`` to ``fixed`` -- ``syntx.syngs``.

    Same calling convention and result as ``syntx.syn`` / ``ants.registration``::

        reg = syntx.syngs(fixed, moving)
        reg['warpedmovout'], reg['fwdtransforms'], reg['invtransforms']

    The deformation is the flow of a smoothed initial velocity field v0 (stationary in the
    default 'transport' mode: ``n_steps`` Euler steps of exp(v0)); a second field gives the
    inverse. See the module docstring. The initial affine comes from ``syntx.robust_affine``
    unless ``initial_transform`` is given. 3-D defaults (``max_step_norm`` 0.3, ``alpha``
    0.675) were tuned on Mindboggle pairs 77 / 44 / 0
    (docs/provenance/tuning/syngs_2026-09-30.md).

    Parameters
    ----------
    fixed, moving : ANTsImage
        2-D or 3-D.
    initial_transform : str, list of str, ANTsTransform, 'identity' / False, or None
        None: ``syntx.robust_affine`` (``affine_dof``, ``affine_mode``, ``affine_seed``).
    affine_dof : {'affine', 'rigid'}, default 'affine'
    affine_mode : str, default 'pytorch'
    affine_seed : int or None
    syn_metric : str, default 'cc2'
        'cc2' (squared local NCC), 'lncc', 'mattes' / 'mattes_mi', 'mse', 'box_lncc'.
        ``similarity_metric=`` is an alias.
    syn_sampling : int, default 2
        Local-correlation radius (window 2 * syn_sampling + 1).
    reg_iterations : list of int, default None
        Iterations per level. None: [60, 40, 20] (3-D), [60, 60, 40, 20] (2-D); the benchmark
        uses [100, 100, 20].
    levels : list of int, default None
        Pyramid shrink factors. None: [2**(L-1), ..., 1] for L = len(reg_iterations), else
        [4, 2, 1] (3-D) / [8, 4, 2, 1] (2-D).
    alpha : float or None, default None
        Spectral strength (larger = smoother; 0 = no smoothing). None:
        ``SYNGS_DEFAULT_ALPHA[dim]`` (3-D 0.675, 2-D 0.06). Spectral regularisers only
        (raises with 'gaussian' / 'bspline'); no aliases: ``sobolev_alpha`` / ``dsti_alpha`` raise TypeError.
        ``integrate_momentum`` uses the same default.
    flow_sigma : float or None, default None
        'gaussian' / 'bspline' regulariser only (None = 3.0, 0 = off); raises with the
        spectral regularisers.
    n_steps : int, default 8
        Euler steps of the shooting.
    optimizer : str, default 'reg_adam'
        'reg_adam' (Adam with the regulariser applied to each step), 'adam', 'adamw', 'sgd';
        other values use LARS.
    optimizer_lr : float or None, default 1.0
        Learning rate (None: 0.6).
    max_step_norm : float or None, default 0.3
        Largest per-iteration update (voxels). None: ``grad_step``.
    grad_step : float, default 0.25
        Used only as ``max_step_norm`` when that is None.
    bootstrap_mode : {'antithetic', 'none'}, default 'antithetic'
        Also evaluate the loss on a randomly jittered grid and its mirror image.
    seed : int, default 42
    backend : {'pytorch', 'jax'}, default 'pytorch'
    verbose : bool, default False
    **kwargs
        ``regularizer`` ('sobolev' default, 'dsti', 'dsti1', 'gaussian', 'bspline'; others
        raise), ``transport_mode`` ('transport', 'scaled', 'recursive'), ``solver``,
        ``symmetric`` (default True: a second, inverse velocity field), ``inverse_identity_weight``
        (default 0.5), ``bootstrap_orig_weight``, ``bootstrap_jitter_scale``,
        ``spline_distance``, ``mesh_size``, ``device``, ``winsorize_quantiles``,
        ``adam_eps_rel``, ``weight_decay``, ``momentum``, ``smoothing_sigmas``, ``mattes_bins``,
        ``similarity_metric`` (alias of ``syn_metric``). Any other keyword raises TypeError --
        in particular parameters SyNGS does not have (``total_sigma``, ``n_time_steps``,
        ``multipoint_loss``, ``cfl_momentum``, ``interpolator``, ``inverse_*``, ``vgg_*``, ...),
        ``sobolev_alpha`` / ``dsti_alpha`` (use ``alpha``), ``fast_smooth``,
        ``affine_iterations``, ``aff_metric``, ``aff_sampling``.

    Returns
    -------
    dict
        ``'fwdtransforms'`` ([warp, affine], fixed-space points to moving space),
        ``'invtransforms'``, ``'whichtoinvert_inv'``, ``'warpedmovout'``, ``'warpedfixout'``,
        ``'fwd_deformation'`` / ``'inv_deformation'`` (displacement images),
        ``'fwd_momentum'`` / ``'inv_momentum'`` (v0 images) and their files
        ``'fwd_momentum_file'`` / ``'inv_momentum_file'``, ``'inverse_identity_errors'``,
        ``'model'`` (the fitted ``GeodesicShootingModel``), ``'provenance'``.
    """
    t_start = _time.time()
    _allowed = {'regularizer', 'transport_mode', 'solver', 'symmetric', 'inverse_identity_weight',
                'bootstrap_orig_weight', 'bootstrap_jitter_scale', 'spline_distance', 'mesh_size',
                'device', 'winsorize_quantiles', 'adam_eps_rel', 'weight_decay', 'momentum',
                'smoothing_sigmas', 'mattes_bins', 'similarity_metric', 'gaussian_sigma',
                'fast_smooth', 'affine_iterations', 'aff_metric', 'aff_sampling'}
    for _alias in ('sobolev_alpha', 'dsti_alpha'):
        if _alias in kwargs:
            raise TypeError(f"syntx.syngs has no {_alias!r}: use alpha= (the strength of the "
                            "spectral regularizer)")
    _unknown = sorted(set(kwargs) - _allowed)
    if _unknown:
        raise TypeError(f"syntx.syngs() got unexpected keyword(s) {_unknown}; SyNGS does not use "
                        "them (see the docstring for the accepted options)")
    _removed_affine_params = {'affine_iterations', 'aff_metric', 'aff_sampling'} & set(kwargs)
    if _removed_affine_params:
        raise TypeError(
            f"syngs_registration() no longer accepts {sorted(_removed_affine_params)}: the "
            f"inline affine optimizer they configured has been removed in favor of always "
            f"using syntx.robust_affine for initial alignment. Use affine_dof/affine_mode/"
            f"affine_seed instead, or pass initial_transform explicitly to bypass alignment "
            f"entirely."
        )

    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    origin = fixed.origin
    direction = fixed.direction

    moving_shape = moving.shape
    moving_spacing = moving.spacing
    moving_origin = moving.origin
    moving_direction = moving.direction

    if 'similarity_metric' in kwargs:
        syn_metric = kwargs.pop('similarity_metric')

    # Defaults matching syntx.syn() / syntx.tvf()
    if levels is None:
        if reg_iterations is not None:
            num_levels = len(reg_iterations)
            levels = [2**i for i in range(num_levels)][::-1]
        else:
            levels = [4, 2, 1] if dim == 3 else [8, 4, 2, 1]

    if reg_iterations is None:
        reg_iterations = [60, 40, 20] if dim == 3 else [60, 60, 40, 20]

    # --- Parameter relevance validation ---
    if 'fast_smooth' in kwargs:
        raise TypeError("syngs_registration() has no fast_smooth parameter (syngs never used it; "
                        "the smoothing is set by regularizer / alpha). Remove the argument.")
    # Each regulariser has exactly one strength parameter; the other one is rejected rather
    # than silently ignored:  gaussian / bspline -> flow_sigma (Gaussian sigma, 0 = off);
    # sobolev / dsti / dsti1 -> alpha (0 = off).
    reg_mode = str(kwargs.get('regularizer', 'sobolev')).lower()
    _SPECTRAL_REGS = {'sobolev', 'dsti', 'dsti1'}
    _SIGMA_REGS = {'gaussian', 'gauss', 'bspline', 'bsplinesyn'}
    if reg_mode in _SPECTRAL_REGS:
        if flow_sigma is not None:
            raise ValueError(
                f"flow_sigma is only used with regularizer='gaussian' (or 'bspline'); with "
                f"regularizer='{reg_mode}' the smoothing strength is alpha (alpha=0 disables it). "
                f"Got flow_sigma={flow_sigma!r}: omit it, or pass regularizer='gaussian'."
            )
        for _p in ('gaussian_sigma',):
            if kwargs.get(_p) is not None:
                raise ValueError(
                    f"{_p} is only used with regularizer='gaussian'; with regularizer='{reg_mode}' "
                    f"the smoothing strength is alpha. Got {_p}={kwargs[_p]!r}."
                )
        fluid_sigma_actual = None
    elif reg_mode in _SIGMA_REGS:
        for _p, _v in (('alpha', alpha),):
            if _v is not None:
                raise ValueError(
                    f"{_p} is only used with the spectral regularizers (sobolev, dsti, dsti1); with "
                    f"regularizer='{reg_mode}' the smoothing strength is flow_sigma. Got {_p}={_v!r}."
                )
        fluid_sigma_actual = 3.0 if flow_sigma is None else float(flow_sigma)
        if fluid_sigma_actual < 0:
            raise ValueError(f"flow_sigma must be >= 0 (0 disables the smoothing); got {flow_sigma!r}")
    else:
        raise ValueError(f"unknown regularizer {reg_mode!r}; expected one of "
                         f"{sorted(_SPECTRAL_REGS | _SIGMA_REGS)}")


    # Extract initial transform (Single Interpolation Policy)
    init_tx_list = []
    init_M_phys, init_t_phys = None, None
    if initial_transform is False or (isinstance(initial_transform, str)
                                      and initial_transform.lower() == "identity"):
        init_tx_list = []  # explicit opt-out of initial alignment (as syntx.syn / greedy)
    elif initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
    else:
        from .robust_affine import robust_affine
        aff_res = robust_affine(
            fixed, moving, dof=affine_dof, mode=affine_mode, seed=affine_seed, verbose=verbose
        )
        init_tx_list = aff_res['fwdtransforms']
    init_M_phys, init_t_phys = parse_ants_affine(init_tx_list, dim)

    # Normalize images
    fi_np = fixed.numpy()
    mi_np = moving.numpy()

    winsorize_quantiles = kwargs.pop('winsorize_quantiles', None)
    if winsorize_quantiles is not None:
        lo_f, hi_f = np.quantile(fi_np[fi_np > 0], winsorize_quantiles) if (fi_np > 0).any() else (fi_np.min(), fi_np.max())
        fi_np = np.clip(fi_np, lo_f, hi_f)
        lo_m, hi_m = np.quantile(mi_np[mi_np > 0], winsorize_quantiles) if (mi_np > 0).any() else (mi_np.min(), mi_np.max())
        mi_np = np.clip(mi_np, lo_m, hi_m)

    # NOTE: SyNGS uses z-score of the incoming [0,1] image rather than the project-standard
    # 2%-98% percentile normalization. This is a known inconsistency: changing to [0,1] inputs
    # causes -0.015 Dice regression and folding violations in 3D (validated Pair 44, 2026-09-19)
    # because the optimizer convergence is calibrated for the z-score gradient scale.
    # Fixing this properly requires 3D-native re-tuning of alpha, lr, flow_sigma — tracked as TODO.
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

    grid_shape_zyx = itk_shape_to_tensor_shape(grid_shape)
    moving_shape_zyx = itk_shape_to_tensor_shape(moving_shape)

    if backend.lower() == 'pytorch':
        device_str = kwargs.pop('device', None)
        if device_str is None:
            if torch.cuda.is_available():
                device_str = 'cuda'
            elif torch.backends.mps.is_available():
                device_str = 'mps'
            else:
                device_str = 'cpu'

        I_tensor = image_to_tensor(fi_norm, device=device_str, to_zyx=True)
        J_tensor = image_to_tensor(mi_norm, device=device_str, to_zyx=True)

        model = GeodesicShootingModel(
            dim=dim,
            image_shape=grid_shape_zyx,
            velocity_shape=grid_shape_zyx,
            n_steps=n_steps,
            spacing=spacing,
            origin=origin,
            direction=direction.tolist() if hasattr(direction, 'tolist') else direction,
            moving_shape=moving_shape_zyx,
            moving_spacing=moving_spacing,
            moving_origin=moving_origin,
            moving_direction=moving_direction.tolist() if hasattr(moving_direction, 'tolist') else moving_direction,
            fluid_sigma=fluid_sigma_actual,
            solver=kwargs.pop('solver', 'euler'),
            similarity_metric=syn_metric,
            alpha=alpha,
            symmetric=bool(kwargs.pop('symmetric', True)),
            inverse_identity_weight=float(kwargs.pop('inverse_identity_weight', 0.50)),
            regularizer=kwargs.pop('regularizer', 'sobolev'),
            spline_distance=kwargs.pop('spline_distance', None),
            mesh_size=kwargs.pop('mesh_size', None),
            bootstrap_mode=kwargs.pop('bootstrap_mode', bootstrap_mode),
            bootstrap_orig_weight=float(kwargs.pop('bootstrap_orig_weight', 0.50)),
            bootstrap_jitter_scale=float(kwargs.pop('bootstrap_jitter_scale', 0.25)),
            transport_mode=kwargs.pop('transport_mode', 'transport'),
            seed=seed,
        ).to(device_str)

        # Single Interpolation Invariant: absorb initial transform into T_init
        if init_M_phys is not None:
            with torch.no_grad():
                dtype_dev = torch.float32
                H_x = compute_grid_to_physical_reference_matrix(fixed.shape, fixed.spacing, fixed.origin, fixed.direction, device=device_str, dtype=dtype_dev)
                H_y = compute_grid_to_physical_reference_matrix(moving.shape, moving.spacing, moving.origin, moving.direction, device=device_str, dtype=dtype_dev)

                T_phys = torch.eye(dim + 1, device=device_str, dtype=dtype_dev)
                T_phys[:dim, :dim] = init_M_phys.to(device=device_str, dtype=dtype_dev)
                T_phys[:dim, dim] = init_t_phys.to(device=device_str, dtype=dtype_dev)

                T_init = torch.inverse(H_y) @ T_phys @ H_x
                model.affine.T_init = T_init

            init_tx_list = []
            if verbose:
                print("[SyNGS] Initialized affine from initial_transform (T_init absorbed)")

        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            similarity_metric=syn_metric,
            lr=optimizer_lr,
            max_step_norm=max_step_norm if max_step_norm is not None else grad_step,
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            moving_spacing=moving_spacing,
            moving_origin=moving_origin,
            moving_direction=moving_direction,
            lncc_radius=syn_sampling,
            optimizer_type=optimizer,
            cfl_step=grad_step,
            seed=seed,
            **kwargs
        )

        with torch.no_grad():
            fwd_disp = model.get_forward_warp(image_shape=grid_shape_zyx)
            inv_disp = model.get_inverse_warp(image_shape=grid_shape_zyx)
            from .core.inverse import calculate_inverse_identity_error
            try:
                inv_err_calc = calculate_inverse_identity_error(
                    fwd_disp, inv_disp, spacing=spacing, origin=origin, direction=direction
                )
                inv_err_dict = {
                    'mean': float(inv_err_calc['mean_error']),
                    'max': float(inv_err_calc['max_error']),
                    'phi_1': {
                        'mean': float(inv_err_calc['mean_error']),
                        'max': float(inv_err_calc['max_error']),
                        # like registration(): lets benchmarks report interior / p95 statistics
                        'max_error': float(inv_err_calc['max_error']),
                        'mean_error': float(inv_err_calc['mean_error']),
                        'error_map': inv_err_calc.get('error_map'),
                    }
                }
            except Exception:
                inv_err_dict = {}

            fwd_np = fwd_disp.cpu().squeeze(0).numpy()
            inv_np = inv_disp.cpu().squeeze(0).numpy()
            v0_fwd_np = model.velocity_0_fwd.detach().cpu().squeeze(0).numpy()
            v0_inv_np = model.velocity_0_inv.detach().cpu().squeeze(0).numpy() if model.velocity_0_inv is not None else -v0_fwd_np

        T_grid = model.affine.get_matrix().detach().cpu().numpy()

    elif backend.lower() == 'jax':
        from .syngs_jax import GeodesicShootingModelJAX
        from .syn_jax import get_affine_matrix_jax
        import jax.numpy as jnp

        device_str = 'cpu'
        perm = [0, 1] + list(range(dim + 1, 1, -1))       # (1, 1, x, y, z) -> (1, 1, z, y, x)
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
            fluid_sigma=fluid_sigma_actual if fluid_sigma_actual is not None else 3.0,  # JAX model: unchanged
            elastic_sigma=0.0,
            solver=kwargs.pop('solver', 'euler'),
        )

        if init_M_phys is not None:
            H_x = compute_grid_to_physical_reference_matrix(fixed.shape, fixed.spacing, fixed.origin, fixed.direction, device='cpu', dtype=torch.float32).numpy()
            H_y = compute_grid_to_physical_reference_matrix(moving.shape, moving.spacing, moving.origin, moving.direction, device='cpu', dtype=torch.float32).numpy()

            T_phys = np.eye(dim + 1, dtype=np.float32)
            T_phys[:dim, :dim] = init_M_phys.numpy() if hasattr(init_M_phys, 'numpy') else np.asarray(init_M_phys)
            T_phys[:dim, dim] = init_t_phys.numpy() if hasattr(init_t_phys, 'numpy') else np.asarray(init_t_phys)

            T_init_jax = jnp.array(np.linalg.inv(H_y) @ T_phys @ H_x)
            model.T_init = T_init_jax
            model.affine_params['T_init'] = T_init_jax

            init_tx_list = []
            if verbose:
                print("[SyNGS-JAX] Initialized affine from initial_transform (T_init absorbed)")

        model.fit(
            I_tensor, J_tensor,
            levels=levels,
            epochs_per_level=reg_iterations,
            affine_epochs=0,
            similarity_metric=syn_metric,
            lr=optimizer_lr,
            verbose=verbose,
            fixed_spacing=spacing,
            fixed_origin=origin,
            fixed_direction=direction,
            lncc_radius=syn_sampling,
            optimizer_type=optimizer,
            cfl_step=grad_step,
            **kwargs
        )

        fwd_disp = np.array(model.get_forward_warp(image_shape=grid_shape_zyx))
        inv_disp = np.array(model.get_inverse_warp(image_shape=grid_shape_zyx))
        fwd_np = fwd_disp.squeeze(0)
        inv_np = inv_disp.squeeze(0)
        v0_fwd_np = np.array(model.velocity_0_fwd).squeeze(0)
        v0_inv_np = np.array(model.velocity_0_inv).squeeze(0) if model.velocity_0_inv is not None else -v0_fwd_np

        T_grid = np.array(get_affine_matrix_jax(model.affine_params, dim, 'Affine'))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Export displacement fields using standardized ITK components via syntx.spatial
    fwd_img = disp_tensor_to_itk(fwd_disp, fixed)
    inv_img = disp_tensor_to_itk(inv_disp, fixed)

    fwd_file = tempfile.NamedTemporaryFile(suffix='_syngs_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_syngs_inv_Warp.nii.gz', delete=False).name
    ants.image_write(fwd_img, fwd_file)
    ants.image_write(inv_img, inv_file)

    # Export initial momentum vector fields (v_0 at t=0)
    v0_inv_disp = model.velocity_0_inv if (getattr(model, 'symmetric', False) and getattr(model, 'velocity_0_inv', None) is not None) else -model.velocity_0_fwd
    fwd_mom_img = disp_tensor_to_itk(model.velocity_0_fwd, fixed)
    inv_mom_img = disp_tensor_to_itk(v0_inv_disp, fixed)

    fwd_mom_file = tempfile.NamedTemporaryFile(suffix='_syngs_fwd_Momentum.nii.gz', delete=False).name
    inv_mom_file = tempfile.NamedTemporaryFile(suffix='_syngs_inv_Momentum.nii.gz', delete=False).name
    ants.image_write(fwd_mom_img, fwd_mom_file)
    ants.image_write(inv_mom_img, inv_mom_file)

    # Export affine transform using standardized reference matrix conversion
    M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving)
    affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    tx_fwd, tx_inv = export_ants_affine_transform(M_phys, t_phys, dim=dim, filename=affine_file)

    # Build transform lists (Single Interpolation Invariant)
    if sum(reg_iterations) > 0:
        fwd_transforms = [fwd_file, affine_file] + init_tx_list
        inv_transforms = init_tx_list + [affine_file, inv_file]
        whichtoinvert_inv = [True] * len(init_tx_list) + [True, False]
    else:
        fwd_transforms = [affine_file] + init_tx_list
        inv_transforms = init_tx_list + [affine_file]
        whichtoinvert_inv = [True] * (len(init_tx_list) + 1)

    # Apply single-interpolation composite transforms
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
        'fwd_deformation': fwd_img,
        'inv_deformation': inv_img,
        'fwd_momentum': fwd_mom_img,
        'inv_momentum': inv_mom_img,
        'fwd_momentum_file': fwd_mom_file,
        'inv_momentum_file': inv_mom_file,
        'inverse_identity_errors': inv_err_dict if 'inv_err_dict' in locals() else {},
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
            affine_dof=affine_dof,
            affine_mode=affine_mode,
            affine_seed=affine_seed,
            solver="GS-Euler",
            fluid_sigma=flow_sigma,
            learning_rate=grad_step,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            similarity_metric=syn_metric,
            syn_sampling=syn_sampling,
            levels=levels,
            fixed_shape=tuple(fixed.shape),
            fixed_spacing=tuple(fixed.spacing),
            fixed_orientation=str(fixed.orientation) if hasattr(fixed, 'orientation') else None,
            moving_shape=tuple(moving.shape),
            moving_spacing=tuple(moving.spacing),
            moving_orientation=str(moving.orientation) if hasattr(moving, 'orientation') else None,
            n_steps=n_steps
        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass

    return ret_dict


def integrate_momentum(
    momentum,
    reference_image=None,
    n_steps: int = 8,
    alpha: float = None,
    t_end: float = 1.0,
    return_trajectory: bool = False,
    device: str = None,
):
    """
    Turn a SyNGS initial velocity field (e.g. ``reg['fwd_momentum']``) back into a
    displacement field -- ``syntx.integrate_momentum`` (aliases ``shoot_geodesic``,
    ``momentum_to_deformation``).

    With the defaults (``t_end=1``, no trajectory) this is exactly ``syntx.syngs``'s forward
    shooting (smooth v0 once, ``n_steps`` Euler steps of the stationary field), so a saved
    momentum reproduces the registration's warp when ``n_steps`` and ``alpha`` match.
    ``return_trajectory`` / ``t_end`` use the same scheme with step ``t_end / n_steps``, so the
    trajectory's last element equals the default result for ``t_end=1``.

    Parameters
    ----------
    momentum : ANTsImage (vector), str (file), np.ndarray or torch.Tensor
        The initial velocity field v0 (physical mm per unit time).
    reference_image : ANTsImage, optional
        Grid and geometry; required for arrays / tensors, defaults to ``momentum`` itself.
    n_steps : int, default 8
        Euler steps (``syntx.syngs``'s default).
    alpha : float, optional
        Sobolev strength; default ``default_alpha(dim)`` (as ``syntx.syngs``).
    t_end : float, default 1.0
        Integration end time (0.5 = halfway, > 1 extrapolates).
    return_trajectory : bool, default False
        Return the displacement after every step instead of only the last.
    device : str, optional
        Default: CUDA, else MPS, else CPU.

    Returns
    -------
    ANTsImage, or list of ANTsImage with ``return_trajectory``
        Displacement field(s) phi(t) - Id in ITK physical coordinates.
    """
    if isinstance(momentum, str):
        momentum = ants.image_read(momentum)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    if isinstance(momentum, ants.ANTsImage):
        if reference_image is None:
            reference_image = momentum
        dim = momentum.dimension
        origin = momentum.origin
        spacing = momentum.spacing
        direction = momentum.direction
        v0_t = disp_itk_to_tensor(momentum, device=device)
        grid_shape_zyx = itk_shape_to_tensor_shape(reference_image.shape)
    elif isinstance(momentum, np.ndarray):
        if reference_image is None:
            raise ValueError("reference_image (ANTsImage) must be provided when momentum is a numpy array.")
        dim = reference_image.dimension
        origin = reference_image.origin
        spacing = reference_image.spacing
        direction = reference_image.direction
        if momentum.shape[-1] != dim:
            raise ValueError(f"momentum last dimension {momentum.shape[-1]} must match dim {dim}")
        if momentum.shape[:dim] == reference_image.shape:
            v0_t = disp_itk_to_tensor(momentum, device=device)
        else:
            v0_t = torch.tensor(momentum, dtype=torch.float32, device=device)
            while v0_t.ndim < dim + 2:
                v0_t = v0_t.unsqueeze(0)
        grid_shape_zyx = itk_shape_to_tensor_shape(reference_image.shape)
    elif isinstance(momentum, torch.Tensor):
        if reference_image is None:
            raise ValueError("reference_image (ANTsImage) must be provided when momentum is a torch Tensor.")
        dim = reference_image.dimension
        origin = reference_image.origin
        spacing = reference_image.spacing
        direction = reference_image.direction
        v0_t = momentum.to(device=device, dtype=torch.float32)
        while v0_t.ndim < dim + 2:
            v0_t = v0_t.unsqueeze(0)
        grid_shape_zyx = itk_shape_to_tensor_shape(reference_image.shape)
    else:
        raise TypeError(f"Unsupported momentum type: {type(momentum)}")

    if alpha is None:
        alpha = default_alpha(dim)

    model = GeodesicShootingModel(
        dim=dim,
        image_shape=grid_shape_zyx,
        velocity_shape=grid_shape_zyx,
        spacing=spacing,
        origin=origin,
        direction=direction,
        n_steps=n_steps,
        alpha=alpha,
        symmetric=False
    ).to(device)
    model.velocity_0_fwd = nn.Parameter(v0_t)

    # If t_end == 1.0 and not return_trajectory, standard shoot:
    if not return_trajectory and math.isclose(t_end, 1.0):
        with torch.no_grad():
            disp_tensor = model.get_forward_warp(image_shape=grid_shape_zyx)
        return disp_tensor_to_itk(disp_tensor, reference_image)

    # Multi-step trajectory or custom t_end integration:
    dt = float(t_end) / float(n_steps)
    phys_grid = get_physical_grid_torch(grid_shape_zyx, spacing, origin, direction, device=device, dtype=torch.float32)
    spacing_rev, origin_rev, direction_rev = reverse_metadata(spacing, origin, direction)

    shape_t = torch.tensor(list(grid_shape_zyx), device=device, dtype=torch.float32)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=torch.float32)
    origin_t = torch.tensor(origin_rev, device=device, dtype=torch.float32)
    direction_t = torch.tensor(direction_rev, device=device, dtype=torch.float32)
    meta = (shape_t, spacing_t, origin_t, direction_t)

    trajectory = []
    with torch.no_grad():
        v = model.apply_green_operator(v0_t, grid_shape_zyx, spacing_rev)
        disp = torch.zeros_like(phys_grid)

        if return_trajectory:
            trajectory.append(disp_tensor_to_itk(disp, reference_image))

        v_cf = torch.movedim(v, -1, 1)          # stationary smoothed v0, channel-first for sampling
        for step in range(n_steps):
            phi_curr = phys_grid + disp
            phi_norm = physical_to_normalized_torch_cached(phi_curr, shape_t, spacing_t, origin_t, direction_t)
            disp = disp + dt * sample_field_cf(v_cf, phi_norm)
            if return_trajectory:
                trajectory.append(disp_tensor_to_itk(disp, reference_image))

        if return_trajectory:
            return trajectory
        else:
            return disp_tensor_to_itk(disp, reference_image)


shoot_geodesic = integrate_momentum
momentum_to_deformation = integrate_momentum
