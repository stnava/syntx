"""
Greedy (one-directional) diffeomorphic registration -- ``syntx.greedy``.

A single forward displacement field u, refined at every iteration by composition with a
small smoothed update v: (Id + u_new) = (Id + u) o (Id + v), the ANTs "Greedy SyN" scheme
without the symmetric midpoint. Each iteration costs two resamplings (warped image, field
composition), so it is much faster than ``syntx.syn``; there is no inverse unless
``return_inverse=True`` (then computed afterwards by a fixed-point solve).

The initial affine (``syntx.robust_affine`` unless given) seeds the optimization;
the result is exported as the standard ANTs transform list ``[warp, affine]`` (or ``[warp]``
if ``initial_transform=False``), adhering to GEMINI.md Rule 2 and ANTs convention so
``ants.apply_transforms(fixed, moving, reg['fwdtransforms'])`` applies the composed transform
in one interpolation.
"""

import os
import time
import math
import tempfile
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ants

from .spatial import (
    disp_tensor_to_itk,
    itk_shape_to_tensor_shape,
    export_ants_affine_transform,
    normalized_to_physical_disp,
)
from .core.initial_warp import resolve_initial_transform_and_warp
from .core.affine import parse_ants_affine
from .core.grid import compose_grids, resize_field, sample_field_cf
from .core.losses import (
    get_similarity_loss,
    parse_similarity_metric,
)
from .core.losses.cross_correlation import _box_pool, box_mean_nd
from .core.smoothing import separable_gaussian_filter
from .core.inverse import update_inverse_field_nd_anderson
from .core.pipeline import auto_detect_device, cleanup_gpu
from .core.convergence import DEFAULT_SLOPE_THRESHOLD, LossSlopeStop
from .core.utils import require_finite_images
from .robust_affine import robust_affine
from .pyramid import build_pyramid


def _build_torch_affine_matrix(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    M_phys: np.ndarray,
    t_phys: np.ndarray,
    device: torch.device
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Constructs the PyTorch affine grid matrix theta (shape [1, dim, dim+1]) mapping
    fixed normalized [-1, 1] coordinates to moving normalized [-1, 1] coordinates.
    
    Coordinates:
        x_phys = P_F @ [x_norm, 1]^T
        y_phys = T_phys @ x_phys
        y_norm = P_M^{-1} @ y_phys
        => y_norm = (P_M^{-1} @ T_phys @ P_F) @ x_norm
    """
    dim = fixed.dimension

    D_F = np.array(fixed.direction).reshape(dim, dim)
    S_F = np.array(fixed.spacing)
    O_F = np.array(fixed.origin)
    N_F = np.array(fixed.shape)

    P_F = np.eye(dim + 1, dtype=np.float64)
    P_F[:dim, :dim] = D_F @ np.diag(S_F) @ np.diag((N_F - 1) / 2.0)
    P_F[:dim, dim] = D_F @ (S_F * (N_F - 1) / 2.0) + O_F

    D_M = np.array(moving.direction).reshape(dim, dim)
    S_M = np.array(moving.spacing)
    O_M = np.array(moving.origin)
    N_M = np.array(moving.shape)

    P_M = np.eye(dim + 1, dtype=np.float64)
    P_M[:dim, :dim] = D_M @ np.diag(S_M) @ np.diag((N_M - 1) / 2.0)
    P_M[:dim, dim] = D_M @ (S_M * (N_M - 1) / 2.0) + O_M

    T_phys = np.eye(dim + 1, dtype=np.float64)
    T_phys[:dim, :dim] = M_phys
    T_phys[:dim, dim] = t_phys

    T_norm = np.linalg.inv(P_M) @ T_phys @ P_F
    theta = torch.from_numpy(T_norm[:dim, :]).float().unsqueeze(0).to(device)
    return theta, P_F, P_M


def _convert_composed_grid_to_ants_displacement(
    total_target_grid: torch.Tensor,
    fixed: ants.ANTsImage,
    P_F: np.ndarray,
    P_M: Optional[np.ndarray] = None,
    restrict_transformation: Optional[Sequence[float]] = None,
) -> ants.ANTsImage:
    """
    Converts a target coordinate grid in normalized coordinates
    into an ANTs-compatible ITK physical displacement vector field defined on the fixed image.
    If P_M is None, target coordinates are in fixed space (pure non-linear field).
    
    Displacement:
        W_phys(x_phys) = y_phys - x_phys
    """
    if P_M is None:
        P_M = P_F
    dim = fixed.dimension
    grid_np = total_target_grid[0].detach().cpu().numpy()
    
    if dim == 2:
        Y, X, _ = grid_np.shape
        grid_flat = grid_np.reshape(-1, 2)
        grid_homo = np.concatenate([grid_flat, np.ones((len(grid_flat), 1))], axis=-1)
        y_phys = (P_M @ grid_homo.T).T[:, :2]

        y_n = np.linspace(-1, 1, Y)
        x_n = np.linspace(-1, 1, X)
        mesh_y, mesh_x = np.meshgrid(y_n, x_n, indexing='ij')
        fixed_norm_flat = np.stack([mesh_x.flatten(), mesh_y.flatten()], axis=-1)
        fixed_homo = np.concatenate([fixed_norm_flat, np.ones((len(fixed_norm_flat), 1))], axis=-1)
        x_phys = (P_F @ fixed_homo.T).T[:, :2]

        disp_phys = (y_phys - x_phys).reshape(Y, X, 2)
        disp_itk = np.transpose(disp_phys, (1, 0, 2)).copy().astype(np.float32)
    elif dim == 3:
        Z, Y, X, _ = grid_np.shape
        grid_flat = grid_np.reshape(-1, 3)
        grid_homo = np.concatenate([grid_flat, np.ones((len(grid_flat), 1))], axis=-1)
        y_phys = (P_M @ grid_homo.T).T[:, :3]

        z_n = np.linspace(-1, 1, Z)
        y_n = np.linspace(-1, 1, Y)
        x_n = np.linspace(-1, 1, X)
        mesh_z, mesh_y, mesh_x = np.meshgrid(z_n, y_n, x_n, indexing='ij')
        fixed_norm_flat = np.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], axis=-1)
        fixed_homo = np.concatenate([fixed_norm_flat, np.ones((len(fixed_norm_flat), 1))], axis=-1)
        x_phys = (P_F @ fixed_homo.T).T[:, :3]

        disp_phys = (y_phys - x_phys).reshape(Z, Y, X, 3)
        disp_itk = np.transpose(disp_phys, (2, 1, 0, 3)).copy().astype(np.float32)
    else:
        raise ValueError(f"Unsupported spatial dimension: {dim}")

    if restrict_transformation is not None:
        for i, w in enumerate(restrict_transformation):
            if w == 0.0:
                disp_itk[..., i] = 0.0

    disp_img = ants.from_numpy(
        disp_itk,
        origin=fixed.origin,
        spacing=fixed.spacing,
        direction=fixed.direction,
        has_components=True
    )
    return disp_img


from .core.smoothing import gaussian_1d_compact, separable_1d_filter

_separable_1d_filter = separable_1d_filter


class _Conv1x1AffineGrid(torch.autograd.Function):
    """Memory-lean closed-form evaluation of ``clamp(coords, -1, 1) @ M^T + t`` using 1x1
    ``F.conv1d`` on MPS (avoiding tall-skinny MPS GEMM slowdown and 144 MB ``affine_grid_cf``
    storage) and direct matmul on CPU/CUDA, saving ``M`` and the ``[-1, 1]`` clamp mask for backward.
    """

    @staticmethod
    def forward(ctx, coords: torch.Tensor, M: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        in_bounds = (coords >= -1.0) & (coords <= 1.0)
        ctx.save_for_backward(M, in_bounds)
        dim = M.shape[0]
        c = coords.clamp(-1.0, 1.0)
        if coords.device.type == 'mps':
            cf = c.view(1, -1, dim).transpose(1, 2).contiguous()
            out_cf = F.conv1d(cf, M.view(dim, dim, 1), t.view(dim))
            return out_cf.transpose(1, 2).contiguous().view_as(coords)
        return torch.addmm(t.view(1, dim), c.reshape(-1, dim), M.t()).view_as(coords)

    @staticmethod
    def backward(ctx, go: torch.Tensor):
        M, in_bounds = ctx.saved_tensors
        dim = M.shape[0]
        if go.device.type == 'mps':
            go_cf = go.reshape(1, -1, dim).transpose(1, 2).contiguous()
            gin_cf = F.conv1d(go_cf, M.t().contiguous().view(dim, dim, 1))
            gin = gin_cf.transpose(1, 2).contiguous().view_as(go)
        else:
            gin = (go.reshape(-1, dim) @ M).view_as(go)
        return gin * in_bounds.to(gin.dtype), None, None


class GreedyRegistrationModel(nn.Module):
    """
    The PyTorch model behind ``syntx.greedy`` (see the module docstring). Most users call
    ``syntx.greedy``; ``fit`` returns the displacement field in normalised [-1, 1]
    coordinates, (1, *spatial, dim), tensor (z, y, x) spatial order, (x, y, z) components.

    Parameters
    ----------
    dim : {2, 3}, default 3
    learning_rate : float, default 0.5625
        Step size (``syntx.greedy`` passes 0.5625): the update's largest displacement is
        learning_rate / (grid size - 1) of the normalised domain.
    flow_sigma : float, default 1.8
        Gaussian sigma (voxels) smoothing the similarity gradient.
    total_sigma : float, default 0.28
        Gaussian sigma (voxels) smoothing the whole field after each update (0 = off).
    optimizer : {'adam', 'regadam' / 'reg_adam'}, default 'adam'
        'regadam' additionally smooths the Adam step (``regadam_sigma``, Gaussian voxels).
    regadam_sigma : float, default 0.8
    sobolev_alpha : float, default 0.035
        Sobolev strength used for the RegAdam step when ``regadam_sigma`` is 0.
    lncc_radius : int, default 2
    similarity_metric : str, default 'cc2'
        'cc2' / 'lncc' / 'cc' / 'ncc': local correlation (``squared=True``: squared, as
        'cc2'); 'mse' / 'l2': mean squared error. Other values raise ValueError.
    squared : bool, default True
    anderson, anderson_steps, anderson_m, anderson_freq
        Optional Anderson-accelerated inverse-consistency projection of the field
        (off; 5 steps, memory 5, 'per_scale' or 'posthoc').
    padding_mode : str, default 'border'
    beta1, beta2, eps : Adam constants (0.9, 0.99, 1e-8).
    device : torch.device, optional (default CPU)
    """
    def __init__(
        self,
        dim: int = 3,
        learning_rate: float = 0.5625,
        flow_sigma: float = 1.8,
        total_sigma: float = 0.28,
        optimizer: str = 'adam',
        regularizer: str = 'gaussian',
        regadam_sigma: float = 0.8,
        sobolev_alpha: float = 0.035,
        lncc_radius: int = 2,
        similarity_metric: str = 'cc2',
        squared: bool = True,
        num_bins: int = 32,
        mattes_bins: Optional[int] = None,
        sampling_percentage: Optional[float] = None,
        anderson: bool = False,
        anderson_steps: int = 5,
        anderson_m: int = 5,
        anderson_freq: str = 'per_scale',
        padding_mode: str = 'border',
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        restrict_transformation: Optional[Union[Sequence[float], Tuple[float, ...]]] = None,
        device: Optional[torch.device] = None,
        extra_reg_kwargs: Optional[Dict[str, Any]] = None,
        convergence_slope: Optional[float] = DEFAULT_SLOPE_THRESHOLD,
        digital_diffeomorphism: bool = True,
        digital_diffeomorphism_floor: float = 0.05,
        digital_diffeomorphism_steps: int = 25,
    ):
        super().__init__()
        self.digital_diffeomorphism = bool(digital_diffeomorphism)
        self.digital_diffeomorphism_floor = float(digital_diffeomorphism_floor)
        self.digital_diffeomorphism_steps = int(digital_diffeomorphism_steps)
        self.stop = LossSlopeStop(slope_threshold=convergence_slope)   # the shared deformable stopping rule (core.convergence)
        self.convergence = []
        self.dim = dim
        if restrict_transformation is not None:
            rt = [float(w) for w in restrict_transformation]
            if len(rt) != dim:
                raise ValueError(
                    f"restrict_transformation must have length {dim} (one weight per physical axis, "
                    f"XYZ order), got length {len(rt)}: {restrict_transformation!r}"
                )
            for w in rt:
                if not (0.0 <= w <= 1.0):
                    raise ValueError(
                        f"restrict_transformation weights must be in [0, 1], got {rt!r}"
                    )
            self.restrict_transformation = tuple(rt)
        else:
            self.restrict_transformation = None
        self.learning_rate = learning_rate
        self.flow_sigma = flow_sigma
        self.total_sigma = total_sigma
        self.optimizer = optimizer.lower()
        self.regularizer = regularizer.lower()
        self.extra_reg_kwargs = extra_reg_kwargs or {}
        self.regadam_sigma = regadam_sigma
        self.sobolev_alpha = sobolev_alpha
        self.lncc_radius = lncc_radius
        self.window_size = 2 * lncc_radius + 1

        sim_lower = similarity_metric.lower()
        if '_' in sim_lower and sim_lower.startswith(('mattes', 'mi')):
            parts = sim_lower.split('_')
            if parts[-1].isdigit():
                num_bins = int(parts[-1])
                similarity_metric = '_'.join(parts[:-1])

        if mattes_bins is not None:
            num_bins = int(mattes_bins)

        self.similarity_metric = similarity_metric.lower()
        self.loss_config = parse_similarity_metric(
            self.similarity_metric,
            window_size=self.window_size,
            num_bins=int(num_bins),
            squared=squared,
            sampling_percentage=sampling_percentage,
            fixed_range=(0.0, 1.0),
            auto_mask=False,
        )
        self.squared = self.loss_config.squared
        self.num_bins = int(self.loss_config.num_bins)
        self.sampling_percentage = self.loss_config.sampling_percentage
        self.anderson = anderson
        self.anderson_steps = anderson_steps
        self.anderson_m = anderson_m
        self.anderson_freq = anderson_freq.lower()
        self.padding_mode = padding_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device or torch.device('cpu')
        self.loss_fn = get_similarity_loss(self.loss_config).to(self.device)
        self.loss_fn._is_default_greedy_lncc = (
            self.loss_config.metric in ('cc2', 'lncc')
            and not bool(getattr(self.loss_config, 'use_analytical_gradients', False))
            and not bool(getattr(self.loss_config, 'single_pass', False))
            and int(getattr(self.loss_config, 'metric_chunks', 1)) == 1
        )
        self.loss_history = []
        self.warp = None
        self.theta = None

    def fit(
        self,
        fixed_tensor: torch.Tensor,
        moving_tensor: torch.Tensor,
        theta: torch.Tensor,
        scales: List[int],
        iterations: List[int],
        fixed_mask_tensor: Optional[torch.Tensor] = None,
        verbose: bool = False,
        spacing: Optional[Sequence[float]] = None,
        moving_spacing: Optional[Sequence[float]] = None,
        origin: Optional[Sequence[float]] = None,
        direction: Optional[Any] = None,
        moving_origin: Optional[Sequence[float]] = None,
        moving_direction: Optional[Any] = None,
        initial_warp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-scale optimisation of the field. ``fixed_tensor`` / ``moving_tensor``:
        (1, 1, *spatial); ``theta``: (1, dim, dim + 1) affine in normalised coordinates
        (fixed -> moving); ``scales`` / ``iterations``: shrink factor and iterations per level.
        ``spacing``, ``origin``, ``direction`` (ITK x, y, z order) are required and feed the
        central ``deformable`` pyramid and physical-space regularisers. ``moving_spacing``,
        ``moving_origin``, ``moving_direction`` default to the fixed geometry when omitted.
        ``initial_warp``: optional normalised ``[-1, 1]`` displacement field ``(1, *spatial, dim)``
        in tensor ``(z, y, x)`` spatial order and ``(x, y, z)`` component order to warm-start
        optimisation at the coarsest scale.
        Returns (and stores as ``self.warp``) the field; ``self.loss_history`` keeps the loss.
        """
        from .core.regularizers import (
            _parse_direction,
            _require_spacing,
            _validate_origin,
            resolve_required_image_geometry,
        )

        self.theta = theta
        full_shape = fixed_tensor.shape[2:]
        dim = self.dim
        mode = 'trilinear' if dim == 3 else 'bilinear'

        # Local helper: smooth a last-channel field [B, *spatial, dim] with 1-D Gaussians.
        # Encapsulates the channel-first permutation required by _separable_1d_filter.
        def smooth_field(f, gaussians):
            return _separable_1d_filter(torch.movedim(f, -1, 1), gaussians).movedim(1, -1)

        sp_f, orig_f, dir_f = resolve_required_image_geometry(
            fixed_tensor,
            spacing,
            origin,
            direction,
            dim,
            "fixed",
            "GreedyRegistrationModel.fit",
            spacing_name="spacing",
            origin_name="origin",
            direction_name="direction",
        )
        sp_m = _require_spacing(moving_spacing, dim, "GreedyRegistrationModel.fit") if moving_spacing is not None else sp_f
        orig_m = _validate_origin(moving_origin, dim, "GreedyRegistrationModel.fit") if moving_origin is not None else orig_f
        dir_m = _parse_direction(moving_direction, dim, "GreedyRegistrationModel.fit") if moving_direction is not None else dir_f

        # the central pyramid: each image on its own grid, the warp lives on the FIXED level grid
        fixed_lv = build_pyramid(fixed_tensor, sp_f, scales, 'deformable', origin=orig_f, direction=dir_f)
        moving_lv = build_pyramid(moving_tensor, sp_m, scales, 'deformable', origin=orig_m, direction=dir_m, reference_spacing=sp_f)
        mask_lv = build_pyramid(fixed_mask_tensor, sp_f, scales, 'deformable', origin=orig_f, direction=dir_f) if fixed_mask_tensor is not None else None

        # Initialize displacement field u at the coarsest scale
        init_size = list(fixed_lv[0].shape)
        if initial_warp is not None:
            iw = initial_warp.to(device=self.device, dtype=torch.float32)
            if iw.ndim == dim + 1:
                iw = iw.unsqueeze(0)
            if iw.ndim != dim + 2 or iw.shape[-1] != dim:
                raise ValueError(
                    f"initial_warp must have shape (1, *spatial, {dim}) or (*spatial, {dim}), "
                    f"got {tuple(initial_warp.shape)}"
                )
            if list(iw.shape[1:-1]) != init_size:
                warp = resize_field(iw, init_size, mode=mode)
            else:
                warp = iw.clone()
            if self.restrict_transformation is not None:
                restr_init = torch.tensor(
                    self.restrict_transformation, device=self.device, dtype=warp.dtype
                ).view(1, *([1] * dim), dim)
                warp = warp * restr_init
        else:
            warp = torch.zeros((1, *init_size, dim), dtype=torch.float32, device=self.device)
        exp_avg = torch.zeros_like(warp)
        exp_avg_sq = torch.zeros_like(warp)

        self.loss_history = []
        self.stop.records.clear()

        for level_idx, (scale, n_iters) in enumerate(zip(scales, iterations)):
            if n_iters <= 0:
                continue
            self.stop.start_level(level_idx, n_iters)

            size_down = list(fixed_lv[level_idx].shape)
            fi_down = fixed_lv[level_idx].tensor
            mi_down = moving_lv[level_idx].tensor
            mask_down = mask_lv[level_idx].tensor if mask_lv is not None else None

            # Interpolate warp and Adam moments to current scale via centralized resize_field
            if list(warp.shape[1:-1]) != size_down:
                warp      = resize_field(warp,      size_down, mode=mode)
                exp_avg   = resize_field(exp_avg,   size_down, mode=mode)
                exp_avg_sq = resize_field(exp_avg_sq, size_down, mode=mode)

            grid_shape = [1, 1, *size_down]
            id_grid = F.affine_grid(torch.eye(dim, dim + 1, device=self.device)[None], grid_shape, align_corners=True)
            M_theta = self.theta[0, :, :dim].contiguous()
            t_theta_flat = self.theta[0, :, dim].contiguous()
            is_identity_theta = bool(
                torch.allclose(
                    self.theta[0],
                    torch.eye(dim, dim + 1, device=self.device, dtype=self.theta.dtype),
                    atol=1e-7,
                )
            )
            half_resolution = 1.0 / (max(size_down) - 1)
            step = 0  # Reset Adam warm-up step at each scale level
            restr_mask = (
                torch.tensor(self.restrict_transformation, device=self.device, dtype=warp.dtype).view(1, *([1] * dim), dim)
                if self.restrict_transformation is not None
                else None
            )

            # Precompute constant fixed-image LNCC local mean and variance once per pyramid level
            use_fast_lncc = bool(getattr(self.loss_fn, '_is_default_greedy_lncc', False))
            if use_fast_lncc:
                ws_lvl = min(int(self.window_size), int(min(size_down)))
                if ws_lvl % 2 == 0:
                    ws_lvl = max(1, ws_lvl - 1)
                with torch.no_grad():
                    fi_f = fi_down.float() if fi_down.dtype in (torch.float16, torch.bfloat16) else fi_down
                    _j_mean = _box_pool(fi_f, ws_lvl, count_include_pad=False)
                    _lncc_M_c = fi_f - _j_mean
                    del _j_mean
                    _lncc_J_var = torch.clamp(_box_pool(_lncc_M_c.square(), ws_lvl, count_include_pad=False), min=0.0)
                    _lncc_safe_J_var = torch.clamp(_lncc_J_var, min=1e-6)
                    _lncc_j_valid = ((_lncc_J_var > 1e-6) & (mask_down > 0.5)) if mask_down is not None else None
                    del _lncc_J_var
            else:
                ws_lvl = self.window_size
                _lncc_M_c = None
                _lncc_safe_J_var = None
                _lncc_j_valid = None

            grad_gaussians = [gaussian_1d_compact(self.flow_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if self.flow_sigma > 0 else None
            reg_gaussians = [gaussian_1d_compact(self.regadam_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if (self.optimizer in ('regadam', 'reg_adam') and self.regadam_sigma > 0) else None
            warp_gaussians = [gaussian_1d_compact(self.total_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if self.total_sigma > 0 else None

            if self.regularizer not in ('gaussian', 'gauss', 'none', 'identity'):
                from .core.regularizers import get_regularizer
                grad_reg_fn = get_regularizer(
                    self.regularizer,
                    alpha=self.sobolev_alpha,
                    sobolev_alpha=self.sobolev_alpha,
                    spacing=fixed_lv[level_idx].spacing,
                    origin=fixed_lv[level_idx].origin,
                    direction=fixed_lv[level_idx].direction,
                    vector_frame="voxel",
                    fluid_sigma=self.flow_sigma,
                    mask=mask_down,
                    fixed_mask=mask_down,
                    **self.extra_reg_kwargs,
                )
            else:
                grad_reg_fn = None

            _stop_pending: List[Tuple[int, torch.Tensor]] = []
            for it in range(n_iters):
                warp_param = warp.detach().requires_grad_(True)

                # Correct A∘(Id+u) composition: deform in fixed-space coords (clamped to [-1, 1]
                # matching sample_field_cf's padding_mode='border'), then map through the affine.
                if is_identity_theta:
                    sample_grid = torch.clamp(id_grid + warp_param, -1.0, 1.0)
                else:
                    sample_grid = _Conv1x1AffineGrid.apply(id_grid + warp_param, M_theta, t_theta_flat)
                moved = F.grid_sample(mi_down, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

                if use_fast_lncc:
                    moved_f = moved.float() if moved.dtype in (torch.float16, torch.bfloat16) else moved
                    I_mean = box_mean_nd(moved_f, ws_lvl)
                    F_c = moved_f - I_mean
                    del I_mean
                    c_ch = moved_f.shape[1]
                    sec = box_mean_nd(torch.cat([F_c.square(), F_c * _lncc_M_c], dim=1), ws_lvl)
                    I_var = torch.clamp(sec[:, :c_ch], min=0.0)
                    IJ_cov = sec[:, c_ch:]
                    del sec, F_c
                    safe_I_var = torch.clamp(I_var, min=1e-6)
                    if self.squared:
                        cc_metric = torch.clamp((IJ_cov ** 2) / (safe_I_var * _lncc_safe_J_var + 1e-8), min=0.0, max=1.0)
                    else:
                        cc_metric = torch.clamp(IJ_cov / (torch.sqrt(safe_I_var * _lncc_safe_J_var) + 1e-6), min=-1.0, max=1.0)
                    if mask_down is not None:
                        active_mask_f = ((I_var > 1e-6) & _lncc_j_valid).to(moved_f.dtype)
                        loss = -torch.sum(cc_metric * active_mask_f) / (torch.sum(active_mask_f) + 1e-8)
                    else:
                        loss = -torch.mean(cc_metric)
                    del I_var, IJ_cov, safe_I_var, cc_metric, moved_f
                else:
                    loss = self.loss_fn(moved, fi_down, mask=mask_down)

                loss_t = loss.detach()
                self.loss_history.append(loss_t)
                loss.backward()

                grad = warp_param.grad.detach()
                del loss, moved, sample_grid, warp_param

                with torch.no_grad():
                    if grad_reg_fn is not None:
                        grad = grad_reg_fn(grad.flip(-1)).flip(-1)
                    elif self.flow_sigma > 0 and grad_gaussians is not None:
                        grad = smooth_field(grad, grad_gaussians)

                    step += 1
                    exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                    exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                    del grad

                    b1_corr = 1.0 - self.beta1 ** step
                    b2_corr = 1.0 - self.beta2 ** step
                    b2_sqrt = math.sqrt(b2_corr)
                    step_size = b2_sqrt / b1_corr
                    eps_scaled = self.eps * b2_sqrt
                    raw_update = (exp_avg * step_size).div_(exp_avg_sq.sqrt().add_(eps_scaled))

                    # Apply RegAdam quotient smoothing if elected
                    if self.optimizer in ('regadam', 'reg_adam'):
                        if grad_reg_fn is not None:
                            u_reg = grad_reg_fn(raw_update.flip(-1)).flip(-1)
                        elif reg_gaussians is not None:
                            u_reg = smooth_field(raw_update, reg_gaussians)
                        elif self.sobolev_alpha > 0:
                            from .core.smoothing import apply_sobolev_green_operator
                            u_reg = apply_sobolev_green_operator(
                                raw_update.squeeze(0),
                                fluid_sigma=self.sobolev_alpha,
                                alpha=self.sobolev_alpha,
                                spacing=fixed_lv[level_idx].spacing,
                            ).unsqueeze(0)
                        else:
                            u_reg = raw_update
                    else:
                        u_reg = raw_update
                    del raw_update

                    # Bound max velocity step
                    gradmax = self.eps + u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                    gradmax = gradmax.reshape(-1, *([1]) * (dim + 1)).clamp(min=1.0)
                    update = -self.learning_rate * half_resolution * (u_reg / gradmax)
                    del u_reg, gradmax
                    if restr_mask is not None:
                        update = update * restr_mask

                    # Eulerian compositive pullback: u_{k+1}(x) = v_k(x) + u_k(x + v_k(x))
                    # compose_grids(warp, id_grid + update) samples warp at (x + update(x)).
                    pulled = compose_grids(warp, id_grid + update.to(warp.dtype))

                    u_new = update + pulled
                    del update, pulled
                    if self.total_sigma > 0 and warp_gaussians is not None:
                        u_new = smooth_field(u_new, warp_gaussians)
                    if restr_mask is not None:
                        u_new = u_new * restr_mask
                    warp = u_new.detach()
                    del u_new

                if verbose and (it % 25 == 0 or it == n_iters - 1):
                    print(f"  [greedy] Level {level_idx} (scale {scale}) Iter {it+1}/{n_iters} | Loss: {loss_t.item():.4f}")

                # Buffer loss samples every 5 iterations and only transfer/test once the window is full
                if (it + 1) % 5 == 0:
                    _stop_pending.append((it, loss_t))
                    if self.stop.enabled and (len(self.stop._losses) + len(_stop_pending)) >= self.stop.window:
                        _vals = torch.stack([t for _, t in _stop_pending]).tolist()
                        _converged_now = False
                        for (_it_p, _), _v_p in zip(_stop_pending, _vals):
                            if self.stop.update(_it_p, float(_v_p)):
                                _converged_now = True
                                break
                        _stop_pending.clear()
                        if _converged_now:
                            break
            if _stop_pending:
                _vals = torch.stack([t for _, t in _stop_pending]).tolist()
                for (_it_p, _), _v_p in zip(_stop_pending, _vals):
                    self.stop.update(_it_p, float(_v_p))
                _stop_pending.clear()
            del _lncc_M_c, _lncc_safe_J_var, _lncc_j_valid
            self.stop.end_level(it + 1)

            # Optional in-loop per-scale Anderson fixed-point projection
            if self.anderson and self.anderson_freq == 'per_scale':
                if verbose:
                    print(f"  [greedy] Applying Anderson projection at scale {scale} ({self.anderson_steps} steps)...")
                warp_inv = update_inverse_field_nd_anderson(warp, None, steps=self.anderson_steps, m=self.anderson_m)
                warp = update_inverse_field_nd_anderson(warp_inv, None, steps=self.anderson_steps, m=self.anderson_m)

        # Final upsample to full shape if required
        if list(warp.shape[1:-1]) != list(full_shape):
            warp = resize_field(warp, list(full_shape), mode=mode)

        # Optional post-hoc Anderson fixed-point projection
        if self.anderson and self.anderson_freq == 'posthoc':
            if verbose:
                print(f"  [greedy] Applying post-hoc Anderson projection ({self.anderson_steps} steps)...")
            warp_inv = update_inverse_field_nd_anderson(warp, None, steps=self.anderson_steps, m=self.anderson_m)
            warp = update_inverse_field_nd_anderson(warp_inv, None, steps=self.anderson_steps, m=self.anderson_m)

        if self.restrict_transformation is not None:
            restr_mask = torch.tensor(
                self.restrict_transformation,
                device=self.device,
                dtype=warp.dtype
            ).view(1, *([1] * dim), dim)
            warp = warp * restr_mask

        if self.loss_history and all(torch.is_tensor(l) for l in self.loss_history):
            self.loss_history = [float(x) for x in torch.stack(self.loss_history).tolist()]
        else:
            self.loss_history = [float(l.item()) if hasattr(l, 'item') else float(l) for l in self.loss_history]
        self.convergence = list(self.stop.records)
        self.warp = warp
        return warp





def greedy_registration(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    reg_iterations: Optional[Union[List[int], Tuple[int, ...]]] = None,
    iterations: Optional[Union[List[int], Tuple[int, ...]]] = None,
    scales: Optional[Union[List[int], Tuple[int, ...]]] = None,
    learning_rate: float = 0.5625,
    flow_sigma: float = 1.8,
    total_sigma: float = 0.28,
    optimizer: str = 'adam',
    regadam_sigma: float = 0.8,
    similarity_metric: str = 'cc2',
    lncc_radius: int = 2,
    anderson: bool = False,
    anderson_steps: int = 5,
    anderson_m: int = 5,
    anderson_freq: str = 'per_scale',
    return_inverse: bool = False,
    initial_transform: Optional[Union[str, List[str], Any, bool]] = None,
    affine_mode: str = 'auto',
    device: Optional[str] = None,
    fixed_mask: Optional[ants.ANTsImage] = None,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    seed: int = 42,
    restrict_transformation: Optional[Union[Sequence[float], Tuple[float, ...]]] = None,
    levels: Optional[Union[List[int], Tuple[int, ...]]] = None,
    initial_warp: Optional[Union[torch.Tensor, ants.ANTsImage, str, os.PathLike]] = None,
    initial_inverse_warp: Optional[Union[torch.Tensor, ants.ANTsImage, str, os.PathLike]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Greedy (one-directional) diffeomorphic registration of ``moving`` to ``fixed`` --
    ``syntx.greedy``.

    ::

        reg = syntx.greedy(fixed, moving)
        reg['warpedmovout'], reg['fwdtransforms']        # [warp, affine] standard ANTs list

    Defaults were tuned on Mindboggle pairs 77 / 44 / 0 (learning_rate 0.375 -> 0.5625,
    docs/provenance/best_parameters.json "syntx.greedy/canonical_2026_11_19").

    Parameters
    ----------
    fixed, moving : ANTsImage
        2-D or 3-D. Both are intensity-normalised (2nd-98th foreground percentile) first.
    reg_iterations, iterations : int or list of int, default None
        Iterations per level. None: [100, 100, 20] (3-D), [100, 100, 100, 50] (2-D).
        Both names are accepted interchangeably.
    scales : int or list of int, default None
        Shrink factor per level. None: [2**(L-1), ..., 1] for L = len(reg_iterations).
    learning_rate : float, default 0.5625
        Step size (``grad_step=`` is an alias); see ``GreedyRegistrationModel``.
    flow_sigma : float, default 1.8
        Gaussian sigma in voxels for smoothing the gradient (not the ITK variance convention
        of ``syntx.syn``).
    total_sigma : float, default 0.28
        Gaussian sigma in voxels for smoothing the whole field after each update.
    optimizer : {'adam', 'regadam', 'reg_adam'}, default 'adam'
        (``optimizer_type=`` is an alias.)
    regadam_sigma : float, default 0.8
        RegAdam step smoothing (voxels).
    similarity_metric : str, default 'cc2'
        'cc2' / 'lncc' / 'cc' / 'ncc' = local correlation; 'mse' / 'l2' = mean squared error;
        other values raise ValueError.
    lncc_radius : int, default 2
        Local-correlation radius (window 2 * radius + 1).
    anderson, anderson_steps, anderson_m, anderson_freq
        Optional Anderson inverse-consistency projection (default off; ``project_inverse=`` /
        ``anderson_projection=`` are aliases for ``anderson``).
    return_inverse : bool, default False
        Also compute the inverse field (fixed-point solve, >= 15 Anderson steps) and return
        it in ``'invtransforms'``. Without it there is no inverse (greedy's inverse metrics
        are NaN in the benchmarks).
    initial_transform : None, False, 'identity', str, list or ANTsTransform
        None: ``syntx.robust_affine(mode=affine_mode)``; False / 'identity': no affine;
        otherwise ANTs transform file(s) -- linear transform(s) or a leading displacement
        field followed by linear transform(s) (e.g. ``[warp_file, affine_file]``). Other
        non-linear combinations (e.g. ``[affine_file, warp_file]`` or composite/BSpline
        transforms) raise ``NotImplementedError``.
    initial_warp : torch.Tensor, ANTsImage, or str, optional
        Dense initial displacement field to warm-start optimization: either a normalised
        ``[-1, 1]`` tensor ``(1, *spatial, dim)`` in tensor ``(z, y, x)`` order and
        ``(x, y, z)`` components, an ITK LPS vector ``ANTsImage``, or a ``.nii.gz``
        displacement field path. Exported as a single composed ``1Warp.nii.gz``.
    affine_mode : str, default 'auto'
        ``syntx.robust_affine`` mode for the automatic alignment.
    device : str, optional
        Default: CUDA, else MPS, else CPU.
    verbose : bool, default False
    outprefix : str, optional
        Write the transforms with this file prefix (default: temporary files).
    seed : int, default 42
    **kwargs
        ``sobolev_alpha`` (0.035), ``padding_mode`` ('border'), ``squared`` (True),
        ``interpolator`` ('linear', for the warped output), ``regularizer`` (only 'gaussian';
        anything else raises ValueError, as does a ``dsti_alpha``), and the aliases
        ``grad_step`` / ``optimizer_type`` / ``project_inverse`` / ``anderson_projection``. Any
        other keyword raises TypeError (they were silently ignored).

    Returns
    -------
    dict
        ``'warpedmovout'`` : ANTsImage warped into fixed space.
        ``'warpedfixout'`` : ANTsImage warped into moving space (if ``return_inverse=True``, else None).
        ``'fwdtransforms'`` : [warp, affine] files mapping fixed-space points to moving space
            (``ants.apply_transforms(fixed, moving, fwdtransforms)``). If no affine was used,
            contains only [warp].
        ``'invtransforms'`` : [affine, inv_warp] files mapping moving-space points to fixed space
            with ``whichtoinvert_inv=[True, False]`` (if ``return_inverse=True``, else []).
        ``'whichtoinvert_inv'`` : list of bools corresponding to ``'invtransforms'``.
        ``'model'`` : GreedyRegistrationModel.
        ``'provenance'`` : dict with runtime and configuration details.
        ``'affine_transform'`` : resolved initial affine transform path or None.
    """
    t0_total = time.time()
    require_finite_images(fixed=fixed, moving=moving)
    dim = fixed.dimension
    torch_device = auto_detect_device(backend='pytorch', requested_device=device)

    # 1. Setup multi-resolution schedule
    if reg_iterations is None:
        reg_iterations = iterations
    if reg_iterations is None and 'iterations' in kwargs:
        reg_iterations = kwargs.pop('iterations')
    elif 'iterations' in kwargs:
        kwargs.pop('iterations')
    if reg_iterations is None:
        # 3D last-stage aligned to syntx.syn()'s newly-updated default (docs/provenance/best_parameters.json,
        # "90pair_population_benchmark_sobolev_mps"); greedy has no benchmarked parameter data of its own,
        # so it is made an interim stand-in match to syn's schedule shape. 2D has no analogous benchmark
        # data in either function and is left unchanged.
        reg_iterations = [100, 100, 20] if dim == 3 else [100, 100, 100, 50]
    elif isinstance(reg_iterations, int):
        reg_iterations = [reg_iterations]

    if scales is None and levels is not None:
        scales = levels

    if scales is None:
        num_levels = len(reg_iterations)
        scales = [2**i for i in range(num_levels)][::-1] if num_levels > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
    elif isinstance(scales, int):
        scales = [scales]

    # 2. Intensity Normalization (Foreground 2nd-98th percentile policy)
    from .benchmark.evaluate import normalize_intensity
    fi_norm = normalize_intensity(fixed)
    mi_norm = normalize_intensity(moving)

    # 3. Initial Affine Alignment & Optional Dense Initial Warp
    t0_aff = time.time()
    raw_init_tx = initial_transform
    if initial_transform is None and initial_warp is None:
        if verbose:
            print("[greedy] Running syntx.robust_affine(mode='auto')...")
        reg_aff = robust_affine(fi_norm, mi_norm, mode=affine_mode, verbose=False, device=torch_device)
        raw_init_tx = reg_aff['fwdtransforms'][0]
    t1_aff = time.time() - t0_aff

    resolved_init = resolve_initial_transform_and_warp(
        fi_norm,
        mi_norm,
        initial_transform=raw_init_tx,
        initial_warp=initial_warp,
        initial_inverse_warp=initial_inverse_warp,
        need_inverse=False,
        device=torch_device,
    )
    aff_tx = resolved_init.aff_tx
    M_phys = resolved_init.M_phys
    t_phys = resolved_init.t_phys
    init_warp_t = resolved_init.u0_norm

    theta, P_F, P_M = _build_torch_affine_matrix(fi_norm, mi_norm, M_phys, t_phys, torch_device)

    # 5. Prepare Image Tensors (ZYX layout)
    fi_np = fi_norm.numpy().T
    mi_np = mi_norm.numpy().T

    fi_t = torch.from_numpy(fi_np).float().unsqueeze(0).unsqueeze(0).to(torch_device)
    mi_t = torch.from_numpy(mi_np).float().unsqueeze(0).unsqueeze(0).to(torch_device)

    mask_t = None
    if fixed_mask is not None:
        mask_np = fixed_mask.numpy().T
        mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(torch_device)

    convergence_slope = kwargs.pop('convergence_slope', DEFAULT_SLOPE_THRESHOLD)   # None / 0 disables the early stop
    # Support aliases and kwargs
    if 'project_inverse' in kwargs:
        anderson = bool(kwargs.pop('project_inverse'))
    if 'anderson_projection' in kwargs:
        anderson = bool(kwargs.pop('anderson_projection'))
    if 'grad_step' in kwargs:
        learning_rate = float(kwargs.pop('grad_step'))
    if 'optimizer_type' in kwargs:
        optimizer = str(kwargs.pop('optimizer_type'))
    if 'optimizer' in kwargs:
        optimizer = str(kwargs.pop('optimizer'))
    regadam_sigma = float(kwargs.pop('regadam_sigma', regadam_sigma))
    sobolev_alpha = float(kwargs.pop('sobolev_alpha', 0.035))
    padding_mode = str(kwargs.pop('padding_mode', 'border'))

    _GREEDY_EXTRA_REG_KEYS = frozenset({
        'beta', 'gamma', 'poisson_ratio', 'darcy_permeability', 'bulk_modulus',
        'num_iters', 'dilatation_weight', 'mask', 'alpha', 'h3_envelope', 's',
    })
    extra_reg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _GREEDY_EXTRA_REG_KEYS}
    if 'alpha' in extra_reg_kwargs:
        sobolev_alpha = float(extra_reg_kwargs['alpha'])

    # --- Parameter relevance validation ---
    reg_mode = str(kwargs.pop('regularizer', 'gaussian')).lower()
    from .core.regularizers import list_regularizers
    _VALID_GREEDY_REGS = set(list_regularizers()) | {'gaussian', 'gauss', 'sobolev'}
    if reg_mode not in _VALID_GREEDY_REGS:
        raise ValueError(f"syntx.greedy: unknown regularizer {reg_mode!r}; expected one of {sorted(_VALID_GREEDY_REGS)}")
    _dsti_alpha_val = kwargs.pop('dsti_alpha', None)
    if _dsti_alpha_val is not None:
        if reg_mode not in ('dsti', 'dsti1', 'dst_i', 'dst1'):
            raise ValueError(
                f"syntx.greedy: dsti_alpha is only supported with DST-I regularizers "
                f"(got regularizer={reg_mode!r})"
            )
        sobolev_alpha = float(_dsti_alpha_val)
    squared = bool(kwargs.pop('squared', True))
    interpolator = kwargs.pop('interpolator', 'linear')

    # Mattes bin aliases: num_bins, mattes_bins, or syn_sampling (ANTs/syntx.syn convention)
    num_bins_val = kwargs.pop('num_bins', None)
    if num_bins_val is None:
        num_bins_val = kwargs.pop('mattes_bins', None)

    if 'syn_sampling' in kwargs:
        syn_samp_val = kwargs.pop('syn_sampling')
        sim_check = str(similarity_metric).lower()
        if sim_check in ('mattes_mi', 'mattes', 'mi', 'mmi') or sim_check.startswith(('mattes', 'mi')):
            if num_bins_val is None:
                num_bins_val = syn_samp_val
        else:
            lncc_radius = int(syn_samp_val)

    cfg = parse_similarity_metric(
        similarity_metric,
        lncc_radius=lncc_radius,
        num_bins=int(num_bins_val) if num_bins_val is not None else 32,
        squared=squared,
        sampling_percentage=kwargs.pop('sampling_percentage', None),
    )
    num_bins = cfg.num_bins
    sampling_percentage = cfg.sampling_percentage
    digital_diffeomorphism = bool(kwargs.pop('digital_diffeomorphism', True))
    digital_diffeomorphism_floor = float(kwargs.pop('digital_diffeomorphism_floor', 0.05))
    digital_diffeomorphism_steps = int(kwargs.pop('digital_diffeomorphism_steps', 25))
    if kwargs:
        raise TypeError(f"syntx.greedy() got unexpected keyword(s) {sorted(kwargs)}")

    # 6. Instantiate & Run Greedy Model
    model = GreedyRegistrationModel(
        dim=dim,
        learning_rate=learning_rate,
        flow_sigma=flow_sigma,
        total_sigma=total_sigma,
        optimizer=optimizer,
        regularizer=reg_mode,
        extra_reg_kwargs=extra_reg_kwargs,
        regadam_sigma=regadam_sigma,
        sobolev_alpha=sobolev_alpha,
        lncc_radius=lncc_radius,
        similarity_metric=similarity_metric,
        squared=squared,
        num_bins=num_bins,
        sampling_percentage=sampling_percentage,
        anderson=anderson,
        anderson_steps=anderson_steps,
        anderson_m=anderson_m,
        anderson_freq=anderson_freq,
        padding_mode=padding_mode,
        restrict_transformation=restrict_transformation,
        device=torch_device,
        convergence_slope=convergence_slope,
        digital_diffeomorphism=digital_diffeomorphism,
        digital_diffeomorphism_floor=digital_diffeomorphism_floor,
        digital_diffeomorphism_steps=digital_diffeomorphism_steps,
    )

    model.has_initial_warp = resolved_init.has_initial_warp
    model.initial_warp_kind = resolved_init.initial_warp_kind

    t0_opt = time.time()
    warp = model.fit(
        fixed_tensor=fi_t,
        moving_tensor=mi_t,
        theta=theta,
        scales=scales,
        iterations=reg_iterations,
        fixed_mask_tensor=mask_t,
        verbose=verbose,
        spacing=tuple(float(x) for x in fixed.spacing),
        moving_spacing=tuple(float(x) for x in moving.spacing),
        origin=tuple(float(x) for x in fixed.origin),
        direction=np.array(fixed.direction, dtype=np.float64),
        moving_origin=tuple(float(x) for x in moving.origin),
        moving_direction=np.array(moving.direction, dtype=np.float64),
        initial_warp=init_warp_t,
    )
    t1_opt = time.time() - t0_opt

    # 7. Non-Linear Displacement Field & Affine Export
    # warp encodes pure non-linear displacement in fixed-space normalized coords u(x).
    # Convert directly on device to physical displacement in fixed ITK LPS space without
    # CPU float64 meshgrid allocations or GPU<->CPU round-trips.
    full_shape = fi_t.shape[2:]
    dim = fi_t.ndim - 2
    with torch.no_grad():
        disp_t = normalized_to_physical_disp(
            warp, full_shape, fixed.spacing, fixed.direction, device=torch_device, dtype=torch.float32
        )
        if restrict_transformation is not None:
            restr_t_zyx = torch.tensor(
                list(reversed([0.0 if float(w) == 0.0 else 1.0 for w in restrict_transformation])),
                device=disp_t.device,
                dtype=disp_t.dtype,
            ).view(1, *([1] * dim), dim)
            disp_t = disp_t * restr_t_zyx
        else:
            restr_t_zyx = None

    # Optional physical inverse and/or digital_diffeomorphism projection
    inv_disp_img = None
    t1_inv = 0.0
    if return_inverse:
        t0_inv = time.time()
        if verbose:
            print("[greedy] Computing physical inverse displacement field with Anderson acceleration...")
        inv_steps = max(anderson_steps, 15)
        if sum(reg_iterations) == 0 and resolved_init.u0_inv_phys is not None:
            inv_disp_t = resolved_init.u0_inv_phys
        else:
            inv_disp_t = update_inverse_field_nd_anderson(
                W_disp=disp_t,
                W_inv_disp=resolved_init.u0_inv_phys,
                steps=inv_steps,
                m=anderson_m,
                spacing=fixed.spacing,
                origin=fixed.origin,
                direction=fixed.direction,
            )
        if digital_diffeomorphism and (sum(reg_iterations) > 0 or init_warp_t is not None):
            from .core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd
            disp_t, inv_disp_t = enforce_digital_diffeomorphism_nd(
                disp_t,
                inv_disp_t,
                spacing=fixed.spacing,
                origin=fixed.origin,
                direction=fixed.direction,
                ice_tol=0.30,
                jac_floor=digital_diffeomorphism_floor,
                steps=digital_diffeomorphism_steps,
                stationary_boundary=(padding_mode == 'border'),
            )
        if restr_t_zyx is not None:
            with torch.no_grad():
                disp_t = disp_t * restr_t_zyx
                inv_disp_t = inv_disp_t * restr_t_zyx
        disp_img = disp_tensor_to_itk(disp_t, fixed)
        inv_disp_img = disp_tensor_to_itk(inv_disp_t, fixed)
        t1_inv = time.time() - t0_inv
    else:
        if digital_diffeomorphism and (sum(reg_iterations) > 0 or init_warp_t is not None):
            from .core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd
            disp_t = enforce_digital_diffeomorphism_nd(
                disp_t,
                None,
                spacing=fixed.spacing,
                origin=fixed.origin,
                direction=fixed.direction,
                jac_floor=digital_diffeomorphism_floor,
                steps=digital_diffeomorphism_steps,
                stationary_boundary=(padding_mode == 'border'),
            )
            if restr_t_zyx is not None:
                with torch.no_grad():
                    disp_t = disp_t * restr_t_zyx
        disp_img = disp_tensor_to_itk(disp_t, fixed)

    # Affine transform handling
    affine_file = None
    if aff_tx is not None:
        if outprefix is not None:
            affine_file = f"{outprefix}0GenericAffine.mat"
            export_ants_affine_transform(M_phys, t_phys, dim=dim, filename=affine_file)
        elif isinstance(aff_tx, str) and os.path.exists(aff_tx):
            affine_file = aff_tx
        elif isinstance(aff_tx, list) and len(aff_tx) == 1 and isinstance(aff_tx[0], str) and os.path.exists(aff_tx[0]):
            affine_file = aff_tx[0]
        else:
            affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
            export_ants_affine_transform(M_phys, t_phys, dim=dim, filename=affine_file)

    # File output paths & fast parallel gzip write + device displacement cache
    from .transforms import (
        _cache_disp_image_and_field,
        _write_ants_images_fast,
        apply_transforms as _apply_transforms,
    )
    _apply_dev = str(torch_device)
    _compress_lvl = 4 if outprefix is not None else 1

    if outprefix is not None:
        os.makedirs(os.path.dirname(outprefix) or '.', exist_ok=True)
        fwd_file = f"{outprefix}1Warp.nii.gz"
    else:
        fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd_warp.nii.gz', delete=False).name

    inv_file = None
    if return_inverse and inv_disp_img is not None:
        if outprefix is not None:
            inv_file = f"{outprefix}1InverseWarp.nii.gz"
        else:
            inv_file = tempfile.NamedTemporaryFile(suffix='_inv_warp.nii.gz', delete=False).name
        _write_ants_images_fast([(disp_img, fwd_file), (inv_disp_img, inv_file)], compresslevel=_compress_lvl)
        _cache_disp_image_and_field(fwd_file, disp_img, _apply_dev)
        _cache_disp_image_and_field(inv_file, inv_disp_img, _apply_dev)
    else:
        _write_ants_images_fast([(disp_img, fwd_file)], compresslevel=_compress_lvl)
        _cache_disp_image_and_field(fwd_file, disp_img, _apply_dev)

    if affine_file is not None:
        if isinstance(aff_tx, list) and len(aff_tx) > 1 and outprefix is None:
            fwd_transforms = [fwd_file] + aff_tx
        else:
            fwd_transforms = [fwd_file, affine_file]
    else:
        fwd_transforms = [fwd_file]

    # 8. Warped Moving Image Output (Single Interpolation directly on native moving image)
    warpedmovout = _apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=fwd_transforms,
        interpolator=interpolator,
        device=_apply_dev,
    )

    # 9. Optional Physical Inverse Transform Export
    inv_transforms = []
    whichtoinvert_inv = []
    warpedfixout = None
    if return_inverse and inv_disp_img is not None and inv_file is not None:
        if affine_file is not None:
            if isinstance(aff_tx, list) and len(aff_tx) > 1 and outprefix is None:
                inv_transforms = aff_tx + [inv_file]
                whichtoinvert_inv = [True] * len(aff_tx) + [False]
            else:
                inv_transforms = [affine_file, inv_file]
                whichtoinvert_inv = [True, False]
        else:
            inv_transforms = [inv_file]
            whichtoinvert_inv = [False]

        warpedfixout = _apply_transforms(
            fixed=moving,
            moving=fixed,
            transformlist=inv_transforms,
            whichtoinvert=whichtoinvert_inv,
            interpolator=interpolator,
            device=_apply_dev,
        )

    total_runtime = time.time() - t0_total

    provenance = {
        'algorithm': 'syntx.greedy',
        'formulation': 'eulerian_compositive',
        'optimizer': optimizer,
        'regadam_sigma': regadam_sigma if optimizer in ('regadam', 'reg_adam') else None,
        'reg_iterations': reg_iterations,
        'scales': scales,
        'learning_rate': learning_rate,
        'flow_sigma': flow_sigma,
        'total_sigma': total_sigma,
        'padding_mode': padding_mode,
        'similarity_metric': similarity_metric,
        'anderson': anderson,
        'anderson_steps': anderson_steps if (anderson or return_inverse) else 0,
        'anderson_freq': anderson_freq if anderson else 'none',
        'return_inverse': return_inverse,
        'digital_diffeomorphism': digital_diffeomorphism,
        'has_initial_warp': resolved_init.has_initial_warp,
        'initial_warp_kind': resolved_init.initial_warp_kind,
        'runtime_total_sec': round(total_runtime, 2),
        'runtime_affine_sec': round(t1_aff, 2),
        'runtime_deformable_sec': round(t1_opt, 2),
        'runtime_inverse_sec': round(t1_inv, 2),
        'device': str(torch_device),
    }

    if verbose:
        print(f"[greedy] Complete in {total_runtime:.2f}s (Affine: {t1_aff:.2f}s, Deformable: {t1_opt:.2f}s, Inverse: {t1_inv:.2f}s)")

    cleanup_gpu(torch_device)

    return {
        'warpedmovout': warpedmovout,
        'warpedfixout': warpedfixout,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': whichtoinvert_inv,
        'model': model,
        'convergence': model.stop.summary(),
        'provenance': provenance,
        # Whatever affine `initial_transform` actually resolved to internally -- a file path
        # from syntx.robust_affine (initial_transform=None), the caller's own supplied
        # transform(s) echoed back, or None (initial_transform was False/'identity', no
        # affine). Exposed so callers (e.g. syntx.build_template) can cache and reuse a
        # freshly-computed affine across repeated calls against a slowly-changing fixed
        # image, instead of re-running robust_affine's search every time.
        'affine_transform': aff_tx,
    }


# Aliases
greedy = greedy_registration
