"""
Greedy (one-directional) diffeomorphic registration -- ``syntx.greedy``.

A single forward displacement field u, refined at every iteration by composition with a
small smoothed update v: (Id + u_new) = (Id + u) o (Id + v), the ANTs "Greedy SyN" scheme
without the symmetric midpoint. Each iteration costs two resamplings (warped image, field
composition), so it is much faster than ``syntx.syn``; there is no inverse unless
``return_inverse=True`` (then computed afterwards by a fixed-point solve).

The initial affine (``syntx.robust_affine`` unless given) is folded into the sampling grid,
and the result is exported as one composed displacement field (affine included), so
``ants.apply_transforms(fixed, moving, reg['fwdtransforms'])`` applies the whole transform in
one interpolation.
"""

import os
import time
import math
import tempfile
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ants

from .spatial import (
    disp_tensor_to_itk,
    itk_shape_to_tensor_shape,
)
from .core.affine import parse_ants_affine
from .core.grid import compose_grids, resize_field, sample_field_cf
from .core.losses import local_ncc_loss_nd
from .core.smoothing import separable_gaussian_filter
from .core.inverse import update_inverse_field_nd_anderson
from .core.pipeline import auto_detect_device, cleanup_gpu
from .robust_affine import robust_affine


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
    P_M: np.ndarray
) -> ants.ANTsImage:
    """
    Converts a total target coordinate grid in moving normalized coordinates
    into an ANTs-compatible ITK physical displacement vector field defined on the fixed image.
    
    Displacement:
        W_phys(x_phys) = y_phys - x_phys
    """
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

    disp_img = ants.from_numpy(
        disp_itk,
        origin=fixed.origin,
        spacing=fixed.spacing,
        direction=fixed.direction,
        has_components=True
    )
    return disp_img


from .core.smoothing import gaussian_1d_compact, separable_1d_filter
from .core.losses import BoxLNCCLoss

_separable_1d_filter = separable_1d_filter


class GreedyRegistrationModel(nn.Module):
    """
    The PyTorch model behind ``syntx.greedy`` (see the module docstring). Most users call
    ``syntx.greedy``; ``fit`` returns the displacement field in normalised [-1, 1]
    coordinates, (1, *spatial, dim), tensor (z, y, x) spatial order, (x, y, z) components.

    Parameters
    ----------
    dim : {2, 3}, default 3
    learning_rate : float, default 0.45
        Step size (``syntx.greedy`` passes 0.375): the update's largest displacement is
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
        learning_rate: float = 0.45,
        flow_sigma: float = 1.8,
        total_sigma: float = 0.28,
        optimizer: str = 'adam',
        regadam_sigma: float = 0.8,
        sobolev_alpha: float = 0.035,
        lncc_radius: int = 2,
        similarity_metric: str = 'cc2',
        squared: bool = True,
        anderson: bool = False,
        anderson_steps: int = 5,
        anderson_m: int = 5,
        anderson_freq: str = 'per_scale',
        padding_mode: str = 'border',
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.learning_rate = learning_rate
        self.flow_sigma = flow_sigma
        self.total_sigma = total_sigma
        self.optimizer = optimizer.lower()
        self.regadam_sigma = regadam_sigma
        self.sobolev_alpha = sobolev_alpha
        self.lncc_radius = lncc_radius
        self.window_size = 2 * lncc_radius + 1
        self.similarity_metric = similarity_metric.lower()
        if self.similarity_metric not in ('cc2', 'lncc', 'cc', 'ncc', 'mse', 'l2'):
            raise ValueError(f"greedy: unknown similarity_metric {similarity_metric!r}; use 'cc2', "
                             "'lncc' / 'cc' / 'ncc' or 'mse' / 'l2'")
        self.squared = squared
        self.anderson = anderson
        self.anderson_steps = anderson_steps
        self.anderson_m = anderson_m
        self.anderson_freq = anderson_freq.lower()
        self.padding_mode = padding_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device or torch.device('cpu')
        self.loss_fn = BoxLNCCLoss(kernel_size=self.window_size, squared=self.squared).to(self.device)
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
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Multi-scale optimisation of the field. ``fixed_tensor`` / ``moving_tensor``:
        (1, 1, *spatial); ``theta``: (1, dim, dim + 1) affine in normalised coordinates
        (fixed -> moving); ``scales`` / ``iterations``: shrink factor and iterations per level.
        Returns (and stores as ``self.warp``) the field; ``self.loss_history`` keeps the loss.
        """
        self.theta = theta
        full_shape = fixed_tensor.shape[2:]
        moving_shape = moving_tensor.shape[2:]
        dim = self.dim
        mode = 'trilinear' if dim == 3 else 'bilinear'

        # Local helper: smooth a last-channel field [B, *spatial, dim] with 1-D Gaussians.
        # Encapsulates the channel-first permutation required by _separable_1d_filter.
        def smooth_field(f, gaussians):
            return _separable_1d_filter(torch.movedim(f, -1, 1), gaussians).movedim(1, -1)

        # Initialize displacement field u at the coarsest scale
        init_size = [max(int(s / scales[0]), 16) for s in full_shape]
        warp = torch.zeros((1, *init_size, dim), dtype=torch.float32, device=self.device)
        exp_avg = torch.zeros_like(warp)
        exp_avg_sq = torch.zeros_like(warp)

        self.loss_history = []

        for level_idx, (scale, n_iters) in enumerate(zip(scales, iterations)):
            if n_iters <= 0:
                continue

            size_down = [max(int(s / scale), 16) for s in full_shape]
            moving_size_down = [max(int(s / scale), 16) for s in moving_shape]

            # Anti-aliased multi-scale downsampling preserving native aspect ratios
            if scale > 1:
                sigmas = [0.5 * (sz / szdown) for sz, szdown in zip(full_shape, size_down)]
                gaussians = [gaussian_1d_compact(s, truncated=2.0, device=self.device) for s in sigmas]
                fi_down = _separable_1d_filter(fixed_tensor, gaussians)
                fi_down = F.interpolate(fi_down, size=size_down, mode=mode, align_corners=True)
                mi_down = _separable_1d_filter(moving_tensor, gaussians)
                mi_down = F.interpolate(mi_down, size=moving_size_down, mode=mode, align_corners=True)
            else:
                fi_down = fixed_tensor
                mi_down = moving_tensor

            # Interpolate warp and Adam moments to current scale via centralized resize_field
            if list(warp.shape[1:-1]) != size_down:
                warp      = resize_field(warp,      size_down, mode=mode)
                exp_avg   = resize_field(exp_avg,   size_down, mode=mode)
                exp_avg_sq = resize_field(exp_avg_sq, size_down, mode=mode)

            grid_shape = [1, 1, *size_down]
            id_grid = F.affine_grid(torch.eye(dim, dim + 1, device=self.device)[None], grid_shape, align_corners=True)
            affine_grid = F.affine_grid(self.theta, grid_shape, align_corners=True)
            affine_grid_cf = torch.movedim(affine_grid, -1, 1)
            half_resolution = 1.0 / (max(size_down) - 1)
            step = 0  # Reset Adam warm-up step at each scale level

            grad_gaussians = [gaussian_1d_compact(self.flow_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if self.flow_sigma > 0 else None
            reg_gaussians = [gaussian_1d_compact(self.regadam_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if (self.optimizer in ('regadam', 'reg_adam') and self.regadam_sigma > 0) else None
            warp_gaussians = [gaussian_1d_compact(self.total_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if self.total_sigma > 0 else None

            for it in range(n_iters):
                warp_param = warp.detach().requires_grad_(True)

                # Correct A∘(Id+u) composition: deform in fixed-space coords, then map through
                # the affine.  sample_field_cf(affine_grid_cf, Id+u) evaluates A at each (x + u(x)) position,
                # yielding moving-space normalized coords A(x+u(x)) for each fixed voxel.
                # This prevents shear-induced folding from affines with negative diagonal entries.
                sample_grid = sample_field_cf(affine_grid_cf, id_grid + warp_param)
                moved = F.grid_sample(mi_down, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

                if self.similarity_metric in ['mse', 'l2']:
                    loss = F.mse_loss(moved, fi_down)
                else:
                    loss = self.loss_fn(moved, fi_down)

                self.loss_history.append(float(loss.item()))
                loss.backward()

                grad = warp_param.grad.data
                if self.flow_sigma > 0 and grad_gaussians is not None:
                    grad = smooth_field(grad, grad_gaussians)

                step += 1
                exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                b1_corr = 1.0 - self.beta1 ** step
                b2_corr = 1.0 - self.beta2 ** step

                denom = (exp_avg_sq / b2_corr).sqrt().add_(self.eps)
                raw_update = (exp_avg / b1_corr) / denom

                # Apply RegAdam quotient smoothing if elected
                if self.optimizer in ('regadam', 'reg_adam'):
                    if reg_gaussians is not None:
                        u_reg = smooth_field(raw_update, reg_gaussians)
                    elif self.sobolev_alpha > 0:
                        from .core.smoothing import apply_sobolev_green_operator
                        u_reg = apply_sobolev_green_operator(raw_update.squeeze(0), fluid_sigma=self.sobolev_alpha, alpha=self.sobolev_alpha).unsqueeze(0)
                    else:
                        u_reg = raw_update
                else:
                    u_reg = raw_update

                # Bound max velocity step
                gradmax = self.eps + u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                gradmax = gradmax.reshape(-1, *([1]) * (dim + 1)).clamp(min=1.0)
                update = -self.learning_rate * half_resolution * (u_reg / gradmax)

                # Eulerian compositive pullback: u_{k+1}(x) = v_k(x) + u_k(x + v_k(x))
                # compose_grids(warp, id_grid + update) samples warp at (x + update(x)).
                pulled = compose_grids(warp, id_grid + update.to(warp.dtype))

                u_new = update + pulled
                if self.total_sigma > 0 and warp_gaussians is not None:
                    u_new = smooth_field(u_new, warp_gaussians)
                warp = u_new.detach()

                if verbose and (it % 25 == 0 or it == n_iters - 1):
                    print(f"  [greedy] Level {level_idx} (scale {scale}) Iter {it+1}/{n_iters} | Loss: {loss.item():.4f}")

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

        self.warp = warp
        return warp





def greedy_registration(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    reg_iterations: Optional[Union[List[int], Tuple[int, ...]]] = None,
    scales: Optional[Union[List[int], Tuple[int, ...]]] = None,
    learning_rate: float = 0.375,
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
    verbose: bool = False,
    outprefix: Optional[str] = None,
    seed: int = 42,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Greedy (one-directional) diffeomorphic registration of ``moving`` to ``fixed`` --
    ``syntx.greedy``.

    ::

        reg = syntx.greedy(fixed, moving)
        reg['warpedmovout'], reg['fwdtransforms']        # one composed field (affine included)

    Defaults were tuned on Mindboggle pairs 77 / 44 / 0 (learning_rate 0.25 -> 0.375,
    docs/provenance/best_parameters.json "syntx.greedy/tuned_greedy_2026_09_29").

    Parameters
    ----------
    fixed, moving : ANTsImage
        2-D or 3-D. Both are intensity-normalised (2nd-98th foreground percentile) first.
    reg_iterations : int or list of int, default None
        Iterations per level. None: [100, 100, 20] (3-D), [100, 100, 100, 50] (2-D).
    scales : int or list of int, default None
        Shrink factor per level. None: [2**(L-1), ..., 1] for L = len(reg_iterations).
    learning_rate : float, default 0.375
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
        otherwise ANTs transform file(s) -- must be linear (affine/rigid/similarity/...); a
        non-linear transform (e.g. a displacement field) raises ``NotImplementedError`` (no
        dense-initial-grid support yet, unlike ``core.affine.parse_ants_affine``'s general
        ``allow_nonlinear`` contract -- nothing in this function's optimisation loop can
        currently consume a per-voxel starting field).
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
        ``'fwdtransforms'`` : [one displacement field file] including the affine.
        ``'invtransforms'`` / ``'whichtoinvert_inv'`` : [inverse field] / [False] with
            ``return_inverse``, else empty lists.
        ``'warpedmovout'``, ``'model'`` (GreedyRegistrationModel), ``'provenance'``.
    """
    t0_total = time.time()
    dim = fixed.dimension
    torch_device = auto_detect_device(backend='pytorch', requested_device=device)

    # 1. Setup multi-resolution schedule
    if reg_iterations is None:
        # 3D last-stage aligned to syntx.syn()'s newly-updated default (docs/provenance/best_parameters.json,
        # "90pair_population_benchmark_sobolev_mps"); greedy has no benchmarked parameter data of its own,
        # so it is made an interim stand-in match to syn's schedule shape. 2D has no analogous benchmark
        # data in either function and is left unchanged.
        reg_iterations = [100, 100, 20] if dim == 3 else [100, 100, 100, 50]
    elif isinstance(reg_iterations, int):
        reg_iterations = [reg_iterations]

    if scales is None:
        num_levels = len(reg_iterations)
        scales = [2**i for i in range(num_levels)][::-1] if num_levels > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
    elif isinstance(scales, int):
        scales = [scales]

    # 2. Intensity Normalization (Foreground 2nd-98th percentile policy)
    from .benchmark.evaluate import normalize_intensity
    fi_norm = normalize_intensity(fixed)
    mi_norm = normalize_intensity(moving)

    # 3. Initial Affine Alignment
    t0_aff = time.time()
    aff_tx = None
    if initial_transform is None:
        if verbose:
            print("[greedy] Running syntx.robust_affine(mode='auto')...")
        reg_aff = robust_affine(fi_norm, mi_norm, mode=affine_mode, verbose=False)
        aff_tx = reg_aff['fwdtransforms'][0]
    elif initial_transform is False or (isinstance(initial_transform, str) and initial_transform.lower() == 'identity'):
        aff_tx = None
    else:
        aff_tx = initial_transform if isinstance(initial_transform, list) else [initial_transform]
    t1_aff = time.time() - t0_aff

    # 4. Extract Physical and Normalized Affine Matrices
    if aff_tx is not None:
        tx_list = aff_tx if isinstance(aff_tx, list) else [aff_tx]
        # allow_nonlinear=True so a non-linear initial_transform (e.g. a displacement field)
        # surfaces here as a clean (None, None) instead of parse_ants_affine's own ValueError
        # -- this function has NO dense-initial-grid support despite what its docstring says
        # ("anything else becomes an initial grid"): GreedyRegistrationModel.fit() only
        # accepts an affine theta, nothing seeds a per-voxel starting field. Since tx_list is
        # guaranteed non-empty here (aff_tx is not None), a (None, None) result can only mean
        # a non-linear item was passed -- raise clearly rather than silently discarding it and
        # starting from identity, which would be a much worse, silent version of this gap.
        M_phys, t_phys = parse_ants_affine(tx_list, dim, allow_nonlinear=True)
        if M_phys is None:
            raise NotImplementedError(
                "greedy_registration: initial_transform contains a non-linear transform (e.g. "
                "a displacement field), which this function does not yet support as a dense "
                "initial grid -- GreedyRegistrationModel.fit() only accepts an affine starting "
                "point. Pass a linear transform (affine/rigid/similarity/...), None (runs "
                "syntx.robust_affine), False, or 'identity' instead."
            )
    else:
        M_phys = np.eye(dim, dtype=np.float64)
        t_phys = np.zeros(dim, dtype=np.float64)

    theta, P_F, P_M = _build_torch_affine_matrix(fi_norm, mi_norm, M_phys, t_phys, torch_device)

    # 5. Prepare Image Tensors (ZYX layout)
    fi_np = fi_norm.numpy().T
    mi_np = mi_norm.numpy().T

    fi_t = torch.from_numpy(fi_np).float().unsqueeze(0).unsqueeze(0).to(torch_device)
    mi_t = torch.from_numpy(mi_np).float().unsqueeze(0).unsqueeze(0).to(torch_device)

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

    # --- Parameter relevance validation ---
    reg_mode = str(kwargs.pop('regularizer', 'gaussian')).lower()
    if reg_mode != 'gaussian':
        raise ValueError(f"syntx.greedy smooths with Gaussians only (flow_sigma / total_sigma / "
                         f"regadam_sigma); regularizer={reg_mode!r} is not supported")
    if kwargs.pop('dsti_alpha', None) is not None:
        raise ValueError("dsti_alpha is only valid with spectral regularizers; syntx.greedy is "
                         "Gaussian-only (smoothing strength: flow_sigma)")
    squared = bool(kwargs.pop('squared', True))
    interpolator = kwargs.pop('interpolator', 'linear')
    if kwargs:
        raise TypeError(f"syntx.greedy() got unexpected keyword(s) {sorted(kwargs)}")

    # 6. Instantiate & Run Greedy Model
    model = GreedyRegistrationModel(
        dim=dim,
        learning_rate=learning_rate,
        flow_sigma=flow_sigma,
        total_sigma=total_sigma,
        optimizer=optimizer,
        regadam_sigma=regadam_sigma,
        sobolev_alpha=sobolev_alpha,
        lncc_radius=lncc_radius,
        similarity_metric=similarity_metric,
        squared=squared,
        anderson=anderson,
        anderson_steps=anderson_steps,
        anderson_m=anderson_m,
        anderson_freq=anderson_freq,
        padding_mode=padding_mode,
        device=torch_device,
    )

    t0_opt = time.time()
    warp = model.fit(
        fixed_tensor=fi_t,
        moving_tensor=mi_t,
        theta=theta,
        scales=scales,
        iterations=reg_iterations,
        verbose=verbose,
    )
    t1_opt = time.time() - t0_opt

    # 7. Single Composed ANTs Displacement Field Export
    # warp encodes displacement in fixed-space normalized coords u(x).
    # compose_grids(A, Id + u) computes A(x + u(x)) for each fixed voxel —
    # the correct A∘(Id+u) composition using the centralized utility.
    full_shape = fi_t.shape[2:]; dim = fi_t.ndim - 2
    full_grid_shape = [1, 1, *full_shape]
    affine_grid = F.affine_grid(theta, full_grid_shape, align_corners=True)
    full_id_grid = F.affine_grid(
        torch.eye(dim, dim + 1, device=warp.device)[None], full_grid_shape, align_corners=True
    )
    total_target_grid = compose_grids(affine_grid, full_id_grid + warp)

    disp_img = _convert_composed_grid_to_ants_displacement(total_target_grid, fixed, P_F, P_M)


    # File output paths
    if outprefix is not None:
        os.makedirs(os.path.dirname(outprefix) or '.', exist_ok=True)
        fwd_file = f"{outprefix}1Warp.nii.gz"
    else:
        fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd_warp.nii.gz', delete=False).name

    ants.image_write(disp_img, fwd_file)
    fwd_transforms = [fwd_file]

    # 8. Warped Moving Image Output (Single Interpolation directly on native moving image)
    warpedmovout = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=fwd_transforms,
        interpolator=interpolator
    )

    # 9. Optional Physical Inverse Transform Export via Anderson Acceleration
    inv_transforms = []
    t1_inv = 0.0
    if return_inverse:
        t0_inv = time.time()
        if verbose:
            print("[greedy] Computing physical inverse displacement field with Anderson acceleration...")
        disp_t = torch.from_numpy(disp_img.numpy().T).float().unsqueeze(0).to(torch_device)
        inv_steps = max(anderson_steps, 15)
        inv_disp_t = update_inverse_field_nd_anderson(
            W_disp=disp_t,
            W_inv_disp=None,
            steps=inv_steps,
            m=anderson_m,
            spacing=fixed.spacing,
            origin=fixed.origin,
            direction=fixed.direction,
        )
        inv_np = inv_disp_t.squeeze(0).cpu().numpy().T
        inv_disp_img = ants.from_numpy(
            inv_np.astype(np.float32),
            origin=fixed.origin,
            spacing=fixed.spacing,
            direction=fixed.direction,
            has_components=True
        )

        if outprefix is not None:
            inv_file = f"{outprefix}1InverseWarp.nii.gz"
        else:
            inv_file = tempfile.NamedTemporaryFile(suffix='_inv_warp.nii.gz', delete=False).name

        ants.image_write(inv_disp_img, inv_file)
        inv_transforms = [inv_file]
        t1_inv = time.time() - t0_inv

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
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': [False] if len(inv_transforms) > 0 else [],
        'model': model,
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
