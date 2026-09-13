"""
greedy.py — Fast Unidirectional Compositive Eulerian Registration Core
=====================================================================

This module implements a lightweight, ultra-fast unidirectional Eulerian compositive
registration model (GreedyRegistration) designed for high-throughput alignment pipelines,
deep-learning preprocessing, and feature extraction where full bidirectional midpoint
symmetry and topological invertibility are not required.

Key Design & Theoretical Highlights:
-------------------------------------
1. Single Interpolation Invariant:
   Avoids any intermediate file-based pre-warping or multi-step interpolation. The initial
   linear alignment (from syntx.robust_affine or user input) is mapped directly into
   normalized coordinate space (theta) and composed with the Eulerian displacement field
   during sampling: x_sample = G_affine + u.
2. Single-Pass Eulerian Composition:
   Each iteration requires only 2 spatial resamplings (1 for the warped moving image and
   1 for the compositive displacement update u_{k+1} = v_k + u_k(x + v_k)), yielding an
   order-of-magnitude reduction in grid evaluations compared to bidirectional SyN (which
   requires 16-24 resamplings per iteration for Anderson-accelerated in-loop inversion).
3. Variance-Floored Similarity Metric:
   Uses analytical or autograd Local Normalized Cross-Correlation (LNCC) with a mandatory
   variance floor (Var_safe >= 1e-6) and Cauchy-Schwarz clamping [-1.0, 1.0] to prevent
   gradient singularities in flat background regions.
4. Robust Affine Multi-Start Integration:
   Automatically initializes with syntx.robust_affine(mode='auto') when no initial transform
   is provided, guaranteeing robust, basin-entrapment-free global initialization.
5. Direct ANTs ITK Physical Field Export:
   Exports a single, fully composed ANTs displacement field (W_phys = y_phys - x_phys)
   in ITK physical space, allowing downstream tools to apply the full transformation
   in a single ants.apply_transforms call.
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
from .core.grid import compose_grids, resize_field
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
    Greedy Eulerian Compositive Registration Model.
    
    Performs multi-scale gradient optimization of a forward displacement field
    u using compositive Eulerian updates:
        (Id + u_{k+1}) = (Id + u_k) o (Id + v_k)
    where v_k is the smoothed Adam descent velocity.
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
        similarity_metric: str = 'lncc',
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
        Executes multi-scale greedy compositive optimization.
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
            half_resolution = 1.0 / (max(size_down) - 1)
            step = 0  # Reset Adam warm-up step at each scale level

            grad_gaussians = [gaussian_1d_compact(self.flow_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if self.flow_sigma > 0 else None
            reg_gaussians = [gaussian_1d_compact(self.regadam_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if (self.optimizer in ('regadam', 'reg_adam') and self.regadam_sigma > 0) else None
            warp_gaussians = [gaussian_1d_compact(self.total_sigma, truncated=2.0, device=self.device) for _ in range(dim)] if self.total_sigma > 0 else None

            for it in range(n_iters):
                warp_param = warp.detach().requires_grad_(True)

                # Correct A∘(Id+u) composition: deform in fixed-space coords, then map through
                # the affine.  compose_grids(A, Id+u) evaluates A at each (x + u(x)) position,
                # yielding moving-space normalized coords A(x+u(x)) for each fixed voxel.
                # This prevents shear-induced folding from affines with negative diagonal entries.
                sample_grid = compose_grids(affine_grid, id_grid + warp_param)
                moved = F.grid_sample(mi_down, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

                if self.similarity_metric in ['lncc', 'cc', 'ncc']:
                    loss = self.loss_fn(moved, fi_down)
                elif self.similarity_metric in ['mse', 'l2']:
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
    learning_rate: float = 0.50,
    flow_sigma: float = 1.8,
    total_sigma: float = 0.28,
    optimizer: str = 'adam',
    regadam_sigma: float = 0.8,
    similarity_metric: str = 'lncc',
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
    High-level, ultra-fast unidirectional Eulerian compositive registration function.
    
    Designed for fast forward alignment, deep-learning feature extraction, and high-throughput
    registration pipelines where sub-25 second runtimes are required without sacrificing
    alignment accuracy.

    Parameters
    ----------
    fixed : ANTsImage
        Fixed reference target image.
    moving : ANTsImage
        Moving source image to be registered to fixed space.
    reg_iterations : list of int, optional
        Number of iterations per pyramid level. Default [100, 100, 80] for 3D, [100, 100, 100, 50] for 2D.
    scales : list of int, optional
        Downsampling factors per pyramid level. Default [4, 2, 1] for 3D, [8, 4, 2, 1] for 2D.
    learning_rate : float, optional
        Descent step size for velocity field. Default 0.50.
    flow_sigma : float, optional
        Gaussian standard deviation in voxels for fluid smoothing of the gradient field. Default 1.8.
    total_sigma : float, optional
        Gaussian standard deviation in voxels for elastic smoothing of the compositive warp field. Default 0.28.
    optimizer : str, optional
        Optimizer type: 'adam' (standard Adam) or 'regadam' / 'reg_adam' (RegAdam with quotient smoothing).
        Default 'adam'.
    regadam_sigma : float, optional
        Gaussian standard deviation in voxels for smoothing the Adam step quotient in RegAdam. Default 0.8.
    similarity_metric : str, optional
        Image similarity metric ('lncc', 'mse'). Default 'lncc'.
    lncc_radius : int, optional
        Window radius for LNCC (window size = 2 * radius + 1). Default 2.
    anderson : bool, optional
        If True, applies Anderson-accelerated fixed-point projection to suppress
        grid folds and regularize deformations. Default False.
    anderson_steps : int, optional
        Number of Anderson fixed-point iterations per projection step. Default 5.
    anderson_m : int, optional
        Anderson history memory window size. Default 5.
    anderson_freq : str, optional
        Frequency of Anderson projection ('per_scale' at scale transitions or 'posthoc' at optimization end).
        Default 'per_scale'.
    return_inverse : bool, optional
        If True, computes and exports the physical inverse displacement field in 'invtransforms'.
        Default False.
    initial_transform : str or list or ANTsTransform or bool or None, optional
        Initial linear transform.
        - If None: Automatically executes syntx.robust_affine(mode='auto').
        - If False or 'identity': Uses identity transformation (no affine initialization).
        - If str/list: Parses the specified ANTs affine transform.
    affine_mode : str, optional
        Mode for robust_affine if initial_transform is None ('auto', 'mmi', etc.). Default 'auto'.
    device : str, optional
        Device to run optimization on ('mps', 'cuda', 'cpu'). Auto-detected if None.
    verbose : bool, optional
        If True, prints progress statements and iteration details. Default False.
    outprefix : str, optional
        File prefix to save output transforms (e.g. '/path/to/prefix').
    seed : int, optional
        Random seed for reproducibility. Default 42.
    **kwargs : dict
        Additional parameters forwarded to robust_affine or optimization routines.

    Returns
    -------
    dict
        Registration result dictionary matching ants.registration interface:
        - 'warpedmovout': ANTsImage (moving image warped directly to fixed space)
        - 'fwdtransforms': list of str (path to single composed ANTs displacement field)
        - 'invtransforms': list of str (empty for unidirectional greedy)
        - 'model': GreedyRegistrationModel object
        - 'provenance': dict of registration metadata and timings
    """
    t0_total = time.time()
    dim = fixed.dimension
    torch_device = auto_detect_device(backend='pytorch', requested_device=device)

    # 1. Setup multi-resolution schedule
    if reg_iterations is None:
        reg_iterations = [100, 100, 80] if dim == 3 else [100, 100, 100, 50]
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
        M_phys, t_phys = parse_ants_affine(tx_list, dim)
        if M_phys is None:
            M_phys = np.eye(dim, dtype=np.float64)
            t_phys = np.zeros(dim, dtype=np.float64)
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
        squared=kwargs.pop('squared', True),
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
        interpolator=kwargs.get('interpolator', 'linear')
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
    }


# Aliases
greedy = greedy_registration
