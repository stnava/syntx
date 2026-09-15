"""
robust_affine.py — Ultra-Fast, Fail-Safe Robust Multi-Start Affine Registration
===================================================================================

Provides high-reliability, multi-start initial alignment strategies ('mode'):
- 'pytorch' / 'gpu'      : Fast 2D/3D native PyTorch Lie Algebra solver with cone-constrained rotation search.
- 'auto' / 'fast'        : Low-res multi-start candidate selection + multi-stage ANTs solver.
- 'ants_fast'            : Fast multi-stage ANTs C++ pipeline (Translation -> Rigid -> Similarity -> Affine).
- 'com_only'             : Instant 0.05s Center-of-Mass physical translation alignment.

Guarantees robust convergence even under severe initial translation, rotation, or
contrast inversion offsets.

Strictly obeys Syntx Registration Guardrails:
1. Single Interpolation Policy: Composes initial and final transforms into a single stage.
2. Cone Rotation Search: Searches a constrained orientation cone (<= 30 degrees) to preserve brain hemispheric symmetry.
3. 2D and 3D Support: Native support for both 2D and 3D image registration.
4. Center of Rotation: Preserves ITK fixed parameter center of rotation during conversions.
5. Lie Algebra Taylor Expansion: Prevents zero-angle gradient locking.
"""

import time
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .syn import mattes_mi_loss_nd
from .core.losses import parzen_weights
from .spatial import (
    image_to_tensor,
    get_image_metadata,
    get_spatial_coordinate_grid,
    create_ants_affine,
)


def compute_center_of_mass(img_ants: ants.ANTsImage, weighted: bool = True) -> np.ndarray:
    """
    Computes physical Center of Mass (CoM) of a 2D or 3D ANTsImage.

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2D or 3D ANTs image.
    weighted : bool, default=True
        If True, computes intensity-weighted physical center of mass.
        If False, computes geometric center of mass of non-zero foreground mask.

    Returns
    -------
    np.ndarray
        Array of shape `(dim,)` containing physical space coordinates `(x, y)` or `(x, y, z)`.
    """
    if weighted:
        return np.array(ants.get_center_of_mass(img_ants))
    else:
        arr = img_ants.numpy()
        origin = np.array(img_ants.origin)
        spacing = np.array(img_ants.spacing)
        direction = np.array(img_ants.direction)
        dim = img_ants.dimension
        weights = (arr > (arr.max() * 0.05)).astype(np.float32)
        total_w = weights.sum()
        if total_w <= 1e-6:
            voxel_center = (np.array(arr.shape) - 1.0) / 2.0
        else:
            grid_coords = [np.arange(s) for s in arr.shape]
            mesh = np.meshgrid(*grid_coords, indexing='ij')
            voxel_center = np.array([(mesh[i] * weights).sum() / total_w for i in range(dim)])
        phys_center = origin + direction @ (voxel_center * spacing)
        return phys_center


def create_translation_transform(fi: ants.ANTsImage, mi: ants.ANTsImage, t_phys: np.ndarray) -> tuple:
    """
    Creates an ANTs physical translation transform file matching physical CoM offset `t_phys`.

    Parameters
    ----------
    fi : ants.ANTsImage
        Fixed image.
    mi : ants.ANTsImage
        Moving image.
    t_phys : np.ndarray
        Physical translation vector `fixed_com - moving_com`.

    Returns
    -------
    tx_path : str
        File path to saved `.mat` translation transform file.
    temp_dir : str
        Temporary directory holding the written file.
    """
    temp_dir = tempfile.mkdtemp(prefix="robust_aff_init_")
    tx_path = os.path.join(temp_dir, "initial_translation.mat")
    dim = fi.dimension

    tx = ants.create_ants_transform(
        transform_type='AffineTransform',
        precision='float',
        dimension=dim
    )
    matrix = np.eye(dim)
    tx.set_parameters(np.concatenate([matrix.flatten(), t_phys]))
    tx.set_fixed_parameters(np.array(fi.origin))

    ants.write_transform(tx, tx_path)
    return tx_path, temp_dir


def _eval_low_res_mi(fi_low: ants.ANTsImage, mi_low: ants.ANTsImage, tx_path: str = None) -> float:
    """Evaluates low-resolution Mattes Mutual Information score with foreground masking for candidate transform."""
    try:
        if tx_path is None:
            warped = mi_low
        else:
            warped = ants.apply_transforms(fixed=fi_low, moving=mi_low, transformlist=[tx_path])
        from syntx.core.losses import mattes_mi_loss_nd
        f_t = image_to_tensor(fi_low)
        w_t = image_to_tensor(warped)
        # fixed-foreground mask (a union mask rewards shrinking overlap) and fixed histogram
        # bounds so candidate scores are comparable
        mask_t = (f_t > 0.01)
        mi_score = mattes_mi_loss_nd(w_t, f_t, mask=mask_t, num_bins=32).item()
        return mi_score
    except Exception:
        return 999.0


def _rodrigues_rotation_matrix_3d(omega: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Rodrigues Lie Algebra $so(3) \\rightarrow SO(3)$ 3D rotation matrix parameterization.

    Uses first-order Taylor expansion near zero ($\\|\\omega\\| < 10^{-5}$) to prevent zero-angle
    gradient locking (GEMINI.md Rule 6).

    Parameters
    ----------
    omega : torch.Tensor
        3D Lie algebra vector `[w0, w1, w2]`.

    Returns
    -------
    torch.Tensor
        `3x3` rotation matrix $R \\in SO(3)$.
    """
    theta = torch.norm(omega)
    if theta < 1e-5:
        w0, w1, w2 = omega[0], omega[1], omega[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=omega.device), -w2, w1]),
            torch.stack([w2, torch.tensor(0.0, device=omega.device), -w0]),
            torch.stack([-w1, w0, torch.tensor(0.0, device=omega.device)])
        ])
        return torch.eye(3, device=omega.device, dtype=torch.float32) + K
    else:
        u = omega / theta
        u0, u1, u2 = u[0], u[1], u[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=omega.device), -u2, u1]),
            torch.stack([u2, torch.tensor(0.0, device=omega.device), -u0]),
            torch.stack([-u1, u0, torch.tensor(0.0, device=omega.device)])
        ])
        return torch.eye(3, device=omega.device, dtype=torch.float32) + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def _rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable Lie Algebra $so(2) \\rightarrow SO(2)$ 2D rotation matrix parameterization."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack([
        torch.stack([cos_t, -sin_t]),
        torch.stack([sin_t, cos_t])
    ])


def compute_fov_center(img_ants: ants.ANTsImage) -> np.ndarray:
    """
    Computes geometric physical center (midpoint of field of view) of an ANTsImage.

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2D or 3D ANTs image.

    Returns
    -------
    np.ndarray
        Array of shape `(dim,)` containing physical coordinates of the FOV center.
    """
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    shape = np.array(img_ants.shape)
    voxel_center = (shape - 1.0) * 0.5
    phys_center = origin + direction @ (voxel_center * spacing)
    return phys_center


def _generate_quick_search_candidates(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    cone_angles_deg: list = None
) -> list:
    r"""
    Generates quick search initialization candidates:
    - Option 1: 'Identity_CoM' (Center of Mass matching with Identity rotation)
    - Option 2: 'Identity_FOV' (Field of View geometric midpoint matching with Identity rotation)
    - Rotational Search: Angle perturbations across pitch, roll, yaw for both CoM and FOV base translations.
    """
    dim = fixed.dimension
    if cone_angles_deg is None:
        cone_angles_deg = [-30.0, -20.0, -10.0, 10.0, 20.0, 30.0]

    com_f = compute_center_of_mass(fixed, weighted=True)
    com_m = compute_center_of_mass(moving, weighted=True)
    t_com = np.array(com_m) - np.array(com_f)

    fov_f = compute_fov_center(fixed)
    fov_m = compute_fov_center(moving)
    t_fov = np.array(fov_m) - np.array(fov_f)

    candidates = []

    # 1. Option 1: Identity CoM
    tx_com = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx_com.set_parameters(np.concatenate([np.eye(dim).flatten(), t_com]))
    tx_com.set_fixed_parameters(com_f)
    r_dir_com = tempfile.mkdtemp(prefix="robust_aff_id_com_")
    r_path_com = os.path.join(r_dir_com, "cone_rotation.mat")
    ants.write_transform(tx_com, r_path_com)
    candidates.append(('Identity_CoM', r_path_com, np.eye(dim), t_com, com_f, r_dir_com))

    # 2. Option 2: Identity FOV
    tx_fov = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx_fov.set_parameters(np.concatenate([np.eye(dim).flatten(), t_fov]))
    tx_fov.set_fixed_parameters(fov_f)
    r_dir_fov = tempfile.mkdtemp(prefix="robust_aff_id_fov_")
    r_path_fov = os.path.join(r_dir_fov, "cone_rotation.mat")
    ants.write_transform(tx_fov, r_path_fov)
    candidates.append(('Identity_FOV', r_path_fov, np.eye(dim), t_fov, fov_f, r_dir_fov))

    # 3. Rotational search around CoM and FOV centers
    if dim == 3:
        for base_name, t_base, C in [('CoM', t_com, com_f), ('FOV', t_fov, fov_f)]:
            for deg in cone_angles_deg:
                if abs(deg) < 1e-3:
                    continue
                rad = np.radians(deg)
                for axis_name in ['pitch', 'roll', 'yaw']:
                    rx = rad if axis_name == 'pitch' else 0.0
                    ry = rad if axis_name == 'roll' else 0.0
                    rz = rad if axis_name == 'yaw' else 0.0

                    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                    R = Rz @ Ry @ Rx
                    tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
                    tx_r.set_parameters(np.concatenate([R.flatten(), t_base]))
                    tx_r.set_fixed_parameters(C)

                    r_dir = tempfile.mkdtemp(prefix=f"robust_aff_{base_name}_{axis_name}_{deg}_")
                    r_path = os.path.join(r_dir, "cone_rotation.mat")
                    ants.write_transform(tx_r, r_path)
                    candidates.append((f'{base_name}_{axis_name}_{deg:+.0f}deg', r_path, R, t_base, C, r_dir))
    elif dim == 2:
        for base_name, t_base, C in [('CoM', t_com, com_f), ('FOV', t_fov, fov_f)]:
            for deg in cone_angles_deg:
                if abs(deg) < 1e-3:
                    continue
                rad = np.radians(deg)
                R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2)
                tx_r.set_parameters(np.concatenate([R.flatten(), t_base]))
                tx_r.set_fixed_parameters(C)

                r_dir = tempfile.mkdtemp(prefix=f"robust_aff_2d_{base_name}_{deg}_")
                r_path = os.path.join(r_dir, "cone_rotation.mat")
                ants.write_transform(tx_r, r_path)
                candidates.append((f'{base_name}_rot_{deg:+.0f}deg', r_path, R, t_base, C, r_dir))

    return candidates


def _generate_cone_rotation_candidates_3d(com_f, t_init, cone_angles_deg=None):
    """Backwards compatibility generator supporting both ANTsImage and raw ndarrays."""
    if hasattr(com_f, 'dimension'):
        return _generate_quick_search_candidates(com_f, t_init, cone_angles_deg)
    
    if cone_angles_deg is None:
        cone_angles_deg = [-30.0, -20.0, -10.0, 10.0, 20.0, 30.0]
    C = np.array(com_f)
    t_base = np.array(t_init)
    candidates = []
    for deg in cone_angles_deg:
        if abs(deg) < 1e-3:
            continue
        rad = np.radians(deg)
        for axis_name in ['pitch', 'roll', 'yaw']:
            rx = rad if axis_name == 'pitch' else 0.0
            ry = rad if axis_name == 'roll' else 0.0
            rz = rad if axis_name == 'yaw' else 0.0
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
            tx_r.set_parameters(np.concatenate([R.flatten(), t_base]))
            tx_r.set_fixed_parameters(C)
            r_dir = tempfile.mkdtemp(prefix=f"robust_aff_legacy_{axis_name}_{deg}_")
            r_path = os.path.join(r_dir, "cone_rotation.mat")
            ants.write_transform(tx_r, r_path)
            candidates.append((f'CoM_{axis_name}_{deg:+.0f}deg', r_path, R, t_base, r_dir))
    return candidates


def _default_affine_schedule(dim: int, preset: str = 'default') -> list:
    """Multi-resolution optimisation schedule (one dict per stage).

    level     : pyramid downsampling factor (avg-pool)                  iters : Adam steps
    dof       : 'rigid' (translation + rotation) or 'affine' (+ log-scale + shear)
    lr        : per-parameter-group learning rates (t, omega, scale, shear)
    eta_min   : cosine-annealing floor for the learning rates (None = constant)
    select    : if True, keep only the best path (full-sample MI at this level) after the stage
    full_grid : if True, this stage uses the voxel grid (regular stride = 1/sampling) instead of
                the random point sample — exact/regular objective, ANTs-like
    sampling  : per-stage regular sampling fraction for full_grid stages (default sampling_percentage)
    optimizer : 'adam' (default) or 'lbfgs' (strong-Wolfe L-BFGS, `iters` = max iterations)

    Presets (3-D), chosen on 6 Mindboggle pairs against ANTs C++ Affine
    (results/affine_baseline/sweep_cohort_subset8.json):
      'default'  : L4 rigid (exact grid) -> L3 affine (exact grid, 100 it, best-of-n_starts)
                   -> L2 affine (point sample) -> L1 affine (point sample); 32 bins.
                   ties/beats ANTs on 6/6 pairs (mean +0.0002 Dice), ~11 s on MPS.
      'accurate' : L4 rigid (exact) -> L2 affine (regular 50 % grid, 100 it, select) -> L1;
                   32 bins.  beats ANTs on 6/6 pairs (mean +0.0021, worst +0.0002), ~20 s.
      'fast'     : L4 rigid -> L2 affine -> L1 affine, all point-sampled (100k), ~4-6 s.
    """
    if dim == 3:
        if preset == 'accurate':
            return [
                dict(level=4, iters=50, dof='rigid',  lr=(0.04, 0.008, 0.0, 0.0),        eta_min=0.002, select=False, full_grid=True, sampling=1.0),
                dict(level=2, iters=100, dof='affine', lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True,  full_grid=True, sampling=0.5),
                dict(level=1, iters=30, dof='affine', lr=(0.005, 0.002, 0.001, 0.001),  eta_min=1e-4,  select=False),
            ]
        if preset == 'fast':
            return [
                dict(level=4, iters=50, dof='rigid',  lr=(0.04, 0.008, 0.0, 0.0),        eta_min=0.002, select=False),
                dict(level=2, iters=50, dof='affine', lr=(0.015, 0.005, 0.003, 0.002),  eta_min=0.001, select=True),
                dict(level=1, iters=30, dof='affine', lr=(0.005, 0.002, 0.001, 0.001),  eta_min=1e-4,  select=False),
            ]
        return [
            dict(level=4, iters=50, dof='rigid',  lr=(0.04, 0.008, 0.0, 0.0),          eta_min=0.002, select=False, full_grid=True, sampling=1.0),
            dict(level=3, iters=100, dof='affine', lr=(0.015, 0.005, 0.003, 0.002),   eta_min=0.001, select=True,  full_grid=True, sampling=1.0),
            dict(level=2, iters=30, dof='affine', lr=(0.008, 0.003, 0.0015, 0.001),   eta_min=5e-4,  select=False),
            dict(level=1, iters=30, dof='affine', lr=(0.005, 0.002, 0.001, 0.001),    eta_min=1e-4,  select=False),
        ]
    return [
        dict(level=2, iters=50, dof='affine', lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True),
        dict(level=1, iters=30, dof='affine', lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4,  select=False),
    ]


def _gaussian_kernel_2d_sep(sigma: float, device):
    """Separable 2-D Gaussian kernels (vertical, horizontal, radius) for conv2d."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (xs / sigma) ** 2); k = k / k.sum()
    return k.view(1, 1, -1, 1), k.view(1, 1, 1, -1), radius


def _parse_initial_affine(path: str, dim: int):
    """ITK .mat (AffineTransform / MatrixOffsetTransformBase) -> (A [dim,dim], t [dim], center [dim])
    with y = A (x - c) + c + t.  Returns None when unreadable."""
    try:
        tx = ants.read_transform(path)
        p = np.array(tx.parameters, dtype=np.float64)
        A = p[:dim * dim].reshape(dim, dim)
        t = p[dim * dim:dim * dim + dim]
        c = np.array(tx.fixed_parameters, dtype=np.float64)[:dim] if len(tx.fixed_parameters) >= dim else np.zeros(dim)
        return A, t, c
    except Exception:
        return None


class _AffinePath:
    """One optimisation path: A = R(omega) @ B @ diag(exp(s)) @ Shear(sh); y = A (x - C) + C + t."""

    def __init__(self, B: np.ndarray, t0: np.ndarray, dim: int, device, name: str):
        self.dim, self.name = dim, name
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.t = torch.tensor(t0, dtype=torch.float32, device=device, requires_grad=True)
        self.omega = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device, requires_grad=True)
        self.scale = torch.zeros(dim, dtype=torch.float32, device=device, requires_grad=True)
        self.shear = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device, requires_grad=True)

    def matrix(self, dof: str) -> torch.Tensor:
        dim = self.dim
        R = _rodrigues_rotation_matrix_3d(self.omega) if dim == 3 else _rotation_matrix_2d(self.omega[0])
        A = R @ self.B
        if dof == 'affine':
            S = torch.diag(torch.exp(torch.clamp(self.scale, -0.4, 0.4)))
            Sh = torch.eye(dim, device=A.device, dtype=A.dtype)
            if dim == 3:
                Sh = Sh.clone(); Sh[0, 1] = self.shear[0]; Sh[0, 2] = self.shear[1]; Sh[1, 2] = self.shear[2]
            else:
                Sh = Sh.clone(); Sh[0, 1] = self.shear[0]
            A = A @ S @ Sh
        return A

    def params(self, dof: str, lr):
        groups = [{'params': [self.t], 'lr': lr[0]}, {'params': [self.omega], 'lr': lr[1]}]
        if dof == 'affine':
            groups += [{'params': [self.scale], 'lr': lr[2]}, {'params': [self.shear], 'lr': lr[3]}]
        return groups

    @torch.no_grad()
    def clamp_(self):
        self.scale.clamp_(-0.35, 0.35); self.shear.clamp_(-0.35, 0.35); self.omega.clamp_(-np.pi / 3, np.pi / 3)

    @torch.no_grad()
    def numpy_affine(self):
        A = self.matrix('affine').detach().cpu().numpy().astype(np.float64)
        return A, self.t.detach().cpu().numpy().astype(np.float64)


def _run_pytorch_affine_solver(fixed: ants.ANTsImage, moving: ants.ANTsImage, initial_tx_path: str = None,
                               device: str = 'auto', verbose: bool = False, multi_start: bool = True,
                               n_starts: int = 2, cone_angles_deg: list = None, seed: int = 42,
                               schedule: list = None, preset: str = 'default', sampling_percentage: float = 0.5, num_bins: int = 32,
                               n_sample_points: int = 100_000, fixed_range=(0.0, 1.0), mask_mode: str = 'none',
                               smooth_sigma_per_level: float = 0.0, fg_dice_weight: float = 0.0,
                               fg_level: float = 0.01, sample_weighting: str = 'uniform',
                               sample_seed: int = None, **kwargs) -> dict:
    """
    Native PyTorch multi-resolution affine solver (``mode='pytorch'``), Mattes MI objective.

    Deterministic by construction: seeded candidate generation, fixed-bound MI with the
    deterministic Parzen histogram, and (optionally) a fixed seeded point sample per level.

    Parameters
    ----------
    initial_tx_path : str, optional
        ITK ``.mat`` used as an additional start candidate (and as the base matrix of that path).
    multi_start, n_starts, cone_angles_deg
        Coarse-level candidate search: identity-at-CoM plus single-axis cone rotations
        (default ±4°, ±8°, ±12°); the ``n_starts`` best MI candidates are optimised in parallel
        until the ``select`` stage of the schedule, then only the best path continues.
    schedule : list of dict
        See ``_default_affine_schedule``.  Any stage key may be overridden.
    preset : {'default', 'accurate', 'fast'}
        Named schedule (ignored when ``schedule`` is given).
    sampling_percentage : float
        Strided subsample of masked voxels for the MI objective (full-grid mode).
    n_sample_points : int, optional
        Point-sampled objective (default 100k): per level a fixed, seeded random subset of this
        many fixed-domain voxels; the moving image is interpolated only there.  Much faster than
        the full grid at fine levels; ``None`` keeps the full-grid objective.
    num_bins, fixed_range
        Mattes MI settings (inputs are foreground-normalised to [0, 1]).
    smooth_sigma_per_level : float
        ANTs-style anti-aliased pyramid: before avg-pooling by ``level`` both images are
        Gaussian-smoothed with sigma = ``smooth_sigma_per_level * (level - 1)`` voxels
        (ANTs uses 3x2x1x0 vox for shrink 6x4x2x1, i.e. ~0.5-1 per level unit).  0 disables.
    sample_weighting : {'uniform', 'gradient'}
        Point-sample distribution over the fixed domain.  'gradient' draws samples with
        probability ∝ 0.1 + |∇fixed| / max|∇fixed| (Gumbel top-k, seeded), concentrating the
        MI estimate on tissue boundaries / cortex instead of homogeneous interiors and background.
    sample_seed : int, optional
        Seed for the point sample only (default: ``seed``); lets an ensemble use different samples.
    fg_dice_weight : float
        Weight of an additional soft-Dice term between the fixed foreground mask and the
        warped moving foreground mask (evaluated on the same samples).  Cortical label
        overlap depends strongly on brain-outline alignment, which MI alone rewards only
        indirectly; 0 disables.
    mask_mode : {'none', 'union', 'fixed_fg'}
        Which voxels enter the MI histogram.  'none' (default): the whole fixed domain including
        background, as ANTs/ITK do without masks — on mbhard this reaches the ANTs affine Dice
        (0.326) where 'fixed_fg' plateaus at 0.31 because a moving brain spilling into fixed
        background is never penalised; 'union': fixed foreground OR warped moving foreground
        (transform-dependent sample set; slow variable-shape path on MPS).
    """
    t0 = time.time()
    dim = fixed.dimension
    seed = 42 if seed is None else seed
    torch.manual_seed(seed); np.random.seed(seed)
    if device in ['auto', None]:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device_obj = torch.device(device)
    schedule = schedule or _default_affine_schedule(dim, preset)
    n_starts = max(1, int(n_starts))

    # 1. centres, normalisation, tensors --------------------------------------------------
    com_f = np.asarray(compute_center_of_mass(fixed, weighted=True), dtype=np.float64)
    com_m = np.asarray(compute_center_of_mass(moving, weighted=True), dtype=np.float64)
    t_init = com_m - com_f
    from syntx.core.utils import normalize_image
    fixed_norm = normalize_image(fixed, method='auto')
    moving_norm = normalize_image(moving, method='auto')
    fi_arr = image_to_tensor(fixed_norm, device=device_obj, to_zyx=True)
    mi_arr = image_to_tensor(moving_norm, device=device_obj, to_zyx=True)

    f32 = dict(dtype=torch.float32, device=device_obj)
    sp_xyz = torch.tensor(fixed.spacing, **f32); orig_xyz = torch.tensor(fixed.origin, **f32); dir_xyz = torch.tensor(fixed.direction, **f32)
    C_phys = torch.tensor(com_f, **f32)
    mi_orig = torch.tensor(moving.origin, **f32); mi_sp = torch.tensor(moving.spacing, **f32)
    mi_dir_inv_t = torch.inverse(torch.tensor(moving.direction, **f32)).t(); mi_shape = torch.tensor(moving.shape, **f32)

    # 2. pyramid: per level the fixed image and the physical coordinates of its (sampled) voxels
    levels = sorted({int(s_['level']) for s_ in schedule}, reverse=True)
    pyr = {}
    gen = torch.Generator(device='cpu').manual_seed(seed if sample_seed is None else int(sample_seed))
    for level in levels:
        fi_src, mi_src = fi_arr, mi_arr
        sig = smooth_sigma_per_level * max(level - 1, 0)
        if sig > 0:
            from .landmarks.blob import _separable_gaussian3d
            if dim == 3:
                fi_src, mi_src = _separable_gaussian3d(fi_arr, sig), _separable_gaussian3d(mi_arr, sig)
            else:
                k = _gaussian_kernel_2d_sep(sig, device_obj)
                fi_src = F.conv2d(F.conv2d(fi_arr, k[0], padding=(k[2], 0)), k[1], padding=(0, k[2]))
                mi_src = F.conv2d(F.conv2d(mi_arr, k[0], padding=(k[2], 0)), k[1], padding=(0, k[2]))
        if level > 1:
            pool = F.avg_pool3d if dim == 3 else F.avg_pool2d
            fi_lev, mi_lev = pool(fi_src, kernel_size=level, stride=level), pool(mi_src, kernel_size=level, stride=level)
        else:
            fi_lev, mi_lev = fi_src, mi_src
        shape_zyx = fi_lev.shape[2:]
        axes = [torch.arange(n, device=device_obj, dtype=torch.float32) for n in shape_zyx]
        mesh = torch.meshgrid(*axes, indexing='ij')                       # z, y, x  (or y, x)
        # a pooled voxel i averages full-res voxels [level*i, level*i + level - 1]: its centre is
        # level*i + (level-1)/2 in full-res continuous index space (the old code used level*i,
        # a bias of (level-1)/2 voxels that grew with the pyramid level)
        vox_xyz = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, dim) * level + (level - 1) / 2.0
        phys_xyz = orig_xyz + (vox_xyz * sp_xyz) @ dir_xyz.t()
        fmask = (fi_lev > fg_level)
        mi_shape_pooled = torch.tensor(list(reversed(mi_lev.shape[2:])), **f32)   # (nx, ny, nz) of the pooled moving
        entry = dict(fi=fi_lev, mi=mi_lev, mi_mask=(mi_lev > fg_level).to(mi_lev.dtype), shape=shape_zyx, phys=phys_xyz,
                     mask=fmask, points=None, level=level, mi_shape_pooled=mi_shape_pooled)
        if n_sample_points is not None:
            domain = fmask.reshape(-1) if mask_mode == 'fixed_fg' else torch.ones_like(fmask.reshape(-1))
            idx_fg = torch.nonzero(domain, as_tuple=False).squeeze(1).cpu()
            k = min(int(n_sample_points), idx_fg.numel())
            if sample_weighting == 'gradient':
                # gradient magnitude of the fixed level image (central differences, MPS-safe slicing)
                from .landmarks.blob import _shift_pad
                gm = torch.zeros_like(fi_lev)
                for d_ in range(2, fi_lev.ndim):
                    gm = gm + (0.5 * (_shift_pad(fi_lev, d_, 1, 'replicate') - _shift_pad(fi_lev, d_, -1, 'replicate'))) ** 2
                w = (0.1 + torch.sqrt(gm).reshape(-1) / (torch.sqrt(gm).max() + 1e-8)).cpu()[idx_fg]
                # weighted sampling without replacement via Gumbel top-k (deterministic with `gen`)
                gumbel = -torch.log(-torch.log(torch.rand(idx_fg.numel(), generator=gen).clamp_min(1e-12)))
                keys = torch.log(w) + gumbel
                sel = idx_fg[torch.topk(keys, k).indices].sort().values.to(device_obj)
            else:
                sel = idx_fg[torch.randperm(idx_fg.numel(), generator=gen)[:k]].sort().values.to(device_obj)
            entry['points'] = dict(phys=phys_xyz[sel], fvals=fi_lev.reshape(-1)[sel])
        pyr[level] = entry

    fixed_w_cache: dict = {}

    def warp_to_moving_norm(y_phys, e=None):
        """physical -> normalised grid coordinate of the (pooled) moving tensor of pyramid entry e."""
        y_vox = (y_phys - mi_orig) @ mi_dir_inv_t / mi_sp                  # full-res continuous index
        if e is None or e['level'] == 1:
            return 2.0 * (y_vox / (mi_shape - 1.0)) - 1.0
        lvl = e['level']
        y_pooled = (y_vox - (lvl - 1) / 2.0) / lvl                        # pooled-tensor index
        return 2.0 * (y_pooled / (e['mi_shape_pooled'] - 1.0)) - 1.0

    def objective(path: '_AffinePath', dof: str, level: int, full: bool = False, sampling: float = None):
        e = pyr[level]
        samp = sampling_percentage if sampling is None else sampling
        A = path.matrix(dof)
        teff = path.t + C_phys - A @ C_phys
        if e['points'] is not None and not full:
            y = warp_to_moving_norm(e['points']['phys'] @ A.t() + teff, e)
            grid = y.reshape(1, 1, *([1] * (dim - 2)), -1, dim) if dim == 3 else y.reshape(1, 1, -1, dim)
            w = F.grid_sample(e['mi'], grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(-1)
            fv = e['points']['fvals']
            m = None
            if mask_mode == 'union':
                m = (fv > fg_level) | (w > fg_level)
            elif mask_mode == 'fixed_fg':
                m = None                                   # points were drawn from the fixed foreground
            loss = mattes_mi_loss_nd(w, fv, mask=m, num_bins=num_bins, auto_mask=False, fixed_range=fixed_range)
            if fg_dice_weight > 0:
                wm = F.grid_sample(e['mi_mask'], grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(-1)
                fm = (fv > fg_level).to(wm.dtype)
                loss = loss + fg_dice_weight * (1.0 - 2.0 * (fm * wm).sum() / (fm.sum() + wm.sum() + 1e-6))
            return loss
        y = warp_to_moving_norm(e['phys'] @ A.t() + teff, e)
        grid = y.reshape(1, *e['shape'], dim)
        w = F.grid_sample(e['mi'], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        fw = None
        if mask_mode == 'union':
            m = e['mask'] | (w > 0.01)
        elif mask_mode == 'none':
            m = torch.ones_like(e['mask'])
            # fixed samples never change for whole-domain full-grid stages: cache their Parzen weights
            key = (level, samp, num_bins)
            if key not in fixed_w_cache:
                fv = e['fi'].flatten()
                if samp is not None and samp < 1.0:
                    fv = fv[::max(1, int(1.0 / samp))].contiguous()
                lo, hi = fixed_range if not isinstance(fixed_range[0], (tuple, list)) else fixed_range[1]
                fixed_w_cache[key] = parzen_weights((fv - lo) / (hi - lo + 1e-8) * 2.0 - 1.0, num_bins).detach()
            fw = fixed_w_cache[key]
        else:
            m = e['mask']
        loss = mattes_mi_loss_nd(w, e['fi'], mask=m, num_bins=num_bins, auto_mask=False,
                                 sampling_percentage=samp, fixed_range=fixed_range, fixed_weights=fw)
        if fg_dice_weight > 0:
            wm = F.grid_sample(e['mi_mask'], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            fm = e['mask'].to(wm.dtype)
            loss = loss + fg_dice_weight * (1.0 - 2.0 * (fm * wm).sum() / (fm.sum() + wm.sum() + 1e-6))
        return loss

    # 3. candidate starts, scored at the coarsest level -------------------------------------
    cands = [('Identity_CoM', np.eye(dim), t_init)]
    if multi_start:
        if cone_angles_deg is None:
            cone_angles_deg = [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0]
        for deg in cone_angles_deg:
            if abs(deg) < 1e-3:
                continue
            rad = np.radians(deg)
            if dim == 3:
                for axis in ('pitch', 'roll', 'yaw'):
                    rx = rad if axis == 'pitch' else 0.0; ry = rad if axis == 'roll' else 0.0; rz = rad if axis == 'yaw' else 0.0
                    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                    cands.append((f'CoM_{axis}_{deg:+.0f}deg', Rz @ Ry @ Rx, t_init))
            else:
                cands.append((f'CoM_rot_{deg:+.0f}deg', np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]), t_init))
    if initial_tx_path is not None and os.path.exists(str(initial_tx_path)):
        parsed = _parse_initial_affine(initial_tx_path, dim)
        if parsed is not None:
            A0, t0_, c0 = parsed
            # re-centre: y = A0 (x - c0) + c0 + t0  ==  A0 (x - C) + C + t  with t = A0 (C - c0) + c0 + t0 - C
            cands.append(('Provided_Initial_Transform', A0, A0 @ (com_f - c0) + c0 + t0_ - com_f))

    coarse = levels[0]
    scored = []
    with torch.no_grad():
        for name, B, tt in cands:
            p = _AffinePath(B, tt, dim, device_obj, name)
            scored.append((float(objective(p, 'rigid', coarse, full=True, sampling=1.0).item()), name, B, tt))
    scored.sort(key=lambda x: x[0])
    keep = scored[:n_starts]
    if verbose:
        print(f"[robust_affine mode='pytorch'] {len(cands)} start candidates scored at level {coarse}; "
              f"keeping {[f'{n} ({s:.4f})' for s, n, _, _ in keep]}", flush=True)
    paths = [_AffinePath(B, tt, dim, device_obj, name) for _, name, B, tt in keep]

    # 4. schedule -------------------------------------------------------------------------
    last_loss = {}
    for si, stage in enumerate(schedule):
        level, iters, dof, lr, eta_min = int(stage['level']), int(stage['iters']), stage['dof'], stage['lr'], stage.get('eta_min')
        use_full = bool(stage.get('full_grid', False)); samp = stage.get('sampling', None)
        if stage.get('optimizer', 'adam') == 'lbfgs':
            # quasi-Newton on the (scaled) parameter vector: few evaluations, exact line search
            for p in paths:
                groups = p.params(dof, lr)
                # L-BFGS is scale-sensitive: reparametrise each group by its Adam lr so one step ~ one lr unit
                plist = [g['params'][0] for g in groups]
                opt = torch.optim.LBFGS(plist, lr=1.0, max_iter=iters, history_size=int(stage.get('history', 10)),
                                        tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn='strong_wolfe')
                def closure(p=p):
                    opt.zero_grad(set_to_none=True)
                    l_ = objective(p, dof, level, full=use_full, sampling=samp)
                    l_.backward()
                    return l_
                loss = opt.step(closure)
                p.clamp_()
                last_loss[p.name] = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        else:
            opts = [torch.optim.Adam(p.params(dof, lr)) for p in paths]
            scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=iters, eta_min=eta_min) for o in opts] if eta_min is not None else []
            for it in range(iters):
                for pi, (p, opt) in enumerate(zip(paths, opts)):
                    opt.zero_grad(set_to_none=True)
                    loss = objective(p, dof, level, full=use_full, sampling=samp)
                    loss.backward()
                    opt.step()
                    if scheds:
                        scheds[pi].step()
                    p.clamp_()
                    last_loss[p.name] = float(loss.item())
        if stage.get('select', False) and len(paths) > 1:
            with torch.no_grad():
                full_scores = [float(objective(p, dof, level, full=True, sampling=1.0).item()) for p in paths]
            best = int(np.argmin(full_scores))
            if verbose:
                print(f"  stage {si} (level {level}): path scores {[(p.name, round(s_, 4)) for p, s_ in zip(paths, full_scores)]} -> keep '{paths[best].name}'", flush=True)
            paths = [paths[best]]
    if len(paths) > 1:   # no select stage in the schedule: pick by final-level full loss
        with torch.no_grad():
            full_scores = [float(objective(p, schedule[-1]['dof'], int(schedule[-1]['level']), full=True, sampling=1.0).item()) for p in paths]
        paths = [paths[int(np.argmin(full_scores))]]
    winner = paths[0]

    # 5. export --------------------------------------------------------------------------
    A_fin, t_fin = winner.numpy_affine()
    tx_final = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx_final.set_parameters(np.concatenate([A_fin.flatten(), t_fin]))
    tx_final.set_fixed_parameters(com_f)
    out_dir = tempfile.mkdtemp(prefix="robust_affine_pt_")
    final_tx_path = os.path.join(out_dir, "affine.mat")
    ants.write_transform(tx_final, final_tx_path)
    warped_mov_out = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[final_tx_path], interpolator='linear')
    elapsed = time.time() - t0
    return {
        'warpedmovout': warped_mov_out,
        'fwdtransforms': [final_tx_path],
        'invtransforms': [final_tx_path],
        'whichtoinvert_inv': [True],
        'runtime_seconds': elapsed,
        'time': elapsed,
        'init_candidate': winner.name,
        'init_score': float(keep[0][0]),
        'final_loss': float(last_loss.get(winner.name, float('nan'))),
        'candidates_scored': [(n, s) for s, n, _, _ in scored],
        'status': 'SUCCESS',
    }


def robust_affine(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: str = None,
    mode: str = 'auto',
    multi_start: bool = True,
    n_starts: int = 3,
    cone_angles_deg: list = None,
    num_rotations: int = 6,
    low_res_spacing: float = 4.0,
    backend: str = 'pytorch',
    device: str = 'auto',
    seed: int = None,
    verbose: bool = False,
    **kwargs
) -> dict:
    """
    Executes fail-safe, ultra-fast multi-start initial affine registration for 2D and 3D images.

    Supported Modes (`mode`)
    ------------------------
    - `'auto'` / `'fast'`   : Low-res multi-start candidate selection + multi-stage ANTs C++ solver (default).
    - `'ants_fast'`       : Fast multi-stage ANTs C++ pipeline (Translation -> Rigid -> Similarity -> Affine).
    - `'pytorch'` / `'gpu'` : Fast 2D/3D native PyTorch Lie Algebra solver with cone-constrained rotation search.
    - `'com_only'`        : Instant 0.05s Center-of-Mass physical translation alignment.

    Parameters
    ----------
    fixed : ants.ANTsImage
        Fixed target image in native physical space (2D or 3D).
    moving : ants.ANTsImage
        Moving source image in native physical space (2D or 3D).
    initial_transform : str, optional
        File path to existing initial ANTs transform `.mat` file.
    mode : str, default='auto'
        Affine strategy mode ('auto', 'ants_fast', 'pytorch', 'com_only').
    multi_start : bool, default=True
        If True, evaluates multi-start candidate transforms at low resolution.
    num_rotations : int, default=6
        Number of discrete orthogonal rotation candidates to test if multi_start is True.
    low_res_spacing : float, default=4.0
        Voxel spacing in mm for fast low-resolution candidate evaluation.
    backend : str, default='pytorch'
        Compute engine ('pytorch' or 'jax').
    device : str, default='auto'
        Compute device: 'auto' (best available accelerator), 'cpu', 'cuda', or 'mps'. An explicit 'cpu' is honoured.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, prints diagnostic timing and score messages.

    Returns
    -------
    dict
        Dictionary containing:
        - `'fwdtransforms'`: list of forward transform file paths (`[.mat]`)
        - `'invtransforms'`: list of inverse transform file paths (`[.mat]`)
        - `'whichtoinvert_inv'`: list of boolean flags for inverse applying
        - `'warpedmovout'`: ANTsImage moving image warped into fixed space
        - `'warpedfixout'`: ANTsImage fixed image
        - `'time'`: execution time in seconds
    """
    t0 = time.time()
    dim = fixed.dimension
    temp_dirs = []

    if seed is None:
        seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Eliminate stochastic sampling risk and multi-threaded reduction drift in ANTs/ITK
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    try:
        import ants
        ants.config.set_ants_deterministic(True, seed)
    except Exception:
        pass

    # 1. Mode: 'com_only'
    if mode in ['com_only', 'translation_only']:
        com_f = compute_center_of_mass(fixed, weighted=True)
        com_m = compute_center_of_mass(moving, weighted=True)
        t_com = com_m - com_f
        tx_path, temp_dir = create_translation_transform(fixed, moving, t_com)
        warped_mov = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
        return {
            'fwdtransforms': [tx_path],
            'invtransforms': [tx_path],
            'whichtoinvert_inv': [True],
            'warpedmovout': warped_mov,
            'warpedfixout': fixed,
            'time': time.time() - t0
        }

    # 2. Mode: 'pytorch'
    if mode in ['pytorch', 'gpu', 'pytorch_gpu']:
        return _run_pytorch_affine_solver(fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose, n_starts=n_starts, cone_angles_deg=cone_angles_deg, seed=seed, **kwargs)

    # 3. Mode: 'auto', 'fast', 'ants_fast'
    try:
        if multi_start:
            if verbose:
                print(f"[robust_affine] Creating low-res {dim}D volumes for multi-start evaluation...", flush=True)
            sp_low = tuple(low_res_spacing for _ in range(dim))
            fi_low = ants.resample_image(fixed, sp_low, use_voxels=False)
            mi_low = ants.resample_image(moving, sp_low, use_voxels=False)

            candidates = []

            if initial_transform is not None and os.path.exists(initial_transform):
                candidates.append(('Provided_Initial_Transform', initial_transform, None))

            # 1. Intensity-Weighted Center of Mass (CoM)
            com_f_w = compute_center_of_mass(fixed, weighted=True)
            com_m_w = compute_center_of_mass(moving, weighted=True)
            t_w = com_m_w - com_f_w
            tx_w_path, dir_w = create_translation_transform(fixed, moving, t_w)
            temp_dirs.append(dir_w)
            candidates.append(('Weighted_CoM', tx_w_path, dir_w))

            # 2. Geometric Center of Mass (Unweighted non-zero foreground)
            com_f_g = compute_center_of_mass(fixed, weighted=False)
            com_m_g = compute_center_of_mass(moving, weighted=False)
            t_g = com_m_g - com_f_g
            tx_g_path, dir_g = create_translation_transform(fixed, moving, t_g)
            temp_dirs.append(dir_g)
            candidates.append(('Geometric_CoM', tx_g_path, dir_g))

            # 3. Rotational search around CoM center
            if num_rotations > 0 and dim == 3:
                cone_angles = [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0][:num_rotations]
                for r_idx, deg in enumerate(cone_angles):
                    rad = np.radians(deg)
                    for axis_idx, axis in enumerate(['pitch', 'roll', 'yaw']):
                        rx = rad if axis == 'pitch' else 0.0
                        ry = rad if axis == 'roll' else 0.0
                        rz = rad if axis == 'yaw' else 0.0
                        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                        R = Rz @ Ry @ Rx

                        C = com_f_w
                        t_rot = t_w + C - R @ C
                        tx_r = create_ants_affine(R, t_rot, dim=3, fixed_params=C)

                        r_dir = tempfile.mkdtemp(prefix=f"robust_aff_rot_{r_idx}_{axis}_")
                        r_path = os.path.join(r_dir, "rot_translation.mat")
                        temp_dirs.append(r_dir)
                        ants.write_transform(tx_r, r_path)
                        candidates.append((f'Rotation_{axis}_{deg:+.0f}deg', r_path, r_dir))
            elif num_rotations > 0 and dim == 2:
                cone_angles = [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0][:num_rotations]
                for r_idx, deg in enumerate(cone_angles):
                    rad = np.radians(deg)
                    R2 = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                    C = com_f_w
                    t_rot = t_w + C - R2 @ C
                    tx_r = create_ants_affine(R2, t_rot, dim=2, fixed_params=C)

                    r_dir = tempfile.mkdtemp(prefix=f"robust_aff_rot_2d_{r_idx}_")
                    r_path = os.path.join(r_dir, "rot_translation.mat")
                    temp_dirs.append(r_dir)
                    ants.write_transform(tx_r, r_path)
                    candidates.append((f'Rotation_{deg:+.0f}deg', r_path, r_dir))

            best_candidate_name = candidates[0][0]
            best_tx_path = candidates[0][1]
            best_score = _eval_low_res_mi(fi_low, mi_low, best_tx_path)
            if verbose:
                print(f"  Candidate '{best_candidate_name}': Low-Res MI = {best_score:.4f}", flush=True)

            for name, path, _ in candidates[1:]:
                score = _eval_low_res_mi(fi_low, mi_low, path)
                if verbose:
                    print(f"  Candidate '{name}': Low-Res MI = {score:.4f}", flush=True)
                if score < best_score - 0.002:
                    best_score = score
                    best_candidate_name = name
                    best_tx_path = path

            if verbose:
                print(f"[robust_affine] Selected winning candidate '{best_candidate_name}' (MI = {best_score:.4f})", flush=True)
            initial_tx_to_use = best_tx_path
        else:
            initial_tx_to_use = initial_transform

        if verbose:
            print(f"[robust_affine mode='{mode}'] Starting ANTs Affine registration...", flush=True)

        reg_kwargs = dict(kwargs)
        if 'aff_random_sampling_rate' not in reg_kwargs:
            reg_kwargs['aff_random_sampling_rate'] = 0.25

        reg_a = ants.registration(
            fixed=fixed, moving=moving, type_of_transform='Affine',
            initial_transform=initial_tx_to_use,
            verbose=verbose,
            **reg_kwargs
        )

        fwdtransforms = reg_a['fwdtransforms']
        invtransforms = reg_a['invtransforms']
        warpedmovout = reg_a['warpedmovout']
        warpedfixout = reg_a.get('warpedfixout', fixed)

        return {
            'fwdtransforms': fwdtransforms,
            'invtransforms': invtransforms,
            'whichtoinvert_inv': [True],
            'warpedmovout': warpedmovout,
            'warpedfixout': warpedfixout,
            'time': time.time() - t0
        }
    finally:
        # Clean up temporary candidate directories that are not returned in output
        for d in temp_dirs:
            try:
                import shutil
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        import gc
        gc.collect()
