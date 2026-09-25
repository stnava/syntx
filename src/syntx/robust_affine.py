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
import logging

logger = logging.getLogger(__name__)

from .syn import mattes_mi_loss_nd
from .core.losses import parzen_weights
from .spatial import (
    image_to_tensor,
    get_image_metadata,
    get_spatial_coordinate_grid,
    create_ants_affine,
)


def robust_center_of_mass(
    img_ants: ants.ANTsImage,
    weighted: bool = True,
    threshold: float = 0.0,
    return_dict: bool = False,
):
    """
    Computes physical Center of Mass (CoM) on the positive and negative halves of the data.

    For medical images with both positive and negative values (such as CT scans where soft
    tissue is positive > 0 HU and air/lung is negative < 0 HU, or standardized/subtraction images),
    standard unpartitioned CoM suffers from severe first-moment cancellation, corrupting the
    center by 20–50+ mm.

    This function independently computes:
    1. Positive CoM (`com_pos`): Center of mass for voxels with value > +threshold (tissue mass).
    2. Negative CoM (`com_neg`): Center of mass for voxels with value < -threshold (air/cavity mass),
       weighted by absolute intensity |I(x)|. Returns `None` if no significant negative voxels exist
       (e.g., standard MRI, PET, histology).

    Uses O(1) memory marginal projection summation, computing CoM in <100 ms even for
    150M+ voxel volumes without allocating 3D coordinate meshgrids.

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2D or 3D ANTs image.
    weighted : bool, default=True
        If True, computes intensity-weighted physical center of mass.
        If False, computes geometric center of mass of the respective mask.
    threshold : float, default=0.0
        Threshold separating positive and negative domains (default 0.0 HU / intensity).
    return_dict : bool, default=False
        If True, returns a dict: {'positive': com_pos, 'negative': com_neg}.
        If False (default), returns a tuple: (com_pos, com_neg).

    Returns
    -------
    tuple of (np.ndarray or None, np.ndarray or None) or dict
        `com_pos` : Physical coordinates (x, y, z) or (x, y) of positive mass, or None.
        `com_neg` : Physical coordinates of negative mass, or None if no negative data.
    """
    arr = img_ants.numpy()
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    dim = img_ants.dimension

    thresh = max(float(threshold), 1e-6)

    # 1. Positive half
    pos_mask = arr > thresh
    com_pos = None
    if pos_mask.sum() >= 10:
        if weighted:
            w_pos = np.where(pos_mask, arr, 0.0).astype(np.float64)
        else:
            w_pos = pos_mask.astype(np.float64)
        tot_pos = float(w_pos.sum())
        if tot_pos > 1e-6:
            vox_pos = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                axes = tuple(j for j in range(dim) if j != i)
                vox_pos[i] = (w_pos.sum(axis=axes) * np.arange(arr.shape[i])).sum() / tot_pos
            com_pos = origin + direction @ (vox_pos * spacing)

    # 2. Negative half
    neg_mask = arr < -thresh
    com_neg = None
    if neg_mask.sum() >= 10:
        if weighted:
            w_neg = np.where(neg_mask, -arr, 0.0).astype(np.float64)
        else:
            w_neg = neg_mask.astype(np.float64)
        tot_neg = float(w_neg.sum())
        if tot_neg > 1e-6:
            vox_neg = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                axes = tuple(j for j in range(dim) if j != i)
                vox_neg[i] = (w_neg.sum(axis=axes) * np.arange(arr.shape[i])).sum() / tot_neg
            com_neg = origin + direction @ (vox_neg * spacing)

    if return_dict:
        return {"positive": com_pos, "negative": com_neg}
    return com_pos, com_neg


def compute_center_of_mass(img_ants: ants.ANTsImage, weighted: bool = True, return_both: bool = False):
    """
    Computes physical Center of Mass (CoM) of a 2D or 3D ANTsImage.
    Gracefully handles negative Hounsfield Unit CT scans and multi-modal images
    without moment cancellation.

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2D or 3D ANTs image.
    weighted : bool, default=True
        If True, computes intensity-weighted physical center of mass.
        If False, computes geometric center of mass of non-zero foreground mask.
    return_both : bool, default=False
        If True, returns `(com_pos, com_neg)` via `robust_center_of_mass`.
        If False (default), returns a single physical coordinate array `(dim,)`.

    Returns
    -------
    np.ndarray or tuple
        Physical space coordinates `(x, y)` or `(x, y, z)` of the predominant mass,
        or `(com_pos, com_neg)` if `return_both=True`.
    """
    com_pos, com_neg = robust_center_of_mass(img_ants, weighted=weighted)
    if return_both:
        return com_pos, com_neg

    if com_pos is not None:
        return com_pos
    if com_neg is not None:
        return com_neg

    # Fallback to geometric center of image volume if image is entirely zero
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    center_vox = (np.array(img_ants.shape) - 1.0) / 2.0
    return origin + direction @ (center_vox * spacing)


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


def _phase_correlation_translation_candidates(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    topk: int = 3,
    target_size: int = 64,
) -> list:
    """
    FFT-based phase correlation: a second, INDEPENDENT translation-init candidate generator
    alongside center-of-mass and field-of-view matching, general-purpose (not
    motion-correction-specific) since it's wired into the same candidate pool every
    `robust_affine` caller (motion_correction, auto_reg, build_template) already uses.

    Why: CoM is outlier-sensitive -- a corrupted or asymmetric image (bias field, partial
    FOV, pathology) can pull the weighted centroid meaningfully off the true rigid offset
    with no warning, and nothing downstream can tell it happened (diagnosed directly this
    session: CoM off by 7-11mm on frames a naive multi-start couldn't then recover, vs 1-2mm
    on well-behaved ones). Phase correlation is a fundamentally different, GLOBALLY exact
    (for pure translation, not iterative/local-optima-prone) estimator -- a single FFT --
    so its failure mode is unrelated to CoM's, making the two genuinely complementary rather
    than redundant. Returns multiple peaks (not just the best): a single peak can be
    fooled by real anatomical structure or by a rotation component invalidating the
    pure-shift assumption (also diagnosed directly -- disambiguating via the existing
    downstream cone-search + MI/correlation scoring, which already runs regardless, is the
    fix, not trusting one peak blindly).

    Returns
    -------
    list of np.ndarray, length `topk`, each a translation (mm) in the same convention as
    `t_com`/`t_fov` in `_generate_quick_search_candidates` (added to an Identity-rotation
    transform centered at the fixed image's own physical origin-relative frame).
    """
    dim = fixed.dimension
    if dim != 3:
        return []  # 2D path not implemented yet; falls back to CoM/FOV candidates only.
    dev = torch.device('cpu')  # one-time, small-tensor cost -- CPU avoids MPS per-call
    # overhead documented elsewhere in this codebase as dominant at this problem size.
    fixed_t = image_to_tensor(fixed, device=dev, to_zyx=True)[0, 0]
    moving_t = image_to_tensor(moving, device=dev, to_zyx=True)[0, 0]
    shape = fixed_t.shape
    level = max(1, int(round(max(shape) / target_size)))
    if level > 1:
        fixed_pool = F.avg_pool3d(fixed_t.unsqueeze(0).unsqueeze(0), kernel_size=level, stride=level)[0, 0]
        moving_pool = F.avg_pool3d(moving_t.unsqueeze(0).unsqueeze(0), kernel_size=level, stride=level)[0, 0]
    else:
        fixed_pool, moving_pool = fixed_t, moving_t
    shape_p = fixed_pool.shape

    def hann(n):
        return torch.hann_window(n, periodic=False, dtype=torch.float32)

    win = hann(shape_p[0]).view(-1, 1, 1) * hann(shape_p[1]).view(1, -1, 1) * hann(shape_p[2]).view(1, 1, -1)
    F_fixed = torch.fft.rfftn(fixed_pool * win)
    F_moving = torch.fft.rfftn(moving_pool * win, dim=(-3, -2, -1))
    Rspec = F_fixed * torch.conj(F_moving)
    Rspec = Rspec / (Rspec.abs() + 1e-8)
    r = torch.fft.irfftn(Rspec, s=shape_p, dim=(-3, -2, -1))

    sp = np.array(fixed.spacing, dtype=np.float64)
    orig_direction = np.array(fixed.direction, dtype=np.float64)
    min_sep = 2
    work = r.clone()
    out = []
    for _ in range(topk):
        flat = work.reshape(-1)
        peak = int(flat.argmax())
        pz = peak // (shape_p[1] * shape_p[2])
        rem = peak % (shape_p[1] * shape_p[2])
        py = rem // shape_p[2]
        px = rem % shape_p[2]

        def unwrap(v, size):
            return v - size if v > size / 2 else v

        vox_shift_xyz = np.array([unwrap(px, shape_p[2]), unwrap(py, shape_p[1]),
                                   unwrap(pz, shape_p[0])], dtype=np.float64) * level
        phys_shift = orig_direction @ (vox_shift_xyz * sp)
        out.append(-phys_shift)  # phase correlation gives moving->fixed; candidates need fixed->moving
        z0, z1 = max(0, pz - min_sep), min(shape_p[0], pz + min_sep + 1)
        y0, y1 = max(0, py - min_sep), min(shape_p[1], py + min_sep + 1)
        x0, x1 = max(0, px - min_sep), min(shape_p[2], px + min_sep + 1)
        work[z0:z1, y0:y1, x0:x1] = -1e9
    return out


def _generate_quick_search_candidates(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    cone_angles_deg: list = None,
    enable_phase_correlation: bool = True,
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

    t_phase_corr = []
    if enable_phase_correlation and dim == 3:
        try:
            t_phase_corr = _phase_correlation_translation_candidates(fixed, moving)
        except Exception as e:
            logger.warning("robust_affine: phase-correlation candidate generation failed (%s); "
                            "continuing with CoM/FOV only", e)

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

    # 2b. Phase-correlation candidates (Identity rotation), same pattern as CoM/FOV above --
    # each gets its own entry, AND (below) its own rotation-cone sweep, not just identity.
    # Giving a candidate only identity rotation was diagnosed directly (this session) as a
    # trap: a still-wrong rotation makes a good translation look bad under any similarity
    # metric, so a translation candidate needs a fair rotation search to be judged fairly.
    pc_base_entries = []
    for k, t_pc in enumerate(t_phase_corr):
        tx_pc = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
        tx_pc.set_parameters(np.concatenate([np.eye(dim).flatten(), t_pc]))
        tx_pc.set_fixed_parameters(com_f)  # rotate about the same physical center as CoM candidates
        r_dir_pc = tempfile.mkdtemp(prefix=f"robust_aff_id_pc{k}_")
        r_path_pc = os.path.join(r_dir_pc, "cone_rotation.mat")
        ants.write_transform(tx_pc, r_path_pc)
        candidates.append((f'Identity_PhaseCorr{k}', r_path_pc, np.eye(dim), t_pc, com_f, r_dir_pc))
        pc_base_entries.append((f'PhaseCorr{k}', t_pc, com_f))

    # 3. Rotational search around CoM, FOV, and phase-correlation centers
    if dim == 3:
        for base_name, t_base, C in [('CoM', t_com, com_f), ('FOV', t_fov, fov_f)] + pc_base_entries:
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


# keyword arguments understood only by _run_pytorch_affine_solver (filtered out of the ANTs path)
_PYTORCH_SOLVER_ONLY_KWARGS = frozenset({
    'schedule', 'preset', 'sampling_percentage', 'num_bins', 'n_sample_points', 'fixed_range', 'mask_mode',
    'smooth_sigma_per_level', 'fg_dice_weight', 'fg_level', 'sample_weighting', 'sample_seed',
    'enable_landmarks', 'lambda_shear', 'lambda_scale', 'cluster_threshold',
})


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

    Presets (3-D), chosen on the 90-pair Mindboggle cohort against ANTs C++ Affine
    (results/affine_baseline/; updated 2026-09-16 with ±24° cone, backend key pt2):
      'default'  : L4 rigid (exact grid) -> L3 affine (exact grid, 100 it, best-of-n_starts)
                   -> L2 affine (point sample) -> L1 affine (point sample); 32 bins.
                   Beats ANTs on most pairs with ±24° sparse cone (6° step), ~11 s on MPS.
      'accurate' : L4 rigid (exact) -> L2 affine (regular 50 % grid, 100 it, select) -> L1;
                   32 bins.  beats ANTs on most pairs (mean +0.0021), ~20 s.
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
                dict(level=4, iters=50, dof='rigid',  lr=(0.04, 0.008, 0.0, 0.0),        eta_min=0.002, select=False, sampling=0.50),
                dict(level=2, iters=50, dof='affine', lr=(0.015, 0.005, 0.003, 0.002),  eta_min=0.001, select=True,  sampling=0.05),
                dict(level=1, iters=30, dof='affine', lr=(0.005, 0.002, 0.001, 0.001),  eta_min=1e-4,  select=False, sampling=0.01),
            ]
        # Hybridized default (pt6): fast 3-level hierarchy (L4 -> L2 -> L1) parameterized by sampling percentages:
        # L4 uses 100% of coarse domain (sampling=1.0), L2 uses 10% (sampling=0.10, 100 iters with path select),
        # L1 uses 2% (sampling=0.02, 40 iters).
        # Accelerates execution from 25.7s to ~5.0s on Apple Silicon MPS (5.2x speedup) while
        # maintaining state-of-the-art cortical Dice (mean ~0.355).
        return [
            dict(level=4, iters=50, dof='rigid',  lr=(0.04, 0.008, 0.0, 0.0),        eta_min=0.002, select=False, sampling=1.0),
            dict(level=2, iters=100, dof='affine', lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True,  sampling=0.10),
            dict(level=1, iters=40, dof='affine', lr=(0.005, 0.002, 0.001, 0.001),  eta_min=1e-4,  select=False, sampling=0.02),
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


def _se3_distance(A1: np.ndarray, t1: np.ndarray, A2: np.ndarray, t2: np.ndarray, domain_diam: float = 200.0) -> float:
    """Riemannian geodesic distance on SE(3) between candidate transforms (A1, t1) and (A2, t2).

    d(T1, T2) = theta(R1, R2) + ||t1 - t2||_2 / domain_diam
    where theta(R1, R2) is the exact geodesic distance on SO(3).
    """
    dim = A1.shape[0]
    if dim == 3:
        U1, _, V1t = np.linalg.svd(A1[:3, :3])
        R1 = U1 @ V1t
        if np.linalg.det(R1) < 0:
            R1 = U1 @ np.diag([1.0, 1.0, -1.0]) @ V1t
        U2, _, V2t = np.linalg.svd(A2[:3, :3])
        R2 = U2 @ V2t
        if np.linalg.det(R2) < 0:
            R2 = U2 @ np.diag([1.0, 1.0, -1.0]) @ V2t
        R_rel = R1.T @ R2
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(tr))  # range [0, pi]
    else:
        theta1 = np.arctan2(A1[1, 0], A1[0, 0])
        theta2 = np.arctan2(A2[1, 0], A2[0, 0])
        dth = abs(theta1 - theta2) % (2.0 * np.pi)
        theta = float(min(dth, 2.0 * np.pi - dth))

    dt = float(np.linalg.norm(t1 - t2) / max(domain_diam, 1.0))
    return theta + dt


def _cluster_candidates_se3(
    scored_candidates: list,
    n_starts: int,
    cluster_threshold: float = 0.35,  # ~20 degrees
    domain_diam: float = 200.0,
) -> list:
    """Clusters candidate transforms by SE(3) Riemannian distance and selects the best representative
    from each distinct cluster to guarantee hypothesis diversity across capture basins."""
    if len(scored_candidates) <= n_starts:
        return scored_candidates

    # scored_candidates is sorted by loss (lowest loss first): [(loss, name, A, t), ...]
    clusters = []
    for cand in scored_candidates:
        loss, name, A, t = cand
        assigned = False
        for cl in clusters:
            rep = cl[0]
            dist = _se3_distance(A, t, rep[2], rep[3], domain_diam=domain_diam)
            if dist < cluster_threshold:
                cl.append(cand)
                assigned = True
                break
        if not assigned:
            clusters.append([cand])

    selected = [cl[0] for cl in clusters]
    if len(selected) >= n_starts:
        return selected[:n_starts]

    already_selected_names = {c[1] for c in selected}
    remaining = [c for c in scored_candidates if c[1] not in already_selected_names]
    selected.extend(remaining[:n_starts - len(selected)])
    return selected


def _extract_landmark_candidate(fixed: ants.ANTsImage, moving: ants.ANTsImage, com_f: np.ndarray,
                                low_res_spacing: float = 4.0, max_kpts: int = 96) -> tuple:
    """Extracts a fast, coarse physical-space Landmark RANSAC candidate (A, t) for robust_affine.

    Uses ``match_sift3d_with_rotation_search`` (PCA-seeded iterative rotation refinement +
    affine RANSAC at 2 mm) instead of the older naive 4 mm SIFT3D approach.  The PCA seeds
    handle large inter-scanner orientation differences; the affine RANSAC provides a full
    12-DOF starting point.  Runtime ≈ 4 s on MPS — well within the ≈24 s budget vs ANTs.
    """
    try:
        from .landmarks.orient import match_sift3d_with_rotation_search
        dim = fixed.dimension
        if dim != 3:
            return None

        sp_coarse = (2.0, 2.0, 2.0)
        fi_coarse = ants.resample_image(fixed, sp_coarse, use_voxels=False)
        mi_coarse = ants.resample_image(moving, sp_coarse, use_voxels=False)

        res = match_sift3d_with_rotation_search(
            fi_coarse, mi_coarse,
            ratio_thresh=0.92,
            ransac_model='affine',
            inlier_thresh_mm=8.0,
            ransac_iter=5000,
        )
        M = res.get('affine')
        inliers = res.get('inliers', [])
        if M is None or len(inliers) < 6:
            return None

        A_lm = M[:3, :3].astype(np.float64)
        det_A = float(np.linalg.det(A_lm))
        if det_A < 0.4 or det_A > 2.5:
            return None

        t_lm = M[:3, 3].astype(np.float64)
        # Re-center to com_f: y = A_lm (x - com_f) + com_f + t_cand  where y = A_lm x + t_lm
        t_cand = A_lm @ com_f + t_lm - com_f
        return ('Landmarks_RANSAC', A_lm, t_cand)
    except Exception as exc:
        logger.debug("Landmark candidate extraction bypassed: %s", exc)
        return None


class _AffinePath:
    """One optimisation path: A = R(omega) @ B @ diag(exp(s)) @ Shear(sh); y = A (x - C) + C + t."""

    def __init__(self, B: np.ndarray, t0: np.ndarray, dim: int, device, name: str, spacing: tuple = None):
        self.dim, self.name = dim, name
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.t = torch.tensor(t0, dtype=torch.float32, device=device, requires_grad=True)
        self.omega = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device, requires_grad=True)
        self.scale = torch.zeros(dim, dtype=torch.float32, device=device, requires_grad=True)
        self.shear = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device, requires_grad=True)

        # Anisotropy weights for Lie algebra Tikhonov shrinkage
        self.w_scale = None
        self.w_shear = None
        if spacing is not None:
            sp = np.asarray(spacing, dtype=np.float32)[:dim]
            s_min = max(float(sp.min()), 1e-4)
            w = sp / s_min  # e.g. [1.0, 1.0, 6.0]
            self.w_scale = torch.tensor(w, dtype=torch.float32, device=device)
            if dim == 3:
                # shear components: 0: xy (w0*w1), 1: xz (w0*w2), 2: yz (w1*w2)
                w_sh = [w[0] * w[1], w[0] * w[2], w[1] * w[2]]
                self.w_shear = torch.tensor(w_sh, dtype=torch.float32, device=device)
            else:
                self.w_shear = torch.tensor([w[0] * w[1]], dtype=torch.float32, device=device)

    def regularization_loss(self, dof: str, lambda_shear: float = 0.02, lambda_scale: float = 0.01) -> torch.Tensor:
        if dof != 'affine':
            return torch.tensor(0.0, device=self.t.device)
        reg = torch.tensor(0.0, device=self.t.device)
        if self.w_scale is not None:
            reg = reg + lambda_scale * torch.sum(self.w_scale * (self.scale ** 2))
        else:
            reg = reg + lambda_scale * torch.sum(self.scale ** 2)
        if self.w_shear is not None:
            reg = reg + lambda_shear * torch.sum(self.w_shear * (self.shear ** 2))
        else:
            reg = reg + lambda_shear * torch.sum(self.shear ** 2)
        return reg

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
                               n_starts: int = 3, cone_angles_deg: list = None, seed: int = 42,
                               schedule: list = None, preset: str = 'default', sampling_percentage: float = None, num_bins: int = 32,
                               n_sample_points: int = None, fixed_range=(0.0, 1.0), mask_mode: str = 'none',
                               smooth_sigma_per_level: float = 0.0, fg_dice_weight: float = 0.0,
                               fg_level: float = 0.01, sample_weighting: str = 'uniform',
                               sample_seed: int = None, enable_landmarks: bool = False,
                               lambda_shear: float = 0.02, lambda_scale: float = 0.01,
                               cluster_threshold: float = 0.35, **kwargs) -> dict:
    """
    Native PyTorch multi-resolution affine solver (``mode='pytorch'``), Mattes MI objective.

    Deterministic by construction: seeded candidate generation, fixed-bound MI with the
    deterministic Parzen histogram, and (optionally) a fixed seeded point sample per level.

    Parameters
    ----------
    initial_tx_path : str, optional
        ITK ``.mat`` used as an additional start candidate (and as the base matrix of that path).
    multi_start, n_starts, cone_angles_deg : list, optional
        Coarse-level candidate search: identity-at-CoM plus single-axis cone rotations
        (default ±6°, ±12°, ±18°, ±24° — 8 values at 6° step); the ``n_starts`` (default 3) best MI
        candidates are optimised in parallel until the ``select`` stage of the schedule, then only the
        best path continues.  The ±24° range is required to capture the large inter-scanner brain
        orientation differences that ANTs finds at its own coarse pyramid levels (see §AFFINE_GUIDE);
        finer steps cluster together and get pruned by the SE(3) diversity filter.
    schedule : list of dict
        See ``_default_affine_schedule``.  Any stage key may be overridden.
    preset : {'default', 'accurate', 'fast'}
        Named schedule (ignored when ``schedule`` is given).
    sampling_percentage : float, optional
        Fraction of domain voxels to sample at each pyramid level (e.g. 0.20 = 20% like ANTs,
        0.02 = 2% for fast high-throughput registration). Overrides or scales stage sampling.
    n_sample_points : int, optional
        Legacy absolute point count override (e.g. 150_000). None (default) calculates sample
        points dynamically from ``sampling_percentage``.
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
    enable_landmarks : bool, default=False
        Whether to add a SIFT3D + PCA-seeded rotation-search (``match_sift3d_with_rotation_search``)
        candidate to the 3D multi-start pool.  Disabled by default because the candidate adds
        ≈4 s overhead and competes with cone candidates at L4 MI scoring; when it scores just
        well enough to enter the top-N, it can displace a superior cone candidate and regress
        the result on hard inter-scanner pairs (Mindboggle 90-pair benchmark, 2026-09-16).
        Enable explicitly for modalities without reliable small-rotation cone coverage, or via
        ``preset='accurate'``.
    lambda_shear, lambda_scale : float
        Anisotropy-weighted Tikhonov regularization penalties for affine shear and scaling.
    cluster_threshold : float, default=0.35
        SE(3) Riemannian geodesic clustering threshold in radians for hypothesis diversity.
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
    # Presets are tuned against real head-sized volumes (hundreds of voxels/axis); on a
    # small volume, downsampling by the coarsest preset level (e.g. 4x on a 16-voxel axis)
    # leaves only a handful of voxels per axis at that pyramid level. The Mattes MI estimate
    # on that few samples is unstable and can converge to a spurious degenerate optimum --
    # observed as scale contraction (det(A) << 1) in an otherwise-correct affine fit -- before
    # the finer levels ever get a chance to correct it. Clamp the coarsest usable level so the
    # smallest spatial axis retains at least MIN_VOXELS_PER_AXIS voxels at every pyramid level.
    MIN_VOXELS_PER_AXIS = 8
    min_shape = min(int(s) for s in fixed.shape)
    max_level = max(1, min_shape // MIN_VOXELS_PER_AXIS)
    if any(int(s_['level']) > max_level for s_ in schedule):
        schedule = [dict(s_, level=min(int(s_['level']), max_level)) for s_ in schedule]
    if sampling_percentage is None and n_sample_points is None:
        if preset == 'fast':
            sampling_percentage = 0.01
        elif preset == 'accurate':
            sampling_percentage = 0.20
        else:
            sampling_percentage = 0.02
    n_starts = max(1, int(n_starts))
    sp_fix = fixed.spacing if hasattr(fixed, 'spacing') else None

    # 1. centres, normalisation, tensors --------------------------------------------------
    com_f_pos, com_f_neg = robust_center_of_mass(fixed, weighted=True)
    com_m_pos, com_m_neg = robust_center_of_mass(moving, weighted=True)
    com_f = np.asarray(com_f_pos if com_f_pos is not None else (com_f_neg if com_f_neg is not None else compute_center_of_mass(fixed, weighted=False)), dtype=np.float64)
    com_m = np.asarray(com_m_pos if com_m_pos is not None else (com_m_neg if com_m_neg is not None else compute_center_of_mass(moving, weighted=False)), dtype=np.float64)
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
        # Determine sampling percentage for this level:
        stage_samplings = [s.get('sampling') for s in schedule if int(s['level']) == level and s.get('sampling') is not None]
        level_sampling = stage_samplings[0] if stage_samplings else sampling_percentage

        k = None
        if n_sample_points is not None:
            domain = fmask.reshape(-1) if mask_mode == 'fixed_fg' else torch.ones_like(fmask.reshape(-1))
            idx_fg = torch.nonzero(domain, as_tuple=False).squeeze(1).cpu()
            k = min(int(n_sample_points), idx_fg.numel())
        elif level_sampling is not None and float(level_sampling) > 0.0 and float(level_sampling) < 1.0:
            domain = fmask.reshape(-1) if mask_mode == 'fixed_fg' else torch.ones_like(fmask.reshape(-1))
            idx_fg = torch.nonzero(domain, as_tuple=False).squeeze(1).cpu()
            k = max(1000, min(int(round(idx_fg.numel() * float(level_sampling))), idx_fg.numel()))

        if k is not None:
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
            if lambda_shear > 0 or lambda_scale > 0:
                loss = loss + path.regularization_loss(dof, lambda_shear=lambda_shear, lambda_scale=lambda_scale)
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
        if lambda_shear > 0 or lambda_scale > 0:
            loss = loss + path.regularization_loss(dof, lambda_shear=lambda_shear, lambda_scale=lambda_scale)
        return loss

    # 3. candidate starts, scored at the coarsest level -------------------------------------
    cands = []
    if initial_tx_path is not None and os.path.exists(str(initial_tx_path)):
        parsed = _parse_initial_affine(initial_tx_path, dim)
        if parsed is not None:
            A0, t0_, c0 = parsed
            # re-centre: y = A0 (x - c0) + c0 + t0  ==  A0 (x - C) + C + t  with t = A0 (C - c0) + c0 + t0 - C
            cands.append(('Provided_Initial_Transform', A0, A0 @ (com_f - c0) + c0 + t0_ - com_f))

    # Phase-correlation translation candidates: a second, INDEPENDENT estimator alongside
    # CoM (see _phase_correlation_translation_candidates's docstring for why -- CoM is
    # outlier-sensitive to bias fields/asymmetric content in a way phase correlation isn't,
    # and the two failure modes are unrelated, so this is complementary, not redundant).
    # topk=2 (not the default 3) and reusing the SAME cone_angles_deg sweep as CoM below,
    # rather than a separate search, to keep the added candidate-pool cost modest -- this
    # already-validated coarse-level scoring step evaluates every candidate before
    # clustering down to n_starts, so candidate count directly drives cost.
    t_phase_corr = []
    if multi_start and dim == 3:
        try:
            t_phase_corr = _phase_correlation_translation_candidates(fixed, moving, topk=2)
        except Exception as e:
            logger.warning("robust_affine: phase-correlation candidate generation failed (%s); "
                            "continuing with CoM candidates only", e)

    if not cands or multi_start:
        cands.append(('Identity_CoM', np.eye(dim), t_init))
        for k, t_pc in enumerate(t_phase_corr):
            cands.append((f'Identity_PhaseCorr{k}', np.eye(dim), t_pc))

        # 0b. Negative Half CoM and Dipole Candidate (when both images contain negative data, e.g. CT air/lungs)
        if com_f_neg is not None and com_m_neg is not None and multi_start:
            t_neg = com_m_neg - com_f_neg
            cands.append(('Identity_CoM_Negative', np.eye(dim), t_neg))
            if com_f_pos is not None and com_m_pos is not None and dim == 3:
                v_f = com_f_pos - com_f_neg
                v_m = com_m_pos - com_m_neg
                len_f, len_m = np.linalg.norm(v_f), np.linalg.norm(v_m)
                if len_f > 10.0 and len_m > 10.0:
                    u_f, u_m = v_f / len_f, v_m / len_m
                    dot_u = float(np.dot(u_f, u_m))
                    if dot_u < 0.98 and dot_u > -0.999:
                        v_cross = np.cross(u_f, u_m)
                        s = np.linalg.norm(v_cross)
                        c = dot_u
                        if s > 1e-4:
                            vx = np.array([[0, -v_cross[2], v_cross[1]],
                                           [v_cross[2], 0, -v_cross[0]],
                                           [-v_cross[1], v_cross[0], 0]])
                            R_dipole = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
                            cands.append(('CoM_Dipole_Vector', R_dipole, t_init))

        # 1. Automated Turnkey Landmark Candidate (3D only)
        if enable_landmarks and dim == 3 and multi_start:
            lm_cand = _extract_landmark_candidate(fixed, moving, com_f)
            if lm_cand is not None:
                cands.append(lm_cand)

        if multi_start:
            if cone_angles_deg is None:
                cone_angles_deg = [-24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0]
            # Give phase-correlation translations the SAME rotation-cone sweep as CoM, not
            # just identity rotation -- diagnosed directly (batched prototype work this
            # session): a translation candidate paired only with identity rotation can score
            # worse than a wrong-but-plausible-looking alternative purely because the true
            # pairing needs a real rotation too, starving a genuinely good translation of a
            # fair chance to be selected.
            rot_sweep_bases = [('CoM', t_init)] + [(f'PhaseCorr{k}', t_pc) for k, t_pc in enumerate(t_phase_corr)]
            for base_name, t_base in rot_sweep_bases:
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
                            cands.append((f'{base_name}_{axis}_{deg:+.0f}deg', Rz @ Ry @ Rx, t_base))
                    else:
                        cands.append((f'{base_name}_rot_{deg:+.0f}deg', np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]), t_base))

    coarse = levels[0]
    scored = []
    with torch.no_grad():
        for name, B, tt in cands:
            p = _AffinePath(B, tt, dim, device_obj, name, spacing=sp_fix)
            scored.append((float(objective(p, 'rigid', coarse, full=True, sampling=1.0).item()), name, B, tt))
    scored.sort(key=lambda x: x[0])


    domain_diam = 200.0
    if hasattr(fixed, 'spacing') and hasattr(fixed, 'shape'):
        domain_diam = float(np.linalg.norm(np.array(fixed.spacing) * np.array(fixed.shape)))

    prov = [s for s in scored if s[1] == 'Provided_Initial_Transform']
    if prov:
        others = [s for s in scored if s[1] != 'Provided_Initial_Transform']
        clustered_others = _cluster_candidates_se3(others, n_starts - len(prov),
                                                   cluster_threshold=cluster_threshold,
                                                   domain_diam=domain_diam)
        keep = (prov + clustered_others)[:n_starts]
    else:
        keep = _cluster_candidates_se3(scored, n_starts,
                                       cluster_threshold=cluster_threshold,
                                       domain_diam=domain_diam)

    if verbose:
        print(f"[robust_affine mode='pytorch'] {len(cands)} start candidates scored at level {coarse}; "
              f"keeping {[f'{n} ({s:.4f})' for s, n, _, _ in keep]}", flush=True)
    paths = [_AffinePath(B, tt, dim, device_obj, name, spacing=sp_fix) for _, name, B, tt in keep]

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


def _compute_salient_gradient_correlation(fi_norm: ants.ANTsImage, warped_norm: ants.ANTsImage) -> float:
    """Computes Pearson correlation strictly on the top 10% gradient magnitude within foreground."""
    f_arr = fi_norm.numpy()
    w_arr = warped_norm.numpy()
    fg = f_arr > 0.05
    if fg.sum() < 100:
        fg = f_arr > 0.0
    if fg.sum() < 20:
        return 0.0
    gx, gy, gz = np.gradient(f_arr)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    thresh = np.percentile(g_mag[fg], 90)
    salient = (g_mag >= thresh) & fg
    if salient.sum() < 50:
        salient = fg
    f_vals = f_arr[salient]
    w_vals = w_arr[salient]
    std_f = float(np.std(f_vals))
    std_w = float(np.std(w_vals))
    if std_f < 1e-6 and std_w < 1e-6:
        return 1.0 if np.allclose(f_vals, w_vals, atol=1e-3) else 0.0
    if std_f < 1e-6 or std_w < 1e-6:
        return 0.0
    c = np.corrcoef(f_vals, w_vals)[0, 1]
    return float(c) if not np.isnan(c) else -1.0


def _run_tournament_affine(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: str = None,
    device: str = 'auto',
    verbose: bool = False,
    seed: int = 42,
    preset: str = 'default',
    sampling_percentage: float = 0.02,
    **kwargs
) -> dict:
    """
    Tournament-based fail-safe robust affine registration pipeline.
    
    Generates multi-modal candidates:
    1. PyTorch fast intensity Lie algebra solver (CoM + orientation cones)
    2. Continuous Sampled Sinkhorn Optimal Transport (percentage-based MIND-SSC)
    3. Discrete SIFT3D Extrema Keypoints + RANSAC
    4. SO(3) sampled wide-angle rotation grid search
    
    Selects winning candidate via Consensus Landmark TRE gating and Salient Structural Correlation,
    then executes multi-resolution polish if non-intensity candidate wins.
    """
    t_start = time.time()
    from syntx.core.utils import normalize_image
    dim = fixed.dimension

    if dim != 3:
        return _run_pytorch_affine_solver(
            fixed, moving,
            initial_tx_path=initial_transform,
            preset=preset,
            device=device,
            verbose=verbose,
            seed=seed,
            **kwargs
        )

    from .landmarks import (
        sampled_optimal_transport_affine,
        score_rotation_candidates_sampled,
        detect_sift3d,
        match_landmarks,
        ransac_filter,
    )

    fi_n = normalize_image(fixed, method="auto")
    mi_n = normalize_image(moving, method="auto")

    if device in ['auto', None]:
        dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        dev = device

    candidates = {}

    # Candidate 1: Standard PyTorch Robust Affine (Fast multi-start)
    t0 = time.time()
    try:
        res_aff = _run_pytorch_affine_solver(
            fi_n, mi_n,
            initial_tx_path=initial_transform,
            preset="fast",
            device=dev,
            verbose=verbose,
            seed=seed,
            **kwargs
        )
        candidates["robust_affine"] = {
            "tx": res_aff["fwdtransforms"][0],
            "res": res_aff,
            "time": time.time() - t0,
            "type": "intensity"
        }
    except Exception as e:
        logger.warning("Tournament candidate 'robust_affine' failed: %s", e)

    # Candidate 2: Continuous Sampled Sinkhorn OT
    t0 = time.time()
    try:
        ot_res = sampled_optimal_transport_affine(
            fi_n, mi_n,
            sampling_percentage=sampling_percentage,
            return_dict=True,
            device=dev
        )
        candidates["sinkhorn_ot"] = {
            "tx": ot_res["transform_path"],
            "time": time.time() - t0,
            "type": "ot"
        }
    except Exception as e:
        logger.debug("Tournament candidate 'sinkhorn_ot' failed: %s", e)

    # Candidate 3: SIFT3D Extrema Keypoints + RANSAC
    has_sift = False
    pts_f_inl, pts_m_inl = None, None
    t0 = time.time()
    try:
        cf, df = detect_sift3d(fi_n, preprocess=False, max_keypoints=250)
        cm, dm = detect_sift3d(mi_n, preprocess=False, max_keypoints=250)
        if len(cf) >= 10 and len(cm) >= 10:
            m = match_landmarks(cf, cm, df, dm, ratio_thresh=0.92, mutual=True, device=dev)
            inl, M_sift = ransac_filter(cf, cm, m, model="rigid", max_iter=2000, inlier_thresh_mm=10.0)
            if len(inl) >= 4:
                has_sift = True
                pts_f_inl = cf[inl[:, 0], :3]
                pts_m_inl = cm[inl[:, 1], :3]
                tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
                tx.set_parameters(np.concatenate([M_sift[:3, :3].ravel(), M_sift[:3, 3]]))
                tx.set_fixed_parameters(np.zeros(3))
                with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
                    sift_tx = f.name
                ants.write_transform(tx, sift_tx)
                candidates["sift3d"] = {
                    "tx": sift_tx,
                    "time": time.time() - t0,
                    "type": "sift"
                }
    except Exception as e:
        logger.debug("Tournament candidate 'sift3d' failed: %s", e)

    # Candidate 4: Fast SO(3) Wide-Angle Rotation Search
    t0 = time.time()
    try:
        cand_rotations = [np.eye(3)]
        for deg in [-90, -45, 45, 90, 180]:
            rad = np.radians(deg)
            # pitch (X)
            cand_rotations.append(np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]]))
            # roll (Y)
            cand_rotations.append(np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]]))
            # yaw (Z)
            cand_rotations.append(np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]))
        rot_scores = score_rotation_candidates_sampled(
            fi_n, mi_n, cand_rotations,
            sampling_percentage=0.01,
            feature_type="mind",
            device=dev
        )
        best_rot_cand = rot_scores[0]
        ident_score = next((r["score"] for r in rot_scores if np.allclose(r["rotation"], np.eye(3))), -1.0)
        if best_rot_cand["score"] > ident_score + 0.05 and not np.allclose(best_rot_cand["rotation"], np.eye(3)):
            R_best = best_rot_cand["rotation"]
            cf_phys = np.array(compute_center_of_mass(fi_n))
            cm_phys = np.array(compute_center_of_mass(mi_n))
            t_rot = cm_phys - R_best @ cf_phys
            tx_r = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
            tx_r.set_parameters(np.concatenate([R_best.ravel(), t_rot]))
            tx_r.set_fixed_parameters(np.zeros(3))
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
                rot_tx = f.name
            ants.write_transform(tx_r, rot_tx)
            candidates["wide_rotation"] = {
                "tx": rot_tx,
                "time": time.time() - t0,
                "type": "rotation_search"
            }
    except Exception as e:
        logger.debug("Tournament wide-angle rotation search skipped: %s", e)

    if not candidates:
        raise RuntimeError("All tournament candidates failed to produce an initial transform.")

    # Tournament Scoring & Gating
    scores = {}
    for cand_name, info in candidates.items():
        tx_path = info["tx"]
        warped_i = ants.apply_transforms(fixed=fi_n, moving=mi_n, transformlist=[tx_path], interpolator="linear")
        c_sal = _compute_salient_gradient_correlation(fi_n, warped_i)

        tre = float("nan")
        if has_sift and pts_f_inl is not None:
            try:
                tx_itk = ants.read_transform(tx_path)
                p = np.array(tx_itk.parameters)
                fp = np.array(tx_itk.fixed_parameters)
                A = p[:9].reshape(3, 3)
                t = p[9:]
                c_itk = fp if len(fp) == 3 else np.zeros(3)
                pred_m = (A @ (pts_f_inl - c_itk).T).T + c_itk + t
                tre = float(np.mean(np.linalg.norm(pred_m - pts_m_inl, axis=1)))
            except Exception:
                pass

        scores[cand_name] = {
            "salient_cc": c_sal,
            "tre": tre,
            "info": info
        }

    # Selection Logic:
    # Rule 1: Consensus Landmark TRE Gating: Reject candidate with TRE > 15mm if a candidate with TRE < 6mm exists.
    tre_gate_passed = {}
    min_tre = min([s["tre"] for s in scores.values() if not np.isnan(s["tre"])], default=float("nan"))

    for c_name, s_val in scores.items():
        if not np.isnan(min_tre) and min_tre < 6.0:
            if not np.isnan(s_val["tre"]) and s_val["tre"] > 15.0:
                continue
        tre_gate_passed[c_name] = s_val

    if not tre_gate_passed:
        tre_gate_passed = scores

    # Rule 2: Pick highest Salient Correlation among surviving candidates
    winner_name = max(tre_gate_passed.keys(), key=lambda k: tre_gate_passed[k]["salient_cc"])
    winner_cand = candidates[winner_name]

    if verbose:
        logger.info(f"[robust_affine tournament] Selected winning candidate: '{winner_name}'")

    # Polish Step:
    if winner_name == "robust_affine":
        final_tx = winner_cand["tx"]
    else:
        try:
            res_polished = _run_pytorch_affine_solver(
                fi_n, mi_n,
                initial_tx_path=winner_cand["tx"],
                preset="fast",
                multi_start=False,
                device=dev,
                verbose=verbose,
                seed=seed,
                **kwargs
            )
            final_tx = res_polished["fwdtransforms"][0]
        except Exception as e:
            logger.warning("Polish of winning candidate '%s' failed (%s); retaining unpolished transform", winner_name, e)
            final_tx = winner_cand["tx"]

    warped_mov = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[final_tx])
    return {
        'fwdtransforms': [final_tx],
        'invtransforms': [final_tx],
        'whichtoinvert_inv': [True],
        'warpedmovout': warped_mov,
        'warpedfixout': fixed,
        'winner': winner_name,
        'candidates': list(candidates.keys()),
        'runtime_seconds': time.time() - t_start,
        'time': time.time() - t_start,
        'status': 'SUCCESS'
    }


def robust_affine(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: str = None,
    mode: str = 'auto',
    tournament: bool = False,
    multi_start: bool = True,
    n_starts: int = 3,
    cone_angles_deg: list = None,
    num_rotations: int = 6,
    low_res_spacing: float = 4.0,
    backend: str = 'pytorch',
    device: str = 'auto',
    seed: int = None,
    verbose: bool = False,
    enable_landmarks: bool = False,
    lambda_shear: float = 0.02,
    lambda_scale: float = 0.01,
    cluster_threshold: float = 0.35,
    dof: str = 'affine',
    **kwargs
) -> dict:
    """
    Executes fail-safe, ultra-fast multi-start initial affine registration for 2D and 3D images.

    dof : {'affine', 'rigid'}, default='affine'
        Degrees of freedom for the `'pytorch'`/`'auto'`/`'fast'` solver's schedule.
        `'affine'` (default) allows scale and shear, matching this function's original
        general-purpose inter-subject affine behaviour. `'rigid'` freezes scale/shear at
        identity for every schedule stage (translation + rotation only) -- use this for
        intra-subject alignment (e.g. frame-to-frame motion tracking), where the free
        `'affine'` solve can drift into spurious scale contraction that a Mattes-MI
        objective alone doesn't reliably penalise on small or low-texture volumes (see
        `docs/antsx_implementation_standards.md`). Ignored when an explicit `schedule`
        kwarg is passed. Has no effect on `mode='ants'`/`'ants_fast'`/`'com_only'`.

    Supported Modes (`mode`)
    ------------------------
    - `'auto'` / `'fast'`   : Native PyTorch multi-resolution Mattes-MI solver (default since 2026-09-15),
                              with automatic fail-safe fallback to the ANTs C++ path on any exception.
                              Equals or beats the ANTs C++ affine on 10/10 Mindboggle pairs
                              (mean Dice 0.3516 vs 0.3503) at ~1/3 the wall time, and is bitwise
                              reproducible per device.  See `docs/AFFINE_GUIDE.md`.
    - `'ants_fast'` / `'ants'`: Legacy multi-stage ANTs C++ pipeline (low-res multi-start candidate
                              selection + `ants.registration(type_of_transform='Affine')`).  Not
                              reproducible run-to-run; kept for provenance comparisons.
    - `'pytorch'` / `'gpu'` : Same solver as `'auto'` but without the ANTs fallback (raises on failure).
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

    try:
        import ants
        ants.config.set_ants_deterministic(True, seed)
    except Exception:
        pass

    # 0. Mode: 'tournament' (General-purpose multi-modal / multi-candidate affine)
    if tournament or mode in ['tournament', 'auto_tournament']:
        return _run_tournament_affine(
            fixed, moving,
            initial_transform=initial_transform,
            device=device,
            verbose=verbose,
            seed=seed,
            enable_landmarks=enable_landmarks,
            lambda_shear=lambda_shear,
            lambda_scale=lambda_scale,
            cluster_threshold=cluster_threshold,
            **kwargs
        )

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

    # 2. Mode: 'pytorch' (also the engine behind 'auto' / 'fast' since 2026-09-15)
    if mode in ['pytorch', 'gpu', 'pytorch_gpu', 'auto', 'fast']:
        if dof == 'rigid' and 'schedule' not in kwargs:
            base_schedule = _default_affine_schedule(dim, kwargs.get('preset', 'default'))
            kwargs['schedule'] = [dict(s_, dof='rigid') for s_ in base_schedule]
        elif dof not in ('rigid', 'affine'):
            raise ValueError(f"dof must be 'rigid' or 'affine', got {dof!r}.")
        try:
            return _run_pytorch_affine_solver(fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose,
                                              multi_start=multi_start, n_starts=n_starts, cone_angles_deg=cone_angles_deg,
                                              seed=seed, enable_landmarks=enable_landmarks, lambda_shear=lambda_shear,
                                              lambda_scale=lambda_scale, cluster_threshold=cluster_threshold, **kwargs)
        except Exception as e:
            if mode in ['pytorch', 'gpu', 'pytorch_gpu']:
                raise
            # 'auto' is fail-safe: fall back to the ANTs C++ path below
            logger.warning("robust_affine: PyTorch solver failed (%s); falling back to the ANTs C++ path", e)
            if verbose:
                print(f"[robust_affine] PyTorch solver failed ({e}); falling back to ANTs C++.", flush=True)
            kwargs = {k: v for k, v in kwargs.items() if k not in _PYTORCH_SOLVER_ONLY_KWARGS}

    # 3. Mode: 'ants_fast' / 'ants' (and the 'auto' fallback)
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

            # Automated Landmark Candidate for ANTs fallback
            if enable_landmarks and dim == 3:
                com_f_init = compute_center_of_mass(fixed, weighted=True)
                lm_cand = _extract_landmark_candidate(fixed, moving, com_f_init, low_res_spacing=low_res_spacing)
                if lm_cand is not None:
                    _, A_lm, t_lm = lm_cand
                    tx_lm = create_ants_affine(A_lm, t_lm, dim=3, fixed_params=com_f_init)
                    lm_dir = tempfile.mkdtemp(prefix="robust_aff_lm_")
                    lm_path = os.path.join(lm_dir, "landmark_affine.mat")
                    temp_dirs.append(lm_dir)
                    ants.write_transform(tx_lm, lm_path)
                    candidates.append(('Landmarks_RANSAC', lm_path, lm_dir))

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

        # solver-only options must not reach ants.registration (they would raise there)
        reg_kwargs = {k: v for k, v in kwargs.items() if k not in _PYTORCH_SOLVER_ONLY_KWARGS}
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
