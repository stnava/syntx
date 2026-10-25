"""
Robust initial linear (affine / rigid) alignment -- ``syntx.robust_affine``.

The standard way to get the initial affine for every syntx deformable registration (``tvf``,
``syngs``, ``syn``, ``greedy`` all call it when no ``initial_transform`` is given). It is built
to succeed from a bad start -- large translations, rotations of tens of degrees, contrast
differences -- by trying several starting poses on a coarse version of the images and refining
the best one.

Modes (``robust_affine(mode=...)``):

- ``'auto'`` (default): the native PyTorch solver below. Deterministic on a given device. If the
  solver raises, ``RobustAffineSolverError`` is raised (chaining the cause); there is NO silent
  fallback to ANTs. Every result carries ``'solver'`` ('pytorch', 'ants' or 'com_only').
- ``'pytorch'``, aliases ``'gpu'``, ``'pytorch_gpu'``: identical to ``'auto'`` (kept as aliases).
- ``'ants_fast'`` / ``'ants'``: low-resolution start-candidate search, then
  ``ants.registration`` with type_of_transform='Affine'. Not reproducible run to run.
- ``'com_only'`` / ``'translation_only'``: centre-of-mass translation only (instant).
- ``'tournament'`` / ``'auto_tournament'`` (or ``tournament=True``): 3-D only; competes several
  candidate generators (intensity solver, optimal transport, SIFT3D keypoints, wide rotation
  grid) and keeps the winner. 2-D inputs use the PyTorch solver.

The PyTorch solver: Mattes mutual information, multi-resolution (pyramid levels 4 -> 2 -> 1 in
3-D), rigid first then affine; start candidates are the centre-of-mass alignment plus
single-axis "cone" rotations (+-6 to +-24 degrees), of which the ``n_starts`` best are optimised in
parallel until a selection stage keeps one. Rotations use an exponential-map (Lie algebra)
parameterisation; the transform is written as an ITK ``.mat`` with its centre at the fixed
image's centre of mass. Details and validation: docs/AFFINE_GUIDE.md.
"""

import logging
import math
import os
import tempfile
import time

import ants
import warnings

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RobustAffineSolverError(RuntimeError):
    """The PyTorch affine solver failed or returned a pose no better than doing nothing.

    ``robust_affine`` never falls back to another solver on its own; when the cause is an
    exception it is chained (``__cause__``). Request ``mode='ants'`` explicitly to use ANTs.
    """

from .core.utils import require_finite_images
from .core.losses import (
    mattes_mi_loss_nd,
    mattes_sample_indices,
    parzen_weights,
    soft_dice_loss_nd,
)
from .pyramid import build_pyramid, cap_ants_image, prebuild_warp
from .slab_capture import check_setting, run_slab_capture
from .spatial import (
    create_ants_affine,
    get_image_metadata,
    get_spatial_coordinate_grid,
    image_to_tensor,
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


def _scoring_foreground(fi_low: ants.ANTsImage) -> np.ndarray:
    """Fixed-image foreground for candidate scoring, computed ONCE per scoring pass: the shared sampling region
    (``syntx.sampling``: above the image's own noise floor). An image with no usable background statistics (e.g. a
    constant non-zero background) falls back to the positive voxels, announced once."""
    from .sampling import strategy_on_grid

    try:
        return strategy_on_grid(fi_low.numpy(), fi_low.spacing, fi_low.origin, fi_low.direction, "uniform")[0]
    except ValueError as exc:
        warnings.warn(f"robust_affine: candidate-scoring foreground falls back to voxels > 0 ({exc})", RuntimeWarning, stacklevel=2)
        return fi_low.numpy() > 0


def _eval_low_res_mi(fi_low: ants.ANTsImage, mi_low: ants.ANTsImage, tx_path: str = None, fg: np.ndarray = None) -> float:
    """Evaluates low-resolution Mattes Mutual Information score with foreground masking for candidate transform
    (``fg``: ``_scoring_foreground(fi_low)``, computed once by the caller for all candidates)."""
    if fg is None:
        fg = _scoring_foreground(fi_low)
    try:
        if tx_path is None:
            warped = mi_low
        else:
            from .transforms import apply_transforms as _apply_transforms
            warped = _apply_transforms(fixed=fi_low, moving=mi_low, transformlist=[tx_path])
        from syntx.core.losses import mattes_mi_loss_nd
        f_t = image_to_tensor(fi_low)
        w_t = image_to_tensor(warped)
        # fixed-foreground mask (a union mask rewards shrinking overlap) and fixed histogram bounds so candidate scores
        # are comparable; the foreground is the shared sampling region (syntx.sampling: above the reference's noise floor)
        mask_t = image_to_tensor(fi_low.new_image_like(np.asarray(fg, dtype=np.float32))) > 0.5
        # histogram bounds from the data: fi_low / mi_low are raw intensities (cap_ants_image, not normalised); the
        # default (0, 1) range clipped them and every candidate scored ~0 (independent re-review)
        mi_score = mattes_mi_loss_nd(w_t, f_t, mask=mask_t, num_bins=32, fixed_range=None).item()
        return mi_score
    except (RuntimeError, ValueError) as exc:
        # a candidate whose transform cannot be applied or scored ranks last, announced (never silently)
        warnings.warn(f"robust_affine: candidate start {tx_path!r} could not be scored ({exc}); ranked last", RuntimeWarning, stacklevel=2)
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
            torch.stack([-u1, w0 if 'w0' in locals() else u0, torch.tensor(0.0, device=omega.device)])
        ]) if False else torch.stack([
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

    from .transforms import resample_image_to_target as _resample_image_to_target
    moving_native = moving
    fixed_aniso = _spacing_ratio(fixed.spacing) > ANISOTROPY_RESAMPLE_RATIO
    if not fixed_aniso and (moving.shape != fixed.shape or tuple(moving.spacing) != tuple(fixed.spacing)):
        moving = _resample_image_to_target(moving, fixed, interp_type=0)   # linear (1 was nearest-neighbour: aliased the pooled image)

    fixed_t = image_to_tensor(fixed, device=dev, to_zyx=True)[0, 0]
    shape = fixed_t.shape
    if fixed_aniso:
        ext_mm = [n * float(sp_) for n, sp_ in zip(fixed.shape, fixed.spacing)]
        tgt_mm = max(min(float(x) for x in fixed.spacing), max(ext_mm) / target_size)
        n_out = [max(1, int(round(e / tgt_mm))) for e in ext_mm]
        _lvl_f = build_pyramid(fixed_t.unsqueeze(0).unsqueeze(0), fixed.spacing, [tgt_mm / min(float(x) for x in fixed.spacing)],
                               policy='mm_iso', centre_to_centre=False, n_out_xyz=n_out)[0]
        fixed_pool, f_xyz = _lvl_f.tensor, list(_lvl_f.f)
        fine = max(min(float(x) for x in moving_native.spacing), min(ext_mm) / 64.0)
        n_fine = [max(1, int(round(e / fine))) for e in ext_mm]
        sp_fine = tuple(e / n for e, n in zip(ext_mm, n_fine))
        dirm = np.array(fixed.direction, dtype=np.float64)
        org_fine = np.array(fixed.origin, dtype=np.float64) + dirm @ np.array(
            [(sf - sx) / 2.0 for sf, sx in zip(sp_fine, fixed.spacing)])
        tpl = ants.make_image(tuple(n_fine), spacing=sp_fine, origin=tuple(org_fine), direction=fixed.direction)
        mov_fine = _resample_image_to_target(moving_native, tpl, interp_type=0)
        mov_t = image_to_tensor(mov_fine, device=dev, to_zyx=True)
        moving_pool = build_pyramid(mov_t, sp_fine, [tgt_mm / min(float(x) for x in fixed.spacing)], policy='mm_iso',
                                    reference_spacing=fixed.spacing, centre_to_centre=False, n_out_xyz=n_out)[0].tensor
        fixed_pool, moving_pool = fixed_pool[0, 0], moving_pool[0, 0]
        level_xyz = np.array(f_xyz, dtype=np.float64)
    else:
        moving_t = image_to_tensor(moving, device=dev, to_zyx=True)[0, 0]
        level = max(1, int(round(max(shape) / target_size)))
        # central pyramid (preset mm_iso: area average by the level); the per-axis factor actually used is returned, so a short axis that
        # cannot be pooled by `level` is reported as such instead of making the voxel-to-mm conversion wrong
        lv_f = build_pyramid(fixed_t.unsqueeze(0).unsqueeze(0), fixed.spacing, [level], policy='mm_iso')[0]
        lv_m = build_pyramid(moving_t.unsqueeze(0).unsqueeze(0), fixed.spacing, [level], policy='mm_iso')[0]
        fixed_pool, moving_pool = lv_f.tensor[0, 0], lv_m.tensor[0, 0]
        level_xyz = np.array(lv_f.f, dtype=np.float64)
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
                                   unwrap(pz, shape_p[0])], dtype=np.float64) * level_xyz
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
ROBUST_AFFINE_MODES = frozenset({
    'auto', 'pytorch', 'gpu', 'pytorch_gpu', 'ants_fast', 'ants', 'com_only', 'translation_only',
    'tournament', 'auto_tournament',
})

_PYTORCH_SOLVER_ONLY_KWARGS = frozenset({
    'schedule', 'preset', 'sampling_percentage', 'num_bins', 'n_sample_points', 'fixed_range', 'mask_mode',
    'smooth_sigma_per_level', 'fg_dice_weight', 'fg_level', 'sample_weighting', 'sample_seed',
    'enable_landmarks', 'lambda_shear', 'lambda_scale', 'cluster_threshold', 'wide_angles',
    'levels', 'iterations', 'reg_iterations', '_level_builder', 'valid_margin_slices',
    'translation_scan_steps',
})


def _default_affine_schedule(dim: int, preset: str = 'default', dof: str = 'affine') -> list:
    """Multi-resolution optimisation schedule (one dict per stage).

    level     : pyramid downsampling factor (avg-pool)                  iters : Adam steps
    dof       : 'rigid' (translation + rotation) or 'affine' (+ log-scale + shear)
    lr        : per-parameter-group learning rates (t, omega, scale, shear)
    eta_min   : cosine-annealing floor for the learning rates in absolute parameter units (None = constant)
    eta_frac  : cosine-annealing floor as a FRACTION of each group's learning rate (preferred; scale-free)
    step      : (translation, rotation, log-scale, shear) displacement of the sampled points per optimiser step, in voxels of the
                finest fixed voxel; converted to parameter units with the physical-shift scales (``core.parameter_scales``), so a
                schedule behaves the same at any image spacing and object size. ``lr`` (absolute parameter units) is still accepted.
    select    : if True, keep only the best path (full-sample MI at this level) after the stage
    full_grid : if True, this stage uses the voxel grid (regular stride = 1/sampling) instead of
                the random point sample — exact/regular objective, ANTs-like
    sampling  : per-stage regular sampling fraction for full_grid stages (default sampling_percentage)
    optimizer : 'adam' (default) or 'lbfgs' (strong-Wolfe L-BFGS, `iters` = max iterations)

    Presets (3-D), chosen on the 90-pair Mindboggle cohort against ANTs C++ Affine
    (results/affine_baseline/; updated 2026-10-21 with foreground-calibrated parameter_scales):
      'default'  : For 12-DOF affine (C3): L4 rigid (100% full_grid, select) -> L3 affine (100% full_grid, 100 it)
                   -> L2 affine (10% strided full_grid, 100 it) -> L1 affine (2% strided full_grid, 40 it);
                   32 bins. Beats ANTs on 8/10 pairs (mean +0.0021, 0.3524 vs 0.3503), ~8 s on MPS.
                   For 6-DOF rigid: L4 -> L2 -> L1 (voxel-scaled L-BFGS polish).
      'accurate' : 4-level dense/strided schedule with 50% L2 and 10% L1 full_grid sampling.
      'fast'     : L4 rigid (select) -> L2 affine -> L1 affine, strided full_grid (50% / 5% / 1%), ~4-6 s.
    """
    if dim == 3:
        if dof == 'rigid':
            if preset == 'accurate':
                return [
                    dict(level=4, iters=50, dof='rigid',  step=(0.02, 0.24, 0.0, 0.0),         eta_frac=0.05, select=False, full_grid=True, sampling=1.0),
                    dict(level=2, iters=100, dof='rigid', step=(0.0075, 0.15, 0.0675, 0.045), eta_frac=0.067, select=True,  full_grid=True, sampling=0.5),
                    dict(level=1, iters=40, dof='rigid',  step=(0.0025, 0.06, 0.0225, 0.0225),  optimizer='lbfgs', select=False, sampling=0.10),
                ]
            if preset == 'fast':
                return [
                    dict(level=4, iters=50, dof='rigid',  step=(0.02, 0.24, 0.0, 0.0),         eta_frac=0.05, select=False, sampling=0.50),
                    dict(level=2, iters=50, dof='rigid',  step=(0.0075, 0.15, 0.0675, 0.045),  eta_frac=0.067, select=True,  sampling=0.05),
                    dict(level=1, iters=30, dof='rigid',  step=(0.0025, 0.06, 0.0225, 0.0225),  eta_frac=0.02,  select=False, sampling=0.01),
                ]
            return [
                dict(level=4, iters=50, dof='rigid',  step=(0.02, 0.24, 0.0, 0.0),         eta_frac=0.05, select=False, sampling=1.0),
                dict(level=2, iters=100, dof='rigid', step=(0.0075, 0.15, 0.0675, 0.045), eta_frac=0.067, select=True,  sampling=0.10),
                dict(level=1, iters=40, dof='rigid',  step=(0.0025, 0.06, 0.0225, 0.0225),  optimizer='lbfgs', select=False, sampling=0.10),
            ]
        if preset == 'accurate':
            return [
                dict(level=4, iters=50,  dof='rigid',  step=(0.04, 0.375, 0.0, 0.0),     eta_frac=0.05,  select=False, full_grid=True, sampling=1.0),
                dict(level=3, iters=100, dof='affine', step=(0.015, 0.235, 0.10, 0.066), eta_frac=0.067, select=True,  full_grid=True, sampling=1.0),
                dict(level=2, iters=100, dof='affine', step=(0.015, 0.235, 0.10, 0.066), eta_frac=0.067, select=False, full_grid=True, sampling=0.50),
                dict(level=1, iters=40,  dof='affine', step=(0.005, 0.094, 0.033, 0.033), eta_frac=0.02, select=False, full_grid=True, sampling=0.10),
            ]
        if preset == 'fast':
            return [
                dict(level=4, iters=50, dof='rigid',  step=(0.04, 0.375, 0.0, 0.0),     eta_frac=0.05,  select=True,  full_grid=True, sampling=0.50),
                dict(level=2, iters=50, dof='affine', step=(0.015, 0.235, 0.10, 0.066), eta_frac=0.067, select=False, full_grid=True, sampling=0.05),
                dict(level=1, iters=30, dof='affine', step=(0.005, 0.094, 0.033, 0.033), eta_frac=0.02, select=False, full_grid=True, sampling=0.01),
            ]
        # Default 3-D 12-DOF affine schedule (C3): 4-level strided full-grid pyramid (L4 -> L3 -> L2 -> L1)
        # with foreground-calibrated physical parameter scales (core.parameter_scales).
        # Selecting the winning multi-start path at L4 (select=True) avoids running multiple paths through
        # dense L3 (~6.2 s saved), while retaining dense L3 for the winning path separates global 12-DOF
        # scale/shear from translation/rotation before fine cortical detail enters at L2/L1.
        # Deterministic strided voxel-centre subsampling (full_grid=True) at L2 (10%) and L1 (2%) avoids both
        # the 12-DOF L-BFGS line-search stall and sub-voxel random jitter noise, beating ANTs C++ on 8/10
        # Mindboggle pairs (0.3524 vs 0.3503, 2026-10-21).
        return [
            dict(level=4, iters=50,  dof='rigid',  lr=(0.04, 0.008, 0.0, 0.0),      eta_min=0.002, select=True,  full_grid=True, sampling=1.0),
            dict(level=3, iters=100, dof='affine', lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=False, full_grid=True, sampling=1.0),
            dict(level=2, iters=50,  dof='affine', lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=False, full_grid=True, sampling=0.10),
            dict(level=1, iters=30,  dof='affine', lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4,  select=False, full_grid=True, sampling=0.02),
        ]
    return [
        # 2-D keeps the validated absolute learning rates and fixed Adam blocks (scattered SyN, dewarp, 2-D benchmarks were tuned on them)
        dict(level=2, iters=50, dof='affine', lr=(0.015, 0.005, 0.003, 0.002), eta_min=0.001, select=True),
        dict(level=1, iters=30, dof='affine', lr=(0.005, 0.002, 0.001, 0.001), eta_min=1e-4,  select=False),
    ]


def _build_schedule_from_levels(
    levels,
    iterations=None,
    dof: str = 'affine',
    preset: str = 'default',
) -> list:
    """Build a multi-resolution affine optimization schedule from explicit pyramid levels.

    Parameters
    ----------
    levels : int or sequence of int
        Pyramid downsampling factors (shrink factors), coarse-to-fine (e.g. [4, 2, 1] or [8, 4, 2, 1]).
    iterations : int or sequence of int, optional
        Optimization iterations per level. If omitted, default iterations matching the preset are used.
    dof : {'affine', 'rigid'}, default 'affine'
        Degrees of freedom. 'rigid' enforces translation + rotation on all stages. 'affine' starts
        with rigid on the coarsest stage and refines affine on subsequent stages.
    preset : {'default', 'accurate', 'fast'}, default 'default'
        Named preset governing sampling fractions and optimizer defaults.
    """
    if isinstance(levels, (int, np.integer)):
        levels_list = [int(levels)]
    else:
        levels_list = [int(lvl) for lvl in levels]

    if not levels_list:
        raise ValueError("levels must be a non-empty sequence of positive integers")

    # Ensure coarse-to-fine order for multi-resolution stages
    if len(levels_list) > 1 and levels_list[0] < levels_list[-1]:
        levels_list = list(reversed(levels_list))
        if iterations is not None and not isinstance(iterations, (int, np.integer)):
            iterations = list(reversed(list(iterations)))

    n_stages = len(levels_list)

    if iterations is not None:
        if isinstance(iterations, (int, np.integer)):
            iters_list = [int(iterations)] * n_stages
        else:
            iters_list = [int(it) for it in iterations]
            if len(iters_list) < n_stages:
                iters_list = iters_list + [iters_list[-1]] * (n_stages - len(iters_list))
            elif len(iters_list) > n_stages:
                iters_list = iters_list[:n_stages]
    else:
        if n_stages == 1:
            iters_list = [50 if preset != 'fast' else 30]
        elif n_stages == 2:
            iters_list = [50, 40] if preset != 'fast' else [30, 20]
        elif n_stages == 3:
            iters_list = [50, 100, 40] if preset != 'fast' else [50, 50, 30]
        else:
            iters_list = [50] + [100] * (n_stages - 2) + [40]

    schedule = []
    for i, (lvl, iters) in enumerate(zip(levels_list, iters_list)):
        stage_dof = 'rigid' if (dof == 'rigid' or (i == 0 and n_stages > 1)) else 'affine'
        select = (i == 1) if n_stages > 1 else True

        if preset == 'fast':
            if lvl >= 4:
                sampling = 0.50
            elif lvl >= 2:
                sampling = 0.05
            else:
                sampling = 0.01
        elif preset == 'accurate':
            if lvl >= 4:
                sampling = 1.0
            elif lvl >= 2:
                sampling = 0.50
            else:
                sampling = 0.10
        else:  # default
            if lvl >= 4:
                sampling = 1.0
            elif lvl >= 2:
                sampling = 0.10
            else:
                sampling = 0.02

        if stage_dof == 'rigid':
            lr = (0.04, 0.008, 0.0, 0.0)
            eta_min = 0.002
        else:
            if lvl >= 2:
                lr = (0.015, 0.005, 0.003, 0.002)
                eta_min = 0.001
            else:
                lr = (0.005, 0.002, 0.001, 0.001)
                eta_min = 1e-4

        stage = dict(
            level=lvl,
            iters=int(iters),
            dof=stage_dof,
            lr=lr,
            eta_min=eta_min,
            select=select,
            sampling=sampling,
        )
        if preset == 'accurate' and lvl >= 2:
            stage['full_grid'] = True

        schedule.append(stage)

    return schedule


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
        # linear (central cap_ants_image); ants.resample_image defaults to NEAREST-NEIGHBOUR, which aliased the SIFT input
        fi_coarse = cap_ants_image(fixed, sp_coarse[0])
        mi_coarse = cap_ants_image(moving, sp_coarse[0])

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
        """Tikhonov penalty lambda_scale * sum(w * s^2) + lambda_shear * sum(w * sh^2) on the
        log-scales and shears (w: spacing anisotropy weights, 1 without spacing); 0 unless
        ``dof == 'affine'``."""
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
        """Linear part A = R(omega) @ B, times diag(exp(clamp(s, -0.4, 0.4))) @ Shear for
        ``dof == 'affine'``."""
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
        """Optimiser parameter groups: t and omega with lr[0], lr[1]; plus scale and shear with
        lr[2], lr[3] for ``dof == 'affine'``."""
        groups = [{'params': [self.t], 'lr': lr[0]}, {'params': [self.omega], 'lr': lr[1]}]
        if dof == 'affine':
            groups += [{'params': [self.scale], 'lr': lr[2]}, {'params': [self.shear], 'lr': lr[3]}]
        return groups

    @torch.no_grad()
    def clamp_(self):
        """In place: log-scales and shears to [-0.35, 0.35], rotation vector to [-pi/3, pi/3]."""
        self.scale.clamp_(-0.35, 0.35); self.shear.clamp_(-0.35, 0.35); self.omega.clamp_(-np.pi / 3, np.pi / 3)

    @torch.no_grad()
    def numpy_affine(self):
        """(A, t) as float64 arrays, A built with the full affine parameterisation."""
        A = self.matrix('affine').detach().cpu().numpy().astype(np.float64)
        return A, self.t.detach().cpu().numpy().astype(np.float64)


# An image whose largest/smallest voxel spacing exceeds this is "anisotropic": its pyramid levels are
# built in millimetres (isotropic working grids) instead of by pooling voxels (see pyramid.mm_iso).
ANISOTROPY_RESAMPLE_RATIO = 1.25

# self-check slack (loss units; Mattes MI loss is O(0.1-1)): the optimised pose may score at most this much worse
# than the best trivial baseline at the coarse level before robust_affine raises. The optimiser works on finer, point-sampled
# levels and the check on the coarse full grid, so for a pair whose start is already the optimum they disagree by a few percent
# of the loss (a 16 x 24 x 20 pair: 0.0098 of 0.42, 2.3 percent): the slack is the larger of an absolute floor and a fraction of the
# baseline loss. A failed solution is far outside it (a pose 60 mm off scores 0.09 against a baseline of 0.18).
SELF_CHECK_TOLERANCE = 5e-3
SELF_CHECK_RELATIVE_TOLERANCE = 0.05


def _check_nonneg_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer (0 disables), got {value!r}")
    return int(value)


def _spacing_ratio(spacing) -> float:
    sp = [float(x) for x in spacing]
    return max(sp) / max(min(sp), 1e-12)


def _run_pytorch_affine_solver(fixed: ants.ANTsImage, moving: ants.ANTsImage, initial_tx_path: str = None,
                               device: str = 'auto', verbose: bool = False, multi_start: bool = True,
                               n_starts: int = 3, cone_angles_deg: list = None, seed: int = 42,
                               schedule: list = None, preset: str = 'default', levels: list = None,
                               iterations: list = None, reg_iterations: list = None,
                               sampling_percentage: float = None, num_bins: int = 32,
                               n_sample_points: int = None, fixed_range=(0.0, 1.0), mask_mode: str = 'none',
                               smooth_sigma_per_level: float = 0.0, fg_dice_weight: float = 0.0,
                               fg_level: float = 0.01, sample_weighting: str = 'uniform',
                               sample_seed: int = None, enable_landmarks: bool = False,
                               lambda_shear: float = 0.02, lambda_scale: float = 0.01,
                               cluster_threshold: float = 0.35,
                               wide_angles: bool = False, max_pyramid_voxels: int = 2 ** 24, _level_builder=None,
                               convergence_tol_vox: float = 0.05, max_extend_factor: int = 4, max_sample_points: int = 250000,
                               valid_margin_slices: int = 1, translation_scan_steps: int = 0, **kwargs) -> dict:
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
    levels : int or sequence of int, optional
        Explicit pyramid downsampling shrink factors (e.g. [4, 2, 1] or [8, 4, 2, 1]).
    iterations : int or sequence of int, optional
        Optimization iterations per level (e.g. [50, 100, 40]).
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
    sample_weighting : {'uniform', 'gradient'} or a ``syntx.sampling`` menu name
        Point-sample distribution over the fixed domain, drawn by ``syntx.sampling.draw_indices`` (seeded).
        'gradient' is the menu's ``gradient_floor`` entry: probability ∝ 0.1 + |∇fixed| / max|∇fixed|
        (Gumbel top-k), concentrating the MI estimate on tissue boundaries. ``mask_mode='fixed_fg'`` samples
        the strategy's region (above the fixed image's own noise floor, below its saturated top).
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
    max_pyramid_voxels : int, default 2**24
        Anisotropic inputs (spacing ratio > ``ANISOTROPY_RESAMPLE_RATIO``) get millimetre-based
        pyramid levels: level L is an isotropic grid of spacing L x (finest native spacing),
        coarsened just enough to stay within this voxel count. Isotropic inputs are pooled by
        ``level`` voxels as before.
    valid_margin_slices : int, default 1
        Anisotropic FIXED image only (a no-op for an isotropic one): the outermost ``valid_margin_slices`` native slices at either
        end of the thick axis do not enter the Mattes metric (the slab-edge slice is partial-volume and has no counterpart in the
        moving image; including it biases a thin-slab pose along z by up to 26 mm on real neuromelanin data). 0 disables it.
        Must be a non-negative integer, otherwise ``ValueError``. See docs/PYRAMID.md.
    translation_scan_steps : int, default 0 (off)
        OPT-IN, anisotropic fixed image and ``multi_start`` only. Scores a lattice of +-``translation_scan_steps`` coarse-voxel
        translations around the CoM and phase-correlation starts (identity rotation, coarse level) and adds the two best well-separated
        points as further bases of the rotation-cone sweep. It fixes the one synthetic thin-slab failure (6:1, pose A: 4.4 -> 0.2 mm) and
        halves the 4:1 and 8:1 errors, but on the real neuromelanin pairs it moved sub-07 from 1.2 to 7.8 mm (a deeper coarse-level minimum
        with a worse native MI) and sub-01 from 0.4 to 1.5 mm, and costs about 40 percent more time, so it is not the default (docs/PYRAMID.md Section 13).

    Intensities: both images are always normalised with ``normalize_image(method='auto',
    force=True)`` (robust percentile clip to [0, 1]); an input already in [0, 1] is NOT passed through
    unchanged, so a min-max-squeezed image keeps no outlier-dominated range.
    """
    t0 = time.time()
    dim = fixed.dimension
    seed = 42 if seed is None else seed
    torch.manual_seed(seed); np.random.seed(seed)
    from .device import resolve_device as _resolve_device
    device_obj = torch.device(_resolve_device(device, task='affine'))
    dof = kwargs.get('dof', 'affine')
    if iterations is None:
        iterations = reg_iterations
    if iterations is None and 'reg_iterations' in kwargs:
        iterations = kwargs.pop('reg_iterations')
    if schedule is None:
        if levels is not None:
            schedule = _build_schedule_from_levels(levels, iterations=iterations, dof=dof, preset=preset)
        else:
            schedule = _default_affine_schedule(dim, preset, dof=dof)
            if iterations is not None:
                iters_list = [iterations] * len(schedule) if isinstance(iterations, (int, np.integer)) else list(iterations)
                for s_, it in zip(schedule, iters_list):
                    s_['iters'] = int(it)
    # Presets are tuned against real head-sized volumes (hundreds of voxels/axis); on a
    # small volume, downsampling by the coarsest preset level (e.g. 4x on a 16-voxel axis)
    # leaves only a handful of voxels per axis at that pyramid level. The Mattes MI estimate
    # on that few samples is unstable and can converge to a spurious degenerate optimum --
    # observed as scale contraction (det(A) << 1) in an otherwise-correct affine fit -- before
    # the finer levels ever get a chance to correct it. Clamp the coarsest usable level so the
    # smallest spatial axis retains at least MIN_VOXELS_PER_AXIS voxels at every pyramid level.
    MIN_VOXELS_PER_AXIS = 8
    aniso_f = dim == 3 and _spacing_ratio(fixed.spacing) > ANISOTROPY_RESAMPLE_RATIO
    aniso_m = dim == 3 and _spacing_ratio(moving.spacing) > ANISOTROPY_RESAMPLE_RATIO
    # private seam (pyramid-policy comparison harness, benchmarks/pyramid_compare.py): a builder may state which
    # coarsest-level clamp applies to it ('voxel' or 'mm'); None keeps the default rule. Default behaviour is unchanged.
    _clamp_rule = getattr(_level_builder, 'min_shape_rule', None) if _level_builder is not None else None
    if _clamp_rule is not None and _clamp_rule not in ('voxel', 'mm', 'none'):
        raise ValueError(f"_level_builder.min_shape_rule must be 'voxel', 'mm' or 'none', got {_clamp_rule!r}")
    if _clamp_rule == 'voxel':
        aniso_clamp = False
    elif _clamp_rule in ('mm', 'none'):
        aniso_clamp = dim == 3
    else:
        aniso_clamp = aniso_f
    if aniso_clamp:
        # millimetre levels: the working grid is isotropic at (level x finest spacing), so the
        # shortest field-of-view extent measured in finest-spacing voxels is what must stay >= 8
        sp_min_f = min(float(x) for x in fixed.spacing)
        min_shape = min(int(n * float(sp_ / sp_min_f)) for n, sp_ in zip(fixed.shape, fixed.spacing))
    else:
        min_shape = min(int(s) for s in fixed.shape)
    max_level = max(1, min_shape // MIN_VOXELS_PER_AXIS)
    if _clamp_rule == 'none':
        max_level = max(int(s_['level']) for s_ in schedule)
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
    from syntx.core.utils import normalize_image
    # force=True: robust percentile normalisation ALWAYS runs. With the default (force=False) an image
    # already in [0, 1] -- e.g. min-max normalised by the caller -- skips it and keeps an
    # outlier-dominated range, which makes the Mattes histogram worse conditioned.
    fixed_norm = normalize_image(fixed, method='auto', force=True)
    moving_norm = normalize_image(moving, method='auto', force=True)
    com_f_pos, com_f_neg = robust_center_of_mass(fixed_norm, weighted=True)
    com_m_pos, com_m_neg = robust_center_of_mass(moving_norm, weighted=True)
    com_f = np.asarray(com_f_pos if com_f_pos is not None else (com_f_neg if com_f_neg is not None else compute_center_of_mass(fixed_norm, weighted=False)), dtype=np.float64)
    com_m = np.asarray(com_m_pos if com_m_pos is not None else (com_m_neg if com_m_neg is not None else compute_center_of_mass(moving_norm, weighted=False)), dtype=np.float64)
    t_init = com_m - com_f
    fi_arr = image_to_tensor(fixed_norm, device=device_obj, to_zyx=True)
    mi_arr = image_to_tensor(moving_norm, device=device_obj, to_zyx=True)

    f32 = dict(dtype=torch.float32, device=device_obj)
    sp_xyz = torch.tensor(fixed.spacing, **f32); orig_xyz = torch.tensor(fixed.origin, **f32); dir_xyz = torch.tensor(fixed.direction, **f32)
    C_phys = torch.tensor(com_f, **f32)

    # 2. pyramid: per level the fixed image and the physical coordinates of its (sampled) voxels
    levels = sorted({int(s_['level']) for s_ in schedule}, reverse=True)
    pyr = {}
    gen = torch.Generator(device='cpu').manual_seed(seed if sample_seed is None else int(sample_seed))
    valid_margin_slices = _check_nonneg_int('valid_margin_slices', valid_margin_slices)
    translation_scan_steps = _check_nonneg_int('translation_scan_steps', translation_scan_steps)

    def _central_level(arr, level, spacing, img, sig_mm, fixed_role):
        """One level of the CENTRAL pyramid. Either image anisotropic: ``mm_iso`` for BOTH images (isotropic working grid of ``level`` x the
        FIXED image's finest spacing, never finer than the image's own spacing, centroid-exact area average, optional Gaussian of ``sig_mm``
        millimetres; the one-slice valid margin on the fixed image); both isotropic: the same ``mm_iso`` plan (area average by ``level``
        voxels, optional Gaussian of ``smooth_sigma_per_level * (level - 1)`` voxels). Both smooth through
        ``core.smoothing`` (replicate padding). Returns the ``Level`` (tensor, f, off, geometry for the warp)."""
        kw = dict(origin=img.origin, direction=img.direction)
        # the moving image is brought to the SAME working resolution (millimetres) as the fixed one whenever either image is anisotropic:
        # an isotropic 1 mm T1 pooled by `level` voxels would be 2-3x coarser than a 0.43 mm in-plane slab at the same level
        if aniso_f or aniso_m:
            return build_pyramid(arr, spacing, [level], policy='mm_iso', sigma=sig_mm, sigma_mode='physical',
                                 reference_spacing=fixed.spacing, max_voxels=max_pyramid_voxels,
                                 valid_margin=valid_margin_slices if fixed_role else 0, **kw)[0]
        return build_pyramid(arr, spacing, [level], policy='mm_iso', sigma=smooth_sigma_per_level * max(level - 1, 0), **kw)[0]

    sp_min_f = min(float(x) for x in fixed.spacing)
    for level in levels:
        # anisotropic images get millimetre-based levels (isotropic working grid of level x the finest
        # spacing of the FIXED image); isotropic ones are pooled by `level` voxels exactly as before
        sig_mm = smooth_sigma_per_level * max(level - 1, 0) * sp_min_f
        valid_lev = None
        if _level_builder is not None:
            # a builder returns (tensor, f_xyz, off_xyz) or (tensor, f_xyz, off_xyz, valid) where ``valid`` is a bool
            # tensor on the level grid selecting the fixed samples that enter the metric
            ctx = dict(level=level, sp_min_f=sp_min_f, sig_mm=sig_mm, budget=max_pyramid_voxels, device=device_obj)
            res_f = _level_builder('fixed', fi_arr, fixed.spacing, ctx)
            fi_lev, f_fix, o_fix = res_f[:3]
            valid_lev = res_f[3] if len(res_f) > 3 else None
            mi_lev, f_mov, o_mov = _level_builder('moving', mi_arr, moving.spacing, ctx)[:3]
        else:
            _lf = _central_level(fi_arr, level, fixed.spacing, fixed, sig_mm, True)
            fi_lev, f_fix, o_fix = _lf.tensor, list(_lf.f), list(_lf.off)
            valid_lev = _lf.valid
        if _level_builder is None:
            _lm = _central_level(mi_arr, level, moving.spacing, moving, sig_mm, False)
            mi_lev, f_mov, o_mov = _lm.tensor, list(_lm.f), list(_lm.off)
        shape_zyx = fi_lev.shape[2:]
        axes = [torch.arange(n, device=device_obj, dtype=torch.float32) for n in shape_zyx]
        mesh = torch.meshgrid(*axes, indexing='ij')                       # z, y, x  (or y, x)
        # working index i sits at full-res continuous index f*i + off: a pooled voxel averages f full-res
        # voxels so off = (f-1)/2 (f = level); the millimetre-based levels of anisotropic images have per-axis,
        # non-integer f and, on finer axes, off = 0 (centre-to-centre grid)
        f_t = torch.tensor(f_fix, **f32); o_t = torch.tensor(o_fix, **f32)
        vox_xyz = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, dim) * f_t + o_t
        phys_xyz = orig_xyz + (vox_xyz * sp_xyz) @ dir_xyz.t()
        fmask = (fi_lev > fg_level)
        if valid_lev is not None:
            valid_lev = valid_lev.to(device=device_obj, dtype=torch.bool).reshape(fmask.shape)
        # prebuilt once per level (the constants are not rebuilt in the objective's thousands of calls)
        warp_mov = prebuild_warp(moving.origin, moving.direction, moving.spacing, f_mov, o_mov, mi_lev.shape[2:], device_obj)
        entry = dict(fi=fi_lev, mi=mi_lev, mi_mask=(mi_lev > fg_level).to(mi_lev.dtype), shape=shape_zyx, phys=phys_xyz,
                     mask=fmask, valid=valid_lev, points=None, level=level, warp=warp_mov)
        # Determine sampling percentage for this level:
        stage_samplings = [s.get('sampling') for s in schedule if int(s['level']) == level and s.get('sampling') is not None]
        level_sampling = stage_samplings[0] if stage_samplings else sampling_percentage
        needs_point_sample = any(not bool(s.get('full_grid', False)) for s in schedule if int(s['level']) == level)

        # sampling region and weights of this level from the shared menu (syntx.sampling), computed only when used:
        # 'fixed_fg' samples the strategy's region (above the fixed image's own noise floor); a non-uniform weighting
        # draws with the strategy's weights. 'gradient' keeps its meaning: the gradient_floor entry (0.1 + |grad| / max).
        strategy_name = {'uniform': 'uniform', 'gradient': 'gradient_floor'}.get(sample_weighting, sample_weighting)
        lvl_region = lvl_w = None
        if mask_mode == 'fixed_fg' or strategy_name != 'uniform':
            from .sampling import strategy_on_grid

            lvl_xyz = fi_lev[0, 0].permute(*reversed(range(dim))).detach().cpu().numpy()
            lvl_sp = [float(f_fix[i]) * float(fixed.spacing[i]) for i in range(dim)]
            dir_np = np.asarray(fixed.direction, dtype=np.float64)
            lvl_origin = np.asarray(fixed.origin, dtype=np.float64) + dir_np @ (np.asarray(o_fix, dtype=np.float64) * np.asarray(fixed.spacing))
            # the weights cover the sampling domain: the strategy region for 'fixed_fg', every voxel otherwise (no region
            # computed then); a fixed image without usable background statistics falls back to its positive voxels
            try:
                reg_np, w_np, _ = strategy_on_grid(lvl_xyz, lvl_sp, lvl_origin, dir_np, strategy_name,
                                                   weights_on_region=(mask_mode == 'fixed_fg'))
            except ValueError as exc:
                warnings.warn(f"robust_affine: level {level} sampling region falls back to voxels > 0 ({exc})", RuntimeWarning, stacklevel=2)
                reg_np = lvl_xyz > 0
                w_np = strategy_on_grid(lvl_xyz, lvl_sp, lvl_origin, dir_np, strategy_name, weights_on_region=False)[1]
            if reg_np is not None:
                lvl_region = torch.from_numpy(np.ascontiguousarray(np.transpose(reg_np, tuple(reversed(range(dim)))))).reshape(-1)
            if w_np is not None:
                lvl_w = torch.from_numpy(np.ascontiguousarray(np.transpose(w_np, tuple(reversed(range(dim)))).astype(np.float32))).reshape(-1)
        domain = lvl_region.to(fmask.device) if mask_mode == 'fixed_fg' else torch.ones_like(fmask.reshape(-1))
        if valid_lev is not None:
            domain = domain & valid_lev.reshape(-1)
        idx_fg = torch.nonzero(domain, as_tuple=False).squeeze(1).cpu()
        sel_all = idx_fg.to(device_obj)
        # Candidate scoring, path selection, full_grid stages and the self-check COMPARE poses with the same bias on both sides, on every domain voxel:
        # they stay on the voxel centres (jittering them mis-ranked starts on sharp synthetic images: scattered-SyN 2-D, 2026-10-21).
        entry['points_full'] = dict(phys=phys_xyz[sel_all], fvals=fi_lev.reshape(-1)[sel_all])
        entry['points_jit'] = None

        if needs_point_sample:
            # Point-sampled stages (full_grid=False) use points drawn uniformly WITHIN the fixed voxels at level 1, with the fixed
            # value interpolated there (still DWI phantom, 2026-10-21). Skipped when every stage at this level uses full_grid=True.
            shape_xyz_t = torch.tensor([float(n) for n in reversed(shape_zyx)], **f32)
            jitter_amp = 1.0 if level == 1 else 0.0   # the pose-dependent noise-averaging bias is a fine-scale effect; pooled levels are smoothed
            work_xyz = (vox_xyz[sel_all] - o_t) / f_t + jitter_amp * (torch.rand((sel_all.numel(), dim), generator=gen) - 0.5).to(device_obj)
            work_xyz = torch.minimum(work_xyz.clamp(min=0.0), shape_xyz_t - 1.0)
            norm = (2.0 * work_xyz / (shape_xyz_t - 1.0) - 1.0).reshape(1, *([1] * (dim - 1)), -1, dim)
            fvals_all = F.grid_sample(fi_lev, norm, mode='bilinear', padding_mode='border', align_corners=True).reshape(-1)
            entry['points_jit'] = dict(phys=orig_xyz + ((work_xyz * f_t + o_t) * sp_xyz) @ dir_xyz.t(), fvals=fvals_all)

            k = None
            if n_sample_points is not None:
                k = min(int(n_sample_points), idx_fg.numel())
            elif level_sampling is not None and float(level_sampling) > 0.0 and float(level_sampling) < 1.0:
                k = max(1000, min(int(round(idx_fg.numel() * float(level_sampling))), idx_fg.numel()))
                if max_sample_points is not None:
                    k = min(k, max(1000, int(max_sample_points)))   # a 12-parameter fit needs ~1e5 points, not 10 % of a 12-million-voxel volume

            if k is not None:
                # the shared draw (syntx.sampling.draw_indices): uniform = the seeded randperm used before; otherwise
                # Gumbel top-k with the strategy's weights (deterministic with `gen`)
                from .sampling import draw_indices

                w_dom = None if lvl_w is None else lvl_w[idx_fg]
                pos = draw_indices(w_dom, int(idx_fg.numel()), k, gen).to(device_obj)
                entry['points'] = dict(phys=entry['points_jit']['phys'][pos], fvals=entry['points_jit']['fvals'][pos])
        pyr[level] = entry

    # Gate default Tikhonov scale/shear shrinkage by anisotropy so isotropic 3-D inter-subject brain scaling (5-12%)
    # is not penalised unless the caller explicitly passes non-default regularization weights.
    eff_lambda_shear = float(lambda_shear) if (aniso_f or aniso_m or lambda_shear != 0.02) else 0.0
    eff_lambda_scale = float(lambda_scale) if (aniso_f or aniso_m or lambda_scale != 0.01) else 0.0

    def objective(path: '_AffinePath', dof: str, level: int, full: bool = False, sampling: float = None):
        e = pyr[level]
        samp = sampling_percentage if sampling is None else sampling
        A = path.matrix(dof)
        teff = path.t + C_phys - A @ C_phys
        if e['points'] is not None and not full:
            phys_p, fv = e['points']['phys'], e['points']['fvals']
        else:
            phys_p, fv = e['points_full']['phys'], e['points_full']['fvals']
            if samp is not None and float(samp) < 1.0:
                eff_samp = min(1.0, max(float(samp), 1000.0 / max(fv.numel(), 1)))
                if eff_samp < 1.0:
                    idx = mattes_sample_indices(fv.numel(), eff_samp, fv.device)   # the regular subsample of the dense evaluation
                    phys_p, fv = phys_p[idx], fv[idx]
        y = e['warp'](phys_p @ A.t() + teff)
        grid = y.reshape(1, 1, *([1] * (dim - 2)), -1, dim) if dim == 3 else y.reshape(1, 1, -1, dim)
        w = F.grid_sample(e['mi'], grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(-1)
        m = None
        if mask_mode == 'union':
            m = (fv > fg_level) | (w > fg_level)
        # 'fixed_fg' and the valid margin are already in the point domain; 'none' is every voxel
        loss = mattes_mi_loss_nd(w, fv, mask=m, num_bins=num_bins, auto_mask=False, fixed_range=fixed_range)
        if fg_dice_weight > 0:
            wm = F.grid_sample(e['mi_mask'], grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(-1)
            fm = (fv > fg_level).to(wm.dtype)
            loss = loss + fg_dice_weight * soft_dice_loss_nd(fm, wm)
        if eff_lambda_shear > 0 or eff_lambda_scale > 0:
            loss = loss + path.regularization_loss(dof, lambda_shear=eff_lambda_shear, lambda_scale=eff_lambda_scale)
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

    # Translation scan (anisotropic fixed image): the capture range of the metric in every translation axis is about one coarse voxel
    # (measured, docs/PYRAMID.md Section 13), so a start whose translation is several coarse voxels off (CoM or a phase-correlation
    # peak on a thin slab) never reaches the right basin. Score a lattice of translations (step = one coarse voxel, +-translation_scan_steps
    # steps along the fixed image's own axes) around every translation base with the identity rotation at the coarse level, and
    # add the best well-separated lattice points as further bases of the rotation-cone sweep.
    scan_bases = []
    if translation_scan_steps > 0 and multi_start and dim == 3 and aniso_f:
        coarse_mm = float(levels[0]) * sp_min_f
        Dm = np.asarray(fixed.direction, dtype=np.float64).reshape(3, 3)
        rng_k = range(-translation_scan_steps, translation_scan_steps + 1)
        lattice = [Dm @ (coarse_mm * np.array([i, j, k], dtype=np.float64)) for i in rng_k for j in rng_k for k in rng_k]
        scanned = []
        with torch.no_grad():
            for b_ in [t_init] + list(t_phase_corr):
                for off_ in lattice:
                    tt_ = b_ + off_
                    scanned.append((float(objective(_AffinePath(np.eye(dim), tt_, dim, device_obj, 'scan', spacing=sp_fix), 'rigid',
                                                    levels[0], full=True, sampling=1.0).item()), tt_))
        scanned.sort(key=lambda x: x[0])
        for _, tt_ in scanned:
            if all(np.linalg.norm(tt_ - o_) >= 1.5 * coarse_mm for o_ in [t_init] + list(t_phase_corr) + scan_bases):
                scan_bases.append(tt_)
            if len(scan_bases) >= 2:
                break

    if not cands or multi_start:
        cands.append(('Identity_CoM', np.eye(dim), t_init))
        # The self-check below uses plain Identity as a baseline, so it must also be a start: a pair that is already
        # aligned in physical space (shared origin, e.g. synthetic or same-scanner data) has its optimum AT identity
        # and a CoM offset is then a worse start than doing nothing. Skipped when CoM coincides with identity.
        if np.linalg.norm(t_init) > 1e-6:
            cands.append(('Identity', np.eye(dim), np.zeros(dim)))
        for k, t_pc in enumerate(t_phase_corr):
            cands.append((f'Identity_PhaseCorr{k}', np.eye(dim), t_pc))
        for k, t_sc in enumerate(scan_bases):
            cands.append((f'Identity_Scan{k}', np.eye(dim), t_sc))

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
                if wide_angles:
                    cone_angles_deg = [-180.0, -90.0, -45.0, -24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0, 45.0, 90.0, 180.0]
                else:
                    cone_angles_deg = [-24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0]
            rot_sweep_bases = ([('CoM', t_init)] + [(f'PhaseCorr{k}', t_pc) for k, t_pc in enumerate(t_phase_corr)]
                               + [(f'Scan{k}', t_sc) for k, t_sc in enumerate(scan_bases)])
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
    # Every stage ends when the path has CONVERGED (the sampled points move < convergence_tol_vox working voxels between
    # checkpoints, twice in a row), not when an iteration count is spent; `iters` is one block, a stage that has not converged
    # is extended up to `max_extend_factor` blocks and then reported as capped. The translation learning rate (mm) is scaled
    # by the finest fixed voxel relative to the 2 mm it was tuned at, so the schedule behaves the same at any image spacing.
    from .core.optimizers import PointStepTracker
    from .core.parameter_scales import ParamAxis, matrix_element_scale, probe_manifold, rotation_axis_scale
    last_loss = {}
    convergence_log = []
    n_check_pts = 2000

    def _check_pts_for_level(level):
        e = pyr[level]
        if '_pts_check' not in e:
            src_dict = e['points'] if e['points'] is not None else e['points_full']
            phys_all, fvals_all = src_dict['phys'], src_dict['fvals']
            # Measure physical lever-arm scales on the foreground anatomy (fvals > fg_level) rather than
            # empty NIfTI background air corners when calibrating full-grid affine stages.
            if e['points'] is None:
                fg_sel = fvals_all > fg_level
                src = phys_all[fg_sel] if int(fg_sel.sum().item()) >= 64 else phys_all
            else:
                src = phys_all
            e['_pts_check'] = src[::max(1, src.shape[0] // n_check_pts)]
        return e['_pts_check']

    def _path_points(p, dof, level):
        pts = _check_pts_for_level(level)
        with torch.no_grad():
            A = p.matrix(dof)
            return (pts @ A.t() + (p.t + C_phys - A @ C_phys)).unsqueeze(0)

    def _path_axes(p, dof, level):
        pts = _check_pts_for_level(level)
        ax = [ParamAxis(p.t, (j,), 1.0) for j in range(dim)]
        if dim == 3:
            ax += [ParamAxis(p.omega, (j,), rotation_axis_scale(pts, C_phys, j)) for j in range(3)]
        else:
            ax += [ParamAxis(p.omega, (0,), float((pts - C_phys).pow(2).sum(-1).mean().sqrt()))]
        if dof == 'affine':
            ax += [ParamAxis(p.scale, (j,), matrix_element_scale(pts, C_phys, j)) for j in range(dim)]
            cols = [1, 2, 2] if dim == 3 else [1]
            ax += [ParamAxis(p.shear, (j,), matrix_element_scale(pts, C_phys, c)) for j, c in enumerate(cols)]
        return ax

    def _run_converging_path(p, dof, level, optimizer, iters, step, lr, eta_frac, eta_min, closure_loss):
        t_stage = time.time()
        if int(iters) <= 0:   # a zero-iteration stage is a stage the caller switched off: nothing to converge, nothing to flag
            return {'stage': f"{optimizer}@L{level}", 'level': int(level), 'path': p.name, 'iterations': 0, 'blocks': 0, 'tol_mm': None,
                    'probe_rejections': 0, 'converged': True, 'skipped': True, 'final_step_mm': 0.0}
        stage_voxel = sp_min_f * max(level, 1)
        # Coarse levels only SEED the finest one (capture range): one block with the early stop, never extended. The finest level is the
        # certificate -- extended up to max_extend_factor blocks, then reported as capped.
        max_ext = int(max_extend_factor) if (level == 1 and optimizer == 'lbfgs') else 1
        # a coarse level never early-stops (tolerance 0): on a small volume a 0.05-coarse-voxel step test fired before the seed was good
        # enough (24^3 rigid phantom: correlation 0.896 < 0.90); the finest level is judged by the step test and the manifold probe
        # Only L-BFGS stages are judged by the step test: an Adam step is learning-rate sized by construction, so a small step is no evidence of
        # convergence (2-D scattered-SyN and tiny-phantom cases ended early and lost accuracy). Adam stages run their block; they seed the finest level.
        judged = (optimizer == 'lbfgs') and level == 1
        tracker = PointStepTracker(tol=(float(convergence_tol_vox) * stage_voxel if judged else 1e-12), patience=2)
        tracker.start(_path_points(p, dof, level))
        done, blocks, rejected = 0, 0, 0
        axes = _path_axes(p, dof, level)
        if step is not None:
            # displacement of the sampled points per step in voxels -> parameter units, through the physical-shift scales:
            # axes are [t (dim), omega (1 or 3), scale (dim), shear (1 or 3)] and each group's lr uses the mean scale of its axes
            n_t, n_r = dim, (3 if dim == 3 else 1)
            sc = [a.scale_mm for a in axes]
            groups_scale = [1.0, float(np.mean(sc[n_t:n_t + n_r]))]
            if dof == 'affine':
                groups_scale += [float(np.mean(sc[n_t + n_r:n_t + n_r + dim])), float(np.mean(sc[n_t + n_r + dim:]))]
            else:
                groups_scale += [1.0, 1.0]
            lr = tuple(float(st) * sp_min_f / max(gs, 1e-12) for st, gs in zip(step, groups_scale))

        def _confirm(newly):
            """A small step is only evidence of convergence; the manifold probe measures the remaining distance (voxels)."""
            nonlocal rejected
            if not bool(newly.any()):
                return
            pr = probe_manifold(closure_loss, _path_axes(p, dof, level), stage_voxel, probe_vox=0.5)   # fresh axes: the L-BFGS stage re-binds the leaves
            if not bool(pr.remaining_vox[0] <= 3.0 * float(convergence_tol_vox)):
                tracker.reject(newly)
                rejected += 1
        if optimizer == 'lbfgs':
            # L-BFGS in SCALED variables: u_a = (displacement of the sampled points caused by parameter a) in voxels of this level
            # (core.parameter_scales). Raw (mm, rad, log-scale, shear) parameters have wildly different conditioning, which is
            # what made the quasi-Newton history crawl on a 12-parameter affine; in voxels every axis has comparable curvature.
            names = [n for n in ('t', 'omega', 'scale', 'shear') if any(getattr(p, n) is ax.tensor for ax in axes)]
            base = {n: getattr(p, n).detach().clone() for n in names}
            slices, k0 = {}, 0
            for n in names:
                slices[n] = slice(k0, k0 + base[n].numel()); k0 += base[n].numel()
            sc = torch.cat([torch.tensor([a_.scale_mm for a_ in axes if a_.tensor is getattr(p, n)], dtype=torch.float32, device=base[n].device) for n in names])
            to_param = stage_voxel / sc                                   # parameter change per voxel of displacement
            u = torch.zeros(k0, dtype=torch.float32, device=sc.device, requires_grad=True)
            check_every = 5
            opt = torch.optim.LBFGS([u], lr=1.0, max_iter=check_every, history_size=10,
                                    tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn='strong_wolfe')

            def _assign(leaf):
                for n in names:
                    v = base[n] + u[slices[n]] * to_param[slices[n]]
                    setattr(p, n, v.detach().requires_grad_(True) if leaf else v)

            def closure():
                opt.zero_grad(set_to_none=True)
                _assign(leaf=False)
                l_ = closure_loss()
                l_.backward()
                return l_
            n_blocks = max(1, math.ceil(iters / check_every))
            for call in range(n_blocks * max_ext):
                loss = opt.step(closure)
                _assign(leaf=True)
                p.clamp_()
                done += check_every
                _confirm(tracker.update(_path_points(p, dof, level)))
                last_loss[p.name] = loss.detach() if torch.is_tensor(loss) else float(loss)
                if tracker.all_converged:
                    break
            blocks = math.ceil((call + 1) / n_blocks)
        else:
            scale = 1.0
            last_check = 0
            pts_ = _path_points(p, dof, level)[0]
            pt_radius = float((pts_ - C_phys).pow(2).sum(-1).mean().sqrt())
            for cycle in range(max_ext):
                blocks = cycle + 1
                lr_c = tuple(v * scale for v in lr)
                opt = torch.optim.Adam(p.params(dof, lr_c))
                if eta_frac is not None:      # cosine floor as a fraction of each group's lr (scale-free)
                    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i, f=float(eta_frac): f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * min(i, iters) / iters)))
                elif eta_min is not None:
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=eta_min * scale)
                else:
                    sched = None
                for it in range(iters):
                    opt.zero_grad(set_to_none=True)
                    loss = closure_loss()
                    loss.backward()
                    opt.step()
                    if sched is not None:
                        sched.step()
                    p.clamp_()
                    last_loss[p.name] = loss.detach()
                    done += 1
                    if done % 10 == 0 or it == iters - 1:
                        cur = 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * (it + 1) / iters))   # cosine fraction of the cycle's lr
                        cap = (done - last_check) * cur * scale * (lr[0] + lr[1] * pt_radius)    # farthest the points could have moved (mm)
                        last_check = done
                        _confirm(tracker.update(_path_points(p, dof, level), capacity=cap))
                        if tracker.all_converged:
                            break
                if tracker.all_converged:
                    break
                scale *= 0.5
        return {'stage': f"{optimizer}@L{level}", 'level': int(level), 'path': p.name, 'iterations': int(done), 'blocks': int(blocks),
                'tol_mm': tracker.tol, 'probe_rejections': int(rejected), 'converged': bool(tracker.all_converged), 'fixed_block': (not judged), 'final_step_mm': float(tracker.last_step[0]),
                'seconds': round(time.time() - t_stage, 2)}

    def _run_schedule(paths):
        snap = {}
        for si, stage in enumerate(schedule):
            if si == len(schedule) - 1:   # the pose entering the finest stage (kept as a fallback for the self-check below)
                snap = {p_.name: {n: getattr(p_, n).detach().clone() for n in ('t', 'omega', 'scale', 'shear')} for p_ in paths}
            level, iters, dof = int(stage['level']), int(stage['iters']), stage['dof']
            use_full = bool(stage.get('full_grid', False)); samp = stage.get('sampling', None)
            for p in paths:
                convergence_log.append(_run_converging_path(
                    p, dof, level, stage.get('optimizer', 'adam'), iters, stage.get('step'), stage.get('lr'), stage.get('eta_frac'), stage.get('eta_min'),
                    lambda p=p: objective(p, dof, level, full=use_full, sampling=samp)))
            if stage.get('select', False) and len(paths) > 1:
                with torch.no_grad():
                    scores_t = torch.stack([objective(p, dof, level, full=True, sampling=1.0) for p in paths])
                    best = int(torch.argmin(scores_t).item())
                if verbose:
                    full_scores = [float(s.item()) for s in scores_t]
                    print(f"  stage {si} (level {level}): path scores {[(p.name, round(s_, 4)) for p, s_ in zip(paths, full_scores)]} -> keep '{paths[best].name}'", flush=True)
                paths = [paths[best]]
        if len(paths) > 1:   # no select stage in the schedule: pick by final-level full loss
            with torch.no_grad():
                scores_t = torch.stack([objective(p, schedule[-1]['dof'], int(schedule[-1]['level']), full=True, sampling=1.0) for p in paths])
                best = int(torch.argmin(scores_t).item())
            paths = [paths[best]]
        return paths[0], snap.get(paths[0].name)

    winner, pre_final = _run_schedule(paths)

    # 4b. self-check: the optimised pose must beat doing nothing. Baselines are the two trivial starts
    # (identity at the origin, and identity at the centre-of-mass offset), scored with the SAME objective
    # on the same full coarse grid as the candidate search. A pose no better than either means the
    # optimiser failed (or the pair is not registrable); that is an error, not a result.
    with torch.no_grad():
        coarse_dof = schedule[-1]['dof']
        final_coarse = float(objective(winner, coarse_dof, coarse, full=True, sampling=1.0).item())
        baselines = {}
        for b_name, b_t in (('Identity', np.zeros(dim)), ('Identity_CoM', t_init)):
            baselines[b_name] = float(objective(_AffinePath(np.eye(dim), b_t, dim, device_obj, b_name, spacing=sp_fix),
                                                coarse_dof, coarse, full=True, sampling=1.0).item())
    best_baseline = min(baselines.values())
    def _fails(loss):
        return not np.isfinite(loss) or loss > best_baseline + max(SELF_CHECK_TOLERANCE, SELF_CHECK_RELATIVE_TOLERANCE * abs(best_baseline))

    if _fails(final_coarse):
        # A multi-start can commit to a rotated start that never returns (a featureless pair: the identity is the best answer and the
        # data cannot pull a candidate back to it). Before declaring failure, run the whole schedule once from each trivial start.
        for b_name, b_t in sorted((('Identity', np.zeros(dim)), ('Identity_CoM', t_init)), key=lambda kv: baselines[kv[0]]):
            retry, retry_pre = _run_schedule([_AffinePath(np.eye(dim), b_t, dim, device_obj, b_name, spacing=sp_fix)])
            with torch.no_grad():
                retry_loss = float(objective(retry, coarse_dof, coarse, full=True, sampling=1.0).item())
            if np.isfinite(retry_loss) and retry_loss < final_coarse:
                winner, final_coarse = retry, retry_loss
            if not _fails(final_coarse):
                break
    if _fails(final_coarse) and pre_final is not None:
        # the finest stage converged to its own optimum, which on a weak-signal pair can score worse on the coarse grid than the pose that
        # entered it; accept the pose entering the finest stage if THAT beats doing nothing
        q = _AffinePath(winner.B.detach().cpu().numpy(), np.zeros(dim), dim, device_obj, winner.name + '_pre_finest', spacing=sp_fix)
        with torch.no_grad():
            for n, v in pre_final.items():
                getattr(q, n).copy_(v)
        with torch.no_grad():
            pre_loss = float(objective(q, coarse_dof, coarse, full=True, sampling=1.0).item())
        if np.isfinite(pre_loss) and pre_loss < final_coarse:
            winner, final_coarse = q, pre_loss
    if _fails(final_coarse):
        raise RobustAffineSolverError(
            f"robust_affine: the optimised pose (loss {final_coarse:.4f} at pyramid level {coarse}) is not better "
            f"than doing nothing (identity {baselines['Identity']:.4f}, centre-of-mass start "
            f"{baselines['Identity_CoM']:.4f}); the solver did not find a usable alignment for this pair.")

    # 5. export --------------------------------------------------------------------------
    A_fin, t_fin = winner.numpy_affine()
    tx_final = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx_final.set_parameters(np.concatenate([A_fin.flatten(), t_fin]))
    tx_final.set_fixed_parameters(com_f)
    out_dir = tempfile.mkdtemp(prefix="robust_affine_pt_")
    final_tx_path = os.path.join(out_dir, "affine.mat")
    ants.write_transform(tx_final, final_tx_path)
    from .transforms import apply_transforms as _apply_transforms
    warped_mov_out = _apply_transforms(fixed=fixed, moving=moving, transformlist=[final_tx_path], interpolator='linear')
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
        'solver': 'pytorch',
        'self_check': {'final_loss_coarse': final_coarse, 'baseline_losses_coarse': baselines, 'coarse_level': coarse},
        'convergence': convergence_log,
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
    enable_pca_candidates: bool = False,
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
    from .landmarks.orient import pca_rotation_candidates

    fi_n = normalize_image(fixed, method="auto")
    mi_n = normalize_image(moving, method="auto")

    from .device import resolve_device as _resolve_device
    dev = _resolve_device(device, task='affine')

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

    # Candidate 5: PCA-Seeded Orientation Tournament Candidates (off by default)
    if enable_pca_candidates:
        t0 = time.time()
        try:
            if 'cf' in locals() and 'cm' in locals() and len(cf) >= 10 and len(cm) >= 10:
                pca_cands = pca_rotation_candidates(cf[:, :3], cm[:, :3])
                pca_scores = score_rotation_candidates_sampled(
                    fi_n, mi_n, pca_cands,
                    sampling_percentage=0.01,
                    feature_type="mind",
                    device=dev
                )
                if pca_scores and len(pca_scores) > 0:
                    best_pca = pca_scores[0]
                    ident_score = next((r["score"] for r in pca_scores if np.allclose(r["rotation"], np.eye(3))), -1.0)
                    if best_pca["score"] > ident_score + 0.05 and not np.allclose(best_pca["rotation"], np.eye(3)):
                        R_pca = best_pca["rotation"]
                        cf_phys = np.array(compute_center_of_mass(fi_n))
                        cm_phys = np.array(compute_center_of_mass(mi_n))
                        t_pca = cm_phys - R_pca @ cf_phys
                        tx_pca = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
                        tx_pca.set_parameters(np.concatenate([R_pca.ravel(), t_pca]))
                        tx_pca.set_fixed_parameters(np.zeros(3))
                        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
                            pca_tx = f.name
                        ants.write_transform(tx_pca, pca_tx)
                        candidates["pca_orientation"] = {
                            "tx": pca_tx,
                            "time": time.time() - t0,
                            "type": "pca_orientation"
                        }
        except Exception as e:
            logger.debug("Tournament PCA orientation candidate skipped: %s", e)

    if not candidates:
        raise RuntimeError("All tournament candidates failed to produce an initial transform.")

    from .transforms import apply_transforms as _apply_transforms
    # Tournament Scoring & Gating
    scores = {}
    for cand_name, info in candidates.items():
        tx_path = info["tx"]
        warped_i = _apply_transforms(fixed=fi_n, moving=mi_n, transformlist=[tx_path], interpolator="linear")
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

    warped_mov = _apply_transforms(fixed=fixed, moving=moving, transformlist=[final_tx])
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
        'status': 'SUCCESS',
        'solver': 'pytorch',
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
    device: str = 'auto',
    seed: int = None,
    verbose: bool = False,
    enable_landmarks: bool = False,
    enable_pca_candidates: bool = False,
    lambda_shear: float = 0.02,
    lambda_scale: float = 0.01,
    cluster_threshold: float = 0.35,
    dof: str = 'affine',
    wide_angles: bool = False,
    levels: list = None,
    iterations: list = None,
    reg_iterations: list = None,
    setting: str = 'default',
    **kwargs
) -> dict:
    """
    Robust affine (or rigid) alignment of ``moving`` to ``fixed``, 2-D or 3-D.

    Returns transforms in the ANTs convention, ready for ``ants.apply_transforms`` or as the
    ``initial_transform`` of a syntx deformable registration::

        aff = syntx.robust_affine(fixed, moving)                 # default: mode='auto'
        warped = aff['warpedmovout']
        reg = syntx.tvf(fixed, moving, initial_transform=aff['fwdtransforms'])

        syntx.robust_affine(fixed, moving, dof='rigid')          # same subject (e.g. motion)
        syntx.robust_affine(fixed, moving, preset='accurate')    # slower, a little more accurate
        syntx.robust_affine(fixed, moving, levels=[4, 2, 1])     # custom multi-resolution levels

    How it works (the default ``mode='auto'``): both images are reduced to a coarse pyramid
    level; a set of starting poses -- the centre-of-mass alignment plus small single-axis
    rotations ("cone" candidates, +-6 to +-24 degrees), plus ``initial_transform`` if given -- is
    scored by Mattes mutual information; the ``n_starts`` best are optimised (rigid first,
    then full affine) through finer levels until one is kept and refined at full resolution.
    If this PyTorch solver fails, ``RobustAffineSolverError`` is raised (no silent fallback to ANTs). Results are
    deterministic for a given device (``seed`` defaults to 42). See docs/AFFINE_GUIDE.md.

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
        Target and source images (same dimension, 2-D or 3-D), in their native physical space.
    initial_transform : str, optional
        Path to an ITK/ANTs ``.mat`` transform. PyTorch solver: added as one more start
        candidate. ANTs path: a start candidate with ``multi_start``, otherwise the
        initialisation of ``ants.registration``. Ignored by ``'com_only'``.
    mode : str, default 'auto'
        Algorithm; see the module docstring. ``'auto'`` (PyTorch; a solver failure raises
        ``RobustAffineSolverError``, never falls back),
        ``'pytorch'`` (aliases ``'gpu'``, ``'pytorch_gpu'``; same as ``'auto'``), ``'ants_fast'`` /
        ``'ants'``, ``'com_only'`` / ``'translation_only'``, ``'tournament'`` (alias
        ``'auto_tournament'``; 3-D). Any other string raises ValueError. For a faster
        schedule use ``preset='fast'``.
    levels : int or list/tuple of int, optional
        Multi-resolution pyramid downsampling shrink factors (e.g. [4, 2, 1] or [8, 4, 2, 1]),
        providing consistent level control across syn, tvf, syngs, greedy, and robust_affine.
    iterations : int or list/tuple of int, optional
        Optimization iterations per level (e.g. [50, 100, 40]). ``reg_iterations`` is supported
        as an alias.
    dof : {'affine', 'rigid'}, default 'affine'
        PyTorch solver only. ``'rigid'`` keeps scale and shear at identity (translation +
        rotation) -- use it for the same subject (e.g. motion correction), where a free affine
        can drift into a spurious scale change. Ignored if a ``schedule`` kwarg is given.
    tournament : bool, default False
        Same as ``mode='tournament'``.
    multi_start : bool, default True
        Score several start candidates instead of one (both PyTorch and ANTs paths).
    n_starts : int, default 3
        PyTorch solver: how many of the best-scoring candidates are optimised in parallel.
    cone_angles_deg : list of float, optional
        PyTorch solver: the rotation angles (degrees, each tried about every axis) of the cone
        candidates. Default +-6, +-12, +-18, +-24.
    num_rotations : int, default 6
        ANTs path only: number of cone angles (from -12, -8, -4, 4, 8, 12 degrees) to try.
    low_res_spacing : float, default 4.0
        ANTs path only: voxel spacing (mm) of the images used to score the start candidates.
    device : str, default 'auto'
        'auto' (CUDA, else MPS, else CPU), 'cpu', 'cuda' or 'mps'.
    seed : int, optional
        Random seed (None = 42). Fixes the candidate generation and point sampling.
    verbose : bool, default False
        Print the candidates, their scores and the timing.
    enable_landmarks : bool, default False
        3-D: add a keypoint-based (SIFT3D) start candidate. Adds ~4 s; off by default because
        it can displace a better cone candidate on hard pairs.
    enable_pca_candidates : bool, default False
        Tournament pool only: add PCA-seeded orientation hypotheses from landmark clouds. Off
        by default because continuous moments are risky on bilateral cranial geometries and
        subject to 180° flips.
    lambda_shear, lambda_scale : float, default 0.02, 0.01
        PyTorch solver: penalties keeping shear and scale near identity.
    cluster_threshold : float, default 0.35
        PyTorch solver: minimum pose difference (radians, SE(3)) between kept start candidates,
        so the ``n_starts`` paths are not near-duplicates.
    levels : list of int, optional
        Pyramid downsampling shrink factors, coarse-to-fine (e.g. [4, 2, 1] or [8, 4, 2, 1]).
    iterations, reg_iterations : int or list of int, optional
        Optimization iterations per pyramid level. Both names are accepted interchangeably.
    **kwargs
        PyTorch solver: ``preset`` ('default', 'accurate', 'fast'), ``schedule`` (list of
        stage dicts), ``sampling_percentage``, ``num_bins``, ``mask_mode``, ``fg_dice_weight``,
        ``sample_weighting``, ``valid_margin_slices`` (default 1: drop the slab-edge slice of an anisotropic fixed image
        from the metric; 0 disables), ... (see ``_run_pytorch_affine_solver``). ANTs path: forwarded
        to ``ants.registration`` (PyTorch-only keys are dropped first).

    setting : {'default', 'intermodality_slab'}, default 'default'
        ``'intermodality_slab'``: a thin anisotropic slab (e.g. neuromelanin) to a whole-brain image of another modality, rigid only
        (``dof='rigid'``, ``mode`` auto / pytorch). Translation-only stage first, then 32 rigid starts (random rotations up to +-30
        degrees composed on that translation), the winner chosen by the Mattes MI of the warped moving image on the fixed image's NATIVE
        grid; blank warps are rejected (``syntx.slab_capture``, docs/AFFINE_GUIDE.md). About 30 times slower than the default.
        ``levels`` / ``iterations`` / ``schedule`` are not supported with it. Any other value raises ``ValueError``.

    Returns
    -------
    dict
        ``'fwdtransforms'`` : [path to the ``.mat``] mapping fixed-space points to moving space
            (use with ``ants.apply_transforms(fixed, moving, ...)``).
        ``'invtransforms'``, ``'whichtoinvert_inv'`` : the same file with ``[True]``, for the
            inverse direction.
        ``'warpedmovout'`` : ``moving`` resampled onto ``fixed``.
        ``'time'`` : wall time (s).
        ``'solver'`` : which engine produced the result: ``'pytorch'``, ``'ants'`` or ``'com_only'``.
        Also, depending on the mode: ``'warpedfixout'`` (ANTs and ``'com_only'`` paths);
        ``'runtime_seconds'``, ``'init_candidate'`` (winning start), ``'init_score'``,
        ``'final_loss'``, ``'candidates_scored'``, ``'status'`` (PyTorch solver);
        ``'winner'``, ``'candidates'`` (tournament).
    """
    require_finite_images(fixed=fixed, moving=moving)
    setting = check_setting(setting)
    for _k in ('valid_margin_slices', 'translation_scan_steps'):
        if _k in kwargs:
            _check_nonneg_int(_k, kwargs[_k])
    if mode not in ROBUST_AFFINE_MODES:
        hint = " (the faster schedule is preset='fast')" if mode == 'fast' else ""
        raise ValueError(f"robust_affine: unknown mode {mode!r}{hint}; expected one of "
                         f"{sorted(ROBUST_AFFINE_MODES)}")
    if 'backend' in kwargs:
        raise TypeError("robust_affine() has no 'backend' parameter (the solver is PyTorch; "
                        "choose the algorithm with mode=)")
    t0 = time.time()
    dim = fixed.dimension
    temp_dirs = []

    if iterations is None:
        iterations = reg_iterations
    if iterations is None and 'reg_iterations' in kwargs:
        iterations = kwargs.pop('reg_iterations')
    elif 'reg_iterations' in kwargs:
        kwargs.pop('reg_iterations')

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

    # 0a. setting='intermodality_slab': translation-first multi-start capture ranked by native-resolution MI (syntx.slab_capture)
    if setting == 'intermodality_slab':
        if tournament or mode not in ('auto', 'pytorch', 'gpu', 'pytorch_gpu'):
            raise ValueError(f"robust_affine(setting='intermodality_slab') runs on the PyTorch solver: mode must be 'auto' or 'pytorch', got {mode!r}")
        if levels is not None or iterations is not None or 'schedule' in kwargs:
            raise ValueError("robust_affine(setting='intermodality_slab') does not take levels / iterations / schedule")
        try:
            return run_slab_capture(fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose, seed=seed,
                                    dof=dof, **kwargs)
        except (RobustAffineSolverError, ValueError):
            raise
        except Exception as e:
            raise RobustAffineSolverError(f"robust_affine(setting='intermodality_slab'): failed with {type(e).__name__}: {e}. "
                                          f"There is no automatic fallback.") from e

    # 0. Mode: 'tournament' (General-purpose multi-modal / multi-candidate affine)
    if tournament or mode in ['tournament', 'auto_tournament']:
        return _run_tournament_affine(
            fixed, moving,
            initial_transform=initial_transform,
            device=device,
            verbose=verbose,
            seed=seed,
            enable_landmarks=enable_landmarks,
            enable_pca_candidates=enable_pca_candidates,
            lambda_shear=lambda_shear,
            lambda_scale=lambda_scale,
            cluster_threshold=cluster_threshold,
            dof=dof,
            levels=levels,
            iterations=iterations,
            **kwargs
        )

    # 1. Mode: 'com_only'
    if mode in ['com_only', 'translation_only']:
        com_f = compute_center_of_mass(fixed, weighted=True)
        com_m = compute_center_of_mass(moving, weighted=True)
        t_com = com_m - com_f
        tx_path, temp_dir = create_translation_transform(fixed, moving, t_com)
        from .transforms import apply_transforms as _apply_transforms
        warped_mov = _apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
        return {
            'fwdtransforms': [tx_path],
            'invtransforms': [tx_path],
            'whichtoinvert_inv': [True],
            'warpedmovout': warped_mov,
            'warpedfixout': fixed,
            'time': time.time() - t0,
            'solver': 'com_only',
        }

    # 2. Mode: 'pytorch' (also the engine behind 'auto'; failures raise RobustAffineSolverError)
    if mode in ['pytorch', 'gpu', 'pytorch_gpu', 'auto']:
        preset = kwargs.get('preset', 'default')
        if 'schedule' not in kwargs:
            if levels is not None:
                kwargs['schedule'] = _build_schedule_from_levels(levels, iterations=iterations, dof=dof, preset=preset)
            elif dof == 'rigid':
                base_schedule = _default_affine_schedule(dim, preset, dof='rigid')
                kwargs['schedule'] = [dict(s_, dof='rigid') for s_ in base_schedule]
                if iterations is not None:
                    iters_list = [iterations] * len(kwargs['schedule']) if isinstance(iterations, (int, np.integer)) else list(iterations)
                    for s_, it in zip(kwargs['schedule'], iters_list):
                        s_['iters'] = int(it)
            elif iterations is not None:
                base_schedule = _default_affine_schedule(dim, preset, dof=dof)
                iters_list = [iterations] * len(base_schedule) if isinstance(iterations, (int, np.integer)) else list(iterations)
                kwargs['schedule'] = [dict(s_, iters=int(it)) for s_, it in zip(base_schedule, iters_list)]
        elif dof not in ('rigid', 'affine'):
            raise ValueError(f"dof must be 'rigid' or 'affine', got {dof!r}.")
        try:
            return _run_pytorch_affine_solver(fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose,
                                              multi_start=multi_start, n_starts=n_starts, cone_angles_deg=cone_angles_deg,
                                              seed=seed, enable_landmarks=enable_landmarks, lambda_shear=lambda_shear,
                                              lambda_scale=lambda_scale, cluster_threshold=cluster_threshold,
                                              wide_angles=wide_angles, levels=levels, iterations=iterations, dof=dof, **kwargs)
        except Exception as e:
            # No silent fallback: a solver failure is the caller's information, whatever the mode.
            # The ANTs path is reachable only by asking for it (mode='ants' / 'ants_fast').
            raise RobustAffineSolverError(
                f"robust_affine(mode={mode!r}): the PyTorch solver failed with {type(e).__name__}: {e}. "
                f"There is no automatic fallback; use mode='ants' to run the ANTs C++ path explicitly."
            ) from e

    # 3. Mode: 'ants_fast' / 'ants' (explicit request only)
    try:
        if multi_start:
            if verbose:
                print(f"[robust_affine] Creating low-res {dim}D volumes for multi-start evaluation...", flush=True)
            # linear (central cap_ants_image, 2-D and 3-D); ants.resample_image defaults to NEAREST-NEIGHBOUR
            fi_low = cap_ants_image(fixed, low_res_spacing)
            mi_low = cap_ants_image(moving, low_res_spacing)

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
            fg_low = _scoring_foreground(fi_low)
            best_score = _eval_low_res_mi(fi_low, mi_low, best_tx_path, fg_low)
            if verbose:
                print(f"  Candidate '{best_candidate_name}': Low-Res MI = {best_score:.4f}", flush=True)

            for name, path, _ in candidates[1:]:
                score = _eval_low_res_mi(fi_low, mi_low, path, fg_low)
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
        if levels is not None and 'aff_shrink_factors' not in reg_kwargs:
            reg_kwargs['aff_shrink_factors'] = tuple(levels) if not isinstance(levels, (int, np.integer)) else (int(levels),)
        if iterations is not None and 'aff_iterations' not in reg_kwargs:
            reg_kwargs['aff_iterations'] = tuple(iterations) if not isinstance(iterations, (int, np.integer)) else (int(iterations),)
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
            'time': time.time() - t0,
            'solver': 'ants',
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


def robust_cross_modal_rigid(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    moving_mask: ants.ANTsImage = None,
    fixed_mask: ants.ANTsImage = None,
    seed: int = None,
    device: str = 'auto',
    aff_sampling: int = 64,
    aff_random_sampling_rate: float = 0.35,
    aff_iterations: tuple = (100, 50),
    aff_shrink_factors: tuple = (4, 2),
    aff_smoothing_sigmas: tuple = (2, 1),
    verbose: bool = False,
) -> dict:
    """Robust two-stage rigid registration for cross-modal image pairs (e.g. T1-to-PET,
    T1-to-CT) where intensity statistics differ enough that a single-stage local optimizer
    risks a poor local optimum without a good initializer.

    Stage 1 (coarse): ``robust_affine(..., dof='rigid')`` -- multi-start center-of-mass +
    cone rotational perturbation search at very low resolution, absorbing large
    inter-scanner table/orientation offsets. Both images are restricted to their own
    foreground first (``moving`` to parenchyma via ``moving_mask``, or an Otsu fallback;
    ``fixed`` to its largest-connected-component Otsu foreground) so a center-of-mass
    estimate isn't dragged off by variable, subject-dependent neck/shoulder/table/halo
    content -- head and neck coverage differs meaningfully between a T1 and a PET
    acquisition even for the same subject, and an unmasked fixed image lets that
    difference bias the coarse CoM/rotation search.
    Stage 2 (fine): ``ants.registration(..., type_of_transform='Rigid')`` multi-resolution
    Mattes Mutual Information gradient descent, initialized from Stage 1. This stage runs
    on the RAW (unmasked) ``fixed``/``moving`` pair, not the foreground-masked images used
    for Stage 1: Mattes MI needs real intensity texture inside and around the mask
    boundary to discriminate a correct pose from a rotated one at full resolution, and
    masking the moving image here (a bug present in an earlier revision of this function)
    starves the fine stage of exactly that texture -- Stage 1's coarse search at low
    resolution can lock onto a plausible-looking but wrong rotation (confirmed on real
    T1-to-PET data: near-tied low-resolution candidate scores, differing only at the 4th
    decimal, with the "winning" candidate sometimes being a ~24 degree wrong yaw), and
    only an unmasked, full-resolution Stage 2 has enough signal to correct it. Passing the
    masked moving image into Stage 2 removed that signal and let the wrong pose survive.

    This is the one documented, blessed use of plain ``ants.registration`` in this
    ecosystem for a fine local-refinement role that ``robust_affine`` itself doesn't cover
    (a bounded multi-resolution gradient-descent polish after a robust global
    initializer) -- callers should not reimplement this pattern themselves; call this
    function instead so any future fix/tuning benefits every caller.

    Parameters
    ----------
    fixed : ants.ANTsImage
        Fixed/target image (e.g. static PET, or any cross-modal reference).
    moving : ants.ANTsImage
        Moving image to register onto ``fixed`` (e.g. native T1).
    moving_mask : ants.ANTsImage, optional
        Binary foreground/parenchyma mask in ``moving``'s native space, used only to
        restrict Stage 1's coarse search. If None, an Otsu foreground mask is used as a
        fallback.
    fixed_mask : ants.ANTsImage, optional
        Binary foreground/parenchyma mask in ``fixed``'s native space, used only to
        restrict Stage 1's coarse search -- the real-brain-mask analogue of
        ``moving_mask``. A real brain mask (e.g. from a modality-matched brain-extraction
        model or a registration-based template fallback) is a strictly better Stage-1
        restriction than the Otsu-foreground fallback below: Otsu only separates
        signal from background, which for tracers with meaningful extracranial uptake
        (e.g. skull/scalp/vasculature activity in some PET tracers) still leaves non-brain
        mass in the coarse CoM/rotation estimate. If None, an Otsu + largest-connected-
        component foreground mask is used as a fallback (unchanged default behavior).
    seed : int, optional
        Random seed forwarded to Stage 1 for deterministic optimization.
    device : str, default='auto'
        Execution device ('auto', 'cpu', 'mps', 'cuda').
    aff_sampling : int, default=64
        Mattes MI histogram bin count for Stage 2.
    aff_random_sampling_rate : float, default=0.35
        Fraction of domain voxels sampled for Stage 2's metric.
    aff_iterations, aff_shrink_factors, aff_smoothing_sigmas : tuple
        Stage 2's multi-resolution schedule (outer-to-inner). Defaults to two levels
        (4x/2x shrink) rather than a third full-resolution level -- PET's low information
        content means a full-resolution level adds wall time without improving alignment.
    verbose : bool, default=False

    Returns
    -------
    dict
        - 'fwdtransforms': list of paths (moving -> fixed)
        - 'invtransforms': list of paths (fixed -> moving)
        - 'warpedmovout': raw moving image warped onto fixed's grid
        - 'time': total wall-clock seconds
        - 'determinant': determinant of the 3x3 rotation matrix (verified ~1.0 -- raises
          if not, since a non-rigid determinant means a genuine bug upstream, not a
          quality issue to silently tolerate)
    """
    t0 = time.time()

    if moving_mask is not None:
        mask_binary = ants.threshold_image(moving_mask, 0.5, 1.5)
        moving_fgd = moving * mask_binary
    else:
        moving_fgd = moving * ants.threshold_image(moving, "Otsu", 1)

    # Restrict the fixed image's own foreground too -- a real brain mask if the caller has
    # one (fixed_mask), else Otsu + largest connected component as a fallback -- so Stage
    # 1's coarse CoM/rotation search isn't biased by variable head/neck/shoulder coverage
    # or non-brain tracer uptake between the fixed and moving acquisitions -- see docstring.
    if fixed_mask is not None:
        fixed_fgd = fixed * ants.threshold_image(fixed_mask, 0.5, 1.5)
    else:
        fixed_fgd_mask = ants.threshold_image(fixed, "Otsu", 1).iMath("GetLargestComponent")
        fixed_fgd = fixed * fixed_fgd_mask

    if verbose:
        print("[robust_cross_modal_rigid] Stage 1: coarse robust initial alignment...")
    reg_coarse = robust_affine(
        fixed=fixed_fgd, moving=moving_fgd, mode='auto', dof='rigid', seed=seed,
        device=device, verbose=verbose,
    )
    initial_tx = reg_coarse['fwdtransforms'][0]

    if verbose:
        print("[robust_cross_modal_rigid] Stage 2: fine multi-resolution Mattes MI refinement...")
    reg_fine = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        initial_transform=initial_tx,
        aff_metric="mattes",
        aff_sampling=aff_sampling,
        aff_random_sampling_rate=aff_random_sampling_rate,
        aff_iterations=aff_iterations,
        aff_shrink_factors=aff_shrink_factors,
        aff_smoothing_sigmas=aff_smoothing_sigmas,
        verbose=verbose,
    )
    fwd = reg_fine['fwdtransforms']
    inv = reg_fine.get('invtransforms', [])

    tx_check = ants.read_transform(fwd[0])
    rot_matrix = np.array(tx_check.parameters)[:9].reshape(3, 3)
    rot_det = float(np.linalg.det(rot_matrix))
    if abs(rot_det - 1.0) > 0.01:
        raise RuntimeError(
            f"robust_cross_modal_rigid: requested rigid transform but returned transform "
            f"has determinant {rot_det:.4f}, not ~1.0 -- not a genuine rigid transform."
        )

    return {
        'fwdtransforms': fwd,
        'invtransforms': inv,
        'warpedmovout': reg_fine.get('warpedmovout', None),
        'time': time.time() - t0,
        'determinant': rot_det,
    }
