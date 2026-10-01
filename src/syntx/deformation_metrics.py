"""
syntx.deformation_metrics — Centralized Topological and Energy Evaluation Framework
=====================================================================================

Provides a unified API for evaluating the topological and energetic properties of
deformation fields, complementing `syntx.image_compare`.

Metrics:
- ``compute_bidirectional_dice``: label overlap after warping in both directions.
- ``compute_harmonic_energy`` / ``compute_bending_energy``: first / second-derivative
  roughness of a displacement field.
- ``compute_jacobian_metrics``: finite-difference Jacobian determinant of an exported warp.
- ``flow_jacobian_metrics``: the same statistics from ``syntx.liouville_determinant`` (the
  per-method determinant used as the folding measure).
"""

import numpy as np
import torch
import ants
import pandas as pd


def compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv=None, metric_column=None):
    """Label Dice after warping in both directions: moving labels into fixed space, and
    fixed labels into moving space.

    Dice is the Sørensen-Dice coefficient 2|A n B| / (|A| + |B|) per label (ITK / ANTs
    ``MeanOverlap``; never target overlap), averaged over the labels present (label 0 and
    the 'All' row excluded; values outside [0, 1] dropped). Labels are warped with
    nearest-neighbour interpolation.

    Parameters
    ----------
    fl, ml : ANTsImage
        Fixed and moving label maps (on the grids of ``fi`` / ``mi``; copies get those
        images' origin / spacing / direction, the inputs are not modified).
    fi, mi : ANTsImage
        Fixed and moving images (define the target grids).
    fwdtransforms, invtransforms : list
        As returned by a registration (``reg['fwdtransforms']``, ``reg['invtransforms']``).
    whichtoinvert_inv : list of bool, optional
        For ``invtransforms``; default [True, False, ...] (first one inverted).
    metric_column : str, optional
        Column of ``ants.label_overlap_measures`` to use instead of the Dice column.

    Returns
    -------
    (dice_fixed, dice_moving, dice_sym) : tuple of float
        Mean Dice in fixed space, in moving space, and their average.
    """
    if whichtoinvert_inv is None:
        whichtoinvert_inv = [True] + [False] * (len(invtransforms) - 1) if len(invtransforms) > 0 else []
    # copies carrying the geometry of their images, for both directions (inputs untouched)
    fl, ml = fl.clone(), ml.clone()
    for lab, img in ((fl, fi), (ml, mi)):
        lab.set_origin(img.origin)
        lab.set_spacing(img.spacing)
        lab.set_direction(img.direction)

    def _get_dice_column(df):
        if metric_column is not None and metric_column in df.columns:
            return metric_column
        # In ITK / ANTsPy, MeanOverlap is the true Sørensen-Dice coefficient: 2|A ∩ B| / (|A| + |B|)
        for col_name in ['MeanOverlap', 'Dice', 'DiceCoefficient', 'SorensenDice']:
            if col_name in df.columns:
                return col_name
        raise KeyError(
            f"Sørensen-Dice column ('MeanOverlap') not found in label overlap DataFrame. "
            f"Available columns: {list(df.columns)}. Do not report non-Dice metrics as Dice."
        )

    # 1. Fixed Space Dice
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    fl.set_origin(fi.origin)
    fl.set_spacing(fi.spacing)
    fl.set_direction(fi.direction)
    ml_warped.set_origin(fi.origin)
    ml_warped.set_spacing(fi.spacing)
    ml_warped.set_direction(fi.direction)
    ov_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = ov_fixed[~ov_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fixed = _get_dice_column(df_fixed)
    vals_fixed = pd.to_numeric(df_fixed[col_fixed], errors='coerce').to_numpy(dtype=np.float64)
    vals_fixed = vals_fixed[np.isfinite(vals_fixed) & (vals_fixed >= 0.0) & (vals_fixed <= 1.0)]
    dice_fixed = float(np.mean(vals_fixed)) if len(vals_fixed) > 0 else 0.0

    # 2. Moving Space Dice
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl,
        transformlist=invtransforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator='nearestNeighbor'
    )
    ml.set_origin(mi.origin)
    ml.set_spacing(mi.spacing)
    ml.set_direction(mi.direction)
    fl_warped.set_origin(mi.origin)
    fl_warped.set_spacing(mi.spacing)
    fl_warped.set_direction(mi.direction)
    ov_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = ov_moving[~ov_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_moving = _get_dice_column(df_moving)
    vals_moving = pd.to_numeric(df_moving[col_moving], errors='coerce').to_numpy(dtype=np.float64)
    vals_moving = vals_moving[np.isfinite(vals_moving) & (vals_moving >= 0.0) & (vals_moving <= 1.0)]
    dice_moving = float(np.mean(vals_moving)) if len(vals_moving) > 0 else 0.0

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym


def _to_numpy_warp(warp) -> np.ndarray:
    """Safely extracts numpy array from ANTsImage, PyTorch tensor, or numpy array, squeezing singleton batch dimensions."""
    if isinstance(warp, str):
        try:
            warp = ants.image_read(warp)
        except Exception:
            raise ValueError(f"Could not load deformation field from path: {warp}")
            
    if hasattr(warp, 'numpy'):
        arr = warp.numpy()
    elif isinstance(warp, torch.Tensor):
        arr = warp.detach().cpu().numpy()
    else:
        arr = np.asarray(warp)

    dim = arr.shape[-1]
    while arr.ndim > dim + 1:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            break
    return arr


def _resolve_warp_and_spacing(warp, spacing=None):
    """(array (..., dim), dim, per-axis spacing in array order): spacing from ``spacing``, else
    from an ANTsImage, else 1; reversed (ITK -> tensor order) for torch tensors."""
    is_tensor = isinstance(warp, torch.Tensor)
    if spacing is None and hasattr(warp, 'spacing'):
        spacing = warp.spacing

    warp_np = _to_numpy_warp(warp)
    dim = warp_np.shape[-1]

    if spacing is None:
        sp_axes = (1.0,) * dim
    else:
        spacing_tuple = tuple(float(s) for s in spacing)
        if is_tensor:
            sp_axes = tuple(reversed(spacing_tuple))
        else:
            sp_axes = spacing_tuple

    return warp_np, dim, sp_axes


def compute_harmonic_energy(warp, spacing=None) -> float:
    """
    Harmonic (membrane) energy of a displacement field u: the sum over components k and
    axes j of mean_x (du_k / dx_j)^2, i.e. the mean squared Frobenius norm of the
    displacement gradient (central differences, physical spacing).

    Parameters
    ----------
    warp : ants.ANTsImage, torch.Tensor, np.ndarray, or str
        Displacement field array of shape [..., dim].
    spacing : tuple of floats, optional
        Physical voxel spacing. If `warp` is an ANTsImage, spacing is auto-extracted.
        Default is 1.0 for all dimensions if not provided.

    Returns
    -------
    float
        Harmonic energy (unitless with physical spacing; 0 for a pure translation).
    """
    warp_np, dim, sp_axes = _resolve_warp_and_spacing(warp, spacing)

    gradient_list = [np.gradient(warp_np[..., k], *sp_axes, axis=tuple(range(dim))) for k in range(dim)]
    
    total_hrm = 0.0
    for k in range(dim):
        for j in range(dim):
            grad_kj = gradient_list[k][j]
            total_hrm += float(np.mean(grad_kj**2))
            
    return total_hrm


def compute_bending_energy(warp, spacing=None) -> float:
    """
    Thin-plate bending energy of a displacement field u: the sum over components k and axes
    i, j of mean_x (d^2 u_k / dx_i dx_j)^2, i.e. the mean squared Frobenius norm of the
    Hessian (repeated central differences, physical spacing).

    Parameters
    ----------
    warp : ants.ANTsImage, torch.Tensor, np.ndarray, or str
        Displacement field array of shape [..., dim].
    spacing : tuple of floats, optional
        Physical voxel spacing. If `warp` is an ANTsImage, spacing is auto-extracted.
        Default is 1.0 for all dimensions if not provided.

    Returns
    -------
    float
        Bending energy (1 / mm^2 with physical spacing; 0 for an affine field).
    """
    warp_np, dim, sp_axes = _resolve_warp_and_spacing(warp, spacing)

    gradient_list = [np.gradient(warp_np[..., k], *sp_axes, axis=tuple(range(dim))) for k in range(dim)]
    
    total_bnd = 0.0
    for k in range(dim):
        for j in range(dim):
            grad_kj = gradient_list[k][j]
            grad2_kj = np.gradient(grad_kj, *sp_axes, axis=tuple(range(dim)))
            for i in range(dim):
                total_bnd += float(np.mean(grad2_kj[i]**2))
                
    return total_bnd


def flow_jacobian_metrics(fixed_image, registration_result):
    """Folding statistics from ``syntx.liouville_determinant`` (the per-method determinant of the
    deformable map: flow product for tvf / syngs, half fields for syn, full field for greedy),
    restricted to the fixed image's foreground, with ``'measure'``; None if it cannot be computed.
    """
    from .liouville import liouville_determinant, determinant_summary
    try:
        img, det = liouville_determinant(registration_result, fixed_image, return_details=True)
    except (ValueError, NotImplementedError, AttributeError, KeyError, StopIteration):
        return None
    out = determinant_summary(img, fixed_image)
    out["measure"] = det["measure"]
    return out


def compute_jacobian_metrics(fixed_image, warp) -> dict:
    """
    Finite-difference Jacobian-determinant statistics of an exported displacement field
    (``ants.create_jacobian_determinant_image``), inside the fixed image's foreground mask
    (``ants.get_mask``; the whole image if the mask is empty).

    For folding, prefer ``flow_jacobian_metrics`` / ``syntx.liouville_determinant``: the
    finite difference reports det <= 0 where a valid map varies sharply within a voxel.

    Parameters
    ----------
    fixed_image : ants.ANTsImage
        The fixed target image used for geometry and foreground masking.
    warp : ants.ANTsImage or str
        The displacement field.

    Returns
    -------
    dict
        ``'min'``, ``'max'``, ``'mean'``, ``'folding_pct'`` (% of masked voxels with det <= 0).
    """
    if isinstance(warp, str):
        warp = ants.image_read(warp)
        
    jac_ants = ants.create_jacobian_determinant_image(fixed_image, warp, do_log=False)
    jac_arr = jac_ants.numpy()
    
    # Restrict to foreground mask
    valid_mask = ants.get_mask(fixed_image).numpy() > 0
    if not np.any(valid_mask):
        valid_mask = np.ones_like(jac_arr, dtype=bool)
        
    valid_jac = jac_arr[valid_mask]
    
    return {
        "min": float(np.min(valid_jac)),
        "max": float(np.max(valid_jac)),
        "mean": float(np.mean(valid_jac)),
        "folding_pct": float(np.mean(valid_jac <= 0.0) * 100.0)
    }
