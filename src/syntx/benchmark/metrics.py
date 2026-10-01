"""Standard metric set for an arbitrary registration result (``compute_pair_metrics``)."""
import numpy as np
import ants
import logging
import syntx
from typing import List, Dict, Any
from syntx.deformation_metrics import compute_bidirectional_dice

logger = logging.getLogger(__name__)

def warp_jacobian_and_energies(fixed: ants.ANTsImage, warp) -> Dict[str, float]:
    """Folding / minimum Jacobian and harmonic / bending energy of a displacement field.

    ``warp`` is a displacement-field file / ANTsImage, or None (affine only: folding 0, min 1,
    energies 0). The Jacobian is ``ants.create_jacobian_determinant_image(fixed, warp)``
    (finite differences) restricted to ``ants.get_mask(fixed)``. Energies of the displacement
    u (ANTs array order, physical spacing): harmonic = sum over k, j of mean((du_k/dx_j)^2),
    bending = sum over k, j, i of mean((d^2 u_k / dx_j dx_i)^2); means over the same mask
    when the field is on ``fixed``'s grid, else over the whole field.

    Returns
    -------
    dict
        'folding_pct' (% of masked voxels with det <= 0), 'min_jacobian', 'harmonic_energy',
        'bending_energy'.
    """
    if warp is None:
        return {"folding_pct": 0.0, "min_jacobian": 1.0, "harmonic_energy": 0.0, "bending_energy": 0.0}
    warp_img = ants.image_read(warp) if isinstance(warp, str) else warp
    dim = warp_img.dimension
    spc = warp_img.spacing
    mask = ants.get_mask(fixed).numpy() > 0
    if not mask.any():
        raise ValueError("warp_jacobian_and_energies: empty foreground mask of the fixed image")
    jac_np = ants.create_jacobian_determinant_image(fixed, warp_img, do_log=False).numpy()
    out = {"folding_pct": float(np.mean(jac_np[mask] <= 0) * 100.0),
           "min_jacobian": float(jac_np[mask].min())}

    warp_np = warp_img.numpy()
    emask = mask if tuple(warp_np.shape[:dim]) == tuple(mask.shape) else np.ones(warp_np.shape[:dim], bool)
    grads = [np.gradient(warp_np[..., k], *spc, axis=range(dim)) for k in range(dim)]
    hrm = bnd = 0.0
    for k in range(dim):
        for j in range(dim):
            g = grads[k][j]
            hrm += float(np.mean(g[emask] ** 2))
            g2 = np.gradient(g, *spc, axis=range(dim))
            for i in range(dim):
                bnd += float(np.mean(g2[i][emask] ** 2))
    out["harmonic_energy"] = hrm
    out["bending_energy"] = bnd
    return out


def compute_pair_metrics(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fixed_label: ants.ANTsImage,
    moving_label: ants.ANTsImage,
    fwdtransforms: List[str],
    invtransforms: List[str],
    reg: Dict[str, Any] = None,
    whichtoinvert_inv: List[bool] = None,
    runtime_seconds: float = None,
) -> Dict[str, Any]:
    """Dice, Jacobian, energy, similarity and inverse-error metrics for one registration.

    Each metric group is computed in its own try block; a failure is logged as a warning and
    that group's values are NaN.

    - Dice: ``compute_bidirectional_dice`` (mean Sørensen-Dice over labels, fixed space,
      moving space, and their average).
    - Jacobian and energies: ``warp_jacobian_and_energies(fixed, <first .nii.gz file in
      fwdtransforms>)`` (finite differences of the exported warp, inside
      ``ants.get_mask(fixed)``). With no such file (affine only): folding 0, min 1,
      energies 0.
    - Similarity: ``syntx.image_compare(fixed, warped moving, 'mattes_mi' / 'lncc')`` with
      the moving image warped by ``fwdtransforms`` (linear interpolation).
    - Inverse error: from ``reg['inverse_identity_error_map']``, else the 'error_map' of
      ``reg['inverse_identity_errors']`` (or of its 'phi_1' entry), else of
      ``reg['inverse_identity_error']``. Max / mean / 95th percentile over the finite values
      of the whole map (no mask; non-finite values are excluded with a warning).

    Parameters
    ----------
    fixed, moving : ANTsImage
        Fixed and moving images.
    fixed_label, moving_label : ANTsImage
        Label maps (not modified).
    fwdtransforms, invtransforms : list of str
        Transform lists as returned by a registration.
    reg : dict, optional
        The registration result; used for the runtime fallback and the inverse error.
    whichtoinvert_inv : list of bool, optional
        Passed to ``compute_bidirectional_dice`` (default there: first transform inverted).
    runtime_seconds : float, optional
        Reported as 'runtime_seconds'; else ``reg['runtime_seconds']`` or ``reg['runtime']``
        if present; otherwise the key is absent.

    Returns
    -------
    dict
        'dice_fixed', 'dice_moving', 'dice_sym', 'folding_pct' (% of masked voxels with
        det <= 0), 'min_jacobian', 'harmonic_energy', 'bending_energy', 'mattes_mi', 'lncc',
        'inverse_error_max', 'inverse_error_mean', 'inverse_error_p95' (NaN without ``reg``
        or an error map), and 'runtime_seconds' when available. All values are floats.
    """
    metrics = {}
    if runtime_seconds is not None:
        metrics["runtime_seconds"] = float(runtime_seconds)
    elif reg is not None and "runtime_seconds" in reg:
        metrics["runtime_seconds"] = float(reg["runtime_seconds"])
    elif reg is not None and "runtime" in reg:
        metrics["runtime_seconds"] = float(reg["runtime"])

    # 1. Bidirectional Dice
    try:
        dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
            fixed_label, moving_label, fixed, moving,
            fwdtransforms, invtransforms,
            whichtoinvert_inv=whichtoinvert_inv,
        )
        metrics["dice_fixed"] = float(dice_fixed)
        metrics["dice_moving"] = float(dice_moving)
        metrics["dice_sym"] = float(dice_sym)
    except Exception as e:
        logger.warning(f"Failed to compute Dice: {e}")
        metrics["dice_fixed"] = float("nan")
        metrics["dice_moving"] = float("nan")
        metrics["dice_sym"] = float("nan")

    # 2. Topological and Energy Metrics
    try:
        warp_path = next(
            (p for p in fwdtransforms if isinstance(p, str) and p.endswith(".nii.gz")),
            None,
        )
        metrics.update(warp_jacobian_and_energies(fixed, warp_path))
    except Exception as e:
        logger.warning(f"Failed to compute topological metrics: {e}")
        metrics["folding_pct"] = float("nan")
        metrics["min_jacobian"] = float("nan")
        metrics["harmonic_energy"] = float("nan")
        metrics["bending_energy"] = float("nan")

    # 3. Image Similarity Metrics
    try:
        mi_warped = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=fwdtransforms
        )
        metrics["mattes_mi"] = float(syntx.image_compare(fixed, mi_warped, "mattes_mi"))
        metrics["lncc"] = float(syntx.image_compare(fixed, mi_warped, "lncc"))
    except Exception as e:
        logger.warning(f"Failed to compute image similarity metrics: {e}")
        metrics["mattes_mi"] = float("nan")
        metrics["lncc"] = float("nan")


    # 4. Inverse Identity Error
    try:
        if reg is not None:
            inv_err_map = None
            if "inverse_identity_error_map" in reg:
                inv_err_map = reg["inverse_identity_error_map"]
            elif "inverse_identity_errors" in reg:
                inv_errs = reg["inverse_identity_errors"]
                if "phi_1" in inv_errs:
                    inv_err_map = inv_errs["phi_1"]["error_map"]
                else:
                    inv_err_map = inv_errs["error_map"]
            elif "inverse_identity_error" in reg:
                inv_errs = reg["inverse_identity_error"]
                if "phi_1" in inv_errs:
                    inv_err_map = inv_errs["phi_1"]["error_map"]
                else:
                    inv_err_map = inv_errs["error_map"]
            
            if inv_err_map is not None:
                if hasattr(inv_err_map, 'cpu'):
                    inv_err_map = inv_err_map.cpu().numpy()
                inv_np = inv_err_map.numpy() if isinstance(inv_err_map, ants.ANTsImage) else np.asarray(inv_err_map)
                
                inv_np = np.asarray(inv_np, dtype=float)
                finite = inv_np[np.isfinite(inv_np)]
                if finite.size < inv_np.size:
                    logger.warning(f"inverse error map: {inv_np.size - finite.size} non-finite values excluded")
                if finite.size:
                    metrics["inverse_error_max"] = float(np.max(finite))
                    metrics["inverse_error_mean"] = float(np.mean(finite))
                    metrics["inverse_error_p95"] = float(np.percentile(finite, 95))
                else:
                    metrics["inverse_error_max"] = metrics["inverse_error_mean"] = \
                        metrics["inverse_error_p95"] = float("nan")
            else:
                metrics["inverse_error_max"] = float("nan")
                metrics["inverse_error_mean"] = float("nan")
                metrics["inverse_error_p95"] = float("nan")
        else:
            metrics["inverse_error_max"] = float("nan")
            metrics["inverse_error_mean"] = float("nan")
            metrics["inverse_error_p95"] = float("nan")
    except Exception as e:
        logger.warning(f"Failed to compute inverse error metrics: {e}")
        metrics["inverse_error_max"] = float("nan")
        metrics["inverse_error_mean"] = float("nan")
        metrics["inverse_error_p95"] = float("nan")

    return metrics
