import numpy as np
import ants
import logging
import syntx
from typing import List, Dict, Any
from syntx.deformation_metrics import compute_bidirectional_dice

logger = logging.getLogger(__name__)

def compute_pair_metrics(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fixed_label: ants.ANTsImage,
    moving_label: ants.ANTsImage,
    fwdtransforms: List[str],
    invtransforms: List[str],
    reg: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Computes all standard benchmarking metrics for a registration result.
    
    Returns
    -------
    dict
        Dictionary containing standardized metric keys:
        dice_fixed, dice_moving, dice_sym, folding_pct, min_jacobian,
        harmonic_energy, bending_energy, mattes_mi, lncc.
    """
    metrics = {}

    # 1. Bidirectional Dice
    try:
        dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
            fixed_label, moving_label, fixed, moving,
            fwdtransforms, invtransforms,
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
        if warp_path is not None:
            warp_img = ants.image_read(warp_path)
            dim = warp_img.dimension
            spc = warp_img.spacing

            # Jacobian
            jac_img = ants.create_jacobian_determinant_image(fixed, warp_img, do_log=False)
            jac_np = jac_img.numpy()
            mask = ants.get_mask(fixed).numpy() > 0

            metrics["folding_pct"] = float(np.mean(jac_np[mask] <= 0) * 100.0) if mask.any() else 0.0
            metrics["min_jacobian"] = float(jac_np[mask].min()) if mask.any() else 1.0

            # Harmonic and Bending Energy
            warp_np = warp_img.numpy()
            gradient_list = [
                np.gradient(warp_np[..., k], *spc, axis=range(dim))
                for k in range(dim)
            ]

            total_hrm = 0.0
            total_bnd = 0.0
            for k in range(dim):
                for j in range(dim):
                    grad_kj = gradient_list[k][j]
                    total_hrm += float(np.mean(grad_kj ** 2))

                    grad2_kj = np.gradient(grad_kj, *spc, axis=range(dim))
                    for i in range(dim):
                        total_bnd += float(np.mean(grad2_kj[i] ** 2))

            metrics["harmonic_energy"] = total_hrm
            metrics["bending_energy"] = total_bnd
        else:
            metrics["folding_pct"] = 0.0
            metrics["min_jacobian"] = 1.0
            metrics["harmonic_energy"] = 0.0
            metrics["bending_energy"] = 0.0
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
                
                # Check for NaNs
                inv_np = np.nan_to_num(inv_np)
                
                metrics["inverse_error_max"] = float(np.max(inv_np))
                metrics["inverse_error_mean"] = float(np.mean(inv_np))
                metrics["inverse_error_p95"] = float(np.percentile(inv_np, 95))
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
