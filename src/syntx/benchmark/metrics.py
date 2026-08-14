"""
Unified Metrics Computation Module
==================================

Computes all relevant benchmark metrics (Dice, energies, Jacobian, similarity) for a 
registration result pair.
"""

import time
import logging
from typing import Dict, Any, List

import numpy as np
import ants

from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.image_compare import image_compare


def compute_pair_metrics(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fixed_label: ants.ANTsImage,
    moving_label: ants.ANTsImage,
    fwdtransforms: List[str],
    invtransforms: List[str],
    runtime_seconds: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes all benchmark metrics for a single registration pair result.

    Parameters
    ----------
    fixed : ants.ANTsImage
        The fixed target image.
    moving : ants.ANTsImage
        The moving source image.
    fixed_label : ants.ANTsImage
        The label image in fixed space.
    moving_label : ants.ANTsImage
        The label image in moving space.
    fwdtransforms : List[str]
        Paths to the forward transform files.
    invtransforms : List[str]
        Paths to the inverse transform files.
    runtime_seconds : float, optional
        The runtime in seconds to include in the output metrics. Default is 0.0.

    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics containing:
        - dice_fixed: Fixed space DKT31 cortical Dice
        - dice_moving: Moving space DKT31 cortical Dice
        - dice_sym: Symmetric mean Dice
        - folding_pct: Percentage of voxels with det(J) <= 0
        - min_jacobian: Minimum Jacobian determinant
        - harmonic_energy: L2 norm of first spatial derivatives
        - bending_energy: L2 norm of second spatial derivatives
        - mattes_mi: Mattes mutual information (lower=more similar)
        - lncc: Local normalized cross-correlation (lower=more similar)
        - runtime_seconds: Elapsed time

    Raises
    ------
    None
        This function catches exceptions and sets the corresponding metrics to NaN.
    """
    logger = logging.getLogger(__name__)

    # Default missing values
    results = {
        'dice_fixed': float('nan'),
        'dice_moving': float('nan'),
        'dice_sym': float('nan'),
        'folding_pct': 0.0,
        'min_jacobian': 1.0,
        'harmonic_energy': 0.0,
        'bending_energy': 0.0,
        'mattes_mi': float('nan'),
        'lncc': float('nan'),
        'runtime_seconds': float(runtime_seconds)
    }

    # 1. Bidirectional Dice
    try:
        df, dm, ds = compute_bidirectional_dice(
            fixed_label, moving_label, fixed, moving, fwdtransforms, invtransforms
        )
        results['dice_fixed'] = float(df)
        results['dice_moving'] = float(dm)
        results['dice_sym'] = float(ds)
    except Exception as e:
        logger.warning(f"Failed to compute Dice metrics: {e}")

    # 2. Find nonlinear warp
    nonlinear_warp_path = None
    for tx in fwdtransforms:
        if tx.endswith('.nii.gz'):
            nonlinear_warp_path = tx
            break

    # 3. Jacobian and Energy (if nonlinear warp exists)
    if nonlinear_warp_path is not None:
        try:
            warp = ants.image_read(nonlinear_warp_path)
            
            # Jacobian
            try:
                jac_ants = ants.create_jacobian_determinant_image(fixed, warp, do_log=False)
                jac_arr = jac_ants.numpy()
                
                valid_mask = ants.get_mask(fixed).numpy() > 0
                if not np.any(valid_mask):
                    valid_mask = np.ones_like(jac_arr, dtype=bool)
                    
                valid_jac = jac_arr[valid_mask]
                results['min_jacobian'] = float(np.min(valid_jac))
                results['folding_pct'] = float(np.mean(valid_jac <= 0.0) * 100.0)
            except Exception as e:
                logger.warning(f"Failed to compute Jacobian metrics: {e}")
                results['min_jacobian'] = float('nan')
                results['folding_pct'] = float('nan')
                
            # Energies
            try:
                warp_np = warp.numpy()
                spacing = warp.spacing
                dim = warp_np.shape[-1]
                
                # 1st order gradients: du_k / dx_j
                gradient_list = [np.gradient(warp_np[..., k], *spacing, axis=range(dim)) for k in range(dim)]
                total_hrm = sum(float(np.mean(g[j]**2)) for k in range(dim) for j in range(dim) for g in [gradient_list[k]])
                
                # 2nd order gradients: d^2 u_k / dx_i dx_j  
                total_bnd = 0.0
                for k in range(dim):
                    for j in range(dim):
                        grad2 = np.gradient(gradient_list[k][j], *spacing, axis=range(dim))
                        for i in range(dim):
                            total_bnd += float(np.mean(grad2[i]**2))
                            
                results['harmonic_energy'] = total_hrm
                results['bending_energy'] = total_bnd
            except Exception as e:
                logger.warning(f"Failed to compute Energy metrics: {e}")
                results['harmonic_energy'] = float('nan')
                results['bending_energy'] = float('nan')
                
        except Exception as e:
            logger.warning(f"Failed to load warp for metrics: {e}")
            results['min_jacobian'] = float('nan')
            results['folding_pct'] = float('nan')
            results['harmonic_energy'] = float('nan')
            results['bending_energy'] = float('nan')

    # 4. Image Similarity
    try:
        moving_warped = ants.apply_transforms(
            fixed=fixed, moving=moving,
            transformlist=fwdtransforms,
            interpolator='linear'
        )
        results['mattes_mi'] = float(image_compare(fixed, moving_warped, 'mattes_mi'))
        results['lncc'] = float(image_compare(fixed, moving_warped, 'lncc'))
    except Exception as e:
        logger.warning(f"Failed to compute image similarity metrics: {e}")

    return results
