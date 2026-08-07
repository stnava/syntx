"""
Peak SyN Benchmark Evaluation and Provenance Exporter for r16_r64.

Runs the exact peak parameter set for syntx.syn:
- regularizer: dsti
- sobolev_alpha: 3.0
- flow_sigma: 2.0
- total_sigma: 0.0
- grad_step: 0.50
- reg_iterations: [100, 100, 20]
- affine_iterations: [100, 50, 20]
- inverse_method: anderson

Evaluates bidirectional fixed, moving, and symmetric Dice across Class 2, Class 3, and Class 2+3,
along with Jacobian regularity, inverse identity error (mm), and runtime.
"""

import time
import json
import numpy as np
import torch
import ants
import syntx
from syntx.syn import calculate_inverse_identity_error


def eval_bidirectional_tissue_dice(fl_binary, ml_binary, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv):
    # Fixed space: warp moving label -> fixed space
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml_binary,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fl_binary, ml_warped)
    dice_fixed = float(ov_fixed['TotalOrTargetOverlap'].iloc[1])

    # Moving space: warp fixed label -> moving space
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl_binary,
        transformlist=invtransforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator='nearestNeighbor'
    )
    ov_moving = ants.label_overlap_measures(ml_binary, fl_warped)
    dice_moving = float(ov_moving['TotalOrTargetOverlap'].iloc[1])

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym


def main():
    print("==================================================", flush=True)
    print(" EVALUATING PEAK SyN PARAMETERS ON r16_r64", flush=True)
    print("==================================================", flush=True)

    data = syntx.benchmark_data('r16_r64')
    fi, mi = data['fixed'], data['moving']
    
    fl_c2 = data['fixed_labels']['class2']
    ml_c2 = data['moving_labels']['class2']
    
    fl_c23 = data['fixed_labels']['class2_3']
    ml_c23 = data['moving_labels']['class2_3']

    # Class 3 (White Matter binary mask)
    fl_otsu = data['fixed_labels']['otsu']
    ml_otsu = data['moving_labels']['otsu']
    fl_c3 = ants.threshold_image(fl_otsu, 3, 3)
    ml_c3 = ants.threshold_image(ml_otsu, 3, 3)

    # 1. Compute robust affine initialization
    print("Computing robust affine initialization...", flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    aff_time = time.time() - t0_aff
    aff_tx = reg_aff['fwdtransforms'][0]

    # 2. Run peak SyN registration
    print("Running SyN (dsti, flow_sigma=2.0, grad_step=0.5, reg_iterations=[100, 100, 20])...", flush=True)
    t0_syn = time.time()
    reg = syntx.syn(
        fixed=fi, moving=mi, initial_transform=aff_tx,
        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
        regularizer='dsti', sobolev_alpha=3.0, flow_sigma=2.0, total_sigma=0.0,
        grad_step=0.50, reg_iterations=[100, 100, 20], affine_iterations=[100, 50, 20],
        inverse_method='anderson', in_loop_inv_steps=10, fast_smooth=True, antisymmetric=True, verbose=False
    )
    syn_time = time.time() - t0_syn
    total_time = aff_time + syn_time

    # 3. Evaluate Bidirectional Dice across tissue classes
    fwd_tx = reg['fwdtransforms']
    inv_tx = reg['invtransforms']
    which_inv = reg.get('whichtoinvert_inv', [True, False])

    c2_fixed, c2_moving, c2_sym = eval_bidirectional_tissue_dice(fl_c2, ml_c2, fi, mi, fwd_tx, inv_tx, which_inv)
    c3_fixed, c3_moving, c3_sym = eval_bidirectional_tissue_dice(fl_c3, ml_c3, fi, mi, fwd_tx, inv_tx, which_inv)
    c23_fixed, c23_moving, c23_sym = eval_bidirectional_tissue_dice(fl_c23, ml_c23, fi, mi, fwd_tx, inv_tx, which_inv)

    # 4. Evaluate Jacobian regularity & folding
    jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx[0], do_log=False)
    jac_np = jac_ants.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0

    folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)
    min_detJ = float(jac_np[mask_eval].min())
    max_detJ = float(jac_np[mask_eval].max())
    mean_detJ = float(jac_np[mask_eval].mean())

    # 5. Evaluate Inverse Identity Error (mm)
    model = reg['model']
    w_l2r = model.warp_l2r.data.cpu()
    w_l2r_inv = model.warp_l2r_inv.data.cpu()
    err_dict = calculate_inverse_identity_error(w_l2r, w_l2r_inv, fi.spacing, fi.origin, fi.direction)
    
    err_map = err_dict['error_map'].numpy()
    mean_inv_err = float(err_dict['mean_error'])
    max_inv_err = float(err_dict['max_error'])
    p95_inv_err = float(np.percentile(err_map[mask_eval], 95))

    results = {
        "otsu_label2_class2_sym_dice": c2_sym,
        "otsu_label2_class2_fixed_dice": c2_fixed,
        "otsu_label2_class2_moving_dice": c2_moving,
        "otsu_label3_class3_sym_dice": c3_sym,
        "otsu_label3_class3_fixed_dice": c3_fixed,
        "otsu_label3_class3_moving_dice": c3_moving,
        "otsu_label23_parenchyma_sym_dice": c23_sym,
        "otsu_label23_parenchyma_fixed_dice": c23_fixed,
        "otsu_label23_parenchyma_moving_dice": c23_moving,
        "runtime_seconds": total_time,
        "folding_percentage": folding_pct,
        "min_jacobian_determinant": min_detJ,
        "max_jacobian_determinant": max_detJ,
        "mean_jacobian_determinant": mean_detJ,
        "mean_inverse_error_mm": mean_inv_err,
        "p95_inverse_error_mm": p95_inv_err,
        "max_inverse_error_mm": max_inv_err,
    }

    print("\n==================================================", flush=True)
    print(" PEAK SyN RESULTS SUMMARY:", flush=True)
    print("==================================================", flush=True)
    print(f" Cortical Gray Matter (Class 2) Dice : {c2_sym:.6f} (Fixed: {c2_fixed:.4f}, Moving: {c2_moving:.4f})")
    print(f" White Matter (Class 3) Dice         : {c3_sym:.6f} (Fixed: {c3_fixed:.4f}, Moving: {c3_moving:.4f})")
    print(f" Parenchyma (Class 2+3) Dice         : {c23_sym:.6f} (Fixed: {c23_fixed:.4f}, Moving: {c23_moving:.4f})")
    print(f" Grid Folding Percentage             : {folding_pct:.6f}%")
    print(f" Minimum Jacobian Determinant        : {min_detJ:.6f}")
    print(f" Mean Inverse Identity Error (mm)    : {mean_inv_err:.6f} mm")
    print(f" 95th %ile Inverse Error (mm)        : {p95_inv_err:.6f} mm")
    print(f" Total Execution Time                : {total_time:.2f} s")
    print("==================================================", flush=True)

    # Save to scratch file for JSON update
    with open('/tmp/peak_syn_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
