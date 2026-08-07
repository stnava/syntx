"""
Side-by-Side Comparison of fast_smooth=True vs fast_smooth=False on Peak SyN Parameters.
"""

import time
import numpy as np
import torch
import ants
import syntx
from syntx.syn import calculate_inverse_identity_error


def eval_bidirectional_tissue_dice(fl_binary, ml_binary, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv):
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml_binary,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fl_binary, ml_warped)
    dice_fixed = float(ov_fixed['TotalOrTargetOverlap'].iloc[1])

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


def run_eval(fast_smooth_flag, fi, mi, fl_c2, ml_c2, fl_c3, ml_c3, fl_c23, ml_c23, aff_tx):
    t0 = time.time()
    reg = syntx.syn(
        fixed=fi, moving=mi, initial_transform=aff_tx,
        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
        regularizer='dsti', sobolev_alpha=3.0, flow_sigma=2.0, total_sigma=0.0,
        grad_step=0.50, reg_iterations=[100, 100, 20], affine_iterations=[100, 50, 20],
        inverse_method='anderson', in_loop_inv_steps=10, fast_smooth=fast_smooth_flag,
        antisymmetric=True, verbose=False
    )
    elapsed = time.time() - t0

    fwd_tx = reg['fwdtransforms']
    inv_tx = reg['invtransforms']
    which_inv = reg.get('whichtoinvert_inv', [True, False])

    c2_fixed, c2_moving, c2_sym = eval_bidirectional_tissue_dice(fl_c2, ml_c2, fi, mi, fwd_tx, inv_tx, which_inv)
    c3_fixed, c3_moving, c3_sym = eval_bidirectional_tissue_dice(fl_c3, ml_c3, fi, mi, fwd_tx, inv_tx, which_inv)
    c23_fixed, c23_moving, c23_sym = eval_bidirectional_tissue_dice(fl_c23, ml_c23, fi, mi, fwd_tx, inv_tx, which_inv)

    jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx[0], do_log=False)
    jac_np = jac_ants.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0

    folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)
    min_detJ = float(jac_np[mask_eval].min())

    model = reg['model']
    w_l2r = model.warp_l2r.data.cpu()
    w_l2r_inv = model.warp_l2r_inv.data.cpu()
    err_dict = calculate_inverse_identity_error(w_l2r, w_l2r_inv, fi.spacing, fi.origin, fi.direction)
    mean_inv_err = float(err_dict['mean_error'])

    return {
        'fast_smooth': fast_smooth_flag,
        'c2_fixed': c2_fixed,
        'c2_moving': c2_moving,
        'c2_sym': c2_sym,
        'c3_fixed': c3_fixed,
        'c3_moving': c3_moving,
        'c3_sym': c3_sym,
        'c23_fixed': c23_fixed,
        'c23_moving': c23_moving,
        'c23_sym': c23_sym,
        'folding_pct': folding_pct,
        'min_detJ': min_detJ,
        'mean_inv_err': mean_inv_err,
        'runtime_seconds': elapsed
    }


def main():
    data = syntx.benchmark_data('r16_r64')
    fi, mi = data['fixed'], data['moving']
    
    fl_c2, ml_c2 = data['fixed_labels']['class2'], data['moving_labels']['class2']
    fl_c23, ml_c23 = data['fixed_labels']['class2_3'], data['moving_labels']['class2_3']

    fl_otsu, ml_otsu = data['fixed_labels']['otsu'], data['moving_labels']['otsu']
    fl_c3 = ants.threshold_image(fl_otsu, 3, 3)
    ml_c3 = ants.threshold_image(ml_otsu, 3, 3)

    print("Computing robust affine initialization...", flush=True)
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]

    print("\nRunning fast_smooth=True (FFT DSTI Green's Operator)...", flush=True)
    res_true = run_eval(True, fi, mi, fl_c2, ml_c2, fl_c3, ml_c3, fl_c23, ml_c23, aff_tx)

    print("Running fast_smooth=False (Spatial Separable Gaussian Filter)...", flush=True)
    res_false = run_eval(False, fi, mi, fl_c2, ml_c2, fl_c3, ml_c3, fl_c23, ml_c23, aff_tx)

    print("\n==========================================================================================", flush=True)
    print(" DIRECT SIDE-BY-SIDE COMPARISON: fast_smooth=True VS fast_smooth=False", flush=True)
    print("==========================================================================================", flush=True)
    print(f"{'Metric':<38} | {'fast_smooth=True (FFT DSTI)':<28} | {'fast_smooth=False (Spatial)':<28}")
    print("-" * 100, flush=True)
    print(f"{'Class 2 Fixed Space Dice':<38} | {res_true['c2_fixed']:<28.6f} | {res_false['c2_fixed']:<28.6f}")
    print(f"{'Class 2 Moving Space Dice':<38} | {res_true['c2_moving']:<28.6f} | {res_false['c2_moving']:<28.6f}")
    print(f"{'Class 2 Symmetric Mean Dice':<38} | {res_true['c2_sym']:<28.6f} | {res_false['c2_sym']:<28.6f}")
    print("-" * 100, flush=True)
    print(f"{'Class 3 White Matter Dice':<38} | {res_true['c3_sym']:<28.6f} | {res_false['c3_sym']:<28.6f}")
    print(f"{'Class 2+3 Parenchyma Dice':<38} | {res_true['c23_sym']:<28.6f} | {res_false['c23_sym']:<28.6f}")
    print(f"{'Grid Folding % (detJ <= 0)':<38} | {res_true['folding_pct']:<28.6f}% | {res_false['folding_pct']:<28.6f}%")
    print(f"{'Minimum Jacobian Det (min_detJ)':<38} | {res_true['min_detJ']:<28.6f} | {res_false['min_detJ']:<28.6f}")
    print(f"{'Mean Inverse Identity Error (mm)':<38} | {res_true['mean_inv_err']:<28.6f} mm | {res_false['mean_inv_err']:<28.6f} mm")
    print(f"{'Execution Runtime (s)':<38} | {res_true['runtime_seconds']:<28.2f} s | {res_false['runtime_seconds']:<28.2f} s")
    print("==========================================================================================", flush=True)


if __name__ == '__main__':
    main()
