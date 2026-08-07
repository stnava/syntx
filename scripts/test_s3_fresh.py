"""
Fresh Run Verification of S3 fastFalse across gaussian, sobolev, and dsti regularizers.
"""

import time
import numpy as np
import torch
import ants
import syntx


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

    s3_params = {'flow_sigma': 3.0, 'grad_step': 0.50}

    print("\n==========================================================================================", flush=True)
    print(" FRESH RUN VERIFICATION: S3 (flow_sigma=3.0, grad_step=0.50, fast_smooth=False)", flush=True)
    print("==========================================================================================", flush=True)
    print(f"{'Regularizer':<12} | {'Class 2 GM':<10} | {'Class 3 WM':<10} | {'Class 2+3 Par':<12} | {'Otsu Sym Dice':<14} | {'Folding %':<10} | {'Runtime':<8}")
    print("-" * 95, flush=True)

    for reg_name in ['gaussian', 'sobolev', 'dsti']:
        t0 = time.time()
        reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
            reg_iterations=[100, 40], affine_iterations=[50, 20],
            similarity_metric='lncc', syn_sampling=2, inverse_method='anderson',
            total_sigma=0.0, regularizer=reg_name, fast_smooth=False,
            antisymmetric=True, verbose=False, **s3_params
        )
        elapsed = time.time() - t0

        fwd_tx = reg['fwdtransforms']
        inv_tx = reg['invtransforms']
        which_inv = reg.get('whichtoinvert_inv', [True, False])

        c2_f, c2_m, c2_s = eval_bidirectional_tissue_dice(fl_c2, ml_c2, fi, mi, fwd_tx, inv_tx, which_inv)
        c3_f, c3_m, c3_s = eval_bidirectional_tissue_dice(fl_c3, ml_c3, fi, mi, fwd_tx, inv_tx, which_inv)
        c23_f, c23_m, c23_s = eval_bidirectional_tissue_dice(fl_c23, ml_c23, fi, mi, fwd_tx, inv_tx, which_inv)

        dice_otsu_f, dice_otsu_m, dice_otsu_s = eval_bidirectional_tissue_dice(data['fixed_label'], data['moving_label'], fi, mi, fwd_tx, inv_tx, which_inv)

        jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx[0], do_log=False)
        jac_np = jac_ants.numpy()
        mask_eval = ants.get_mask(fi).numpy() > 0
        folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)

        print(f"{reg_name:<12} | {c2_s:<10.4f} | {c3_s:<10.4f} | {c23_s:<12.4f} | {dice_otsu_s:<14.4f} | {folding_pct:<10.4f}% | {elapsed:<8.2f}s", flush=True)

    print("==========================================================================================", flush=True)


if __name__ == '__main__':
    main()
