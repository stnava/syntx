import os
import sys
import time
import numpy as np
import torch
import ants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from syntx.spatial import jacobian_determinant

def main():
    print("=" * 80)
    print("  STAGE 2: MID-LEVEL DEFORMABLE ALIGNMENT (syntx.syn [100,10,0]) ON MBHARD")
    print("=" * 80)

    data = syntx.benchmark_data('mbhard')
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    print("[1/2] Computing Stage 1 Robust Affine Initializer...", flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    t_aff = time.time() - t0_aff
    aff_tx = reg_aff['fwdtransforms'][0]
    aff_inv_tx = reg_aff['invtransforms'][0]

    print("[2/2] Running Stage 2 syntx.syn with reg_iterations=[100, 10, 0]...", flush=True)
    t0_syn = time.time()
    reg_syn = syntx.syn(
        fixed=fi,
        moving=mi,
        initial_transform=aff_tx,
        reg_iterations=[100, 10, 0],
        similarity_metric='lncc',
        regularizer='dsti',
        flow_sigma=3.0,
        total_sigma=0.0,
        grad_step=0.25,
        in_loop_inv_steps=10,
        backend='pytorch',
        verbose=True
    )
    t_syn = time.time() - t0_syn

    syn_fwd = reg_syn['fwdtransforms']
    syn_inv = reg_syn['invtransforms']

    # Evaluate Fixed Space Dice
    warped_ml_syn = ants.apply_transforms(fixed=fi, moving=ml, transformlist=syn_fwd, interpolator='nearestNeighbor')
    ov_syn_f = ants.label_overlap_measures(fl, warped_ml_syn)
    df_syn_f = ov_syn_f[~ov_syn_f['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_sf = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_syn_f.columns else 'TargetOverlap'
    dice_syn_fixed = float(df_syn_f[col_sf].mean())

    # Evaluate Moving Space Dice
    warped_fl_syn = ants.apply_transforms(
        fixed=mi, moving=fl,
        transformlist=syn_inv,
        whichtoinvert=[True, False] if len(syn_inv) > 1 else [False] * len(syn_inv),
        interpolator='nearestNeighbor'
    )
    ov_syn_m = ants.label_overlap_measures(ml, warped_fl_syn)
    df_syn_m = ov_syn_m[~ov_syn_m['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_sm = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_syn_m.columns else 'TargetOverlap'
    dice_syn_moving = float(df_syn_m[col_sm].mean())

    dice_syn_sym = 0.5 * (dice_syn_fixed + dice_syn_moving)

    # Evaluate Jacobian for SyN stage
    syn_warp_img = ants.image_read(syn_fwd[0])
    syn_jac = jacobian_determinant(syn_warp_img, ref_image=fi)
    mask = ants.get_mask(fi).numpy() > 0
    syn_jac_vals = syn_jac[mask]
    syn_min_detJ = float(syn_jac_vals.min())
    syn_max_detJ = float(syn_jac_vals.max())
    syn_folding_pct = float(np.mean(syn_jac_vals <= 0.0) * 100.0)

    print("=" * 80)
    print("  STAGE 2 SYN RESULTS")
    print("=" * 80)
    print(f"SyN Execution Time          : {t_syn:.2f} seconds (Total incl affine: {t_aff + t_syn:.2f}s)")
    print(f"Fixed Space Cortical Dice   : {dice_syn_fixed:.6f}")
    print(f"Moving Space Cortical Dice  : {dice_syn_moving:.6f}")
    print(f"Symmetric Mean Cortical Dice: {dice_syn_sym:.6f}")
    print(f"Jacobian det(J) Range       : [{syn_min_detJ:+.6f}, {syn_max_detJ:.6f}]")
    print(f"Grid Folding Rate           : {syn_folding_pct:.4f}% (Fold-Free: {syn_min_detJ > 0.0})")
    print("=" * 80)

if __name__ == "__main__":
    main()
