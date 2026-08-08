import os
import sys
import time
import numpy as np
import ants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx

def main():
    print("=" * 80)
    print("  STAGE 1: ROBUST AFFINE BASELINE ALIGNMENT ON MBHARD")
    print("=" * 80)

    data = syntx.benchmark_data('mbhard')
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    print(f"Fixed shape: {fi.shape}, spacing: {fi.spacing}")
    print(f"Moving shape: {mi.shape}, spacing: {mi.spacing}\n")

    t0 = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=True)
    t_aff = time.time() - t0

    aff_tx = reg_aff['fwdtransforms'][0]
    aff_inv_tx = reg_aff['invtransforms'][0]

    # Evaluate Fixed Space Dice
    warped_ml_aff = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[aff_tx], interpolator='nearestNeighbor')
    ov_aff_f = ants.label_overlap_measures(fl, warped_ml_aff)
    df_aff_f = ov_aff_f[~ov_aff_f['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_af = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_aff_f.columns else 'TargetOverlap'
    dice_aff_fixed = float(df_aff_f[col_af].mean())

    # Evaluate Moving Space Dice
    warped_fl_aff = ants.apply_transforms(fixed=mi, moving=fl, transformlist=[aff_inv_tx], interpolator='nearestNeighbor')
    ov_aff_m = ants.label_overlap_measures(ml, warped_fl_aff)
    df_aff_m = ov_aff_m[~ov_aff_m['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_am = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_aff_m.columns else 'TargetOverlap'
    dice_aff_moving = float(df_aff_m[col_am].mean())

    dice_aff_sym = 0.5 * (dice_aff_fixed + dice_aff_moving)

    print("=" * 80)
    print("  STAGE 1 AFFINE RESULTS")
    print("=" * 80)
    print(f"Runtime                     : {t_aff:.2f} seconds")
    print(f"Fixed Space Cortical Dice   : {dice_aff_fixed:.6f}")
    print(f"Moving Space Cortical Dice  : {dice_aff_moving:.6f}")
    print(f"Symmetric Mean Cortical Dice: {dice_aff_sym:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
