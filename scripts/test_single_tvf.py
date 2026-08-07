import os
import sys
import time
import torch
import numpy as np
import ants
import syntx

def main():
    print("================================================================================")
    print("           SINGLE PAIR TVF VERIFICATION (antisymmetric=False)                   ")
    print("================================================================================")

    data = syntx.benchmark_data('2d')
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    device = 'cpu' if not torch.backends.mps.is_available() else 'mps'

    print("\n[1/2] Computing Robust Affine Baseline...")
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', device=device, verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]
    t_aff = time.time() - t0_aff

    # Evaluate Affine Baseline
    ml_aff = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[aff_tx], interpolator='nearestNeighbor')
    ov_aff = ants.label_overlap_measures(fl, ml_aff)
    df_aff = ov_aff[~ov_aff['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_aff = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_aff.columns else 'TargetOverlap'
    dice_aff = float(df_aff[col_aff].mean())
    print(f"  -> Robust Affine Baseline Dice: {dice_aff:.4f} (computed in {t_aff:.2f}s)")

    print("\n[2/2] Running TVF (antisymmetric=False, constant_speed=True)...")
    t0_tvf = time.time()
    reg = syntx.tvf(
        fixed=fi, moving=mi, initial_transform=aff_tx,
        backend='pytorch', device=device,
        reg_iterations=[100, 100, 20],
        similarity_metric='lncc', syn_sampling=2,
        multipoint_loss=[0.0, 0.5, 1.0],
        optimizer='cfl', cfl_max=0.0, cfl_momentum=0.95,
        n_time_steps=3, constant_speed=True, constant_speed_relaxation=0.10,
        use_analytical_gradients=True,
        antisymmetric=False,  # RULE 14: Never force zero midpoint velocity
        flow_sigma=1.5,
        total_sigma=0.05,
        grad_step=0.90,
        regularizer='gaussian',
        fast_smooth=True,
        verbose=True
    )
    t_tvf = time.time() - t0_tvf

    # Fixed Space Dice
    ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    ov_fwd = ants.label_overlap_measures(fl, ml_warped)
    df_fwd = ov_fwd[~ov_fwd['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fwd = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fwd.columns else 'TargetOverlap'
    dice_fixed = float(df_fwd[col_fwd].mean())

    # Moving Space Dice
    fl_warped = ants.apply_transforms(fixed=mi, moving=fl, transformlist=reg['invtransforms'], whichtoinvert=reg.get('whichtoinvert_inv', [True, False]), interpolator='nearestNeighbor')
    ov_inv = ants.label_overlap_measures(ml, fl_warped)
    df_inv = ov_inv[~ov_inv['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_inv = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_inv.columns else 'TargetOverlap'
    dice_moving = float(df_inv[col_inv].mean())

    dice_sym = 0.5 * (dice_fixed + dice_moving)

    # Jacobian & Folding
    jac = ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0])
    jac_np = jac.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0
    fold_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100)
    min_detJ = float(jac_np[mask_eval].min())

    # Displacement Field Stats
    warp_img = ants.image_read(reg['fwdtransforms'][0])
    warp_np = warp_img.numpy()
    disp_max = float(warp_np.max())
    disp_min = float(warp_np.min())

    print("\n================================================================================")
    print("                              VERIFICATION RESULTS                              ")
    print("================================================================================")
    print(f"Robust Affine Baseline Dice : {dice_aff:.4f}")
    print(f"TVF Symmetric Mean Dice     : {dice_sym:.4f} (Fixed: {dice_fixed:.4f} / Moving: {dice_moving:.4f})")
    print(f"Dice Gain Over Baseline     : +{dice_sym - dice_aff:.4f}")
    print(f"Grid Folding Percentage     : {fold_pct:.4f}%")
    print(f"Minimum Jacobian det(J)    : {min_detJ:.4f}")
    print(f"Displacement Bounds (mm)    : [{disp_min:.3f}, {disp_max:.3f}]")
    print(f"Execution Runtime           : {t_tvf:.2f}s")
    print("================================================================================")

if __name__ == "__main__":
    main()
