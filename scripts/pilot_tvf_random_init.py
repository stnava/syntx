import os
import sys
import time
import torch
import numpy as np
import ants
import syntx
from syntx.tvf import TVFModel, separable_gaussian_filter

def main():
    print("================================================================================")
    print("             PILOT TEST: DETERMINISTIC RANDOM INIT + LARS FOR TVF                ")
    print("================================================================================")

    data = syntx.benchmark_data('2d')
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    device = 'cpu' if not torch.backends.mps.is_available() else 'mps'

    print("\n[1/3] Computing robust affine baseline...")
    t0 = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', device=device, verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]

    # Evaluate Affine Baseline Dice
    ml_aff = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[aff_tx], interpolator='nearestNeighbor')
    ov_aff = ants.label_overlap_measures(fl, ml_aff)
    df_aff = ov_aff[~ov_aff['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_aff = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_aff.columns else 'TargetOverlap'
    dice_aff = float(df_aff[col_aff].mean())
    print(f"  -> Robust Affine Baseline Dice: {dice_aff:.4f} (computed in {time.time()-t0:.2f}s)")

    print("\n[2/3] Running TVF with Deterministic Random Field Initialization...")
    # Parameters for pilot run
    flow_sigma = 1.5
    total_sigma = 0.05
    grad_step = 0.90
    trust_coeff = 0.80

    t0_tvf = time.time()

    # We will subclass / patch TVFModel's fit method or run tvf directly with deterministic seeding hook
    torch.manual_seed(42)

    reg = syntx.tvf(
        fixed=fi, moving=mi, initial_transform=aff_tx,
        backend='pytorch', device=device,
        reg_iterations=[100, 100, 20],
        similarity_metric='lncc', syn_sampling=2,
        multipoint_loss=[0.0, 0.5, 1.0],
        optimizer='lars', cfl_max=0.0, cfl_momentum=0.95,
        n_time_steps=3, constant_speed=True, constant_speed_relaxation=0.10,
        use_analytical_gradients=True, antisymmetric=True,
        flow_sigma=flow_sigma,
        total_sigma=total_sigma,
        grad_step=grad_step,
        regularizer='gaussian',
        fast_smooth=True,
        trust_coefficient=trust_coeff,
        convergence_threshold=0.0,  # Disable premature early stopping
        verbose=True
    )
    t_tvf = time.time() - t0_tvf

    print("\n[3/3] Evaluating TVF Pilot Results...")

    # Fixed Space Dice
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml, 
        transformlist=reg['fwdtransforms'], 
        interpolator='nearestNeighbor'
    )
    overlap_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = overlap_fixed[~overlap_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fixed = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fixed.columns else 'TargetOverlap'
    dice_fixed = float(df_fixed[col_fixed].mean())

    # Moving Space Dice
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl, 
        transformlist=reg['invtransforms'], 
        whichtoinvert=reg.get('whichtoinvert_inv', [True, False]), 
        interpolator='nearestNeighbor'
    )
    overlap_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = overlap_moving[~overlap_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_moving = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_moving.columns else 'TargetOverlap'
    dice_moving = float(df_moving[col_moving].mean())

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

    print("================================================================================")
    print("                             PILOT RESULTS SUMMARY                              ")
    print("================================================================================")
    print(f"Robust Affine Baseline Dice : {dice_aff:.4f}")
    print(f"TVF Pilot Symmetric Dice    : {dice_sym:.4f} (Fixed: {dice_fixed:.4f} / Moving: {dice_moving:.4f})")
    print(f"Dice Gain Over Baseline     : +{dice_sym - dice_aff:.4f}")
    print(f"Grid Folding Percentage     : {fold_pct:.4f}%")
    print(f"Minimum Jacobian det(J)    : {min_detJ:.4f}")
    print(f"Max / Min Displacement (mm) : [{disp_min:.3f}, {disp_max:.3f}]")
    print(f"TVF Execution Runtime       : {t_tvf:.2f}s")
    print("================================================================================")

if __name__ == "__main__":
    main()
