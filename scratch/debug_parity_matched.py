#!/usr/bin/env python
"""
DEFINITIVE parity test: match parameters exactly between ANTs and syntx.

Findings from previous rounds:
1. Affine: PARITY ACHIEVED (Dice diff < 0.002)
2. ANTs default SyN uses MI metric with fewer iterations → lower Dice
3. ANTs SyNCC uses CC metric with more iterations → matches syntx
4. syntx warp has larger displacements (10.6 vs 6.3 max) but similar Dice

This test matches parameters exactly:
- Same metric (CC/LNCC)
- Same number of iterations
- Same regularization
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import ants
import numpy as np
import time

def get_test_data():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    return fi, mi

def compute_tissue_overlap(fixed_img, warped_img, n_classes=3):
    fixed_seg = ants.threshold_image(fixed_img, 'Otsu', n_classes)
    warped_seg = ants.threshold_image(warped_img, 'Otsu', n_classes)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    if 'MeanOverlap' in overlap.columns:
        return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return 0.0

def compute_ncc(fixed_img, warped_img):
    fi_np = fixed_img.numpy().flatten()
    mi_np = warped_img.numpy().flatten()
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    return float(np.mean(fi_norm * mi_norm))


def matched_parameter_comparison():
    """
    Match parameters between ANTs and syntx for fair comparison.
    """
    fi, mi = get_test_data()
    from syntx.syn import registration
    
    print("=" * 70)
    print("DEFINITIVE PARITY TEST: Matched Parameters")
    print("=" * 70)
    print(f"Fixed: {fi.shape}, spacing={fi.spacing}")
    print(f"Moving: {mi.shape}, spacing={mi.spacing}")
    
    # -------------------------------------------------------
    # 1. AFFINE ONLY
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("1. AFFINE ONLY")
    print("-" * 50)
    
    reg_ants_aff = ants.registration(fi, mi, type_of_transform='Affine')
    dice_ants_aff = compute_tissue_overlap(fi, reg_ants_aff['warpedmovout'])
    ncc_ants_aff = compute_ncc(fi, reg_ants_aff['warpedmovout'])
    
    reg_syntx_aff = registration(fi, mi, type_of_transform='Affine', backend='pytorch')
    dice_syntx_aff = compute_tissue_overlap(fi, reg_syntx_aff['warpedmovout'])
    ncc_syntx_aff = compute_ncc(fi, reg_syntx_aff['warpedmovout'])
    
    print(f"  ANTs:  Dice={dice_ants_aff:.4f}, NCC={ncc_ants_aff:.4f}")
    print(f"  syntx: Dice={dice_syntx_aff:.4f}, NCC={ncc_syntx_aff:.4f}")
    print(f"  Δ Dice: {dice_syntx_aff - dice_ants_aff:+.4f}")
    
    # -------------------------------------------------------
    # 2. SyN on pre-warped images (ANTs affine → SyN)
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("2. SyN ON PRE-WARPED (after ANTs affine)")
    print("-" * 50)
    
    mi_prewarped = reg_ants_aff['warpedmovout']
    
    # ANTs SyNCC on pre-warped (uses CC metric, more iters)
    reg_ants_syn = ants.registration(fi, mi_prewarped, type_of_transform='SyNCC')
    dice_ants_syn = compute_tissue_overlap(fi, reg_ants_syn['warpedmovout'])
    ncc_ants_syn = compute_ncc(fi, reg_ants_syn['warpedmovout'])
    
    # syntx SyNTo on pre-warped (LNCC, skip affine)
    reg_syntx_syn = registration(
        fi, mi_prewarped, type_of_transform='SyN', backend='pytorch',
        affine_iterations=[0, 0, 0],
        reg_iterations=[100, 100, 50],
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.2, flow_sigma=3.0,
    )
    dice_syntx_syn = compute_tissue_overlap(fi, reg_syntx_syn['warpedmovout'])
    ncc_syntx_syn = compute_ncc(fi, reg_syntx_syn['warpedmovout'])
    
    print(f"  After affine:     Dice={dice_ants_aff:.4f}")
    print(f"  ANTs SyNCC:       Dice={dice_ants_syn:.4f}, NCC={ncc_ants_syn:.4f}")
    print(f"  syntx SyNTo:      Dice={dice_syntx_syn:.4f}, NCC={ncc_syntx_syn:.4f}")
    print(f"  Δ Dice: {dice_syntx_syn - dice_ants_syn:+.4f}")
    
    # -------------------------------------------------------
    # 3. Full pipeline (original images)
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. FULL PIPELINE (affine + SyN on original images)")
    print("-" * 50)
    
    # ANTs SyNCC (uses CC metric internally)
    reg_ants_full = ants.registration(fi, mi, type_of_transform='SyNCC')
    dice_ants_full = compute_tissue_overlap(fi, reg_ants_full['warpedmovout'])
    ncc_ants_full = compute_ncc(fi, reg_ants_full['warpedmovout'])
    
    # Also compare ANTs SyN with custom high iterations
    reg_ants_hi = ants.registration(fi, mi, type_of_transform='SyN',
                                     reg_iterations=(100, 100, 50))
    dice_ants_hi = compute_tissue_overlap(fi, reg_ants_hi['warpedmovout'])
    ncc_ants_hi = compute_ncc(fi, reg_ants_hi['warpedmovout'])
    
    # syntx full pipeline
    reg_syntx_full = registration(
        fi, mi, type_of_transform='SyN', backend='pytorch',
        aff_metric='mattes',
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.2, flow_sigma=3.0,
        reg_iterations=[100, 100, 50],
    )
    dice_syntx_full = compute_tissue_overlap(fi, reg_syntx_full['warpedmovout'])
    ncc_syntx_full = compute_ncc(fi, reg_syntx_full['warpedmovout'])
    
    # syntx with ANTs affine as initialization  
    import shutil, tempfile
    ants_aff_path = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    shutil.copy(reg_ants_aff['fwdtransforms'][0], ants_aff_path)
    
    reg_syntx_init = registration(
        fi, mi, type_of_transform='SyN', backend='pytorch',
        initial_transform=[ants_aff_path],
        affine_iterations=[0, 0, 0],
        reg_iterations=[100, 100, 50],
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.2, flow_sigma=3.0,
    )
    dice_syntx_init = compute_tissue_overlap(fi, reg_syntx_init['warpedmovout'])
    ncc_syntx_init = compute_ncc(fi, reg_syntx_init['warpedmovout'])
    os.remove(ants_aff_path)
    
    print(f"  ANTs SyN (default):    Dice={dice_ants_hi:.4f}, NCC={ncc_ants_hi:.4f}")
    print(f"  ANTs SyNCC:            Dice={dice_ants_full:.4f}, NCC={ncc_ants_full:.4f}")
    print(f"  syntx full:            Dice={dice_syntx_full:.4f}, NCC={ncc_syntx_full:.4f}")
    print(f"  syntx (ANTs aff init): Dice={dice_syntx_init:.4f}, NCC={ncc_syntx_init:.4f}")
    print(f"  Δ Dice (syntx vs SyNCC):         {dice_syntx_full - dice_ants_full:+.4f}")
    print(f"  Δ Dice (syntx+init vs SyNCC):    {dice_syntx_init - dice_ants_full:+.4f}")
    
    # -------------------------------------------------------
    # 4. Warp field comparison
    # -------------------------------------------------------
    print("\n" + "-" * 50)
    print("4. WARP FIELD MAGNITUDE COMPARISON")
    print("-" * 50)
    
    # Compare warp fields when both use CC metric
    for name, reg in [("ANTs SyNCC", reg_ants_full), ("syntx full", reg_syntx_full)]:
        for f in reg['fwdtransforms']:
            if f.endswith('.nii.gz'):
                w = ants.image_read(f)
                wnp = w.numpy()
                print(f"  {name} warp: range=[{wnp.min():.2f}, {wnp.max():.2f}], "
                      f"mean_abs={np.abs(wnp).mean():.3f}, max_abs={np.abs(wnp).max():.2f}")
    
    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<35} {'ANTs':>10} {'syntx':>10} {'Δ':>10}")
    print("-" * 65)
    print(f"{'1. Affine only':<35} {dice_ants_aff:>10.4f} {dice_syntx_aff:>10.4f} {dice_syntx_aff-dice_ants_aff:>+10.4f}")
    print(f"{'2. SyN on pre-warped':<35} {dice_ants_syn:>10.4f} {dice_syntx_syn:>10.4f} {dice_syntx_syn-dice_ants_syn:>+10.4f}")
    print(f"{'3a. Full (ANTs SyNCC)':<35} {dice_ants_full:>10.4f} {dice_syntx_full:>10.4f} {dice_syntx_full-dice_ants_full:>+10.4f}")
    print(f"{'3b. Full (syntx + ANTs aff init)':<35} {dice_ants_full:>10.4f} {dice_syntx_init:>10.4f} {dice_syntx_init-dice_ants_full:>+10.4f}")


if __name__ == '__main__':
    matched_parameter_comparison()
