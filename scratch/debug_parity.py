#!/usr/bin/env python
"""
Systematic parity debug: ANTs vs syntx (PyTorch)
Step 1: Affine-only parity
Step 2: SyN/SyNTo after affine pre-warp (pass ANTs-affine-warped image to SyNTo)
Step 3: SyN/SyNTo on original images (full pipeline)

Uses 3D brain data with DKT label overlap (DICE) for evaluation.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import ants
import numpy as np
import time

def get_test_data():
    """Get 3D brain images from ANTs test data."""
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    return fi, mi

def compute_tissue_overlap(fixed_img, warped_img, n_classes=3):
    """Compute tissue overlap (Otsu segmentation) as a quick Dice proxy."""
    fixed_seg = ants.threshold_image(fixed_img, 'Otsu', n_classes)
    warped_seg = ants.threshold_image(warped_img, 'Otsu', n_classes)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    if 'MeanOverlap' in overlap.columns:
        return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return 0.0

def compute_image_similarity(fixed_img, warped_img):
    """Compute NCC between fixed and warped images."""
    fi_np = fixed_img.numpy().flatten()
    mi_np = warped_img.numpy().flatten()
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    ncc = np.mean(fi_norm * mi_norm)
    return ncc

def step1_affine_only():
    """
    Step 1: Check affine-only parity.
    Compare ANTs Affine vs syntx Affine registration.
    """
    print("=" * 70)
    print("STEP 1: AFFINE-ONLY PARITY")
    print("=" * 70)
    
    fi, mi = get_test_data()
    print(f"Fixed: {fi.shape}, spacing={fi.spacing}, origin={fi.origin}")
    print(f"Moving: {mi.shape}, spacing={mi.spacing}, origin={mi.origin}")
    
    # ANTs Affine
    print("\n--- ANTs Affine ---")
    t0 = time.time()
    reg_ants = ants.registration(fi, mi, type_of_transform='Affine')
    t_ants = time.time() - t0
    dice_ants = compute_tissue_overlap(fi, reg_ants['warpedmovout'])
    ncc_ants = compute_image_similarity(fi, reg_ants['warpedmovout'])
    print(f"  ANTs Affine: Dice={dice_ants:.4f}, NCC={ncc_ants:.4f}, time={t_ants:.1f}s")
    print(f"  ANTs fwd transforms: {reg_ants['fwdtransforms']}")
    
    # Read ANTs affine transform parameters
    ants_tx = ants.read_transform(reg_ants['fwdtransforms'][0])
    print(f"  ANTs transform params: {ants_tx.parameters}")
    print(f"  ANTs fixed params: {ants_tx.fixed_parameters}")
    
    # syntx Affine
    from syntx.syn import registration
    print("\n--- syntx Affine ---")
    t0 = time.time()
    reg_syntx = registration(
        fi, mi, type_of_transform='Affine', backend='pytorch',
        affine_iterations=[100, 50, 50, 20],
        aff_metric='mattes',
        verbose=True
    )
    t_syntx = time.time() - t0
    dice_syntx = compute_tissue_overlap(fi, reg_syntx['warpedmovout'])
    ncc_syntx = compute_image_similarity(fi, reg_syntx['warpedmovout'])
    print(f"  syntx Affine: Dice={dice_syntx:.4f}, NCC={ncc_syntx:.4f}, time={t_syntx:.1f}s")
    print(f"  syntx fwd transforms: {reg_syntx['fwdtransforms']}")
    
    # Read syntx affine transform
    if reg_syntx['fwdtransforms']:
        syntx_tx = ants.read_transform(reg_syntx['fwdtransforms'][0])
        print(f"  syntx transform params: {syntx_tx.parameters}")
        print(f"  syntx fixed params: {syntx_tx.fixed_parameters}")
    
    # Summary
    print(f"\n--- Step 1 Summary ---")
    print(f"  ANTs Affine:  Dice={dice_ants:.4f}, NCC={ncc_ants:.4f}")
    print(f"  syntx Affine: Dice={dice_syntx:.4f}, NCC={ncc_syntx:.4f}")
    print(f"  Delta Dice: {dice_syntx - dice_ants:+.4f}")
    print(f"  Delta NCC:  {ncc_syntx - ncc_ants:+.4f}")
    
    return reg_ants, reg_syntx, fi, mi


def step2_syn_on_prewarped():
    """
    Step 2: Check SyN/SyNTo parity when applied to pre-warped images.
    First do ANTs Affine, warp the moving image, then run SyN on
    (fixed, warped_moving) with both ANTs and syntx.
    """
    print("\n" + "=" * 70)
    print("STEP 2: SYN ON PRE-WARPED IMAGES (affine pre-alignment)")
    print("=" * 70)
    
    fi, mi = get_test_data()
    
    # Step 2a: ANTs Affine
    print("\n--- ANTs Affine pre-alignment ---")
    reg_aff = ants.registration(fi, mi, type_of_transform='Affine')
    mi_prewarped = reg_aff['warpedmovout']
    dice_aff = compute_tissue_overlap(fi, mi_prewarped)
    print(f"  After ANTs Affine: Dice={dice_aff:.4f}")
    
    # Step 2b: ANTs SyN on pre-warped
    print("\n--- ANTs SyN on pre-warped ---")
    t0 = time.time()
    reg_ants_syn = ants.registration(fi, mi_prewarped, type_of_transform='SyN')
    t_ants = time.time() - t0
    dice_ants_syn = compute_tissue_overlap(fi, reg_ants_syn['warpedmovout'])
    ncc_ants_syn = compute_image_similarity(fi, reg_ants_syn['warpedmovout'])
    print(f"  ANTs SyN: Dice={dice_ants_syn:.4f}, NCC={ncc_ants_syn:.4f}, time={t_ants:.1f}s")
    
    # Step 2c: syntx SyNTo on pre-warped (no affine since already aligned)
    from syntx.syn import registration
    print("\n--- syntx SyNTo on pre-warped (skip affine) ---")
    t0 = time.time()
    reg_syntx_syn = registration(
        fi, mi_prewarped, type_of_transform='SyN', backend='pytorch',
        affine_iterations=[0, 0, 0],
        reg_iterations=[100, 100, 50],
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.5, flow_sigma=3.0,
    )
    t_syntx = time.time() - t0
    dice_syntx_syn = compute_tissue_overlap(fi, reg_syntx_syn['warpedmovout'])
    ncc_syntx_syn = compute_image_similarity(fi, reg_syntx_syn['warpedmovout'])
    print(f"  syntx SyNTo: Dice={dice_syntx_syn:.4f}, NCC={ncc_syntx_syn:.4f}, time={t_syntx:.1f}s")
    
    print(f"\n--- Step 2 Summary ---")
    print(f"  After Affine pre-warp: Dice={dice_aff:.4f}")
    print(f"  ANTs SyN:             Dice={dice_ants_syn:.4f}, NCC={ncc_ants_syn:.4f}")
    print(f"  syntx SyNTo:          Dice={dice_syntx_syn:.4f}, NCC={ncc_syntx_syn:.4f}")
    print(f"  Delta Dice (syn): {dice_syntx_syn - dice_ants_syn:+.4f}")
    print(f"  Delta NCC (syn):  {ncc_syntx_syn - ncc_ants_syn:+.4f}")
    
    return reg_ants_syn, reg_syntx_syn


def step3_full_pipeline():
    """
    Step 3: Check full pipeline parity (SyN/SyNTo on original images).
    Both ANTs and syntx do their own affine + deformable.
    """
    print("\n" + "=" * 70)
    print("STEP 3: FULL PIPELINE (affine + SyN on original images)")
    print("=" * 70)
    
    fi, mi = get_test_data()
    
    # ANTs SyN (full pipeline)
    print("\n--- ANTs SyN (full) ---")
    t0 = time.time()
    reg_ants = ants.registration(fi, mi, type_of_transform='SyN')
    t_ants = time.time() - t0
    dice_ants = compute_tissue_overlap(fi, reg_ants['warpedmovout'])
    ncc_ants = compute_image_similarity(fi, reg_ants['warpedmovout'])
    print(f"  ANTs SyN: Dice={dice_ants:.4f}, NCC={ncc_ants:.4f}, time={t_ants:.1f}s")
    print(f"  ANTs fwd transforms: {reg_ants['fwdtransforms']}")
    
    # syntx SyNTo (full pipeline)
    from syntx.syn import registration
    print("\n--- syntx SyNTo (full) ---")
    t0 = time.time()
    reg_syntx = registration(
        fi, mi, type_of_transform='SyN', backend='pytorch',
        affine_iterations=[100, 50, 50, 20],
        reg_iterations=[100, 100, 50],
        aff_metric='mattes',
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.5, flow_sigma=3.0,
    )
    t_syntx = time.time() - t0
    dice_syntx = compute_tissue_overlap(fi, reg_syntx['warpedmovout'])
    ncc_syntx = compute_image_similarity(fi, reg_syntx['warpedmovout'])
    print(f"  syntx SyNTo: Dice={dice_syntx:.4f}, NCC={ncc_syntx:.4f}, time={t_syntx:.1f}s")
    print(f"  syntx fwd transforms: {reg_syntx['fwdtransforms']}")
    
    # syntx SyNTo with initial_transform from ANTs affine
    print("\n--- syntx SyNTo (with ANTs affine as initial_transform) ---")
    # First get ANTs affine
    reg_ants_aff = ants.registration(fi, mi, type_of_transform='Affine')
    import shutil, tempfile
    ants_aff_path = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    shutil.copy(reg_ants_aff['fwdtransforms'][0], ants_aff_path)
    
    t0 = time.time()
    reg_syntx_init = registration(
        fi, mi, type_of_transform='SyN', backend='pytorch',
        initial_transform=[ants_aff_path],
        affine_iterations=[0, 0, 0],
        reg_iterations=[100, 100, 50],
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.5, flow_sigma=3.0,
    )
    t_syntx_init = time.time() - t0
    dice_syntx_init = compute_tissue_overlap(fi, reg_syntx_init['warpedmovout'])
    ncc_syntx_init = compute_image_similarity(fi, reg_syntx_init['warpedmovout'])
    print(f"  syntx SyNTo (ANTs init): Dice={dice_syntx_init:.4f}, NCC={ncc_syntx_init:.4f}, time={t_syntx_init:.1f}s")
    
    # Cleanup
    os.remove(ants_aff_path)
    
    print(f"\n--- Step 3 Summary ---")
    print(f"  ANTs SyN:              Dice={dice_ants:.4f}, NCC={ncc_ants:.4f}")
    print(f"  syntx SyNTo (full):    Dice={dice_syntx:.4f}, NCC={ncc_syntx:.4f}")
    print(f"  syntx SyNTo (ANTs init): Dice={dice_syntx_init:.4f}, NCC={ncc_syntx_init:.4f}")
    print(f"  Delta Dice (full):       {dice_syntx - dice_ants:+.4f}")
    print(f"  Delta Dice (ANTs init):  {dice_syntx_init - dice_ants:+.4f}")
    
    return reg_ants, reg_syntx, reg_syntx_init


if __name__ == '__main__':
    print("Parity Debug: ANTs vs syntx (PyTorch)")
    print("Using 2D test data: r16 (fixed) vs r64 (moving)")
    print()
    
    # Step 1
    reg_ants_aff, reg_syntx_aff, fi, mi = step1_affine_only()
    
    # Step 2
    reg_ants_syn_pw, reg_syntx_syn_pw = step2_syn_on_prewarped()
    
    # Step 3
    reg_ants_full, reg_syntx_full, reg_syntx_init = step3_full_pipeline()
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
