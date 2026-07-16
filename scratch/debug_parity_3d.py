#!/usr/bin/env python
"""
3D Parity Test: ANTs vs syntx on real brain data.

Uses MNI template as fixed and CH2 template as moving.
These are different atlases with different shapes — a realistic 3D registration task.

Tests:
1. Affine only  
2. SyN full pipeline
3. Metrics: tissue overlap (Otsu Dice), NCC, warp magnitude
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import ants
import numpy as np
import time

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

def warp_stats(reg):
    """Get displacement field statistics."""
    for f in reg.get('fwdtransforms', []):
        if f.endswith('.nii.gz') and os.path.exists(f):
            w = ants.image_read(f)
            wnp = w.numpy()
            return {
                'max_disp': float(np.abs(wnp).max()),
                'mean_disp': float(np.abs(wnp).mean()),
                'range': f"[{wnp.min():.1f}, {wnp.max():.1f}]"
            }
    return None

def cleanup(reg):
    for path in reg.get('fwdtransforms', []) + reg.get('invtransforms', []):
        if os.path.exists(path):
            try: os.remove(path)
            except: pass


def main():
    print("=" * 70)
    print("3D PARITY TEST: ANTs vs syntx")
    print("=" * 70)
    
    # Load 3D brain data
    fi = ants.image_read(ants.get_ants_data('mni'))
    mi = ants.image_read(ants.get_ants_data('ch2'))
    print(f"Fixed (MNI): shape={fi.shape}, spacing={fi.spacing}, origin={fi.origin}")
    print(f"Moving (CH2): shape={mi.shape}, spacing={mi.spacing}, origin={mi.origin}")
    
    # Downsample to 2mm for faster testing (keeps it manageable on CPU)
    fi_ds = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
    mi_ds = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)
    print(f"\nDownsampled Fixed: shape={fi_ds.shape}, spacing={fi_ds.spacing}")
    print(f"Downsampled Moving: shape={mi_ds.shape}, spacing={mi_ds.spacing}")
    
    # Baseline: no registration
    baseline_ncc = compute_ncc(fi_ds, ants.resample_image_to_target(mi_ds, fi_ds))
    print(f"\nBaseline NCC (no registration): {baseline_ncc:.4f}")
    
    from syntx.syn import registration
    
    # -------------------------------------------------------
    # 1. AFFINE ONLY
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: AFFINE ONLY (3D)")
    print("=" * 70)
    
    t0 = time.time()
    reg_ants_aff = ants.registration(fi_ds, mi_ds, type_of_transform='Affine')
    t_ants_aff = time.time() - t0
    dice_ants_aff = compute_tissue_overlap(fi_ds, reg_ants_aff['warpedmovout'])
    ncc_ants_aff = compute_ncc(fi_ds, reg_ants_aff['warpedmovout'])
    print(f"  ANTs Affine:  Dice={dice_ants_aff:.4f}, NCC={ncc_ants_aff:.4f}, time={t_ants_aff:.1f}s")
    
    t0 = time.time()
    reg_syntx_aff = registration(fi_ds, mi_ds, type_of_transform='Affine', backend='pytorch')
    t_syntx_aff = time.time() - t0
    dice_syntx_aff = compute_tissue_overlap(fi_ds, reg_syntx_aff['warpedmovout'])
    ncc_syntx_aff = compute_ncc(fi_ds, reg_syntx_aff['warpedmovout'])
    print(f"  syntx Affine: Dice={dice_syntx_aff:.4f}, NCC={ncc_syntx_aff:.4f}, time={t_syntx_aff:.1f}s")
    print(f"  Δ Dice: {dice_syntx_aff - dice_ants_aff:+.4f}, Δ NCC: {ncc_syntx_aff - ncc_ants_aff:+.4f}")
    
    # -------------------------------------------------------
    # 2. FULL SYN PIPELINE
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: FULL SYN PIPELINE (3D)")
    print("=" * 70)
    
    # ANTs SyN (default — uses MI)
    t0 = time.time()
    reg_ants_syn = ants.registration(fi_ds, mi_ds, type_of_transform='SyN')
    t_ants_syn = time.time() - t0
    dice_ants_syn = compute_tissue_overlap(fi_ds, reg_ants_syn['warpedmovout'])
    ncc_ants_syn = compute_ncc(fi_ds, reg_ants_syn['warpedmovout'])
    ws_ants_syn = warp_stats(reg_ants_syn)
    print(f"  ANTs SyN:     Dice={dice_ants_syn:.4f}, NCC={ncc_ants_syn:.4f}, time={t_ants_syn:.1f}s")
    if ws_ants_syn:
        print(f"    warp: max={ws_ants_syn['max_disp']:.2f}, mean={ws_ants_syn['mean_disp']:.3f}, range={ws_ants_syn['range']}")
    
    # ANTs SyNCC (uses CC metric)
    t0 = time.time()
    reg_ants_cc = ants.registration(fi_ds, mi_ds, type_of_transform='SyNCC')
    t_ants_cc = time.time() - t0
    dice_ants_cc = compute_tissue_overlap(fi_ds, reg_ants_cc['warpedmovout'])
    ncc_ants_cc = compute_ncc(fi_ds, reg_ants_cc['warpedmovout'])
    ws_ants_cc = warp_stats(reg_ants_cc)
    print(f"  ANTs SyNCC:   Dice={dice_ants_cc:.4f}, NCC={ncc_ants_cc:.4f}, time={t_ants_cc:.1f}s")
    if ws_ants_cc:
        print(f"    warp: max={ws_ants_cc['max_disp']:.2f}, mean={ws_ants_cc['mean_disp']:.3f}, range={ws_ants_cc['range']}")
    
    # syntx SyNTo (new defaults: syn_sampling=4)
    t0 = time.time()
    reg_syntx_syn = registration(
        fi_ds, mi_ds, type_of_transform='SyN', backend='pytorch',
        aff_metric='mattes', syn_metric='lncc',
    )
    t_syntx_syn = time.time() - t0
    dice_syntx_syn = compute_tissue_overlap(fi_ds, reg_syntx_syn['warpedmovout'])
    ncc_syntx_syn = compute_ncc(fi_ds, reg_syntx_syn['warpedmovout'])
    ws_syntx = warp_stats(reg_syntx_syn)
    n_aff = len(reg_syntx_syn.get('affine_losses', []))
    n_syn = len(reg_syntx_syn.get('syn_losses', []))
    print(f"  syntx SyNTo:  Dice={dice_syntx_syn:.4f}, NCC={ncc_syntx_syn:.4f}, time={t_syntx_syn:.1f}s (aff={n_aff}, syn={n_syn})")
    if ws_syntx:
        print(f"    warp: max={ws_syntx['max_disp']:.2f}, mean={ws_syntx['mean_disp']:.3f}, range={ws_syntx['range']}")
    
    # -------------------------------------------------------
    # 3. PARAMETER VARIATIONS (if gap exists)
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: SYNTX PARAMETER VARIATIONS (3D)")
    print("=" * 70)
    
    configs = [
        {"name": "gs=0.2 sigma=3 r=4 (default)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "syn_sampling": 4}},
        {"name": "gs=0.5 sigma=3 r=4",
         "kwargs": {"grad_step": 0.5, "flow_sigma": 3.0, "syn_sampling": 4}},
        {"name": "gs=0.75 sigma=3 r=4",
         "kwargs": {"grad_step": 0.75, "flow_sigma": 3.0, "syn_sampling": 4}},
        {"name": "gs=0.2 sigma=3 r=2 (old default)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "syn_sampling": 2}},
        {"name": "gs=0.75 sigma=3 r=2",
         "kwargs": {"grad_step": 0.75, "flow_sigma": 3.0, "syn_sampling": 2}},
    ]
    
    print(f"\n{'Config':<40} {'Dice':>8} {'NCC':>8} {'Time':>8} {'n_syn':>6}")
    print("-" * 75)
    
    for cfg in configs:
        try:
            t0 = time.time()
            reg = registration(
                fi_ds, mi_ds, type_of_transform='SyN', backend='pytorch',
                aff_metric='mattes', syn_metric='lncc',
                **cfg["kwargs"]
            )
            t1 = time.time() - t0
            dice = compute_tissue_overlap(fi_ds, reg['warpedmovout'])
            ncc = compute_ncc(fi_ds, reg['warpedmovout'])
            n_syn = len(reg.get('syn_losses', []))
            print(f"{cfg['name']:<40} {dice:>8.4f} {ncc:>8.4f} {t1:>7.1f}s {n_syn:>6}")
            cleanup(reg)
        except Exception as e:
            print(f"{cfg['name']:<40} ERROR: {e}")
            import traceback; traceback.print_exc()
    
    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (3D Brain Registration)")
    print("=" * 70)
    print(f"{'Test':<35} {'ANTs':>10} {'syntx':>10} {'Δ':>10}")
    print("-" * 65)
    print(f"{'Affine (Dice)':<35} {dice_ants_aff:>10.4f} {dice_syntx_aff:>10.4f} {dice_syntx_aff-dice_ants_aff:>+10.4f}")
    print(f"{'Affine (NCC)':<35} {ncc_ants_aff:>10.4f} {ncc_syntx_aff:>10.4f} {ncc_syntx_aff-ncc_ants_aff:>+10.4f}")
    print(f"{'SyN vs SyNCC (Dice)':<35} {dice_ants_cc:>10.4f} {dice_syntx_syn:>10.4f} {dice_syntx_syn-dice_ants_cc:>+10.4f}")
    print(f"{'SyN vs SyNCC (NCC)':<35} {ncc_ants_cc:>10.4f} {ncc_syntx_syn:>10.4f} {ncc_syntx_syn-ncc_ants_cc:>+10.4f}")
    
    # Cleanup
    cleanup(reg_ants_aff)
    cleanup(reg_syntx_aff)
    cleanup(reg_ants_syn)
    cleanup(reg_ants_cc)
    cleanup(reg_syntx_syn)


if __name__ == '__main__':
    main()
