#!/usr/bin/env python
"""
Targeted SyN deformable debug: find the ~1.2% Dice gap.

Known: Affine is at parity. SyN deformable gap is ~0.012 Dice.
Test: Run syntx SyN on pre-warped images with various settings to narrow the gap.

Hypotheses:
H1: syntx convergence check is too aggressive (early stopping)
H2: syntx LNCC window size differs from ANTs CC radius  
H3: syntx grad_step or CFL scaling differs
H4: syntx midpoint composition introduces error
H5: syntx fluid sigma (3.0 voxels) differs from ANTs default
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


def main():
    fi, mi = get_test_data()
    from syntx.syn import registration
    
    # Pre-warp with ANTs Affine
    reg_aff = ants.registration(fi, mi, type_of_transform='Affine')
    mi_pw = reg_aff['warpedmovout']
    dice_aff = compute_tissue_overlap(fi, mi_pw)
    
    print("=" * 70)
    print("SYN DEFORMABLE GAP ANALYSIS")
    print(f"Starting from ANTs affine: Dice={dice_aff:.4f}")
    print("=" * 70)
    
    # Reference: ANTs SyNCC  
    reg_ref = ants.registration(fi, mi_pw, type_of_transform='SyNCC')
    dice_ref = compute_tissue_overlap(fi, reg_ref['warpedmovout'])
    ncc_ref = compute_ncc(fi, reg_ref['warpedmovout'])
    print(f"\nReference: ANTs SyNCC = Dice={dice_ref:.4f}, NCC={ncc_ref:.4f}")
    
    configs = [
        # Baseline: current default
        {"name": "default (gs=0.2, sigma=3, iters=100x3)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [100, 100, 50],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        
        # H1: More iterations (disable convergence check by running more)
        {"name": "H1a: more iters (200x3)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        {"name": "H1b: even more iters (300x3)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [300, 300, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        
        # H2: Different LNCC window sizes (ANTs CC uses radius=4 → window=9 by default)
        {"name": "H2a: lncc radius=1 (window=3)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 1}},
        {"name": "H2b: lncc radius=4 (window=9)",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 4}},
        
        # H3: Different grad_step
        {"name": "H3a: grad_step=0.1",
         "kwargs": {"grad_step": 0.1, "flow_sigma": 3.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        {"name": "H3b: grad_step=0.5",
         "kwargs": {"grad_step": 0.5, "flow_sigma": 3.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        
        # H5: Different fluid sigma
        {"name": "H5a: sigma=1.5",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 1.5, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        {"name": "H5b: sigma=2.0",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 2.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        {"name": "H5c: sigma=6.0",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 6.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 2}},
        
        # Combined: match ANTs CC defaults more closely
        {"name": "matched: gs=0.2, sigma=3, radius=4, 200 iters",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [200, 200, 100],
                    "syn_metric": "lncc", "syn_sampling": 4}},
        
        # Try different levels
        {"name": "4 levels [8,4,2,1]",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [100, 100, 100, 50],
                    "syn_metric": "lncc", "syn_sampling": 2, "levels": [8, 4, 2, 1]}},
    ]
    
    print(f"\n{'Config':<50} {'Dice':>8} {'NCC':>8} {'Time':>6} {'Δ Dice':>8} {'n_syn':>6}")
    print("-" * 90)
    
    for cfg in configs:
        try:
            t0 = time.time()
            reg = registration(
                fi, mi_pw, type_of_transform='SyN', backend='pytorch',
                affine_iterations=[0, 0, 0],
                aff_metric='mattes',
                **cfg["kwargs"]
            )
            t1 = time.time() - t0
            dice = compute_tissue_overlap(fi, reg['warpedmovout'])
            ncc = compute_ncc(fi, reg['warpedmovout'])
            n_syn = len(reg.get('syn_losses', []))
            delta = dice - dice_ref
            print(f"{cfg['name']:<50} {dice:>8.4f} {ncc:>8.4f} {t1:>5.1f}s {delta:>+8.4f} {n_syn:>6}")
            
            # Cleanup
            for path in reg['fwdtransforms'] + reg['invtransforms']:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
        except Exception as e:
            print(f"{cfg['name']:<50} ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
