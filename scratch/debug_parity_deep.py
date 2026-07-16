#!/usr/bin/env python
"""
Deeper parity debug: investigating the SOURCES of differences.

Key observations from initial run:
1. Affine: syntx matches ANTs (Delta Dice = +0.0013) ✓
2. SyN on pre-warped: syntx > ANTs by +0.0644 Dice
3. Full pipeline: syntx > ANTs by +0.0527 Dice

Questions:
- Why is syntx doing better? Is ANTs underperforming or syntx overperforming?
- Is ANTs using different default parameters (iterations, regularization)?
- Does the fixed_parameters difference in affine matter?
- Does the warp field composition/application differ?
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


def test_ants_parameter_sweep():
    """
    Test ANTs SyN with different parameter configurations to find its best.
    """
    print("=" * 70)
    print("TEST A: ANTs SyN Parameter Sweep")
    print("=" * 70)
    fi, mi = get_test_data()
    
    configs = [
        # Default SyN
        {"name": "SyN (default)", "type": "SyN"},
        # SyNCC (uses CC metric explicitly)
        {"name": "SyNCC", "type": "SyNCC"},
        # SyNOnly (skip affine)
        {"name": "SyNOnly", "type": "SyNOnly"},
        # ElasticSyN
        {"name": "ElasticSyN", "type": "ElasticSyN"},
        # SyN with more iterations  
        {"name": "SyN custom iters", "type": "SyN",
         "kwargs": {"reg_iterations": (100, 100, 50)}},
        # SyNAggro
        {"name": "SyNAggro", "type": "SyNAggro"},
    ]
    
    results = {}
    for cfg in configs:
        name = cfg["name"]
        try:
            t0 = time.time()
            kwargs = cfg.get("kwargs", {})
            reg = ants.registration(fi, mi, type_of_transform=cfg["type"], **kwargs)
            t1 = time.time() - t0
            dice = compute_tissue_overlap(fi, reg['warpedmovout'])
            ncc = compute_ncc(fi, reg['warpedmovout'])
            print(f"  {name:30s}: Dice={dice:.4f}, NCC={ncc:.4f}, time={t1:.1f}s")
            print(f"    fwd transforms: {[os.path.basename(f) for f in reg['fwdtransforms']]}")
            results[name] = {"dice": dice, "ncc": ncc, "time": t1}
        except Exception as e:
            print(f"  {name:30s}: ERROR - {e}")
    
    return results


def test_syntx_parameter_sweep():
    """
    Test syntx SyN with different parameter configurations.
    """
    print("\n" + "=" * 70)
    print("TEST B: syntx SyNTo Parameter Sweep")
    print("=" * 70)
    fi, mi = get_test_data()
    from syntx.syn import registration
    
    configs = [
        # Match ANTs SyN defaults more closely
        {"name": "syntx grad=0.2 sigma=3",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [100, 100, 50]}},
        {"name": "syntx grad=0.5 sigma=3",
         "kwargs": {"grad_step": 0.5, "flow_sigma": 3.0, "reg_iterations": [100, 100, 50]}},
        {"name": "syntx grad=0.75 sigma=3",
         "kwargs": {"grad_step": 0.75, "flow_sigma": 3.0, "reg_iterations": [100, 100, 50]}},
        # Try matching ANTs SyN's default 3-level [40,20,0]
        {"name": "syntx grad=0.2 iters=[40,20,0]",
         "kwargs": {"grad_step": 0.2, "flow_sigma": 3.0, "reg_iterations": [40, 20, 0]}},
        # With elastic regularization
        {"name": "syntx grad=0.5 sigma=3 elastic=0.5",
         "kwargs": {"grad_step": 0.5, "flow_sigma": 3.0, "total_sigma": 0.5, "reg_iterations": [100, 100, 50]}},
        # Metric: mattes_mi for SyN (like ANTs default?)
        {"name": "syntx mattes_mi for syn",
         "kwargs": {"grad_step": 0.5, "flow_sigma": 3.0, "syn_metric": "mattes_mi", "reg_iterations": [100, 100, 50]}},
    ]
    
    results = {}
    for cfg in configs:
        name = cfg["name"]
        try:
            t0 = time.time()
            reg = registration(
                fi, mi, type_of_transform='SyN', backend='pytorch',
                aff_metric='mattes',
                **cfg["kwargs"]
            )
            t1 = time.time() - t0
            dice = compute_tissue_overlap(fi, reg['warpedmovout'])
            ncc = compute_ncc(fi, reg['warpedmovout'])
            print(f"  {name:40s}: Dice={dice:.4f}, NCC={ncc:.4f}, time={t1:.1f}s")
            results[name] = {"dice": dice, "ncc": ncc, "time": t1}
            # Cleanup
            for path in reg['fwdtransforms'] + reg['invtransforms']:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
        except Exception as e:
            print(f"  {name:40s}: ERROR - {e}")
            import traceback; traceback.print_exc()
    
    return results


def test_warp_field_analysis():
    """
    Analyze the actual warp fields produced by both ANTs and syntx.
    """
    print("\n" + "=" * 70)
    print("TEST C: Warp Field Analysis (ANTs vs syntx)")
    print("=" * 70)
    fi, mi = get_test_data()
    from syntx.syn import registration
    
    # Run ANTs SyN
    reg_ants = ants.registration(fi, mi, type_of_transform='SyN')
    
    # Run syntx SyNTo with matched parameters
    reg_syntx = registration(
        fi, mi, type_of_transform='SyN', backend='pytorch',
        aff_metric='mattes',
        syn_metric='lncc', syn_sampling=2,
        grad_step=0.5, flow_sigma=3.0,
        reg_iterations=[100, 100, 50],
    )
    
    # Analyze ANTs warp field
    print("\n--- ANTs Warp Field ---")
    ants_fwd = reg_ants['fwdtransforms']
    for i, f in enumerate(ants_fwd):
        if f.endswith('.nii.gz'):
            warp_img = ants.image_read(f)
            warp_np = warp_img.numpy()
            print(f"  Transform[{i}]: {os.path.basename(f)}")
            print(f"    Shape: {warp_np.shape}")
            print(f"    Displacement range: [{warp_np.min():.4f}, {warp_np.max():.4f}]")
            print(f"    Mean displacement: {np.abs(warp_np).mean():.4f}")
            print(f"    Max |displacement|: {np.abs(warp_np).max():.4f}")
        else:
            print(f"  Transform[{i}]: {os.path.basename(f)} (affine)")
            tx = ants.read_transform(f)
            print(f"    Params: {tx.parameters}")
            print(f"    Fixed params: {tx.fixed_parameters}")
    
    # Analyze syntx warp field
    print("\n--- syntx Warp Field ---")
    syntx_fwd = reg_syntx['fwdtransforms']
    for i, f in enumerate(syntx_fwd):
        if f.endswith('.nii.gz'):
            warp_img = ants.image_read(f)
            warp_np = warp_img.numpy()
            print(f"  Transform[{i}]: {os.path.basename(f)}")
            print(f"    Shape: {warp_np.shape}")
            print(f"    Displacement range: [{warp_np.min():.4f}, {warp_np.max():.4f}]")
            print(f"    Mean displacement: {np.abs(warp_np).mean():.4f}")
            print(f"    Max |displacement|: {np.abs(warp_np).max():.4f}")
        else:
            print(f"  Transform[{i}]: {os.path.basename(f)} (affine)")
            tx = ants.read_transform(f)
            print(f"    Params: {tx.parameters}")
            print(f"    Fixed params: {tx.fixed_parameters}")
    
    # Compare the model's internal state
    model = reg_syntx['model']
    print("\n--- syntx Model Internal State ---")
    print(f"  Affine losses: first={model.affine_losses[0]:.6f}, last={model.affine_losses[-1]:.6f}" if model.affine_losses else "  No affine losses")
    print(f"  SyN losses: first={model.syn_losses[0]:.6f}, last={model.syn_losses[-1]:.6f}" if model.syn_losses else "  No SyN losses")
    print(f"  num affine iters: {len(model.affine_losses)}")
    print(f"  num syn iters: {len(model.syn_losses)}")
    
    # warp_l2r internal
    import torch
    with torch.no_grad():
        wl2r = model.warp_l2r.data
        print(f"\n  warp_l2r shape: {wl2r.shape}")
        print(f"  warp_l2r range: [{wl2r.min():.4f}, {wl2r.max():.4f}]")
        print(f"  warp_l2r mean abs: {wl2r.abs().mean():.4f}")
        print(f"  warp_l2r max abs: {wl2r.abs().max():.4f}")
        
        # Compute folding (Jacobian < 0)
        disp_norm = torch.sqrt(torch.sum(wl2r**2, dim=-1))
        print(f"  warp_l2r Euclidean disp range: [{disp_norm.min():.4f}, {disp_norm.max():.4f}]")
    
    return reg_ants, reg_syntx


def test_affine_fixed_params():
    """
    Investigate: does fixed_parameters=[0,0] vs center-based make a difference?
    ANTs reports fixed_params=[125.27, 129.21] — this is the center of rotation.
    syntx reports fixed_params=[0, 0].
    
    Test: read syntx affine, apply manually with center at [0,0] vs with
    ANTs center. Check if results differ.
    """
    print("\n" + "=" * 70)
    print("TEST D: Affine fixed_parameters (center of rotation) analysis")
    print("=" * 70)
    fi, mi = get_test_data()
    from syntx.syn import registration
    
    # Get ANTs affine
    reg_ants = ants.registration(fi, mi, type_of_transform='Affine')
    ants_tx = ants.read_transform(reg_ants['fwdtransforms'][0])
    print(f"ANTs affine params: {ants_tx.parameters}")
    print(f"ANTs fixed params (center): {ants_tx.fixed_parameters}")
    
    # Get syntx affine  
    reg_syntx = registration(fi, mi, type_of_transform='Affine', backend='pytorch')
    syntx_tx = ants.read_transform(reg_syntx['fwdtransforms'][0])
    print(f"syntx affine params: {syntx_tx.parameters}")
    print(f"syntx fixed params (center): {syntx_tx.fixed_parameters}")
    
    # The ITK AffineTransform convention:
    # mapped_point = M @ (point - fixed_center) + fixed_center + translation
    # So for fixed_center=0: mapped_point = M @ point + translation
    # For fixed_center=c: mapped_point = M @ (point - c) + c + t = M @ point + (c - M@c + t)
    # They're mathematically equivalent if the translation is adjusted.
    
    # Let's verify: compute the effective M and t for both
    dim = 2
    M_ants = np.array(ants_tx.parameters[:dim*dim]).reshape(dim, dim)
    t_ants = np.array(ants_tx.parameters[dim*dim:])
    c_ants = np.array(ants_tx.fixed_parameters)
    
    M_syntx = np.array(syntx_tx.parameters[:dim*dim]).reshape(dim, dim)
    t_syntx = np.array(syntx_tx.parameters[dim*dim:])
    c_syntx = np.array(syntx_tx.fixed_parameters)
    
    # Effective affine: point -> M @ point + (c - M @ c + t)
    t_eff_ants = c_ants - M_ants @ c_ants + t_ants
    t_eff_syntx = c_syntx - M_syntx @ c_syntx + t_syntx
    
    print(f"\nEffective (center-free) comparison:")
    print(f"  ANTs M:\n{M_ants}")
    print(f"  syntx M:\n{M_syntx}")
    print(f"  Delta M:\n{M_ants - M_syntx}")
    print(f"  Max |Delta M|: {np.abs(M_ants - M_syntx).max():.6f}")
    print(f"\n  ANTs effective t: {t_eff_ants}")
    print(f"  syntx effective t: {t_eff_syntx}")
    print(f"  Delta effective t: {t_eff_ants - t_eff_syntx}")
    print(f"  Max |Delta t|: {np.abs(t_eff_ants - t_eff_syntx).max():.4f}")
    
    # Apply both transforms to the same test point
    test_point = np.array([128.0, 128.0])  # center of image
    ants_mapped = M_ants @ (test_point - c_ants) + c_ants + t_ants
    syntx_mapped = M_syntx @ (test_point - c_syntx) + c_syntx + t_syntx
    print(f"\n  Test point {test_point} mapped:")
    print(f"    ANTs:  {ants_mapped}")
    print(f"    syntx: {syntx_mapped}")
    print(f"    diff:  {ants_mapped - syntx_mapped}")
    
    # Apply to corner of image
    test_point2 = np.array([0.0, 0.0])
    ants_mapped2 = M_ants @ (test_point2 - c_ants) + c_ants + t_ants
    syntx_mapped2 = M_syntx @ (test_point2 - c_syntx) + c_syntx + t_syntx
    print(f"\n  Test point {test_point2} mapped:")
    print(f"    ANTs:  {ants_mapped2}")
    print(f"    syntx: {syntx_mapped2}")
    print(f"    diff:  {ants_mapped2 - syntx_mapped2}")


if __name__ == '__main__':
    print("DEEP PARITY DEBUG: ANTs vs syntx")
    print()
    
    # Test A: ANTs parameter sweep to understand its best performance
    ants_results = test_ants_parameter_sweep()
    
    # Test B: syntx parameter sweep
    syntx_results = test_syntx_parameter_sweep()
    
    # Test C: Warp field analysis
    test_warp_field_analysis()
    
    # Test D: Affine center analysis
    test_affine_fixed_params()
