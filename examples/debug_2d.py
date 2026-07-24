"""
Fast 2D parity test: stripped-down version of generate_ants_2d_comparison_report.py
for quick correctness checking during development.
Runs ANTs, PyTorch, and JAX SyN on r16/r64 with reduced iterations.
Prints MI, Dice, and inverse identity errors.
"""
import ants
import numpy as np
import time
import syntx

def compute_tissue_overlap(fi, warped):
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return dice

def main():
    print("="*60)
    print("  Fast 2D Parity Test (debug_2d.py)")
    print("  Defaults: project_inverse=True, projection_frequency=20")
    print("="*60)
    
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    reg_iterations=[100,100,100,20]
    aff_iterations=[1000,1000,1000,1000]
    # --- ANTs ---
    print("\n[1/3] Running ANTs SyN...")
    t0 = time.time()
    reg_ants = ants.registration(
        fi, mi, 'SyN',
        reg_iterations=reg_iterations,
        syn_metric='cc', syn_sampling=2
    )
    ants_time = time.time() - t0
    mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
    dice_ants = compute_tissue_overlap(fi, reg_ants['warpedmovout'])
    
    mygs=0.2
    myfs=4.0
    s=0.2
    myinv=5
    # --- PyTorch (using defaults) ---
    print("[2/3] Running PyTorch SyN (default settings)...")
    t0 = time.time()
    reg_py = syntx.syn(
        fixed=fi, moving=mi,
        reg_iterations=reg_iterations,
        affine_iterations=aff_iterations,
        grad_step=mygs, flow_sigma=myfs,
        sampling_percentage=s,
        syn_metric='lncc', lncc_radius=2,
        backend='pytorch', inverse_steps=myinv
    )
    py_time = time.time() - t0
    mi_py = ants.image_mutual_information(fi, reg_py['warpedmovout'])
    dice_py = compute_tissue_overlap(fi, reg_py['warpedmovout'])
    
    # --- JAX (using defaults) ---
    print("[3/3] Running JAX SyN (default settings)...")
    t0 = time.time()
    reg_jax = syntx.syn(
        fixed=fi, moving=mi,
        reg_iterations=reg_iterations,
        affine_iterations=aff_iterations,
        grad_step=mygs, flow_sigma=myfs,
        sampling_percentage=s,
        syn_metric='lncc', lncc_radius=2,
        backend='jax', inverse_steps=myinv
    )
    jax_time = time.time() - t0
    mi_jax = ants.image_mutual_information(fi, reg_jax['warpedmovout'])
    dice_jax = compute_tissue_overlap(fi, reg_jax['warpedmovout'])
    
    # --- Results ---
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"{'Method':<20} {'MI':>10} {'Dice':>10} {'Time(s)':>10}")
    print("-"*50)
    print(f"{'ANTs':<20} {mi_ants:>10.4f} {dice_ants:>10.4f} {ants_time:>10.1f}")
    print(f"{'PyTorch':<20} {mi_py:>10.4f} {dice_py:>10.4f} {py_time:>10.1f}")
    print(f"{'JAX':<20} {mi_jax:>10.4f} {dice_jax:>10.4f} {jax_time:>10.1f}")
    
    print("\nInverse Identity Errors:")
    print(f"  PyTorch: {reg_py['inverse_identity_errors']}")
    print(f"  JAX:     {reg_jax['inverse_identity_errors']}")
    
    # --- Checks ---
    print("\n" + "="*60)
    print("  CHECKS")
    print("="*60)
    
    all_pass = True
    
    # Dice within 1.0% of ANTs (GEMINI.md Rule 2)
    dice_gap_py = dice_ants - dice_py
    dice_gap_jax = dice_ants - dice_jax
    
    if dice_gap_py > 0.01:
        print(f"  FAIL: PyTorch Dice gap {dice_gap_py:.4f} > 0.01")
        all_pass = False
    else:
        print(f"  PASS: PyTorch Dice gap {dice_gap_py:.4f}")
        
    if dice_gap_jax > 0.01:
        print(f"  FAIL: JAX Dice gap {dice_gap_jax:.4f} > 0.01")
        all_pass = False
    else:
        print(f"  PASS: JAX Dice gap {dice_gap_jax:.4f}")
    
    # Inverse identity max errors should be < 3.0 mm and mean error < 0.05 mm
    for name, errors in [('PyTorch', reg_py['inverse_identity_errors']), ('JAX', reg_jax['inverse_identity_errors'])]:
        for field_name, field_errors in errors.items():
            if field_errors['max_error'] >= 3.0 or field_errors['mean_error'] >= 0.05:
                print(f"  FAIL: {name} {field_name} max_error {field_errors['max_error']:.4f} >= 3.0 or mean {field_errors['mean_error']:.4f} >= 0.05")
                all_pass = False
            else:
                print(f"  PASS: {name} {field_name} max_error {field_errors['max_error']:.4f} (mean {field_errors['mean_error']:.4f} mm)")
    
    if all_pass:
        print("\n  ALL CHECKS PASSED ✓")
    else:
        print("\n  SOME CHECKS FAILED ✗")
    
    return all_pass

if __name__ == "__main__":
    main()
