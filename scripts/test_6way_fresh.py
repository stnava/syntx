#!/usr/bin/env python
"""Fresh run: 6-way verification of {gaussian, sobolev, dsti} × {fast_smooth=True, False}."""
import syntx, ants, time

data = syntx.benchmark_data('r16_r64')
fi, mi = data['fixed'], data['moving']
fl_c2 = data['fixed_labels']['class2']
ml_c2 = data['moving_labels']['class2']
fl_p = data['fixed_labels']['class2_3']
ml_p = data['moving_labels']['class2_3']

print("Computing robust affine initialization...")
reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
aff_tx = reg_aff['fwdtransforms'][0]

configs = [
    ('gaussian', True),
    ('gaussian', False),
    ('sobolev', True),
    ('sobolev', False),
    ('dsti', True),
    ('dsti', False),
]

print()
print("=" * 100)
print(" 6-WAY FRESH VERIFICATION: {gaussian, sobolev, dsti} × {fast_smooth=True, False}")
print(" Parameters: flow_sigma=2.0, grad_step=0.50, reg_iterations=[100, 100, 20], sobolev_alpha=3.0")
print("=" * 100)
print(f"{'Config':<30s} | {'Class 2 GM':>10s} | {'Class 2+3 Par':>13s} | {'Folding %':>10s} | {'Runtime':>8s}")
print("-" * 100)

for reg_name, fs in configs:
    t0 = time.time()
    reg = syntx.syn(
        fixed=fi, moving=mi, initial_transform=aff_tx, backend='pytorch',
        regularizer=reg_name, sobolev_alpha=3.0, flow_sigma=2.0, total_sigma=0.0,
        grad_step=0.50, reg_iterations=[100, 100, 20], affine_iterations=[100, 50, 20],
        inverse_method='anderson', in_loop_inv_steps=10, fast_smooth=fs,
        antisymmetric=True, verbose=False)
    elapsed = time.time() - t0

    # Fixed space evaluation
    ml_w_c2 = ants.apply_transforms(fixed=fi, moving=ml_c2, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    ml_w_p = ants.apply_transforms(fixed=fi, moving=ml_p, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    
    # Moving space evaluation
    fl_w_c2 = ants.apply_transforms(fixed=mi, moving=fl_c2, transformlist=reg['invtransforms'], interpolator='nearestNeighbor')
    fl_w_p = ants.apply_transforms(fixed=mi, moving=fl_p, transformlist=reg['invtransforms'], interpolator='nearestNeighbor')
    
    ov_c2_fwd = ants.label_overlap_measures(fl_c2, ml_w_c2)
    ov_c2_inv = ants.label_overlap_measures(ml_c2, fl_w_c2)
    ov_p_fwd = ants.label_overlap_measures(fl_p, ml_w_p)
    ov_p_inv = ants.label_overlap_measures(ml_p, fl_w_p)
    
    d_c2 = 0.5 * (float(ov_c2_fwd['TotalOrTargetOverlap'].iloc[1]) + float(ov_c2_inv['TotalOrTargetOverlap'].iloc[1]))
    d_p = 0.5 * (float(ov_p_fwd['TotalOrTargetOverlap'].iloc[1]) + float(ov_p_inv['TotalOrTargetOverlap'].iloc[1]))
    
    fold_pct = float(reg.get('grid_folding_percentage', 0.0))
    
    label = f"{reg_name}_fast{fs}"
    print(f"{label:<30s} | {d_c2:10.4f} | {d_p:13.4f} | {fold_pct:9.4f}% | {elapsed:7.2f}s")

print("=" * 100)
