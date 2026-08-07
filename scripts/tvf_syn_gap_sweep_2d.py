import os
import sys
import time
import json
import torch
import numpy as np
import ants
import syntx

def compute_jacobian_folding(fi, fwdtransforms):
    jac = ants.create_jacobian_determinant_image(fi, fwdtransforms[0])
    jac_np = jac.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0
    return float(np.mean(jac_np[mask_eval] <= 0) * 100)

def main():
    print("Loading 2D benchmark data...")
    data = syntx.benchmark_data('2d')
    fi = data['fixed']
    mi = data['moving']
    fl = data['fixed_label']
    ml = data['moving_label']

    print("Computing robust affine...")
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]
    
    # Base configuration
    fast_smooth = True
    
    # Grid search configs
    syn_fluid_sigmas = [1.0, 2.0, 3.0]
    tvf_fluid_sigmas = [0.2, 0.4, 0.8, 1.6]
    
    syn_grad_steps = [0.25, 0.50, 1.0]
    tvf_grad_steps = [0.25, 0.50, 1.0]
    
    tvf_total_sigmas = [0.05, 0.10]
    
    results = []

    # --- SyN Sweep ---
    for fsig in syn_fluid_sigmas:
        for gs in syn_grad_steps:
            cfg_name = f"syn_fsig{fsig}_gstep{gs}"
            print(f"\nRunning {cfg_name}...")
            t0 = time.time()
            reg = syntx.syn(
                fixed=fi, moving=mi, initial_transform=aff_tx,
                backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
                reg_iterations=[100, 40], affine_iterations=[50, 20],
                similarity_metric='lncc', syn_sampling=2,
                inverse_method='anderson',
                flow_sigma=fsig, total_sigma=0.0, grad_step=gs,
                regularizer='gaussian', antisymmetric=True,
                fast_smooth=fast_smooth
            )
            t1 = time.time()
            ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
            overlap = ants.label_overlap_measures(fl, ml_warped)
            df = overlap[~overlap['Label'].astype(str).isin(['All', '0', '0.0'])]
            col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
            dice = float(df[col].mean()) if len(df) > 0 else 0.0
            fold = compute_jacobian_folding(fi, reg['fwdtransforms'])
            print(f"  -> Dice: {dice:.4f} | Folding: {fold:.4f}% | Time: {t1-t0:.1f}s")
            results.append({'model': 'syn', 'fsig': fsig, 'gstep': gs, 'tsig': 0.0, 'dice': dice, 'fold': fold, 'time': t1-t0})

    # --- TVF Sweep ---
    for fsig in tvf_fluid_sigmas:
        for tsig in tvf_total_sigmas:
            for gs in tvf_grad_steps:
                cfg_name = f"tvf_fsig{fsig}_tsig{tsig}_gstep{gs}"
                print(f"\nRunning {cfg_name}...")
                t0 = time.time()
                reg = syntx.tvf(
                    fixed=fi, moving=mi, initial_transform=aff_tx,
                    backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
                    reg_iterations=[100, 100, 20],
                    similarity_metric='lncc', syn_sampling=2, multipoint_loss=[0.0, 0.5, 1.0],
                    flow_sigma=fsig, total_sigma=tsig, grad_step=gs,
                    regularizer='gaussian',
                    optimizer='lars', cfl_max=0.0, cfl_momentum=0.95, n_time_steps=3, 
                    constant_speed=True, use_analytical_gradients=True, 
                    antisymmetric=True
                )
                t1 = time.time()
                ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
                overlap = ants.label_overlap_measures(fl, ml_warped)
                df = overlap[~overlap['Label'].astype(str).isin(['All', '0', '0.0'])]
                col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
                dice = float(df[col].mean()) if len(df) > 0 else 0.0
                fold = compute_jacobian_folding(fi, reg['fwdtransforms'])
                print(f"  -> Dice: {dice:.4f} | Folding: {fold:.4f}% | Time: {t1-t0:.1f}s")
                results.append({'model': 'tvf', 'fsig': fsig, 'gstep': gs, 'tsig': tsig, 'dice': dice, 'fold': fold, 'time': t1-t0})
                
    with open('benchmark_syn_tvf_gap.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
