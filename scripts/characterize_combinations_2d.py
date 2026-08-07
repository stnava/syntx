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

    models = ['syn', 'tvf', 'syngs']
    regs = ['gaussian', 'sobolev', 'dsti']
    fast_smooths = [True, False]
    
    # Pre-compute robust affine
    print("Computing robust affine...")
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]

    results = []

    for m in models:
        for r in regs:
            for fs in fast_smooths:
                cfg_id = f"{m}_{r}_fast{fs}"
                print(f"Running {cfg_id}...")
                
                try:
                    t0 = time.time()
                    if m == 'syn':
                        reg = syntx.syn(
                            fixed=fi, moving=mi, initial_transform=aff_tx,
                            backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
                            reg_iterations=[100, 40], affine_iterations=[50, 20],
                            similarity_metric='lncc', syn_sampling=2,
                            inverse_method='anderson',
                            flow_sigma=3.0, total_sigma=0.0, grad_step=0.50,
                            regularizer=r, antisymmetric=True,
                            fast_smooth=fs
                        )
                    elif m == 'syngs':
                        reg = syntx.syngs(
                            fixed=fi, moving=mi, initial_transform=aff_tx,
                            backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
                            reg_iterations=[100, 40], affine_iterations=[50, 20],
                            similarity_metric='lncc', syn_sampling=2,
                            flow_sigma=1.6 if r == 'dsti' else 0.4, 
                            total_sigma=0.05, 
                            grad_step=1.0 if r == 'dsti' else 0.5,
                            regularizer=r, fast_smooth=fs
                        )
                    else:
                        reg = syntx.tvf(
                            fixed=fi, moving=mi, initial_transform=aff_tx,
                            backend='pytorch', device='cpu' if not torch.backends.mps.is_available() else 'mps',
                            reg_iterations=[100, 40], affine_iterations=[50, 20],
                            similarity_metric='lncc', syn_sampling=2, multipoint_loss=[0.0, 0.5, 1.0],
                            flow_sigma=1.6 if r == 'dsti' else 0.4, 
                            total_sigma=0.05, 
                            grad_step=1.0 if r == 'dsti' else 0.5,
                            optimizer='lars', cfl_max=0.0, cfl_momentum=0.95, n_time_steps=3, 
                            constant_speed=True, use_analytical_gradients=True, 
                            regularizer=r, fast_smooth=fs
                        )
                    elapsed = time.time() - t0
                    
                    # Evaluate Dice
                    ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
                    overlap = ants.label_overlap_measures(fl, ml_warped)
                    df = overlap[~overlap['Label'].astype(str).isin(['All', '0', '0.0'])]
                    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
                    dice_fixed = float(df[col].mean()) if len(df) > 0 else 0.0
                    
                    # Inverse Dice
                    fl_warped = ants.apply_transforms(fixed=mi, moving=fl, transformlist=reg['invtransforms'], whichtoinvert=reg.get('whichtoinvert_inv', [True, False]), interpolator='nearestNeighbor')
                    overlap_inv = ants.label_overlap_measures(ml, fl_warped)
                    df_inv = overlap_inv[~overlap_inv['Label'].astype(str).isin(['All', '0', '0.0'])]
                    col_inv = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_inv.columns else 'TargetOverlap'
                    dice_moving = float(df_inv[col_inv].mean()) if len(df_inv) > 0 else 0.0
                    
                    dice_sym = 0.5 * (dice_fixed + dice_moving)
                    
                    folding = compute_jacobian_folding(fi, reg['fwdtransforms'])
                    
                    res = {
                        'config': cfg_id,
                        'model': m,
                        'regularizer': r,
                        'fast_smooth': fs,
                        'time': elapsed,
                        'dice_sym': dice_sym,
                        'folding_pct': folding
                    }
                    print(f"  -> Dice: {dice_sym:.4f} | Folding: {folding:.4f}% | Time: {elapsed:.1f}s")
                    results.append(res)
                    
                except Exception as e:
                    print(f"  -> Failed: {e}")
                    import traceback
                    traceback.print_exc()

    with open('benchmark_2d_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
