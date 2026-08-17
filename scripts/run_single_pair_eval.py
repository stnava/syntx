#!/usr/bin/env python
"""
Single-Pair Isolated Benchmark Evaluation Worker
================================================
Executes a single Mindboggle pair in an isolated process to prevent GPU memory leaks.
Reuses existing official ANTs C++ baseline files from results/pair_{idx:03d}_ants_syn.json.
Supports Sobolev SyN, Gaussian SyN, and comparisons.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

def normalize_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
    """Foreground 2nd-to-98th percentile intensity normalization."""
    arr = img.numpy()
    pos = arr[arr > 0]
    if len(pos) > 0:
        p02 = float(np.percentile(pos, 2.0))
        p98 = float(np.percentile(pos, 98.0))
        if p98 <= p02 + 1e-4:
            p02 = 0.0
            p98 = float(pos.max())
    else:
        p02 = float(arr.min())
        p98 = float(arr.max())
    norm_arr = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)
    return img.new_image_like(norm_arr)

def run_single_eval(pair_idx: int, model_type: str = "sobolev", out_dir: str = "results/reproducible_eval", ants_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"pair_{pair_idx:03d}_{model_type}.json")
    
    # 1. Load Existing Official ANTs Baseline
    ants_baseline_file = os.path.join(ants_dir, f"pair_{pair_idx:03d}_ants_syn.json")
    ants_rec = {}
    if os.path.exists(ants_baseline_file):
        try:
            with open(ants_baseline_file, "r") as f:
                ants_rec = json.load(f)
        except Exception:
            pass
            
    # Deterministic seeding
    torch.manual_seed(42 + pair_idx)
    np.random.seed(42 + pair_idx)
    
    t0_start = time.time()
    p = load_mindboggle_pair(pair_idx, "examples/pairs.csv")
    fi_raw, mi_raw = p['fixed'], p['moving']
    fl, ml = p['fixed_label'], p['moving_label']
    
    # 2. Percentile Normalization
    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)
    
    # 3. Robust Affine Pre-Alignment
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode='pytorch', n_starts=3, verbose=False)
    aff_0 = reg_aff['fwdtransforms'][0]
    t_aff = time.time() - t0_aff
    
    # 4. Syntx Deformable SyN
    t0_syn = time.time()
    if model_type == "sobolev":
        res_syn = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
            grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
            reg_iterations=[80, 80, 20], similarity_metric='cc2',
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=False, inverse_method='anderson',
            formulation='eulerian', regularizer='sobolev', sobolev_alpha=1.5,
            antisymmetric=True, verbose=False
        )
    else:  # gaussian
        res_syn = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
            grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
            reg_iterations=[80, 80, 20], similarity_metric='cc2',
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=False, inverse_method='anderson',
            formulation='eulerian', regularizer='gaussian',
            antisymmetric=True, verbose=False
        )
    t_syn = time.time() - t0_syn + t_aff
    
    # 5. Evaluate Metrics
    df_f_s, df_m_s, sym_s = compute_bidirectional_dice(
        fl, ml, fi, mi,
        res_syn['fwdtransforms'], res_syn['invtransforms'],
        res_syn['whichtoinvert_inv']
    )
    fwd_warp_s = next(x for x in res_syn['fwdtransforms'] if isinstance(x, str) and x.endswith('.nii.gz'))
    jac_s = compute_jacobian_metrics(fi, fwd_warp_s)
    
    inv_errs = res_syn.get('inverse_identity_errors', {})
    if 'phi_1' in inv_errs:
        inv_mean_s = float(inv_errs['phi_1'].get('mean', float('nan')))
        inv_p95_s = float(inv_errs['phi_1'].get('p95', float('nan')))
    else:
        inv_mean_s = float(inv_errs.get('mean', float('nan')))
        inv_p95_s = float(inv_errs.get('p95', float('nan')))
        
    ants_dice_sym = float(ants_rec.get("dice_sym", float("nan")))
    ants_dice_f = float(ants_rec.get("dice_fixed", float("nan")))
    ants_dice_m = float(ants_rec.get("dice_moving", float("nan")))
    ants_fold = float(ants_rec.get("folding_pct", 0.0))
    ants_min_jac = float(ants_rec.get("min_jacobian", 0.0))
    ants_time = float(ants_rec.get("runtime_seconds", float("nan")))
    
    rec = {
        'pair_idx': pair_idx,
        'model_type': model_type,
        'cohort_type': p.get('type', ants_rec.get('pair_type', 'unknown')),
        'fixed_id': ants_rec.get('fixed_id', f"pair_{pair_idx:03d}_fix"),
        'moving_id': ants_rec.get('moving_id', f"pair_{pair_idx:03d}_mov"),
        'status': 'SUCCESS',
        'syntx_dice_sym': float(sym_s),
        'syntx_dice_fixed': float(df_f_s),
        'syntx_dice_moving': float(df_m_s),
        'syntx_fold': float(jac_s['folding_pct']),
        'syntx_min_jac': float(jac_s['min']),
        'syntx_inv_mean': float(inv_mean_s),
        'syntx_inv_p95': float(inv_p95_s),
        'syntx_time': float(t_syn),
        'ants_baseline': {
            'dice_sym': ants_dice_sym,
            'dice_fixed': ants_dice_f,
            'dice_moving': ants_dice_m,
            'folding_pct': ants_fold,
            'min_jacobian': ants_min_jac,
            'runtime_seconds': ants_time
        }
    }
    
    with open(out_file, "w") as f:
        json.dump(rec, f, indent=2)
        
    diff_vs_ants = (sym_s - ants_dice_sym) * 100.0 if np.isfinite(ants_dice_sym) else float("nan")
    win_str = "WIN" if (np.isfinite(ants_dice_sym) and sym_s >= ants_dice_sym) else "LOSS"
    print(f"CASE_COMPLETE: Pair {pair_idx:02d} [{model_type.upper()}] | Sym Dice: {sym_s:.4f} (ANTs: {ants_dice_sym:.4f}, diff: {diff_vs_ants:+.2f}%) | Fold: {jac_s['folding_pct']:.4f}% | MinJac: {jac_s['min']:.4f} | Time: {t_syn:.1f}s (ANTs: {ants_time:.1f}s) | Result: {win_str}", flush=True)
    return rec

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_single_pair_eval.py <pair_idx> [model_type: sobolev | gaussian]")
        sys.exit(1)
        
    p_idx = int(sys.argv[1])
    m_type = sys.argv[2] if len(sys.argv) > 2 else "sobolev"
    run_single_eval(p_idx, m_type)
