#!/usr/bin/env python3
"""
Benchmark Evaluation Suite: Sulcal Soft Dice Guided SyN vs Sobolev Baseline
===========================================================================
Evaluates 10 representative Mindboggle benchmark pairs:
- 5 Intra-Study pairs (OASIS-TRT-20): Pairs 0, 1, 2, 3, 4
- 5 Inter-Study pairs (Cross-Site): Pairs 42, 43, 44, 53, 54

Compares:
1. Baseline: Standard Sobolev SyN (1-channel CC2, sobolev_alpha=1.5)
2. Guided: Sulcal Soft Dice Guided SyN (guided='sulcal')
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import ants
import antstorch

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.robust_affine import robust_affine
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

def evaluate_pair(pair_idx: int, device: str = 'cpu'):
    print(f"\n{'='*75}\nEvaluating Mindboggle Pair {pair_idx:02d}...\n{'='*75}", flush=True)
    p = load_mindboggle_pair(pair_idx)
    fi, mi = p['fixed'], p['moving']
    fl, ml = p['fixed_label'], p['moving_label']
    pair_type = p.get('type', 'inter' if pair_idx >= 42 else 'intra')

    # 1. Preprocessing: N4 / Denoise
    fi_dn = antstorch.denoise_image(fi, shrink_factor=2, p=1, r=1, noise_model="Rician")
    mi_dn = antstorch.denoise_image(mi, shrink_factor=2, p=1, r=1, noise_model="Rician")

    # 2. Locked Deterministic Affine
    t_aff_start = time.time()
    aff_res = robust_affine(fi_dn, mi_dn, mode='auto', seed=42)
    aff_time = time.time() - t_aff_start
    aff_fwd = aff_res['fwdtransforms'][0]

    d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(fl, ml, fi, mi, [aff_fwd], [aff_fwd], [False])
    print(f"  Affine: Sym Dice = {d_sym_aff:.4f} ({aff_time:.1f}s)", flush=True)

    # 3. Method 1: Baseline Sobolev SyN (1-channel CC2)
    print("  Running Baseline Sobolev SyN (1-channel CC2)...", flush=True)
    t0 = time.time()
    res_base = syntx.syn(
        fixed=fi_dn,
        moving=mi_dn,
        initial_transform=[aff_fwd],
        reg_iterations=[100, 100, 20],
        levels=[4, 2, 1],
        affine_iterations=0,
        regularizer='sobolev',
        sobolev_alpha=1.5,
        grad_step=0.25,
        similarity_metric='cc2',
        verbose=0
    )
    time_base = time.time() - t0
    fwd_base = res_base['fwdtransforms']
    inv_base = res_base['invtransforms']
    which_inv_base = res_base.get('whichtoinvert_inv', [True, False])

    d_fix_b, d_mov_b, d_sym_b = compute_bidirectional_dice(fl, ml, fi, mi, fwd_base, inv_base, which_inv_base)
    warp_b = ants.image_read(fwd_base[0])
    jac_b = compute_jacobian_metrics(fi, warp_b)
    print(f"    Base Result: Sym Dice={d_sym_b:.4f} (Fix={d_fix_b:.4f}, Mov={d_mov_b:.4f}) | Folds={jac_b['folding_pct']:.5f}% | Time={time_base:.1f}s", flush=True)

    # 4. Method 2: Sulcal Soft Dice Guided SyN
    print("  Running Sulcal Soft Dice Guided SyN (guided='sulcal')...", flush=True)
    t0 = time.time()
    res_guided = syntx.syn(
        fixed=fi_dn,
        moving=mi_dn,
        initial_transform=[aff_fwd],
        reg_iterations=[100, 100, 20],
        levels=[4, 2, 1],
        affine_iterations=0,
        regularizer='sobolev',
        sobolev_alpha=1.5,
        grad_step=0.25,
        guided='sulcal',
        cohort_type=pair_type,
        verbose=0
    )
    time_guided = time.time() - t0
    fwd_g = res_guided['fwdtransforms']
    inv_g = res_guided['invtransforms']
    which_inv_g = res_guided.get('whichtoinvert_inv', [True, False])

    d_fix_g, d_mov_g, d_sym_g = compute_bidirectional_dice(fl, ml, fi, mi, fwd_g, inv_g, which_inv_g)
    warp_g = ants.image_read(fwd_g[0])
    jac_g = compute_jacobian_metrics(fi, warp_g)
    diff = d_sym_g - d_sym_b
    print(f"    Guided Result: Sym Dice={d_sym_g:.4f} (Fix={d_fix_g:.4f}, Mov={d_mov_g:.4f}) | Diff={diff:+.4f} | Folds={jac_g['folding_pct']:.5f}% | Time={time_guided:.1f}s", flush=True)

    return {
        'pair_idx': pair_idx,
        'pair_type': pair_type,
        'affine_sym_dice': float(d_sym_aff),
        'baseline': {
            'sym_dice': float(d_sym_b),
            'fixed_dice': float(d_fix_b),
            'moving_dice': float(d_mov_b),
            'folding_pct': float(jac_b['folding_pct']),
            'min_jac': float(jac_b['min']),
            'time_sec': float(time_base),
        },
        'guided': {
            'sym_dice': float(d_sym_g),
            'fixed_dice': float(d_fix_g),
            'moving_dice': float(d_mov_g),
            'folding_pct': float(jac_g['folding_pct']),
            'min_jac': float(jac_g['min']),
            'time_sec': float(time_guided),
        },
        'dice_diff': float(diff),
        'win': bool(d_sym_g >= d_sym_b)
    }

def main():
    parser = argparse.ArgumentParser(description="Run 10-pair Mindboggle benchmark comparing Sobolev baseline vs Sulcal-guided SyN")
    parser.add_argument('--pairs', nargs='+', type=int, default=[0, 1, 2, 3, 4, 42, 43, 44, 53, 54],
                        help="List of pair indices to evaluate (default: 5 intra, 5 inter)")
    parser.add_argument('--output', type=str,
                        default="/Users/stnava/.gemini/antigravity-cli/brain/ed9af813-1310-43df-bc86-a32aeec400a0/scratch/sulcal_guided_10pair_results.json",
                        help="Path to save output JSON")
    args = parser.parse_args()

    print("=" * 85, flush=True)
    print("MINDBOGGLE 10-PAIR COHORT BENCHMARK: SULCAL-GUIDED SyN VS SOBOLEV BASELINE", flush=True)
    print("=" * 85, flush=True)

    results = []
    for p_idx in args.pairs:
        res = evaluate_pair(p_idx)
        results.append(res)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary Statistics
    base_dices = [r['baseline']['sym_dice'] for r in results]
    guided_dices = [r['guided']['sym_dice'] for r in results]
    base_folds = [r['baseline']['folding_pct'] for r in results]
    guided_folds = [r['guided']['folding_pct'] for r in results]
    wins = [r['win'] for r in results]

    print("\n" + "=" * 90, flush=True)
    print(f"{'Pair':<6} | {'Type':<6} | {'Base Dice':<11} | {'Guided Dice':<12} | {'Diff':<9} | {'Base Folds':<11} | {'Guided Folds':<12} | {'Win':<5}", flush=True)
    print("-" * 90, flush=True)
    for r in results:
        w_str = "WIN" if r['win'] else "LOSS"
        print(f"Pair {r['pair_idx']:02d} | {r['pair_type']:<6} | {r['baseline']['sym_dice']:.4f}      | {r['guided']['sym_dice']:.4f}       | {r['dice_diff']:+.4f}   | {r['baseline']['folding_pct']:.5f}%   | {r['guided']['folding_pct']:.5f}%    | {w_str}", flush=True)
    print("=" * 90, flush=True)
    print(f"Mean Baseline Symmetric Dice : {np.mean(base_dices):.4f}", flush=True)
    print(f"Mean Guided Symmetric Dice   : {np.mean(guided_dices):.4f} ({np.mean(guided_dices) - np.mean(base_dices):+.4f})", flush=True)
    print(f"Mean Baseline Folding        : {np.mean(base_folds):.5f}%", flush=True)
    print(f"Mean Guided Folding          : {np.mean(guided_folds):.5f}%", flush=True)
    print(f"Overall Guided Win Rate      : {np.sum(wins)}/{len(wins)} ({np.mean(wins)*100:.1f}%)", flush=True)
    print("=" * 90, flush=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {args.output}", flush=True)

if __name__ == '__main__':
    main()
