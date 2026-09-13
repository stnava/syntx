#!/usr/bin/env python3
"""Example: Ultra-Fast Forward Registration on 3D Mindboggle mbhard using syntx.greedy.

Demonstrates syntx.greedy — a unidirectional Eulerian compositive registration method
delivering sub-25s forward alignments with robust affine initialization and
variance-floored LNCC.

Usage:
    python examples/run_greedy_mbhard.py
"""

import time
import os
import numpy as np
import pandas as pd
import torch
import ants
import syntx
from syntx.benchmark.evaluate import normalize_intensity


def compute_cortical_dice_and_jacobian(fixed_img, moving_label, fixed_label, warp_path):
    warped_lbl = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_label,
        transformlist=[warp_path],
        interpolator='nearestNeighbor'
    )
    ov = ants.label_overlap_measures(fixed_label, warped_lbl)
    df = ov[~ov['Label'].astype(str).isin(['All', '0', '0.0'])]
    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'MeanOverlap'
    vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals >= 0.0) & (vals <= 1.0)]
    dice = float(np.mean(vals)) if len(vals) > 0 else 0.0

    jac_img = ants.create_jacobian_determinant_image(fixed_img, warp_path, do_log=False)
    jac_np = jac_img.numpy()
    mask_eval = ants.get_mask(fixed_img).numpy() > 0
    min_det = float(np.min(jac_np[mask_eval]))
    fold_pct = float(np.mean(jac_np[mask_eval] <= 0.0) * 100.0)

    return dice, min_det, fold_pct, warped_lbl


def main():
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"      syntx.greedy: FAST UNIDIRECTIONAL COMPOSITIVE REGISTRATION")
    print("=" * 80)
    print(f"Using device: {device.upper()}")

    # 1. Load Mindboggle mbhard dataset
    print("\n[1/3] Loading 'mbhard' benchmark dataset...")
    data = syntx.benchmark_data('mbhard')
    fi_raw, mi_raw = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)
    print(f"  Fixed shape : {fi.shape}, spacing: {fi.spacing}")
    print(f"  Moving shape: {mi.shape}, spacing: {mi.spacing}")

    # 2. Run syntx.greedy (Baseline)
    print("\n[2/4] Running baseline syntx.greedy (raw speed)...")
    reg_base = syntx.greedy(
        fixed=fi,
        moving=mi,
        reg_iterations=[100, 100, 50],
        scales=[4, 2, 1],
        learning_rate=0.4,
        flow_sigma=2.0,
        total_sigma=0.35,
        anderson=False,
        similarity_metric='lncc',
        verbose=False
    )
    dice_base, min_j_base, folds_base, _ = compute_cortical_dice_and_jacobian(
        fi, ml, fl, reg_base['fwdtransforms'][0]
    )

    # 3. Run syntx.greedy with Anderson Fold Suppression
    print("\n[3/4] Running syntx.greedy with Anderson Fold Suppression (anderson=True)...")
    reg_and = syntx.greedy(
        fixed=fi,
        moving=mi,
        reg_iterations=[100, 100, 50],
        scales=[4, 2, 1],
        learning_rate=0.4,
        flow_sigma=2.0,
        total_sigma=0.35,
        anderson=True,
        anderson_steps=5,
        return_inverse=True,
        similarity_metric='lncc',
        verbose=False
    )
    dice_and, min_j_and, folds_and, _ = compute_cortical_dice_and_jacobian(
        fi, ml, fl, reg_and['fwdtransforms'][0]
    )

    # 4. Results Comparison Table
    prov_b = reg_base['provenance']
    prov_a = reg_and['provenance']
    print("\n" + "=" * 80)
    print("                syntx.greedy FOLDING OPTIMIZATION COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Baseline Greedy':<20} | {'Anderson Greedy':<20}")
    print("-" * 80)
    print(f"{'Fixed Cortical DKT Dice':<30} | {dice_base:<20.4f} | {dice_and:<20.4f}")
    print(f"{'Minimum det(J)':<30} | {min_j_base:<20.4f} | {min_j_and:<20.4f}")
    print(f"{'Grid Folding %':<30} | {folds_base:<19.3f}% | {folds_and:<19.3f}%")
    print(f"{'Deformable Runtime (s)':<30} | {prov_b['runtime_deformable_sec']:<20.2f} | {prov_a['runtime_deformable_sec']:<20.2f}")
    print(f"{'Total Runtime (s)':<30} | {prov_b['runtime_total_sec']:<20.2f} | {prov_a['runtime_total_sec']:<20.2f}")
    print(f"{'Inverse Transforms Exported':<30} | {len(reg_base['invtransforms']):<20} | {len(reg_and['invtransforms']):<20}")
    fold_red = (folds_base - folds_and) / max(folds_base, 1e-6) * 100.0
    print("-" * 80)
    print(f"Folding Reduction: {fold_red:.1f}% reduction in grid folds with Anderson projection!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
