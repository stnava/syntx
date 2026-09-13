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

    # 2. Run syntx.greedy
    print("\n[2/3] Running syntx.greedy registration (robust_affine + Eulerian compositive)...")
    t0 = time.time()
    reg = syntx.greedy(
        fixed=fi,
        moving=mi,
        reg_iterations=[100, 100, 50],
        scales=[4, 2, 1],
        learning_rate=0.4,
        flow_sigma=1.5,
        total_sigma=0.20,
        similarity_metric='lncc',
        verbose=True
    )
    total_time = time.time() - t0
    fwd_warp = reg['fwdtransforms'][0]

    # 3. Evaluate Metrics
    print("\n[3/3] Evaluating Cortical Overlap & Regularity...")
    dice, min_j, folds, _ = compute_cortical_dice_and_jacobian(fi, ml, fl, fwd_warp)

    prov = reg['provenance']
    print("\n" + "=" * 80)
    print("                      syntx.greedy RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Fixed Cortical DKT Dice : {dice:.4f}")
    print(f"  Minimum det(J)          : {min_j:.4f}")
    print(f"  Grid Folding %          : {folds:.3f}%")
    print(f"  Affine Time             : {prov['runtime_affine_sec']:.2f} s")
    print(f"  Deformable Time         : {prov['runtime_deformable_sec']:.2f} s")
    print(f"  Total Registration Time : {prov['runtime_total_sec']:.2f} s")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
