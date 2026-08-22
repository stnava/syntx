#!/usr/bin/env python3
"""
Rerun Losses from the 90-Pair Mindboggle Benchmark
===================================================
Evaluates the 4 loss pairs (Pairs 4, 48, 49, 62) using:
- Locked Canonical Affine Initialization
- ANTs C++ SyN Baseline
- syntx.syn (Eulerian, Antithetic Bootstrapping, flow_sigma=5.0 mm, MPS)
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import torch
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_harmonic_energy,
    compute_bending_energy
)

LOSS_PAIRS = [4, 48, 49, 62]

def normalize_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
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

def rerun_losses(
    loss_pairs=LOSS_PAIRS,
    flow_sigma=5.0,
    bootstrap_mode='antithetic',
    bootstrap_orig_weight=0.50,
    bootstrap_jitter_scale=0.25,
    output_csv="results/rerun_losses_summary.csv"
):
    os.makedirs("results", exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    df_pairs = pd.read_csv("examples/pairs.csv")
    print("=" * 90)
    print(f" RERUNNING MINDBOGGLE LOSS PAIRS: ANTITHETIC BOOTSTRAPPING (σ={flow_sigma} mm, Device: {device})")
    print(f" Target Pairs: {loss_pairs}")
    print("=" * 90 + "\n", flush=True)

    results = []

    for i, idx in enumerate(loss_pairs):
        row = df_pairs.iloc[idx]
        ptype = row['type']
        sub1, sub2 = row['subject1'], row['subject2']

        print(f"[{i+1}/{len(loss_pairs)}] Starting Pair {idx:02d} ({ptype.upper()}: {sub1} -> {sub2})...", flush=True)
        t_pair_start = time.time()

        p = load_mindboggle_pair(idx, "examples/pairs.csv")
        fi = normalize_intensity(p['fixed'])
        mi = normalize_intensity(p['moving'])
        fl, ml = p['fixed_label'], p['moving_label']
        fl_arr = fl.numpy()
        brain_mask = fl_arr > 0

        # Deterministic Robust Affine
        t_aff_0 = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
        aff_0 = reg_aff["fwdtransforms"][0]
        aff_inv = reg_aff["invtransforms"][0]
        t_aff = time.time() - t_aff_0

        # Initial Affine DICE
        _, _, dice_aff = compute_bidirectional_dice(fl, ml, fi, mi, [aff_0], [aff_inv])

        # 1. ANTs C++ SyN Baseline
        print(f"  -> Running ANTs C++ SyN baseline...", flush=True)
        t0 = time.time()
        res_ants = ants.registration(
            fixed=fi, moving=mi, type_of_transform="SyN",
            initial_transform=aff_0,
            syn_metric="CC", syn_sampling=2,
            reg_iterations=(100, 100, 20),
            flow_sigma=3.0, total_sigma=0.0, grad_step=0.25, verbose=False
        )
        t_ants = time.time() - t0

        dfix_ants, dmov_ants, dice_ants = compute_bidirectional_dice(
            fl, ml, fi, mi, res_ants["fwdtransforms"], res_ants["invtransforms"]
        )
        warp_ants = res_ants["fwdtransforms"][0]
        jac_ants = ants.create_jacobian_determinant_image(fi, warp_ants, do_log=False).numpy()
        fold_whole_ants = float(np.mean(jac_ants <= 0.0) * 100.0)
        fold_brain_ants = float(np.mean(jac_ants[brain_mask] <= 0.0) * 100.0)
        min_jac_whole_ants = float(np.min(jac_ants))
        min_jac_brain_ants = float(np.min(jac_ants[brain_mask]))
        harm_ants = compute_harmonic_energy(warp_ants, fi.spacing)
        bend_ants = compute_bending_energy(warp_ants, fi.spacing)

        # 2. syntx.syn with Antithetic Bootstrapping & flow_sigma=5.0
        print(f"  -> Running syntx.syn (Eulerian + Antithetic σ={flow_sigma} mm)...", flush=True)
        t0 = time.time()
        res_syntx = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device=device,
            grad_step=0.25, flow_sigma=flow_sigma, total_sigma=0.0,
            reg_iterations=[100, 100, 20], similarity_metric='cc2',
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=True, inverse_method='anderson',
            in_loop_inv_steps=10, formulation='eulerian', regularizer='gaussian',
            smooth_in_deformed_space=False, antisymmetric=True,
            bootstrap_mode=bootstrap_mode,
            bootstrap_orig_weight=bootstrap_orig_weight,
            bootstrap_jitter_scale=bootstrap_jitter_scale,
            verbose=False
        )
        t_syntx = time.time() - t0

        dfix_syntx, dmov_syntx, dice_syntx = compute_bidirectional_dice(
            fl, ml, fi, mi, res_syntx["fwdtransforms"], res_syntx["invtransforms"]
        )
        warp_syntx = res_syntx["fwdtransforms"][0]
        jac_syntx = ants.create_jacobian_determinant_image(fi, warp_syntx, do_log=False).numpy()
        fold_whole_syntx = float(np.mean(jac_syntx <= 0.0) * 100.0)
        fold_brain_syntx = float(np.mean(jac_syntx[brain_mask] <= 0.0) * 100.0)
        min_jac_whole_syntx = float(np.min(jac_syntx))
        min_jac_brain_syntx = float(np.min(jac_syntx[brain_mask]))
        harm_syntx = compute_harmonic_energy(warp_syntx, fi.spacing)
        bend_syntx = compute_bending_energy(warp_syntx, fi.spacing)

        gain_pct = (dice_syntx - dice_ants) * 100.0
        speedup = t_ants / max(t_syntx, 1e-3)
        win = 1 if dice_syntx > dice_ants + 1e-4 else (0 if dice_syntx < dice_ants - 1e-4 else 0.5)
        win_str = "WIN" if win == 1 else ("LOSS" if win == 0 else "TIE")

        rec = {
            "pair": idx,
            "type": ptype,
            "subject1": sub1,
            "subject2": sub2,
            "flow_sigma": flow_sigma,
            "boot_mode": bootstrap_mode,
            "w_orig": bootstrap_orig_weight,
            "dice_affine": dice_aff,
            "dice_ants": dice_ants,
            "dice_ants_fixed": dfix_ants,
            "dice_ants_moving": dmov_ants,
            "dice_syntx": dice_syntx,
            "dice_syntx_fixed": dfix_syntx,
            "dice_syntx_moving": dmov_syntx,
            "dice_gain_pct": gain_pct,
            "win": win_str,
            "fold_brain_ants_pct": fold_brain_ants,
            "fold_brain_syntx_pct": fold_brain_syntx,
            "fold_whole_ants_pct": fold_whole_ants,
            "fold_whole_syntx_pct": fold_whole_syntx,
            "min_jac_brain_ants": min_jac_brain_ants,
            "min_jac_brain_syntx": min_jac_brain_syntx,
            "harm_ants": harm_ants,
            "harm_syntx": harm_syntx,
            "bend_ants": bend_ants,
            "bend_syntx": bend_syntx,
            "time_affine_s": t_aff,
            "time_ants_s": t_ants,
            "time_syntx_s": t_syntx,
            "speedup": speedup,
            "time_total_s": time.time() - t_pair_start
        }
        results.append(rec)

        print(f"  * Pair {idx:02d} Results: syntx DICE = {dice_syntx:.4f} vs ANTs = {dice_ants:.4f} ({gain_pct:+5.2f}% [{win_str}]) | Brain Fold: {fold_brain_syntx:.5f}% (min detJ: {min_jac_brain_syntx:+.4f}) | Harm E: {harm_syntx:.4f} (ANTs: {harm_ants:.4f}) | syntx Time: {t_syntx:.1f}s ({speedup:.2f}x speedup)\n", flush=True)

        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)

    print("=" * 90)
    print(" ALL LOSS PAIRS RE-EVALUATION COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    rerun_losses()
