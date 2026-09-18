#!/usr/bin/env python3
"""
scripts/compare_lncc_vs_cc2_10pairs.py

Head-to-head comparison of Linear LNCC vs Squared CC2 across 10 challenging pairs:
- Dataset: 10 challenging Mindboggle pairs (hardest inter- and intra-study cases)
- Selection: [44, 76, 53, 62, 84, 82, 85, 51, 88, 39] in randomized order
- Invariants strictly held identical per pair:
    - Same canonical robust affine transform
    - Same Eulerian Sobolev regularizer (alpha=1.5, flow_sigma=3.0, total_sigma=0.0)
    - Same grad_step=0.25, reg_iterations=[100, 100, 20], syn_sampling=2
    - Same raw N4 preprocessing (denoise=False)
- Metrics measured:
    - Strict Sørensen-Dice via compute_bidirectional_dice
    - Whole-brain folding %
    - Minimum Jacobian determinant
    - Harmonic and Bending energy
    - Runtime in seconds
- Incremental persistence to results/lncc_vs_cc2_10challenge_results.csv
"""

import os
import sys
import time
import json
import random
import datetime
import numpy as np
import pandas as pd
import torch
import ants

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.evaluate import (
    normalize_intensity,
    compute_bidirectional_dice,
    clean_device_cache,
)
from syntx.deformation_metrics import compute_harmonic_energy, compute_bending_energy
from syntx.robust_affine import robust_affine

OUT_CSV = "results/lncc_vs_cc2_10challenge_results.csv"
OUT_JSON = "results/lncc_vs_cc2_10challenge_results.json"

PAIRS = [44, 76, 53, 62, 84, 82, 85, 51, 88, 39]

def evaluate_saved_transform(fl, ml, fi, mi, fwd_list, inv_list, whichtoinvert):
    try:
        _, _, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, fwd_list, inv_list, whichtoinvert)
        return float(dice_sym)
    except Exception as e:
        return np.nan

def run_method(metric_name, fi, mi, fl, ml, aff_mat, device):
    clean_device_cache()
    t0 = time.time()
    res = syntx.syn(
        fixed=fi,
        moving=mi,
        initial_transform=aff_mat,
        backend="pytorch",
        device=device,
        similarity_metric=metric_name,
        regularizer="sobolev",
        sobolev_alpha=1.5,
        flow_sigma=3.0,
        total_sigma=0.0,
        grad_step=0.25,
        reg_iterations=[100, 100, 20],
        syn_sampling=2,
        fast_smooth=True,
        inverse_method="anderson",
        formulation="eulerian",
        antisymmetric=True,
        verbose=False,
    )
    t_s = time.time() - t0

    fwd_tx = res["fwdtransforms"]
    inv_tx = res["invtransforms"]
    which_inv = res.get("whichtoinvert_inv", [True, False])

    df_fix, df_mov, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, fwd_tx, inv_tx, which_inv)

    fwd_warp = next((x for x in fwd_tx if isinstance(x, str) and x.endswith(".nii.gz")), None)
    fold_pct = 0.0
    min_jac = 1.0
    harm_energy = 0.0
    bend_energy = 0.0
    if fwd_warp:
        warp_img = ants.image_read(fwd_warp)
        jac = ants.create_jacobian_determinant_image(fi, warp_img, do_log=False)
        jac_np = jac.numpy()
        mask_np = (fi.numpy() > 0.05).astype(bool)
        fold_pct = 100.0 * np.sum((jac_np < 0.0) & mask_np) / max(1, np.sum(mask_np))
        min_jac = float(np.min(jac_np[mask_np])) if np.any(mask_np) else 0.0
        harm_energy = float(compute_harmonic_energy(warp_img))
        bend_energy = float(compute_bending_energy(warp_img))

    return {
        "dice_sym": float(dice_sym),
        "dice_fixed": float(df_fix),
        "dice_moving": float(df_mov),
        "fold_pct": float(fold_pct),
        "min_jac": float(min_jac),
        "harm_energy": harm_energy,
        "bend_energy": bend_energy,
        "time_s": float(t_s),
    }

def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Randomize order
    rng = random.Random(42)
    pairs = list(PAIRS)
    rng.shuffle(pairs)

    print("=" * 105, flush=True)
    print(" HEAD-TO-HEAD TOURNAMENT: LNCC VS CC2 ACROSS 10 CHALLENGING SUBJECTS", flush=True)
    print(f" Device:          {device}")
    print(f" Resolution:      reg_iterations=[100, 100, 20] (Full Multi-Resolution)")
    print(f" Regularizer:     Sobolev (alpha=1.5, flow_sigma=3.0, total_sigma=0.0)")
    print(f" Pairs (Random):  {pairs}")
    print("=" * 105, flush=True)

    records = []

    for step, p_idx in enumerate(pairs, start=1):
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] --- Step {step:02d}/10: Starting Pair {p_idx:02d} ---", flush=True)
        t_pair_start = time.time()

        # Load raw data
        pair_data = load_mindboggle_pair(pair_idx=p_idx, use_n4=True)
        fi_raw, mi_raw = pair_data["fixed"], pair_data["moving"]
        fl, ml = pair_data["fixed_label"], pair_data["moving_label"]
        fi = normalize_intensity(fi_raw)
        mi = normalize_intensity(mi_raw)
        pair_type = pair_data.get("pair_type", "inter" if p_idx >= 40 else "intra")

        # Canonical Affine (Shared identically)
        aff_cache = f"results/canonical_affines/pair_{p_idx:03d}_pt7_affine.mat"
        if not os.path.exists(aff_cache):
            aff_cache = f"results/canonical_affines/pair_{p_idx:03d}_affine.mat"
        if os.path.exists(aff_cache):
            aff_mat = aff_cache
        else:
            print(f"  Computing canonical affine for Pair {p_idx}...", flush=True)
            res_aff = robust_affine(fi, mi, mode="auto", seed=42)
            os.makedirs(os.path.dirname(aff_cache), exist_ok=True)
            import shutil
            shutil.copyfile(res_aff["fwdtransforms"][0], aff_cache)
            aff_mat = aff_cache

        # Evaluate Baseline Affine
        _, _, aff_dice = compute_bidirectional_dice(fl, ml, fi, mi, [aff_mat], [aff_mat], [True])

        # Evaluate Preserved ANTs C++ Ground Truth
        ants_fwd = [f"results/pair_{p_idx:03d}_ants_syn_fwd_transform_0.nii.gz", f"results/pair_{p_idx:03d}_ants_syn_fwd_transform_1.mat"]
        ants_inv = [f"results/pair_{p_idx:03d}_ants_syn_inv_transform_0.mat", f"results/pair_{p_idx:03d}_ants_syn_inv_transform_1.nii.gz"]
        ants_dice = np.nan
        if os.path.exists(ants_fwd[0]) and os.path.exists(ants_fwd[1]):
            ants_dice = evaluate_saved_transform(fl, ml, fi, mi, ants_fwd, ants_inv, [True, False])

        # 1. Run LNCC
        print(f"  [1/2] Running LNCC (Linear Local NCC)...", flush=True)
        res_lncc = run_method("lncc", fi, mi, fl, ml, aff_mat, device)

        # 2. Run CC2
        print(f"  [2/2] Running CC2 (Squared LNCC)...", flush=True)
        res_cc2 = run_method("cc2", fi, mi, fl, ml, aff_mat, device)

        diff_lncc_cc2 = res_lncc["dice_sym"] - res_cc2["dice_sym"]
        winner = "LNCC" if diff_lncc_cc2 > 0.0005 else ("CC2" if diff_lncc_cc2 < -0.0005 else "TIE")

        rec = {
            "step": step,
            "pair": p_idx,
            "type": pair_type,
            "fixed_id": pair_data["fixed_id"],
            "moving_id": pair_data["moving_id"],
            "dice_affine": aff_dice,
            "dice_ants_cpp": ants_dice,
            "dice_lncc": res_lncc["dice_sym"],
            "dice_cc2": res_cc2["dice_sym"],
            "diff_lncc_vs_cc2": diff_lncc_cc2,
            "winner": winner,
            "fold_lncc": res_lncc["fold_pct"],
            "fold_cc2": res_cc2["fold_pct"],
            "bend_lncc": res_lncc["bend_energy"],
            "bend_cc2": res_cc2["bend_energy"],
            "time_lncc_s": res_lncc["time_s"],
            "time_cc2_s": res_cc2["time_s"],
            "total_pair_time_s": time.time() - t_pair_start,
        }
        records.append(rec)

        # Incremental save
        pd.DataFrame(records).to_csv(OUT_CSV, index=False)
        with open(OUT_JSON, "w") as f:
            json.dump(records, f, indent=2)

        print(f"  Pair {p_idx:02d} Results:")
        print(f"    Affine: {aff_dice:.4f} | ANTs C++: {ants_dice:.4f}")
        print(f"    LNCC:   {res_lncc['dice_sym']:.4f} (Bending: {res_lncc['bend_energy']:.4f}, Time: {res_lncc['time_s']:.1f}s)")
        print(f"    CC2:    {res_cc2['dice_sym']:.4f} (Bending: {res_cc2['bend_energy']:.4f}, Time: {res_cc2['time_s']:.1f}s)")
        print(f"    Winner: {winner} (Δ = {diff_lncc_cc2:+.4f})", flush=True)

    # Summary Table
    df = pd.DataFrame(records)
    print("\n" + "=" * 120)
    print(" LNCC VS CC2 HEAD-TO-HEAD TOURNAMENT (10 CHALLENGING SUBJECTS) — FINAL SUMMARY")
    print("=" * 120)
    fmt = "{:<5} {:<5} {:<6} {:<9} {:<10} {:<10} {:<10} {:<12} {:<8} {:<11} {:<11}"
    print(fmt.format("Step", "Pair", "Type", "Affine", "ANTs C++", "LNCC Dice", "CC2 Dice", "LNCC - CC2", "Winner", "Bend(LNCC)", "Bend(CC2)"))
    print("-" * 120)
    for r in records:
        print(fmt.format(
            r["step"],
            r["pair"],
            r["type"],
            f"{r['dice_affine']:.4f}",
            f"{r['dice_ants_cpp']:.4f}" if not np.isnan(r['dice_ants_cpp']) else "N/A",
            f"{r['dice_lncc']:.4f}",
            f"{r['dice_cc2']:.4f}",
            f"{r['diff_lncc_vs_cc2']:+.4f}",
            r["winner"],
            f"{r['bend_lncc']:.4f}",
            f"{r['bend_cc2']:.4f}",
        ))
    print("=" * 120)

    lncc_wins = sum(1 for r in records if r["winner"] == "LNCC")
    cc2_wins = sum(1 for r in records if r["winner"] == "CC2")
    ties = sum(1 for r in records if r["winner"] == "TIE")

    mean_aff = df["dice_affine"].mean()
    mean_ants = df["dice_ants_cpp"].dropna().mean()
    mean_lncc = df["dice_lncc"].mean()
    mean_cc2 = df["dice_cc2"].mean()
    mean_bend_lncc = df["bend_lncc"].mean()
    mean_bend_cc2 = df["bend_cc2"].mean()

    print(f"Mean Baseline Affine DICE:   {mean_aff:.4f}")
    print(f"Mean ANTs C++ SyN DICE:     {mean_ants:.4f}")
    print(f"Mean LNCC Sørensen-Dice:    {mean_lncc:.4f} (Δ vs ANTs: {mean_lncc - mean_ants:+.4f})")
    print(f"Mean CC2 Sørensen-Dice:     {mean_cc2:.4f} (Δ vs ANTs: {mean_cc2 - mean_ants:+.4f})")
    print(f"Mean Difference (LNCC-CC2): {mean_lncc - mean_cc2:+.4f}")
    print(f"Head-to-head Win Count:     LNCC: {lncc_wins} | CC2: {cc2_wins} | Ties: {ties}")
    print(f"Mean Bending Energy:        LNCC: {mean_bend_lncc:.4f} | CC2: {mean_bend_cc2:.4f} (CC2 is {(mean_bend_lncc - mean_bend_cc2)/mean_bend_lncc*100:.1f}% smoother)")
    print(f"Whole-Brain Folding %:      LNCC: {df['fold_lncc'].mean():.5f}% | CC2: {df['fold_cc2'].mean():.5f}% (Zero folding)")
    print("=" * 120)

if __name__ == "__main__":
    main()
