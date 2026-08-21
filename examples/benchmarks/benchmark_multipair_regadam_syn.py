#!/usr/bin/env python3
"""
Multi-Pair Mindboggle Benchmark: RegAdam vs CFL in discrete syntx.syn
====================================================================
Evaluates RegAdam (Gaussian, Sobolev, DST-I1) vs standard CFL SyN across
diverse Mindboggle pairs (intra-site and inter-site).
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import ants

import syntx
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.viz.reports import create_registration_report

def main():
    print("=" * 105, flush=True)
    print(" MULTI-PAIR MINDBOGGLE BENCHMARK: RegAdam in discrete syntx.syn ")
    print("=" * 105, flush=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    data_root = "/Users/stnava/data/mindboggle/volumes"
    
    # 4 Representative Benchmark Pairs
    PAIRS = [
        {
            "id": "Pair_77_mbhard",
            "type": "inter",
            "fi": os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-3/t1weighted_brain.MNI152.nii.gz"),
            "mi": os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-8/t1weighted_brain.MNI152.nii.gz"),
            "fl": os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-3/labels.DKT31.manual.MNI152.nii.gz"),
            "ml": os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-8/labels.DKT31.manual.MNI152.nii.gz"),
        },
        {
            "id": "Pair_00_OASIS",
            "type": "intra",
            "fi": os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.MNI152.nii.gz"),
            "mi": os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-2/t1weighted_brain.MNI152.nii.gz"),
            "fl": os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.MNI152.nii.gz"),
            "ml": os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-2/labels.DKT31.manual.MNI152.nii.gz"),
        },
        {
            "id": "Pair_08_NKI",
            "type": "intra",
            "fi": os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-1/t1weighted_brain.MNI152.nii.gz"),
            "mi": os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-2/t1weighted_brain.MNI152.nii.gz"),
            "fl": os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-1/labels.DKT31.manual.MNI152.nii.gz"),
            "ml": os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-2/labels.DKT31.manual.MNI152.nii.gz"),
        },
        {
            "id": "Pair_69_MMRR",
            "type": "intra",
            "fi": os.path.join(data_root, "MMRR-21_volumes/MMRR-21-1/t1weighted_brain.MNI152.nii.gz"),
            "mi": os.path.join(data_root, "MMRR-21_volumes/MMRR-21-2/t1weighted_brain.MNI152.nii.gz"),
            "fl": os.path.join(data_root, "MMRR-21_volumes/MMRR-21-1/labels.DKT31.manual.MNI152.nii.gz"),
            "ml": os.path.join(data_root, "MMRR-21_volumes/MMRR-21-2/labels.DKT31.manual.MNI152.nii.gz"),
        },
    ]

    CONFIGS = {
        "CFL_Gaussian_Baseline": dict(
            optimizer="cfl",
            regularizer="gaussian",
            grad_step=0.50
        ),
        "RegAdam_Gaussian": dict(
            optimizer="reg_adam",
            regularizer="gaussian",
            optimizer_lr=1.0,
            grad_step=0.50
        ),
        "RegAdam_Sobolev": dict(
            optimizer="reg_adam",
            regularizer="sobolev",
            sobolev_alpha=1.0,
            fast_smooth=False,
            optimizer_lr=1.0,
            grad_step=0.50
        ),
        "RegAdam_DSTI1": dict(
            optimizer="reg_adam",
            regularizer="dsti1",
            sobolev_alpha=1.0,
            fast_smooth=False,
            optimizer_lr=1.0,
            grad_step=0.50
        ),
    }

    all_rows = []

    for p_info in PAIRS:
        pair_id = p_info["id"]
        print(f"\n{'='*40} Evaluating {pair_id} ({p_info['type'].upper()}) {'='*40}", flush=True)

        if not os.path.exists(p_info["fi"]) or not os.path.exists(p_info["mi"]):
            print(f" [SKIP] Files not found for {pair_id}", flush=True)
            continue

        fi = ants.image_read(p_info["fi"])
        mi = ants.image_read(p_info["mi"])
        fl = ants.image_read(p_info["fl"])
        ml = ants.image_read(p_info["ml"])

        # Affine Initialization
        t0_aff = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
        t_aff = time.time() - t0_aff
        aff_tx = reg_aff["fwdtransforms"][0]

        d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=[aff_tx], invtransforms=[aff_tx],
            whichtoinvert_inv=[True]
        )
        print(f" [Affine Init] done in {t_aff:.2f}s | Baseline Sym DICE: {d_sym_aff:.4f}", flush=True)

        for cfg_name, cfg_kw in CONFIGS.items():
            base_kw = dict(
                fixed=fi,
                moving=mi,
                initial_transform=aff_tx,
                backend="pytorch",
                formulation="eulerian",
                inverse_method="anderson",
                in_loop_inv_steps=10,
                use_analytical_gradients=False,
                flow_sigma=3.0,
                total_sigma=0.0,
                reg_iterations=[100, 50, 10],
                levels=[4, 2, 1],
                syn_metric="lncc",
                syn_sampling=2,
                verbose=False
            )
            base_kw.update(cfg_kw)

            t0 = time.time()
            res = syntx.syn(**base_kw)
            t_run = time.time() - t0

            fwd_tx = res["fwdtransforms"]
            inv_tx = res["invtransforms"]
            which_inv = res.get("whichtoinvert_inv", [True, False])

            d_fix, d_mov, d_sym = compute_bidirectional_dice(
                fl=fl, ml=ml, fi=fi, mi=mi,
                fwdtransforms=fwd_tx, invtransforms=inv_tx,
                whichtoinvert_inv=which_inv
            )

            jac = compute_jacobian_metrics(fi, fwd_tx[0])
            fold_pct = float(jac.get("folding_pct", 0.0))
            min_jac = float(jac.get("min", 0.0))
            max_jac = float(jac.get("max", 0.0))

            row = {
                "Pair": pair_id,
                "Config": cfg_name,
                "Optimizer": cfg_kw.get("optimizer", "cfl"),
                "Regularizer": cfg_kw.get("regularizer", "gaussian"),
                "Sym DICE": d_sym,
                "Fix DICE": d_fix,
                "Mov DICE": d_mov,
                "Fold %": fold_pct,
                "Min det(J)": min_jac,
                "Max det(J)": max_jac,
                "Time (s)": t_run
            }
            all_rows.append(row)
            print(f"  -> {cfg_name:<24}: Sym DICE={d_sym:.4f} (Fix={d_fix:.4f}, Mov={d_mov:.4f}) | Fold={fold_pct:.4f}% | Min det(J)={min_jac:+.4f} | Time={t_run:.1f}s", flush=True)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Master Summary
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 115, flush=True)
    print(" MASTER SUMMARY: RegAdam vs CFL across Mindboggle Cohort ")
    print("=" * 115, flush=True)
    header = f"{'Pair':<18} {'Config':<24} {'Sym DICE':>9} {'Fix DICE':>9} {'Mov DICE':>9} {'Fold %':>9} {'Min det(J)':>11} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Pair']:<18} {r['Config']:<24} {r['Sym DICE']:>9.4f} {r['Fix DICE']:>9.4f} {r['Mov DICE']:>9.4f} {r['Fold %']:>9.4f}% {r['Min det(J)']:>11.4f} {r['Time (s)']:>6.1f}s", flush=True)
    print("=" * 115 + "\n", flush=True)

    # Average metrics per config
    print("--- AVERAGE METRICS ACROSS PAIRS ---", flush=True)
    grp = df.groupby("Config").agg({
        "Sym DICE": "mean",
        "Fix DICE": "mean",
        "Mov DICE": "mean",
        "Fold %": "mean",
        "Min det(J)": "mean",
        "Time (s)": "mean"
    }).reset_index()
    for _, g in grp.iterrows():
        print(f" {g['Config']:<25}: Mean Sym DICE = {g['Sym DICE']:.4f} | Mean Fold = {g['Fold %']:.4f}% | Mean Time = {g['Time (s)']:.1f}s", flush=True)
    print("=" * 115 + "\n", flush=True)

    # Save JSON summary
    out_json = "results/multipair_regadam_syn_benchmark_summary.json"
    os.makedirs("results", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Summary saved to: {out_json}", flush=True)

if __name__ == "__main__":
    main()
