#!/usr/bin/env python3
"""
Parameter search for Sobolev and DST-I1 regularizers in syntx.syn on mbhard (Pair 77: OASIS-8 -> NKI-3).
Sweeps alpha values (0.1, 0.5, 1.0, 1.5, 3.0, 5.0), fast_smooth (True vs False), and grad_step (0.25 vs 0.50).
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import ants

import syntx
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

def main():
    print("=" * 95, flush=True)
    print(" PARAMETER SEARCH: Sobolev & DST-I1 in syntx.syn on mbhard (Pair 77) ")
    print("=" * 95, flush=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    fi_path = "/Users/stnava/data/mindboggle/volumes/NKI-TRT-20_volumes/NKI-TRT-20-3/t1weighted_brain.MNI152.nii.gz"
    mi_path = "/Users/stnava/data/mindboggle/volumes/OASIS-TRT-20_volumes/OASIS-TRT-20-8/t1weighted_brain.MNI152.nii.gz"
    fl_path = "/Users/stnava/data/mindboggle/volumes/NKI-TRT-20_volumes/NKI-TRT-20-3/labels.DKT31.manual.MNI152.nii.gz"
    ml_path = "/Users/stnava/data/mindboggle/volumes/OASIS-TRT-20_volumes/OASIS-TRT-20-8/labels.DKT31.manual.MNI152.nii.gz"

    fi = ants.image_read(fi_path)
    mi = ants.image_read(mi_path)
    fl = ants.image_read(fl_path)
    ml = ants.image_read(ml_path)

    # 1. Deterministic Multi-Start Robust Affine Initialization
    print("\n--- Running Multi-Start Robust Affine Initialization ---", flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
    t_aff = time.time() - t0_aff
    aff_tx = reg_aff["fwdtransforms"][0]

    d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
        fl=fl, ml=ml, fi=fi, mi=mi,
        fwdtransforms=[aff_tx], invtransforms=[aff_tx],
        whichtoinvert_inv=[True]
    )
    print(f"   => Affine done in {t_aff:.2f}s | Baseline Sym DICE: {d_sym_aff:.4f} (Fix: {d_fix_aff:.4f}, Mov: {d_mov_aff:.4f})\n", flush=True)

    def run_eval(name, **kwargs):
        base = dict(
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
            verbose=False
        )
        base.update(kwargs)

        t0_run = time.time()
        res = syntx.syn(**base)
        t_run = time.time() - t0_run

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

        row = {
            "Config": name,
            "Reg": base.get("regularizer", "gaussian"),
            "Alpha": base.get("sobolev_alpha", "default"),
            "FastSmooth": base.get("fast_smooth", False),
            "Step": base.get("grad_step", 0.25),
            "Sym DICE": d_sym,
            "Fix DICE": d_fix,
            "Mov DICE": d_mov,
            "Fold %": fold_pct,
            "Min det(J)": min_jac,
            "Time (s)": t_run
        }
        print(f" >> {name:<36}: Sym DICE={d_sym:.4f} (Fix={d_fix:.4f}, Mov={d_mov:.4f}) | Fold={fold_pct:.4f}% | MinJ={min_jac:+.4f} | Time={t_run:.1f}s", flush=True)
        return row

    results = []

    # 1. Baseline References
    print("--- 1. Baseline References ---", flush=True)
    results.append(run_eval("Gaussian_Standard_Step0.25", regularizer="gaussian", grad_step=0.25))
    results.append(run_eval("Gaussian_Relaxed_Step0.50", regularizer="gaussian", grad_step=0.50))

    # 2. Sobolev Alpha Sweep (Dual Smooth: fast_smooth=False, standard SyN mode)
    print("\n--- 2. Sobolev Alpha Sweep (Dual Mode: fast_smooth=False) ---", flush=True)
    for a in [0.5, 1.0, 1.5, 3.0, 5.0]:
        results.append(run_eval(f"Sobolev_Dual_Alpha{a}_Step0.25", regularizer="sobolev", sobolev_alpha=a, fast_smooth=False, grad_step=0.25))
        results.append(run_eval(f"Sobolev_Dual_Alpha{a}_Step0.50", regularizer="sobolev", sobolev_alpha=a, fast_smooth=False, grad_step=0.50))

    # 3. Sobolev Alpha Sweep (Pure Spectral: fast_smooth=True)
    print("\n--- 3. Sobolev Alpha Sweep (Pure Spectral: fast_smooth=True) ---", flush=True)
    for a in [1.5, 3.0, 5.0, 10.0]:
        results.append(run_eval(f"Sobolev_Pure_Alpha{a}_Step0.25", regularizer="sobolev", sobolev_alpha=a, fast_smooth=True, grad_step=0.25))

    # 4. DST-I1 Alpha Sweep (Dual & Pure)
    print("\n--- 4. DST-I1 Alpha Sweep ---", flush=True)
    for a in [0.5, 1.5, 3.0]:
        results.append(run_eval(f"DSTI1_Dual_Alpha{a}_Step0.25", regularizer="dsti1", sobolev_alpha=a, fast_smooth=False, grad_step=0.25))
        results.append(run_eval(f"DSTI1_Dual_Alpha{a}_Step0.50", regularizer="dsti1", sobolev_alpha=a, fast_smooth=False, grad_step=0.50))
        results.append(run_eval(f"DSTI1_Pure_Alpha{a}_Step0.25", regularizer="dsti1", sobolev_alpha=a, fast_smooth=True, grad_step=0.25))

    # Print Master Summary Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 115, flush=True)
    print(" MASTER SYNTX.SYN SPECTRAL PARAMETER SEARCH SUMMARY (mbhard: OASIS-8 -> NKI-3) ")
    print("=" * 115, flush=True)
    header = f"{'Config':<34} {'Reg':<10} {'Alpha':<8} {'Fast':<6} {'Step':<6} {'Sym DICE':>9} {'Fix DICE':>9} {'Mov DICE':>9} {'Fold %':>9} {'Min det(J)':>11} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Config']:<34} {r['Reg']:<10} {str(r['Alpha']):<8} {str(r['FastSmooth']):<6} {r['Step']:<6.2f} {r['Sym DICE']:>9.4f} {r['Fix DICE']:>9.4f} {r['Mov DICE']:>9.4f} {r['Fold %']:>9.4f} {r['Min det(J)']:>11.4f} {r['Time (s)']:>6.1f}s", flush=True)
    print("=" * 115 + "\n", flush=True)

    # Sort by Sym DICE
    df_sorted = df.sort_values(by="Sym DICE", ascending=False)
    print("--- Top 10 Configurations by Mean Symmetric DICE ---", flush=True)
    for rank, (_, r) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f" {rank:2d}. {r['Config']:<34} | DICE: {r['Sym DICE']:.4f} | Folds: {r['Fold %']:.4f}% | MinJ: {r['Min det(J)']:+.4f} | Time: {r['Time (s)']:.1f}s", flush=True)
    print("=" * 90 + "\n", flush=True)

if __name__ == "__main__":
    main()
