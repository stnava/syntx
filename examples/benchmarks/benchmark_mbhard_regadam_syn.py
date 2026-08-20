#!/usr/bin/env python3
"""
Benchmark: RegAdam vs CFL in syntx.syn on mbhard (Pair 77: OASIS-TRT-20-8 -> NKI-TRT-20-3)
Testing RegAdam (Gaussian, Sobolev, DST-I1) vs CFL Gaussian baseline.
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
from syntx.viz.reports import create_registration_report

def main():
    print("=" * 95, flush=True)
    print(" BENCHMARK: RegAdam in discrete syntx.syn on mbhard (Pair 77) ")
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

    def run_case(name, **kwargs):
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
            levels=[4, 2, 1],
            syn_metric="lncc",
            syn_sampling=2,
            verbose=False
        )
        base.update(kwargs)

        t0 = time.time()
        res = syntx.syn(**base)
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
            "Config": name,
            "Optimizer": base.get("optimizer", "cfl"),
            "Regularizer": base.get("regularizer", "gaussian"),
            "LR": base.get("optimizer_lr", "default"),
            "Step": base.get("grad_step", 0.50),
            "Sym DICE": d_sym,
            "Fix DICE": d_fix,
            "Mov DICE": d_mov,
            "Fold %": fold_pct,
            "Min det(J)": min_jac,
            "Max det(J)": max_jac,
            "Time (s)": t_run
        }
        print(f" >> {name:<32}: Sym DICE={d_sym:.4f} (Fix={d_fix:.4f}, Mov={d_mov:.4f}) | Fold={fold_pct:.4f}% | MinJ={min_jac:+.4f} | Time={t_run:.1f}s", flush=True)
        return row, res

    CONFIGS = {
        "1_CFL_Gaussian_Reference": dict(
            optimizer="cfl",
            regularizer="gaussian",
            grad_step=0.50
        ),
        "2_RegAdam_Gaussian": dict(
            optimizer="reg_adam",
            regularizer="gaussian",
            optimizer_lr=1.0,
            grad_step=0.50
        ),
        "3_RegAdam_Sobolev_Dual": dict(
            optimizer="reg_adam",
            regularizer="sobolev",
            sobolev_alpha=1.0,
            fast_smooth=False,
            optimizer_lr=1.0,
            grad_step=0.50
        ),
        "4_RegAdam_DSTI1_Dual": dict(
            optimizer="reg_adam",
            regularizer="dsti1",
            sobolev_alpha=1.0,
            fast_smooth=False,
            optimizer_lr=1.0,
            grad_step=0.50
        ),
    }

    results = []
    saved_outputs = {}

    print("--- Running mbhard Evaluation (Schedule [100, 50, 10], Step=0.50) ---", flush=True)
    for name, kw in CONFIGS.items():
        row, res = run_case(name, **kw)
        results.append(row)
        saved_outputs[name] = res

    # Master Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 115, flush=True)
    print(" MASTER mbhard (Pair 77) RESULTS: RegAdam vs CFL in discrete syntx.syn ")
    print("=" * 115, flush=True)
    header = f"{'Config':<30} {'Opt':<10} {'Reg':<10} {'LR':<6} {'Step':<5} {'Sym DICE':>9} {'Fix DICE':>9} {'Mov DICE':>9} {'Fold %':>9} {'Min det(J)':>11} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Config']:<30} {r['Optimizer']:<10} {r['Regularizer']:<10} {str(r['LR']):<6} {r['Step']:<5.2f} {r['Sym DICE']:>9.4f} {r['Fix DICE']:>9.4f} {r['Mov DICE']:>9.4f} {r['Fold %']:>9.4f}% {r['Min det(J)']:>11.4f} {r['Time (s)']:>6.1f}s", flush=True)
    print("=" * 115 + "\n", flush=True)

    # HTML Report for best config
    best_config_name = df.sort_values(by="Sym DICE", ascending=False).iloc[0]["Config"]
    best_res = saved_outputs[best_config_name]
    out_dir = "/Users/stnava/data/syntx/docs/reports"
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "mbhard_regadam_syn_comparison_report.html")

    print(f"Generating HTML report for best model ({best_config_name}) at: {html_path} ...", flush=True)
    create_registration_report(
        fixed=fi,
        moving=mi,
        reg=best_res,
        fixed_label=fl,
        moving_label=ml,
        output_html=html_path,
        title=f"mbhard: RegAdam vs CFL discrete SyN Benchmark (Best: {best_config_name})"
    )
    print(f"Report available at: {html_path}\n", flush=True)

if __name__ == "__main__":
    main()
