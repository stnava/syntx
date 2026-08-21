#!/usr/bin/env python3
"""
Evaluate Pair 69 across syntx.syn (RegAdam vs CFL) and syntx.tvf
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
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

def main():
    print("=" * 95, flush=True)
    print(" EVALUATING MINDBOGGLE PAIR 69 (OASIS-TRT-20-2 -> MMRR-21-18) ")
    print("=" * 95, flush=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    p = load_mindboggle_pair(69, "examples/pairs.csv")
    fi_raw, mi_raw = p['fixed'], p['moving']
    fl, ml = p['fixed_label'], p['moving_label']

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

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    # 1. Deterministic Multi-Start Robust Affine
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
    t_aff = time.time() - t0_aff
    aff_0 = reg_aff["fwdtransforms"][0]

    d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
        fl=fl, ml=ml, fi=fi, mi=mi,
        fwdtransforms=[aff_0], invtransforms=[aff_0],
        whichtoinvert_inv=[True]
    )
    print(f"\n[Affine Initializer] done in {t_aff:.2f}s | Sym DICE: {d_sym_aff:.4f} (Fix={d_fix_aff:.4f}, Mov={d_mov_aff:.4f})\n", flush=True)

    CONFIGS = {
        "1_CFL_Gaussian_Baseline": dict(
            model="syn",
            optimizer="cfl",
            regularizer="gaussian",
            grad_step=0.25,
            flow_sigma=3.0,
            reg_iterations=[100, 100, 20]
        ),
        "2_RegAdam_Gaussian_Syn": dict(
            model="syn",
            optimizer="reg_adam",
            regularizer="gaussian",
            optimizer_lr=1.0,
            grad_step=0.50,
            flow_sigma=3.0,
            reg_iterations=[100, 100, 20]
        ),
        "3_RegAdam_Sobolev_Syn": dict(
            model="syn",
            optimizer="reg_adam",
            regularizer="sobolev",
            sobolev_alpha=1.0,
            optimizer_lr=1.0,
            grad_step=0.50,
            flow_sigma=3.0,
            reg_iterations=[100, 100, 20]
        ),
        "4_RegAdam_DSTI1_Syn": dict(
            model="syn",
            optimizer="reg_adam",
            regularizer="dsti1",
            sobolev_alpha=1.0,
            optimizer_lr=1.0,
            grad_step=0.50,
            flow_sigma=3.0,
            reg_iterations=[100, 100, 20]
        ),
        "5_Dirichlet_Shield_TVF": dict(
            model="tvf",
            regularizer="dsti1",
            flow_sigma=1.0,
            total_sigma=0.035,
            dsti_alpha=0.035,
            optimizer="reg_adam",
            optimizer_lr=1.2,
            max_step_norm=0.50,
            multipoint_loss=[0.0, 0.5, 1.0],
            constant_speed=True,
            reg_iterations=[100, 100, 20]
        )
    }

    rows = []

    for name, cfg in CONFIGS.items():
        print(f"--- Running {name} ---", flush=True)
        model = cfg.pop("model")
        t0 = time.time()

        if model == "tvf":
            res = syntx.tvf(
                fixed=fi,
                moving=mi,
                initial_transform=aff_0,
                backend="pytorch",
                device=device,
                **cfg
            )
        else:
            res = syntx.syn(
                fixed=fi,
                moving=mi,
                initial_transform=aff_0,
                backend="pytorch",
                device=device,
                formulation="eulerian",
                inverse_method="anderson",
                in_loop_inv_steps=10,
                use_analytical_gradients=False,
                syn_metric="lncc",
                syn_sampling=2,
                total_sigma=0.0,
                levels=[4, 2, 1],
                **cfg
            )

        t_run = time.time() - t0
        fwd_tx = res["fwdtransforms"]
        inv_tx = res["invtransforms"]
        which_inv = res.get("whichtoinvert_inv", [True, False])

        d_fix, d_mov, d_sym = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=fwd_tx, invtransforms=inv_tx,
            whichtoinvert_inv=which_inv
        )

        warp_file = next((tx for tx in fwd_tx if isinstance(tx, str) and tx.endswith(('.nii', '.nii.gz'))), None)
        jac = compute_jacobian_metrics(fi, warp_file) if warp_file else {}
        fold_pct = float(jac.get("folding_pct", 0.0))
        min_jac = float(jac.get("min", 0.0))
        max_jac = float(jac.get("max", 0.0))

        row = {
            "Config": name,
            "Sym DICE": d_sym,
            "Fix DICE": d_fix,
            "Mov DICE": d_mov,
            "Fold %": fold_pct,
            "Min det(J)": min_jac,
            "Max det(J)": max_jac,
            "Time (s)": t_run
        }
        rows.append(row)
        print(f" >> {name:<26}: Sym DICE={d_sym:.4f} (Fix={d_fix:.4f}, Mov={d_mov:.4f}) | Fold={fold_pct:.4f}% | MinJ={min_jac:+.4f} | Time={t_run:.1f}s", flush=True)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n" + "=" * 105, flush=True)
    print(" MASTER SUMMARY: PAIR 69 REGADAM vs CFL vs TVF ")
    print("=" * 105, flush=True)
    header = f"{'Config':<28} {'Sym DICE':>9} {'Fix DICE':>9} {'Mov DICE':>9} {'Fold %':>9} {'Min det(J)':>11} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Config']:<28} {r['Sym DICE']:>9.4f} {r['Fix DICE']:>9.4f} {r['Mov DICE']:>9.4f} {r['Fold %']:>9.4f}% {r['Min det(J)']:>11.4f} {r['Time (s)']:>6.1f}s", flush=True)
    print("=" * 105 + "\n", flush=True)

if __name__ == "__main__":
    main()
