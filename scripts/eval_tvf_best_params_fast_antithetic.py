#!/usr/bin/env python3
"""
Evaluation Script: Peak TVF with Fast Antithetic Bootstrapping on Canonical mbhard (Pair 75)
===========================================================================================
Compares Peak DSTI1 Shield Baseline vs Fast Antithetic TVF using ONLY our best parameters:
- Regularizer: 'dsti1' (alpha=0.035, flow_sigma=1.0, total_sigma=0.035)
- Optimizer: 'reg_adam' (lr=1.2, max_step_norm=0.50, momentum=0.90)
- Loss: multipoint_loss=[0.0, 0.5, 1.0], syn_sampling=2
- Solver: 'euler', constant_speed=True, constant_speed_relaxation=0.10
- Full Peak Schedule: reg_iterations=[100, 100, 20]
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
from syntx.core.inverse import calculate_inverse_identity_error

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

def run_evaluation(output_csv="results/tvf_best_params_fast_antithetic_summary.csv"):
    os.makedirs("results", exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print("=" * 110)
    print(f" EVALUATING PEAK DSTI1 TVF + FAST ANTITHETIC ON CANONICAL MBHARD (PAIR 75, Device: {device})")
    print("=" * 110 + "\n", flush=True)

    print("1. Loading canonical mbhard (Pair 75: NKI-TRT-20-3 -> OASIS-TRT-20-8)...", flush=True)
    p = load_mindboggle_pair(75, "examples/pairs.csv")
    fi = normalize_intensity(p['fixed'])
    mi = normalize_intensity(p['moving'])
    fl = p['fixed_label']
    ml = p['moving_label']
    fl_arr = fl.numpy()
    brain_mask = fl_arr > 0

    print("2. Computing locked canonical affine transform (mode='auto')...", flush=True)
    t_aff_0 = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
    aff_fwd = reg_aff["fwdtransforms"][0]
    aff_inv = reg_aff["invtransforms"][0]
    t_aff = time.time() - t_aff_0

    dfix_aff, dmov_aff, dice_aff = compute_bidirectional_dice(fl, ml, fi, mi, [aff_fwd], [aff_inv])
    print(f"   Baseline Affine DICE: {dice_aff:.4f} (Fixed: {dfix_aff:.4f}, Moving: {dmov_aff:.4f}, Time: {t_aff:.2f}s)\n", flush=True)

    configs = [
        {
            "name": "TVF Peak DSTI1 Shield (Baseline)",
            "regularizer": "dsti1",
            "dsti_alpha": 0.035,
            "flow_sigma": 1.0,
            "total_sigma": 0.035,
            "optimizer": "reg_adam",
            "optimizer_lr": 1.2,
            "max_step_norm": 0.50,
            "multipoint_loss": [0.0, 0.5, 1.0],
            "antisymmetric": False,
            "solver": "euler",
            "constant_speed": True,
            "constant_speed_relaxation": 0.10,
            "cfl_momentum": 0.90,
            "fast_smooth": False,
            "bootstrap_mode": None,
            "bootstrap_orig_weight": 0.50,
            "bootstrap_jitter_scale": 0.25,
            "reg_iterations": [100, 100, 20]
        },
        {
            "name": "TVF Peak DSTI1 Shield + Fast Antithetic",
            "regularizer": "dsti1",
            "dsti_alpha": 0.035,
            "flow_sigma": 1.0,
            "total_sigma": 0.035,
            "optimizer": "reg_adam",
            "optimizer_lr": 1.2,
            "max_step_norm": 0.50,
            "multipoint_loss": [0.0, 0.5, 1.0],
            "antisymmetric": False,
            "solver": "euler",
            "constant_speed": True,
            "constant_speed_relaxation": 0.10,
            "cfl_momentum": 0.90,
            "fast_smooth": False,
            "bootstrap_mode": "antithetic",
            "bootstrap_orig_weight": 0.50,
            "bootstrap_jitter_scale": 0.25,
            "reg_iterations": [100, 100, 20]
        }
    ]

    results = []

    for idx, cfg in enumerate(configs):
        name = cfg['name']
        print("-" * 110)
        print(f"[{idx+1}/{len(configs)}] Running Arm {idx+1}: {name}...", flush=True)
        print(f"     Regularizer: {cfg['regularizer']} | Flow σ: {cfg['flow_sigma']} | Elastic σ: {cfg['total_sigma']} | Schedule: {cfg['reg_iterations']} | Bootstrap: {cfg['bootstrap_mode']}", flush=True)
        
        t0 = time.time()
        res_tvf = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_fwd,
            backend='pytorch',
            regularizer=cfg['regularizer'],
            dsti_alpha=cfg.get('dsti_alpha', 0.035),
            flow_sigma=cfg['flow_sigma'],
            total_sigma=cfg['total_sigma'],
            optimizer=cfg['optimizer'],
            optimizer_lr=cfg['optimizer_lr'],
            max_step_norm=cfg['max_step_norm'],
            multipoint_loss=cfg['multipoint_loss'],
            antisymmetric=cfg['antisymmetric'],
            solver=cfg['solver'],
            constant_speed=cfg['constant_speed'],
            constant_speed_relaxation=cfg['constant_speed_relaxation'],
            cfl_momentum=cfg['cfl_momentum'],
            fast_smooth=cfg['fast_smooth'],
            bootstrap_mode=cfg['bootstrap_mode'],
            bootstrap_orig_weight=cfg['bootstrap_orig_weight'],
            bootstrap_jitter_scale=cfg['bootstrap_jitter_scale'],
            reg_iterations=cfg['reg_iterations'],
            verbose=False
        )
        t_tvf = time.time() - t0

        dfix, dmov, dsym = compute_bidirectional_dice(
            fl, ml, fi, mi, res_tvf["fwdtransforms"], res_tvf["invtransforms"]
        )

        warp_fwd = res_tvf["fwdtransforms"][0]
        warp_inv = res_tvf["invtransforms"][1]

        jac = ants.create_jacobian_determinant_image(fi, warp_fwd, do_log=False).numpy()
        fold_whole = float(np.mean(jac <= 0.0) * 100.0)
        fold_brain = float(np.mean(jac[brain_mask] <= 0.0) * 100.0)
        min_jac_whole = float(np.min(jac))
        min_jac_brain = float(np.min(jac[brain_mask]))

        harm = compute_harmonic_energy(warp_fwd, fi.spacing)
        bend = compute_bending_energy(warp_fwd, fi.spacing)

        # Real Physical Inverse Identity Error
        w_fwd_arr = ants.image_read(warp_fwd).numpy()
        w_inv_arr = ants.image_read(warp_inv).numpy()
        w_fwd_t = torch.from_numpy(w_fwd_arr).float().squeeze()
        w_inv_t = torch.from_numpy(w_inv_arr).float().squeeze()
        if w_fwd_t.ndim == 4:
            w_fwd_zyx = w_fwd_t.permute(2, 1, 0, 3)
            w_inv_zyx = w_inv_t.permute(2, 1, 0, 3)
        else:
            w_fwd_zyx = w_fwd_t.permute(1, 0, 2)
            w_inv_zyx = w_inv_t.permute(1, 0, 2)

        inv_res = calculate_inverse_identity_error(w_fwd_zyx, w_inv_zyx, fi.spacing, fi.origin, fi.direction)
        mean_inv_err = inv_res['mean_error']
        max_inv_err = inv_res['max_error']
        p95_inv_err = float(np.percentile(inv_res['error_map'].cpu().numpy(), 95.0))

        rec = {
            "arm": idx + 1,
            "configuration": name,
            "regularizer": cfg['regularizer'],
            "flow_sigma": cfg['flow_sigma'],
            "total_sigma": cfg['total_sigma'],
            "bootstrap_mode": str(cfg['bootstrap_mode']),
            "dice_sym": dsym,
            "dice_fixed": dfix,
            "dice_moving": dmov,
            "fold_brain_pct": fold_brain,
            "fold_whole_pct": fold_whole,
            "min_jac_brain": min_jac_brain,
            "min_jac_whole": min_jac_whole,
            "harmonic_energy": harm,
            "bending_energy": bend,
            "mean_inv_err_mm": mean_inv_err,
            "p95_inv_err_mm": p95_inv_err,
            "max_inv_err_mm": max_inv_err,
            "runtime_s": t_tvf
        }
        results.append(rec)

        print(f"  --> Arm {idx+1} Completed: Sym DICE = {dsym:.4f} (Fix: {dfix:.4f}, Mov: {dmov:.4f}) | Brain Folds: {fold_brain:.5f}% (min detJ: {min_jac_brain:+.4f}) | Bnd: {bend:.4e} | Inv Err: {mean_inv_err:.3f}mm | Time: {t_tvf:.1f}s\n", flush=True)

        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)

    print("=" * 110)
    print(" SUMMARY OF PEAK TVF + FAST ANTITHETIC EXPERIMENT ON CANONICAL MBHARD (PAIR 75)")
    print("=" * 110)
    df_res = pd.DataFrame(results)
    print(df_res[["arm", "configuration", "bootstrap_mode", "dice_sym", "fold_brain_pct", "min_jac_brain", "bending_energy", "runtime_s"]].to_string(index=False))
    print("=" * 110)

if __name__ == "__main__":
    run_evaluation()
