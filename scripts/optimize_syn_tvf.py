#!/usr/bin/env python
"""
scripts/optimize_syn_tvf.py — Controlled Parameter Optimization for SyN & TVF
==============================================================================

Systematically evaluates optimization and regularization variations for:
  1. syntx.syn (Eulerian Gaussian & Sobolev SyN)
  2. syntx.tvf (Time-Varying Velocity Field)

Strict Constraints:
  - Similarity metric is FIXED to 'cc2'
  - Multi-resolution schedule is FIXED to [100, 100, 20]
  - Topological folding percentage must NOT be compromised (folding <= baseline)

Evaluates on canonical calibration pairs:
  - Pair 000: OASIS Intra-subject
  - Pair 018: OASIS Intra-subject
  - Pair 044: Mindboggle Hard Inter-subject (mbhard)
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
import torch
import ants

from syntx.benchmark import evaluate_pair
from syntx.benchmark.config import get_model_config


def evaluate_config(pair_idx: int, model: str, config_overrides: dict, device: str = "mps"):
    """Evaluates a single pair with specific parameter overrides."""
    t0 = time.time()
    res = evaluate_pair(
        pair_idx=pair_idx,
        model=model,
        device=device,
        **config_overrides
    )
    t_el = time.time() - t0
    dice = res.get("dice_sym", res.get("syntx_dice_sym", 0.0))
    fold = res.get("folding_pct", res.get("syntx_fold", 0.0))
    min_jac = res.get("min_jacobian", res.get("syntx_min_jac", 0.0))
    return {
        "pair": pair_idx,
        "model": model,
        "dice": float(dice),
        "fold": float(fold),
        "min_jac": float(min_jac),
        "time": float(t_el),
        "params": config_overrides
    }


def run_syn_experiments(pairs=[0, 18, 44], device="mps"):
    print("=" * 80)
    print("STARTING SyN PARAMETER OPTIMIZATION SWEEPS")
    print("=" * 80)
    
    # Candidate configurations for SyN
    candidates = [
        {"name": "Baseline (grad_step=0.25, flow=3.0)", "model": "gaussian", "cfg": {}},
        {"name": "Step 0.35 (grad_step=0.35, flow=3.0)", "model": "gaussian", "cfg": {"grad_step": 0.35}},
        {"name": "Step 0.40 (grad_step=0.40, flow=3.0)", "model": "gaussian", "cfg": {"grad_step": 0.40}},
        {"name": "Flow 2.5 (grad_step=0.25, flow=2.5)", "model": "gaussian", "cfg": {"flow_sigma": 2.5}},
        {"name": "Step 0.35 + Flow 2.5", "model": "gaussian", "cfg": {"grad_step": 0.35, "flow_sigma": 2.5}},
        {"name": "Deformed Smooth (grad_step=0.25, deformed=True)", "model": "gaussian", "cfg": {"smooth_in_deformed_space": True}},
        {"name": "Sobolev Baseline (alpha=1.5, flow=3.0)", "model": "sobolev", "cfg": {}},
        {"name": "Sobolev Alpha 1.0 (sobolev_alpha=1.0)", "model": "sobolev", "cfg": {"sobolev_alpha": 1.0}},
        {"name": "Sobolev Alpha 2.0 (sobolev_alpha=2.0)", "model": "sobolev", "cfg": {"sobolev_alpha": 2.0}},
        {"name": "Sobolev Step 0.35 (grad_step=0.35)", "model": "sobolev", "cfg": {"grad_step": 0.35}},
    ]
    
    records = []
    for cand in candidates:
        name = cand["name"]
        model = cand["model"]
        cfg = cand["cfg"]
        print(f"\n--- Testing Candidate: {name} [{model.upper()}] ---")
        for p in pairs:
            out = evaluate_config(p, model, cfg, device=device)
            out["candidate"] = name
            records.append(out)
            print(f"  Pair {p:03d}: Dice={out['dice']:.5f} | Fold={out['fold']:.5f}% | Time={out['time']:.1f}s")
            
    df = pd.DataFrame(records)
    summary = df.groupby(["candidate", "model"]).agg({
        "dice": ["mean", "std"],
        "fold": "mean",
        "time": "mean"
    }).reset_index()
    summary.columns = ["Candidate", "Model", "Mean Dice", "Std Dice", "Mean Fold %", "Mean Time (s)"]
    summary = summary.sort_values(by="Mean Dice", ascending=False)
    print("\n" + "=" * 80)
    print("SyN SWEEP SUMMARY (RANKED BY MEAN DICE):")
    print("=" * 80)
    print(summary.to_string(index=False))
    return summary


def run_tvf_experiments(pairs=[0, 18, 44], device="mps"):
    print("=" * 80)
    print("STARTING TVF PARAMETER OPTIMIZATION SWEEPS")
    print("=" * 80)
    
    candidates = [
        {"name": "Baseline (lr=1.2, flow=1.0, total=0.035, mom=0.90)", "cfg": {}},
        {"name": "Momentum 0.95 (cfl_momentum=0.95)", "cfg": {"cfl_momentum": 0.95}},
        {"name": "LR 1.4 (optimizer_lr=1.4)", "cfg": {"optimizer_lr": 1.4}},
        {"name": "Flow 0.8 (tvf_flow_sigma=0.8)", "cfg": {"tvf_flow_sigma": 0.8}},
        {"name": "Total 0.02 (tvf_total_sigma=0.020)", "cfg": {"tvf_total_sigma": 0.020}},
        {"name": "Time Steps 4 (tvf_n_time_steps=4)", "cfg": {"tvf_n_time_steps": 4}},
        {"name": "Pyramid Flow Decay (flow=[1.5, 0.8, 0.2])", "cfg": {"tvf_flow_sigma": [1.5, 0.8, 0.2]}},
        {"name": "Relaxation 0.05 (relaxation=0.05)", "cfg": {"tvf_constant_speed_relaxation": 0.05}},
    ]
    
    records = []
    for cand in candidates:
        name = cand["name"]
        cfg = cand["cfg"]
        print(f"\n--- Testing TVF Candidate: {name} ---")
        for p in pairs:
            out = evaluate_config(p, "tvf", cfg, device=device)
            out["candidate"] = name
            records.append(out)
            print(f"  Pair {p:03d}: Dice={out['dice']:.5f} | Fold={out['fold']:.5f}% | Time={out['time']:.1f}s")
            
    df = pd.DataFrame(records)
    summary = df.groupby("candidate").agg({
        "dice": ["mean", "std"],
        "fold": "mean",
        "time": "mean"
    }).reset_index()
    summary.columns = ["Candidate", "Mean Dice", "Std Dice", "Mean Fold %", "Mean Time (s)"]
    summary = summary.sort_values(by="Mean Dice", ascending=False)
    print("\n" + "=" * 80)
    print("TVF SWEEP SUMMARY (RANKED BY MEAN DICE):")
    print("=" * 80)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["syn", "tvf", "all"], default="syn")
    parser.add_argument("--pairs", type=int, nargs="+", default=[0, 18, 44])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.model in ("syn", "all"):
        run_syn_experiments(pairs=args.pairs, device=args.device)
    if args.model in ("tvf", "all"):
        run_tvf_experiments(pairs=args.pairs, device=args.device)


if __name__ == "__main__":
    main()
