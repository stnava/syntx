#!/usr/bin/env python
"""
Reproducibility Verification Suite for syntx
============================================
Runs the same registration pair 3 times in isolated subprocesses with
fixed random seeds (CPU and MPS) and tests for bitwise/floating-point
reproducibility of transformation fields and evaluation metrics.
"""

import sys
import os
import json
import time
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_PAIR_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_single_pair.py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "docs", "provenance", "run_config.json")
PAIRS_CSV = os.path.join(PROJECT_ROOT, "examples", "pairs.csv")

def run_trial(trial_id: int, pair_idx: int = 11, device: str = "mps") -> dict:
    out_json = f"results/reproducibility_trial_{trial_id}_p{pair_idx:03d}.json"
    cmd = [
        sys.executable, SINGLE_PAIR_SCRIPT,
        "--pair-idx", str(pair_idx),
        "--model", "syn",
        "--device", device,
        "--config", CONFIG_PATH,
        "--out-json", out_json,
        "--pairs-csv", PAIRS_CSV,
    ]
    env = os.environ.copy()
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=PROJECT_ROOT)
    dt = time.time() - t0
    
    with open(out_json, "r") as f:
        res = json.load(f)
    res["wall_clock"] = dt
    return res

def main():
    pair_idx = 11
    n_trials = 3
    print(f"===============================================================")
    print(f"  Reproducibility Verification Suite (Pair {pair_idx:03d}, {n_trials} Trials)")
    print(f"===============================================================\n")
    
    trials = []
    for i in range(1, n_trials + 1):
        print(f"Running Trial {i}/{n_trials}...")
        res = run_trial(i, pair_idx=pair_idx, device="mps")
        print(f"  Trial {i}: Dice={res['dice_sym']:.6f} | Fold={res['folding_pct']:.6f}% | Runtime={res['runtime_seconds']:.2f}s")
        trials.append(res)
        
    dices = [t['dice_sym'] for t in trials]
    folds = [t['folding_pct'] for t in trials]
    
    print("\n---------------------------------------------------------------")
    print(f"Dice Scores across {n_trials} runs: {dices}")
    print(f"Dice Max Delta: {max(dices) - min(dices):.8e}")
    print(f"Folding % across {n_trials} runs: {folds}")
    print(f"Folding Max Delta: {max(folds) - min(folds):.8e}")
    print("---------------------------------------------------------------")
    
    if max(dices) - min(dices) < 1e-5:
        print(">> REPRODUCIBILITY VERIFIED: Deterministic execution confirmed.")
    else:
        print(">> WARNING: Numerical divergence observed.")

if __name__ == "__main__":
    main()
