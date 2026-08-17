#!/usr/bin/env python
"""
Rerun Losses, Ties, and Sobolev Benchmark for Mindboggle
========================================================
Systematically re-runs:
- Losses: Pairs 11, 49, 67, 75, 86
- Tie: Pair 54
- Sobolev evaluation on target Mindboggle pairs
"""

import os
import sys
import time
import json
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("syntx.rerun")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_PAIR_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_single_pair.py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "docs", "provenance", "run_config.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "pairs_90", "syn_mps")
PAIRS_CSV = os.path.join(PROJECT_ROOT, "examples", "pairs.csv")

TARGET_PAIRS = [11, 49, 67, 75, 86, 54]

def run_single_pair_subprocess(pair_idx: int, config_override: dict = None, out_json: str = None) -> dict:
    if out_json is None:
        out_json = os.path.join(RESULTS_DIR, f"pair_{pair_idx:03d}_syn.json")
        
    cfg_file = CONFIG_PATH
    if config_override is not None:
        import tempfile
        fd, cfg_file = tempfile.mkstemp(prefix=f"cfg_p{pair_idx}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(config_override, f, indent=2)

    cmd = [
        sys.executable, SINGLE_PAIR_SCRIPT,
        "--pair-idx", str(pair_idx),
        "--model", "syn",
        "--device", "mps",
        "--config", cfg_file,
        "--out-json", out_json,
        "--pairs-csv", PAIRS_CSV,
    ]
    
    env = os.environ.copy()
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    t0 = time.time()
    logger.info(f"Running Pair {pair_idx:03d}...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=PROJECT_ROOT)
    dt = time.time() - t0
    
    for line in proc.stdout.strip().split("\n"):
        if line.startswith("DONE:") or line.startswith("FAIL:"):
            logger.info(f"  Worker: {line}")
            
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            res = json.load(f)
            res["elapsed"] = dt
            return res
    else:
        logger.error(f"  Worker failed: {proc.stderr[-300:]}")
        return {"status": "FAILED", "error": proc.stderr[-300:]}

def main():
    logger.info(f"Starting Re-runs of Losses and Ties: {TARGET_PAIRS}")
    
    # 1. Re-run the standard configuration on all losses and ties
    for p in TARGET_PAIRS:
        res = run_single_pair_subprocess(p)
        logger.info(f"Pair {p:03d} Standard Result: Dice={res.get('dice_sym')} | Fold={res.get('folding_pct')}%")
        
    # 2. Run Sobolev on Pair 67 and Pair 11
    sobolev_cfg = {
        "syn_config": {
            "grad_step": 0.25,
            "fluid_sigma": 3.0,
            "elastic_sigma": 0.0,
            "lncc_radius": 2,
            "inverse_steps": 10,
            "syn_metric": "cc2",
            "syn_regularizer": "sobolev",
            "kernel_type": "sobolev",
            "syn_fast_smooth": False,
            "syn_use_analytical_gradients": False,
            "syn_inverse_method": "anderson",
            "syn_formulation": "eulerian",
            "reg_iterations": [100, 100, 20],
            "n_starts": 3
        }
    }
    
    for sp_idx in [67, 11]:
        out_sob = os.path.join(RESULTS_DIR, f"pair_{sp_idx:03d}_syn_sobolev.json")
        res_sob = run_single_pair_subprocess(sp_idx, config_override=sobolev_cfg, out_json=out_sob)
        logger.info(f"Pair {sp_idx:03d} Sobolev Result: Dice={res_sob.get('dice_sym')} | Fold={res_sob.get('folding_pct')}%")

    logger.info("All re-runs and Sobolev experiments complete!")

if __name__ == "__main__":
    main()
