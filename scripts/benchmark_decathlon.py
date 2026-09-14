#!/usr/bin/env python3
"""
benchmark_decathlon.py — Automated Evaluation of syntx.auto_reg on Medical Segmentation Decathlon
================================================================================================

Evaluates auto_reg() across downloaded Decathlon tasks:
- Verifies autonomous modality and anatomy diagnosis
- Evaluates zero-effort registration accuracy (bidirectional label DICE)
- Evaluates topological quality (folding % and Jacobian metrics)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import syntx
from syntx.benchmark.msd import run_msd_task_benchmark, evaluate_msd_pair


def main():
    parser = argparse.ArgumentParser(description="Evaluate syntx.auto_reg across Medical Segmentation Decathlon tasks")
    parser.add_argument("--data-dir", type=str, default="/Users/stnava/data/decathlon",
                        help="Root directory where MSD tasks are stored")
    parser.add_argument("--tasks", nargs="+", type=str,
                        default=[
                            "Task01_BrainTumour", "Task02_Heart", "Task03_Liver",
                            "Task04_Hippocampus", "Task05_Prostate", "Task06_Lung",
                            "Task07_Pancreas", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"
                        ],
                        help="Tasks to evaluate (default: all 10 MSD tasks)")
    parser.add_argument("--num-pairs", type=int, default=2,
                        help="Number of evaluation pairs per task (default: 2)")
    parser.add_argument("--max-dimension", type=int, default=256,
                        help="Maximum spatial dimension for evaluation volumes (default: 256)")
    parser.add_argument("--output", type=str,
                        default="/Users/stnava/.gemini/antigravity-cli/brain/ed9af813-1310-43df-bc86-a32aeec400a0/scratch/decathlon_benchmark_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 85)
    print("SYNTX AUTOMATED DECATHLON BENCHMARK EVALUATION")
    print("=" * 85, flush=True)

    all_results = {}

    for task_name in args.tasks:
        task_dir = os.path.join(args.data_dir, task_name)
        if not os.path.exists(task_dir):
            print(f"Warning: Task directory not found: {task_dir}. Skipping.")
            continue

        meta_p = os.path.join(task_dir, "dataset.json")
        if not os.path.exists(meta_p):
            print(f"Warning: dataset.json not found in {task_dir}. Skipping.")
            continue

        with open(meta_p) as f:
            meta = json.load(f)

        n_train = len(meta.get("training", []))
        if n_train < 2:
            print(f"Not enough cases in {task_name} ({n_train} cases). Skipping.")
            continue

        # Form evaluation pairs: (case 0, case 1), (case 1, case 2), etc.
        pairs = []
        for i in range(min(args.num_pairs, n_train - 1)):
            pairs.append((i, i + 1))

        print(f"\nEvaluating {len(pairs)} pairs on {task_name} (Total cases: {n_train})...", flush=True)
        task_res = run_msd_task_benchmark(
            task_dir=task_dir,
            pairs=pairs,
            reg_iterations=[30, 15],
            affine_iterations=None,
            max_dimension=args.max_dimension
        )
        all_results[task_name] = task_res

    # Overall Combined Summary
    total_pairs = sum(len(res) for res in all_results.values())
    if total_pairs > 0:
        all_d0 = [r["initial_dice_sym"] for res in all_results.values() for r in res]
        all_df = [r["final_dice_sym"] for res in all_results.values() for r in res]
        all_dg = [r["dice_gain"] for res in all_results.values() for r in res]
        all_folds = [r["folding_pct"] for res in all_results.values() for r in res]
        all_times = [r["time_seconds"] for res in all_results.values() for r in res]

        print("\n" + "=" * 90)
        print("OVERALL MULTI-TASK DECATHLON BENCHMARK SUMMARY")
        print("=" * 90)
        print(f"Total Evaluated Pairs : {total_pairs}")
        print(f"Mean Initial DICE     : {np.mean(all_d0):.4f}")
        print(f"Mean Final DICE       : {np.mean(all_df):.4f} (Mean Gain: {np.mean(all_dg):+.4f})")
        print(f"Mean Folding %        : {np.mean(all_folds):.5f}%")
        print(f"Mean Registration Time: {np.mean(all_times):.1f}s")
        print(f"Overall Win Rate      : {sum(g > 0 for g in all_dg)}/{total_pairs} ({np.mean([g > 0 for g in all_dg])*100:.1f}%)")
        print("=" * 90, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full benchmark results to {args.output}")


if __name__ == "__main__":
    main()
