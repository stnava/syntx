#!/usr/bin/env python3
"""
scripts/add_syngs_to_cohort_benchmark.py

Adds syntx.syngs (Geodesic Shooting) as the 4th Arm to the 90-Pair Mindboggle Benchmark:
- Reuses the identical canonical robust_affine transforms (seeded via evaluate_mindboggle_pair)
- Evaluates full multi-resolution pyramid reg_iterations=[100, 100, 20]
- Computes strict bidirectional Sørensen-Dice, Jacobian min/folding %
- Updates results/cohort_90pair_fullres_random_summary.csv incrementally
"""

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import torch

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import syntx
from syntx.benchmark.evaluate import evaluate_mindboggle_pair, clean_device_cache

CSV_PATH = "results/cohort_90pair_fullres_random_summary.csv"
REPORT_PATH = "docs/reports/cohort_90pair_4arms_final_report.md"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} does not exist.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print("=" * 80)
    print("ADDING SYNGS (GEODESIC SHOOTING) AS 4TH ARM TO 90-PAIR BENCHMARK")
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print("=" * 80, flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Ensure syngs columns exist
    for col in ["dice_syngs", "syngs_fold_brain_pct", "syngs_min_jac", "time_syngs_s", "diff_syngs_vs_ants"]:
        if col not in df.columns:
            df[col] = np.nan

    total_pairs = len(df)
    for idx, row in df.iterrows():
        p_idx = int(row["pair"])
        step = int(row["random_step"])

        if not np.isnan(row["dice_syngs"]):
            continue

        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] [Step {step:02d}/{total_pairs}: Pair {p_idx:02d}] Running syntx.syngs...", flush=True)
        t0 = time.time()

        res_syngs = evaluate_mindboggle_pair(
            pair_idx=p_idx,
            model="syngs",
            device=device,
            reg_iterations=[100, 100, 20],
            denoise=False,
            use_n4=True,
            verbose=False,
        )
        clean_device_cache()

        t_elapsed = time.time() - t0
        dice_syngs = res_syngs["syntx_dice_sym"]
        fold_syngs = res_syngs["syntx_fold"]
        jac_syngs = res_syngs["syntx_min_jac"]
        diff_vs_ants = dice_syngs - row["dice_ants_live"]

        df.at[idx, "dice_syngs"] = dice_syngs
        df.at[idx, "syngs_fold_brain_pct"] = fold_syngs
        df.at[idx, "syngs_min_jac"] = jac_syngs
        df.at[idx, "time_syngs_s"] = t_elapsed
        df.at[idx, "diff_syngs_vs_ants"] = diff_vs_ants

        # Save immediately
        df.to_csv(CSV_PATH, index=False)

        print(f"  syngs Dice: {dice_syngs:.4f} (vs ANTs: {diff_vs_ants:+.4f}) | Fold: {fold_syngs:.4f}% | Time: {t_elapsed:.1f}s", flush=True)

    print("\n" + "=" * 80)
    print("ALL 90 PAIRS EVALUATED FOR SYNGS!")
    print("=" * 80)

    # Generate comprehensive report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    n = len(df)
    win_syn = (df["dice_syn_current"] > df["dice_ants_live"]).sum()
    win_tvf = (df["dice_tvf_current"] > df["dice_ants_live"]).sum()
    win_syngs = (df["dice_syngs"] > df["dice_ants_live"]).sum()

    report = f"""# Full Population 90-Pair 4-Arm Mindboggle-101 Benchmark Report

**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Pairs:** {n} (40 intra-study, 50 inter-study)
**Seeding:** Single canonical affine seed (`syntx.robust_affine`, mode='auto') shared across all methods.

## Aggregate Results Across Full Cohort (N = {n})

| Method | Mean Dice | Median Dice | Std Dev | Win vs ANTs | Win Rate | Mean Time (s) | Mean Folding % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Robust Affine Seed** | {df['dice_affine'].mean():.4f} | {df['dice_affine'].median():.4f} | {df['dice_affine'].std():.4f} | — | — | {df['time_affine_s'].mean():.1f}s | — |
| **ANTs C++ SyN (Live)** | {df['dice_ants_live'].mean():.4f} | {df['dice_ants_live'].median():.4f} | {df['dice_ants_live'].std():.4f} | — | — | {df['time_ants_s'].mean():.1f}s | {df['ants_fold_brain_pct'].mean():.4f}% |
| **syntx.syn (Sobolev)** | **{df['dice_syn_current'].mean():.4f}** | **{df['dice_syn_current'].median():.4f}** | {df['dice_syn_current'].std():.4f} | {win_syn}/{n} | **{win_syn/n*100:.1f}%** | **{df['time_syn_s'].mean():.1f}s** | {df['syn_fold_brain_pct'].mean():.4f}% |
| **syntx.tvf (DSTI-1)** | **{df['dice_tvf_current'].mean():.4f}** | **{df['dice_tvf_current'].median():.4f}** | {df['dice_tvf_current'].std():.4f} | {win_tvf}/{n} | **{win_tvf/n*100:.1f}%** | {df['time_tvf_s'].mean():.1f}s | **{df['tvf_fold_brain_pct'].mean():.5f}%** |
| **syntx.syngs (Shooting)** | **{df['dice_syngs'].mean():.4f}** | **{df['dice_syngs'].median():.4f}** | {df['dice_syngs'].std():.4f} | {win_syngs}/{n} | **{win_syngs/n*100:.1f}%** | {df['time_syngs_s'].mean():.1f}s | {df['syngs_fold_brain_pct'].mean():.4f}% |

## Performance by Sub-Cohort

### Intra-Study (N = {(df['type'] == 'intra').sum()})
- **Affine**: {df[df['type'] == 'intra']['dice_affine'].mean():.4f}
- **ANTs C++**: {df[df['type'] == 'intra']['dice_ants_live'].mean():.4f}
- **syntx.syn**: {df[df['type'] == 'intra']['dice_syn_current'].mean():.4f} (+{df[df['type'] == 'intra']['dice_syn_current'].mean() - df[df['type'] == 'intra']['dice_ants_live'].mean():.4f})
- **syntx.tvf**: {df[df['type'] == 'intra']['dice_tvf_current'].mean():.4f} (+{df[df['type'] == 'intra']['dice_tvf_current'].mean() - df[df['type'] == 'intra']['dice_ants_live'].mean():.4f})
- **syntx.syngs**: {df[df['type'] == 'intra']['dice_syngs'].mean():.4f} (+{df[df['type'] == 'intra']['dice_syngs'].mean() - df[df['type'] == 'intra']['dice_ants_live'].mean():.4f})

### Inter-Study (N = {(df['type'] == 'inter').sum()})
- **Affine**: {df[df['type'] == 'inter']['dice_affine'].mean():.4f}
- **ANTs C++**: {df[df['type'] == 'inter']['dice_ants_live'].mean():.4f}
- **syntx.syn**: {df[df['type'] == 'inter']['dice_syn_current'].mean():.4f} (+{df[df['type'] == 'inter']['dice_syn_current'].mean() - df[df['type'] == 'inter']['dice_ants_live'].mean():.4f})
- **syntx.tvf**: {df[df['type'] == 'inter']['dice_tvf_current'].mean():.4f} (+{df[df['type'] == 'inter']['dice_tvf_current'].mean() - df[df['type'] == 'inter']['dice_ants_live'].mean():.4f})
- **syntx.syngs**: {df[df['type'] == 'inter']['dice_syngs'].mean():.4f} (+{df[df['type'] == 'inter']['dice_syngs'].mean() - df[df['type'] == 'inter']['dice_ants_live'].mean():.4f})
"""
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
