#!/usr/bin/env python3
"""
Comprehensive 10-Pair Parameter Sweep Benchmark Suite
=====================================================

Evaluates all relevant combinations of registration parameters across 10 randomly
sampled Mindboggle benchmark pairs (5 intra-subject, 5 inter-subject):
  - Models: TVF, Sobolev SyN, Gaussian SyN, Greedy RegAdam, Greedy Adam
  - Similarity Metrics: BoxLNCC (squared), CC2, LNCC (linear), Mattes MI
  - Regularizers: Gaussian (sigma=3.0, 2.0), Sobolev FFT (alpha=1.5, 1.0), DSTI-1 DCT, RegAdam
  - Optimizers & Step Sizes: CFL (0.15, 0.25, 0.35), RegAdam (lr=1.0)
  - Preprocessing: Standard vs ANTsTorch Adaptive Non-Local Means Denoising (`antstorch.denoise_image`)

Invariants strictly preserved:
  - Multi-resolution levels & iterations constant: [100, 100, 20]
  - Native space evaluation (Single Interpolation Policy)
  - Process isolation: Each registration executed in a dedicated subprocess
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, "results", "parameter_sweep_10pairs")
REPORT_HTML = os.path.join(REPO_ROOT, "docs", "parameter_sweep_10pairs_report.html")
SUMMARY_CSV = os.path.join(OUT_DIR, "sweep_summary.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "sweep_summary.json")

# 10 Representative sampled pairs (5 intra, 5 inter) with deterministic seed 42
PAIRS_10 = [0, 10, 22, 26, 39, 40, 44, 55, 65, 70]

# Multi-dimensional parameter sweep definitions
# (reg_iterations=[100, 100, 20] strictly held constant across all configurations)
SWEEP_CONFIGS = [
    # -------------------------------------------------------------
    # AXIS 1: MODEL ARCHITECTURES (Baseline BoxLNCC / CC2)
    # -------------------------------------------------------------
    {
        "id": "model_tvf",
        "name": "TVF (Time-Varying Field + DSTI-1)",
        "dimension": "Model Architecture",
        "model": "tvf",
        "args": ["--model", "tvf"],
    },
    {
        "id": "model_sobolev",
        "name": "Sobolev SyN (Eulerian FFT)",
        "dimension": "Model Architecture",
        "model": "sobolev",
        "args": ["--model", "sobolev"],
    },
    {
        "id": "model_gaussian",
        "name": "Gaussian SyN (Eulerian)",
        "dimension": "Model Architecture",
        "model": "gaussian",
        "args": ["--model", "gaussian"],
    },
    {
        "id": "model_greedy_regadam",
        "name": "Greedy RegAdam",
        "dimension": "Model Architecture",
        "model": "greedy_regadam",
        "args": ["--model", "greedy_regadam"],
    },
    {
        "id": "model_greedy_adam",
        "name": "Greedy Adam",
        "dimension": "Model Architecture",
        "model": "greedy",
        "args": ["--model", "greedy"],
    },

    # -------------------------------------------------------------
    # AXIS 2: SIMILARITY METRICS (SyN Gaussian)
    # -------------------------------------------------------------
    {
        "id": "metric_box_lncc",
        "name": "SyN + BoxLNCC (Squared CC)",
        "dimension": "Similarity Metric",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--similarity-metric", "box_lncc"],
    },
    {
        "id": "metric_cc2",
        "name": "SyN + CC2 (Pseudo-Gradient Squared LNCC)",
        "dimension": "Similarity Metric",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--similarity-metric", "cc2"],
    },
    {
        "id": "metric_lncc_linear",
        "name": "SyN + LNCC (Linear Unsquared)",
        "dimension": "Similarity Metric",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--similarity-metric", "lncc"],
    },
    {
        "id": "metric_mattes_mi",
        "name": "SyN + Mattes Mutual Information",
        "dimension": "Similarity Metric",
        "model": "syn_mi",
        "args": ["--model", "syn_mi"],
    },

    # -------------------------------------------------------------
    # AXIS 3: REGULARIZATION OPERATORS (SyN)
    # -------------------------------------------------------------
    {
        "id": "reg_gaussian_sig3",
        "name": "SyN + Gaussian (sigma=3.0)",
        "dimension": "Regularization",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--flow-sigma", "3.0"],
    },
    {
        "id": "reg_gaussian_sig2",
        "name": "SyN + Gaussian (sigma=2.0)",
        "dimension": "Regularization",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--flow-sigma", "2.0"],
    },
    {
        "id": "reg_sobolev_a15",
        "name": "SyN + Sobolev FFT (alpha=1.5)",
        "dimension": "Regularization",
        "model": "sobolev",
        "args": ["--model", "sobolev"],
    },
    {
        "id": "reg_dsti1",
        "name": "SyN + DSTI-1 DCT",
        "dimension": "Regularization",
        "model": "syn_dsti1",
        "args": ["--model", "syn_dsti1"],
    },

    # -------------------------------------------------------------
    # AXIS 4: OPTIMIZERS & GRADIENT STEP SIZES (SyN)
    # -------------------------------------------------------------
    {
        "id": "opt_cfl_015",
        "name": "SyN + CFL Step 0.15 (Conservative)",
        "dimension": "Optimizer & Step",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--grad-step", "0.15"],
    },
    {
        "id": "opt_cfl_025",
        "name": "SyN + CFL Step 0.25 (Standard)",
        "dimension": "Optimizer & Step",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--grad-step", "0.25"],
    },
    {
        "id": "opt_cfl_035",
        "name": "SyN + CFL Step 0.35 (Aggressive)",
        "dimension": "Optimizer & Step",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--grad-step", "0.35"],
    },
    {
        "id": "opt_regadam",
        "name": "SyN + RegAdam Quotient Optimizer",
        "dimension": "Optimizer & Step",
        "model": "syn_regadam",
        "args": ["--model", "syn_regadam", "--learning-rate", "1.0"],
    },

    # -------------------------------------------------------------
    # AXIS 5: PREPROCESSING — ANTsTorch Adaptive Denoising
    # -------------------------------------------------------------
    {
        "id": "denoise_sobolev",
        "name": "Sobolev SyN + Denoised",
        "dimension": "Preprocessing (Denoise)",
        "model": "sobolev",
        "args": ["--model", "sobolev", "--denoise"],
    },
    {
        "id": "denoise_tvf",
        "name": "TVF + Denoised",
        "dimension": "Preprocessing (Denoise)",
        "model": "tvf",
        "args": ["--model", "tvf", "--denoise"],
    },
    {
        "id": "denoise_gaussian",
        "name": "Gaussian SyN + Denoised",
        "dimension": "Preprocessing (Denoise)",
        "model": "gaussian",
        "args": ["--model", "gaussian", "--denoise"],
    },
    {
        "id": "denoise_greedy_regadam",
        "name": "Greedy RegAdam + Denoised",
        "dimension": "Preprocessing (Denoise)",
        "model": "greedy_regadam",
        "args": ["--model", "greedy_regadam", "--denoise"],
    },
]


def run_parameter_sweep():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_HTML), exist_ok=True)

    print("=" * 86)
    print("  SYNTX 10-PAIR MULTI-DIMENSIONAL PARAMETER SWEEP")
    print("  Evaluating Models, Metrics, Regularizers, Optimizers, and Denoising Preprocessing")
    print(f"  Target Sampled Pairs (10): {PAIRS_10}")
    print(f"  Total Configurations per Pair: {len(SWEEP_CONFIGS)}")
    print(f"  Total Evaluations: {len(PAIRS_10) * len(SWEEP_CONFIGS)}")
    print("=" * 86, flush=True)

    results = []
    t0_all = time.time()
    total_runs = len(PAIRS_10) * len(SWEEP_CONFIGS)
    run_idx = 0

    for pair_idx in PAIRS_10:
        pair_type = "intra" if pair_idx < 40 else "inter"
        pair_label = f"Pair {pair_idx:02d} ({pair_type})"

        for cfg in SWEEP_CONFIGS:
            run_idx += 1
            cfg_id = cfg["id"]
            cfg_name = cfg["name"]
            dimension = cfg["dimension"]
            m_name = cfg["model"]
            is_denoised = "--denoise" in cfg["args"]

            out_suffix = f"{cfg_id}"
            json_file = os.path.join(OUT_DIR, f"pair_{pair_idx:03d}_{out_suffix}.json")

            # Check cache
            if os.path.exists(json_file):
                try:
                    with open(json_file, "r") as f:
                        rec = json.load(f)
                    if rec.get("status") == "SUCCESS":
                        dice_val = rec.get("syntx_dice_sym", float("nan"))
                        fold_val = rec.get("syntx_fold", float("nan"))
                        min_j = rec.get("syntx_min_j", float("nan"))
                        time_val = rec.get("syntx_time", 0.0)
                        ants_dice = rec.get("ants_baseline", {}).get("dice_sym", float("nan"))
                        diff_ants = (dice_val - ants_dice) * 100.0 if np.isfinite(ants_dice) else float("nan")

                        print(f"[{run_idx:03d}/{total_runs:03d}] ⚡ CACHE: {pair_label} | {cfg_name:38s} | Dice: {dice_val:.4f} | Folds: {fold_val:.4f}% | vs ANTs: {diff_ants:+.2f}%", flush=True)
                        results.append({
                            "config_id": cfg_id,
                            "config_name": cfg_name,
                            "dimension": dimension,
                            "pair_idx": pair_idx,
                            "pair_type": pair_type,
                            "pair_label": pair_label,
                            "status": "SUCCESS",
                            "dice_sym": dice_val,
                            "folding_pct": fold_val,
                            "min_j": min_j,
                            "runtime_s": time_val,
                            "ants_dice": ants_dice,
                            "diff_vs_ants": diff_ants,
                            "denoised": is_denoised,
                        })
                        continue
                except Exception:
                    pass

            print(f"\n[{run_idx:03d}/{total_runs:03d}] Running {pair_label} | {cfg_name}...", flush=True)

            cmd = [
                sys.executable, "-u", "-m", "syntx.benchmark.cli",
                "--pair-idx", str(pair_idx),
                "--out-dir", OUT_DIR,
                "--out-name", cfg_id,
                "--reg-iterations", "100", "100", "20",
            ] + cfg["args"]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0

            if proc.returncode != 0:
                print(f"  ❌ ERROR (exit {proc.returncode}): {proc.stderr[:300]}", flush=True)
                results.append({
                    "config_id": cfg_id,
                    "config_name": cfg_name,
                    "dimension": dimension,
                    "pair_idx": pair_idx,
                    "pair_type": pair_type,
                    "pair_label": pair_label,
                    "status": "FAILED",
                    "dice_sym": float("nan"),
                    "folding_pct": float("nan"),
                    "min_j": float("nan"),
                    "runtime_s": elapsed,
                    "ants_dice": float("nan"),
                    "diff_vs_ants": float("nan"),
                    "denoised": is_denoised,
                })
            else:
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        rec = json.load(f)
                    dice_val = rec.get("syntx_dice_sym", float("nan"))
                    fold_val = rec.get("syntx_fold", float("nan"))
                    min_j = rec.get("syntx_min_j", float("nan"))
                    time_val = rec.get("syntx_time", elapsed)
                    ants_dice = rec.get("ants_baseline", {}).get("dice_sym", float("nan"))
                    diff_ants = (dice_val - ants_dice) * 100.0 if np.isfinite(ants_dice) else float("nan")

                    print(f"  ✅ Dice: {dice_val:.4f} | Folds: {fold_val:.4f}% | Time: {time_val:.1f}s | vs ANTs: {diff_ants:+.2f}%", flush=True)
                    results.append({
                        "config_id": cfg_id,
                        "config_name": cfg_name,
                        "dimension": dimension,
                        "pair_idx": pair_idx,
                        "pair_type": pair_type,
                        "pair_label": pair_label,
                        "status": "SUCCESS",
                        "dice_sym": dice_val,
                        "folding_pct": fold_val,
                        "min_j": min_j,
                        "runtime_s": time_val,
                        "ants_dice": ants_dice,
                        "diff_vs_ants": diff_ants,
                        "denoised": is_denoised,
                    })

            # Intermediate save
            df_curr = pd.DataFrame(results)
            df_curr.to_csv(SUMMARY_CSV, index=False)

    total_time = time.time() - t0_all
    print("\n" + "=" * 86)
    print(f"  SWEEP COMPLETE in {total_time/60.0:.1f} minutes")
    print("=" * 86, flush=True)

    with open(SUMMARY_JSON, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_runs": len(results),
            "sampled_pairs": PAIRS_10,
            "total_runtime_minutes": total_time / 60.0,
            "results": results,
        }, f, indent=2)

    generate_sweep_html_report(results, SUMMARY_JSON, REPORT_HTML)


def generate_sweep_html_report(results, summary_json, output_html):
    df = pd.DataFrame(results)
    if df.empty:
        return

    agg_df = df[df["status"] == "SUCCESS"].groupby(["dimension", "config_name"]).agg(
        mean_dice=("dice_sym", "mean"),
        std_dice=("dice_sym", "std"),
        mean_fold=("folding_pct", "mean"),
        max_fold=("folding_pct", "max"),
        mean_time=("runtime_s", "mean"),
        mean_diff_ants=("diff_vs_ants", "mean"),
        win_rate=("diff_vs_ants", lambda x: (x > 0).mean() * 100.0),
        n_pairs=("dice_sym", "count"),
    ).reset_index()

    agg_df = agg_df.sort_values(by=["dimension", "mean_dice"], ascending=[True, False])

    table_rows = []
    for _, r in agg_df.iterrows():
        diff_str = f"{r['mean_diff_ants']:+.2f}%"
        diff_color = "#10b981" if r['mean_diff_ants'] > 0 else "#ef4444"
        win_color = "#10b981" if r['win_rate'] >= 50.0 else "#ef4444"
        table_rows.append(f"""
        <tr>
            <td style="font-weight: 600; color: #6366f1;">{r['dimension']}</td>
            <td style="font-weight: 500;">{r['config_name']}</td>
            <td style="text-align: right; font-weight: 700; color: #1e293b;">{r['mean_dice']:.4f} &plusmn; {r['std_dice']:.4f}</td>
            <td style="text-align: right; color: {'#10b981' if r['mean_fold'] < 0.05 else '#f59e0b'};">{r['mean_fold']:.4f}% (max {r['max_fold']:.4f}%)</td>
            <td style="text-align: right;">{r['mean_time']:.1f}s</td>
            <td style="text-align: right; font-weight: 700; color: {diff_color};">{diff_str}</td>
            <td style="text-align: right; font-weight: 700; color: {win_color};">{r['win_rate']:.1f}%</td>
            <td style="text-align: center;">{int(r['n_pairs'])}/10</td>
        </tr>
        """)

    pair_table_rows = []
    for _, r in df.iterrows():
        diff_val = r.get("diff_vs_ants", float("nan"))
        diff_str = f"{diff_val:+.2f}%" if np.isfinite(diff_val) else "N/A"
        diff_color = "#10b981" if diff_val > 0 else "#ef4444"
        pair_table_rows.append(f"""
        <tr>
            <td>Pair {r['pair_idx']:02d} ({r['pair_type']})</td>
            <td>{r['dimension']}</td>
            <td>{r['config_name']}</td>
            <td style="text-align: right; font-weight: 600;">{r['dice_sym']:.4f}</td>
            <td style="text-align: right;">{r['folding_pct']:.4f}%</td>
            <td style="text-align: right;">{r['runtime_s']:.1f}s</td>
            <td style="text-align: right; color: {diff_color};">{diff_str}</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Syntx 10-Pair Parameter Sweep Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 32px; background: #f8fafc; color: #0f172a; }}
        .header {{ max-width: 1200px; margin: 0 auto 32px; padding: 32px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); }}
        .header h1 {{ margin: 0 0 12px; font-size: 30px; font-weight: 800; letter-spacing: -0.025em; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 15px; max-width: 800px; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 16px; padding: 28px; margin-bottom: 32px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; }}
        h2 {{ margin: 0 0 20px; font-size: 20px; font-weight: 700; color: #1e293b; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th {{ background: #f1f5f9; color: #475569; font-weight: 600; text-align: left; padding: 12px 14px; border-bottom: 2px solid #cbd5e1; }}
        td {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
        tr:hover td {{ background: #f8fafc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Syntx 10-Pair Multi-Dimensional Parameter Sweep Report</h1>
        <p>Thorough evaluation across Models (TVF, Sobolev, Gaussian, Greedy), Similarity Metrics (BoxLNCC, CC2, LNCC, Mattes MI), Regularizers (Sobolev, DSTI-1, Gaussian), Optimizers, and ANTsTorch Denoising Preprocessing across 10 representative Mindboggle pairs.</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>Summary Across Dimensions (10 Sampled Pairs)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Dimension</th>
                        <th>Configuration</th>
                        <th style="text-align: right;">Mean Dice &plusmn; Std</th>
                        <th style="text-align: right;">Folding %</th>
                        <th style="text-align: right;">Runtime</th>
                        <th style="text-align: right;">vs ANTs C++</th>
                        <th style="text-align: right;">Win Rate</th>
                        <th style="text-align: center;">Evaluated</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Detailed Per-Pair Evaluation Log</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pair</th>
                        <th>Dimension</th>
                        <th>Configuration</th>
                        <th style="text-align: right;">Dice Sym</th>
                        <th style="text-align: right;">Folding %</th>
                        <th style="text-align: right;">Runtime</th>
                        <th style="text-align: right;">vs ANTs</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(pair_table_rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"[generate_sweep_html_report] Interactive report generated at: {output_html}")


if __name__ == "__main__":
    run_parameter_sweep()
