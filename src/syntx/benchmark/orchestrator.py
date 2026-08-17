"""
syntx.benchmark.orchestrator — Mindboggle Population Benchmark Orchestrator
===========================================================================

Orchestrates multi-pair and full 90-pair randomized benchmarks with strict
process isolation, progress streaming, and automated HTML dashboard compilation.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
from typing import List, Dict, Any, Optional, Set

from syntx.benchmark.data import check_mindboggle_data, DEFAULT_PAIRS_CSV


def run_mindboggle_benchmark(
    pairs: Optional[List[int]] = None,
    model: str = "sobolev",
    probe_pairs: Optional[Set[int]] = None,
    random_order: bool = True,
    seed: int = 42,
    out_dir: str = "results/reproducible_eval",
    summary_json: str = "results/reproducible_90pair_master_summary.json",
    report_html: str = "docs/reproducible_90pair_report.html",
    generate_example_reports: bool = False,
    example_report_pairs: Optional[List[int]] = None,
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes a comprehensive Mindboggle benchmark with isolated subprocesses.

    Parameters
    ----------
    pairs : List[int], optional
        List of pair indices to evaluate. If None, runs all 90 pairs [0..89].
    model : str
        Primary registration model ('sobolev', 'gaussian', 'tvf'). Default: 'sobolev'.
    probe_pairs : Set[int], optional
        Set of probe pair indices to evaluate dual-arm Gaussian SyN ablation.
        Default: {0, 1, 2, 45, 67, 82}.
    random_order : bool
        If True, evaluates pairs in a deterministic pseudo-random permutation.
    seed : int
        Random seed for permutation and reproducibility. Default: 42.
    out_dir : str
        Directory to store per-pair JSON result files.
    summary_json : str
        Master summary JSON file path.
    report_html : str
        Path to compile the interactive Plotly HTML dashboard.
    generate_example_reports : bool
        If True, renders standard 5-figure visual reports for specified example pairs.
    example_report_pairs : List[int], optional
        List of pair indices for standard visual reports. Default: [0, 67].
    pairs_csv : str
        Path to pairs CSV configuration.
    data_dir : str, optional
        Mindboggle data root directory.
    verbose : bool
        If True, logs progress to stdout.

    Returns
    -------
    Dict[str, Any]
        Master benchmark summary dictionary.
    """
    # 1. Check Dataset Integrity
    is_valid, report = check_mindboggle_data(pairs_csv=pairs_csv, data_dir=data_dir, verbose=verbose)
    if not is_valid:
        raise RuntimeError(
            f"Cannot run Mindboggle benchmark: Missing data. "
            f"Found {report['available_pairs']}/{report['total_pairs_in_csv']} pairs."
        )

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_html)), exist_ok=True)

    if pairs is None:
        pairs = list(range(report["total_pairs_in_csv"]))

    if probe_pairs is None:
        probe_pairs = {0, 1, 2, 45, 67, 82}

    if example_report_pairs is None:
        example_report_pairs = [0, 67]

    if random_order:
        rng = np.random.RandomState(seed)
        permuted_order = rng.permutation(len(pairs)).tolist()
        ordered_pairs = [pairs[i] for i in permuted_order]
    else:
        ordered_pairs = list(pairs)

    total_pairs = len(ordered_pairs)
    if verbose:
        print("=" * 78)
        print(f"  Syntx Mindboggle Benchmark Runner ({total_pairs} Pairs)")
        print(f"  Primary Model: {model.upper()} | Randomized: {random_order} (Seed {seed})")
        print(f"  Dual-Arm Probe Set (Gaussian): {sorted(list(probe_pairs))}")
        print("=" * 78, flush=True)

    t0_benchmark = time.time()
    sobolev_results = {}
    gaussian_results = {}

    for step_num, pair_idx in enumerate(ordered_pairs, start=1):
        models_to_run = [model]
        if pair_idx in probe_pairs and model != "gaussian":
            models_to_run.append("gaussian")

        for m_type in models_to_run:
            out_file = os.path.join(out_dir, f"pair_{pair_idx:03d}_{m_type}.json")

            # Check cache / resume
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r") as f:
                        rec = json.load(f)
                    if rec.get("status") == "SUCCESS":
                        if m_type == "gaussian":
                            gaussian_results[pair_idx] = rec
                        else:
                            sobolev_results[pair_idx] = rec
                        if verbose:
                            print(f"[{step_num}/{total_pairs}] Pair {pair_idx:02d} [{m_type.upper()}]: Resumed from cache (Dice = {rec.get('syntx_dice_sym', 0.0):.4f})", flush=True)
                        continue
                except Exception:
                    pass

            # Run in isolated subprocess
            if verbose:
                print(f"[{step_num}/{total_pairs}] Launching Pair {pair_idx:02d} [{m_type.upper()}] in isolated subprocess...", flush=True)

            cmd = [
                sys.executable, "-u", "-m", "syntx.benchmark.cli",
                "--pair-idx", str(pair_idx),
                "--model", str(m_type),
                "--out-dir", str(out_dir),
                "--pairs-csv", str(pairs_csv),
            ]
            if data_dir:
                cmd.extend(["--data-dir", str(data_dir)])
            if generate_example_reports and pair_idx in example_report_pairs:
                cmd.append("--generate-report")

            res = subprocess.run(cmd, capture_output=False)
            if res.returncode != 0:
                print(f"[syntx.benchmark] ERROR: Subprocess failed on Pair {pair_idx:02d} [{m_type}] (exit {res.returncode})", file=sys.stderr)
            else:
                if os.path.exists(out_file):
                    with open(out_file, "r") as f:
                        rec = json.load(f)
                    if m_type == "gaussian":
                        gaussian_results[pair_idx] = rec
                    else:
                        sobolev_results[pair_idx] = rec

        # Intermediate progress logging and master summary sync
        n_done = len(sobolev_results)
        if n_done > 0 and verbose:
            primary_dices = [r.get("syntx_dice_sym", float("nan")) for r in sobolev_results.values()]
            ants_dices = [r.get("ants_baseline", {}).get("dice_sym", float("nan")) for r in sobolev_results.values()]
            aff_dices = [r.get("syntx_affine_dice_sym", float("nan")) for r in sobolev_results.values()]
            valid_pairs = [(s, a) for s, a in zip(primary_dices, ants_dices) if np.isfinite(s) and np.isfinite(a)]
            valid_aff = [a for a in aff_dices if np.isfinite(a)]
            
            if valid_pairs:
                wins = sum(1 for s, a in valid_pairs if s >= a)
                mean_prim = float(np.mean([p[0] for p in valid_pairs]))
                mean_ants = float(np.mean([p[1] for p in valid_pairs]))
                mean_aff = float(np.mean(valid_aff)) if valid_aff else float("nan")
                win_pct = (wins / len(valid_pairs)) * 100.0
                aff_str = f" | Affine Mean: {mean_aff:.4f}" if np.isfinite(mean_aff) else ""
                print(f"  PROGRESS: {n_done}/{total_pairs} Completed{aff_str} | {model.title()} Deform Mean: {mean_prim:.4f} vs ANTs: {mean_ants:.4f} | Win Rate: {win_pct:.1f}% ({wins}/{len(valid_pairs)})", flush=True)

            master_summary = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_completed": n_done,
                "total_planned": total_pairs,
                "permutation_order": ordered_pairs,
                "primary_model": model,
                "sobolev_results": sobolev_results,
                "gaussian_results": gaussian_results
            }
            with open(summary_json, "w") as f:
                json.dump(master_summary, f, indent=2)

    # 4. Compile Master Population Benchmark HTML Report
    try:
        from syntx.viz import create_population_benchmark_report
        create_population_benchmark_report(
            results_source=summary_json,
            output_html=report_html,
            title=f"Syntx {model.title()} SyN vs ANTs C++ — {total_pairs}-Pair Mindboggle Benchmark Report"
        )
        if verbose:
            print(f"[syntx.benchmark] Master HTML dashboard compiled at: {report_html}")
    except Exception as e:
        if verbose:
            print(f"[syntx.benchmark] Warning: Failed to compile HTML dashboard: {e}", file=sys.stderr)

    total_time = time.time() - t0_benchmark
    if verbose:
        print("=" * 78)
        print(f"  BENCHMARK COMPLETE: {len(sobolev_results)}/{total_pairs} in {total_time/60.0:.1f} minutes")
        print("=" * 78, flush=True)

    return {
        "total_completed": len(sobolev_results),
        "summary_json": os.path.abspath(summary_json),
        "report_html": os.path.abspath(report_html),
        "runtime_minutes": total_time / 60.0
    }
