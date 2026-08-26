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
    force: bool = False,
    verbose: bool = False,
    use_n4: bool = True
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
    else:
        if probe_pairs is None:
            probe_pairs = set()

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

    ants_results = {}
    sobolev_results = {}
    gaussian_results = {}
    tvf_results = {}
    syngs_results = {}

    if os.path.exists(summary_json):
        try:
            with open(summary_json, "r") as f:
                existing_summary = json.load(f)
            if isinstance(existing_summary.get("ants_results"), dict):
                for k, v in existing_summary["ants_results"].items():
                    try:
                        ants_results[int(k)] = v
                    except ValueError:
                        pass
            if isinstance(existing_summary.get("sobolev_results"), dict):
                for k, v in existing_summary["sobolev_results"].items():
                    try:
                        sobolev_results[int(k)] = v
                    except ValueError:
                        pass
            if isinstance(existing_summary.get("gaussian_results"), dict):
                for k, v in existing_summary["gaussian_results"].items():
                    try:
                        gaussian_results[int(k)] = v
                    except ValueError:
                        pass
            if isinstance(existing_summary.get("tvf_results"), dict):
                for k, v in existing_summary["tvf_results"].items():
                    try:
                        tvf_results[int(k)] = v
                    except ValueError:
                        pass
            if isinstance(existing_summary.get("syngs_results"), dict):
                for k, v in existing_summary["syngs_results"].items():
                    try:
                        syngs_results[int(k)] = v
                    except ValueError:
                        pass
        except Exception:
            pass

    # Ensure all available ANTs baselines are loaded from disk cache
    for p_i in range(90):
        if p_i not in ants_results:
            ants_file_eval = os.path.join(out_dir, f"pair_{p_i:03d}_ants.json")
            ants_file_root = os.path.join("results", f"pair_{p_i:03d}_ants_syn.json")
            for af in [ants_file_eval, ants_file_root]:
                if os.path.exists(af):
                    try:
                        with open(af, "r") as f:
                            ants_results[p_i] = json.load(f)
                        break
                    except Exception:
                        pass

    for step_num, pair_idx in enumerate(ordered_pairs, start=1):
        if model in ("all", "all_5", "all5"):
            models_to_run = ["ants", "gaussian", "sobolev", "tvf", "syngs"]
        elif model in ("all_4", "all4"):
            models_to_run = ["ants", "gaussian", "sobolev", "tvf"]
        elif model in ("syn_tvf", "sobolev_tvf", "syntx"):
            models_to_run = ["sobolev", "tvf"]
        elif model in ("both", "gauss_sobolev"):
            models_to_run = ["gaussian", "sobolev"]
        elif pair_idx in probe_pairs and model not in ("gaussian", "all", "all_5"):
            models_to_run = [model, "gaussian"]
        else:
            models_to_run = [model]

        for m_type in models_to_run:
            out_file = os.path.join(out_dir, f"pair_{pair_idx:03d}_{m_type}.json")

            # Check cache / resume
            if not force and os.path.exists(out_file):
                try:
                    with open(out_file, "r") as f:
                        rec = json.load(f)
                    if rec.get("status") == "SUCCESS":
                        if m_type in ("ants", "ants_syn"):
                            ants_results[pair_idx] = rec
                        elif m_type == "gaussian":
                            gaussian_results[pair_idx] = rec
                        elif m_type == "sobolev":
                            sobolev_results[pair_idx] = rec
                        elif m_type == "tvf":
                            tvf_results[pair_idx] = rec
                        elif m_type == "syngs":
                            syngs_results[pair_idx] = rec
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
            if not use_n4:
                cmd.append("--no-n4")
            if generate_example_reports and pair_idx in example_report_pairs:
                cmd.append("--generate-report")

            res = subprocess.run(cmd, capture_output=False)
            if res.returncode != 0:
                print(f"[syntx.benchmark] ERROR: Subprocess failed on Pair {pair_idx:02d} [{m_type}] (exit {res.returncode})", file=sys.stderr)
            else:
                if os.path.exists(out_file):
                    with open(out_file, "r") as f:
                        rec = json.load(f)
                    if m_type in ("ants", "ants_syn"):
                        ants_results[pair_idx] = rec
                    elif m_type == "gaussian":
                        gaussian_results[pair_idx] = rec
                    elif m_type == "sobolev":
                        sobolev_results[pair_idx] = rec
                    elif m_type == "tvf":
                        tvf_results[pair_idx] = rec
                    elif m_type == "syngs":
                        syngs_results[pair_idx] = rec

                    diff = rec.get("diff_vs_ants", 0.0)
                    if diff < 0.0 and np.isfinite(diff):
                        aff_d = rec.get("syntx_affine_dice_sym", float("nan"))
                        def_d = rec.get("syntx_dice_sym", float("nan"))
                        ants_d = rec.get("ants_baseline", {}).get("dice_sym", float("nan"))
                        print(f"  ⚠️ OUTLIER DETECTED: Pair {pair_idx:02d} [{m_type.upper()}] | Deform: {def_d:.4f} vs ANTs: {ants_d:.4f} ({diff:+.2f}%) | Affine Dice: {aff_d:.4f}", flush=True)

        # Intermediate progress logging and master summary sync
        n_done = max(len(ants_results), len(sobolev_results), len(gaussian_results), len(tvf_results), len(syngs_results))
        if n_done > 0:
            if verbose:
                # Affine metrics
                all_recs = list(ants_results.values()) + list(sobolev_results.values()) + list(gaussian_results.values()) + list(tvf_results.values()) + list(syngs_results.values())
                aff_all = [r.get("syntx_affine_dice_sym", float("nan")) for r in all_recs]
                aff_valid = [a for a in aff_all if np.isfinite(a)]
                mean_aff = float(np.mean(aff_valid)) if aff_valid else float("nan")

                # ANTs metrics
                a_valid = [r.get("syntx_dice_sym", float("nan")) for r in ants_results.values() if np.isfinite(r.get("syntx_dice_sym", float("nan")))]
                a_mean = float(np.mean(a_valid)) if a_valid else float("nan")

                # Gaussian metrics
                g_valid = [r.get("syntx_dice_sym", float("nan")) for r in gaussian_results.values() if np.isfinite(r.get("syntx_dice_sym", float("nan")))]
                g_mean = float(np.mean(g_valid)) if g_valid else float("nan")

                # Sobolev metrics
                s_valid = [r.get("syntx_dice_sym", float("nan")) for r in sobolev_results.values() if np.isfinite(r.get("syntx_dice_sym", float("nan")))]
                s_mean = float(np.mean(s_valid)) if s_valid else float("nan")

                # TVF metrics
                t_valid = [r.get("syntx_dice_sym", float("nan")) for r in tvf_results.values() if np.isfinite(r.get("syntx_dice_sym", float("nan")))]
                t_mean = float(np.mean(t_valid)) if t_valid else float("nan")

                # SyNGS metrics
                gs_valid = [r.get("syntx_dice_sym", float("nan")) for r in syngs_results.values() if np.isfinite(r.get("syntx_dice_sym", float("nan")))]
                gs_mean = float(np.mean(gs_valid)) if gs_valid else float("nan")

                aff_str = f" | Affine: {mean_aff:.4f}" if np.isfinite(mean_aff) else ""
                a_str = f" | ANTs: {a_mean:.4f}" if np.isfinite(a_mean) else ""
                g_str = f" | Gauss: {g_mean:.4f}" if np.isfinite(g_mean) else ""
                s_str = f" | Sobolev: {s_mean:.4f}" if np.isfinite(s_mean) else ""
                t_str = f" | TVF: {t_mean:.4f}" if np.isfinite(t_mean) else ""
                gs_str = f" | SyNGS: {gs_mean:.4f}" if np.isfinite(gs_mean) else ""
                print(f"  PROGRESS: {n_done}/{total_pairs} Completed{aff_str}{a_str}{g_str}{s_str}{t_str}{gs_str}", flush=True)

            master_summary = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_completed": n_done,
                "total_planned": total_pairs,
                "permutation_order": ordered_pairs,
                "primary_model": model,
                "ants_results": ants_results,
                "sobolev_results": sobolev_results,
                "gaussian_results": gaussian_results,
                "tvf_results": tvf_results,
                "syngs_results": syngs_results
            }
            with open(summary_json, "w") as f:
                json.dump(master_summary, f, indent=2)

            try:
                from syntx.viz import create_population_benchmark_report
                report_title = f"Syntx 5-Arm Population Benchmark — {total_pairs}-Pair Mindboggle Report" if model in ("all", "all_5") else f"Syntx {model.title()} SyN vs ANTs C++ — {total_pairs}-Pair Mindboggle Benchmark Report"
                create_population_benchmark_report(
                    results_source=summary_json,
                    output_html=report_html,
                    title=report_title
                )
            except Exception:
                pass

    # 4. Compile Master Population Benchmark HTML Report
    try:
        from syntx.viz import create_population_benchmark_report
        report_title = f"Syntx (Gaussian & Sobolev) SyN vs ANTs C++ — {total_pairs}-Pair Mindboggle Benchmark Report" if model == "both" else f"Syntx {model.title()} SyN vs ANTs C++ — {total_pairs}-Pair Mindboggle Benchmark Report"
        create_population_benchmark_report(
            results_source=summary_json,
            output_html=report_html,
            title=report_title
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
