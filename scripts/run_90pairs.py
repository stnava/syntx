#!/usr/bin/env python
"""
syntx Benchmark: 90-Pair Mindboggle Population Evaluation
==========================================================

Runs the full 90-pair Mindboggle DKT31 benchmark with process isolation,
automatic crash recovery, and comprehensive aggregate reporting.

Each pair is evaluated in a separate subprocess to guarantee full memory
reclamation (PyTorch MPS Metal allocations, ANTsPy ITK heap) between pairs.

Features
--------
- **Process Isolation**: Each pair runs in its own Python subprocess
- **Crash Recovery**: Automatically resumes from the last successful pair
- **Atomic Writes**: Per-pair JSON results use write-to-temp + os.replace
- **Aggregate Reports**: Generates summary statistics and Markdown report
- **Progress Logging**: Real-time stdout progress with ETA estimates

Usage
-----
    # Run all 90 pairs with SyN on MPS
    python scripts/run_90pairs.py --model syn --device mps

    # Resume an interrupted run (skips completed pairs)
    python scripts/run_90pairs.py --model syn --device mps --resume

    # Run a subset of pairs
    python scripts/run_90pairs.py --model syn --device mps --start 10 --end 20

    # Force restart (re-run all pairs)
    python scripts/run_90pairs.py --model syn --device mps --force-restart

Output
------
    results/
        pair_000_syn.json       # Per-pair result files
        pair_001_syn.json
        ...
        pair_089_syn.json
        summary_syn_mps.json    # Aggregate statistics
        summary_syn_mps.md      # Markdown report
"""

import sys
import os
import argparse
import json
import time
import subprocess
import statistics
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("syntx.benchmark.90pair")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "docs", "provenance", "run_config.json")
DEFAULT_PAIRS_CSV = os.path.join(PROJECT_ROOT, "examples", "pairs.csv")
SINGLE_PAIR_SCRIPT = os.path.join(SCRIPT_DIR, "run_single_pair.py")
TOTAL_PAIRS = 90
DEFAULT_TIMEOUT = 1200  # 20 minutes per pair

if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from syntx.benchmark.config import get_model_config, compute_config_hash


def count_pairs(pairs_csv: str) -> int:
    """Returns the number of data rows in the pairs CSV."""
    with open(pairs_csv, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header


def is_pair_completed(
    results_dir: str,
    pair_idx: int,
    model: str,
    expected_config_hash: Optional[str] = None,
) -> bool:
    """Checks if a pair result file exists, completed successfully, and matches the active config.

    Parameters
    ----------
    results_dir : str
        Directory containing per-pair JSON results.
    pair_idx : int
        Pair index.
    model : str
        Model name ('syn', 'syngs', 'tvf', 'greedy').
    expected_config_hash : str, optional
        Expected SHA-256 configuration hash. If provided, cached results with a
        mismatched or missing hash are rejected and marked for rerun.

    Returns
    -------
    bool
        True if the pair completed successfully with matching config hash.
    """
    path = os.path.join(results_dir, f"pair_{pair_idx:03d}_{model}.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            result = json.load(f)
        if result.get("status") != "SUCCESS":
            return False
        if expected_config_hash:
            cached_hash = result.get("config_hash")
            if cached_hash != expected_config_hash:
                logger.warning(
                    f"Pair {pair_idx:03d} [{model}] cached config_hash ({cached_hash}) != "
                    f"expected ({expected_config_hash}); invalidating stale cache."
                )
                return False
        return True
    except (json.JSONDecodeError, KeyError, IOError):
        return False


def load_pair_result(results_dir: str, pair_idx: int, model: str) -> Optional[dict]:
    """Loads a completed pair result from disk.

    Parameters
    ----------
    results_dir : str
        Directory containing per-pair JSON results.
    pair_idx : int
        Pair index.
    model : str
        Model name.

    Returns
    -------
    dict or None
        Parsed result dictionary, or None if not available.
    """
    path = os.path.join(results_dir, f"pair_{pair_idx:03d}_{model}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def run_pair_isolated(
    pair_idx: int,
    model: str,
    device: str,
    config_path: str,
    results_dir: str,
    pairs_csv: str,
    data_dir: Optional[str],
    timeout: int,
) -> dict:
    """Runs a single pair in an isolated subprocess.

    Parameters
    ----------
    pair_idx : int
        Pair index.
    model : str
        Model type ('syn' or 'tvf').
    device : str
        Compute device.
    config_path : str
        Path to config JSON.
    results_dir : str
        Directory for output JSON.
    pairs_csv : str
        Path to pairs CSV.
    data_dir : str or None
        Data directory override.
    timeout : int
        Maximum seconds per pair.

    Returns
    -------
    dict
        Result dictionary (from file or error stub).
    """
    out_json = os.path.join(results_dir, f"pair_{pair_idx:03d}_{model}.json")

    cmd = [
        sys.executable, SINGLE_PAIR_SCRIPT,
        "--pair-idx", str(pair_idx),
        "--model", model,
        "--device", device,
        "--config", config_path,
        "--out-json", out_json,
        "--pairs-csv", pairs_csv,
    ]
    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    env = os.environ.copy()
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
            cwd=PROJECT_ROOT,
        )

        # Parse stdout for the machine-readable line
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("DONE:") or line.startswith("FAIL:"):
                logger.info(f"  Worker output: {line}")

        if proc.returncode != 0 and not os.path.exists(out_json):
            # Worker crashed without writing output
            error_msg = proc.stderr[-500:] if proc.stderr else "Unknown error"
            result = {
                "pair_idx": pair_idx,
                "model": model,
                "device": device,
                "status": "FAILED",
                "error": f"Worker exited with code {proc.returncode}: {error_msg}",
            }
            # Write the error result
            os.makedirs(results_dir, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            return result

    except subprocess.TimeoutExpired:
        result = {
            "pair_idx": pair_idx,
            "model": model,
            "device": device,
            "status": "FAILED",
            "error": f"Timed out after {timeout}s",
        }
        os.makedirs(results_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        result = {
            "pair_idx": pair_idx,
            "model": model,
            "device": device,
            "status": "FAILED",
            "error": str(e),
        }
        os.makedirs(results_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Load the result written by the worker
    return load_pair_result(results_dir, pair_idx, model) or {"status": "UNKNOWN"}


def compute_summary(results: List[dict]) -> dict:
    """Computes aggregate statistics from a list of pair results.

    Parameters
    ----------
    results : list of dict
        List of per-pair result dictionaries.

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    successful = [r for r in results if r.get("status") == "SUCCESS"]
    failed = [r for r in results if r.get("status") != "SUCCESS"]

    if not successful:
        return {
            "total_pairs": len(results),
            "successful": 0,
            "failed": len(failed),
        }

    def safe_stats(key):
        values = [r[key] for r in successful if key in r and r[key] is not None and str(r[key]) != "nan"]
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None, "median": None}
        return {
            "mean": round(statistics.mean(values), 6),
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "median": round(statistics.median(values), 6),
        }

    summary = {
        "total_pairs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "failed_indices": [r.get("pair_idx") for r in failed],
        "dice_sym": safe_stats("dice_sym"),
        "dice_fixed": safe_stats("dice_fixed"),
        "dice_moving": safe_stats("dice_moving"),
        "folding_pct": safe_stats("folding_pct"),
        "min_jacobian": safe_stats("min_jacobian"),
        "harmonic_energy": safe_stats("harmonic_energy"),
        "bending_energy": safe_stats("bending_energy"),
        "mattes_mi": safe_stats("mattes_mi"),
        "lncc": safe_stats("lncc"),
        "runtime_seconds": safe_stats("runtime_seconds"),
        "total_runtime_seconds": sum(
            r.get("runtime_seconds", 0) for r in successful
        ),
    }

    hashes = sorted(list({r.get("config_hash") for r in successful if r.get("config_hash")}))
    if hashes:
        summary["config_hash"] = hashes[0] if len(hashes) == 1 else hashes

    return summary


def write_markdown_report(summary: dict, results: List[dict], model: str, device: str, out_path: str) -> None:
    """Generates a Markdown summary report.

    Parameters
    ----------
    summary : dict
        Aggregate statistics from ``compute_summary()``.
    results : list of dict
        Per-pair result dictionaries.
    model : str
        Model name.
    device : str
        Device name.
    out_path : str
        Output Markdown file path.
    """
    lines = [
        f"# syntx {model.upper()} Benchmark Results ({device.upper()})\n",
        f"\n**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        f"\n## Summary\n",
        f"\n| Metric | Mean | Std | Min | Median | Max |",
        f"\n|--------|------|-----|-----|--------|-----|",
    ]

    metrics_display = [
        ("Symmetric Dice", "dice_sym"),
        ("Fixed Dice", "dice_fixed"),
        ("Moving Dice", "dice_moving"),
        ("Folding (%)", "folding_pct"),
        ("Min Jacobian", "min_jacobian"),
        ("Harmonic Energy", "harmonic_energy"),
        ("Bending Energy", "bending_energy"),
        ("Mattes MI", "mattes_mi"),
        ("LNCC", "lncc"),
        ("Runtime (s)", "runtime_seconds"),
    ]

    for display_name, key in metrics_display:
        s = summary.get(key, {})
        if s.get("mean") is not None:
            lines.append(
                f"\n| {display_name} | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['min']:.4f} | {s['median']:.4f} | {s['max']:.4f} |"
            )

    lines.append(f"\n\n**Total Pairs**: {summary['successful']}/{summary['total_pairs']} successful")
    if summary.get("failed_indices"):
        lines.append(f"\n**Failed Indices**: {summary['failed_indices']}")
    lines.append(
        f"\n**Total Compute Time**: {summary.get('total_runtime_seconds', 0):.0f}s "
        f"({summary.get('total_runtime_seconds', 0) / 3600:.1f}h)"
    )

    # Per-pair table
    successful = sorted(
        [r for r in results if r.get("status") == "SUCCESS"],
        key=lambda r: r.get("pair_idx", 0),
    )
    if successful:
        lines.append("\n\n## Per-Pair Results\n")
        lines.append("\n| Pair | Fixed ID | Moving ID | Dice Sym | Fold% | Time (s) |")
        lines.append("\n|------|----------|-----------|----------|-------|----------|")
        for r in successful:
            lines.append(
                f"\n| {r.get('pair_idx', '?'):>3} | "
                f"{r.get('fixed_id', '?')[:15]:>15} | "
                f"{r.get('moving_id', '?')[:15]:>15} | "
                f"{r.get('dice_sym', 0):.4f} | "
                f"{r.get('folding_pct', 0):.3f} | "
                f"{r.get('runtime_seconds', 0):.1f} |"
            )

    lines.append("\n")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)


def write_cross_model_report(all_model_results: Dict[str, List[dict]], device: str, out_path: str) -> None:
    """Generates a cross-model head-to-head comparison Markdown report."""
    lines = [
        f"# syntx Multi-Model Benchmark Comparison ({device.upper()})\n",
        f"\n**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        f"\n## Method Comparison Summary (cc2 metric, [100, 100, 20] iterations)\n",
        f"\n| Method | Model Description | Completed | Mean Dice (±std) | Median Dice | Mean Fold (%) | Mean Time (s) | Win Rate vs ANTs |",
        f"\n|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    method_names = {
        "syn": "Eulerian Sobolev SyN",
        "syngs": "Geodesic Shooting SyN (SyNGS)",
        "tvf": "Time-Varying Velocity Field (TVF)",
        "greedy": "Compositive Greedy (LDdMM)",
        "ants_syn": "ANTs C++ SyN Reference",
    }

    for model, results in all_model_results.items():
        succ = [r for r in results if r.get("status") == "SUCCESS"]
        desc = method_names.get(model, model.upper())
        if not succ:
            lines.append(f"\n| **{model.upper()}** | {desc} | 0/{len(results)} | N/A | N/A | N/A | N/A | N/A |")
            continue

        dices = [r["dice_sym"] for r in succ if "dice_sym" in r and r["dice_sym"] is not None and str(r["dice_sym"]) != "nan"]
        folds = [r["folding_pct"] for r in succ if "folding_pct" in r and r["folding_pct"] is not None and str(r["folding_pct"]) != "nan"]
        times = [r["runtime_seconds"] for r in succ if "runtime_seconds" in r and r["runtime_seconds"] is not None and str(r["runtime_seconds"]) != "nan"]
        wins = [r.get("win", False) for r in succ]

        mean_dice = statistics.mean(dices) if dices else 0.0
        std_dice = statistics.stdev(dices) if len(dices) > 1 else 0.0
        med_dice = statistics.median(dices) if dices else 0.0
        mean_fold = statistics.mean(folds) if folds else 0.0
        mean_time = statistics.mean(times) if times else 0.0
        win_rate = (sum(1 for w in wins if w) / len(wins) * 100.0) if wins else 0.0

        lines.append(
            f"\n| **{model.upper()}** | {desc} | {len(succ)}/{len(results)} | "
            f"**{mean_dice:.4f}** ± {std_dice:.4f} | {med_dice:.4f} | "
            f"{mean_fold:.4f}% | {mean_time:.1f}s | {win_rate:.1f}% |"
        )

    lines.append("\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)


def main():
    """CLI entry point for the 90-pair orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run the full 90-pair Mindboggle benchmark across all valid methods with process isolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model syn --device mps
  %(prog)s --model all --device mps
  %(prog)s --models syn syngs tvf greedy --device mps
  %(prog)s --model all --device mps --start 0 --end 10
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["syn", "syngs", "tvf", "greedy", "ants_syn", "all"],
        help="Single model name, or 'all' to run all four valid methods.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more model names to run (e.g. syn syngs tvf greedy, or 'all').",
    )
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--pairs-csv", type=str, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--start", type=int, default=0, help="First pair index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last pair index (exclusive)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed pairs")
    parser.add_argument("--force-restart", action="store_true", help="Re-run all pairs (deletes existing results)")
    parser.add_argument("--random-order", action="store_true", help="Evaluate pairs in randomized order")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random order shuffling (default: 42)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Max seconds per pair")

    args = parser.parse_args()

    # Determine models to evaluate
    if args.models:
        if "all" in args.models:
            models_to_run = ["syn", "syngs", "tvf", "greedy"]
        else:
            models_to_run = args.models
    elif args.model:
        if args.model == "all":
            models_to_run = ["syn", "syngs", "tvf", "greedy"]
        else:
            models_to_run = [args.model]
    else:
        models_to_run = ["syn"]

    # Resolve number of pairs
    total = count_pairs(args.pairs_csv)
    end_idx = min(args.end or total, total)
    start_idx = max(args.start, 0)
    pair_indices = list(range(start_idx, end_idx))

    if args.random_order:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(pair_indices)
        logger.info(f"Random order enabled: pair sequence shuffled with seed {args.seed}")

    # Load configuration and compute expected hashes
    run_cfg = None
    if os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                run_cfg = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config file {args.config}: {e}")

    expected_hashes = {
        m: compute_config_hash(get_model_config(m, run_cfg))
        for m in models_to_run
    }

    # Banner
    print("=" * 72)
    print(f"  syntx Comprehensive Benchmark — Device: {args.device.upper()}")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"  Metric: cc2 (Local Normalized Cross-Correlation)")
    print(f"  Iterations: [100, 100, 20] (Constant Across All Methods)")
    print(f"  Pairs: {len(pair_indices)} total (range: {start_idx}..{end_idx - 1})")
    print(f"  Random Order: {args.random_order} (seed: {args.seed if args.random_order else 'N/A'})")
    if args.random_order:
        print(f"  Execution Sequence: {pair_indices[:15]} ...")
    print(f"  Config: {args.config}")
    for m, h in expected_hashes.items():
        print(f"    - {m}: hash={h}")
    print(f"  Resume: {args.resume}  |  Timeout: {args.timeout}s")
    print("=" * 72)
    print()

    all_model_results: Dict[str, List[dict]] = {m: [] for m in models_to_run}

    for model in models_to_run:
        m_results_dir = os.path.join(args.results_dir, f"{model}_{args.device}")
        os.makedirs(m_results_dir, exist_ok=True)

        if args.force_restart:
            logger.warning(f"Force restart for {model}: deleting existing results in {m_results_dir}")
            for idx in pair_indices:
                path = os.path.join(m_results_dir, f"pair_{idx:03d}_{model}.json")
                if os.path.exists(path):
                    os.remove(path)

    for i, pair_idx in enumerate(pair_indices):
        print(f"\n--- Processing Pair {pair_idx:03d} [{i + 1}/{len(pair_indices)}] ---")
        for model in models_to_run:
            m_results_dir = os.path.join(args.results_dir, f"{model}_{args.device}")

            # Check for resume (with SHA-256 config hash validation)
            if args.resume and is_pair_completed(
                m_results_dir, pair_idx, model, expected_config_hash=expected_hashes.get(model)
            ):
                result = load_pair_result(m_results_dir, pair_idx, model)
                all_model_results[model].append(result)
                logger.info(
                    f"  [{model.upper()}] SKIP pair {pair_idx:03d} (Dice={result.get('dice_sym', 0):.4f}, Hash={result.get('config_hash', 'N/A')})"
                )
                continue

            logger.info(f"  [{model.upper()}] Running pair {pair_idx:03d}...")
            t0 = time.time()
            result = run_pair_isolated(
                pair_idx=pair_idx,
                model=model,
                device=args.device,
                config_path=args.config,
                results_dir=m_results_dir,
                pairs_csv=args.pairs_csv,
                data_dir=args.data_dir,
                timeout=args.timeout,
            )
            t1 = time.time()
            all_model_results[model].append(result)

            if result.get("status") == "SUCCESS":
                logger.info(
                    f"  [{model.upper()}] SUCCESS | Dice={result.get('dice_sym', 0):.4f} | "
                    f"Fold={result.get('folding_pct', 0):.3f}% | Time={t1 - t0:.1f}s"
                )
            else:
                logger.error(f"  [{model.upper()}] FAILED | {result.get('error', 'Unknown')[:100]}")

        # Update live cross-model summary report
        if len(models_to_run) > 1:
            live_md = os.path.join(args.results_dir, "multimodel_live_comparison.md")
            try:
                write_cross_model_report(all_model_results, args.device, live_md)
            except Exception:
                pass

    # Generate per-model and cross-model summaries
    print()
    print("=" * 72)
    print(f"  ALL BENCHMARKS COMPLETE")
    print("=" * 72)

    for model in models_to_run:
        m_results_dir = os.path.join(args.results_dir, f"{model}_{args.device}")
        summary = compute_summary(all_model_results[model])
        summary["model"] = model
        summary["device"] = args.device
        summary["config_path"] = args.config
        summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Write summary JSON
        summary_json = os.path.join(m_results_dir, f"summary_{model}_{args.device}.json")
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        # Write Markdown report
        summary_md = os.path.join(m_results_dir, f"summary_{model}_{args.device}.md")
        write_markdown_report(summary, all_model_results[model], model, args.device, summary_md)
        logger.info(f"[{model.upper()}] Summary written to {summary_md}")

    # Cross-model comparison if multiple models evaluated
    if len(models_to_run) > 1:
        cross_json = os.path.join(args.results_dir, f"multimodel_{len(pair_indices)}pairs_comparison.json")
        cross_md = os.path.join(args.results_dir, f"multimodel_{len(pair_indices)}pairs_comparison.md")

        cross_summary = {
            m: compute_summary(all_model_results[m]) for m in models_to_run
        }
        with open(cross_json, "w") as f:
            json.dump(cross_summary, f, indent=2)

        write_cross_model_report(all_model_results, args.device, cross_md)
        logger.info(f"Cross-model report written to {cross_md}")

        # Print comparison table to stdout
        with open(cross_md) as f:
            print(f.read())


if __name__ == "__main__":
    main()
