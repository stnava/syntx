"""
syntx.benchmark.cli — Command-Line Interface for Syntx Registration Benchmarking
=================================================================================

Provides a unified CLI for checking data, running single pairs, and orchestrating
full 90-pair cohort evaluations.
"""

import os
import sys
import json
import numpy as np
import argparse

from syntx.benchmark.data import check_mindboggle_data, DEFAULT_PAIRS_CSV
from syntx.benchmark.evaluate import evaluate_mindboggle_pair
from syntx.benchmark.orchestrator import run_mindboggle_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Syntx Mindboggle-101 Registration Benchmark Suite"
    )
    parser.add_argument(
        "--check-data", action="store_true",
        help="Check Mindboggle dataset existence and display setup instructions if missing."
    )
    parser.add_argument(
        "--pair-idx", type=int, default=None,
        help="Evaluate a single pair index (0 to 89)."
    )
    parser.add_argument(
        "--model", type=str, default="both", choices=["both", "gaussian", "sobolev", "tvf"],
        help="Registration model / regularizer variant ('both' evaluates Gaussian and Sobolev on every pair)."
    )
    parser.add_argument(
        "--cohort", action="store_true",
        help="Run full cohort benchmark across pairs."
    )
    parser.add_argument(
        "--pairs", type=int, nargs="+", default=None,
        help="Subset of pair indices to evaluate (e.g. --pairs 0 1 2 45 67 82)."
    )
    parser.add_argument(
        "--pairs-csv", type=str, default=DEFAULT_PAIRS_CSV,
        help="Path to pairs.csv configuration file."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Mindboggle volumes root directory."
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/reproducible_eval",
        help="Output directory for JSON result files."
    )
    parser.add_argument(
        "--summary-json", type=str, default="results/reproducible_90pair_master_summary.json",
        help="Master summary JSON file path."
    )
    parser.add_argument(
        "--report-html", type=str, default="docs/reproducible_90pair_report.html",
        help="Master interactive HTML report path."
    )
    parser.add_argument(
        "--affine-report", action="store_true",
        help="Generate dedicated 90-Pair Affine Population Benchmark Report (docs/reproducible_90pair_affine_report.html)."
    )
    parser.add_argument(
        "--affine-report-html", type=str, default="docs/reproducible_90pair_affine_report.html",
        help="Output filepath for dedicated Affine HTML benchmark report."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run quick single-pair demonstration on mbhard (or 2D r16_r64) and create standard 5-figure visual report."
    )
    parser.add_argument(
        "--demo-dataset", type=str, default="mbhard",
        help="Dataset for demo mode ('mbhard', 'r16_r64', 'c', 'ellipse')."
    )
    parser.add_argument(
        "--demo-html", type=str, default="docs/reports/mbhard_standard_report.html",
        help="Output filepath for demo HTML report."
    )
    parser.add_argument(
        "--generate-report", action="store_true",
        help="Generate standalone 5-figure visual HTML diagnostic report."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-running all evaluations from scratch, ignoring cached JSON results."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic permutation and initialization."
    )

    args = parser.parse_args()

    # 1. Dataset verification mode
    if args.check_data:
        is_valid, rep = check_mindboggle_data(pairs_csv=args.pairs_csv, data_dir=args.data_dir, verbose=True)
        if is_valid:
            print(f"[syntx.benchmark] Dataset verified successfully! All {rep['available_pairs']} pairs ready.")
            sys.exit(0)
        else:
            sys.exit(1)

    # 2. Demo Report mode
    if args.demo:
        from syntx.benchmark.evaluate import run_standard_report_demo
        rep_path = run_standard_report_demo(
            dataset_key=args.demo_dataset,
            output_html=args.demo_html,
            model=args.model if args.model != "both" else "gaussian",
            verbose=True
        )
        print(f"[syntx.benchmark] Demo report generated: {rep_path}")
        sys.exit(0)

    # 3. Affine Report generation mode
    if args.affine_report:
        from syntx.viz.reports import create_affine_benchmark_report
        rep_path = create_affine_benchmark_report(
            summary_source=args.summary_json,
            output_html=args.affine_report_html
        )
        print(f"[syntx.benchmark] Generated 90-Pair Affine Benchmark Report: {rep_path}")
        sys.exit(0)

    # 3. Single Pair Evaluation mode
    if args.pair_idx is not None:
        models_to_eval = ["gaussian", "sobolev"] if args.model == "both" else [args.model]
        os.makedirs(args.out_dir, exist_ok=True)
        for m_name in models_to_eval:
            out_file = os.path.join(args.out_dir, f"pair_{args.pair_idx:03d}_{m_name}.json")
            rec = evaluate_mindboggle_pair(
                pair_idx=args.pair_idx,
                model=m_name,
                pairs_csv=args.pairs_csv,
                data_dir=args.data_dir,
                generate_report=args.generate_report,
                report_out_dir=os.path.join(args.out_dir, "reports"),
                verbose=True,
                seed=args.seed
            )
            with open(out_file, "w") as f:
                json.dump(rec, f, indent=2)

            win_str = "WIN" if rec.get("win") else "LOSS"
            diff = rec.get("diff_vs_ants", 0.0)
            ants_dice = rec.get("ants_baseline", {}).get("dice_sym", 0.0)
            aff_dice = rec.get('syntx_affine_dice_sym', float('nan'))
            aff_str = f"{aff_dice:.4f}" if np.isfinite(aff_dice) else "N/A"
            print(f"CASE_COMPLETE: Pair {args.pair_idx:02d} [{m_name.upper()}] | Affine Dice: {aff_str} | Deform Sym Dice: {rec['syntx_dice_sym']:.4f} (ANTs: {ants_dice:.4f}, diff: {diff:+.2f}%) | Fold: {rec['syntx_fold']:.4f}% | Time: {rec['syntx_time']:.1f}s | Result: {win_str}", flush=True)
        sys.exit(0)

    # 3. Cohort Benchmark mode
    if args.cohort or args.pairs is not None:
        run_mindboggle_benchmark(
            pairs=args.pairs,
            model=args.model,
            pairs_csv=args.pairs_csv,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            summary_json=args.summary_json,
            report_html=args.report_html,
            generate_example_reports=args.generate_report,
            seed=args.seed,
            random_order=False if args.pairs is not None else True,
            force=args.force,
            verbose=True
        )
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
