"""
syntx.benchmark.cli — Mindboggle benchmark command line (``python -m syntx.benchmark``)
======================================================================================

One mode per invocation, checked in this order (the first that applies runs, then exits):

1. ``--precompute-n4``: fill the N4 cache for the subjects of ``--pairs-csv``.
2. ``--organize-data SOURCE_PATH``: ``organize_mindboggle_data`` into ``--target-dir``
   (default ``$SYNTX_DATA_DIR`` or ``DEFAULT_DATA_DIR``); exit 1 if the result is incomplete.
3. ``--check-data``: ``check_mindboggle_data``; exit 0 / 1.
4. ``--demo``: ``run_standard_report_demo`` on ``--demo-dataset`` (model ``--model``, with
   'both' meaning 'sobolev').
5. ``--affine-report``: ``create_affine_benchmark_report`` from ``--summary-json``.
6. ``--pair-idx N``: ``evaluate_mindboggle_pair`` for each model of ``--model``
   (``expand_model_set``: e.g. the default 'syn_tvf' = sobolev then tvf), written to
   ``<out-dir>/pair_<N>_<name>.json`` with name ``--out-name`` (one model only), else the
   model name (+ '_denoised' with ``--denoise``); prints a ``CASE_COMPLETE`` line.
7. ``--cohort`` or ``--pairs ...``: ``run_mindboggle_benchmark`` (pairs in order when
   ``--pairs`` is given, else all pairs in seeded random order).

Otherwise the help is printed. The parameter options (``--reg-iterations``,
``--learning-rate`` / ``--grad-step``, ``--flow-sigma``, ``--total-sigma``, ``--optimizer``,
``--similarity-metric`` / ``--metric``, ``--regularizer``) are forwarded as keyword overrides
in modes 6 and 7 only.
"""

import os
import sys
import json
import numpy as np
import argparse

from syntx.benchmark.data import check_mindboggle_data, DEFAULT_PAIRS_CSV
from syntx.benchmark.evaluate import evaluate_mindboggle_pair
from syntx.benchmark.orchestrator import run_mindboggle_benchmark, expand_model_set


def main():
    """Parse the command line and run the selected mode (see module docstring).

    Notes:

    - ``--model`` is a model set or one model (``expand_model_set``) in both the
      ``--pair-idx`` and the cohort modes; ``--out-name`` needs a single model.
    - ``--no-n4`` and ``--denoise`` / ``--no-denoise`` (denoising is on only with ``--denoise``
      and without ``--no-denoise``) apply in the ``--pair-idx`` and cohort modes.
    - ``--seed`` seeds the evaluation in ``--pair-idx`` mode and the pair permutation in
      cohort mode.
    - ``--force`` and ``--summary-json`` / ``--report-html`` apply to cohort mode
      (``--summary-json`` also to ``--affine-report``); ``--generate-report`` writes reports
      to ``<out-dir>/reports`` in ``--pair-idx`` mode and for the example pairs in cohort mode.
    - ``--verbose`` also controls ``--organize-data`` output.

    Exits via ``sys.exit`` in every mode except when only the help is printed.
    """
    parser = argparse.ArgumentParser(
        description="Syntx Mindboggle-101 Registration Benchmark Suite"
    )
    parser.add_argument(
        "--precompute-n4", action="store_true",
        help="Precompute and disk-cache ANTsTorch N4 bias field correction for the subjects named in --pairs-csv."
    )
    parser.add_argument(
        "--no-n4", action="store_true",
        help="Disable ANTsTorch N4 bias field correction preprocessing."
    )
    parser.add_argument(
        "--no-denoise", action="store_true",
        help="Disable ANTsTorch non-local means denoising (antstorch.denoise_image) preprocessing."
    )
    parser.add_argument(
        "--denoise", action="store_true", default=False,
        help="Apply ANTsTorch non-local means denoising (antstorch.denoise_image) prior to intensity normalization (default: False)."
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
        "--model", type=str, default="syn_tvf",
        help="A model ('greedy', 'greedy_regadam', 'syn_regadam', 'syn_dsti1', 'gaussian', 'sobolev', 'tvf', 'syngs', 'fireants', 'ants', ...) or a model set ('syn_tvf' (default), 'all', 'all_4', 'both')."
    )
    parser.add_argument(
        "--reg-iterations", type=int, nargs="+", default=None,
        help="Deformable registration iterations per level (e.g. --reg-iterations 100 100 40)."
    )
    parser.add_argument(
        "--learning-rate", "--grad-step", type=float, default=None, dest="learning_rate",
        help="Learning rate / grad step size."
    )
    parser.add_argument(
        "--flow-sigma", type=float, default=None,
        help="Fluid regularization standard deviation."
    )
    parser.add_argument(
        "--total-sigma", type=float, default=None,
        help="Elastic regularization standard deviation."
    )
    parser.add_argument(
        "--optimizer", type=str, default=None,
        help="Optimization algorithm ('adam', 'regadam', 'cfl')."
    )
    parser.add_argument(
        "--similarity-metric", "--metric", type=str, default=None,
        help="Similarity metric ('lncc', 'cc2', 'mattes_mi', 'vgg_4_lncc', 'dino_2_lncc', etc.)."
    )
    parser.add_argument(
        "--regularizer", type=str, default=None,
        help="Regularization operator ('gaussian', 'sobolev', 'dsti1', 'bspline')."
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
        "--out-name", type=str, default=None,
        help="Custom output filename identifier suffix (e.g. 'reg_gaussian_sig2')."
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
        "--organize-data", type=str, default=None, metavar="SOURCE_PATH",
        help="Organize raw Mindboggle volumes/archives from SOURCE_PATH into standardized directory hierarchy."
    )
    parser.add_argument(
        "--target-dir", type=str, default=None,
        help="Target directory to organize Mindboggle volumes into (default: $SYNTX_DATA_DIR, else syntx.benchmark.data.DEFAULT_DATA_DIR)."
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
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose diagnostic output during optimization and evaluation."
    )

    args = parser.parse_args()

    # 1. N4 precomputation mode
    if args.precompute_n4:
        from syntx.benchmark.data import precompute_mindboggle_n4
        res = precompute_mindboggle_n4(
            pairs_csv=args.pairs_csv,
            data_dir=args.data_dir,
            verbose=True
        )
        sys.exit(0)

    # 2. Dataset organization mode
    if args.organize_data:
        from syntx.benchmark.data import organize_mindboggle_data, DEFAULT_DATA_DIR, DEFAULT_DATA_DIR_ENV
        target = args.target_dir or os.environ.get(DEFAULT_DATA_DIR_ENV, DEFAULT_DATA_DIR)
        is_valid, rep = organize_mindboggle_data(
            source_path=args.organize_data,
            target_dir=target,
            pairs_csv=args.pairs_csv,
            verbose=args.verbose
        )
        sys.exit(0 if is_valid else 1)

    # 3. Dataset verification mode
    if args.check_data:
        is_valid, rep = check_mindboggle_data(pairs_csv=args.pairs_csv, data_dir=args.data_dir, verbose=True)
        if is_valid:
            print(f"[syntx.benchmark] Dataset verified successfully! All {rep['available_pairs']} pairs ready at: '{rep['data_dir']}'")
            sys.exit(0)
        else:
            sys.exit(1)

    # 4. Demo Report mode
    if args.demo:
        from syntx.benchmark.evaluate import run_standard_report_demo
        rep_path = run_standard_report_demo(
            dataset_key=args.demo_dataset,
            output_html=args.demo_html,
            model=args.model if args.model != "both" else "sobolev",
            verbose=args.verbose
        )
        print(f"[syntx.benchmark] Demo report generated: {rep_path}")
        sys.exit(0)

    # 5. Affine Report generation mode
    if args.affine_report:
        from syntx.viz.reports import create_affine_benchmark_report
        rep_path = create_affine_benchmark_report(
            summary_source=args.summary_json,
            output_html=args.affine_report_html
        )
        print(f"[syntx.benchmark] Generated 90-Pair Affine Benchmark Report: {rep_path}")
        sys.exit(0)

    # 6. Single Pair Evaluation mode
    use_n4 = not args.no_n4
    if args.pair_idx is not None:
        models_to_eval = expand_model_set(args.model)
        if args.out_name and len(models_to_eval) > 1:
            parser.error(f"--out-name needs a single model; --model {args.model} runs {models_to_eval}")
        os.makedirs(args.out_dir, exist_ok=True)
        use_denoise = args.denoise and not args.no_denoise
        for m_name in models_to_eval:
            out_name = args.out_name or (f"{m_name}_denoised" if use_denoise else m_name)
            out_file = os.path.join(args.out_dir, f"pair_{args.pair_idx:03d}_{out_name}.json")
            kwargs = {}
            if args.reg_iterations is not None:
                kwargs["reg_iterations"] = args.reg_iterations
            if args.learning_rate is not None:
                kwargs["learning_rate"] = args.learning_rate
            if args.flow_sigma is not None:
                kwargs["flow_sigma"] = args.flow_sigma
            if args.total_sigma is not None:
                kwargs["total_sigma"] = args.total_sigma
            if args.optimizer is not None:
                kwargs["optimizer"] = args.optimizer
            if args.similarity_metric is not None:
                kwargs["similarity_metric"] = args.similarity_metric
            if args.regularizer is not None:
                kwargs["regularizer"] = args.regularizer

            rec = evaluate_mindboggle_pair(
                pair_idx=args.pair_idx,
                model=m_name,
                pairs_csv=args.pairs_csv,
                data_dir=args.data_dir,
                generate_report=args.generate_report,
                report_out_dir=os.path.join(args.out_dir, "reports"),
                verbose=args.verbose,
                seed=args.seed,
                use_n4=use_n4,
                denoise=use_denoise,
                **kwargs
            )
            with open(out_file, "w") as f:
                json.dump(rec, f, indent=2)

            win_str = "WIN" if rec.get("win") else "LOSS"
            diff = rec.get("diff_vs_ants", float("nan"))
            ants_dice = rec.get("ants_baseline", {}).get("dice_sym", float("nan"))
            aff_dice = rec.get('syntx_affine_dice_sym', float('nan'))
            aff_str = f"{aff_dice:.4f}" if np.isfinite(aff_dice) else "N/A"
            print(f"CASE_COMPLETE: Pair {args.pair_idx:02d} [{m_name.upper()}] | Affine Dice: {aff_str} | Deform Sym Dice: {rec['syntx_dice_sym']:.4f} (ANTs: {ants_dice:.4f}, diff: {diff:+.2f}%) | Fold: {rec['syntx_fold']:.4f}% | Time: {rec['syntx_time']:.1f}s | Result: {win_str}", flush=True)
        sys.exit(0)

    # 3. Cohort Benchmark mode
    if args.cohort or args.pairs is not None:
        cohort_kwargs = {}
        if args.reg_iterations is not None:
            cohort_kwargs["reg_iterations"] = args.reg_iterations
        if args.learning_rate is not None:
            cohort_kwargs["learning_rate"] = args.learning_rate
        if args.flow_sigma is not None:
            cohort_kwargs["flow_sigma"] = args.flow_sigma
        if args.total_sigma is not None:
            cohort_kwargs["total_sigma"] = args.total_sigma
        if args.optimizer is not None:
            cohort_kwargs["optimizer"] = args.optimizer
        if args.similarity_metric is not None:
            cohort_kwargs["similarity_metric"] = args.similarity_metric
        if args.regularizer is not None:
            cohort_kwargs["regularizer"] = args.regularizer

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
            verbose=args.verbose,
            use_n4=use_n4,
            denoise=args.denoise and not args.no_denoise,
            **cohort_kwargs
        )
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
