"""
syntx.benchmark.orchestrator — multi-pair Mindboggle runs
=========================================================

``run_mindboggle_benchmark`` runs each (pair, model) as a separate
``python -m syntx.benchmark.cli --pair-idx ...`` subprocess, caches each result as
``<out_dir>/pair_<idx>_<model>.json``, and after every pair rewrites a summary JSON and the
population HTML report (``syntx.viz.create_population_benchmark_report``).
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
from typing import List, Dict, Any, Optional, Set

from syntx.benchmark.data import check_mindboggle_data, DEFAULT_PAIRS_CSV


_MODEL_SETS = {
    ("all", "all_5", "all5"): ["ants", "gaussian", "sobolev", "tvf", "syngs"],
    ("all_4", "all4"): ["ants", "gaussian", "sobolev", "tvf"],
    ("syn_tvf", "sobolev_tvf", "syntx"): ["sobolev", "tvf"],
    ("both", "gauss_sobolev"): ["gaussian", "sobolev"],
    ("continuum_syn", "syn_continuum"): ["sobolev", "syn_navier", "syn_divcurl", "syn_hyperelastic"],
    ("top5_continuum", "continuum_top5", "continuum_5"): ["sobolev", "syn_divcurl", "syn_hyperelastic", "syn_navier", "gaussian"],
    ("all_continuum", "continuum_all"): ["sobolev", "syn_navier", "syn_divcurl", "syn_hyperelastic", "syn_solenoidal", "gaussian"],
}

# CLI options forwarded to the per-pair subprocess (keyword -> flag)
_FORWARDED_OPTIONS = {
    "reg_iterations": "--reg-iterations", "learning_rate": "--learning-rate",
    "flow_sigma": "--flow-sigma", "total_sigma": "--total-sigma", "optimizer": "--optimizer",
    "similarity_metric": "--similarity-metric", "regularizer": "--regularizer",
    "poisson_ratio": "--poisson-ratio", "bulk_modulus": "--bulk-modulus",
}

# result-dict names of the summary arms (ants_syn / greedy_regadam share an arm)
_ARM = {"ants_syn": "ants", "greedy_regadam": "greedy"}


def expand_model_set(model: str) -> List[str]:
    """Models run for ``model``: a model-set name ('all' / 'all_5' / 'all5', 'all_4' /
    'all4', 'syn_tvf' / 'sobolev_tvf' / 'syntx', 'both' / 'gauss_sobolev') or one
    ``evaluate_mindboggle_pair`` model name."""
    for names, models in _MODEL_SETS.items():
        if model in names:
            return list(models)
    return [model]


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
    use_n4: bool = True,
    denoise: bool = False,
    ants_baseline_dir: str = "results",
    **kwargs: Any
) -> Dict[str, Any]:
    """Run a set of Mindboggle pairs through one or more models, one subprocess per run.

    Procedure: check the data (``check_mindboggle_data``; RuntimeError if incomplete); load
    previous per-model results from ``summary_json`` if it exists (a corrupt file warns);
    load ANTs baselines for every pair of ``pairs_csv`` from ``<out_dir>/pair_<idx>_ants.json``
    or ``<ants_baseline_dir>/pair_<idx>_ants_syn.json`` when not already in the summary; then
    for each pair (in the chosen order) and each model of that pair, reuse
    ``<out_dir>/pair_<idx>_<name>.json`` (name = model, + '_denoised' with ``denoise``) if it
    holds status 'SUCCESS' (unless ``force``), otherwise run ``python -u -m
    syntx.benchmark.cli --pair-idx <idx> --model <model> --out-dir <out_dir> --pairs-csv
    <pairs_csv> --seed <seed> ...``. A failing subprocess is reported on stderr and skipped.
    Pairs where the result's 'diff_vs_ants' is negative are printed as outliers. After each
    pair the summary JSON and the HTML report are rewritten (a report error warns).

    Parameters
    ----------
    pairs : list of int, optional
        Pair indices; default all rows of ``pairs_csv``.
    model : str, default 'sobolev'
        A model set or one model (``expand_model_set``); a single model also runs
        'gaussian' on ``probe_pairs`` (unless the model is 'gaussian'). Every model's
        results are summarised under '<arm>_results' ('ants_syn' -> 'ants',
        'greedy_regadam' -> 'greedy').
    probe_pairs : set of int, optional
        Pairs that also get a 'gaussian' run in single-model mode. Default {0, 1, 2, 45, 67,
        82} when ``pairs`` is None, else empty.
    random_order : bool, default True
        Visit the pairs in ``numpy.random.RandomState(seed).permutation`` order.
    seed : int, default 42
        Seed of that permutation, also passed to every run (``--seed``).
    out_dir : str, default 'results/reproducible_eval'
        Per-run JSON directory (created).
    summary_json : str, default 'results/reproducible_90pair_master_summary.json'
        Summary file: 'timestamp', 'total_completed', 'total_planned', 'permutation_order',
        'primary_model', 'planned_models', and '<arm>_results' dicts (pair -> record). Read
        at start (resume) and rewritten after each pair.
    report_html : str, default 'docs/reproducible_90pair_report.html'
        Population HTML report path.
    generate_example_reports : bool, default False
        Pass ``--generate-report`` for pairs in ``example_report_pairs``.
    example_report_pairs : list of int, optional
        Default [0, 67].
    pairs_csv : str, default ``DEFAULT_PAIRS_CSV``
        Pairs CSV.
    data_dir : str, optional
        Mindboggle data directory (passed as ``--data-dir``).
    force : bool, default False
        Rerun even when a cached per-run JSON exists. Results already in ``summary_json``
        are still loaded first, and are replaced only when the rerun succeeds.
    verbose : bool, default False
        Print progress (running means of 'syntx_dice_sym' per arm). The outlier and error
        lines are printed regardless.
    use_n4 : bool, default True
        False passes ``--no-n4``.
    denoise : bool, default False
        True passes ``--denoise``.
    ants_baseline_dir : str, default 'results'
        Directory of the ``pair_<idx>_ants_syn.json`` baselines.
    **kwargs
        Forwarded as CLI options when not None: ``reg_iterations``, ``learning_rate``,
        ``flow_sigma``, ``total_sigma``, ``optimizer``, ``similarity_metric``,
        ``regularizer``. Any other keyword raises TypeError.

    Returns
    -------
    dict
        'total_completed' (planned pairs with a result for every planned model),
        'completed_by_model' ({model: number of planned pairs with a result}),
        'summary_json', 'report_html' (absolute paths), 'runtime_minutes'.
    """
    unknown = sorted(set(kwargs) - set(_FORWARDED_OPTIONS))
    if unknown:
        raise TypeError(f"run_mindboggle_benchmark: unknown keyword(s) {unknown}; "
                        f"forwarded options are {sorted(_FORWARDED_OPTIONS)}")
    import warnings

    # 1. Check Dataset Integrity
    t0_benchmark = time.time()
    is_valid, report = check_mindboggle_data(pairs_csv=pairs_csv, data_dir=data_dir, verbose=verbose)
    if not is_valid:
        raise RuntimeError(
            f"Cannot run Mindboggle benchmark: Missing data. "
            f"Found {report['available_pairs']}/{report['total_pairs_in_csv']} pairs."
        )

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_html)), exist_ok=True)

    n_csv = int(report["total_pairs_in_csv"])
    if pairs is None:
        pairs = list(range(n_csv))
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

    def models_for(pair_idx):
        ms = expand_model_set(model)
        if len(ms) == 1 and pair_idx in probe_pairs and model != "gaussian":
            ms = ms + ["gaussian"]
        return ms

    total_pairs = len(ordered_pairs)
    if verbose:
        print("=" * 78)
        print(f"  Syntx Mindboggle Benchmark Runner ({total_pairs} Pairs)")
        print(f"  Primary Model: {model.upper()} | Randomized: {random_order} (Seed {seed})")
        print(f"  Dual-Arm Probe Set (Gaussian): {sorted(list(probe_pairs))}")
        print("=" * 78, flush=True)

    # arm name -> {pair_idx: record}
    results: Dict[str, Dict[int, Any]] = {
        a: {} for a in ("ants", "sobolev", "gaussian", "tvf", "syngs", "greedy",
                        "syn_navier", "syn_divcurl", "syn_hyperelastic", "syn_solenoidal")
    }
    arm = lambda m: _ARM.get(m, m)

    if os.path.exists(summary_json):
        try:
            with open(summary_json, "r") as f:
                existing_summary = json.load(f)
            for key, val in existing_summary.items():
                if key.endswith("_results") and isinstance(val, dict):
                    bucket = results.setdefault(key[:-len("_results")], {})
                    for k, v in val.items():
                        try:
                            bucket[int(k)] = v
                        except ValueError:
                            pass
        except Exception as e:
            warnings.warn(f"run_mindboggle_benchmark: could not read {summary_json} ({e}); starting fresh")

    # Ensure all available ANTs baselines are loaded from disk cache
    for p_i in range(n_csv):
        if p_i not in results["ants"]:
            for af in (os.path.join(out_dir, f"pair_{p_i:03d}_ants.json"),
                       os.path.join(ants_baseline_dir, f"pair_{p_i:03d}_ants_syn.json")):
                if os.path.exists(af):
                    try:
                        with open(af, "r") as f:
                            results["ants"][p_i] = json.load(f)
                        break
                    except Exception as e:
                        warnings.warn(f"run_mindboggle_benchmark: unreadable baseline {af} ({e})")

    suffix = "_denoised" if denoise else ""

    def completed_counts():
        by_model = {}
        done = 0
        for p in ordered_pairs:
            ms = models_for(p)
            ok = [p in results.get(arm(m), {}) for m in ms]
            for m, o in zip(ms, ok):
                by_model[m] = by_model.get(m, 0) + int(o)
            done += int(all(ok))
        return done, by_model

    for step_num, pair_idx in enumerate(ordered_pairs, start=1):
        for m_type in models_for(pair_idx):
            out_file = os.path.join(out_dir, f"pair_{pair_idx:03d}_{m_type}{suffix}.json")

            # Check cache / resume
            if not force and os.path.exists(out_file):
                try:
                    with open(out_file, "r") as f:
                        rec = json.load(f)
                    if rec.get("status") == "SUCCESS":
                        results.setdefault(arm(m_type), {})[pair_idx] = rec
                        if verbose:
                            print(f"[{step_num}/{total_pairs}] Pair {pair_idx:02d} [{m_type.upper()}]: Resumed from cache (Dice = {rec.get('syntx_dice_sym', float('nan')):.4f})", flush=True)
                        continue
                except Exception as e:
                    warnings.warn(f"run_mindboggle_benchmark: unreadable cached result {out_file} ({e}); rerunning")

            # Run in isolated subprocess
            if verbose:
                print(f"[{step_num}/{total_pairs}] Launching Pair {pair_idx:02d} [{m_type.upper()}] in isolated subprocess...", flush=True)

            cmd = [
                sys.executable, "-u", "-m", "syntx.benchmark.cli",
                "--pair-idx", str(pair_idx),
                "--model", str(m_type),
                "--out-dir", str(out_dir),
                "--pairs-csv", str(pairs_csv),
                "--seed", str(seed),
            ]
            if data_dir:
                cmd.extend(["--data-dir", str(data_dir)])
            if not use_n4:
                cmd.append("--no-n4")
            if denoise:
                cmd.append("--denoise")
            if generate_example_reports and pair_idx in example_report_pairs:
                cmd.append("--generate-report")
            for key, flag in _FORWARDED_OPTIONS.items():
                v = kwargs.get(key)
                if v is None:
                    continue
                cmd.extend([flag] + ([str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]))

            res = subprocess.run(cmd, capture_output=False)
            if res.returncode != 0:
                print(f"[syntx.benchmark] ERROR: Subprocess failed on Pair {pair_idx:02d} [{m_type}] (exit {res.returncode})", file=sys.stderr)
            elif os.path.exists(out_file):
                with open(out_file, "r") as f:
                    rec = json.load(f)
                results.setdefault(arm(m_type), {})[pair_idx] = rec

                diff = rec.get("diff_vs_ants", float("nan"))
                if np.isfinite(diff) and diff < 0.0:
                    aff_d = rec.get("syntx_affine_dice_sym", float("nan"))
                    def_d = rec.get("syntx_dice_sym", float("nan"))
                    ants_d = rec.get("ants_baseline", {}).get("dice_sym", float("nan"))
                    print(f"  ⚠️ OUTLIER DETECTED: Pair {pair_idx:02d} [{m_type.upper()}] | Deform: {def_d:.4f} vs ANTs: {ants_d:.4f} ({diff:+.2f}%) | Affine Dice: {aff_d:.4f}", flush=True)

        # Intermediate progress logging and master summary sync
        n_done, _ = completed_counts()
        if verbose:
            means = []
            for a_name, recs in results.items():
                vals = [r.get("syntx_dice_sym", float("nan")) for r in recs.values()]
                vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
                if vals:
                    means.append(f"{a_name}: {np.mean(vals):.4f}")
            print(f"  PROGRESS: {n_done}/{total_pairs} pairs complete | " + " | ".join(means), flush=True)

        master_summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_completed": n_done,
            "total_planned": total_pairs,
            "permutation_order": ordered_pairs,
            "primary_model": model,
            "planned_models": sorted({m for p in ordered_pairs for m in models_for(p)}),
        }
        master_summary.update({f"{a_name}_results": recs for a_name, recs in results.items()})
        with open(summary_json, "w") as f:
            json.dump(master_summary, f, indent=2)
        _write_population_report(summary_json, report_html, model, total_pairs)

    total_time = time.time() - t0_benchmark
    n_done, by_model = completed_counts()
    if verbose:
        print(f"[syntx.benchmark] Master HTML dashboard: {report_html}")
        print("=" * 78)
        print(f"  BENCHMARK COMPLETE: {n_done}/{total_pairs} pairs in {total_time/60.0:.1f} minutes")
        print("=" * 78, flush=True)

    return {
        "total_completed": n_done,
        "completed_by_model": by_model,
        "summary_json": os.path.abspath(summary_json),
        "report_html": os.path.abspath(report_html),
        "runtime_minutes": total_time / 60.0
    }


def _write_population_report(summary_json, report_html, model, total_pairs):
    """Rewrite the population report from the summary; a failure warns."""
    import warnings
    if not os.path.exists(summary_json):
        return
    try:
        from syntx.viz import create_population_benchmark_report
        create_population_benchmark_report(
            results_source=summary_json,
            output_html=report_html,
            title=f"Syntx {model} vs ANTs C++ — {total_pairs}-Pair Mindboggle Benchmark Report",
        )
    except Exception as e:
        warnings.warn(f"run_mindboggle_benchmark: population report failed ({e})")
