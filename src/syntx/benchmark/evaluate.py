"""
syntx.benchmark.evaluate — Standardized Single-Pair Registration & Metric Evaluation
=====================================================================================

Executes robust affine pre-alignment and nonlinear deformable registration
(Sobolev SyN, Gaussian SyN, or TVF) on a single Mindboggle evaluation pair,
extracting complete topological, accuracy, and inverse consistency metrics.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import ants
from typing import Dict, Any, Optional, Union, List

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.core.utils import normalize_image


def clean_device_cache():
    """
    Clears PyTorch GPU / Apple Silicon MPS memory allocator cache and runs garbage collection.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def normalize_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
    """
    Automatic entropy-optimal foreground intensity normalization.
    Conforms to syntx registration guardrails.
    """
    return normalize_image(img, method='auto')


def evaluate_mindboggle_pair(
    pair_idx: int = 0,
    model: str = "sobolev",
    device: Optional[str] = None,
    pairs_csv: str = "examples/pairs.csv",
    data_dir: Optional[str] = None,
    ants_baseline_dir: str = "results",
    generate_report: bool = False,
    report_out_dir: Optional[str] = None,
    verbose: bool = False,
    seed: int = 42,
    dataset_key: Optional[str] = None,
    config: Optional[dict] = None,
    use_n4: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates a single Mindboggle registration pair under the specified model variant.

    Parameters
    ----------
    pair_idx : int
        Index of the pair (0 to 89).
    model : str
        Registration algorithm/regularizer ('sobolev', 'gaussian', 'tvf').
    device : str, optional
        Compute device ('mps', 'cuda', 'cpu'). If None, automatically detected.
    pairs_csv : str
        Path to pairs CSV configuration file.
    data_dir : str, optional
        Mindboggle data root directory.
    ants_baseline_dir : str
        Directory containing existing ANTs C++ baseline result files.
    generate_report : bool
        If True, generates a standalone interactive HTML diagnostic report with
        the complete visual verification suite via `syntx.viz`.
    report_out_dir : str, optional
        Output directory for single-pair HTML reports.
    verbose : bool
        If True, prints intermediate progress details.
    seed : int
        Random seed for reproducibility.
    use_n4 : bool, default=True
        If True, preprocesses input images with ANTsTorch N4 bias field correction.

    Returns
    -------
    Dict[str, Any]
        Structured benchmark metrics dictionary.
    """
    clean_device_cache()

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Deterministic seeding
    torch.manual_seed(seed + pair_idx)
    np.random.seed(seed + pair_idx)

    # 1. Load Pair Data
    pair_data = load_mindboggle_pair(pair_idx=pair_idx, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=use_n4)
    fi_raw, mi_raw = pair_data["fixed"], pair_data["moving"]
    fl, ml = pair_data["fixed_label"], pair_data["moving_label"]
    fixed_id = pair_data["fixed_id"]
    moving_id = pair_data["moving_id"]
    cohort_type = pair_data["pair_type"]

    # 2. Intensity Normalization
    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    # 3. Canonical Affine Alignment (Shared Across All 4 Methods)
    canonical_affine_dir = "results/canonical_affines"
    os.makedirs(canonical_affine_dir, exist_ok=True)
    aff_mat_path = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}_affine.mat")
    aff_info_path = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}_affine_info.json")

    aff_0 = None
    if os.path.exists(aff_mat_path) and os.path.exists(aff_info_path):
        try:
            with open(aff_info_path, "r") as f:
                aff_info = json.load(f)
            aff_0 = aff_mat_path
            t_aff = float(aff_info.get("runtime_seconds", 0.0))
            aff_dice_sym = float(aff_info.get("dice_sym", 0.0))
        except Exception:
            aff_0 = None

    if aff_0 is None:
        t0_aff = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=verbose)
        t_aff = time.time() - t0_aff
        import shutil
        shutil.copyfile(reg_aff["fwdtransforms"][0], aff_mat_path)
        aff_0 = aff_mat_path

        clean_device_cache()
        _, _, aff_dice_sym = compute_bidirectional_dice(
            fl, ml, fi, mi, [aff_mat_path], [aff_mat_path], [True]
        )
        with open(aff_info_path, "w") as f:
            json.dump({
                "dice_sym": float(aff_dice_sym),
                "runtime_seconds": float(t_aff),
                "pair_idx": pair_idx
            }, f, indent=2)

    clean_device_cache()

    # 4. Deformable Registration
    t0_reg = time.time()
    model_lower = str(model).lower()

    # Allow parameter overrides from kwargs or config
    reg_iters = kwargs.get("reg_iterations") or (config and config.get("params", {}).get("reg_iterations")) or [100, 100, 20]
    grad_step = kwargs.get("grad_step") or (config and config.get("params", {}).get("grad_step")) or 0.25
    flow_sigma = kwargs.get("flow_sigma") if "flow_sigma" in kwargs else (config and config.get("params", {}).get("flow_sigma", 3.0)) if config else 3.0
    total_sigma = kwargs.get("total_sigma") if "total_sigma" in kwargs else (config and config.get("params", {}).get("total_sigma", 0.0)) if config else 0.0
    fast_smooth = kwargs.get("fast_smooth") if "fast_smooth" in kwargs else (config and config.get("fast_smooth", False)) if config else False

    if model_lower in ("sobolev", "syn_sobolev") or (model_lower == "syn" and (kwargs.get("regularizer") == "sobolev" or (config and config.get("regularizer") == "sobolev"))):
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=grad_step, flow_sigma=flow_sigma, total_sigma=total_sigma,
            reg_iterations=reg_iters, similarity_metric="cc2",
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=fast_smooth, inverse_method="anderson",
            formulation="eulerian", regularizer="sobolev", sobolev_alpha=1.5,
            antisymmetric=True, verbose=verbose
        )
    elif model_lower in ("gaussian", "syn_gaussian", "syn"):
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=grad_step, flow_sigma=flow_sigma, total_sigma=total_sigma,
            reg_iterations=reg_iters, similarity_metric="cc2",
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=fast_smooth, inverse_method="anderson",
            formulation="eulerian", regularizer="gaussian",
            antisymmetric=True, verbose=verbose
        )
    elif model_lower == "tvf":
        res_reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            regularizer="sobolev",
            flow_sigma=0.0,
            total_sigma=0.035,
            alpha=0.035,
            sobolev_alpha=0.035,
            optimizer="adam",
            optimizer_lr=0.8,
            multipoint_loss=[0.0, 0.5, 1.0],
            antisymmetric=False,
            reg_iterations=reg_iters if reg_iters != [100, 100, 20] else [100, 100, 6],
            solver="euler",
            integration_steps_per_interval=6,
            constant_speed=True,
            constant_speed_relaxation=0.10,
            verbose=verbose
        )
    elif model_lower in ("ants", "ants_syn"):
        res_reg = ants.registration(
            fixed=fi, moving=mi, typeofTransform="SyN",
            initial_transform=aff_0, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown registration model: '{model}'. Supported: 'ants', 'sobolev', 'gaussian', 'tvf'")

    t_reg = time.time() - t0_reg + t_aff

    # 5. Evaluate Structural and Topological Metrics
    fwd_tx = res_reg["fwdtransforms"]
    inv_tx = res_reg["invtransforms"]
    which_inv = res_reg.get("whichtoinvert_inv", [True, False])

    df_fixed, df_moving, dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, fwd_tx, inv_tx, which_inv
    )

    fwd_warp_file = next(x for x in fwd_tx if isinstance(x, str) and x.endswith(".nii.gz"))
    jac = compute_jacobian_metrics(fi, fwd_warp_file)

    inv_errs = res_reg.get("inverse_identity_errors", {})
    if "phi_1" in inv_errs:
        inv_mean = float(inv_errs["phi_1"].get("mean", float("nan")))
        inv_p95 = float(inv_errs["phi_1"].get("p95", float("nan")))
    else:
        inv_mean = float(inv_errs.get("mean", float("nan")))
        inv_p95 = float(inv_errs.get("p95", float("nan")))

    # 6. Load Matched ANTs C++ Baseline
    ants_baseline_file = os.path.join(ants_baseline_dir, f"pair_{pair_idx:03d}_ants_syn.json")
    ants_rec = {}
    if os.path.exists(ants_baseline_file):
        try:
            with open(ants_baseline_file, "r") as f:
                ants_rec = json.load(f)
        except Exception:
            pass

    ants_dice_sym = float(ants_rec.get("dice_sym", float("nan")))
    ants_dice_f = float(ants_rec.get("dice_fixed", float("nan")))
    ants_dice_m = float(ants_rec.get("dice_moving", float("nan")))
    ants_fold = float(ants_rec.get("folding_pct", 0.0))
    ants_min_jac = float(ants_rec.get("min_jacobian", 0.0))
    ants_time = float(ants_rec.get("runtime_seconds", float("nan")))

    diff_vs_ants = (dice_sym - ants_dice_sym) * 100.0 if np.isfinite(ants_dice_sym) else float("nan")
    win = bool(np.isfinite(ants_dice_sym) and dice_sym >= ants_dice_sym)

    record = {
        "pair_idx": int(pair_idx),
        "model_type": model_lower,
        "cohort_type": cohort_type,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "use_n4": use_n4,
        "status": "SUCCESS",
        "syntx_affine_dice_sym": float(aff_dice_sym),
        "syntx_dice_sym": float(dice_sym),
        "syntx_dice_fixed": float(df_fixed),
        "syntx_dice_moving": float(df_moving),
        "syntx_fold": float(jac["folding_pct"]),
        "syntx_min_jac": float(jac["min"]),
        "syntx_inv_mean": float(inv_mean),
        "syntx_inv_p95": float(inv_p95),
        "syntx_time": float(t_reg),
        "diff_vs_ants": float(diff_vs_ants),
        "win": win,
        "ants_baseline": {
            "dice_sym": ants_dice_sym,
            "dice_fixed": ants_dice_f,
            "dice_moving": ants_dice_m,
            "folding_pct": ants_fold,
            "min_jacobian": ants_min_jac,
            "runtime_seconds": ants_time
        },
        "transforms": {
            "fwdtransforms": [str(x) for x in fwd_tx],
            "invtransforms": [str(x) for x in inv_tx],
            "whichtoinvert_inv": which_inv
        }
    }

    # 7. Optional Standalone HTML Report Generation
    if generate_report:
        try:
            from syntx.viz import create_registration_report
            if report_out_dir is None:
                report_out_dir = "docs/reports"
            os.makedirs(report_out_dir, exist_ok=True)
            report_path = os.path.join(
                report_out_dir, f"report_pair_{pair_idx:03d}_{model_lower}.html"
            )
            create_registration_report(
                fixed=fi,
                moving=mi,
                warped=res_reg.get("warpedmovout", fi),
                warp=fwd_warp_file,
                output_html=report_path,
                fixed_name=f"Fixed ({fixed_id})",
                moving_name=f"Moving ({moving_id})",
                reg=res_reg,
                dice_overlap=float(dice_sym)
            )
            record["report_html"] = os.path.abspath(report_path)
        except Exception as e:
            if verbose:
                print(f"[syntx.benchmark] Report generation skipped or failed: {e}", file=sys.stderr)

    clean_device_cache()
    return record


# Backward compatibility alias
evaluate_pair = evaluate_mindboggle_pair


def run_standard_report_demo(
    dataset_key: str = "mbhard",
    output_html: str = "docs/reports/mbhard_standard_report.html",
    model: str = "gaussian",
    device: Optional[str] = None,
    reg_iterations: list = None,
    verbose: bool = False
) -> str:
    """
    Runs a demonstration deformable registration on `mbhard` (or 2D `r16_r64`)
    and generates a complete publication-quality 5-figure HTML diagnostic report.

    Parameters
    ----------
    dataset_key : str
        Dataset identifier ('mbhard', 'r16_r64', 'c', 'ellipse').
    output_html : str
        Target filepath for generated HTML diagnostic report.
    model : str
        Registration regularizer ('gaussian', 'sobolev', 'tvf').
    device : str, optional
        Compute device ('cuda', 'mps', 'cpu'). If None, automatically detected.
    reg_iterations : list, optional
        Multi-resolution iteration schedule (e.g. [100, 100, 20] or [40, 40, 10]).
    verbose : bool
        If True, prints progress details.

    Returns
    -------
    str
        Absolute path to the generated HTML diagnostic report.
    """
    from syntx.generators import benchmark_data
    from syntx.robust_affine import robust_affine
    from syntx.viz import create_registration_report

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if verbose:
        print(f"[run_standard_report_demo] Loading dataset '{dataset_key}' on device: {device.upper()}...", flush=True)

    data = benchmark_data(dataset_key)
    fi_raw = data["fixed"]
    mi_raw = data["moving"]
    fl = data.get("fixed_label")
    ml = data.get("moving_label")

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    if verbose:
        print("[run_standard_report_demo] Step 1/2: Running robust multi-start affine initialization...", flush=True)

    reg_aff = robust_affine(fi, mi, mode="auto", verbose=False)
    aff_tx = reg_aff["fwdtransforms"][0]

    if reg_iterations is None:
        reg_iterations = [80, 80, 20] if fi.dimension == 3 else [100, 100, 50]

    if verbose:
        print(f"[run_standard_report_demo] Step 2/2: Running deformable {model.upper()} SyN ({reg_iterations})...", flush=True)

    if model.lower() == "tvf":
        res_reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            device=device, reg_iterations=reg_iterations,
            verbose=verbose
        )
    else:
        regularizer = "sobolev" if model.lower() in ("sobolev", "syn_sobolev") else "gaussian"
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            backend="pytorch", device=device,
            grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
            reg_iterations=reg_iterations, similarity_metric="cc2",
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, inverse_method="anderson", formulation="eulerian",
            regularizer=regularizer, antisymmetric=True,
            verbose=verbose
        )

    dice_val = None
    if fl is not None and ml is not None:
        try:
            _, _, dice_val = compute_bidirectional_dice(
                fl, ml, fi, mi,
                res_reg["fwdtransforms"],
                res_reg["invtransforms"],
                res_reg.get("whichtoinvert_inv", [True, False])
            )
            if verbose:
                print(f"[run_standard_report_demo] Symmetric Cortical Mean Dice: {dice_val:.4f}", flush=True)
        except Exception:
            pass

    fwd_warp = next((x for x in res_reg["fwdtransforms"] if isinstance(x, str) and x.endswith(".nii.gz")), None)

    rep_dict = create_registration_report(
        fixed=fi,
        moving=mi,
        warped=res_reg.get("warpedmovout", fi),
        warp=fwd_warp,
        fixed_label=fl,
        moving_label=ml,
        output_html=output_html,
        fixed_name=f"Fixed ({data.get('description', dataset_key)})",
        moving_name=f"Moving ({data.get('description', dataset_key)})",
        reg=res_reg,
        dice_overlap=float(dice_val) if dice_val is not None else None
    )

    out_path = rep_dict.get("html_path", os.path.abspath(output_html))
    if verbose:
        print(f"[run_standard_report_demo] Report generated successfully: {out_path}", flush=True)

    return out_path


def evaluate_affine_benchmark(
    pairs: Union[int, List[int], str] = "inter16",
    modes: List[str] = ['ants_fast', 'pytorch', 'auto', 'com_only'],
    pairs_csv: str = "examples/pairs.csv",
    data_dir: Optional[str] = None,
    verbose: bool = True,
    generate_report: bool = False,
    output_html: Optional[str] = None,
    use_n4: bool = True
) -> Any:
    """
    Official Mindboggle Affine Registration Benchmark Suite.

    Evaluates and benchmarks multiple affine registration modes ('ants_fast', 'pytorch', 'auto', 'com_only')
    across single or multi-pair Mindboggle cohorts (intra-site and inter-site).

    Parameters
    ----------
    pairs : int, list of int, or str
        Pair index (e.g. 0), list of pair indices (e.g. range(40, 56) for 16 inter-study pairs),
        or special keywords: 'mbhard', 'inter16', 'intra16', 'all'.
    modes : list of str
        Affine registration modes to benchmark. Default: ['ants_fast', 'pytorch', 'auto', 'com_only'].
    pairs_csv : str
        Path to pairs CSV configuration file.
    data_dir : str, optional
        Root directory of Mindboggle dataset.
    verbose : bool
        Whether to print progress.
    generate_report : bool
        Whether to compile interactive HTML benchmark report.
    output_html : str, optional
        File path to save the HTML benchmark report.

    Returns
    -------
    pd.DataFrame
        Structured DataFrame containing benchmark results per pair and mode.
    """
    import pandas as pd
    from syntx.deformation_metrics import compute_bidirectional_dice
    from syntx.benchmark.data import load_mindboggle_pair

    # Resolve pair index list
    if isinstance(pairs, str):
        pairs_str = pairs.lower().strip()
        if pairs_str == "mbhard":
            pair_list = [44]
        elif pairs_str == "inter16":
            # 16 inter-study pairs starting from index 40
            pair_list = list(range(40, 56))
        elif pairs_str == "intra16":
            # First 16 intra-study pairs
            pair_list = list(range(0, 16))
        elif pairs_str == "all":
            pair_list = list(range(0, 90))
        else:
            try:
                pair_list = [int(pairs_str)]
            except ValueError:
                pair_list = [0]
    elif isinstance(pairs, int):
        pair_list = [pairs]
    else:
        pair_list = list(pairs)

    records = []

    for idx in pair_list:
        try:
            pair_data = load_mindboggle_pair(idx, pairs_csv=pairs_csv, data_dir=data_dir, use_n4=use_n4)
        except Exception as e:
            if verbose:
                print(f"[Affine Benchmark] Skipping pair {idx} due to loading error: {e}", flush=True)
            continue

        fi = pair_data['fixed']
        mi = pair_data['moving']
        fl = pair_data['fixed_label']
        ml = pair_data['moving_label']
        pair_type = pair_data.get('pair_type', 'inter' if idx >= 40 else 'intra')
        cohort1 = pair_data.get('cohort1', '')
        cohort2 = pair_data.get('cohort2', '')

        if verbose:
            print(f"\n--- [Affine Benchmark] Evaluating Pair {idx:02d} ({pair_type.upper()}: {cohort1} -> {cohort2}) ---", flush=True)

        for m in modes:
            t0 = time.time()
            try:
                reg = syntx.robust_affine(fi, mi, mode=m, verbose=False)
                t_el = time.time() - t0
                fwd = reg['fwdtransforms']
                inv = reg['invtransforms']
                d_f, d_m, d_sym = compute_bidirectional_dice(
                    fl, ml, fi, mi, fwd, inv,
                    whichtoinvert_inv=reg.get('whichtoinvert_inv', [True] + [False]*(len(inv)-1))
                )
            except Exception as e:
                if verbose:
                    print(f"  Mode '{m}' failed on Pair {idx}: {e}", flush=True)
                d_f, d_m, d_sym, t_el = 0.0, 0.0, 0.0, 0.0

            if verbose:
                print(f"  Mode: {m:<10} | Sym DICE: {d_sym:.4f} (Fixed: {d_f:.4f}, Moving: {d_m:.4f}) | Time: {t_el:.2f}s", flush=True)

            records.append({
                'pair_idx': idx,
                'pair_type': pair_type,
                'cohorts': f"{cohort1}->{cohort2}",
                'mode': m,
                'dice_fixed': d_f,
                'dice_moving': d_m,
                'dice_sym': d_sym,
                'runtime_seconds': t_el
            })

    df = pd.DataFrame(records)

    if generate_report or output_html is not None:
        from syntx.viz.reports import create_affine_benchmark_report
        out_file = output_html if output_html else "docs/reports/affine_benchmark_report.html"
        create_affine_benchmark_report(df, output_html=out_file)
        if verbose:
            print(f"\n[Affine Benchmark] HTML report saved to: {out_file}", flush=True)

    return df

