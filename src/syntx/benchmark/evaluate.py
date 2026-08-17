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
from typing import Dict, Any, Optional

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.core.utils import normalize_image


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

    Returns
    -------
    Dict[str, Any]
        Structured benchmark metrics dictionary.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Deterministic seeding
    torch.manual_seed(seed + pair_idx)
    np.random.seed(seed + pair_idx)

    # 1. Load Pair Data
    pair_data = load_mindboggle_pair(pair_idx=pair_idx, pairs_csv=pairs_csv, data_dir=data_dir)
    fi_raw, mi_raw = pair_data["fixed"], pair_data["moving"]
    fl, ml = pair_data["fixed_label"], pair_data["moving_label"]
    fixed_id = pair_data["fixed_id"]
    moving_id = pair_data["moving_id"]
    cohort_type = pair_data["pair_type"]

    # 2. Intensity Normalization
    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    # 3. Robust Quick-Search Affine Alignment
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=verbose)
    aff_0 = reg_aff["fwdtransforms"][0]
    t_aff = time.time() - t0_aff

    _, _, aff_dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, reg_aff["fwdtransforms"], reg_aff["invtransforms"], reg_aff.get("whichtoinvert_inv", [True])
    )

    # 4. Deformable Registration
    t0_reg = time.time()
    model_lower = str(model).lower()

    if model_lower == "sobolev":
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
            reg_iterations=[80, 80, 20], similarity_metric="cc2",
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=False, inverse_method="anderson",
            formulation="eulerian", regularizer="sobolev", sobolev_alpha=1.5,
            antisymmetric=True, verbose=verbose
        )
    elif model_lower == "gaussian":
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
            reg_iterations=[80, 80, 20], similarity_metric="cc2",
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=False, inverse_method="anderson",
            formulation="eulerian", regularizer="gaussian",
            antisymmetric=True, verbose=verbose
        )
    elif model_lower == "tvf":
        res_reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=0.211, flow_sigma=0.0, total_sigma=0.2,
            reg_iterations=[80, 80, 20], similarity_metric="cc2",
            multipoint_loss=[0.0, 0.5, 1.0], solver="euler",
            regularizer="gaussian", fast_smooth=True, antisymmetric=True,
            constant_speed=True, constant_speed_relaxation=0.10,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown registration model: '{model}'. Supported: 'sobolev', 'gaussian', 'tvf'")

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
            if verbose:
                print(f"[syntx.benchmark] Generated HTML report at: {report_path}")
        except Exception as e:
            if verbose:
                print(f"[syntx.benchmark] Report generation skipped or failed: {e}", file=sys.stderr)

    return record


# Backward compatibility alias
evaluate_pair = evaluate_mindboggle_pair
