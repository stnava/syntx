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


# Identifies which affine engine produced a cached canonical affine.  Bump when the affine
# backend or its defaults change so stale caches are recomputed instead of silently reused.
AFFINE_BACKEND_KEY = "pt7"


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
    denoise: bool = False,
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
    denoise : bool, default=False
        If True, applies ANTsTorch non-local means denoising (`antstorch.denoise_image`)
        prior to intensity normalization.

    Returns
    -------
    Dict[str, Any]
        Structured benchmark metrics dictionary.
    """
    clean_device_cache()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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

    # 2. Intensity Normalization & Optional Denoising
    denoise_device = None
    if denoise:
        try:
            import antstorch
            # Be explicit about the device rather than relying on antstorch's default: a silent
            # drop to CPU here costs minutes per pair and is invisible in the results.
            denoise_device = device or ("cuda" if torch.cuda.is_available()
                                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
            fi_raw = antstorch.denoise_image(fi_raw, shrink_factor=2, p=1, r=1, noise_model="Rician", device=denoise_device)
            mi_raw = antstorch.denoise_image(mi_raw, shrink_factor=2, p=1, r=1, noise_model="Rician", device=denoise_device)
            if verbose:
                print(f"[evaluate_mindboggle_pair] Applied antstorch.denoise_image on {denoise_device} (Rician, shrink_factor=2, r=1)")
        except Exception as e:
            denoise_device = None
            if verbose:
                print(f"[evaluate_mindboggle_pair] Warning: antstorch.denoise_image failed ({e}), continuing with raw.")

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    # 3. Canonical Affine Alignment (Shared Across All 4 Methods)
    canonical_affine_dir = "results/canonical_affines"
    os.makedirs(canonical_affine_dir, exist_ok=True)
    aff_suffix = "_denoised" if denoise else ""
    # Cache key includes the affine backend: 'auto' dispatches to the PyTorch solver since
    # 2026-09-15, so ANTs-seeded caches from earlier runs must not be silently reused.
    aff_suffix += f"_{AFFINE_BACKEND_KEY}"
    aff_mat_path = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}{aff_suffix}_affine.mat")
    aff_info_path = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}{aff_suffix}_affine_info.json")

    aff_0 = None
    if os.path.exists(aff_mat_path) and os.path.exists(aff_info_path):
        try:
            with open(aff_info_path, "r") as f:
                aff_info = json.load(f)
            if aff_info.get("affine_backend") != AFFINE_BACKEND_KEY:
                raise ValueError("cached affine was produced by a different backend")
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
                "pair_idx": pair_idx,
                "affine_backend": AFFINE_BACKEND_KEY
            }, f, indent=2)

    clean_device_cache()

    # 4. Deformable Registration
    t0_reg = time.time()
    model_lower = str(model).lower()

    # Allow parameter overrides from kwargs or config
    user_reg_iters = kwargs.pop("reg_iterations", None) or (config and config.get("params", {}).get("reg_iterations"))
    user_grad_step = kwargs.pop("learning_rate", kwargs.pop("grad_step", None)) or (config and config.get("params", {}).get("grad_step"))
    user_flow_sigma = kwargs.pop("flow_sigma", None) if "flow_sigma" in kwargs else (config and config.get("params", {}).get("flow_sigma"))
    user_total_sigma = kwargs.pop("total_sigma", None) if "total_sigma" in kwargs else (config and config.get("params", {}).get("total_sigma"))
    fast_smooth = kwargs.pop("fast_smooth", None) if "fast_smooth" in kwargs else ((config and config.get("fast_smooth", False)) if config else False)

    if model_lower in ("affine", "affine_default"):
        res_reg = {
            "fwdtransforms": [aff_0],
            "invtransforms": [aff_0],
            "whichtoinvert_inv": [True],
            "warpedmovout": ants.apply_transforms(fi, mi, [aff_0]),
            "runtime_seconds": t_aff,
            "inverse_identity_errors": {"mean": 0.0, "p95": 0.0},
        }
    elif model_lower in ("affine_fast", "affine_accurate"):
        preset = "fast" if "fast" in model_lower else "accurate"
        t0_aff_preset = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", preset=preset, verbose=verbose)
        t_aff_preset = time.time() - t0_aff_preset
        res_reg = {
            "fwdtransforms": reg_aff["fwdtransforms"],
            "invtransforms": reg_aff["fwdtransforms"],
            "whichtoinvert_inv": [True],
            "warpedmovout": reg_aff.get("warpedmovout", ants.apply_transforms(fi, mi, reg_aff["fwdtransforms"])),
            "runtime_seconds": t_aff_preset,
            "inverse_identity_errors": {"mean": 0.0, "p95": 0.0},
        }
    elif model_lower in ("sobolev", "syn_sobolev"):
        syn_iters = user_reg_iters if user_reg_iters is not None else [100, 100, 20]
        syn_step = user_grad_step if user_grad_step is not None else 0.25
        syn_flow = user_flow_sigma if user_flow_sigma is not None else 3.0
        syn_total = user_total_sigma if user_total_sigma is not None else 0.0
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=syn_step, flow_sigma=syn_flow, total_sigma=syn_total,
            reg_iterations=syn_iters, similarity_metric="cc2",
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=True, inverse_method="anderson",
            formulation="eulerian", regularizer="sobolev", sobolev_alpha=1.5,
            antisymmetric=True, verbose=verbose
        )
    elif model_lower in ("gaussian", "syn_gaussian", "syn", "syn_mi"):
        syn_iters = user_reg_iters if user_reg_iters is not None else [100, 100, 20]
        syn_step = user_grad_step if user_grad_step is not None else 0.25
        syn_flow = user_flow_sigma if user_flow_sigma is not None else 3.0
        syn_total = user_total_sigma if user_total_sigma is not None else 0.0
        default_metric = "mattes_mi" if model_lower == "syn_mi" else "cc2"
        syn_metric = kwargs.pop("similarity_metric", default_metric)
        syn_kernel = kwargs.pop("kernel_type", "gaussian")
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=syn_step, flow_sigma=syn_flow, total_sigma=syn_total,
            reg_iterations=syn_iters, similarity_metric=syn_metric,
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=True, inverse_method="anderson",
            formulation="eulerian", regularizer="gaussian", kernel_type=syn_kernel,
            antisymmetric=True, verbose=verbose, **kwargs
        )
    elif model_lower in ("syn_regadam", "syn_dsti1", "regadam_syn"):
        syn_iters = user_reg_iters if user_reg_iters is not None else [100, 50, 10]
        syn_step = user_grad_step if user_grad_step is not None else 0.50
        syn_flow = user_flow_sigma if user_flow_sigma is not None else 3.0
        syn_total = user_total_sigma if user_total_sigma is not None else 0.0
        syn_metric = kwargs.pop("similarity_metric", "cc2")
        syn_reg = "dsti1" if "dsti" in model_lower else kwargs.pop("regularizer", "dsti1")
        syn_opt_lr = kwargs.pop("optimizer_lr", 1.0)
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=syn_step, flow_sigma=syn_flow, total_sigma=syn_total,
            reg_iterations=syn_iters, similarity_metric=syn_metric,
            optimizer="reg_adam", optimizer_lr=syn_opt_lr,
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=False, inverse_method="anderson",
            in_loop_inv_steps=10, formulation="eulerian", regularizer=syn_reg,
            sobolev_alpha=1.0, antisymmetric=True, verbose=verbose, **kwargs
        )
    elif model_lower == "tvf":
        tvf_flow_sig = kwargs.pop("flow_sigma", config.get("params", {}).get("flow_sigma", 2.5) if config else 2.5)
        tvf_total_sig = kwargs.pop("total_sigma", config.get("params", {}).get("total_sigma", 0.012) if config else 0.012)
        tvf_alpha = kwargs.pop("sobolev_alpha", kwargs.pop("dsti_alpha", config.get("params", {}).get("sobolev_alpha", 0.018) if config else 0.018))
        tvf_reg = kwargs.pop("regularizer", (config.get("params", {}).get("regularizer", "sobolev") if config else "sobolev"))
        tvf_opt = kwargs.pop("optimizer", (config and config.get("params", {}).get("optimizer")) or "reg_adam")
        tvf_opt_lr = kwargs.pop("optimizer_lr", config.get("params", {}).get("optimizer_lr", 1.2) if config else 1.2)
        tvf_max_step = kwargs.pop("max_step_norm", config.get("params", {}).get("max_step_norm", 0.38) if config else 0.38)
        tvf_fast_smooth = kwargs.pop("fast_smooth", True)
        tvf_metric = kwargs.pop("similarity_metric", "cc2")
        res_reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            regularizer=tvf_reg,
            flow_sigma=tvf_flow_sig,
            total_sigma=tvf_total_sig,
            alpha=tvf_alpha,
            dsti_alpha=tvf_alpha,
            sobolev_alpha=tvf_alpha,
            optimizer=tvf_opt,
            optimizer_lr=tvf_opt_lr,
            max_step_norm=tvf_max_step,
            multipoint_loss=kwargs.pop("multipoint_loss", [0.0, 0.5, 1.0]),
            antisymmetric=kwargs.pop("antisymmetric", False),
            reg_iterations=user_reg_iters if user_reg_iters is not None else [100, 100, 20],
            solver=kwargs.pop("solver", "euler"),
            constant_speed=kwargs.pop("constant_speed", True),
            constant_speed_relaxation=kwargs.pop("constant_speed_relaxation", 0.10),
            cfl_momentum=kwargs.pop("cfl_momentum", 0.9),
            fast_smooth=tvf_fast_smooth,
            use_analytical_gradients=kwargs.pop("use_analytical_gradients", False),
            similarity_metric=tvf_metric,
            amp=False,
            verbose=verbose,
            **kwargs
        )
    elif model_lower in ("syngs", "geodesic", "syn_gs"):
        gs_flow_sig = user_flow_sigma if user_flow_sigma is not None else 3.1
        gs_total_sig = user_total_sigma if user_total_sigma is not None else 0.0
        gs_alpha = kwargs.pop("alpha", (config.get("params", {}).get("alpha", 0.42) if config else 0.42))
        gs_opt = kwargs.pop("optimizer", (config and config.get("params", {}).get("optimizer")) or "reg_adam")
        gs_opt_lr = kwargs.pop("optimizer_lr", (config.get("params", {}).get("optimizer_lr", 1.0) if config else 1.0))
        gs_max_step = kwargs.pop("max_step_norm", (config.get("params", {}).get("max_step_norm", 0.20) if config else 0.20))
        gs_reg = kwargs.pop("regularizer", (config.get("params", {}).get("regularizer", "sobolev") if config else "sobolev"))
        gs_trans = kwargs.pop("transport_mode", (config.get("params", {}).get("transport_mode", "transport") if config else "transport"))
        gs_metric = kwargs.pop("similarity_metric", "cc2")
        gs_boot = kwargs.pop("bootstrap_mode", (config.get("params", {}).get("bootstrap_mode", "antithetic") if config else "antithetic"))
        gs_orig_w = kwargs.pop("bootstrap_orig_weight", 0.50)
        gs_jitter = kwargs.pop("bootstrap_jitter_scale", 0.25)
        res_reg = syntx.syngs(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            flow_sigma=gs_flow_sig,
            total_sigma=gs_total_sig,
            alpha=gs_alpha,
            regularizer=gs_reg,
            transport_mode=gs_trans,
            optimizer=gs_opt,
            optimizer_lr=gs_opt_lr,
            max_step_norm=gs_max_step,
            reg_iterations=user_reg_iters if user_reg_iters is not None else [100, 100, 20],
            similarity_metric=gs_metric,
            bootstrap_mode=gs_boot,
            bootstrap_orig_weight=gs_orig_w,
            bootstrap_jitter_scale=gs_jitter,
            n_steps=kwargs.pop("n_steps", 8),
            solver=kwargs.pop("solver", "euler"),
            verbose=verbose,
            **kwargs
        )
    elif model_lower in ("greedy", "syntx_greedy", "greedy_regadam", "regadam_greedy"):
        greedy_iters = user_reg_iters if user_reg_iters is not None else [100, 100, 20]
        greedy_flow_sig = user_flow_sigma if user_flow_sigma is not None else (config.get("params", {}).get("flow_sigma", 1.8) if config else 1.8)
        greedy_total_sig = user_total_sigma if user_total_sigma is not None else (config.get("params", {}).get("total_sigma", 0.28) if config else 0.28)
        greedy_grad_step = user_grad_step if user_grad_step is not None else (config.get("params", {}).get("grad_step", 0.50) if config else 0.50)
        greedy_opt = "regadam" if "regadam" in model_lower else kwargs.pop("optimizer", (config and config.get("params", {}).get("optimizer")) or "adam")
        greedy_regadam_sig = kwargs.pop("regadam_sigma", (config and config.get("params", {}).get("regadam_sigma", 0.8)) if config else 0.8)
        greedy_anderson = kwargs.pop("anderson", False)
        greedy_anderson_steps = kwargs.pop("anderson_steps", 5)
        greedy_return_inv = kwargs.pop("return_inverse", False)
        res_reg = syntx.greedy(
            fixed=fi, moving=mi, initial_transform=aff_0,
            reg_iterations=greedy_iters,
            learning_rate=greedy_grad_step,
            flow_sigma=greedy_flow_sig,
            total_sigma=greedy_total_sig,
            optimizer=greedy_opt,
            regadam_sigma=greedy_regadam_sig,
            anderson=greedy_anderson,
            anderson_steps=greedy_anderson_steps,
            return_inverse=greedy_return_inv,
            similarity_metric=kwargs.pop("similarity_metric", "lncc"),
            device=device,
            verbose=verbose,
            **kwargs
        )
    elif model_lower in ("fireants", "fireants_greedy"):
        import tempfile
        from fireants.io import Image as FAImage, BatchedImages as FABatchedImages
        from fireants.registration.greedy import GreedyRegistration as FireANTsGreedy

        # Convert canonical affine to physical 4x4 matrix
        tx_obj = ants.read_transform(aff_0)
        A = np.array(tx_obj.parameters[:9]).reshape(3, 3)
        t = np.array(tx_obj.parameters[9:12])
        c = np.array(tx_obj.fixed_parameters[:3])
        offset = t + c - A @ c
        M = np.eye(4)
        M[:3, :3] = A
        M[:3, 3] = offset
        M_t = torch.from_numpy(M).float().unsqueeze(0).to(device)

        fa_tmpdir = tempfile.mkdtemp(prefix="fireants_bench_")
        f_path = os.path.join(fa_tmpdir, "f.nii.gz")
        m_path = os.path.join(fa_tmpdir, "m.nii.gz")
        ants.image_write(fi, f_path)
        ants.image_write(mi, m_path)
        b_f = FABatchedImages([FAImage.load_file(f_path, device=device)])
        b_m = FABatchedImages([FAImage.load_file(m_path, device=device)])

        fa_lr = kwargs.pop("optimizer_lr", user_grad_step if user_grad_step is not None else 0.50)
        fa_flow = user_flow_sigma if user_flow_sigma is not None else 1.0
        fa_total = user_total_sigma if user_total_sigma is not None else 0.25
        fa_iters = user_reg_iters if user_reg_iters is not None else [100, 100, 50]

        fa_reg = FireANTsGreedy(
            scales=[4, 2, 1],
            iterations=fa_iters,
            fixed_images=b_f,
            moving_images=b_m,
            loss_type='cc',
            cc_kernel_size=5,
            deformation_type='compositive',
            optimizer='Adam',
            optimizer_lr=fa_lr,
            smooth_grad_sigma=fa_flow,
            smooth_warp_sigma=fa_total,
            init_affine=M_t
        )
        fa_reg.optimize()
        w_file = os.path.join(fa_tmpdir, "fireants_fwd_warp.nii.gz")
        fa_reg.save_as_ants_transforms(w_file, save_inverse=False)
        res_reg = {
            'fwdtransforms': [w_file],
            'invtransforms': [],
            'whichtoinvert_inv': [],
            'warpedmovout': ants.apply_transforms(fi, mi, [w_file]),
        }
    elif model_lower in ("ants", "ants_syn"):
        res_reg = ants.registration(
            fixed=fi, moving=mi, type_of_transform="SyN",
            initial_transform=aff_0,
            syn_metric="CC", syn_sampling=2,
            reg_iterations=(100, 100, 20),
            flow_sigma=3.0, total_sigma=0.0,
            grad_step=0.25,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown registration model: '{model}'. Supported: 'affine', 'affine_fast', 'affine_accurate', 'ants', 'sobolev', 'gaussian', 'syn', 'syn_regadam', 'tvf', 'syngs', 'greedy', 'greedy_regadam', 'fireants'")

    t_reg = (t_aff if model_lower in ("affine", "affine_default") else (res_reg.get("runtime_seconds", time.time() - t0_reg) if "affine" in model_lower else (time.time() - t0_reg + t_aff)))

    # 5. Evaluate Structural and Topological Metrics
    fwd_tx = res_reg["fwdtransforms"]
    inv_tx = res_reg["invtransforms"]
    which_inv = res_reg.get("whichtoinvert_inv", [True, False])

    df_fixed, df_moving, dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, fwd_tx, inv_tx, which_inv
    )
    if (inv_tx is None or len(inv_tx) == 0) or ("greedy" in model_lower or "fireants" in model_lower):
        dice_sym = df_fixed
        df_moving = float("nan")

    fwd_warp_file = next((x for x in fwd_tx if isinstance(x, str) and x.endswith(".nii.gz")), None)
    if fwd_warp_file is not None:
        jac = compute_jacobian_metrics(fi, fwd_warp_file)
    else:
        jac = {"folding_pct": 0.0, "min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0}

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
        "denoise": bool(denoise),
        "status": "SUCCESS",
        "affine_backend": AFFINE_BACKEND_KEY,
        "denoise_device": denoise_device,
        "syntx_affine_dice_sym": float(aff_dice_sym),
        "syntx_affine_time": float(t_aff),
        "syntx_dice_sym": float(dice_sym),
        "syntx_dice_fixed": float(df_fixed),
        "syntx_dice_moving": float(df_moving),
        "syntx_fold": float(jac["folding_pct"]),
        "syntx_min_jac": float(jac["min"]),
        "syntx_inv_mean": float(inv_mean),
        "syntx_inv_p95": float(inv_p95),
        "syntx_time": float(t_reg),
        "dice_sym": float(dice_sym),
        "dice_fixed": float(df_fixed),
        "dice_moving": float(df_moving),
        "folding_pct": float(jac["folding_pct"]),
        "min_jacobian": float(jac["min"]),
        "runtime_seconds": float(t_reg),
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
                fixed_label=fl,
                moving_label=ml,
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
            similarity_metric="cc2",
            verbose=verbose
        )
    else:
        regularizer = "gaussian" if model.lower() in ("gaussian", "syn_gaussian") else "sobolev"
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

