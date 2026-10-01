"""
syntx.benchmark.evaluate — one Mindboggle pair, one method: register and score
==============================================================================

``evaluate_mindboggle_pair`` (alias ``evaluate_pair``) loads a pair, normalizes intensities,
reuses or computes the pair's cached canonical affine, runs one deformable method on top of
it (syntx.syn, syntx.tvf, syntx.syngs, syntx.greedy, ANTs SyN, FireANTs, or affine only) and
returns label Dice in both directions, Jacobian folding statistics, inverse-identity errors,
runtimes and a comparison with a stored ANTs baseline. Results carry a provenance manifest.

Also: ``evaluate_affine_benchmark`` (affine modes over many pairs) and
``run_standard_report_demo`` (one registration on a ``syntx.benchmark_data`` dataset plus an
HTML report). All relative paths (affine cache, baselines, reports) are relative to the
current working directory.
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
from syntx.provenance import with_provenance
from syntx.benchmark.config import get_model_config, compute_config_hash, syn_config_to_syn_kwargs
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics, flow_jacobian_metrics
from syntx.core.utils import normalize_image


def clean_device_cache():
    """Run ``gc.collect()`` and empty the CUDA cache if CUDA is available, else the MPS cache
    (errors from the MPS call are ignored)."""
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
    """``syntx.core.utils.normalize_image(img, method='auto')``: clip to automatically chosen
    foreground percentiles and rescale to [0, 1] (an image already in [0, 1] is only clipped)."""
    return normalize_image(img, method='auto')


# Identifies which affine engine produced a cached canonical affine.  Bump when the affine
# backend or its defaults change so stale caches are recomputed instead of silently reused.
AFFINE_BACKEND_KEY = "pt7"


def _inverse_error_stats(err: Dict[str, Any], fixed) -> Dict[str, float]:
    """Summarise an inverse-identity error record (mm).

    ``registration()`` returns ``{'max_error', 'mean_error', 'error_map'}`` (error_map in
    tensor order (z, y, x)); older records used ``{'mean', 'p95'}``. With an error map, it is
    squeezed, transposed to ANTs order (x, y, z) if its shape is the reversed fixed shape, and
    summarised over all voxels; interior values use ``fixed``'s ``get_mask`` eroded by 5 voxels
    (``iMath 'ME' 5``). Without a map, only mean / p95 / max are read from the record.

    Returns
    -------
    dict
        'mean', 'p95', 'max', 'interior_mean', 'interior_max' (float; NaN when unavailable,
        e.g. interior values when the map's shape does not match ``fixed``).
    """
    nan = float("nan")
    out = {k: nan for k in ("mean", "p95", "max", "interior_mean", "interior_max")}
    emap = err.get("error_map") if isinstance(err, dict) else None
    if emap is None:
        out["mean"] = float(err.get("mean", err.get("mean_error", nan)))
        out["p95"] = float(err.get("p95", nan))
        out["max"] = float(err.get("max", err.get("max_error", nan)))
        return out
    e = emap.detach().cpu().numpy() if hasattr(emap, "detach") else np.asarray(emap)
    e = np.squeeze(e)
    if e.shape != tuple(fixed.shape) and e.shape == tuple(fixed.shape)[::-1]:
        e = np.transpose(e, tuple(range(e.ndim))[::-1])
    out.update(mean=float(e.mean()), p95=float(np.percentile(e, 95)), max=float(e.max()))
    if e.shape == tuple(fixed.shape):
        interior = ants.iMath(ants.get_mask(fixed), "ME", 5).numpy() > 0
        if interior.any():
            out.update(interior_mean=float(e[interior].mean()),
                       interior_max=float(e[interior].max()))
    return out


def _evaluate_mindboggle_pair_impl(
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
    """Register one Mindboggle pair with one method and return its benchmark record.

    Steps:

    1. Seed torch and numpy with ``seed + pair_idx``; load the pair
       (``load_mindboggle_pair``, fixed = subject 1, moving = subject 2).
    2. Optionally denoise both brains (``antstorch.denoise_image``, Rician, shrink_factor 2,
       p 1, r 1, on ``device``); if that fails, continue with the undenoised images (warning
       only when ``verbose``). Then ``normalize_intensity`` both.
    3. Canonical affine: reuse ``results/canonical_affines/pair_<idx>[_denoised]_pt7_affine.mat``
       if it and its ``_affine_info.json`` exist and the info's 'affine_backend' equals
       ``AFFINE_BACKEND_KEY``; otherwise run ``syntx.robust_affine(fi, mi, mode='auto')``,
       copy its transform there, and store the affine-only symmetric Dice and runtime in the
       info file. The cache name does not depend on ``use_n4``, ``pairs_csv`` or ``data_dir``.
    4. Run the deformable method with ``initial_transform`` = the affine file (see ``model``).
    5. Score: ``compute_bidirectional_dice`` on the DKT31 labels; Jacobian statistics from
       ``flow_jacobian_metrics`` (``syntx.liouville_determinant`` of the method's own map,
       available for syntx syn / tvf / syngs / greedy results), else from the finite-difference
       determinant of the first ``.nii.gz`` forward transform (``compute_jacobian_metrics``);
       inverse-identity error from ``res['inverse_identity_errors']`` (its 'phi_1' entry if
       present) via ``_inverse_error_stats``.
    6. Read the ANTs baseline ``<ants_baseline_dir>/pair_<idx>_ants_syn.json`` if present.
    7. Optionally write an HTML report (``syntx.viz.create_registration_report``).

    Parameters
    ----------
    pair_idx : int, default 0
        Row of ``pairs_csv`` (0..89 for the standard CSV).
    model : str, default 'sobolev'
        Case-insensitive. Supported values:

        - 'sobolev' / 'syn_sobolev' / 'syn': ``syntx.syn`` (pytorch) with its own defaults.
        - 'gaussian' / 'syn_gaussian' / 'syn_mi': ``syntx.syn`` with regularizer 'gaussian'
          and explicit settings: parameters from the 'gaussian_config' block (grad_step 0.25,
          fluid_sigma 3.0, elastic_sigma 0.0, reg_iterations [100, 100, 20], metric 'cc2';
          'syn_mi' uses 'mattes_mi'), syn_sampling 2, anderson inverse, antisymmetric.
        - 'syn_regadam' / 'syn_dsti1' / 'regadam_syn': ``syntx.syn`` with optimizer
          'reg_adam' (optimizer_lr 1.0), regularizer 'dsti1' (overridable via
          ``regularizer`` except for 'syn_dsti1'), grad_step 0.5, flow_sigma 3.0,
          reg_iterations [100, 50, 10].
        - 'tvf': ``syntx.tvf`` with its own defaults.
        - 'syngs' / 'geodesic' / 'syn_gs': ``syntx.syngs`` with its own defaults.
        - 'greedy' / 'syntx_greedy' / 'greedy_regadam' / 'regadam_greedy': ``syntx.greedy``
          with its own defaults (the 'regadam' names force optimizer 'regadam'). Greedy
          returns no inverse.
        - 'fireants' / 'fireants_greedy': FireANTs ``GreedyRegistration`` (scales [4, 2, 1],
          iterations [100, 100, 50], CC kernel 5, Adam lr 0.5, smooth_grad_sigma 1.0,
          smooth_warp_sigma 0.25) initialised with the affine; forward warp only. 3-D only.
        - 'ants' / 'ants_syn': ``ants.registration(type_of_transform='SyNOnly')`` with CC,
          syn_sampling 2, reg_iterations (100, 100, 20), flow_sigma 3, total_sigma 0,
          grad_step 0.25.
        - 'affine' / 'affine_default': the canonical affine only.
        - 'affine_fast' / 'affine_accurate': a fresh ``robust_affine(mode='auto',
          preset='fast' / 'accurate')``.

        Any other value raises ValueError.
    device : str, optional
        Torch device; default 'cuda' if available, else 'mps', else 'cpu'. Not used by the
        ANTs and affine arms (``robust_affine`` picks its own device).
    pairs_csv : str, default 'examples/pairs.csv'
        Pairs CSV.
    data_dir : str, optional
        Mindboggle data directory (``resolve_data_dir``).
    ants_baseline_dir : str, default 'results'
        Directory of the ANTs baseline JSON files.
    generate_report : bool, default False
        Write ``<report_out_dir>/report_pair_<idx>_<model>.html``. Report errors are
        swallowed (printed only when ``verbose``).
    report_out_dir : str, optional
        Report directory; default 'docs/reports'.
    verbose : bool, default False
        Passed to the registration call; also prints warnings.
    seed : int, default 42
        torch / numpy seed offset (the seed used is ``seed + pair_idx``).
    dataset_key : str, optional
        Accepted and ignored (the Mindboggle pair is always used).
    config : dict, optional
        Parameter configuration (``DEFAULT_BENCHMARK_CONFIG`` layout or a grid entry with
        'params'); resolved with ``get_model_config``. With ``config=None`` the syn / tvf /
        syngs / greedy arms run with the function's own defaults; with a config, the
        block's values are passed to them (for 'sobolev' via ``syn_config_to_syn_kwargs``,
        which raises KeyError on unmapped keys). The resolved block and its hash are stored
        in the record either way.
    use_n4 : bool, default True
        Load N4-corrected brains (cached; see ``load_mindboggle_pair``).
    denoise : bool, default False
        Denoise before normalisation (see step 2); the affine cache gets a '_denoised' suffix.
    **kwargs
        Explicit parameter overrides. ``reg_iterations``, ``grad_step`` (alias
        ``learning_rate``), ``flow_sigma``, ``total_sigma``, ``fast_smooth`` and
        ``similarity_metric`` take precedence over ``config`` for every arm that uses them;
        all other keywords are passed through to the syntx registration call. The 'affine*'
        and 'ants' arms ignore all keywords; 'fireants' uses ``reg_iterations``,
        ``grad_step`` / ``optimizer_lr`` (lr), ``flow_sigma`` and ``total_sigma`` and ignores
        the rest; the regadam arm always runs with ``fast_smooth=False``.

    Returns
    -------
    dict
        - 'pair_idx', 'model_type' (lower-cased ``model``), 'cohort_type', 'fixed_id',
          'moving_id', 'use_n4', 'denoise', 'status' ('SUCCESS'), 'affine_backend',
          'denoise_device' (None if not denoised).
        - 'syntx_affine_dice_sym', 'syntx_affine_time': the canonical affine's symmetric
          Dice and runtime (s), from the cache info file when the affine was reused.
        - 'syntx_dice_sym', 'syntx_dice_fixed', 'syntx_dice_moving' (also as 'dice_sym',
          'dice_fixed', 'dice_moving'). For methods without an inverse (greedy, FireANTs, or an
          empty inverse list) 'dice_sym' is the fixed-space Dice and 'dice_moving' is NaN.
        - 'syntx_fold', 'syntx_min_jac', 'syntx_max_jac' (also 'folding_pct', 'min_jacobian'),
          'syntx_jacobian_measure' (which determinant was used), 'syntx_fd_fold',
          'syntx_fd_min_jac' (finite-difference determinant of the exported warp; 0 % / 1.0
          when there is no warp file).
        - 'syntx_inv_mean', 'syntx_inv_p95', 'syntx_inv_max', 'syntx_inv_interior_mean',
          'syntx_inv_interior_max' (mm; NaN when the method reports no inverse error; 0 for
          mean / p95 of the 'affine*' arms).
        - 'syntx_time' (also 'runtime_seconds'): deformable runtime plus the affine time
          ``syntx_affine_time`` (which may come from the cache); for the 'affine*' arms the
          affine time alone.
        - 'diff_vs_ants': (dice_sym - ANTs dice_sym) x 100; 'win': dice_sym >= ANTs dice_sym
          (False without a baseline). 'ants_baseline': the baseline's 'dice_sym',
          'dice_fixed', 'dice_moving', 'runtime_seconds' (NaN if missing) and 'folding_pct',
          'min_jacobian' (0.0 if missing).
        - 'transforms': 'fwdtransforms', 'invtransforms', 'whichtoinvert_inv' (as strings;
          deformable fields are typically temporary files).
        - 'config', 'config_hash'; 'report_html' when a report was written.
        - 'provenance': added by the ``with_provenance`` wrapper of ``evaluate_mindboggle_pair``
          (code state, environment, and each registration call with its resolved
          parameters).

    Raises
    ------
    ValueError
        Unknown ``model``.
    FileNotFoundError, IndexError
        From ``load_mindboggle_pair``.
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

    # Resolve model-specific configuration block from centralized config
    model_cfg = get_model_config(model_lower, config)
    config_hash = compute_config_hash(model_cfg)

    # Parameters the caller passed explicitly (kept apart from config-derived values so the
    # standard SyN run can use syntx.syn()'s own -- canonical -- defaults).
    explicit_syn = {k: kwargs[k] for k in ("reg_iterations", "grad_step", "learning_rate", "flow_sigma",
                                           "total_sigma", "fast_smooth", "similarity_metric")
                    if k in kwargs}

    # Allow parameter overrides from kwargs or resolved config block
    user_reg_iters = kwargs.pop("reg_iterations", None) or model_cfg.get("reg_iterations")
    user_grad_step = kwargs.pop("learning_rate", kwargs.pop("grad_step", None)) or model_cfg.get("grad_step")
    user_flow_sigma = kwargs.pop("flow_sigma", None) if "flow_sigma" in kwargs else model_cfg.get("flow_sigma", model_cfg.get("fluid_sigma", model_cfg.get("tvf_flow_sigma")))
    user_total_sigma = kwargs.pop("total_sigma", None) if "total_sigma" in kwargs else model_cfg.get("total_sigma", model_cfg.get("elastic_sigma", model_cfg.get("tvf_total_sigma")))
    fast_smooth = kwargs.pop("fast_smooth", None) if "fast_smooth" in kwargs else model_cfg.get("fast_smooth", model_cfg.get("syn_fast_smooth", model_cfg.get("tvf_fast_smooth", False)))

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
    elif model_lower in ("sobolev", "syn_sobolev", "syn"):
        # Standard run: syntx.syn()'s defaults ARE the canonical benchmark parameters
        # (docs/provenance/best_parameters.json; tests/test_canonical_parameters.py).
        # A caller-supplied config or explicit keyword overrides them.
        syn_kwargs = syn_config_to_syn_kwargs(model_cfg) if config is not None else {}
        for k, v in explicit_syn.items():
            syn_kwargs[{"learning_rate": "grad_step", "similarity_metric": "syn_metric"}.get(k, k)] = v
        kwargs.pop("similarity_metric", None)
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device, verbose=verbose, **syn_kwargs, **kwargs
        )
    elif model_lower in ("gaussian", "syn_gaussian", "syn_mi"):
        syn_iters = user_reg_iters if user_reg_iters is not None else [100, 100, 20]
        syn_step = user_grad_step if user_grad_step is not None else 0.25
        syn_flow = user_flow_sigma if user_flow_sigma is not None else 3.0
        syn_total = user_total_sigma if user_total_sigma is not None else 0.0
        default_metric = "mattes_mi" if model_lower == "syn_mi" else "cc2"
        syn_metric = kwargs.pop("similarity_metric", model_cfg.get("syn_metric", default_metric))
        syn_kernel = kwargs.pop("kernel_type", model_cfg.get("kernel_type", "gaussian"))
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device,
            grad_step=syn_step, flow_sigma=syn_flow, total_sigma=syn_total,
            reg_iterations=syn_iters, similarity_metric=syn_metric,
            use_ants_pseudo_gradient=False, use_analytical_gradients=False,
            syn_sampling=2, fast_smooth=fast_smooth, inverse_method="anderson",
            formulation="eulerian", regularizer="gaussian", kernel_type=syn_kernel,
            antisymmetric=True, verbose=verbose, **kwargs
        )
    elif model_lower in ("syn_regadam", "syn_dsti1", "regadam_syn"):
        syn_iters = user_reg_iters if user_reg_iters is not None else [100, 50, 10]
        syn_step = user_grad_step if user_grad_step is not None else 0.50
        syn_flow = user_flow_sigma if user_flow_sigma is not None else 3.0
        syn_total = user_total_sigma if user_total_sigma is not None else 0.0
        syn_metric = kwargs.pop("similarity_metric", model_cfg.get("syn_metric", "cc2"))
        syn_reg = "dsti1" if "dsti" in model_lower else kwargs.pop("regularizer", model_cfg.get("regularizer", "dsti1"))
        syn_opt_lr = kwargs.pop("optimizer_lr", model_cfg.get("optimizer_lr", 1.0))
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
        # Standard run: syntx.tvf's own defaults (tests/test_canonical_parameters.py).
        # A caller-supplied config or explicit keyword overrides them.
        tvf_kwargs = {}
        if config is not None:
            _keys = ("regularizer", "alpha", "total_alpha", "flow_sigma", "total_sigma", "optimizer",
                     "optimizer_lr", "max_step_norm", "grad_step", "cfl_momentum", "cfl_max",
                     "n_time_steps", "multipoint_loss", "fast_smooth", "syn_metric", "syn_sampling",
                     "reg_iterations", "constant_speed", "constant_speed_relaxation")
            tvf_kwargs = {k: model_cfg[k] for k in _keys if k in model_cfg}
        for k, v in explicit_syn.items():
            tvf_kwargs[{"similarity_metric": "syn_metric", "learning_rate": "grad_step"}.get(k, k)] = v
        kwargs.pop("similarity_metric", None)
        res_reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device, verbose=verbose, **tvf_kwargs, **kwargs
        )
    elif model_lower in ("syngs", "geodesic", "syn_gs"):
        # Standard run: syntx.syngs's own defaults (tests/test_canonical_parameters.py).
        # A caller-supplied config or explicit keyword overrides them.
        gs_kwargs = {}
        if config is not None:
            _map = {"grad_step": "grad_step", "flow_sigma": "flow_sigma", "total_sigma": "total_sigma",
                    "alpha": "alpha", "regularizer": "regularizer", "optimizer": "optimizer",
                    "optimizer_lr": "optimizer_lr", "max_step_norm": "max_step_norm",
                    "syn_metric": "syn_metric", "similarity_metric": "syn_metric", "n_steps": "n_steps",
                    "bootstrap_mode": "bootstrap_mode", "reg_iterations": "reg_iterations",
                    "transport_mode": "transport_mode"}
            gs_kwargs = {_map[k]: v for k, v in model_cfg.items() if k in _map}
        for k, v in explicit_syn.items():
            gs_kwargs[{"similarity_metric": "syn_metric", "learning_rate": "grad_step"}.get(k, k)] = v
        kwargs.pop("similarity_metric", None)
        res_reg = syntx.syngs(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend="pytorch", device=device, verbose=verbose, **gs_kwargs, **kwargs
        )
    elif model_lower in ("greedy", "syntx_greedy", "greedy_regadam", "regadam_greedy"):
        # Standard run: syntx.greedy's own defaults (tests/test_canonical_parameters.py).
        # A caller-supplied config or explicit keyword overrides them. greedy produces no
        # inverse, so the inverse-consistency metrics of this arm are NaN.
        greedy_kwargs = {}
        if config is not None:
            _map = {"grad_step": "learning_rate", "learning_rate": "learning_rate",
                    "flow_sigma": "flow_sigma", "total_sigma": "total_sigma",
                    "optimizer": "optimizer", "regadam_sigma": "regadam_sigma",
                    "similarity_metric": "similarity_metric", "reg_iterations": "reg_iterations",
                    "lncc_radius": "lncc_radius"}
            greedy_kwargs = {_map[k]: v for k, v in model_cfg.items() if k in _map}
        for k, v in explicit_syn.items():
            greedy_kwargs[{"grad_step": "learning_rate"}.get(k, k)] = v
        greedy_kwargs.pop("fast_smooth", None)  # not a greedy parameter
        if "regadam" in model_lower:
            greedy_kwargs["optimizer"] = "regadam"
        res_reg = syntx.greedy(
            fixed=fi, moving=mi, initial_transform=aff_0, device=device, verbose=verbose,
            **greedy_kwargs, **kwargs
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
            fixed=fi, moving=mi, type_of_transform="SyNOnly",
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
        jac_fd = compute_jacobian_metrics(fi, fwd_warp_file)
    else:
        jac_fd = {"folding_pct": 0.0, "min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0}
    # Folding measure: the flow's exact (Liouville) determinant where the method has one (tvf);
    # the finite-difference determinant of the exported warp otherwise (reported for context).
    jac = flow_jacobian_metrics(fi, res_reg) or dict(jac_fd, measure="finite_difference")

    inv_errs = res_reg.get("inverse_identity_errors", {})
    inv_stats = _inverse_error_stats(inv_errs.get("phi_1", inv_errs), fi)
    inv_mean, inv_p95 = inv_stats["mean"], inv_stats["p95"]

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
        "syntx_max_jac": float(jac.get("max", float("nan"))),
        "syntx_jacobian_measure": jac["measure"],
        "syntx_fd_fold": float(jac_fd["folding_pct"]),
        "syntx_fd_min_jac": float(jac_fd["min"]),
        "syntx_inv_mean": float(inv_mean),
        "syntx_inv_p95": float(inv_p95),
        "syntx_inv_max": float(inv_stats["max"]),
        "syntx_inv_interior_mean": float(inv_stats["interior_mean"]),
        "syntx_inv_interior_max": float(inv_stats["interior_max"]),
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
        },
        "config": model_cfg,
        "config_hash": config_hash,
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
# Every result carries a provenance manifest (syntx.provenance): git commit + full
# uncommitted diff, the invoking script's text, environment, and each registration call
# with the parameters the method actually resolved. See docs/BENCHMARKING_GUIDE.md.
evaluate_mindboggle_pair = with_provenance("syntx.benchmark.evaluate_mindboggle_pair")(
    _evaluate_mindboggle_pair_impl)
evaluate_mindboggle_pair.__name__ = "evaluate_mindboggle_pair"


evaluate_pair = evaluate_mindboggle_pair


def run_standard_report_demo(
    dataset_key: str = "mbhard",
    output_html: str = "docs/reports/mbhard_standard_report.html",
    model: str = "sobolev",
    device: Optional[str] = None,
    reg_iterations: list = None,
    verbose: bool = False
) -> str:
    """Register one ``syntx.benchmark_data`` pair and write an HTML registration report.

    The images are normalised (``normalize_intensity``), aligned with
    ``robust_affine(mode='auto')``, then registered with ``model`` using the method's own
    defaults. If the dataset has labels, the symmetric Dice is computed (errors ignored) and
    shown in the report, which is built by ``syntx.viz.create_registration_report``.

    Parameters
    ----------
    dataset_key : str, default 'mbhard'
        ``syntx.benchmark_data`` key ('mbhard', 'r16_r64', 'c', 'ellipse', ...).
    output_html : str, default 'docs/reports/mbhard_standard_report.html'
        Report path.
    model : str, default 'sobolev'
        'tvf' -> ``syntx.tvf``; 'gaussian' / 'syn_gaussian' -> ``syntx.syn`` with
        ``regularizer='gaussian'``; any other value -> ``syntx.syn`` with its defaults.
    device : str, optional
        Torch device for the deformable step; default 'cuda' > 'mps' > 'cpu'.
    reg_iterations : list of int, optional
        Replaces the method's default schedule.
    verbose : bool, default False
        Print progress (and pass ``verbose`` to the registration).

    Returns
    -------
    str
        The report path returned by ``create_registration_report`` ('html_path'), else the
        absolute ``output_html``.
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

    # Each method runs with its own defaults (syntx.syn's are the canonical benchmark
    # parameters); only an explicit reg_iterations overrides them.
    overrides = {} if reg_iterations is None else {"reg_iterations": reg_iterations}

    if verbose:
        print(f"[run_standard_report_demo] Step 2/2: Running deformable {model.upper()} ({reg_iterations or 'default schedule'})...", flush=True)

    if model.lower() == "tvf":
        res_reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            device=device, verbose=verbose, **overrides
        )
    else:
        if model.lower() in ("gaussian", "syn_gaussian"):
            overrides["regularizer"] = "gaussian"
        res_reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            backend="pytorch", device=device, verbose=verbose, **overrides
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
    """Compare ``syntx.robust_affine`` modes by label Dice over Mindboggle pairs.

    For each pair and mode: ``robust_affine(fi, mi, mode=m)`` on the loaded (not
    intensity-normalised) brains, then ``compute_bidirectional_dice`` with its transforms.

    Parameters
    ----------
    pairs : int, list of int, or str, default 'inter16'
        A pair index, a list of indices, or a keyword: 'mbhard' -> [44], 'inter16' ->
        40..55 (the first 16 inter-cohort pairs), 'intra16' -> 0..15, 'all' -> 0..89. Another
        string is parsed as an integer index; if that fails, pair 0 is used silently.
    modes : list of str, default ['ants_fast', 'pytorch', 'auto', 'com_only']
        ``robust_affine`` modes.
    pairs_csv : str, default 'examples/pairs.csv'
        Pairs CSV.
    data_dir : str, optional
        Mindboggle data directory.
    verbose : bool, default True
        Print per-pair / per-mode results.
    generate_report : bool, default False
        Call ``syntx.viz.reports.create_affine_benchmark_report`` (also done when
        ``output_html`` is given). Note that the DataFrame is passed as that function's
        ``summary_source``, which accepts only a JSON path or a dict.
    output_html : str, optional
        Report path; default 'docs/reports/affine_benchmark_report.html'.
    use_n4 : bool, default True
        Load N4-corrected brains.

    Returns
    -------
    pandas.DataFrame
        One row per (pair, mode): 'pair_idx', 'pair_type', 'cohorts' ('<cohort1>-><cohort2>'),
        'mode', 'dice_fixed', 'dice_moving', 'dice_sym', 'runtime_seconds'. A mode that raises
        is recorded with all four numbers 0.0; a pair that fails to load is skipped.
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

