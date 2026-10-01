"""
syntx.benchmark.high_level — compare ANTs and syntx backends on one pair
========================================================================

``high_level_benchmark_run`` registers one pair (a ``syntx.benchmark_data`` key, a dataset dict,
or user images) with ANTsPy SyN and with ``syntx.syn`` / ``syntx.tvf`` on PyTorch (MPS or CPU)
and JAX (CPU), all starting from one shared CPU ``robust_affine``, and returns a DataFrame of
Dice, Jacobian, energy and runtime per method.
"""

import time
import os
import gc
import json
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import ants
import syntx


def _evaluate_2d_r16_r64(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fixed_label: ants.ANTsImage,
    moving_label: ants.ANTsImage,
    fwdtransforms: List[str],
    invtransforms: List[str],
    runtime: float,
    whichtoinvert_inv: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """2-D scoring: Dice of label 2 and of label 3 (for the r16/r64 3-class Otsu labels,
    roughly grey and white matter), each in fixed and moving space. Label maps without
    labels 2 and 3 (e.g. the binary 'c' / 'ellipse' masks) are scored on their positive
    labels instead (keys 'label<k>_*' for each).

    Each label is binarised, warped with nearest-neighbour interpolation (moving -> fixed by
    ``fwdtransforms``, fixed -> moving by ``invtransforms`` with ``whichtoinvert_inv``;
    default: invert the ``.mat`` affines, the ANTs convention), and scored with
    ``ants.label_overlap_measures`` ('MeanOverlap').

    Returns
    -------
    dict
        'label<k>_fix_dice', 'label<k>_mov_dice', 'label<k>_sym_dice' (mean of the two) per
        scored label, 'mean_sym_dice' (mean of the symmetric values), 'dice_fixed' /
        'dice_moving' (means of the fixed- / moving-space values), 'dice_sym'
        (= 'mean_sym_dice'), 'runtime_seconds' (= ``runtime``).
    """
    which = _whichtoinvert(invtransforms, whichtoinvert_inv)
    present = set(np.unique(fixed_label.numpy()).astype(int)) | set(np.unique(moving_label.numpy()).astype(int))
    score_labels = [2, 3] if {2, 3} <= present else sorted(l for l in present if l > 0)
    if not score_labels:
        raise ValueError("_evaluate_2d_r16_r64: the label maps have no positive labels")

    def get_dice(target, warped):
        ov = ants.label_overlap_measures(target.clone('unsigned int'), warped.clone('unsigned int'))
        ov['L_num'] = pd.to_numeric(ov['Label'], errors='coerce')
        filtered = ov[ov['L_num'] > 0]
        # Guaranteed Sørensen-Dice Invariant: MeanOverlap is 2|A ∩ B| / (|A| + |B|)
        col = 'MeanOverlap' if 'MeanOverlap' in ov.columns else ('Dice' if 'Dice' in ov.columns else 'MeanOverlap')
        if len(filtered) > 0:
            return float(pd.to_numeric(filtered[col], errors='coerce').dropna().mean())
        else:
            return float(pd.to_numeric(ov[col], errors='coerce').iloc[0])

    out, fixes, movs = {}, [], []
    for k in score_labels:
        fl_k = fixed_label.threshold_image(k, k)
        ml_k = moving_label.threshold_image(k, k)
        # moving labels to fixed space, fixed labels to moving space
        ml_k_w = ants.apply_transforms(fixed=fixed, moving=ml_k, transformlist=fwdtransforms, interpolator='nearestNeighbor')
        fl_k_w = ants.apply_transforms(fixed=moving, moving=fl_k, transformlist=invtransforms,
                                       whichtoinvert=which, interpolator='nearestNeighbor')
        d_fix, d_mov = get_dice(fl_k, ml_k_w), get_dice(ml_k, fl_k_w)
        out[f'label{k}_fix_dice'], out[f'label{k}_mov_dice'] = d_fix, d_mov
        out[f'label{k}_sym_dice'] = 0.5 * (d_fix + d_mov)
        fixes.append(d_fix)
        movs.append(d_mov)

    mean_sym = float(np.mean([out[f'label{k}_sym_dice'] for k in score_labels]))
    out.update({
        'mean_sym_dice': mean_sym,
        'dice_fixed': float(np.mean(fixes)),
        'dice_moving': float(np.mean(movs)),
        'dice_sym': mean_sym,
        'runtime_seconds': runtime,
    })
    return out


def _whichtoinvert(invtransforms, whichtoinvert_inv=None):
    """``whichtoinvert`` for ``invtransforms``: the given list, else the ANTs convention
    (invert the ``.mat`` affines, not the inverse warps)."""
    if whichtoinvert_inv is not None:
        return list(whichtoinvert_inv)
    return [isinstance(t, str) and t.endswith('.mat') for t in invtransforms]


def _evaluate_3d_mbhard(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fixed_label: ants.ANTsImage,
    moving_label: ants.ANTsImage,
    fwdtransforms: List[str],
    invtransforms: List[str],
    runtime: float,
    whichtoinvert_inv: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """3-D scoring: mean DKT31 label Dice in fixed and moving space, plus image similarity.

    Labels are warped with nearest-neighbour interpolation (moving -> fixed by
    ``fwdtransforms``, fixed -> moving by ``invtransforms`` with ``whichtoinvert_inv``;
    default: invert the ``.mat`` affines); Dice is 'MeanOverlap' of ``ants.label_overlap_measures`` averaged over
    labels > 0. ``syntx.image_compare`` 'mattes_mi' and 'lncc' compare the fixed image with the
    moving image warped by ``fwdtransforms``.

    Returns
    -------
    dict
        'fixed_dkt31_dice', 'moving_dkt31_dice', 'sym_mean_dkt31_dice' / 'mean_sym_dice' (mean
        of the two), aliases 'dice_fixed', 'dice_moving', 'dice_sym', 'mattes_mi', 'lncc',
        'runtime_seconds' (= ``runtime``).
    """
    ml_w = ants.apply_transforms(fixed=fixed, moving=moving_label, transformlist=fwdtransforms, interpolator='nearestNeighbor')
    fl_w = ants.apply_transforms(fixed=moving, moving=fixed_label, transformlist=invtransforms,
                                 whichtoinvert=_whichtoinvert(invtransforms, whichtoinvert_inv),
                                 interpolator='nearestNeighbor')

    ov_fix = ants.label_overlap_measures(fixed_label.clone('unsigned int'), ml_w.clone('unsigned int'))
    ov_fix['L_num'] = pd.to_numeric(ov_fix['Label'], errors='coerce')
    dkt_fix_df = ov_fix[ov_fix['L_num'] > 0]
    # Guaranteed Sørensen-Dice Invariant: MeanOverlap is 2|A ∩ B| / (|A| + |B|)
    col_fix = 'MeanOverlap' if 'MeanOverlap' in dkt_fix_df.columns else ('Dice' if 'Dice' in dkt_fix_df.columns else 'MeanOverlap')
    fix_d = float(pd.to_numeric(dkt_fix_df[col_fix], errors='coerce').dropna().mean())

    ov_mov = ants.label_overlap_measures(moving_label.clone('unsigned int'), fl_w.clone('unsigned int'))
    ov_mov['L_num'] = pd.to_numeric(ov_mov['Label'], errors='coerce')
    dkt_mov_df = ov_mov[ov_mov['L_num'] > 0]
    col_mov = 'MeanOverlap' if 'MeanOverlap' in dkt_mov_df.columns else ('Dice' if 'Dice' in dkt_mov_df.columns else 'MeanOverlap')
    mov_d = float(pd.to_numeric(dkt_mov_df[col_mov], errors='coerce').dropna().mean())

    sym_d = 0.5 * (fix_d + mov_d)

    # Warp moving image to fixed space to compute image metrics
    mi_w = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwdtransforms)
    import syntx
    mattes_mi = syntx.image_compare(fixed, mi_w, 'mattes_mi')
    lncc = syntx.image_compare(fixed, mi_w, 'lncc')

    return {
        'fixed_dkt31_dice': fix_d,
        'moving_dkt31_dice': mov_d,
        'sym_mean_dkt31_dice': sym_d,
        'mean_sym_dice': sym_d,
        # Unified metric aliases for cross-module consistency
        'dice_fixed': fix_d,
        'dice_moving': mov_d,
        'dice_sym': sym_d,
        'mattes_mi': mattes_mi,
        'lncc': lncc,
        'runtime_seconds': runtime
    }


def high_level_benchmark_run(
    benchmark_name: Union[str, Dict[str, Any]] = 'r16_r64',
    methods: Optional[List[str]] = None,
    model: str = 'syn',
    fixed: Optional[ants.ANTsImage] = None,
    moving: Optional[ants.ANTsImage] = None,
    fixed_label: Optional[ants.ANTsImage] = None,
    moving_label: Optional[ants.ANTsImage] = None,
    grad_step: Optional[float] = None,
    fluid_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    lncc_radius: Optional[int] = None,
    inverse_steps: Optional[int] = None,
    syn_regularizer: Optional[str] = None,
    syn_fast_smooth: Optional[bool] = None,
    syn_use_analytical_gradients: Optional[bool] = None,
    syn_inverse_method: Optional[str] = None,
    tvf_overrides: Optional[Dict[str, Any]] = None,
    reg_iterations: Optional[List[int]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Register one pair with several ANTs / syntx backends and tabulate the metrics.

    All methods start from the same affine: ``syntx.robust_affine(mode='auto', device='cpu',
    multi_start=True)`` computed once after seeding torch / numpy / random with 42. Each method
    is then timed (the affine time is not included) and scored with ``_evaluate_2d_r16_r64``
    (2-D images) or ``_evaluate_3d_mbhard`` (3-D images), plus Jacobian / energy statistics of
    the first ``.nii.gz`` forward transform (``syntx.benchmark.metrics.
    warp_jacobian_and_energies``: finite differences inside ``ants.get_mask(fixed)``). If
    those statistics fail, they are NaN for that method (with a warning).

    Parameters
    ----------
    benchmark_name : str or dict, default 'r16_r64'
        - 'r16_r64' / '2d' / 'r16' / 'r64': ``syntx.benchmark_data('r16_r64')`` (2-D slices,
          3-class Otsu labels).
        - 'mbhard' / '3d' / 'mindboggle' / 'mindboggle_hard' / 'mb_hard' / 'hard_pair':
          ``syntx.benchmark_data('mbhard')`` (Mindboggle NKI-TRT-20-2 -> MMRR-21-2, pairs.csv
          row 44, DKT31 labels).
        - any other string: passed to ``syntx.benchmark_data``. The 2-D scorer thresholds
          labels 2 and 3, so binary-mask datasets ('c', 'ellipse') give empty labels.
        - dict with 'fixed', 'moving', 'fixed_label', 'moving_label' (optional 'key').
        Ignored when all four image arguments are given.
    methods : list of str, optional
        Any of 'antspy_cpp' (aliases 'antspy', 'cpp'), 'pytorch_mps' ('mps', 'syn_mps'),
        'pytorch_cpu' ('pytorch', 'py_cpu', 'syn_cpu'), 'jax_cpu' ('jax', 'jax_backend',
        'syn_jax'), 'tvf_pytorch_mps' ('tvf_mps'), 'tvf_pytorch_cpu' ('tvf_pytorch',
        'tvf_py_cpu'), 'tvf_jax_cpu' ('tvf_jax', 'tvf_jax_backend'). Unknown names raise
        ValueError. Repeating a method runs it once. ``syntx.tvf`` with backend 'jax' accepts
        only regularizer 'gaussian' with optimizer 'cfl', so 'tvf_jax_cpu' needs
        ``tvf_overrides`` such as ``{'regularizer': 'gaussian'}``.
    model : str, default 'syn'
        Used only when ``methods`` is None: 'syn' -> the four SyN methods, 'tvf' -> the three
        TVF methods, 'all' / 'both' -> all seven; any other value -> the SyN methods.
    fixed, moving, fixed_label, moving_label : ANTsImage, optional
        User images; used only when all four are given.
    grad_step, fluid_sigma, elastic_sigma, lncc_radius, inverse_steps : optional
        ``syntx.syn`` overrides, passed as ``grad_step``, ``flow_sigma``, ``total_sigma``,
        ``syn_sampling`` and ``inverse_steps``. ``grad_step`` is also passed to the ANTs arm.
    syn_regularizer, syn_fast_smooth, syn_use_analytical_gradients, syn_inverse_method : optional
        ``syntx.syn`` overrides ``regularizer``, ``fast_smooth``, ``use_analytical_gradients``,
        ``inverse_method``. For all SyN overrides, None (default) keeps ``syntx.syn``'s own
        default. None of them reach the TVF methods.
    tvf_overrides : dict, optional
        Extra keywords for ``syntx.tvf`` (e.g. ``{'alpha': 1.5}``); None runs its defaults.
    reg_iterations : list of int, optional
        Schedule for every method (ANTs included). For 3-D images, None becomes
        [100, 100, 20]; for 2-D, None keeps each method's default.
    verbose : bool, default False
        Print progress and the final table.

    Returns
    -------
    pandas.DataFrame
        One row per method, 'method' first (display names such as
        '2. syntx.syn PyTorch MPS'), then the scorer's columns plus 'folding_pct',
        'min_jacobian', 'harmonic_energy', 'bending_energy' (0 % / 1 / 0 / 0 when there is no
        warp file).

    Notes
    -----
    Every arm runs only its deformable stage on the shared affine (the ANTs arm uses
    ``type_of_transform='SyNOnly'``). The '... MPS' methods pass ``device='mps'``; the '... CPU' ones ``device='cpu'``.
    """
    # 1. Parse dataset inputs
    canonical_key = 'custom_pair'
    
    if fixed is not None and moving is not None and fixed_label is not None and moving_label is not None:
        ds_fixed, ds_moving, ds_fixed_lbl, ds_moving_lbl = fixed, moving, fixed_label, moving_label
        canonical_key = 'custom_user_images'
    elif isinstance(benchmark_name, dict):
        ds_fixed = benchmark_name['fixed']
        ds_moving = benchmark_name['moving']
        ds_fixed_lbl = benchmark_name['fixed_label']
        ds_moving_lbl = benchmark_name['moving_label']
        canonical_key = benchmark_name.get('key', 'custom_dict_pair')
    elif isinstance(benchmark_name, str):
        key_lower = benchmark_name.lower().strip()
        if key_lower in ('r16_r64', '2d', 'r16', 'r64'):
            canonical_key = 'r16_r64'
            ds = syntx.benchmark_data('r16_r64')
        elif key_lower in ('mbhard', '3d', 'mindboggle', 'mindboggle_hard', 'mb_hard', 'hard_pair'):
            canonical_key = 'mbhard'
            ds = syntx.benchmark_data('mbhard')
        else:
            canonical_key = key_lower
            ds = syntx.benchmark_data(benchmark_name)
            
        ds_fixed = ds['fixed']
        ds_moving = ds['moving']
        ds_fixed_lbl = ds['fixed_label']
        ds_moving_lbl = ds['moving_label']
    else:
        raise ValueError(
            "Invalid benchmark_name or image inputs. Provide a string key, a dataset dict, "
            "or fixed/moving/fixed_label/moving_label ANTsImage arguments."
        )

    # 2. Select evaluation paradigm based on image dimensionality
    dim = ds_fixed.dimension
    if dim == 2:
        eval_fn = _evaluate_2d_r16_r64
    else:
        eval_fn = _evaluate_3d_mbhard
        if reg_iterations is None:
            reg_iterations = [100, 100, 20]

    # Standardize methods list based on model choice if methods is None
    if methods is None:
        model_lower = str(model).lower().strip()
        if model_lower == 'syn':
            methods = ['antspy_cpp', 'pytorch_mps', 'pytorch_cpu', 'jax_cpu']
        elif model_lower == 'tvf':
            methods = ['tvf_pytorch_mps', 'tvf_pytorch_cpu', 'tvf_jax_cpu']
        elif model_lower in ('all', 'both'):
            methods = ['antspy_cpp', 'pytorch_mps', 'pytorch_cpu', 'jax_cpu', 'tvf_pytorch_mps', 'tvf_pytorch_cpu', 'tvf_jax_cpu']
        else:
            methods = ['antspy_cpp', 'pytorch_mps', 'pytorch_cpu', 'jax_cpu']

    method_map = {}
    for m in methods:
        m_lower = str(m).lower().strip()
        if m_lower in ('antspy_cpp', 'antspy', 'cpp'):
            method_map['1. ANTsPy C++ SyN (cc)'] = ('syn', 'antspy_cpp', None, None)
        elif m_lower in ('pytorch_mps', 'mps', 'syn_mps'):
            method_map['2. syntx.syn PyTorch MPS'] = ('syn', 'pytorch', 'mps', 'pytorch')
        elif m_lower in ('pytorch_cpu', 'pytorch', 'py_cpu', 'syn_cpu'):
            method_map['3. syntx.syn PyTorch CPU'] = ('syn', 'pytorch', 'cpu', 'pytorch')
        elif m_lower in ('jax_cpu', 'jax', 'jax_backend', 'syn_jax'):
            method_map['4. syntx.syn JAX CPU'] = ('syn', 'jax', 'cpu', 'jax')
        elif m_lower in ('tvf_pytorch_mps', 'tvf_mps'):
            method_map['5. syntx.tvf PyTorch MPS'] = ('tvf', 'pytorch', 'mps', 'pytorch')
        elif m_lower in ('tvf_pytorch_cpu', 'tvf_pytorch', 'tvf_py_cpu'):
            method_map['6. syntx.tvf PyTorch CPU'] = ('tvf', 'pytorch', 'cpu', 'pytorch')
        elif m_lower in ('tvf_jax_cpu', 'tvf_jax', 'tvf_jax_backend'):
            method_map['7. syntx.tvf JAX CPU'] = ('tvf', 'jax', 'cpu', 'jax')
        else:
            raise ValueError(f"Unknown method '{m}'. Supported methods: 'antspy_cpp', 'pytorch_mps', 'pytorch_cpu', 'jax_cpu', 'tvf_pytorch_mps', 'tvf_pytorch_cpu', 'tvf_jax_cpu'.")

    if verbose:
        print(f"\n==========================================================================")
        print(f" STARTING HIGH-LEVEL BENCHMARK: `{canonical_key.upper()}` ({dim}D)")
        print(f" Methods Queued: {list(method_map.keys())}")
        print(f"==========================================================================")

    # 2. Base Affine Initialization (Hybrid Deterministic: Always on CPU)
    initial_transform = None
    if initial_transform is None:
        if verbose:
            print(f"Computing deterministic robust affine alignment on CPU...")
        import torch
        import numpy as np
        import random
        # Lock global seeds before affine to ensure perfect CPU reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        res_aff = syntx.robust_affine(fixed=ds_fixed, moving=ds_moving, mode='auto', device='cpu', multi_start=True, verbose=False)
        initial_transform = res_aff['fwdtransforms'][0]


    if verbose:
        print(f"[*] Initial ANTsPy Affine alignment completed.\n")

    records = []

    for display_name, (model_type, backend_type, device_val, backend_val) in method_map.items():
        if verbose:
            print(f"[*] Executing {display_name}...", flush=True)

        t_start = time.time()

        if backend_type == 'antspy_cpp':
            reg_args = {
                'fixed': ds_fixed,
                'moving': ds_moving,
                'type_of_transform': 'SyNOnly',
                'initial_transform': initial_transform,
                'syn_metric': 'cc',
                'syn_sampling': 2,
                'verbose': False
            }
            if grad_step is not None:
                reg_args['grad_step'] = grad_step
            if reg_iterations is not None:
                reg_args['reg_iterations'] = reg_iterations

            res = ants.registration(**reg_args)
            fwdtransforms = res['fwdtransforms']
            invtransforms = res['invtransforms']

        elif model_type == 'syn':
            syn_kwargs = {
                'fixed': ds_fixed,
                'moving': ds_moving,
                'initial_transform': initial_transform,
                'backend': backend_val,
                'verbose': False
            }
            syn_overrides = {
                'grad_step': grad_step,
                'flow_sigma': fluid_sigma,
                'total_sigma': elastic_sigma,
                'syn_sampling': lncc_radius,
                'inverse_steps': inverse_steps,
                'regularizer': syn_regularizer,
                'fast_smooth': syn_fast_smooth,
                'use_analytical_gradients': syn_use_analytical_gradients,
                'inverse_method': syn_inverse_method,
            }
            syn_kwargs.update({k: v for k, v in syn_overrides.items() if v is not None})
            if device_val is not None:
                syn_kwargs['device'] = device_val
            if reg_iterations is not None:
                syn_kwargs['reg_iterations'] = reg_iterations

            res = syntx.syn(**syn_kwargs)
            fwdtransforms = res['fwdtransforms']
            invtransforms = res['invtransforms']

        elif model_type == 'tvf':
            tvf_kwargs = {
                'fixed': ds_fixed,
                'moving': ds_moving,
                'initial_transform': initial_transform,
                'backend': backend_val,
                'verbose': False,
                **(tvf_overrides or {}),
            }
            if device_val is not None:
                tvf_kwargs['device'] = device_val
            if reg_iterations is not None:
                tvf_kwargs['reg_iterations'] = reg_iterations

            res = syntx.tvf(**tvf_kwargs)
            fwdtransforms = res['fwdtransforms']
            invtransforms = res['invtransforms']

        t_elapsed = time.time() - t_start

        # Evaluate quantitative overlap metrics
        rec = eval_fn(ds_fixed, ds_moving, ds_fixed_lbl, ds_moving_lbl, fwdtransforms, invtransforms, t_elapsed,
                      whichtoinvert_inv=res.get('whichtoinvert_inv'))
        rec['method'] = display_name
        
        # Jacobian / energy statistics of the exported warp (NaN with a warning on failure)
        try:
            from .metrics import warp_jacobian_and_energies
            warp_path = next((p for p in fwdtransforms if isinstance(p, str) and p.endswith('.nii.gz')), None)
            rec.update(warp_jacobian_and_energies(ds_fixed, warp_path))
        except Exception as e:
            import warnings
            warnings.warn(f"high_level_benchmark_run: Jacobian / energy metrics failed for {rec.get('method')}: {e}")
            rec.update({k: float('nan') for k in ('folding_pct', 'min_jacobian', 'harmonic_energy', 'bending_energy')})

        records.append(rec)

        # Memory cleanup between methods to prevent accumulation
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        if verbose:
            print(f"    Completed {display_name} | Mean Sym Dice: {rec['mean_sym_dice']:.4f} [{t_elapsed:.2f}s]")

    df_results = pd.DataFrame(records)
    # Re-order columns to put method first
    cols = ['method'] + [c for c in df_results.columns if c != 'method']
    df_results = df_results[cols]

    if verbose:
        print(f"\n==========================================================================")
        print(f" BENCHMARK `{canonical_key.upper()}` RESULTS SUMMARY")
        print(f"==========================================================================")
        print(df_results.to_string(index=False))
        print(f"==========================================================================\n")

    return df_results
