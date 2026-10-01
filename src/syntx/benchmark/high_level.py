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
    runtime: float
) -> Dict[str, Any]:
    """2-D scoring: Dice of label 2 and of label 3 (for the r16/r64 3-class Otsu labels,
    roughly grey and white matter), each in fixed and moving space.

    Each label is binarised, warped with nearest-neighbour interpolation (moving -> fixed by
    ``fwdtransforms``, fixed -> moving by ``invtransforms`` as given, with no
    ``whichtoinvert``), and scored with ``ants.label_overlap_measures`` ('MeanOverlap').

    Returns
    -------
    dict
        'label2_fix_dice', 'label2_mov_dice', 'label2_sym_dice' (mean of the two), the same
        for label 3, 'mean_sym_dice' (mean of the two symmetric values), 'dice_fixed',
        'dice_moving', 'dice_sym' (all three equal to 'mean_sym_dice', not separate fixed /
        moving values), 'runtime_seconds' (= ``runtime``).
    """
    fl_l2 = fixed_label.threshold_image(2, 2)
    ml_l2 = moving_label.threshold_image(2, 2)

    fl_l3 = fixed_label.threshold_image(3, 3)
    ml_l3 = moving_label.threshold_image(3, 3)

    # Warp moving labels to fixed space, and fixed labels to moving space
    ml_l2_w = ants.apply_transforms(fixed=fixed, moving=ml_l2, transformlist=fwdtransforms, interpolator='nearestNeighbor')
    fl_l2_w = ants.apply_transforms(fixed=moving, moving=fl_l2, transformlist=invtransforms, interpolator='nearestNeighbor')

    ml_l3_w = ants.apply_transforms(fixed=fixed, moving=ml_l3, transformlist=fwdtransforms, interpolator='nearestNeighbor')
    fl_l3_w = ants.apply_transforms(fixed=moving, moving=fl_l3, transformlist=invtransforms, interpolator='nearestNeighbor')

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

    d2_fix = get_dice(fl_l2, ml_l2_w)
    d2_mov = get_dice(ml_l2, fl_l2_w)
    d2_sym = 0.5 * (d2_fix + d2_mov)

    d3_fix = get_dice(fl_l3, ml_l3_w)
    d3_mov = get_dice(ml_l3, fl_l3_w)
    d3_sym = 0.5 * (d3_fix + d3_mov)

    mean_sym = 0.5 * (d2_sym + d3_sym)

    return {
        'label2_fix_dice': d2_fix,
        'label2_mov_dice': d2_mov,
        'label2_sym_dice': d2_sym,
        'label3_fix_dice': d3_fix,
        'label3_mov_dice': d3_mov,
        'label3_sym_dice': d3_sym,
        'mean_sym_dice': mean_sym,
        # Unified metric aliases for cross-module consistency
        'dice_fixed': mean_sym,
        'dice_moving': mean_sym,
        'dice_sym': mean_sym,
        'runtime_seconds': runtime
    }


def _evaluate_3d_mbhard(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fixed_label: ants.ANTsImage,
    moving_label: ants.ANTsImage,
    fwdtransforms: List[str],
    invtransforms: List[str],
    runtime: float
) -> Dict[str, Any]:
    """3-D scoring: mean DKT31 label Dice in fixed and moving space, plus image similarity.

    Labels are warped with nearest-neighbour interpolation (moving -> fixed by
    ``fwdtransforms``, fixed -> moving by ``invtransforms`` as given, with no
    ``whichtoinvert``); Dice is 'MeanOverlap' of ``ants.label_overlap_measures`` averaged over
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
    fl_w = ants.apply_transforms(fixed=moving, moving=fixed_label, transformlist=invtransforms, interpolator='nearestNeighbor')

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
    the first ``.nii.gz`` forward transform (finite-difference
    ``ants.create_jacobian_determinant_image`` inside ``ants.get_mask(fixed)``; energies as in
    ``syntx.benchmark.compute_pair_metrics``). If those statistics fail, the columns are
    missing for that method (warning printed with ``verbose``).

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
    The ANTs arm uses ``type_of_transform='SyN'``, which runs ANTs' own affine stage after the
    initial transform, while the syntx arms run only their deformable stage on the shared
    affine. The '... MPS' methods pass ``device='mps'``; the '... CPU' ones ``device='cpu'``.
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
                'type_of_transform': 'SyN',
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
        rec = eval_fn(ds_fixed, ds_moving, ds_fixed_lbl, ds_moving_lbl, fwdtransforms, invtransforms, t_elapsed)
        rec['method'] = display_name
        
        # Calculate jacobian and topological metrics
        try:
            # Find the nonlinear warp in the list
            warp_path = next((p for p in fwdtransforms if isinstance(p, str) and p.endswith('.nii.gz')), None)
            if warp_path is not None:
                warp_img = ants.image_read(warp_path)
                
                # 1. Jacobian Metrics
                jac_ants = ants.create_jacobian_determinant_image(ds_fixed, warp_img, do_log=False)
                jac_arr = jac_ants.numpy()
                valid_mask = ants.get_mask(ds_fixed).numpy() > 0
                
                rec['folding_pct'] = float(np.mean(jac_arr[valid_mask] <= 0) * 100)
                rec['min_jacobian'] = float(jac_arr[valid_mask].min())
                
                # 2. Harmonic & Bending Energy (from non-linear warp)
                dim = warp_img.dimension
                spc = warp_img.spacing
                warpnp = warp_img.numpy()
                
                # 1st order gradients: du_k / dx_i
                gradient_list = [np.gradient(warpnp[..., k], *spc, axis=range(dim)) for k in range(dim)]
                total_bnd, total_hrm = 0.0, 0.0
                
                for k in range(dim):
                    for j in range(dim):
                        grad_kj = gradient_list[k][j]
                        total_hrm += float(np.mean(grad_kj**2))
                        
                        # 2nd order gradients: d^2 u_k / dx_i dx_j
                        grad2_kj = np.gradient(grad_kj, *spc, axis=range(dim))
                        for i in range(dim):
                            total_bnd += float(np.mean(grad2_kj[i]**2))
                            
                rec['harmonic_energy'] = total_hrm
                rec['bending_energy'] = total_bnd
            else:
                rec['folding_pct'] = 0.0
                rec['min_jacobian'] = 1.0
                rec['harmonic_energy'] = 0.0
                rec['bending_energy'] = 0.0
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to compute topological metrics: {e}")
            
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
