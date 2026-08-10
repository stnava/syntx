"""
syntx.benchmark.high_level — Unified High-Level Registration Benchmark Suite
==============================================================================

Provides a single 1-liner orchestrator function `high_level_benchmark_run` to execute
2D (r16 -> r64) and 3D (Mindboggle Hard or any arbitrary Mindboggle pair) diffeomorphic registration
benchmarks across SyN and TVF (Time-Varying Velocity Field) models in native C++ ANTsPy, PyTorch MPS GPU,
PyTorch CPU, and JAX CPU backends.
"""

import time
import os
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
    """Evaluates 2D r16 -> r64 Label 2 (Gray Matter) and Label 3 (White Matter) Otsu Dice scores."""
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
        if len(filtered) > 0:
            return float(pd.to_numeric(filtered['TotalOrTargetOverlap'], errors='coerce').dropna().mean())
        else:
            return float(pd.to_numeric(ov['TotalOrTargetOverlap'], errors='coerce').iloc[0])

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
    """Evaluates 3D Mindboggle DKT31 Cortical Dice scores symmetrically."""
    ml_w = ants.apply_transforms(fixed=fixed, moving=moving_label, transformlist=fwdtransforms, interpolator='nearestNeighbor')
    fl_w = ants.apply_transforms(fixed=moving, moving=fixed_label, transformlist=invtransforms, interpolator='nearestNeighbor')

    ov_fix = ants.label_overlap_measures(fixed_label.clone('unsigned int'), ml_w.clone('unsigned int'))
    ov_fix['L_num'] = pd.to_numeric(ov_fix['Label'], errors='coerce')
    dkt_fix_df = ov_fix[ov_fix['L_num'] > 0]
    fix_d = float(pd.to_numeric(dkt_fix_df['TotalOrTargetOverlap'], errors='coerce').dropna().mean())

    ov_mov = ants.label_overlap_measures(moving_label.clone('unsigned int'), fl_w.clone('unsigned int'))
    ov_mov['L_num'] = pd.to_numeric(ov_mov['Label'], errors='coerce')
    dkt_mov_df = ov_mov[ov_mov['L_num'] > 0]
    mov_d = float(pd.to_numeric(dkt_mov_df['TotalOrTargetOverlap'], errors='coerce').dropna().mean())

    sym_d = 0.5 * (fix_d + mov_d)

    return {
        'fixed_dkt31_dice': fix_d,
        'moving_dkt31_dice': mov_d,
        'sym_mean_dkt31_dice': sym_d,
        'mean_sym_dice': sym_d,
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
    grad_step: float = 0.5,
    fluid_sigma: float = 3.0,
    elastic_sigma: float = 0.0,
    lncc_radius: int = 2,
    inverse_steps: int = 10,
    tvf_grad_step: float = 0.35,
    tvf_flow_sigma: float = 0.4,
    tvf_total_sigma: float = 0.5,
    tvf_cfl_momentum: float = 0.95,
    tvf_n_time_steps: int = 3,
    reg_iterations: Optional[List[int]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Executes high-level diffeomorphic registration benchmark suites across SyN and TVF models.

    Parameters
    ----------
    benchmark_name : str or dict, default='r16_r64'
        Canonical benchmark dataset name or dataset dictionary:
        - `'r16_r64'`, `'2d'`, or `'r16'`: 2D brain slice registration with Label 2 & 3 Otsu Dice evaluation.
        - `'mbhard'`, `'3d'`, or `'mindboggle'`: 3D Mindboggle Hard Pair 00 with Cortical DKT31 Dice evaluation.
        - `dict`: Dataset dictionary with keys `{'fixed', 'moving', 'fixed_label', 'moving_label'}`.
    methods : list of str, optional
        Methods/backends to execute. Supported keys:
        - SyN: `'antspy_cpp'`, `'pytorch_mps'`, `'pytorch_cpu'`, `'jax_cpu'`
        - TVF: `'tvf_pytorch_mps'`, `'tvf_pytorch_cpu'`, `'tvf_jax_cpu'`
    model : str, default='syn'
        Model family to execute if methods is None: `'syn'`, `'tvf'`, or `'all'`.
    fixed, moving, fixed_label, moving_label : ANTsImage, optional
        Direct ANTsImage instances.
    grad_step, fluid_sigma, elastic_sigma, lncc_radius, inverse_steps : float/int
        Peak SyN model parameters.
    tvf_grad_step, tvf_flow_sigma, tvf_total_sigma, tvf_cfl_momentum, tvf_n_time_steps : float/int
        Peak TVF model parameters.
    reg_iterations : list of int, optional
        Multiresolution pyramid iteration schedule.
    verbose : bool, default=True
        If True, prints progress updates and formatted summary table.

    Returns
    -------
    pd.DataFrame
        Formatted pandas DataFrame containing quantitative metrics and runtimes per method.
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

    # Pre-align with fast ANTsPy Affine
    t0 = time.time()
    res_aff = ants.registration(fixed=ds_fixed, moving=ds_moving, type_of_transform='Affine', verbose=False)
    t_aff = time.time() - t0
    initial_transform = res_aff['fwdtransforms']


    if verbose:
        print(f"[*] Initial ANTsPy Affine alignment completed in {t_aff:.2f}s.\n")

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
                'grad_step': grad_step,
                'verbose': False
            }
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
                'grad_step': grad_step,
                'fluid_sigma': fluid_sigma,
                'elastic_sigma': elastic_sigma,
                'lncc_radius': lncc_radius,
                'inverse_steps': inverse_steps,
                'antisymmetric': True,
                'backend': backend_val,
                'verbose': False
            }
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
                'grad_step': tvf_grad_step,
                'flow_sigma': tvf_flow_sigma,
                'total_sigma': tvf_total_sigma,
                'regularizer': 'dsti',
                'fast_smooth': False,
                'antisymmetric': True,
                'cfl_momentum': tvf_cfl_momentum,
                'n_time_steps': tvf_n_time_steps,
                'use_analytical_gradients': True,
                'constant_speed': True,
                'constant_speed_relaxation': 0.10,
                'multipoint_loss': [0.0, 0.5, 1.0],
                'backend': backend_val,
                'verbose': False
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
        records.append(rec)

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
