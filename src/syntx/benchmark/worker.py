"""
Isolated worker entry point for running a single benchmark task in a separate process.

Executing each benchmark unit in a fresh Python process guarantees:
- 100% reclamation of PyTorch MPS Metal allocations & C++ ANTsPy ITK heap memory upon exit.
- Bound memory usage (O(1) memory overhead across long population sweeps).
"""

import sys
import os
import argparse
import json
import time
import gc
import numpy as np
import torch
import ants
import syntx


def compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv=None):
    """Computes bidirectional fixed, moving, and symmetric mean Dice scores."""
    if whichtoinvert_inv is None:
        whichtoinvert_inv = [True, False]

    # 1. Fixed Space Dice
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = ov_fixed[~ov_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fixed = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fixed.columns else 'TargetOverlap'
    dice_fixed = float(df_fixed[col_fixed].mean()) if len(df_fixed) > 0 else 0.0

    # 2. Moving Space Dice
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl,
        transformlist=invtransforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator='nearestNeighbor'
    )
    ov_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = ov_moving[~ov_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_moving = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_moving.columns else 'TargetOverlap'
    dice_moving = float(df_moving[col_moving].mean()) if len(df_moving) > 0 else 0.0

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym


def run_task(task_def: dict) -> dict:
    """Executes a single benchmark task and returns structured metrics."""
    phase = task_def.get('phase', 1)
    ds_key = task_def.get('dataset', 'r16_r64')
    cfg = task_def['config']
    config_id = cfg['id']

    t0 = time.time()
    
    # Load dataset
    data = syntx.benchmark_data(ds_key)
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    # Physical device selection
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Compute robust affine initialization if not provided
    aff_tx = cfg.get('initial_transform', None)
    if aff_tx is None:
        reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
        aff_tx = reg_aff['fwdtransforms'][0]

    # Execute non-linear registration
    if cfg['model'] == 'syn':
        reg = syntx.syn(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            backend='pytorch', device=device,
            reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
            similarity_metric='lncc', syn_sampling=2, inverse_method='anderson',
            total_sigma=0.0, regularizer=cfg['regularizer'], fast_smooth=cfg['fast_smooth'],
            antisymmetric=True, verbose=False, **cfg['params']
        )
    else:
        reg = syntx.tvf(
            fixed=fi, moving=mi, initial_transform=aff_tx,
            backend='pytorch', device=device,
            reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
            similarity_metric='lncc', syn_sampling=2, multipoint_loss=[0.0, 0.5, 1.0],
            optimizer='lars', cfl_max=0.0, cfl_momentum=0.95, n_time_steps=3,
            constant_speed=True, constant_speed_relaxation=0.10, use_analytical_gradients=True,
            regularizer=cfg['regularizer'], fast_smooth=cfg['fast_smooth'],
            antisymmetric=True, verbose=False, **cfg['params']
        )

    elapsed = time.time() - t0

    # Compute bidirectional Dice
    dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(
        fl, ml, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv')
    )

    # Class-specific scores for r16_r64
    class_scores = {}
    if ds_key == 'r16_r64' and 'fixed_labels' in data:
        fl_c2, ml_c2 = data['fixed_labels']['class2'], data['moving_labels']['class2']
        fl_c23, ml_c23 = data['fixed_labels']['class2_3'], data['moving_labels']['class2_3']
        fl_c3 = ants.threshold_image(data['fixed_labels']['otsu'], 3, 3)
        ml_c3 = ants.threshold_image(data['moving_labels']['otsu'], 3, 3)

        c2_f, c2_m, c2_s = compute_bidirectional_dice(fl_c2, ml_c2, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))
        c3_f, c3_m, c3_s = compute_bidirectional_dice(fl_c3, ml_c3, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))
        c23_f, c23_m, c23_s = compute_bidirectional_dice(fl_c23, ml_c23, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))

        class_scores = {
            'cortical_gm_c2_dice': c2_s,
            'white_matter_c3_dice': c3_s,
            'parenchyma_c23_dice': c23_s
        }

    # Jacobian determinant metrics
    jac_ants = ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0], do_log=False)
    jac_np = jac_ants.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0

    folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)
    min_j = float(jac_np[mask_eval].min())

    record = {
        'task_id': task_def.get('task_id', config_id),
        'phase': phase,
        'dataset': ds_key,
        'config_id': config_id,
        'model': cfg['model'],
        'regularizer': cfg['regularizer'],
        'fast_smooth': cfg['fast_smooth'],
        'tuple_name': cfg['tuple_name'],
        'dice_fixed': dice_fixed,
        'dice_moving': dice_moving,
        'dice_sym': dice_sym,
        'folding_pct': folding_pct,
        'min_jacobian': min_j,
        'runtime_seconds': elapsed,
        'class_scores': class_scores,
        'device': device,
        'status': 'SUCCESS'
    }

    # Clean up memory buffers before exiting worker
    del reg, jac_ants
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return record


def main():
    parser = argparse.ArgumentParser(description="syntx benchmark worker process")
    parser.add_argument("--task-json", required=True, help="Path to JSON file specifying task definition")
    parser.add_argument("--out-json", required=True, help="Path to write output result JSON")
    args = parser.parse_args()

    with open(args.task_json, 'r', encoding='utf-8') as f:
        task_def = json.load(f)

    try:
        record = run_task(task_def)
    except Exception as e:
        record = {
            'task_id': task_def.get('task_id', 'unknown'),
            'status': 'FAILED',
            'error': str(e),
            'runtime_seconds': 0.0
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)


if __name__ == '__main__':
    main()
