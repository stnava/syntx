"""
Subprocess worker for ``syntx.benchmark.runner``: runs one grid task and writes its record.

    python -m syntx.benchmark.worker --task-json TASK.json --out-json OUT.json

The task (a ``syntx.benchmark.grid`` entry) is run through ``evaluate_pair``
(``evaluate_mindboggle_pair``) on 'mps' if available, else 'cpu'. That evaluator scores
Mindboggle pairs only: dataset 'mindboggle' is pairs.csv row 0, 'mbhard' row 44; any other
dataset (the 2-D grid keys) fails the task. Of the configuration only the ``params`` entries
take effect (as keyword overrides); the top-level ``regularizer`` / ``fast_smooth`` entries
are only copied into the output record.
"""
import os
import json
import argparse
import traceback
import torch
import syntx
from syntx.benchmark import evaluate_pair
from syntx.deformation_metrics import compute_bidirectional_dice

def run_task(task_def: dict) -> dict:
    """Run one task and return its flat result record.

    Parameters
    ----------
    task_def : dict
        Keys 'config' (required; a ``build_30_grid`` entry), 'dataset' (default
        'mindboggle'), 'task_id', 'phase'.

    Returns
    -------
    dict
        'task_id', 'phase', 'dataset', 'config_id', 'model', 'regularizer', 'fast_smooth',
        'tuple_name', 'dice_fixed', 'dice_moving', 'dice_sym', 'folding_pct', 'min_jacobian',
        'runtime_seconds', 'device', 'status' ('SUCCESS'), 'provenance' (the evaluator's
        manifest). Exceptions from the evaluator propagate; a dataset that is not a Mindboggle
        pair raises ValueError.
    """
    from syntx.benchmark.evaluate import MBHARD_PAIR_IDX, _MBHARD_KEYS
    ds_key = task_def.get('dataset', 'mindboggle')
    cfg = task_def['config']
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pair_idx = MBHARD_PAIR_IDX if str(ds_key).lower() in _MBHARD_KEYS else 0

    metrics = evaluate_pair(
        pair_idx=pair_idx,
        model=cfg['model'],
        device=device,
        dataset_key=ds_key,
        config=cfg,
        **cfg.get('params', {})
    )

    record = {
        'task_id': task_def.get('task_id', cfg['id']),
        'phase': task_def.get('phase', 1),
        'dataset': ds_key,
        'config_id': cfg['id'],
        'model': cfg['model'],
        'regularizer': cfg.get('regularizer'),
        'fast_smooth': cfg.get('fast_smooth'),
        'tuple_name': cfg.get('tuple_name'),
        'dice_fixed': metrics.get('dice_fixed', metrics.get('syntx_dice_fixed')),
        'dice_moving': metrics.get('dice_moving', metrics.get('syntx_dice_moving')),
        'dice_sym': metrics.get('dice_sym', metrics.get('syntx_dice_sym')),
        'folding_pct': metrics.get('folding_pct', metrics.get('syntx_fold')),
        'min_jacobian': metrics.get('min_jacobian', metrics.get('syntx_min_jac')),
        'runtime_seconds': metrics.get('runtime_seconds', metrics.get('syntx_time')),
        'device': device,
        'status': 'SUCCESS',
        'provenance': metrics.get('provenance'),
    }
    return record

def main():
    """CLI entry: read ``--task-json``, run it, write the record to ``--out-json``.

    Any exception is caught and written as ``{'task_id', 'status': 'FAILED', 'error',
    'traceback'}``; the process then exits with code 1.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.task_json, 'r') as f:
        task_def = json.load(f)

    try:
        record = run_task(task_def)
    except Exception as e:
        record = {
            'task_id': task_def.get('task_id', 'unknown'),
            'status': 'FAILED',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(record, f, indent=2)
    if record.get('status') != 'SUCCESS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
