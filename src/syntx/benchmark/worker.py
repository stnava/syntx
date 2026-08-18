import os
import json
import argparse
import traceback
import torch
import syntx
from syntx.benchmark import evaluate_pair
from syntx.deformation_metrics import compute_bidirectional_dice

def run_task(task_def: dict) -> dict:
    ds_key = task_def.get('dataset', 'r16_r64')
    cfg = task_def['config']
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    metrics = evaluate_pair(
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
        'status': 'SUCCESS'
    }
    return record

def main():
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

if __name__ == '__main__':
    main()
