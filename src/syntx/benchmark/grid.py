"""
Benchmark grid definitions for syntx registration suite.

Provides standardized configuration grids for:
- Phase 1: 2D Dataset Sweeps ('r16_r64', 'c', 'ellipse') across 30 parameter combinations.
- Phase 2: 3D Dataset Sweep ('mbhard') across 30 parameter combinations.
- Phase 3: 90-Pair Mindboggle Population Evaluation across Top 5 parameter configurations.
"""

from typing import List, Dict, Any


def build_30_grid() -> List[Dict[str, Any]]:
    """Builds the canonical 30-combination parameter grid across SyN and TVF."""
    grid = []
    
    # SyN Tuples: S1, S2, S3
    syn_tuples = {
        'S1': {'flow_sigma': 1.0, 'grad_step': 0.25},
        'S2': {'flow_sigma': 3.0, 'grad_step': 0.25},
        'S3': {'flow_sigma': 3.0, 'grad_step': 0.50},
    }
    for tuple_name, params in syn_tuples.items():
        for reg in ['gaussian', 'sobolev', 'dsti']:
            for fast in [True, False]:
                grid.append({
                    'id': f"syn_{reg}_fast{fast}_{tuple_name}",
                    'model': 'syn',
                    'regularizer': reg,
                    'fast_smooth': fast,
                    'tuple_name': tuple_name,
                    'params': params
                })

    # TVF Tuples: T1, T2
    tvf_tuples = {
        'T1': {'flow_sigma': 1.5, 'grad_step': 0.90, 'total_sigma': 0.05},
        'T2': {'flow_sigma': 0.4, 'grad_step': 0.50, 'total_sigma': 0.05},
    }
    for tuple_name, params in tvf_tuples.items():
        for reg in ['gaussian', 'sobolev', 'dsti']:
            for fast in [True, False]:
                grid.append({
                    'id': f"tvf_{reg}_fast{fast}_{tuple_name}",
                    'model': 'tvf',
                    'regularizer': reg,
                    'fast_smooth': fast,
                    'tuple_name': tuple_name,
                    'params': params
                })

    return grid


def get_phase1_tasks() -> List[Dict[str, Any]]:
    """Generates all task definitions for Phase 1 (2D sweeps)."""
    grid = build_30_grid()
    datasets = ['r16_r64', 'c', 'ellipse']
    tasks = []
    for ds in datasets:
        for cfg in grid:
            task_id = f"phase1_{ds}_{cfg['id']}"
            tasks.append({
                'task_id': task_id,
                'phase': 1,
                'dataset': ds,
                'config': cfg
            })
    return tasks


def get_phase2_tasks() -> List[Dict[str, Any]]:
    """Generates all task definitions for Phase 2 (3D 'mbhard' sweep)."""
    grid = build_30_grid()
    tasks = []
    for cfg in grid:
        task_id = f"phase2_mbhard_{cfg['id']}"
        tasks.append({
            'task_id': task_id,
            'phase': 2,
            'dataset': 'mbhard',
            'config': cfg
        })
    return tasks
