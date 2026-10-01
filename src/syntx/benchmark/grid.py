"""
Configuration grid for the restartable benchmark suite (``syntx.benchmark.runner``).

``build_30_grid`` returns 30 configurations: (3 SyN parameter tuples + 2 TVF tuples) x 3
regularizers ('gaussian', 'sobolev', 'dsti') x ``fast_smooth`` in (True, False). Phase 1
runs the grid on the 2-D keys 'r16_r64', 'c' and 'ellipse' (90 tasks); phase 2 on 'mbhard'
(30 tasks). There is no phase-3 task generator.

Caveat (see ``syntx.benchmark.worker``): the evaluator these tasks are run through,
``evaluate_mindboggle_pair``, ignores the dataset key and the top-level ``regularizer`` /
``fast_smooth`` entries of a configuration; only the ``params`` values are applied.
"""

from typing import List, Dict, Any


def build_30_grid() -> List[Dict[str, Any]]:
    """Return the 30 grid configurations.

    SyN tuples: S1 (flow_sigma 1.0, grad_step 0.25), S2 (3.0, 0.25), S3 (3.0, 0.50).
    TVF tuples: T1 (flow_sigma 1.5, grad_step 0.90, total_sigma 0.05),
    T2 (0.4, 0.50, 0.05). Each tuple is crossed with regularizer in ('gaussian', 'sobolev',
    'dsti') and fast_smooth in (True, False).

    Returns
    -------
    list of dict
        Keys: 'id' (e.g. ``'syn_gaussian_fastTrue_S1'``), 'model' ('syn' or 'tvf'),
        'regularizer', 'fast_smooth', 'tuple_name', 'params' (the tuple's dict; the same dict
        object is shared by the six configurations of a tuple).
    """
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
    """Phase-1 tasks: the 30 grid configurations on each of 'r16_r64', 'c', 'ellipse' (90 tasks).

    Returns
    -------
    list of dict
        Keys: 'task_id' (``'phase1_<dataset>_<config id>'``), 'phase' (1), 'dataset',
        'config' (a ``build_30_grid`` entry).
    """
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
    """Phase-2 tasks: the 30 grid configurations on 'mbhard' (30 tasks).

    Returns
    -------
    list of dict
        Keys: 'task_id' (``'phase2_mbhard_<config id>'``), 'phase' (2), 'dataset' ('mbhard'),
        'config'.
    """
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
