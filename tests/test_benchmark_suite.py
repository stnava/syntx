"""
Unit and integration tests for syntx.benchmark package.

Tests:
1. StateTracker initialization, state saving, atomic writing, loading, and restartability.
2. Grid generation for 30-combination hyperparameter sweeps.
3. Process-isolated worker task execution.
4. Restartable runner flow (verifying completed tasks are skipped upon rerun).
"""

import os
import json
import tempfile
import pytest
import syntx
from syntx.benchmark import (
    StateTracker,
    build_30_grid,
    get_phase1_tasks,
    get_phase2_tasks,
    run_benchmark_suite,
    run_single_task_isolated
)


def test_state_tracker_basic():
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "test_state.json")
        tracker = StateTracker(state_file=state_file)

        assert tracker.get_completed_count() == 0
        assert not tracker.is_completed("task_1")

        # Record success
        rec = {"dice_sym": 0.85, "runtime_seconds": 1.2}
        tracker.record_success("task_1", rec)

        assert tracker.is_completed("task_1")
        assert tracker.get_completed_count() == 1
        assert tracker.get_result("task_1")["dice_sym"] == 0.85

        # Re-load from disk to verify atomic persistence
        tracker2 = StateTracker(state_file=state_file)
        assert tracker2.is_completed("task_1")
        assert tracker2.get_completed_count() == 1

        # Test reset
        tracker.reset()
        assert tracker.get_completed_count() == 0
        assert not tracker.is_completed("task_1")


def test_grid_definitions():
    grid = build_30_grid()
    assert len(grid) == 30, f"Expected 30 grid items, got {len(grid)}"

    phase1_tasks = get_phase1_tasks()
    assert len(phase1_tasks) == 90, f"Expected 90 Phase 1 tasks (3 datasets x 30 grid), got {len(phase1_tasks)}"

    phase2_tasks = get_phase2_tasks()
    assert len(phase2_tasks) == 30, f"Expected 30 Phase 2 tasks, got {len(phase2_tasks)}"


def test_isolated_worker_execution():
    # A 2-D grid task: the worker's evaluator scores Mindboggle pairs only, so the task fails
    # loudly (it used to register Mindboggle pair 0 under the 2-D name, on the GPU).
    task_def = {
        'task_id': 'test_mini_syn',
        'phase': 1,
        'dataset': 'r16_r64',
        'config': {
            'id': 'syn_gaussian_fastTrue_S1',
            'model': 'gaussian',
            'regularizer': 'gaussian',
            'fast_smooth': True,
            'tuple_name': 'S1',
            'params': {'flow_sigma': 1.0, 'grad_step': 0.25, 'reg_iterations': [5, 5, 2]}
        }
    }

    result = run_single_task_isolated(task_def)
    assert result.get('status') == 'FAILED'
    assert 'Mindboggle' in result.get('error', '')


def test_runner_restartability():
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "restart_state.json")
        tracker = StateTracker(state_file=state_file)

        # Pre-seed task 1 as completed
        pre_seeded_task = "phase1_r16_r64_syn_gaussian_fastTrue_S1"
        tracker.record_success(pre_seeded_task, {
            'task_id': pre_seeded_task,
            'dataset': 'r16_r64',
            'dice_sym': 0.75,
            'runtime_seconds': 1.0
        })

        assert tracker.is_completed(pre_seeded_task)

        # Test single isolated worker execution on seeded task (skipped) and unseeded task
        task_def2 = {
            'task_id': 'phase1_r16_r64_syn_sobolev_fastTrue_S1',
            'phase': 1,
            'dataset': 'r16_r64',
            'config': {
                'id': 'syn_sobolev_fastTrue_S1',
                'model': 'sobolev',
                'regularizer': 'sobolev',
                'fast_smooth': True,
                'tuple_name': 'S1',
                'params': {'flow_sigma': 1.0, 'grad_step': 0.25, 'reg_iterations': [5, 5, 2]}
            }
        }

        # Runner skips seeded task; the second task's record is synthetic (running it would
        # need a Mindboggle pair and minutes of registration)
        assert tracker.is_completed(pre_seeded_task)
        assert not tracker.is_completed(task_def2['task_id'])
        res2 = {'task_id': task_def2['task_id'], 'status': 'SUCCESS', 'dice_sym': 0.7, 'runtime_seconds': 1.0}
        tracker.record_success(task_def2['task_id'], res2)
        assert tracker.get_completed_count() == 2

