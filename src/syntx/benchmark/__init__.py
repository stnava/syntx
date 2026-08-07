"""
syntx.benchmark — Restartable, Process-Isolated Benchmark Suite
================================================================

Provides automated, memory-safe, restartable benchmarking for PyTorch/JAX diffeomorphic registration.

Modules & Entry Points
----------------------
syntx.benchmark.run_benchmark_suite
    Main orchestrator function for running Phase 1 (2D), Phase 2 (3D), and Phase 3.
syntx.benchmark.StateTracker
    Atomic state tracking & JSON provenance manager.
syntx.benchmark.build_30_grid
    Standardized 30-combination hyperparameter grid definitions.
"""

from .state import StateTracker
from .grid import build_30_grid, get_phase1_tasks, get_phase2_tasks
from .runner import run_benchmark_suite, run_single_task_isolated

__all__ = [
    "StateTracker",
    "build_30_grid",
    "get_phase1_tasks",
    "get_phase2_tasks",
    "run_benchmark_suite",
    "run_single_task_isolated"
]
