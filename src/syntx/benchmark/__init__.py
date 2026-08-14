"""
syntx.benchmark — Professional Registration Evaluation System
===============================================================

Provides automated, memory-safe, restartable benchmarking for PyTorch/JAX
diffeomorphic registration on Mindboggle DKT31 cortical labels.

Architecture
------------
The benchmark system has four layers:

1. **Metrics** (``syntx.benchmark.metrics.compute_pair_metrics``):
   Unified function computing all mandated metrics (bidirectional Dice,
   Jacobian folding, harmonic/bending energy, MI, LNCC) from a single
   registration result.

2. **Single-Pair Runner** (``scripts/run_single_pair.py``):
   CLI script that loads one pair from ``examples/pairs.csv``, runs
   registration with config from ``docs/provenance/run_config.json``,
   and writes an atomic JSON result file.

3. **90-Pair Orchestrator** (``scripts/run_90pairs.py``):
   Launches single-pair runners in isolated subprocesses with automatic
   crash recovery (``--resume``), ETA estimation, and aggregate reporting.

4. **High-Level API** (``syntx.benchmark.high_level_benchmark_run``):
   Programmatic one-liner for quick interactive benchmarks across
   multiple backends (ANTsPy C++, PyTorch MPS/CPU, JAX CPU).

Configuration
-------------
Parameters are defined in ``docs/provenance/run_config.json`` with
schema documentation in ``docs/provenance/CONFIG_SCHEMA.md``.

Quick Start
-----------
    # Single pair
    python scripts/run_single_pair.py --pair-idx 0 --model syn --device mps

    # Full 90-pair benchmark with resume
    python scripts/run_90pairs.py --model syn --device mps --resume
"""


from .state import StateTracker
from .grid import build_30_grid, get_phase1_tasks, get_phase2_tasks
from .runner import run_benchmark_suite, run_single_task_isolated
from .high_level import high_level_benchmark_run
from .metrics import compute_pair_metrics

__all__ = [
    "StateTracker",
    "build_30_grid",
    "get_phase1_tasks",
    "get_phase2_tasks",
    "run_benchmark_suite",
    "run_single_task_isolated",
    "high_level_benchmark_run",
    "compute_pair_metrics"
]

