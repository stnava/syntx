"""
syntx.benchmark — Mindboggle-101 Registration Evaluation Suite
==============================================================

Core tools for dataset integrity checking, pair loading, standardized
subprocess-isolated execution, metric computation, and HTML dashboard compilation.
"""

from .data import (
    check_mindboggle_data,
    load_mindboggle_pair,
    get_n4_cached_subject_volume,
    precompute_mindboggle_n4,
    resolve_data_dir,
    organize_mindboggle_data,
    MINDBOGGLE_SETUP_INSTRUCTIONS,
    DEFAULT_PAIRS_CSV,
    DEFAULT_DATA_DIR,
    DEFAULT_DATA_DIR_ENV,
)
from .evaluate import (
    evaluate_mindboggle_pair,
    evaluate_pair,
    evaluate_affine_benchmark,
    normalize_intensity,
    run_standard_report_demo,
)
from .orchestrator import (
    run_mindboggle_benchmark,
)
from .runner import (
    run_benchmark_suite,
    run_single_task_isolated,
)
from .high_level import (
    high_level_benchmark_run,
)
from .metrics import (
    compute_pair_metrics,
)
from .state import (
    StateTracker,
)
from .grid import (
    build_30_grid,
    get_phase1_tasks,
    get_phase2_tasks,
)
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_jacobian_metrics,
)
from syntx.viz.reports import (
    create_affine_benchmark_report,
    create_population_benchmark_report,
)

__all__ = [
    "check_mindboggle_data",
    "load_mindboggle_pair",
    "get_n4_cached_subject_volume",
    "precompute_mindboggle_n4",
    "resolve_data_dir",
    "organize_mindboggle_data",
    "evaluate_mindboggle_pair",
    "evaluate_pair",
    "normalize_intensity",
    "run_mindboggle_benchmark",
    "run_benchmark_suite",
    "run_single_task_isolated",
    "high_level_benchmark_run",
    "compute_pair_metrics",
    "compute_bidirectional_dice",
    "compute_jacobian_metrics",
    "create_affine_benchmark_report",
    "create_population_benchmark_report",
    "run_standard_report_demo",
    "StateTracker",
    "build_30_grid",
    "get_phase1_tasks",
    "get_phase2_tasks",
    "MINDBOGGLE_SETUP_INSTRUCTIONS",
    "DEFAULT_PAIRS_CSV",
    "DEFAULT_DATA_DIR",
    "DEFAULT_DATA_DIR_ENV",
]
