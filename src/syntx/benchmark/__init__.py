"""
syntx.benchmark — registration benchmarks (Mindboggle-101 pairs and 2-D examples)
=================================================================================

What is here:

- ``data``: Mindboggle data directory resolution, integrity check, pair loading (with a disk
  cache of ANTsTorch N4-corrected volumes), and a helper that organizes a raw download.
- ``evaluate``: ``evaluate_mindboggle_pair`` (alias ``evaluate_pair``) -- canonical affine +
  one deformable method on one pair, returning Dice / Jacobian / inverse-error metrics with a
  provenance manifest; ``evaluate_affine_benchmark``; ``run_standard_report_demo``.
- ``orchestrator`` / ``cli``: multi-pair Mindboggle runs, one subprocess per (pair, model),
  with a resumable per-pair JSON cache and a summary JSON + HTML report
  (``python -m syntx.benchmark``).
- ``runner`` / ``grid`` / ``state`` / ``worker``: a restartable 30-configuration grid suite
  run task by task in subprocesses.
- ``high_level``: ``high_level_benchmark_run`` -- ANTs vs syntx.syn / syntx.tvf backends on
  one pair, returned as a DataFrame.
- ``metrics``: ``compute_pair_metrics`` for an arbitrary registration result.
- ``config``: ``DEFAULT_BENCHMARK_CONFIG`` (per-method parameter blocks) and hashing helpers.
- ``html_report``: an auto-refreshing per-model dashboard read from result directories.
- ``tune`` / ``codify`` (not imported here): automated parameter tuning and committing a
  tuning winner as new defaults on a git branch.
- ``msd`` (not imported here): ``auto_reg`` evaluation on Medical Segmentation Decathlon tasks.

``compute_bidirectional_dice`` / ``compute_jacobian_metrics`` (from
``syntx.deformation_metrics``) and the two report builders from ``syntx.viz.reports`` are
re-exported for convenience. ``evaluate_affine_benchmark`` is importable from this package
but is not listed in ``__all__``.
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
from .config import (
    DEFAULT_BENCHMARK_CONFIG,
    get_default_config,
    get_model_config,
    compute_config_hash,
    validate_config_compatibility,
)
from .html_report import (
    generate_live_html_report,
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
    "DEFAULT_BENCHMARK_CONFIG",
    "get_default_config",
    "get_model_config",
    "compute_config_hash",
    "validate_config_compatibility",
    "generate_live_html_report",
]
