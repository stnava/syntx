# Syntx Examples & Benchmarks

This directory contains tutorials, visual report generators, and benchmark scripts for `syntx` (SyN, TVF, SyNGS).

## Directory Structure

* **`tutorials/`**: User-facing runnable usage guides and API examples.
  * [`compare_metrics_tutorial.py`](file:///Users/stnava/data/syntx/examples/tutorials/compare_metrics_tutorial.py): Detailed guide comparing metric configurations.
  * [`run_auto_reg_example.py`](file:///Users/stnava/data/syntx/examples/tutorials/run_auto_reg_example.py): Example demonstrating automatic registration mode.
* **`benchmarks/`**: Suite benchmark runners and comparison report generators.
  * [`benchmark_suite.py`](file:///Users/stnava/data/syntx/examples/benchmarks/benchmark_suite.py): Standard benchmark suite runner.
  * [`compare_registration_backends_3d.py`](file:///Users/stnava/data/syntx/examples/benchmarks/compare_registration_backends_3d.py): 3D backend parity comparison (PyTorch vs JAX vs ANTs C++).
  * [`generate_ants_3d_comparison_report.py`](file:///Users/stnava/data/syntx/examples/benchmarks/generate_ants_3d_comparison_report.py): 3D registration report generator.
* **`pairs.csv`**: Standard dataset subject pair index mapping.
