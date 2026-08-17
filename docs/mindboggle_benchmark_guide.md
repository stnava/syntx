# Mindboggle-101 Registration Benchmark Guide

`syntx.benchmark` is the core evaluation framework in `syntx` for standardized, reproducible, and topology-preserving registration benchmarking against the 101 manually labeled Mindboggle brain volumes (Klein & Tourville, 2012).

---

## 1. Dataset Setup & Organization

The benchmark requires the T1-weighted brain MRI volumes (`t1weighted_brain.nii.gz`) and manual DKT31 cortical label maps (`labels.DKT31.manual.nii.gz`).

### Expected Directory Hierarchy
```text
$SYNTX_DATA_DIR/ (default: /Users/stnava/data/mindboggle/volumes)
├── OASIS-TRT-20_volumes/
│   ├── OASIS-TRT-20-1/
│   │   ├── t1weighted_brain.nii.gz
│   │   └── labels.DKT31.manual.nii.gz
│   └── ... (20 subjects)
├── NKI-RS-22_volumes/
│   ├── NKI-RS-22-1/
│   │   ├── t1weighted_brain.nii.gz
│   │   └── labels.DKT31.manual.nii.gz
│   └── ... (22 subjects)
├── NKI-TRT-20_volumes/
│   ├── NKI-TRT-20-1/
│   │   ├── t1weighted_brain.nii.gz
│   │   └── labels.DKT31.manual.nii.gz
│   └── ... (20 subjects)
└── MMRR-21_volumes/
    ├── MMRR-21-1/
    │   ├── t1weighted_brain.nii.gz
    │   └── labels.DKT31.manual.nii.gz
    └── ... (21 subjects)
```

### Environment Variable Configuration
Set the path to your dataset in your shell profile:
```bash
export SYNTX_DATA_DIR="/path/to/mindboggle/volumes"
```

To verify dataset completeness from the command line:
```bash
syntx-benchmark --check-data
# or
python -m syntx.benchmark --check-data
```

---

## 2. Python API Usage

### Single-Pair Evaluation
```python
import syntx
from syntx.benchmark import evaluate_mindboggle_pair

# Evaluate Pair 0 (Sobolev SyN) with automated HTML report generation
rec = evaluate_mindboggle_pair(
    pair_idx=0,
    model="sobolev",              # 'sobolev', 'gaussian', or 'tvf'
    generate_report=True,
    report_out_dir="docs/examples",
    verbose=True
)

print(f"Symmetric Cortical Dice: {rec['syntx_dice_sym']:.4f}")
print(f"Grid Folding Rate:       {rec['syntx_fold']:.4f}%")
print(f"ANTs Baseline Dice:       {rec['ants_baseline']['dice_sym']:.4f}")
print(f"Performance vs ANTs:     {rec['diff_vs_ants']:+.2f}%")
```

### Cohort / Multi-Pair Benchmarking
```python
from syntx.benchmark import run_mindboggle_benchmark

# Run the 6 probe pairs in isolated subprocesses
summary = run_mindboggle_benchmark(
    pairs=[0, 1, 2, 45, 67, 82],
    model="sobolev",
    probe_pairs={0, 1, 2, 45, 67, 82},   # Evaluates matched Gaussian SyN ablation
    out_dir="results/probe_eval",
    report_html="docs/probe_benchmark_report.html",
    generate_example_reports=True
)
```

---

## 3. Command-Line Interface (CLI)

`syntx` provides the `syntx-benchmark` command (and `python -m syntx.benchmark`):

### Verify Dataset
```bash
syntx-benchmark --check-data
```

### Run Single Pair Evaluation
```bash
syntx-benchmark --pair-idx 0 --model sobolev --generate-report
```

### Run Probe Subset (6 Pairs)
```bash
syntx-benchmark --pairs 0 1 2 45 67 82 --model sobolev --out-dir results/probe_eval --report-html docs/probe_report.html
```

### Run Full 90-Pair Randomized Cohort
```bash
syntx-benchmark --cohort --model sobolev --seed 42 --report-html docs/reproducible_90pair_report.html
```

---

## 4. Standardized Reporting & Visual Diagnostics

Every benchmark evaluation produces structured JSON records and connects directly to `syntx.viz`:
- **Single-Pair Reports**: Complete 5-figure visual suite (input pair orthoslices, deformed mesh grids, seismic Jacobian determinant maps, inverse identity error maps in mm, and Canny edge overlays).
- **Population Dashboards**: Interactive Plotly scatterplots comparing symmetric cortical Dice, per-cohort distributions (intra vs. inter), runtime speedups, and head-to-head Gaussian vs. Sobolev SyN ablation tables.

---

## 5. Provenance Hyperparameters

| Parameter | Sobolev SyN | Gaussian SyN | Time-Varying Velocity (TVF) |
| :--- | :--- | :--- | :--- |
| **Regularizer** | `sobolev` ($\alpha=1.5, k=5$) | `gaussian` (`flow_sigma=3.0`) | `gaussian` (`total_sigma=0.2`) |
| **Step Size** | `grad_step=0.25` | `grad_step=0.25` | `grad_step=0.211` |
| **Similarity Metric** | `cc2` (LNCC autograd) | `cc2` (LNCC autograd) | `cc2` (multipoint `[0.0, 0.5, 1.0]`) |
| **Pyramid Levels** | `[80, 80, 20]` | `[80, 80, 20]` | `[80, 80, 20]` |
| **Inverse Method** | `anderson` (10 steps) | `anderson` (10 steps) | `euler` (3 time steps) |
| **Folding Rate** | **`0.0000%`** (5/6 pairs) | `0.0002%` | `0.0100%` |
| **Mean Cortical Dice** | **`0.6378`** (+1.60% vs ANTs) | **`0.6399`** (+1.80% vs ANTs) | **`0.6476`** (+2.40% vs ANTs) |
