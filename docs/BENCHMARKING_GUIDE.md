# Syntx Benchmarking Guide

> **Definitive Reference Document — Version 1.0**
>
> This document specifies the complete protocol for benchmarking `syntx` registration algorithms: datasets, parameter grids, evaluation metrics, provenance persistence, and reporting standards.

---

## 1. Overview

Benchmarking `syntx` serves two purposes:

1. **Verification** — Confirm that every valid model × regularizer × parameter combination produces expected accuracy and diffeomorphic regularity on canonical datasets.
2. **Optimization** — Identify peak-performing parameter configurations for production use and persist them in `docs/provenance/best_parameters.json`.

All benchmarks use `syntx.benchmark_data()` for reproducible data access and `ants.label_overlap_measures()` for standardized Dice evaluation.

---

## 2. Data

### 2.1 Built-in Benchmark Datasets (`syntx.benchmark_data`)

The `syntx.benchmark_data(key)` function provides four canonical datasets with paired images and ground-truth segmentation labels. Each returns a dict with keys `fixed`, `moving`, `fixed_label`, `moving_label`, `fixed_labels`, `moving_labels`, and `description`.

| Key | Alias | Dim | Fixed | Moving | Labels | Purpose |
|-----|-------|-----|-------|--------|--------|---------|
| `'r16_r64'` | `'2d'` | 2D | ANTs `r16` brain slice | ANTs `r64` brain slice | 3-class Otsu segmentation | Fast 2D verification (~5s per run) |
| `'c'` | `'c_halfc'` | 2D | C-shape phantom | Half-C phantom | Binary masks | Topological expansion stress test |
| `'ellipse'` | `'circle'` | 2D | Ellipse phantom | Circle phantom | Binary masks | Simple shape deformation test |
| `'mbhard'` | `'3d'` | 3D | NKI-TRT-20-2 | MMRR-21-2 | DKT31 manual cortical labels | Inter-cohort 3D stress test |

**Usage:**

```python
import syntx

# 2D brain benchmark
data_2d = syntx.benchmark_data('2d')
fi, mi = data_2d['fixed'], data_2d['moving']
fl, ml = data_2d['fixed_label'], data_2d['moving_label']

# 3D Mindboggle hard pair
data_3d = syntx.benchmark_data('3d')
fi, mi = data_3d['fixed'], data_3d['moving']
fl, ml = data_3d['fixed_label'], data_3d['moving_label']
```

#### 2D Label Evaluation (Otsu Classes)

For the `r16_r64` benchmark, labels are derived from 3-class Otsu thresholding:

- **Class 2 (Cortical Gray Matter):** `ants.threshold_image(img, "Otsu", 3).threshold_image(2, 2)`
- **Class 2+3 (Parenchymal Brain Tissue):** `ants.threshold_image(img, "Otsu", 3).threshold_image(2, 3)`

When computing Dice, filter out background labels:

```python
overlap = ants.label_overlap_measures(fl, ml_warped)
df = overlap[~overlap['Label'].astype(str).isin(['All', '0', '0.0'])]
col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
dice = float(df[col].mean())
```

#### 3D Label Evaluation (Mindboggle DKT31)

The `mbhard` dataset uses manually annotated DKT31 cortical labels with 31 discrete regions. Dice is computed identically but over many more label classes, yielding lower absolute scores.

### 2.2 90-Pair Mindboggle Benchmark

The definitive large-scale benchmark evaluates registration across **90 image pairs** drawn from 5 Mindboggle sub-cohorts. The pair list is stored in `examples/pairs.csv` (91 lines: 1 header + 90 data rows).

| Cohort | Subjects | Origin | Spacing |
|--------|----------|--------|---------|
| OASIS-TRT-20 | 20 | `[-80, 128, -128]` | `[1.0, 1.0, 1.0]` |
| MMRR-21 | 21 | `[202.8, 0, 0]` | `[1.2, 1.0, 1.0]` |
| NKI-RS-22 | 22 | varies | `[1.0, 1.0, 1.0]` |
| NKI-TRT-20 | 20 | `[0, 0, 0]` | `[1.0, 1.0, 1.0]` |
| Extra-18 | 18 | varies | varies |

**Pair Types:**
- **Intra-cohort** — Same scanner/protocol, minimal header mismatch.
- **Inter-cohort** — Different origins, spacings, and scan protocols. These are the hardest cases that stress-test physical coordinate handling.

**Data Location:** `/Users/stnava/data/mindboggle/volumes/`

The 90-pair benchmark is the **gold standard** for final validation. It should only be run with the top-performing parameter configurations identified by the 30-combination grid sweep.

---

## 3. Models Under Evaluation

### 3.1 Current Scope (PyTorch)

| Model | Function | Description |
|-------|----------|-------------|
| **SyN** | `syntx.syn()` | Symmetric diffeomorphic normalization via greedy CFL velocity steps with antisymmetric geodesic midpoint anchoring |
| **TVF** | `syntx.tvf()` | Time-Varying Velocity Field registration via ODE trajectory integration with LARS/CFL optimizer |

### 3.2 Future Scope (Not Yet Included)

> [!NOTE]
> The following models will be included in future benchmark sweeps once the PyTorch `syn` and `tvf` characterization is complete:
>
> - **SyNGS** (`syntx.syngs()`) — EPDiff Geodesic Shooting via initial momentum optimization
> - **SyN JAX** (`syntx.syn()` with `backend='jax'`) — JAX backend parity validation
> - **TVF JAX** (`syntx.tvf()` with `backend='jax'`) — JAX backend parity validation
>
> Backend parity requirement: JAX and PyTorch results must match within floating-point tolerance (~0.001 Dice). Any larger discrepancy indicates an implementation bug (see GEMINI.md §9).

---

## 4. The 30-Combination Parameter Grid

The verification grid sweeps **30 distinct parameter combinations** across 2 models, 3 regularizers, 2 `fast_smooth` settings, and model-specific parameter tuples.

### 4.1 Axes

| Axis | Values | Count |
|------|--------|-------|
| **Model** | `syn`, `tvf` | 2 |
| **Regularizer** | `gaussian`, `sobolev`, `dsti` | 3 |
| **`fast_smooth`** | `True`, `False` | 2 |
| **Parameter Tuples** | 3 for SyN, 2 for TVF | — |
| **SyN subtotal** | 3 × 2 × 3 | **18** |
| **TVF subtotal** | 3 × 2 × 2 | **12** |
| **Total** | 18 + 12 | **30** |

### 4.2 SyN Parameter Sets (18 combinations)

All SyN configurations use `total_sigma=0.0` (pure fluid deformation, no elastic field smoothing).

**3 parameter tuples × 3 regularizers × 2 `fast_smooth` = 18:**

| Tuple | `flow_sigma` | `grad_step` | `total_sigma` |
|-------|-------------|-------------|---------------|
| S1 | 1.0 | 0.25 | 0.0 |
| S2 | 3.0 | 0.25 | 0.0 |
| S3 | 3.0 | 0.50 | 0.0 |

Each tuple is run with every combination of:
- `regularizer` ∈ {`gaussian`, `sobolev`, `dsti`}
- `fast_smooth` ∈ {`True`, `False`}

**Fixed SyN parameters (all 18 combinations):**

```python
syntx.syn(
    fixed=fi, moving=mi, initial_transform=aff_tx,
    backend='pytorch',
    reg_iterations=[100, 40],
    affine_iterations=[50, 20],
    similarity_metric='lncc',
    syn_sampling=2,
    inverse_method='anderson',
    antisymmetric=True,
    total_sigma=0.0,
    # --- swept ---
    flow_sigma=<swept>,
    grad_step=<swept>,
    regularizer=<swept>,
    fast_smooth=<swept>,
)
```

### 4.3 TVF Parameter Sets

Because `shrink_ratio` properly bounds coarse resolutions, we sweep `grad_step` and `total_sigma` while enforcing `flow_sigma=0.0` and `regularizer='gaussian'`.

| Tuple | `grad_step` | `total_sigma` | `cfl_momentum` |
|-------|-------------|---------------|----------------|
| T1 | 0.25 | 0.0 | 0.9 |
| T2 | 0.30 | 0.0 | 0.9 |
| T3 | 0.35 | 0.0 | 0.9 |
| T4 | 0.30 | 0.1 | 0.9 |

Each tuple is run with:
- `regularizer` = `gaussian`
- `fast_smooth` = `False`

**Fixed TVF parameters (all 12 combinations):**

```python
syntx.tvf(
    fixed=fi, moving=mi, initial_transform=aff_tx,
    backend='pytorch',
    reg_iterations=[80, 80, 20],
    similarity_metric='lncc',
    syn_sampling=2,
    multipoint_loss=[0.0, 0.5, 1.0],
    solver='euler',
    cfl_max=0.0,
    cfl_momentum=0.90,
    n_time_steps=3,
    constant_speed=True,
    constant_speed_relaxation=0.10,
    use_analytical_gradients=False,
    antisymmetric=True,
    # --- swept ---
    flow_sigma=0.0,  # MUST be 0.0 (flow_sigma > 0 degrades Dice by 2.5-3.5%)
    total_sigma=<swept>,
    grad_step=<swept>,
    cfl_momentum=<swept>,
    regularizer='gaussian',
    fast_smooth=False,
)
```

### 4.4 Naming Convention

Each combination is identified by a canonical string:

```
{model}_{regularizer}_fast{True|False}_{param_tuple}
```

Examples: `syn_gaussian_fastTrue_S1`, `tvf_sobolev_fastFalse_T2`, `syn_dsti_fastTrue_S3`

---

## 5. Evaluation Metrics

Every benchmark run must report the **complete standard quantitative deformation metrics suite**.

### 5.1 Required Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| **Fixed Space Dice** | Dice_fixed | Warp moving labels → fixed space, compare with fixed labels |
| **Moving Space Dice** | Dice_moving | Warp fixed labels → moving space, compare with moving labels |
| **Symmetric Mean Dice** | Dice_sym | `0.5 × (Dice_fixed + Dice_moving)` |
| **Grid Folding %** | fold% | Percentage of voxels where det(J) ≤ 0 |
| **Min Jacobian** | min_detJ | Minimum Jacobian determinant (must be > 0 for diffeomorphism) |
| **Mean Inverse Error (mm)** | e_mean | Mean ‖φ_inv(x + φ_fwd(x)) + φ_fwd(x)‖₂ |
| **95th %ile Inverse Error (mm)** | e_p95 | 95th percentile of inverse identity error |
| **Peak Inverse Error (mm)** | e_max | Maximum inverse identity error |
| **Runtime (s)** | time_s | Total wall-clock execution time |

### 5.2 Bidirectional Dice Protocol

Dice **must** be evaluated symmetrically in both image spaces:

```python
# Fixed Space Dice
ml_warped = ants.apply_transforms(
    fixed=fi, moving=ml,
    transformlist=reg['fwdtransforms'],
    interpolator='nearestNeighbor'
)
dice_fixed = compute_mean_dice(fl, ml_warped)

# Moving Space Dice
fl_warped = ants.apply_transforms(
    fixed=mi, moving=fl,
    transformlist=reg['invtransforms'],
    whichtoinvert=reg.get('whichtoinvert_inv', [True, False]),
    interpolator='nearestNeighbor'
)
dice_moving = compute_mean_dice(ml, fl_warped)

# Symmetric Mean
dice_sym = 0.5 * (dice_fixed + dice_moving)
```

> [!IMPORTANT]
> Always use `interpolator='nearestNeighbor'` when warping discrete label maps. Linear or B-spline interpolation on integer segmentations produces meaningless fractional labels.

### 5.3 Grid Folding & Jacobian

```python
jac = ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0])
jac_np = jac.numpy()
mask = ants.get_mask(fi).numpy() > 0
fold_pct = float(np.mean(jac_np[mask] <= 0) * 100)
min_detJ = float(jac_np[mask].min())
```

---

## 6. Standard Registration Report Figures

Every benchmark evaluation should generate the **standard 5-figure visual suite** (when producing HTML reports):

| Figure | Content | Function |
|--------|---------|----------|
| **Fig 1** | Fixed & Moving input pair | `syntx.viz.render_input_pair_figure()` |
| **Fig 2** | 4-Panel: Mesh Grid, Seismic log-det(J), Inverse Error (mm), Canny Edge Overlap | `syntx.viz.render_standard_4panel()` |
| **Fig 3** | TVF keyframe velocity fields (quiver + heatmap) | `syntx.viz.plot_time_varying_velocity_grid()` |
| **Fig 4** | Multi-resolution loss convergence curves | Custom matplotlib |
| **Fig 5** | Epoch-by-epoch Dice progression | Custom matplotlib |

For grid sweep summaries (30-combination characterization), a condensed table format is sufficient. Full 5-figure reports are required for final peak-configuration validation.

---

## 7. Provenance Persistence (`best_parameters.json`)

### 7.1 Purpose

`docs/provenance/best_parameters.json` is the canonical record of peak-performing parameter configurations discovered through benchmarking. It must be updated whenever a new configuration achieves higher Dice or better regularity than the current record.

### 7.2 Schema

```json
{
  "syntx_version": "3.x.x",
  "last_updated": "ISO-8601 timestamp",
  "algorithms": {
    "syntx.syn": {
      "name": "Human-readable algorithm name",
      "provenance": {
        "algorithm": "syntx.syn",
        "backend": "pytorch",
        "device": "mps | cuda | cpu",
        "similarity_metric": "lncc",
        "regularizer": "gaussian | sobolev | dsti",
        "flow_sigma": 3.0,
        "total_sigma": 0.0,
        "grad_step": 0.25,
        "fast_smooth": true,
        "reg_iterations": [100, 40],
        "affine_iterations": [50, 20],
        "initial_transform": "syntx.robust_affine(mode='pytorch')",
        "inverse_method": "anderson",
        "antisymmetric": true
      },
      "performance_benchmarks": {
        "2d_r16_r64": {
          "dice_fixed": 0.0,
          "dice_moving": 0.0,
          "dice_sym": 0.0,
          "folding_percentage": 0.0,
          "min_jacobian": 0.0,
          "mean_inverse_error_mm": 0.0,
          "p95_inverse_error_mm": 0.0,
          "max_inverse_error_mm": 0.0,
          "runtime_seconds": 0.0
        },
        "3d_mbhard": {
          "...same fields..."
        },
        "90pair_mindboggle": {
          "mean_dice_sym": 0.0,
          "std_dice_sym": 0.0,
          "min_dice_sym": 0.0,
          "max_dice_sym": 0.0,
          "mean_folding_percentage": 0.0,
          "mean_runtime_seconds": 0.0,
          "n_pairs": 90
        }
      }
    },
    "syntx.tvf": {
      "...same structure..."
    },
    "syntx.robust_affine": {
      "...affine-specific fields..."
    }
  }
}
```

### 7.3 Benchmark Case Keys

Each benchmark case in `performance_benchmarks` must use a standardized key:

| Key | Dataset | Description |
|-----|---------|-------------|
| `2d_r16_r64` | `syntx.benchmark_data('2d')` | 2D brain slice Otsu Dice |
| `2d_c_halfc` | `syntx.benchmark_data('c')` | 2D C-shape phantom |
| `2d_ellipse` | `syntx.benchmark_data('ellipse')` | 2D ellipse phantom |
| `3d_mbhard` | `syntx.benchmark_data('3d')` | 3D Mindboggle hard pair |
| `90pair_mindboggle` | `examples/pairs.csv` | 90-pair population benchmark |

---

## 8. Execution Protocol

### Phase 1: 2D Grid Sweep

1. Load `syntx.benchmark_data('2d')`.
2. Compute `syntx.robust_affine(fixed, moving, multi_start=True, mode='pytorch')` once.
3. Run all 30 combinations using the shared affine transform.
4. Record all 9 metrics per combination.
5. Rank by `dice_sym` descending, breaking ties by `fold%` ascending.

### Phase 2: 3D Grid Sweep

1. Load `syntx.benchmark_data('3d')`.
2. Compute `syntx.robust_affine(fixed, moving, multi_start=True, mode='pytorch')` once.
3. Run all 30 combinations.
4. Record all 9 metrics.
5. Rank and select top 5 distinct parameter configurations per model.

### Phase 3: 90-Pair Mindboggle Validation

1. Select the 5 best parameter sets from Phase 2 (across both `syn` and `tvf`).
2. Run each parameter set across all 90 pairs from `examples/pairs.csv`.
3. Report population-level statistics: mean, std, min, max Dice_sym.
4. Update `docs/provenance/best_parameters.json` with the winning configuration.

### Accuracy Thresholds

> [!WARNING]
> A drop in Mean Dice ≥ 0.01 (1%) relative to the current best is considered a **massive, unacceptable regression**. Any parameter configuration showing such a drop must be flagged and investigated.

---

## 9. Affine Initialization

All benchmark runs must use `syntx.robust_affine()` for fair comparison:

```python
reg_aff = syntx.robust_affine(
    fixed=fi, moving=mi,
    multi_start=True,
    mode='pytorch',
    verbose=False
)
aff_tx = reg_aff['fwdtransforms'][0]
```

The same affine transform is shared across all 30 combinations within a given dataset to isolate deformable registration quality.

---

## 10. Benchmark Script Template

```python
import syntx, ants, time, json, numpy as np

data = syntx.benchmark_data('2d')  # or '3d'
fi, mi = data['fixed'], data['moving']
fl, ml = data['fixed_label'], data['moving_label']

reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch')
aff_tx = reg_aff['fwdtransforms'][0]

results = []

# SyN tuples: S1, S2, S3 (3 tuples)
# TVF tuples: T1, T2 (2 tuples)

for model in ['syn', 'tvf']:
    for regularizer in ['gaussian', 'sobolev', 'dsti']:
        for fast_smooth in [True, False]:
            param_tuples = get_param_tuples(model)  # 3 for syn, 2 for tvf
            for tuple_name, params in param_tuples.items():
                config_id = f"{model}_{regularizer}_fast{fast_smooth}_{tuple_name}"
                t0 = time.time()

                if model == 'syn':
                    reg = syntx.syn(
                        fixed=fi, moving=mi, initial_transform=aff_tx,
                        regularizer=regularizer, fast_smooth=fast_smooth,
                        total_sigma=0.0, **params
                    )
                else:
                    reg = syntx.tvf(
                        fixed=fi, moving=mi, initial_transform=aff_tx,
                        regularizer=regularizer, fast_smooth=fast_smooth,
                        cfl_max=0.0, **params
                    )

                elapsed = time.time() - t0

                # Evaluate all 9 metrics
                metrics = evaluate_registration(fi, mi, fl, ml, reg)
                metrics['config'] = config_id
                metrics['time_s'] = elapsed
                results.append(metrics)

# Persist
with open('docs/provenance/grid_30_characterization_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 11. Key Constraints & Invariants

These invariants are **non-negotiable** and must be respected in every benchmark run:

1. **Single Interpolation Policy** — Never pre-warp images before optimization. Compose transforms in a single `ants.apply_transforms` call.
2. **Nearest Neighbor for Labels** — Always `interpolator='nearestNeighbor'` for segmentation maps.
3. **total_sigma=0.0 for SyN** — All SyN models use pure fluid deformation without elastic field smoothing.
4. **cfl_max=0.0 for TVF** — Never cap TVF velocity norms during benchmarking.
5. **flow_sigma=0.0 for TVF** — TVF models MUST use zero fluid gradient smoothing (`flow_sigma=0.0`) and rely exclusively on `total_sigma` for elastic velocity field regularization.
6. **Variance Floor** — LNCC implementations must enforce `Var_safe = max(Var, 1e-6)`.
7. **Cauchy-Schwarz Clamping** — LNCC cross-correlation must be clamped to `[-1.0, 1.0]`.
8. **Bidirectional Dice** — Always evaluate in both fixed and moving space.

---

## 12. Document History

| Date | Version | Change |
|------|---------|--------|
| 2026-08-07 | 1.0 | Initial definitive benchmarking guide |
| 2026-08-12 | 1.1 | Updated TVF peak invariants (`flow_sigma=0.0`, `total_sigma=0.2`, `solver='euler'`) |
