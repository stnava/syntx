# Comprehensive Mindboggle 90-Pair Diffeomorphic Registration Benchmark Plan

> **Definitive Plan & Protocol**
>
> This plan specifies the end-to-end execution of the 90-pair Mindboggle population benchmark, comparing **ANTs C++ SyN**, **PyTorch `syntx.syn`**, and **PyTorch `syntx.tvf`** from scratch under strictly fair and legitimate conditions.

---

## 1. Executive Summary & Goals

The objective of this benchmark is to provide a rigorous, uncompromised, and reproducible evaluation of non-linear diffeomorphic brain image registration across the complete 90-pair Mindboggle population dataset (`examples/pairs.csv`).

### Target Models
1. **`syn c++`**: ANTs C++ SyN via ANTsPy (`ants.registration(type_of_transform='SyN')`).
2. **`syntx.syn` PyTorch**: Diffeomorphic greedy SyN (`syntx.syn(backend='pytorch')`).
3. **`syntx.tvf` PyTorch**: Diffeomorphic Time-Varying Velocity Field (`syntx.tvf(backend='pytorch')`).

### Fairness & Legitimacy Guarantee
- **Identical Affine Initialization**: Every pair computes a shared initial affine transform via `syntx.robust_affine(fixed, moving, multi_start=True, mode='pytorch')`. All three algorithms (`syn c++`, `syntx.syn`, `syntx.tvf`) receive this exact same transform as their starting point (`initial_transform`).
- **No Intermediate Pre-Warping**: In accordance with the Single Interpolation Policy, optimization occurs on native-space images; transform lists are composed in a single evaluation step.
- **Nearest-Neighbor Segmentation Warping**: All discrete DKT31 label maps are warped using `interpolator='nearestNeighbor'`.
- **Exact Physical Metrics**: Jacobian determinants are evaluated without log-compression (`do_log=False`), and inverse identity errors are computed in physical millimeters.

---

## 2. Hardware & Device Partitioning

To benchmark compute hardware scalability across Apple Silicon and CPU execution:

- **Pairs 0 to 44 (45 pairs)**:
  - `syn c++`: CPU (native ITK threads)
  - `syntx.syn` PyTorch: **MPS** (`device='mps'`)
  - `syntx.tvf` PyTorch: **MPS** (`device='mps'`)

- **Pairs 45 to 89 (45 pairs)**:
  - `syn c++`: CPU (native ITK threads)
  - `syntx.syn` PyTorch: **CPU** (`device='cpu'`)
  - `syntx.tvf` PyTorch: **CPU** (`device='cpu'`)

---

## 3. Parameter Specifications

### 3.1 ANTs C++ SyN (`syn c++`)
```python
ants.registration(
    fixed=fi,
    moving=mi,
    initial_transform=aff_tx,
    type_of_transform='SyN',
    grad_step=0.25,
    reg_iterations=[100, 100, 20],
    syn_metric='cc',
    syn_sampling=2,
    verbose=False
)
```

### 3.2 PyTorch SyN (`syntx.syn`)
```python
syntx.syn(
    fixed=fi,
    moving=mi,
    initial_transform=aff_tx,
    backend='pytorch',
    device=device,
    grad_step=0.25,
    flow_sigma=3.0,
    total_sigma=0.0,
    reg_iterations=[100, 100, 20],
    similarity_metric='lncc',
    syn_sampling=2,
    fast_smooth=True,
    inverse_method='anderson',
    antisymmetric=True,
    verbose=False
)
```

### 3.3 PyTorch TVF (`syntx.tvf`)
```python
syntx.tvf(
    fixed=fi,
    moving=mi,
    initial_transform=aff_tx,
    backend='pytorch',
    device=device,
    flow_sigma=0.0,
    total_sigma=0.2,
    grad_step=0.211,
    reg_iterations=[80, 80, 20],
    similarity_metric='lncc',
    syn_sampling=2,
    solver='euler',
    cfl_max=0.0,
    n_time_steps=3,
    constant_speed=True,
    antisymmetric=True,
    fast_smooth=True,
    verbose=False
)
```

---

## 4. Metric Suite

For each pair $i \in [0, 89]$ and algorithm $m \in \{\text{syn\_cpp}, \text{syntx\_syn}, \text{syntx\_tvf}\}$, the benchmark computes and records:

| Metric | Code Key | Description |
|--------|----------|-------------|
| **Fixed Space Dice** | `dice_fixed` | Warped moving labels to fixed space, mean DKT31 Dice |
| **Moving Space Dice** | `dice_moving` | Warped fixed labels to moving space, mean DKT31 Dice |
| **Symmetric Mean Dice** | `dice_sym` | $0.5 \times (\text{dice\_fixed} + \text{dice\_moving})$ |
| **Grid Folding %** | `fold_pct` | Voxel percentage in brain mask with $\det(J) \le 0$ |
| **Min Jacobian** | `min_detJ` | Minimum $\det(J)$ value in brain mask |
| **Mean Inverse Error** | `e_mean` | Mean physical inverse identity error (mm) |
| **95th %ile Inverse Error** | `e_p95` | 95th percentile physical inverse identity error (mm) |
| **Peak Inverse Error** | `e_max` | Maximum physical inverse identity error (mm) |
| **Runtime (s)** | `time_s` | Total wall-clock execution time (seconds) |

---

## 5. Persistence & Deliverables

1. **Raw Benchmark Results**: Saved incrementally to `docs/provenance/mindboggle_90pair_fair_results.json`.
2. **Provenance Update**: Top configuration summary updated in `docs/provenance/best_parameters.json`.
3. **Comprehensive Report**: Final markdown synthesis including cohort-by-cohort statistics, MPS vs CPU speedup, Dice distributions, and topology regularity analysis.
