# Syntx Benchmarking Guide

> **Definitive Reference Document — Version 1.2**
>
> This document specifies the complete protocol for benchmarking `syntx` registration algorithms: datasets, parameter grids, gradient evaluation modes, evaluation metrics, provenance persistence, process isolation, and reporting standards.

---

## 1. Overview

Benchmarking `syntx` serves two purposes:

1. **Verification** — Confirm that every valid model × regularizer × gradient formulation combination produces expected accuracy and diffeomorphic regularity on canonical datasets.
2. **Optimization** — Identify peak-performing parameter configurations for production use and persist them in `docs/provenance/best_parameters.json`.

All benchmarks use `syntx.benchmark.data` for reproducible data access and `compute_bidirectional_dice()` for standardized symmetric Dice evaluation.

---

## 2. Benchmark Datasets

### 2.1 Built-in Benchmark Datasets (`syntx.benchmark_data`)

The `syntx.benchmark_data(key)` function provides canonical datasets with paired images and ground-truth segmentation labels. Each returns a dict with keys `fixed`, `moving`, `fixed_label`, `moving_label`, `fixed_labels`, `moving_labels`, and `description`.

| Key | Alias | Dim | Fixed | Moving | Labels | Purpose |
|:---|:---|:---:|:---|:---|:---|:---|
| `'r16_r64'` | `'2d'` | 2D | ANTs `r16` brain slice | ANTs `r64` brain slice | 3-class Otsu segmentation | Fast 2D verification (~5s per run) |
| `'c'` | `'c_halfc'` | 2D | C-shape phantom | Half-C phantom | Binary masks | Topological expansion stress test |
| `'ellipse'` | `'circle'` | 2D | Ellipse phantom | Circle phantom | Binary masks | Simple shape deformation test |
| `'mbhard'` | `'3d'` | 3D | NKI-TRT-20-2 | MMRR-21-2 | DKT31 manual cortical labels | Inter-cohort 3D stress test (Pair 41) |

#### 2D Label Evaluation (Otsu Classes)
For the `r16_r64` benchmark, labels are derived from 3-class Otsu thresholding:
- **Class 2 (Cortical Gray Matter):** `ants.threshold_image(img, "Otsu", 3).threshold_image(2, 2)`
- **Class 2+3 (Parenchymal Brain Tissue):** `ants.threshold_image(img, "Otsu", 3).threshold_image(2, 3)`

#### 3D Label Evaluation (Mindboggle DKT31)
The 3D Mindboggle datasets use manually annotated DKT31 cortical labels with 31 discrete regions. Dice is computed symmetrically in both Fixed and Moving spaces using nearest-neighbor interpolation.

---

### 2.2 Canonical 6-Pair Diagnostic Suite (`syntx.benchmark.data.load_mindboggle_pair`)

Before scaling to the full 90-pair population, algorithms must be evaluated on the **Canonical 6-Pair Diagnostic Suite**. This suite spans both intra-subject acquisitions and high-contrast anisotropic inter-subject cohort transfers:

| Pair Index | Dataset Type | Fixed Cohort | Moving Cohort | Challenges & Header Properties |
|:---:|:---|:---|:---|:---|
| **Pair 57** | Inter-Subject | OASIS-TRT-20 | MMRR-21 | Severe FOV mismatch, anisotropic voxel grids ($160\times 256 \times 256$ vs $189\times 233\times 197$) |
| **Pair 00** | Intra-Subject | OASIS-TRT-20-1 | OASIS-TRT-20-2 | Baseline intra-subject longitudinal scan |
| **Pair 01** | Intra-Subject | NKI-RS-22-1 | NKI-RS-22-2 | Intra-scanner repeatability |
| **Pair 02** | Intra-Subject | NKI-TRT-20-1 | NKI-TRT-20-2 | High-resolution test-retest |
| **Pair 41** | Inter-Subject (`mbhard`) | NKI-TRT-20-2 | MMRR-21-2 | Cross-scanner morphometry transfer |
| **Pair 45** | Inter-Subject | MMRR-21-1 | NKI-RS-22-1 | Contrast inversion & morphological variance |

---

### 2.3 90-Pair Mindboggle Population Benchmark

The definitive large-scale benchmark evaluates registration across **90 image pairs** drawn from 5 Mindboggle sub-cohorts. The pair list is stored in `examples/pairs.csv` (91 lines: 1 header + 90 data rows).

| Cohort | Subjects | Origin | Spacing |
|:---|:---:|:---|:---|
| OASIS-TRT-20 | 20 | `[-80, 128, -128]` | `[1.0, 1.0, 1.0]` |
| MMRR-21 | 21 | `[202.8, 0, 0]` | `[1.2, 1.0, 1.0]` |
| NKI-RS-22 | 22 | varies | `[1.0, 1.0, 1.0]` |
| NKI-TRT-20 | 20 | `[0, 0, 0]` | `[1.0, 1.0, 1.0]` |
| Extra-18 | 18 | varies | varies |

---

## 3. Models Under Evaluation

| Model | Function | Description | Peak Formulation |
|:---|:---|:---|:---|
| **SyN** | `syntx.syn()` | Symmetric diffeomorphic normalization via greedy Eulerian velocity steps with antisymmetric geodesic midpoint anchoring | `formulation='eulerian'`, `use_analytical_gradients=False`, `kernel_type='gaussian'`, `inverse_method='anderson'` |
| **TVF** | `syntx.tvf()` | Time-Varying Velocity Field registration via ODE trajectory integration with LARS/CFL optimizer | `flow_sigma=0.0`, `total_sigma=0.2`, `solver='euler'`, `regularizer='gaussian'` |
| **SyNGS** | `syntx.syngs()` | EPDiff Geodesic Shooting via initial momentum optimization | Geodesic shooting trajectory optimization |

---

## 4. Parameter Sweeps & Gradient Formulation Taxonomy

### 4.1 Gradient Evaluation Modes

`syntx.syn` supports three distinct gradient backpropagation modes:

1. **Pure Autograd (`use_analytical_gradients=False`, Primary Standard)**:
   - Evaluates full PyTorch autograd graph backpropagation through sliding-window box filters.
   - **Vector-Channel Physical Scaling**: Converts normalized grid gradients $\frac{\partial \mathcal{L}}{\partial \phi}$ to physical displacement units ($1/\text{mm}$) using dimension-flipped spatial scales:
     $$\mathbf{s}_{\text{phys}} = \text{flip}\left(\frac{(\mathbf{N} - 1) \odot \mathbf{s}}{2}, \text{dim}=0\right)$$
   - Verified Peak: **$0.6476$ Mean Symmetric Dice** across Mindboggle ($+2.40\%$ gain over ANTs C++ with $0.0005\%$ folding).

2. **Analytical Variational Pseudo-Gradient (`use_analytical_gradients=True`)**:
   - ITK-style center-of-window approximation ($CC^2 = \frac{s_{FM}^2}{s_{FF} \cdot s_{MM}}$) with decoupled image spatial gradients $\nabla I_{\text{mid}}$.
   - Produces sharp sulcal boundary alignment but requires heavier Gaussian regularization to prevent folding.

3. **Dual-Gradient Averaging (`dual_gradient=True`, `dual_gradient_weight=0.5`)**:
   - Computes the convex combination of analytical pseudo-gradients and end-to-end autograd gradients:
     $$\mathbf{g}_{\text{dual}} = (1 - w) \mathbf{g}_{\text{analytic}} + w \mathbf{g}_{\text{autograd}}$$
   - Acts as a strong low-frequency regularizer, suppressing grid folding to near $0.000\%$.

---

### 4.2 Regularization Kernels

1. **ITK Truncated Gaussian Kernel (`kernel_type='gaussian'`, Verified Standard)**:
   - Sampled Gaussian operator truncated at radius $\lfloor 3\sigma + 0.5 \rfloor$.
   - Yields exact mathematical parity with ITK/ANTs C++ Gaussian smoothing.
2. **Spectral Sobolev Regularizer (`regularizer='sobolev'`)**:
   - Frequency-domain spectral damping $(1 + \alpha \|\mathbf{k}\|^2)^{-s}$.
3. **Discrete Sine Transform Regularizer (`regularizer='dsti'`)**:
   - Imposes zero-Dirichlet boundary conditions on displacement fields.

---

### 4.3 SyN Standard Configurations

```python
syntx.syn(
    fixed=fi, moving=mi, initial_transform=aff_tx,
    backend='pytorch', device='mps',
    reg_iterations=[100, 100, 20],
    similarity_metric='cc2',
    syn_sampling=2,
    inverse_method='anderson',
    formulation='eulerian',
    kernel_type='gaussian',
    flow_sigma=3.0,
    total_sigma=0.0,
    grad_step=0.25,
    use_analytical_gradients=False,  # Peak Autograd Standard
    antisymmetric=True,
    verbose=False
)
```

---

## 5. Evaluation Metrics

Every benchmark run must report the **complete standard quantitative deformation metrics suite**:

| Metric | Symbol | Description | Target Standard |
|:---|:---:|:---|:---:|
| **Fixed Space Dice** | $\text{Dice}_{\text{fixed}}$ | Warp moving labels $\to$ fixed space, compare with fixed labels | $> 0.60$ (DKT31) |
| **Moving Space Dice** | $\text{Dice}_{\text{moving}}$ | Warp fixed labels $\to$ moving space, compare with moving labels | $> 0.60$ (DKT31) |
| **Symmetric Mean Dice** | $\text{Dice}_{\text{sym}}$ | $0.5 \cdot (\text{Dice}_{\text{fixed}} + \text{Dice}_{\text{moving}})$ | **Peak: $0.6476$** |
| **Grid Folding %** | $\text{Fold}\%$ | Percentage of voxels where $\det(J) \le 0$ | **$< 0.005\%$** |
| **Min Jacobian** | $\min \det(J)$ | Minimum Jacobian determinant | $> 0.0$ |
| **Mean Inverse Error** | $e_{\text{mean}}$ | Mean $\|\phi_{\text{inv}}(x + \phi_{\text{fwd}}(x)) + \phi_{\text{fwd}}(x)\|_2$ (mm) | **$< 0.030\text{ mm}$** |
| **95th %ile Inverse Error**| $e_{p95}$ | 95th percentile of inverse identity error (mm) | $< 0.080\text{ mm}$ |
| **Runtime** | $t_{\text{sec}}$ | Total execution time in seconds | **$< 50\text{ s}$ on GPU** |

---

## 6. Execution Protocol & Process Isolation

### 6.1 Subprocess Isolation Mandate (GPU / MPS)

> [!IMPORTANT]
> When executing benchmark suites on Apple Silicon GPU (`mps`) or CUDA, each registration evaluation **MUST execute in a dedicated, isolated Python subprocess** (`subprocess.run([sys.executable, "-u", "scripts/run_single_pair_eval.py", str(pair_idx)])`).
>
> Running multiple large 3D registrations sequentially inside a single Python process causes PyTorch graph retention, GPU memory leaks, and MPS allocator fragmentation.

---

### 6.2 Benchmarking Workflow Phases

1. **Phase 1: Fast 2D Verification** (`r16_r64`):
   - Confirms optimization sanity and gradient backpropagation in $< 5\text{ s}$.
2. **Phase 2: Canonical 6-Pair Diagnostic Suite** (Pairs 57, 00, 01, 02, 41, 45):
   - Evaluates performance across intra-subject and anisotropic cross-subject acquisitions.
   - Requires isolated subprocess worker execution.
3. **Phase 3: 90-Pair Mindboggle Validation**:
   - Executes full population benchmark across all 90 pairs in `examples/pairs.csv`.
   - Records population mean, standard deviation, and diffeomorphism statistics.

---

## 7. Provenance Persistence (`best_parameters.json`)

Whenever parameter sweeps discover new peak performance records, the full configuration and benchmark metrics must be immediately persisted to [`docs/provenance/best_parameters.json`](file:///Users/stnava/code/syntx/docs/provenance/best_parameters.json).

### Schema Reference

```json
{
  "syntx.syn": {
    "3D_peak_autograd": {
      "formulation": "eulerian",
      "use_analytical_gradients": false,
      "use_ants_pseudo_gradient": false,
      "kernel_type": "gaussian",
      "grad_step": 0.25,
      "flow_sigma": 3.0,
      "total_sigma": 0.0,
      "fast_smooth": false,
      "inverse_method": "anderson",
      "in_loop_inv_steps": 10,
      "syn_metric": "cc2",
      "reg_iterations": [100, 100, 20],
      "mean_symmetric_dice_6pairs": 0.6476,
      "mean_folding_pct": 0.0005,
      "mean_speedup_vs_ants": 4.45
    }
  }
}
```

---

## 8. Benchmark Script Architecture

### 8.1 Single-Pair Worker Script (`scripts/run_single_pair_eval.py`)

```python
import os, sys, time, json, ants, torch, syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

pair_idx = int(sys.argv[1])
p = load_mindboggle_pair(pair_idx)
fi, mi, fl, ml = p['fixed'], p['moving'], p['fixed_label'], p['moving_label']

# Initial Affine Alignment
res_aff = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine', verbose=False)
aff_0 = res_aff['fwdtransforms'][0]

# Syntx SyN with Autograd + ITK Gaussian Kernel
res_syn = syntx.syn(
    fixed=fi, moving=mi, initial_transform=aff_0,
    backend='pytorch', device='mps',
    grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
    reg_iterations=[100, 100, 20], similarity_metric='cc2',
    use_analytical_gradients=False,
    kernel_type='gaussian',
    inverse_method='anderson', formulation='eulerian',
    antisymmetric=True, verbose=False
)

df_f, df_m, sym = compute_bidirectional_dice(fl, ml, fi, mi, res_syn['fwdtransforms'], res_syn['invtransforms'], res_syn['whichtoinvert_inv'])
fwd_w = next(x for x in res_syn['fwdtransforms'] if isinstance(x, str) and x.endswith('.nii.gz'))
jac = compute_jacobian_metrics(fi, fwd_w)

print(f"CASE_COMPLETE: Pair {pair_idx:02d} | Sym: {sym:.4f} | Fold: {jac['folding_pct']:.4f}% | MinJac: {jac['min']:.4f}", flush=True)
```

---

### 8.2 Master Subprocess Runner (`scripts/run_isolated_suite.py`)

```python
import sys, subprocess

pairs = [57, 0, 1, 2, 41, 45]
for pair_idx in pairs:
    cmd = [sys.executable, "-u", "scripts/run_single_pair_eval.py", str(pair_idx)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, flush=True)
```

---

## 9. Non-Negotiable Invariants

1. **Single Interpolation Policy** — Never pre-warp images before optimization. Always compose transforms in a single `ants.apply_transforms` call.
2. **Nearest Neighbor for Labels** — Strictly use `interpolator='nearestNeighbor'` when warping discrete label segmentations.
3. **Subprocess Isolation on GPU** — Serial benchmark runs must execute each evaluation in an isolated Python subprocess to prevent MPS/CUDA allocator fragmentation.
4. **Autograd Physical Scaling Channel Order** — Autograd scaling vectors MUST be flipped along spatial dimensions (`torch.flip(scale, dims=[0])`) to match `(dx, dy, dz)` vector channels.
5. **Bidirectional Symmetric Dice** — Always evaluate Dice in both fixed and moving space ($\text{Dice}_{\text{sym}} = 0.5 \cdot (\text{Dice}_{\text{fixed}} + \text{Dice}_{\text{moving}})$).
6. **Variance Floor** — All LNCC implementations must enforce $\text{Var}_{\text{safe}} = \max(\text{Var}, 10^{-6})$.

---

## 10. Document History

| Date | Version | Change |
|:---|:---:|:---|
| 2026-08-07 | 1.0 | Initial definitive benchmarking guide |
| 2026-08-12 | 1.1 | Updated TVF peak invariants (`flow_sigma=0.0`, `total_sigma=0.2`, `solver='euler'`) |
| 2026-08-16 | 1.2 | Added Autograd Gaussian peak standard, dual-gradient taxonomy, canonical 6-pair diagnostic suite, and subprocess isolation mandate |
