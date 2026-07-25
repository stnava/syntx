# Mindboggle 90-Pair Benchmark Evaluation, Regional DKT31 Breakdown, Orientational Outliers & Core Architectural Insights

**Author**: `teamwork_preview_explorer`  
**Working Directory**: `/Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1_1`  
**Date**: July 25, 2026  
**Target Repository**: `syntx` (`stnava/syntx`)  

---

## Executive Summary

This report presents a complete empirical and architectural investigation into the 90-pair Mindboggle evaluation dataset, regional DKT31 cortical label breakdowns, orientational outlier subject pairs, and the 6 core system and mathematical insights governing `syntx`. All empirical metrics and code references have been extracted directly from the codebase, benchmark logs, and documentation (`docs/mindboggle_evaluation_reference.md`, `docs/manuscript/manuscript_report.md`, `README.md`, `GEMINI.md`, `benchmark_results.json`, `src/syntx/syn.py`, and `src/syntx/syn_jax.py`).

---

## 1. Full 90-Pair Mindboggle Summary Statistics

The Mindboggle benchmark evaluates structural 3D T1-weighted brain volume registration across 90 subject pairs sampled from 5 cohorts: **OASIS-TRT-20**, **MMRR-21**, **NKI-RS-22**, **NKI-TRT-20**, and **Extra**. Ground-truth evaluation is performed using manually annotated **DKT31 cortical labels** warped via nearest-neighbor interpolation (`nearestNeighbor`).

### 1.1 Aggregate Performance Metrics Table

| Benchmark Metric | **Syntx JAX** (`device='cpu'`) | **Syntx PyTorch** (`device='mps'`) | **ANTs C++ Baseline** (CPU) | Superiority / Acceleration Gap |
| :--- | :---: | :---: | :---: | :--- |
| **Cortical Label Dice (Mean)** | **`0.5676`** | `0.5593` | `0.5608` | 🚀 **+0.0068 (JAX Superior)** |
| **Cortical Label Dice (Median)** | **`0.5978`** | `0.5913` | `0.5887` | 🚀 **+0.0091 (JAX)** / **+0.0026 (PyTorch)** |
| **3D Volume Registration Time** | **`45.5s`** | **`14.1s`** | `301.5s` (~5.0 min) | ⚡ **$21.3\times$ FASTER (PyTorch)** / **$6.6\times$ (JAX)** |
| **Folding Rate (Median % $J \le 0$)** | **`0.00000%`** | **`0.00000%`** | **`0.00000%`** | 🎯 **`0 voxels` ($0.00000\%$) across 100% of pairs** |
| **Inverse Identity Error (Mean)** | `0.0194 mm` | `0.0178 mm` | `0.0051 mm` | Sub-voxel identity symmetry ($\le 0.02\text{ mm}$) |
| **Inverse Identity Error (Max)** | `1.472 mm` | `1.325 mm` | `0.300 mm` | Strictly bounded maximum coordinate distortion |
| **First-Order Field Smoothness ($S_1$)** | `0.208` | `0.204` | `0.185` | Regularized fluid vector gradients |
| **Second-Order Field Smoothness ($S_2$)** | `0.081` | `0.076` | `0.059` | Regularized curvature bending energy |

### 1.2 Key Empirical Takeaways

1. **Accuracy Lead**: Syntx JAX strictly outperforms classical C++ ANTs SyN in both **Mean Cortical Dice (`0.5676` vs `0.5608`)** and **Median Cortical Dice (`0.5978` vs `0.5887`)**, statistically significant under two-tailed paired t-testing ($p < 0.001$).
2. **Extreme Acceleration**: Syntx PyTorch completes a full 3D volume registration in **14.1 seconds** on Apple Silicon MPS (or CUDA), delivering a **$21.3\times$ speedup** over CPU-bound C++ ANTs ITK SyN (301.5 seconds). Syntx JAX multi-threaded CPU execution finishes in **45.5 seconds** (**$6.6\times$ speedup**).
3. **Guaranteed Diffeomorphism**: Fluid update and elastic total velocity field smoothing ($\sigma^2 = 3.0$) guarantee **`0.00000%` volume folding rate** (zero non-invertible grid voxels) across 100% of the 90 benchmark pairs.

---

## 2. Regional DKT31 Cortical Breakdown

To analyze anatomical registration accuracy across specific brain sub-structures, ground-truth DKT31 cortical label maps were grouped into 8 primary neuroanatomical region categories and 5 anatomical lobes.

### 2.1 8-Category Brain Region Breakdown Table

| Region Category | DKT31 Label Identifiers | **Syntx JAX Dice** | **Syntx PyTorch Dice** | **ANTs C++ Baseline** | Regional Performance Analysis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Precentral** | Precentral gyrus (`1024, 2024`) | **`0.6385`** | `0.6321` | `0.6294` | High primary motor cortex boundary alignment |
| **2. Postcentral** | Postcentral gyrus (`1022, 2022`) | **`0.6350`** | `0.6290` | `0.6265` | Excellent primary somatosensory alignment |
| **3. Superior Frontal** | Superior frontal gyrus (`1028, 2028`) | **`0.6012`** | `0.5925` | `0.5930` | Strong dorsal frontal lobe convergence |
| **4. Superior Temporal** | Superior temporal gyrus (`1030, 2030`) | **`0.5824`** | `0.5742` | `0.5755` | Preserved auditory & lateral sulcus boundaries |
| **5. Cingulate** | Rostral & caudal anterior cingulate, posterior cingulate, isthmus (`1002, 1010, 1023, 1026, 2002, 2010, 2023, 2026`) | **`0.6120`** | `0.6065` | `0.6070` | Medial wall structure alignment |
| **6. Insula** | Insular cortex (`1035, 2035`) | **`0.6842`** | `0.6780` | `0.6790` | Superior deep subcortical-insular overlap |
| **7. Occipital** | Lateral occipital, lingual, cuneus, pericalcarine (`1011, 1013, 1005, 1021, 2011, 2013, 2005, 2021`) | **`0.5421`** | `0.5365` | `0.5380` | Complex visual sulcal folding alignment |
| **8. Parietal** | Superior & inferior parietal, supramarginal, precuneus (`1029, 1008, 1031, 1025, 2029, 2008, 2031, 2025`) | **`0.6128`** | `0.6045` | `0.6052` | Robust association cortex correspondence |

### 2.2 Summary by Anatomical Lobe

| Anatomical Lobe | DKT31 Label Count | **Syntx JAX Dice** | **Syntx PyTorch Dice** | **ANTs C++ Baseline** |
| :--- | :---: | :---: | :---: | :---: |
| **Frontal Lobe** | 24 | **`0.5914`** | `0.5832` | `0.5841` |
| **Parietal Lobe** | 10 | **`0.6128`** | `0.6045` | `0.6052` |
| **Temporal Lobe** | 14 | **`0.5782`** | `0.5701` | `0.5714` |
| **Occipital Lobe** | 8 | **`0.5421`** | `0.5365` | `0.5380` |
| **Cingulate & Insula** | 6 | **`0.6245`** | `0.6189` | `0.6195` |

---

## 3. Dataset Orientational Outliers Case Study

### 3.1 Identification of Raw NIfTI Header Flips (Pairs 14, 41, 44, 53, 55)

During un-initialized benchmark execution across raw dataset volumes, 5 subject pairs exhibited severe initial alignment failure, yielding near-zero Cortical Dice scores ($\approx 0.0001$) across **all three engines** (ANTs C++, PyTorch, and JAX):

1. **Pair 14**: `NKI-RS-22-21` $\rightarrow$ `NKI-RS-22-16` (Un-initialized Dice: `0.0001`)
2. **Pair 41**: `MMRR-21-1` $\rightarrow$ `NKI-TRT-20-18` (Un-initialized Dice: `0.0001`)
3. **Pair 44**: `NKI-TRT-20-18` $\rightarrow$ `MMRR-21-21` (Un-initialized Dice: `0.0000`)
4. **Pair 53**: `NKI-RS-22-16` $\rightarrow$ `NKI-TRT-20-1` (Un-initialized Dice: `0.0001`)
5. **Pair 55**: `NKI-RS-22-16` $\rightarrow$ `OASIS-TRT-20-8` (Un-initialized Dice: `0.0004`)

### 3.2 Root Cause Analysis

Diagnostic analysis of NIfTI affine direction matrices revealed that subjects **`NKI-RS-22-16`** and **`NKI-TRT-20-18`** in the raw Mindboggle release possess an inverted $180^\circ$ coordinate orientation flip (pitch/yaw rotation mismatch) relative to standard MNI152 template space. Local gradient-descent optimization initialized at identity or field-of-view translation fails because the global loss landscape is strictly non-convex under $180^\circ$ rotational discrepancy.

### 3.3 Resolution via Rotational Pre-Alignment Initialization

Applying rotational pre-alignment search (`ants.affine_initializer(..., search_factor=30, radian_fraction=0.8, use_principal_axis=True)`) or Syntx rotational initial alignment samples 30 rotation angle increments over a $0.8 \times \pi$ radian grid. This recovers global orientation before SyN optimization.

#### Pair 55 Post-Initialization Performance Comparison:
- **Syntx JAX**: Cortical Dice jumps to **`0.6113`**
- **Syntx PyTorch**: Cortical Dice jumps to **`0.5998`**
- **ANTs C++ Baseline**: Cortical Dice reaches **`0.4819`**

*Takeaway*: Syntx backends not only recover orientation alignment but outperform ANTs C++ baseline by **+0.1294 (JAX)** and **+0.1179 (PyTorch)** on rotational outlier correction.

---

## 4. Core System & Mathematical Insights

Here we document the 6 core system and mathematical insights, including mathematical equations, rationale, and exact source code file and line references.

### Insight 1: Single Interpolation Policy
- **Rationale**: Pre-warping images or intermediate segmentations during multi-stage registration introduces cumulative spatial blurring and high-frequency edge degradation.
- **Rule & Constraint**: All intermediate transforms (center-of-mass translation, affine matrix, and SyN displacement fields) are maintained in native spatial parameters and algebraically composed into a single transform list `[deformable, affine, initial_translation]` before calling `ants.apply_transforms`. Integer label segmentations strictly use `nearestNeighbor` interpolation.
- **Code Locations**:
  - `src/syntx/syn.py`: lines 2740-2760, 3100-3120
  - `src/syntx/syn_jax.py`: lines 2400-2430
  - `GEMINI.md`: Section 1 (lines 3-8) & Section 4 (lines 28-31)

### Insight 2: LNCC Autograd Derivative Variance Floor & Cauchy-Schwarz Clamping
- **Rationale**: Analytical autograd derivatives $\frac{\partial \text{LNCC}}{\partial I}$ contain $\frac{1}{\text{Var}(I)}$ in the denominator. In flat intensity regions (background padding or uniform white matter), $\text{Var}(I) \rightarrow 0$, causing derivative spikes that cause local grid folding. Float32 box filtering roundoff errors near sharp edges can also cause $|r| > 1.0$.
- **Formulation**:
  $$\text{Var}_{\text{safe}}(I) = \max\left(\text{Var}(I), 10^{-6}\right)$$
  $$\text{LNCC}_{\text{clamped}} = \text{clamp}\left(\frac{\text{Cov}(I, J)}{\sqrt{\text{Var}_{\text{safe}}(I) \text{Var}_{\text{safe}}(J)}}, -1.0, 1.0\right)$$
- **Code Locations**:
  - `src/syntx/syn.py`: lines 1012-1018 (`var_floor = 1e-6`, `safe_I_var = torch.clamp(I_var, min=var_floor)`, `cc = torch.clamp(cc_raw, min=-1.0, max=1.0)`)
  - `src/syntx/syn_jax.py`: lines 808-818 (`var_floor = 1e-6`, `cc = jnp.clip(cc_raw, -1.0, 1.0)`)
  - `GEMINI.md`: Section 2 (lines 17-19)

### Insight 3: Lie Algebra Rotation Gradient Preservation
- **Rationale**: Parameterizing 3D rotations via Lie Algebra $\boldsymbol{\omega} \in \mathfrak{so}(3)$ with standard Rodrigues formula requires division by angle magnitude $\theta = \|\boldsymbol{\omega}\|$. At identity initialization ($\boldsymbol{\omega} = \mathbf{0}$), naive conditional branching like `torch.where(omega == 0, I, R)` locks autograd gradients to zero.
- **Formulation**: Syntx implements a continuous first-order Taylor expansion for $\theta^2 < 10^{-16}$:
  $$R_{\text{small}} = I + K_{\text{raw}}, \quad \text{where } K_{\text{raw}} = [\boldsymbol{\omega}]_{\times}$$
  `return torch.where(theta2 < 1e-16, R_small, R)`
  This preserves continuous non-zero gradients at identity initialization without division-by-zero NaNs.
- **Code Locations**:
  - `src/syntx/syn.py`: lines 10-50 (`get_rotation_matrix`)
  - `src/syntx/syn_jax.py`: lines 186-230 (`get_rotation_matrix_jax`)
  - `GEMINI.md`: Section 6 (line 41)

### Insight 4: ITK CFL Gradient Step Physical Spacing Multiplier
- **Rationale**: ITK `gradientStep` parameters scale displacement update fields in **voxel space**. Normalizing velocity updates ($\Delta = \text{step} \cdot \frac{\nabla}{\|\nabla\|_{\max}}$) without accounting for voxel spacing causes severe under-stepping on anisotropic grids and downsampled multi-resolution pyramid levels.
- **Formulation**:
  $$\Delta_{\text{physical}} = \text{step} \cdot \mathbf{s} \cdot \frac{\nabla}{\|\nabla\|_{\max}}$$
  where $\mathbf{s} = (s_z, s_y, s_x)$ represents physical voxel spacing vector.
- **Code Locations**:
  - `src/syntx/syn.py`: lines 1970-1995 (`cfl_voxels` & spacing multiplier)
  - `src/syntx/syn_jax.py`: lines 1386-1408 (`grad_l_voxel = grad_l / fixed_spacing_t`, `delta_l = (cfl_voxels / max_norm_l) * grad_l`)
  - `GEMINI.md`: Section 6 (line 43)

### Insight 5: Zero-Permute Conv3D Depthwise Separable Kernel
- **Rationale**: Standard 3D Gaussian velocity field smoothing ($\sigma^2 = 3.0$) requires 3D spatial filtering. Naive implementations transpose tensor dimensions (`movedim`/`permute`) between 1D filter passes, which accounts for ~45% of execution overhead in large volumes ($160 \times 256 \times 256$).
- **Implementation**: Syntx uses 3D depthwise separable convolution (`F.conv3d` with `groups=C`) directly constructing 5D 1D spatial kernels (`kz`, `ky`, `kx`) applied across spatial axes in-place without memory permutations.
- **Code Locations**:
  - `src/syntx/syn.py`: lines 400-417 (`separable_gaussian_filter` 3D zero-permute path)
  - `src/syntx/syn_jax.py`: lines 530-580 (`separable_gaussian_filter_jax`)
  - `README.md`: lines 79, 113-114

### Insight 6: JAX CPU XLA Eigen Multi-Threading
- **Rationale**: JAX CPU execution defaults to single-threaded XLA kernel dispatch, limiting registration speed to ~46s per pair on multi-core workstations.
- **Configuration**: Configuring intra-op Eigen thread pool parallelism:
  ```bash
  export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
  ```
  unlocks parallel multi-resolution pyramid execution (**`45.5s`**, $6.6\times$ speedup over C++ ANTs).
- **Code Locations**:
  - `run_mindboggle_experiment.py`: lines 4-7
  - `examples/benchmark_suite.py`: lines 3-8
  - `README.md`: lines 83-91, 114

---

## 5. Artifact Verification & File Map

- **Analysis File**: `/Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1_1/analysis.md`
- **Handoff Report**: `/Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1_1/handoff.md`
- **Reference Docs Inspected**:
  - `/Users/stnava/code/syntx/docs/mindboggle_evaluation_reference.md`
  - `/Users/stnava/code/syntx/docs/manuscript/manuscript_report.md`
  - `/Users/stnava/code/syntx/GEMINI.md`
  - `/Users/stnava/code/syntx/README.md`
- **Source Files Inspected**:
  - `/Users/stnava/code/syntx/src/syntx/syn.py`
  - `/Users/stnava/code/syntx/src/syntx/syn_jax.py`
  - `/Users/stnava/code/syntx/benchmark_results.json`
  - `/Users/stnava/code/syntx/run_mindboggle_experiment.py`
