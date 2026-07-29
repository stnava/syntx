# Analysis Report: SyN PyTorch vs JAX Implementation Parity & Evaluation Strategy

**Agent**: teamwork_preview_explorer  
**Working Directory**: `/Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1_1`  
**Date**: 2026-07-26  

---

## Executive Summary

An in-depth read-only investigation of the `syntx` codebase (`src/syntx/syn.py`, `src/syntx/syn_jax.py`, `src/syntx/tvf.py`, `tests/test_syn.py`, `tests/test_syn_jax.py`, `scratch/compare_torch_jax_3d.py`, `scratch/debug_parity_3d.py`, and `run_mindboggle_experiment.py`) was performed to evaluate PyTorch (`syntx.syn`) and JAX (`syntx.syn_jax`) SyN implementation parity and formulate the Evaluation Strategy (R1) and Algorithmic Parity Design Goals (R2).

Both backends adhere strictly to GEMINI.md guardrail rules—including the Single Interpolation Policy, variance-floored LNCC, Cauchy-Schwarz clamping, physical coordinate domain mapping, fixed-point inverse field step bounding, and antisymmetric velocity projection. Default configurations across both PyTorch and JAX models standardize on Anderson Acceleration (`inverse_method='anderson'`, `inverse_steps=30`).

---

## 1. Default Parameter & GEMINI.md Rule Compliance Analysis

### 1.1 Default Inverse Solver Parameters
* **PyTorch (`syntx.syn.SyNTo` and `syntx.syn.registration`)**:
  - `inverse_method`: Defaulted to `'anderson'` (Line 1747, 1760, 3431 in `syn.py`).
  - `inverse_steps`: Defaulted to `30` (Line 1747, 1761, 3430 in `syn.py`).
* **JAX (`syntx.syn_jax.SyNJAX` and `syntx.syn_jax.registration`)**:
  - `inverse_method`: Defaulted to `'anderson'` (Line 1954, 1967 in `syn_jax.py`).
  - `inverse_steps`: Defaulted to `30` (Line 1954, 1968 in `syn_jax.py`).
* **Observation / Audit Note**: Lower-level utility functions `update_inverse_field_nd` (PyTorch line 1311) and `update_inverse_field_nd_jax` (JAX line 998) default to `steps=10`, but high-level registration models (`SyNTo`, `SyNJAX`, `registration`) override this to `inverse_steps=30`. In `run_mindboggle_experiment.py`, `inverse_steps=10` is passed explicitly for benchmark speed.

### 1.2 LNCC Loss Implementation & Guardrail Compliance (Rule 2)
* **PyTorch LNCC (`local_ncc_loss_nd` in `syn.py:1467-1515`)**:
  - **Variance Floor**: Enforces $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$ via `safe_I_var = torch.clamp(I_var, min=1e-6)` (Lines 1504–1505).
  - **Cauchy-Schwarz Clamping**: Enforces $cc \in [-1.0, 1.0]$ via `cc = torch.clamp(cc_raw, min=-1.0, max=1.0)` (Line 1509) to prevent float32 roundoff singularity.
* **JAX LNCC (`local_ncc_loss_nd_jax` in `syn_jax.py:1191-1215`)**:
  - **Variance Floor**: Enforces $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$ via `safe_I_var = jnp.maximum(I_var, 1e-6)` (Lines 1202–1203).
  - **Cauchy-Schwarz Clamping**: Enforces $cc \in [-1.0, 1.0]$ via `cc = jnp.clip(cc_raw, -1.0, 1.0)` (Line 1207).

### 1.3 Single Interpolation Policy Adherence (Rule 1)
* **Pre-warping Prohibition**: Neither backend generates or saves intermediate pre-aligned image files (e.g., calling `ants.apply_transforms` on moving images prior to non-linear optimization).
* **Single-Step Composition**: In both backends (`prepare_mid_images_and_gradients_torch` and `prepare_mid_images_and_gradients_jax`), initial affine matrices ($M_{\text{phys}}, t_{\text{phys}}$) and SyN deformation fields ($\phi_{l2r}, \phi_{r2l}$) are composed directly in coordinate space before a single `grid_sample` call samples native-space image arrays.

### 1.4 Physical Coordinate Domain Mapping & ITK Parity Rules (Rules 6, 8, 9, 10, 11)
* **Rule 6 (Lie Algebra SO(d) Rotation & ANTs Center of Rotation)**:
  - Both `get_rotation_matrix` (PyTorch) and `get_rotation_matrix_jax` (JAX) use a 1st-order Taylor expansion `R_small = I + K_raw` for infinitesimally small angles (`theta2 < 1e-16`) to guarantee non-zero continuous gradient flow at identity initialization.
  - Coordinate translation functions (`grid_to_physical_affine_torch` and `grid_to_physical_affine_jax`) map physical space differences and ANTs fixed parameters (center of rotation $C$) via $t_{\text{new}} = t + C - A \cdot C$.
  - Affine initialization dynamically evaluates FOV vs Foreground Center of Mass (CoM) translations via downsampled MI/LNCC (`eval_translation_jax`).
* **Rule 8 (Inverse Displacement Field Evaluation)**:
  - Fixed-point convergence condition uses `logical_or(max > max_threshold, mean > mean_threshold)` matching ITK `InvertDisplacementFieldImageFilter`.
  - In-loop inverse updates are bounded (`in_loop_inv_steps`).
  - Displacement fields use `padding_mode='border'` during fixed-point inversion and composition, whereas intensity images use `padding_mode='zeros'`.
* **Rule 9 (Backend Parity)**:
  - Algorithmic steps (parameter updates, clamping, inversion bounds, warp composition) are synchronized symmetrically across PyTorch and JAX.
* **Rule 10 (Gaussian Smoothing Space & Units)**:
  - `flow_sigma` and `total_sigma` inputs represent variance ($\sigma^2$) in ITK/ANTs convention and are converted via $\sigma = \sqrt{\text{variance}}$ (Line 3446 in `syn.py`).
  - `separable_gaussian_filter` and `separable_gaussian_filter_jax` perform isotropic filtering in voxel index space across downsampled multi-resolution levels.
* **Rule 11 (Midpoint Warp Field Preservation & Antisymmetric Projection)**:
  - Half-warps (`w_l2r`, `w_r2l`) defining the geodesic midpoint are preserved as separate attributes (`midpoint_warp_l2r`, `midpoint_warp_r2l`) before full geodesic composition.
  - Antisymmetric velocity projection $e_0 = \delta_l + \delta_r$, $\delta_l \leftarrow \delta_l - 0.5 e_0$, $\delta_r \leftarrow \delta_r - 0.5 e_0$ anchors the midpoint at the Fréchet mean with zero drift and zero hyperparameters.

---

## 2. Step-by-Step Code Comparison: PyTorch vs JAX SyN

| Pipeline Stage | PyTorch Implementation (`syn.py`) | JAX Implementation (`syn_jax.py`) | Parity Status |
|---|---|---|---|
| **Data Types & Dispatch** | `torch.Tensor`, autograd graph, PyTorch device placement (`cpu`, `mps`, `cuda`). | `jnp.ndarray`, functional transforms (`jax.jit`, `jax.vjp`, `jax.lax.fori_loop`). | **Synchronized via DLPack** (`to_torch_tensor`, `to_jax_array_dl`). |
| **Rotation Matrix (Lie Algebra)** | `get_rotation_matrix` with PyTorch `torch.where` and matrix operations. | `get_rotation_matrix_jax` with `jnp.where` and JAX array indexing. | **Exact Parity** (Taylor expansion at $\mathbf{\omega}=0$). |
| **Affine Parameter Clamping** | Post-step parameter clamping in `HierarchicalAffine.clamp_parameters()`. | Post-step parameter clamping in `affine_step_jax`. | **Exact Parity** (`scale` $\in [0.05, 20]$, `shear` $\in [-5, 5]$, `omega` $\in [-\pi, \pi]$). |
| **Midpoint Image Sampling** | `prepare_mid_images_and_gradients_torch` using `grid_sample_nd`. | `prepare_mid_images_and_gradients_jax` using `jax_grid_sample`. | **Exact Parity** (Single interpolation policy). |
| **Velocity Field Update** | `syn_update_step_torch` or manual optimizer step. | `syn_update_step_jax` with JIT-compiled helper functions. | **Exact Parity** (Antisymmetric velocity projection $e_0 = \delta_l + \delta_r$). |
| **Inverse Field Solver** | `update_inverse_field_nd_anderson` using PyTorch tensor ops. | `update_inverse_field_nd_jax_anderson` & `update_inverse_field_jax_hybrid_lm`. | **Exact Parity** (Anderson Acceleration & Damped LM). |
| **Gaussian Smoothing** | `separable_gaussian_filter` (conv1d/conv3d). | `separable_gaussian_filter_jax` (`_conv1d_axis_edge`). | **Exact Parity** (Voxel index space isotropic smoothing). |

---

## 3. Mindboggle DKT 3D Benchmark Strategy

### 3.1 Subject Pair Corpus & Data Layout
* **Data Location**: `/Users/stnava/data/mindboggle/volumes`
* **Pair Configuration**: `examples/pairs.csv` (91 pairs: 40 intra-cohort pairs, 50 inter-cohort pairs across OASIS-TRT-20, NKI-RS-22, NKI-TRT-20, MMRR-21).
* **Per-Subject Files**:
  - T1 Brain Volume: `<cohort>_volumes/<subject>/t1weighted_brain.MNI152.nii.gz`
  - Cortical DKT Labels: `<cohort>_volumes/<subject>/labels.DKT31.manual.MNI152.nii.gz`

### 3.2 Evaluation Metrics Specification
1. **Cortical Mindboggle DKT Dice Overlap**: Target overlap calculated via `ants.label_overlap_measures` on nearest-neighbor warped label maps (`interpolator='nearestNeighbor'`). Target accuracy drop $\ge 0.01$ (1%) is flagged as a regression.
2. **Inverse Identity Error**: $\|\phi_{\text{inv}} \circ \phi_{\text{fwd}} - I\|$ in physical space (mm), reporting mean and max error via `calculate_inverse_identity_error`.
3. **Bending Energy ($E_{\text{2nd}}$)**: 2nd derivative smoothness $\sqrt{(\partial^2 u/\partial x^2)^2 + (\partial^2 u/\partial y^2)^2 + (\partial^2 u/\partial z^2)^2}$ computed by `compute_smoothness_metrics`.
4. **Jacobian Determinants**: Min, Max, Mean, Std, and Folding Rate (% negative $|J| \le 0$) computed via `compute_jacobian_and_folding`.
5. **Execution Runtime**: Wall-clock seconds per pair execution.

### 3.3 Reproducible Execution Protocol
`run_mindboggle_experiment.py` executes three benchmark arms across all pairs:
1. **ANTs C++/Py SyN Baseline**: `ants.registration(type_of_transform='SyN', syn_metric='cc', syn_sampling=2, grad_step=0.25)`.
2. **syntx.syn (PyTorch)**: Executed on MPS / CPU with `backend='pytorch'`, `syn_metric='lncc'`, `inverse_steps=10` (or `30` for full precision).
3. **syntx.syn (JAX)**: Executed on CPU / Metal XLA with `backend='jax'`, `syn_metric='lncc'`, `inverse_steps=10` (or `30` for full precision).

Results are written incrementally to `benchmark_results.json` to allow summary table generation and progress recovery.

---

## 4. Recommendations & Design Goals for Milestone Implementation

1. **R1 Evaluation Strategy**:
   - Maintain `run_mindboggle_experiment.py` as the benchmark script.
   - Verify that all summary reports include cortical Dice, inverse identity error (mean/max), 2nd derivative smoothness, Jacobian folding rate, and runtime.
2. **R2 Algorithmic Parity Goals**:
   - Ensure backend results match within floating-point tolerance ($\sim 0.001$ Dice gain/loss).
   - Enforce `inverse_method='anderson'` and `inverse_steps=30` as standard defaults in all top-level constructors.
   - Maintain strict symmetric velocity projection ($e_0 = \delta_l + \delta_r$) and single interpolation policy across PyTorch and JAX backends.
