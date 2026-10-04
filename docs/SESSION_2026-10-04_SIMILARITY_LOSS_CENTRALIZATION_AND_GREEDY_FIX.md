# Session 2026-10-04: Canonical Similarity Loss Centralization & Greedy Registration Conformance

## 1. Executive Summary

This session executed two major architectural advancements for `syntx`:
1. **Fixing the `syntx.greedy` Transform Composition Bug**: Eliminated the premature composition of initial affine matrices into greedy non-linear displacement fields. `syntx.greedy` now adheres strictly to standard ANTs / GEMINI.md pipeline invariants by returning `fwdtransforms = [1Warp.nii.gz, 0GenericAffine.mat]` and `invtransforms = [0GenericAffine.mat, 1InverseWarp.nii.gz]` with `whichtoinvert_inv = [True, False]` and direct `warpedfixout` generation.
2. **Centralizing All Image Similarity Losses (`syntx.core.losses`)**: Consolidated all similarity metric families into a canonical modular subpackage, eliminating duplicated string parsing, private loss calculation loops, and GEMINI.md Rule 3 violations across all 7 registration drivers (`syn.py`, `greedy.py`, `robust_affine.py`, `syngs.py`, `tvf.py`, `syn_jax.py`, `tvf_jax.py`).

All verification suites pass 100% across CPU and Apple Silicon MPS with zero test regressions.

---

## 2. Greedy Registration Bug Fix

### A. Root Cause & Mathematical Analysis
Previously, `syntx.greedy` was the lone outlier across registration drivers baking the initial affine transform $A$ directly into its displacement field:
$$\Phi_{\text{composed}} = A \circ (\text{Id} + u) - \text{Id}_{\text{phys}}$$
When downstream pipelines or benchmark runners passed standard ANTs forward transform lists `transformlist = [warp, affine]`, ANTs re-evaluated the affine transform a second time, catastrophically degrading alignment quality (Pearson correlation dropped from **0.9911** down to **0.5796**).

### B. Canonical Resolution
1. **Pure Physical Displacement**: In [`src/syntx/greedy.py`](../src/syntx/greedy.py), the optimized fixed-space normalized displacement field $u \in [-1, 1]$ is converted directly to fixed physical space (ITK LPS coordinates) using `P_F`:
   $$u_{\text{phys}} = P_F \cdot u_{\text{norm}}$$
   without composing the moving affine grid.
2. **Exported Transform Pair**:
   - **Forward**: Returns `fwdtransforms = [1Warp.nii.gz, 0GenericAffine.mat]` (or `[1Warp.nii.gz]` if `initial_transform=False`). Single-interpolation resampling in ANTs now evaluates distortion-free ($r > 0.991$).
   - **Inverse**: Anderson acceleration inverts the pure non-linear displacement field $\Phi^{-1}$. It returns `invtransforms = [0GenericAffine.mat, 1InverseWarp.nii.gz]` with `whichtoinvert_inv = [True, False]` satisfying $(A \circ \Phi)^{-1} = \Phi^{-1} \circ A^{-1}$ ($r > 0.975$ on moving image reconstruction).
3. **Execution Speed**: `return_inverse=False` remains the default to maintain unidirectional greedy speed, avoiding unnecessary fixed-point Anderson iterations when inverse fields are not requested.

---

## 3. Canonical Similarity Loss Centralization

### A. Modular Loss Core (`src/syntx/core/losses/`)
All similarity loss calculations are consolidated into dedicated functional modules:
- **`registry.py`**: Unified `SimilarityLossConfig` and factory `get_similarity_loss` accepting `(I, J, mask=None)`. Unified alias resolver supporting `'cc2'`, `'lncc'`, `'mattes'`, `'mattes_mi'`, `'mse'`, `'soft_dice'`, `'vgg19'`, `'resnet10'`, `'swinunetr'`. Configured bin suffixes (e.g. `'mattes_64'`) to override generic default parameters.
- **`cross_correlation.py`**: Canonical `local_ncc_loss_nd`, `BoxLNCCLoss`, `AnalyticalLNCC`, and `ANTsPseudoLNCC` enforcing a strict variance floor ($\ge 10^{-6}$), Cauchy-Schwarz bounds $[-1, 1]$, and `float64` autograd gradcheck compatibility.
- **`mutual_information.py`**: Canonical Mattes MI with boundary padding (`pad=2.0`), fixed bounds `(0.0, 1.0)`, and chunked Parzen joint histogram accumulation keeping 3D volume peak memory strictly **$< 15\text{ MB}$**.
- **`pointwise.py`**: Canonical `mse_loss_nd`, `l2_loss_nd`, and `mae_loss_nd` with uniform masked reductions and AMP `float16` scalar overflow safeguards (accumulating in `float32`).
- **`structural.py`**: Canonical `soft_dice_loss_nd` with half-precision overflow protection and distance transform losses.
- **`deep_features.py`**: Canonical `FeatureSpaceLoss` eliminating circular dependencies with `features.py`.

### B. Backward-Compatible Façade
[`src/syntx/core/losses.py`](../src/syntx/core/losses.py) re-exports all public loss functions, ensuring 100% backward compatibility for existing code and external callers.

---

## 4. Registration Drivers Refactoring

All registration drivers were migrated to consume similarity losses via `get_similarity_loss`:
- **`syn.py`**: Eliminated 6 triple-nested `try/except` fallback ladders; replaced inline metric construction; routed CoM initialization and deep feature fallbacks through `get_similarity_loss('lncc')` with full mask support.
- **`greedy.py`**: Centralized metric parsing with `parse_similarity_metric` and removed redundant `BoxLNCCLoss` allocations.
- **`robust_affine.py`**: Replaced private inline soft Dice with `soft_dice_loss_nd`; migrated all imports to `syntx.core.losses`.
- **`syngs.py` & `tvf.py`**: Eliminated dynamic union foreground masks in strict compliance with **GEMINI.md Rule 3** (whole fixed-domain evaluation).
- **`syn_jax.py` & JAX drivers**: Synchronized Mattes MI boundary padding (`pad=2.0`), achieving $< 10^{-7}$ numerical parity with PyTorch.

---

## 5. Verification Test Suite Matrix

| Test Suite | Pass Count | Status | Description |
| :--- | :---: | :---: | :--- |
| `tests/test_greedy.py` | **12 / 12** | **PASS** | Pure displacement field, `[warp, affine]`, Anderson inverse, RegAdam, Mattes MI |
| `tests/test_syn.py` | **13 / 13** (4 skip) | **PASS** | SyN standard runs, CoM init, deep feature fallback on degenerate grids |
| `tests/test_template.py` | **31 / 31** (8 skip) | **PASS** | Greedy template construction, adaptive affine caching, forward transform lists |
| `tests/test_robust_affine_general.py` | **8 / 8** | **PASS** | Canonical Mattes MI and soft Dice integration in robust affine |
| `tests/test_syngs_coverage.py` | **7 / 7** | **PASS** | Diffeomorphic velocity integration with centralized metric dispatcher |
| `tests/test_tvf.py` | **8 / 8** (1 skip) | **PASS** | Time-varying velocity fields with canonical loss factory |
| `tests/test_canonical_losses_gradcheck.py` | **26 / 26** | **PASS** | Float64 analytical and autograd gradchecks across all exact losses |
| `tests/test_canonical_losses_accuracy.py` | **34 / 34** | **PASS** | Cosine similarity $\ge 0.999999$ against numerical reference autograd |
| `tests/test_canonical_losses_memory_amp.py` | **16 / 16** | **PASS** | Peak transient memory $<15\text{ MB}$ on 3D volumes; AMP float16 stability |
| `tests/test_canonical_losses_parity.py` | **20 / 20** | **PASS** | Bitwise deterministic accumulation and PyTorch $\leftrightarrow$ JAX parity |
| `tests/test_canonical_losses_adversarial_stress.py` | **39 / 39** (1 skip) | **PASS** | Dynamic range $[0, 4000]$, zero-variance images, extreme aspect ratios |

---

## 6. Published Visual HTML Reports

All reports adhere strictly to GEMINI.md Rule 5 light theme (`#ffffff` background, `#1e293b` text, cyan/emerald accents):
- [`docs/reports/loss_centralization_final_report.html`](reports/loss_centralization_final_report.html)
- [`docs/reports/milestone2_interface_conformance_regression_report.html`](reports/milestone2_interface_conformance_regression_report.html)
- [`docs/reports/milestone2_jax_memory_challenge_report.html`](reports/milestone2_jax_memory_challenge_report.html)
- [`docs/reports/syn_deep_feature_fallback_remediation_report.html`](reports/syn_deep_feature_fallback_remediation_report.html)
- [`docs/reports/canonical_mattes_mi_report.html`](reports/canonical_mattes_mi_report.html)
