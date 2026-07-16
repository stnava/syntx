# Verification Review Report — Milestones 2, 3, and 4

## Review Summary

**Verdict**: REQUEST_CHANGES

This review evaluates the implementation of Milestones 2, 3, and 4 in the `syntx` package against correctness, robustness, and constraints specified in `GEMINI.md`. All 94 unit tests successfully pass with high coverage (~92% overall). The Single Interpolation Policy and VGG similarity metric guidelines are successfully implemented and verified. However, a **Critical Finding** is identified: the generated visual dashboard (`docs/parity_report.html`) lacks required spatial visualizations—specifically edge overlap maps, deformed coordinate grids, and Jacobian determinant maps. Ripgrep and Python analysis confirmed that these visual helpers are defined in the generator script but never called, and the HTML contains no references to Jacobian data, violating Rule 3 of the project guardrails.

---

## Findings

### [Critical] Finding 1: Lack of Required Report Visualizations in `docs/parity_report.html`

- **What**: The HTML dashboard `docs/parity_report.html` does not display spatial/structural maps for edge/region overlap, deformed grids, or Jacobian determinants.
- **Where**: `docs/parity_report.html` and `examples/generate_ants_2d_comparison_report.py`.
- **Why**: `GEMINI.md` Rule 3 ("Reporting and Visualization Guidelines") explicitly demands:
  > Any HTML or artifact reports summarizing registration performance comparisons must always display structural/spatial images to visually inspect registration quality, including:
  > - Edge and/or region overlap between the registered image and the target image.
  > - Deformed grids visualizing the coordinate warping.
  > - Jacobian determinant maps illustrating local compression and expansion.
  > - Deformed/Warped images shown side-by-side (next to) target/fixed images.
  While warped and target images are shown, edge overlap maps, deformed grids, and Jacobian determinant maps are missing. The plotting functions `plot_warp_grid_2d` and `plot_jacobian_slice` are defined in `generate_ants_2d_comparison_report.py` but never called, resulting in their omission from the HTML.
- **Suggestion**: Update `examples/generate_ants_2d_comparison_report.py` to call `plot_warp_grid_2d` and `plot_jacobian_slice` for the PyTorch LNCC, JAX LNCC, and VGG-based registrations, convert them to base64, and embed the generated figures in `html_content` under each registration panel. Add a spatial visualization of edge overlap (e.g. subtracting warped from fixed intensity or plotting contours).

---

## Verified Claims

- **Single Interpolation Policy** (Transforms composed and applied in a single step) $\rightarrow$ verified via code inspection of `src/syntx/syn.py` (lines 1427, 1530-1533, and 1743-1781) and execution of test suite $\rightarrow$ **PASS**
  * *Method*: Confirmed that `compute_initial_grid` translates coordinates into an `initial_grid` tensor, which is composed dynamically inside `compose_grids` with learned affine and SyN deformation fields. The moving image is sampled exactly once during optimization and once at the end using `ants.apply_transforms` on the combined list `[fwd_file, affine_file] + tx_list`. No intermediate file-based pre-warping occurs.
- **Similarity Metric & VGG Guidelines** (No VGG 2D mode for accuracy tasks, only VGG 3D LNCC Layer 4) $\rightarrow$ verified via inspection of `registration()` signature in `src/syntx/syn.py` and `FeatureSpaceLoss` in `src/syntx/features.py` $\rightarrow$ **PASS**
  * *Method*: Verified that `vgg_mode` defaults to `'lncc_3d'` and `vgg_layers` defaults to `[4]`. Confirmed that `FeatureSpaceLoss` maps 3D inputs to 2D orthogonal slices, runs VGG19 features, reconstructs the 3D feature volume, and applies 3D local NCC to prevent spatial blurring and preserve high-frequency cortical boundaries.
- **Test Suite Execution** (All 94 unit tests pass successfully) $\rightarrow$ verified via running `pytest --runslow` $\rightarrow$ **PASS**
  * *Method*: Executed the full test suite. 94 passed, 0 failed, 6 warnings in 254.42 seconds. Overall test coverage is 92% (94% for `features.py`, 92% for `syn.py`, 90% for `syn_jax.py`).

---

## Coverage Gaps

- **Anisotropic Voxel Spacing Regularizer Scaling** — risk level: **Medium** — recommendation: **Investigate**
  * The separable Gaussian fluid and elastic filters assume isotropic voxel size or uniform coordinate scaling. In highly anisotropic spacing (e.g., sagittal slices spaced 5mm apart), the deformation field smoothing will be distorted. A future investigation should scale regularizer sigmas by voxel spacing per axis.

---

## Unverified Items

- **GPU Acceleration on CUDA Device** — reason not verified: Review executed on a macOS system without CUDA-capable hardware (using CPU fallback). PyTorch/MPS backends were executed and verified locally.

---

## Challenge Summary (Adversarial Critic)

**Overall risk assessment**: MEDIUM

---

## Challenges

### [Medium] Challenge 1: JAX-to-PyTorch DLPack Context Sharing

- **Assumption challenged**: The JAX SyN backend borrows the PyTorch-based modular VGG features extractor using DLPack sharing in `make_pytorch_loss_jax` and custom VJP. It assumes zero-copy tensors are safely managed across JAX and PyTorch without locking or memory leaks.
- **Attack scenario**: When running on multi-GPU environments where PyTorch and JAX allocate to different default CUDA device contexts, DLPack transfer may fail, block, or cause silent memory duplication and OOM.
- **Blast radius**: Registration fails dynamically or crashes with hardware context errors when switching to JAX backend on multi-GPU nodes.
- **Mitigation**: Enforce device synchronization and explicitly assert that both PyTorch and JAX tensors reside on the same logical device before DLPack conversion.

### [Medium] Challenge 2: Anisotropic Image Spacing Distortion in Gaussian Regularization

- **Assumption challenged**: The Gaussian fluid and elastic filters in `separable_gaussian_filter` apply isotropic smoothing in voxel space.
- **Attack scenario**: If a volume is highly anisotropic (e.g., spacing = 1.0mm x 1.0mm x 4.0mm), isotropic voxel-based filtering results in 4x physical smoothing in the Z-axis. This causes severe regularization distortion and can lead to topology folds.
- **Blast radius**: Folds and topological violations (negative Jacobian determinants) on anisotropic volumes.
- **Mitigation**: Adjust the smoothing kernel width (sigma) along each dimension to be inversely proportional to the spacing.

---

## Stress Test Results

- **Run JAX-to-PyTorch Loss Bridge** $\rightarrow$ executed JAX VGG test cases (`test_pytorch_loss_jax_jit` and `test_dlpack_empty_tensor`) $\rightarrow$ successfully compiled and propagated gradients without crashes $\rightarrow$ **PASS**
- **Test Topology Preservation on Extreme Deformation** $\rightarrow$ ran `tests/test_syn.py` with large displacement grids $\rightarrow$ Jacobian determinants remained strictly positive $\rightarrow$ **PASS**
