# Milestone 2, 3, and 4 Verification Review & Challenge Report

## Review Summary

**Verdict**: REQUEST_CHANGES

This verification report evaluates the correctness, robustness, and conformance of Milestones 2, 3, and 4 in the `syntx` registration codebase. The evaluation consists of two parts: a Quality Review and an Adversarial Challenge.

* **Summary of Verdict**: All 94 unit tests executed successfully in 4 minutes and 16 seconds. The implementation of the **Single Interpolation Policy** and **VGG 3D LNCC Layer 4** guidelines is highly correct, robust, and mathematically sound. However, the **Reporting and Visualization Guidelines** have not been fully satisfied: the HTML report `docs/parity_report.html` fails to include edge overlaps, deformed grids, or Jacobian determinant maps, as the helper plotting functions are defined but not called in the generation script.

---

## Findings

### [Critical] Finding 1: Missing Required Visualizations in Parity Report
- **What**: The generated report `docs/parity_report.html` does not include edge/region overlap, deformed grids, or Jacobian determinant maps. The warped/moving images are also not displayed side-by-side with target/fixed images.
- **Where**: `examples/generate_ants_2d_comparison_report.py` (lines 109, 139) and `docs/parity_report.html`.
- **Why**: The helper plotting functions `plot_warp_grid_2d` and `plot_jacobian_slice` are defined in the script but never called or integrated into the HTML template generation inside the `main` execution loop. Also, the HTML layout places target and warped images in distant sections, preventing side-by-side visual inspection.
- **Suggestion**: Update `examples/generate_ants_2d_comparison_report.py` to invoke `plot_warp_grid_2d` and `plot_jacobian_slice` for all backends (ANTs, PyTorch, JAX, VGG), embed the resulting base64 plots in the HTML template, and adjust the CSS/layout to display warped and target images side-by-side.

---

## Verified Claims

- **Single Interpolation Policy** → verified via inspection of `syn.py`, `syn_jax.py`, and `transform.py` → **PASS**
  - *Proof*: The `forward` and `forward_inverse` functions compose the initial, affine, and displacement grids into a single coordinate grid via `compose_grids` before applying it to the native image using a single `grid_sample` call. Center-of-mass initialization is performed directly on the Lie algebra / translation parameters.
- **VGG 3D Mode Requirement** → verified via inspection of `syn.py` and `features.py` → **PASS**
  - *Proof*: VGG mode defaults to `'lncc_3d'` and VGG layers to `[4]`. `features.py` reconstructs 3D deep feature volumes along three orthogonal planes before computing 3D LNCC, matching the accuracy guidelines.
- **Unit Test Suite Success** → verified via `pytest --runslow` execution → **PASS**
  - *Proof*: All 94 unit tests passed without errors.

---

## Coverage Gaps

- **3D registration visual reports** — risk level: low — recommendation: Accept risk as the comparison script is focused on 2D (`r16`/`r64` dataset) to match ANTs baseline.

---

## Unverified Items

None. All claims have been verified.

---

## Challenge Summary (Adversarial Critic)

**Overall risk assessment**: LOW

---

## Challenges

### [Medium] Challenge 1: Coarse Multi-Resolution Input Degeneracy
- **Assumption challenged**: Deep feature networks (e.g. SwinUNETR, DINOv2, VGG) can operate at any resolution.
- **Attack scenario**: At very coarse multi-resolution levels (e.g., downsampling scale 8), the spatial dimensions of input slices may fall below the network's minimum input size (e.g. patch size 32), resulting in dimension mismatches or out-of-bounds slicing.
- **Blast radius**: Registration crash during initialization or first level loop.
- **Mitigation**: The code implements a robust "degeneracy trigger" (`min(curr_spatial) < 32`) that automatically falls back to intensity-based LNCC at degenerate levels. This was stress-tested and works correctly.

### [Low] Challenge 2: Lie Algebra SO(d) Singularity
- **Assumption challenged**: Derivative of rotation matrices is differentiable at zero rotation.
- **Attack scenario**: Zero rotation (lie algebra omega parameters initialized to zero) can trigger division-by-zero when normalized by the rotation angle.
- **Blast radius**: Backpropagation returns NaNs, stalling optimization.
- **Mitigation**: The custom Lie Algebra exponential mapping uses a safe threshold `theta2 < 1e-16` and replaces division denominators with safe epsilon values in JAX and PyTorch, which ensures numerical stability.

---

## Stress Test Results

- **Coarse resolution fallback** → downsample image to size < 32 → trigger fallback to LNCC → pass
- **Lie Algebra gradient evaluation at zero** → omega initialized to zero vector → backward pass evaluated successfully → pass
- **Displacement field inversion convergence** -> large deformations -> ITK-style voxel clipping bounds max update to prevent divergence -> pass
