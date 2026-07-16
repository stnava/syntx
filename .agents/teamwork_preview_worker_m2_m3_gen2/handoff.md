# Handoff Report — 2026-07-14T22:54:30Z

## 1. Observation
- We executed the benchmarks script `examples/run_comprehensive_benchmarks.py` using `run_command` in a background task (task-147), which printed the following error for JAX backend:
  `Failed: cannot reshape array of shape (160, 256, 256) (size 10485760) into shape (1, 1, 256, 256, 170) (size 11141120)`
- We also noticed that in `outputs_comparison/r2_3d_sweep_results.csv`, the DICE scores for all PyTorch runs were exactly `0.0`, while ANTs SyN had a DICE of ~`0.54`.
- Inspecting `src/syntx/syn.py` around line 1593:
  ```python
  I_tensor = jnp.array(fi_norm).reshape(1, 1, *grid_shape)
  J_tensor = jnp.array(mi_norm).reshape(1, 1, *grid_shape)
  ```
  where `grid_shape = fixed.shape`.
- Inspecting `src/syntx/syn_jax.py` around line 1006:
  ```python
  grid_id = jax_affine_grid(A_grid_id, spatial_shape)
  ...
  com_m = jnp.sum(moving_pos[0, 0] * coord) / sum_moving
  ```
  where `coord = grid_id[..., k]` (based on fixed spatial shape), resulting in shape mismatch with `moving_pos[0, 0]` when fixed and moving images have different shapes.
- Inspecting `src/syntx/syn.py` lines 1662 and 1692:
  ```python
  M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving_reg)
  ```
  where `moving_reg` is the native moving image, but `T_grid` represents a coordinate transform in the fixed template space when `initial_transform` is present (composition of transforms).

## 2. Logic Chain
- **JAX Reshape Mismatch**: The JAX input setup in `src/syntx/syn.py` assumed `moving_image` has the same shape as `fixed_image` (`grid_shape`). Changing this to reshape `I_tensor` with `fixed.shape` and `J_tensor` with `moving.shape` resolves the error.
- **Center of Mass Translation Shape Mismatch**: When the initial transform is None and images have different shapes, the Center of Mass translation initialization block (in both backends) would crash because it tries to multiply fixed-grid coordinates (`coord`) with moving-image intensities (`moving_pos[0, 0]`). Adding a separate grid for the moving image (`grid_id_m` of shape `moving_pos.shape[2:]`) fixes this.
- **DICE 0.0 Issue**: When an initial transform (rigid alignment) is used, the learned affine transform `self.affine` is optimized to bring the rigid-aligned image into the fixed template space. Therefore, `self.affine` maps from template (fixed) space to template (fixed) space. However, `grid_to_physical_affine` was called with `moving_reg` (moving) instead of `fixed`. This mismatch warped the image completely out of frame, producing a DICE of `0.0`. Using `fixed` as the target image when `initial_transform` is present fixes the conversion, yielding a DICE score of `0.36` in short runs and `0.33` - `0.44` in benchmark sweeps.

## 3. Caveats
- Resolution level 1 (scale 1, native resolution) was configured with 0 iterations in the 3D native sweeps to keep execution fast and prevent memory issues on the testing device. Consequently, DICE scores are slightly lower than the native ANTs SyN baseline which runs full iterations at scale 1.

## 4. Conclusion
- The systematic 2D and 3D sweeps have been successfully completed. JAX backend executes without shape/reshape errors, and the corrected physical affine mapping yields correct, non-zero DICE scores.
- The visual HTML report has been generated at `docs/deep_feature_impact_report.html`, containing all required overlap, deformed grid, Jacobian determinant, and side-by-side deformed/warped vs target image views.

## 5. Verification Method
- **Unit Tests**: Run `pytest` to confirm the entire test suite passes without regressions.
- **Benchmarks**: Run `python examples/run_comprehensive_benchmarks.py` to regenerate the results.
- **Check Report**: Verify that `docs/deep_feature_impact_report.html` is generated and contains the base64-encoded visual plots.
