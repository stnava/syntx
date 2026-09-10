# Test Infrastructure Specification: Spatial Coordinate Centralization & Invariance

## 1. Objectives & Testing Strategy
This test infrastructure specification establishes the rigorous verification framework for the centralization of physical space and coordinate management into `syntx.spatial`. It ensures mathematical precision, elimination of ad-hoc coordinate conversions, and complete backward compatibility across the entire repository.

The test suite follows the **Dual-Track 4-Tier Testing Methodology**:
- **Track A (Roundtrip & Invariance)**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) across all coordinate domains (PyTorch/JAX tensor domain $\leftrightarrow$ ITK/ANTs physical scanner space), across 2D/3D isotropic and anisotropic grids with arbitrary direction cosines, batched fields, and disk persistence.
- **Track B (Centralization & Backward Compatibility)**: Verifies that `syntx.spatial` serves as the single source of truth, top-level accessibility via `syntx.spatial`, `__all__` inclusion, complete backward compatibility of re-exports across `syntx.transform`, `syntx.core.grid`, and `syntx.core.affine`, affine parameter export fidelity, and coordinate scaling accuracy.

---

## 2. 4-Tier Test Matrix

### Tier 1: Feature Coverage (Primary Capabilities)
*Requirement: $\ge 5$ test cases per feature*

#### Feature 1: Displacement Field Domain Bridges (`disp_tensor_to_itk` $\leftrightarrow$ `disp_itk_to_tensor`)
1. **`test_roundtrip_2d_isotropic`**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) for 2D isotropic non-square grids with identity direction matrix.
2. **`test_roundtrip_2d_anisotropic_rotation`**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) for 2D anisotropic non-square grids with non-trivial 2D planar rotation direction cosines.
3. **`test_roundtrip_3d_isotropic_non_cubic`**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) for 3D isotropic non-cubic grids (e.g. $24 \times 28 \times 32$).
4. **`test_roundtrip_3d_anisotropic_arbitrary_direction`**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) for 3D anisotropic grids with fully arbitrary orthonormal direction cosine matrices.
5. **`test_roundtrip_disk_nifti`**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) for displacement fields exported to temporary NIfTI files via `ants.image_write` and re-imported via `disp_itk_to_tensor(filepath)`.
6. **`test_roundtrip_batched_tensor`**: Verifies exact numerical identity ($L_\infty < 10^{-6}$) for batched displacement fields ($B > 1$) converting to lists of ANTsImages and stacking back to full $(B, *spatial, dim)$ tensors.

#### Feature 2: Spatial Coordinate Centralization & Module Harmonization
7. **`test_top_level_spatial_import`**: Verifies that `import syntx` provides direct, clean access to `syntx.spatial` without requiring manual submodule import.
8. **`test_spatial_all_inclusion`**: Verifies that `'spatial'` is explicitly declared in `syntx.__all__`.
9. **`test_transform_backward_compatibility`**: Verifies that `export_ants_displacement_field` and `export_ants_affine_transform` in `syntx.transform` remain available and functional.
10. **`test_core_grid_backward_compatibility`**: Verifies that `get_physical_grid_torch`, `physical_to_normalized_torch`, and `physical_to_normalized_torch_cached` in `syntx.core.grid` remain available and functional.
11. **`test_core_affine_backward_compatibility`**: Verifies that `grid_to_physical_affine` and `parse_ants_affine` in `syntx.core.affine` remain available and functional.
12. **`test_affine_parameter_export_3d`**: Verifies that `export_ants_affine_transform` exports forward and inverse `ants.ANTsTransform` objects with parameter layout matching ITK AffineTransform ($3\times 3$ matrix + translation, fixed parameters = 0).
13. **`test_affine_parameter_export_2d`**: Verifies that `export_ants_affine_transform` exports 2D forward and inverse `ants.ANTsTransform` objects ($2\times 2$ matrix + translation).

---

### Tier 2: Boundary, Edge & Corner Cases
*Requirement: $\ge 5$ test cases per feature*

#### Feature 1: Displacement Field Domain Bridges
14. **`test_roundtrip_zero_identity_field`**: Verifies that zero displacement fields (identity transforms) preserve exact numerical zero ($L_\infty = 0.0$) across all dimensions.
15. **`test_roundtrip_extreme_magnitudes`**: Verifies numerical stability and precision when displacements contain large physical offsets ($\pm 100.0\text{ mm}$).
16. **`test_roundtrip_unbatched_tensor`**: Verifies that unbatched displacement tensors $(*spatial, dim)$ are accepted and processed correctly without requiring manual `unsqueeze(0)`.
17. **`test_disp_itk_to_tensor_empty_sequence_raises`**: Verifies that passing an empty list or tuple `[]` to `disp_itk_to_tensor` raises `ValueError`.
18. **`test_disp_itk_to_tensor_sequence_of_files`**: Verifies that a sequence of NIfTI file paths stacks into a single batched tensor of shape $(B, *spatial, dim)$.
19. **`test_roundtrip_numpy_input`**: Verifies that `disp_tensor_to_itk` correctly accepts raw NumPy arrays `np.ndarray` without requiring PyTorch tensors.
20. **`test_metadata_preservation`**: Verifies that reference image origin, spacing, and direction cosines are preserved verbatim through the conversion pipeline.

#### Feature 2: Spatial Coordinate Centralization & Module Harmonization
21. **`test_affine_identity_roundtrip`**: Verifies that identity physical affine parameters ($M = I, t = 0$) yield identity forward and inverse ITK transforms.
22. **`test_affine_pure_translation`**: Verifies that pure translation transforms ($M = I, t \ne 0$) yield inverse translation $t_{inv} = -t$.
23. **`test_affine_tensor_and_numpy_inputs`**: Verifies that `export_ants_affine_transform` accepts both PyTorch tensors and NumPy arrays for $M_{phys}$ and $t_{phys}$.
24. **`test_affine_file_export_and_read`**: Verifies that exporting to a `.mat` transform file produces an artifact that can be read back and evaluated by `ants.read_transform`.
25. **`test_coordinate_grid_scaling_boundaries`**: Verifies that `get_physical_grid_torch` produces physical grid points whose extreme coordinates match origin and $(N-1) \times spacing$.
26. **`test_coordinate_grid_normalization_range`**: Verifies that `physical_to_normalized_torch` maps the physical domain strictly into $[-1.0, 1.0]$.
27. **`test_autograd_physical_scale_channel_flip`**: Verifies that coordinate scaling for autograd backpropagation flips dimension 0 (`torch.flip`), correctly mapping $x$-displacements by $x$-spacing/dimensions rather than $z$-dimensions on anisotropic acquisitions (GEMINI.md Rule 2).

---

### Tier 3: Cross-Feature Interactions & Architectural Integration
28. **`test_roundtrip_with_reverse_components_invariance`**: Verifies that `reverse_components` is strictly self-inverting ($f(f(x)) = x$) and matches the channel reversal in `disp_tensor_to_itk`.
29. **`test_roundtrip_with_jacobian_evaluation`**: Verifies that calculating Jacobian determinant maps via `jacobian_determinant` on a displacement field yields identical results before and after ITK roundtrip ($L_\infty < 10^{-6}$).
30. **`test_roundtrip_with_image_resampling`**: Verifies that resampling a scalar image using exported displacement fields matches PyTorch internal `F.grid_sample` within linear interpolation tolerances.
31. **`test_grid_to_physical_affine_and_export`**: Verifies that converting a normalized grid affine matrix $T_{grid}$ into physical affine parameters and exporting to ANTsTransform preserves spatial mapping consistency.
32. **`test_physical_grid_to_normalized_roundtrip`**: Verifies that passing coordinates generated by `get_physical_grid_torch` into `physical_to_normalized_torch` reproduces the canonical normalized coordinate grid.

---

### Tier 4: Real-World Workload Scenarios
33. **`test_real_world_anisotropic_neuroimaging_geometry`**: Simulates a high-resolution anisotropic clinical neuroimaging acquisition (sagittal spacing $0.8 \times 0.8 \times 2.5\text{ mm}$, volume $240 \times 256 \times 40$, oblique scan plane) and verifies roundtrip invariance ($L_\infty < 10^{-6}$).
34. **`test_real_world_deformation_stats_pipeline`**: Verifies the end-to-end deformation analysis pipeline: synthetic non-rigid warp $\rightarrow$ `disp_tensor_to_itk` $\rightarrow$ `deformation_stats` $\rightarrow$ `disp_itk_to_tensor` $\rightarrow$ verification of folding percentage, min/max $\det(J)$, and harmonic norms.
35. **`test_affine_transform_application_on_image`**: Exports physical affine parameters to an ANTsTransform, applies it to a 3D volume using `ants.apply_transforms`, and confirms geometric accuracy.
36. **`test_multi_resolution_grid_generation`**: Evaluates physical and normalized coordinate grid generators across multi-resolution pyramid levels (levels 1, 2, 4) and confirms spatial extent invariance.

---

## 3. Test Execution Framework & Commands

### Test Execution Commands
- **Run Roundtrip Invariance Test Suite**:
  ```bash
  pytest -v tests/test_spatial_roundtrip.py
  ```
- **Run Centralization & Backward Compatibility Test Suite**:
  ```bash
  pytest -v tests/test_spatial_centralization.py
  ```
- **Run Complete E2E Suite**:
  ```bash
  pytest -v tests/test_spatial_roundtrip.py tests/test_spatial_centralization.py
  ```
- **Run Full Regression Suite**:
  ```bash
  pytest tests/
  ```

### Pass/Fail Criteria & Acceptance Standards
- **Numerical Identity Invariant**: $L_\infty = \max |x_{\text{orig}} - x_{\text{rec}}| < 10^{-6}$ across all roundtrip conversions.
- **Strict Backward Compatibility**: 100% of legacy imports from `syntx.transform`, `syntx.core.grid`, and `syntx.core.affine` continue to function identically.
- **Zero Regressions**: 100% of tests in `tests/test_spatial_roundtrip.py` and `tests/test_spatial_centralization.py` must pass cleanly.
