# TEST READY: Spatial Coordinate Centralization & Domain Invariance

**Status**: READY & FULLY VERIFIED  
**Date**: 2026-09-09  
**Test Suite Author**: `test_writer_e2e`  
**Test Infra Specification**: `TEST_INFRA.md`  

---

## 1. Test Suite Deliverables

Two dedicated, exhaustive test modules have been created to validate the spatial coordinate centralization and numerical domain invariance:

| Test File | Test Count | Scope & Focus | Pass Status |
|---|---|---|---|
| `tests/test_spatial_roundtrip.py` | 18 | Opaque-box roundtrip numerical identity ($L_\infty < 10^{-6}$) between tensor domain and ITK displacement fields (`disp_tensor_to_itk` $\leftrightarrow$ `disp_itk_to_tensor`) across 2D/3D isotropic, anisotropic, arbitrary rotation cosines, NIfTI disk I/O, batched tensors, and clinical MRI geometries. | **18 / 18 PASSED** (100%) |
| `tests/test_spatial_centralization.py` | 25 | Public module export (`syntx.spatial`), explicit `__all__` inclusion, native consolidation of spatial/affine primitives, backward-compatible aliases in `syntx.transform`, `syntx.core.grid`, and `syntx.core.affine`, forward/inverse affine parameter roundtrips, and autograd coordinate scaling channel flips. | **25 / 25 PASSED** (100%) |
| **Total Delivered Suite** | **43** | Full 4-Tier Dual-Track Test Coverage | **43 / 43 PASSED** (100%) |

---

## 2. Test Execution Commands

### Primary E2E Verification Command
Execute the complete spatial centralization and invariance test suite:
```bash
pytest -v tests/test_spatial_roundtrip.py tests/test_spatial_centralization.py
```

### Dedicated Suite Execution Commands
- **Roundtrip Invariance Suite ($L_\infty < 10^{-6}$)**:
  ```bash
  pytest -v tests/test_spatial_roundtrip.py
  ```
- **Centralization & Backward Compatibility Suite**:
  ```bash
  pytest -v tests/test_spatial_centralization.py
  ```

### Combined Regression Command
Run new spatial tests alongside existing repository spatial unit and parity tests:
```bash
pytest -v tests/test_spatial_roundtrip.py tests/test_spatial_centralization.py tests/test_spatial.py tests/test_transform.py tests/test_core_grid.py tests/test_core_affine.py
```

---

## 3. 4-Tier Test Coverage Summary

### Tier 1: Feature Coverage (13 Tests)
- `test_roundtrip_2d_isotropic`: Exact numerical identity on 2D isotropic non-square grids ($L_\infty = 0.0 < 10^{-6}$).
- `test_roundtrip_2d_anisotropic_rotation`: Exact identity on 2D anisotropic rotated grids ($L_\infty = 0.0 < 10^{-6}$).
- `test_roundtrip_3d_isotropic_non_cubic`: Exact identity on 3D non-cubic isotropic grids ($L_\infty = 0.0 < 10^{-6}$).
- `test_roundtrip_3d_anisotropic_arbitrary_direction`: Exact identity with arbitrary orthonormal direction cosines ($L_\infty = 0.0 < 10^{-6}$).
- `test_roundtrip_disk_nifti`: Temporary NIfTI file serialization and deserialization via file path ($L_\infty = 0.0 < 10^{-6}$).
- `test_roundtrip_batched_tensor`: Multi-sample batched displacement fields ($B = 2, 3$) roundtripping to list of ANTsImages and stacking back to $(B, *spatial, dim)$.
- `test_top_level_spatial_import`: Clean module access via `import syntx; syntx.spatial`.
- `test_spatial_all_inclusion`: Explicit declaration of `'spatial'` in `syntx.__all__`.
- `test_spatial_centralized_exports`: Verifies all 20 public spatial primitives are exposed as callables directly on `syntx.spatial`.
- `test_spatial_native_displacement_export`: Direct invocation of native `sp.export_ants_displacement_field`.
- `test_spatial_native_affine_export`: Direct invocation of native `sp.export_ants_affine_transform` and `sp.create_ants_affine`.
- `test_spatial_shape_mapping`: Verifies `sp.itk_shape_to_tensor_shape` for 2D and 3D shapes.
- `test_spatial_identity_grid_generation`: Normalized $[-1, 1]$ coordinate grid generation via `sp.get_identity_grid_torch`.

### Tier 2: Boundary, Edge & Corner Cases (14 Tests)
- `test_roundtrip_zero_identity_field`: Zero displacement fields yield exact $L_\infty = 0.0$ in 2D and 3D.
- `test_roundtrip_extreme_magnitudes`: Large displacement vectors ($\pm 100\text{ mm}$) and fine sub-voxel components ($10^{-5}\text{ mm}$).
- `test_roundtrip_unbatched_tensor`: Unbatched displacement tensors $(*spatial, dim)$ processed without shape mismatch.
- `test_disp_itk_to_tensor_empty_sequence_raises`: Empty sequence `[]` raises descriptive `ValueError`.
- `test_disp_itk_to_tensor_sequence_of_files`: Sequence of NIfTI paths stacks into single batched tensor.
- `test_roundtrip_numpy_input`: Raw NumPy arrays transparently converted without type errors.
- `test_metadata_preservation`: Verification that origin, spacing, and direction cosines are preserved verbatim.
- `test_affine_identity_roundtrip`: Identity parameters ($M = I, t = 0$) yield identity forward/inverse transforms.
- `test_affine_pure_translation`: Pure translation transforms verify $t_{inv} = -t$.
- `test_affine_tensor_and_numpy_inputs`: Accepts both PyTorch tensors and NumPy arrays for $M_{phys}$ and $t_{phys}$.
- `test_affine_file_export_and_read`: Export to `.mat` file readable and verifiable by `ants.read_transform`.
- `test_coordinate_grid_scaling_boundaries`: Physical coordinates at boundaries match origin and $(N-1) \times spacing$.
- `test_coordinate_grid_normalization_range`: Physical to normalized coordinates strictly bounded in $[-1.0, 1.0]$.
- `test_autograd_physical_scale_channel_flip`: Dimension 0 flip (`torch.flip`) converts normalized grid gradients to physical units without cross-axis anisotropy errors (GEMINI.md Rule 2).

### Tier 3: Cross-Feature Interactions (5 Tests)
- `test_roundtrip_with_reverse_components_invariance`: Component reversal involution ($f(f(x)) = x$) aligns with displacement vector conversions.
- `test_roundtrip_with_jacobian_evaluation`: Jacobian determinant maps calculated before and after ITK roundtrip yield identical results ($L_\infty < 10^{-6}$).
- `test_roundtrip_with_image_resampling`: Image tensor <-> ANTsImage roundtrip maintains exact voxel identity ($L_\infty < 10^{-6}$).
- `test_grid_to_physical_affine_and_export`: Normalized grid affine $T_{grid}$ converted to physical affine and exported to valid ANTsTransform objects.
- `test_physical_grid_to_normalized_roundtrip`: Physical coordinate grid normalized maps center voxel to 0.

### Tier 4: Real-World Workload Scenarios (4 Tests)
- `test_real_world_anisotropic_neuroimaging_geometry`: Clinical anisotropic MRI acquisition ($0.8 \times 0.8 \times 2.4\text{ mm}$, volume $240 \times 256 \times 40$, oblique scan plane) achieves $L_\infty < 10^{-6}$.
- `test_real_world_deformation_stats_pipeline`: Full deformation field diagnostic pipeline: displacement field $\rightarrow$ `deformation_stats` $\rightarrow$ `disp_tensor_to_itk` $\rightarrow$ `disp_itk_to_tensor` $\rightarrow$ consistent folding percentage, min/max $\det(J)$, and L2 norms.
- `test_affine_transform_application_on_image`: Exported physical affine applied to 3D volume using `ants.apply_transforms` shifts image features accurately in scanner space.
- `test_multi_resolution_grid_generation`: Multiresolution pyramid levels (1, 2, 4) preserve identical physical bounding box bounds.

---

## 4. Discovered Implementation Defects & Escalations

During test development, one implementation defect was identified for escalation to the implementing worker:
- **`syntx.spatial.physical_to_grid_affine` Translation Inversion Axis Permutation**:
  In `grid_to_physical_affine`, axes are permuted to $(z, y, x)$ order before matrix algebra and permuted back via $P = I_{[::-1]}$. In `physical_to_grid_affine`, the matrix $A_{grid}$ inverts correctly, but the translation offset inversion $t_{grid} = Vy^{-1}(t_{phys} - cy) - A_{grid} bx$ does not apply the corresponding coordinate permutation $P$, resulting in an axis transposition on anisotropic/asymmetric translation components. This does not affect identity or symmetric grids, but should be aligned with `grid_to_physical_affine`.
