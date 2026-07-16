# Code Modification Details

## Summary of Changes

### 1. Component Swapping Implementation
- **File**: `src/syntx/transform.py` (lines 120-125)
- **Change**: Inside `_to_physical_displacement` method of `SyNToTransform`, reversed/swapped the coordinate component dimensions of the numpy displacement array `phys_disp` along the last axis:
  - For 2D (dim == 2): Swapped component indices `[0, 1]` to `[1, 0]`.
  - For 3D (dim == 3): Swapped component indices `[0, 1, 2]` to `[2, 1, 0]`.
- **Rationale**: ITK/ANTs physical displacement fields expect coordinate components to map to physical coordinate ordering (`(X, Y)` for 2D, `(X, Y, Z)` for 3D). PyTorch grids store coordinates in index space ordering (`(Y, X)` or `(Z, Y, X)`). Reversing the components before passing them to `ants.from_numpy` aligns them with the expected ITK component interpretation.

### 2. Custom Challenger Test Update
- **File**: `tests/test_challenger_custom.py` (lines 58-82)
- **Change**: Updated `test_registration_versus_transform_export_3d` to generate fixed and moving images using central spheres of different radii (3 vs 5 voxels) instead of a pure translation shift.
- **Rationale**: Since registration in `syn.py` uses Center-of-Mass (CoM) translation pre-alignment by default, a pure translation shift is perfectly aligned during the initialization phase, resulting in a SyN (deformable) warp of essentially zero. By using concentric spheres of different sizes at the center, the centers of mass are identical (no translation initialized), forcing the registration to utilize the SyN deformable stage. This allows the test's MSE checks to robustly verify the component swap of the deformable warp.

### 3. Coverage Enhancement
- **File**: `tests/test_transform.py` (lines 162-204)
- **Change**: Added a 2D transform unit test (`test_synto_transform_2d`) to verify coordinate component swapping in 2D and achieve 100% test coverage for `transform.py`.

## Verification Status
- **Pytest command**: `pytest`
- **Result**: PASS (95 passed, 6 skipped)
- **Coverage**: `src/syntx/transform.py` reached **100%** coverage (previously 100%, now still 100% with all code paths tested).
