# Original User Request

## Initial Request — 2026-09-09T21:58:43Z

Centralize all management of physical space for PyTorch and JAX into `syntx.spatial` to establish a general, mathematically unified approach and a single debugging point across the entire repository. Eliminate all scattered, ad-hoc transpose operations, coordinate flips, and channel reversals across `syn.py`, `tvf.py`, `syngs.py`, `robust_affine.py`, and `transform.py`.

Integrity mode: development

Requirements:
### R1. Single Source of Truth in `syntx.spatial`
- Consolidate all coordinate conversions, displacement field domain bridges, and affine exports natively inside `src/syntx/spatial.py`.
- Implement native `disp_tensor_to_itk` and `export_ants_displacement_field` directly in `spatial.py` (no downstream delegation to `transform.py`).
- Implement native `export_ants_affine_transform` and `grid_to_physical_affine` directly in `spatial.py`.
- Consolidate physical coordinate grid generation (`get_physical_grid_torch`, `physical_to_normalized_torch`) into `syntx.spatial`.
- Export `spatial` as a top-level module in `src/syntx/__init__.py` and include it in `__all__`.
- Provide backward-compatible import aliases in `src/syntx/transform.py` and `src/syntx/core/grid.py`.

### R2. Eradicate Scattered Transpose & Channel Reversal Snippets
- Audit and eliminate all manual array transpositions (e.g. `.transpose(0, 3, 2, 1, 4)` or `.transpose(0, 2, 1, 3)`) and channel reversals (`[..., ::-1]`) in:
  - `src/syntx/syn.py`
  - `src/syntx/tvf.py`
  - `src/syntx/syngs.py`
  - `src/syntx/robust_affine.py`
  - `src/syntx/scattered/mapping.py`
- Replace all ad-hoc conversions with direct calls to `syntx.spatial` conversion primitives.

### R3. Harmonize `SyNToTransform` with `syntx.spatial`
- Refactor `SyNToTransform` in `src/syntx/transform.py` so that all internal domain conversions, grid normalizations, Jacobian determinant maps, and export routines route directly through `syntx.spatial`.

### R4. Verification and Non-Regression Invariant
- **Roundtrip Invariance**: Verify that converting between tensor domain and ITK displacement fields (`disp_tensor_to_itk` <-> `disp_itk_to_tensor`) achieves exact numerical identity ($L_\infty < 10^{-6}$) for 2D, 3D isotropic, and 3D anisotropic volumes with arbitrary direction cosines.
- **Unit & Regression Tests**: Verify that 100% of tests in `pytest tests/` pass cleanly without regressions.
- **Benchmark Parity**: Verify that `syntx.syn` on canonical `mbhard` achieves peak performance:
  - Symmetric Cortical DICE >= 0.630
  - Whole-volume grid folding <= 0.015% and strictly positive minimum Jacobian (min det(J) > 0).

Acceptance Criteria:
- Centralization & Single Debugging Point:
  - `syntx.spatial` is the sole module in the repository containing array transposition and component channel reversal logic for ITK <-> Tensor coordinate conversion.
  - `syn.py`, `tvf.py`, `syngs.py`, and `robust_affine.py` contain zero ad-hoc `.transpose()` or `[::-1]` for coordinate domain conversions; all route through `syntx.spatial`.
  - `import syntx; syntx.spatial` is accessible, and `spatial` is included in `syntx.__all__`.
  - Backward compatibility: existing imports from `syntx.transform` (`export_ants_displacement_field`, `export_ants_affine_transform`) work without breakage.
- Mathematical Precision & Parity:
  - Roundtrip fidelity test passes with $L_\infty < 10^{-6}$ across 2D, 3D isotropic, and 3D anisotropic inputs.
  - Image resampling using exported warps matches PyTorch internal `grid_sample` within interpolation tolerances.
  - All test cases in `pytest tests/` pass cleanly.
- Registration Performance Verification:
  - `syntx.syn` on `mbhard` completes successfully with the centralized `syntx.spatial` pipeline.
  - Symmetric Cortical DICE on `mbhard` >= 0.630.
  - Grid folding on `mbhard` <= 0.015% with strictly positive interior determinants.
