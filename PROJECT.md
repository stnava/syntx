# Project: Centralize Physical Space Management into `syntx.spatial`

## Architecture
- **Single Source of Truth**: All coordinate domain conversions between ITK scanner space (Cartesian XYZ, mm, direction cosines, spacing, origin) and Tensor domain (PyTorch/JAX C-contiguous ZYX array layout, physical vector channels, normalized grid $[-1, 1]$) are consolidated natively within `src/syntx/spatial.py`.
- **Elimination of Circular / Inverted Dependencies**: `syntx.spatial` does not import or delegate downstream to `syntx.transform` or `syntx.syn`. Instead, `syntx.transform`, `syntx.core.grid`, and `syntx.core.affine` import and re-export primitives from `syntx.spatial` to maintain complete backward compatibility.
- **Top-Level Accessibility**: `syntx.spatial` is imported in `src/syntx/__init__.py` and explicitly exposed in `__all__`.
- **Eradication of Scattered Snippets**: Registration algorithms (`syn.py`, `tvf.py`, `syngs.py`, `robust_affine.py`), transform pipelines (`transform.py`), and scattered coordinate mappers (`scattered/mapping.py`, `scattered/transport.py`) route all domain bridging through `syntx.spatial` primitives.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Native `disp_tensor_to_itk` & `export_ants_displacement_field` | Native conversion of PyTorch/JAX displacement tensors `(B, *spatial, dim)` to `ants.ANTsImage` displacement fields without delegating to `transform.py` | M1 (DONE) | Survey (R1) |
| 2 | Native `disp_itk_to_tensor` | Converts ANTs displacement image(s) or file paths to batched tensor `(B, *spatial, dim)` | M1 (DONE) | Survey (R1) |
| 3 | Native `export_ants_affine_transform` & `create_ants_affine` | Standardized export of physical affine parameters $(M_{phys}, t_{phys})$ to forward/inverse `ants.ANTsTransform` objects | M1 (DONE) | Survey (R1) |
| 4 | Native `grid_to_physical_affine` & `grid_to_physical_affine_torch` | Converts normalized grid affine matrix $T_{grid}$ into physical affine parameters $(M_{phys}, t_{phys})$ in NumPy and PyTorch | M1 (DONE) | Survey (R1) |
| 5 | Native `physical_to_grid_affine` | Converts physical affine parameters back to normalized grid affine matrix $T_{grid}$ without axis permutation | M1 (DONE) | Survey (R1) |
| 6 | Native `get_physical_grid_torch` | Generates physical coordinate grid tensor $X_{phys}$ `(1, *shape, dim)` from spatial metadata | M1 (DONE) | Survey (R1) |
| 7 | Native `physical_to_normalized_torch` & cached variant | Maps physical coordinates to PyTorch normalized grid coordinates in $[-1, 1]$ | M1 (DONE) | Survey (R1) |
| 8 | Native `get_identity_grid_torch` | Standardized normalized identity grid generator for `F.grid_sample` | M1 (DONE) | Survey (R1, R3) |
| 9 | Native `compute_autograd_physical_scale` | Computes correct coordinate scaling vector converting autograd normalized grid gradients to physical displacement gradients on anisotropic grids | M1 (DONE) | Survey (R1, R2) |
| 10 | Native `image_to_tensor` & `tensor_to_image` | Standardized scalar image conversion bridging ITK `(X, Y, Z)` and PyTorch `(1, 1, Z, Y, X)` | M1 (DONE) | Survey (R1, R2) |
| 11 | Component & Metadata Reversal Primitives | `reverse_components`, `reverse_metadata`, `itk_shape_to_tensor_shape` | M1 (DONE) | Survey (R1, R2) |
| 12 | Top-level Export & Backward Compatibility | Export `spatial` in `syntx/__init__.py` (`__all__`), provide re-exports in `transform.py`, `core/grid.py`, and `core/affine.py` | M1 (DONE) | Survey (R1) |
| 13 | Harmonize `SyNToTransform` | Route all grid normalizations, physical displacement conversions, Jacobian maps, and export routines in `SyNToTransform` through `syntx.spatial` | M2 | Survey (R3) |
| 14 | Eradicate Double Component Reversal Hack | Remove redundant `phys_disp[..., ::-1]` flip in `SyNToTransform._to_physical_displacement` | M2 | Survey (R3) |
| 15 | Eradicate Scattered Transposes in `syn.py` | Replace 10 scattered transpose/flip sites (SYN-1 through SYN-10) with `syntx.spatial` primitives | M3 | Survey (R2) |
| 16 | Eradicate Scattered Transposes in `tvf.py` | Replace 4 scattered transpose/flip sites (TVF-1 through TVF-4) with `syntx.spatial` primitives | M3 | Survey (R2) |
| 17 | Eradicate Scattered Transposes in `syngs.py` | Replace 4 scattered transpose/flip sites (SYNGS-1 through SYNGS-4) with `syntx.spatial` primitives | M3 | Survey (R2) |
| 18 | Eradicate Scattered Transposes in `robust_affine.py` | Replace 3 scattered transpose/flip sites (ROB-1 through ROB-3) with `syntx.spatial` primitives | M3 | Survey (R2) |
| 19 | Eradicate Scattered Transposes in `scattered/` | Replace coordinate convention flips (SCAT-1, SCAT-2) in `mapping.py` and `transport.py` | M3 | Survey (R2) |
| 20 | Roundtrip Invariance Test Suite | Enforce exact numerical identity $L_\infty < 10^{-6}$ for `disp_tensor_to_itk` <-> `disp_itk_to_tensor` across 2D, 3D isotropic, and 3D anisotropic volumes with arbitrary direction cosines | M4 | Survey (R4) |
| 21 | Full Regression Test Suite Pass | 100% of tests in `pytest tests/` pass cleanly without regressions | M4 | Survey (R4) |
| 22 | Mindboggle `mbhard` Benchmark Parity | `syntx.syn` on Pair 44 (`NKI-TRT-20-2` -> `MMRR-21-2`) achieves Symmetric Cortical DICE $\ge 0.630$, whole-volume grid folding $\le 0.015\%$, and $\min \det(J) > 0$ | M4 | Survey (R4) |
| 23 | Adversarial Coverage Hardening | White-box edge-case stress testing of spatial transformations across singular matrices, extreme anisotropies, and batching | M4 | Survey (R4) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Native `syntx.spatial` Primitives & Aliases | Implement all native conversion and coordinate primitives in `src/syntx/spatial.py`; establish backward-compatible aliases in `transform.py`, `core/grid.py`, `core/affine.py`; export `spatial` in `src/syntx/__init__.py` | none | DONE |
| 2 | M2: Harmonize `SyNToTransform` | Refactor `SyNToTransform` in `src/syntx/transform.py` to route through `syntx.spatial` primitives; eliminate double component reversal hack | M1 | DONE |
| 3 | M3: Eradicate Scattered Transposes in Registration Modules | Eliminate ad-hoc `.transpose()` and channel flips in `syn.py`, `tvf.py`, `syngs.py`, `robust_affine.py`, and `scattered/mapping.py` | M1, M2 | DONE |
| 4 | M4: Comprehensive E2E Verification & Benchmark Parity | Verify roundtrip invariance ($L_\infty < 10^{-6}$), pass 100% of `pytest tests/`, verify `mbhard` benchmark parity (DICE $\ge 0.630$, folding $\le 0.015\%$, $\min \det(J) > 0$), and harden via adversarial tests | M1, M2, M3 | DONE |

## Interface Contracts
### `syntx.spatial` ↔ `syntx.syn` / `syntx.tvf` / `syntx.syngs` / `syntx.robust_affine`
- `disp_tensor_to_itk(disp: Tensor | ndarray, ref_image: ANTsImage) -> ANTsImage | list[ANTsImage]`
- `disp_itk_to_tensor(disp_img: ANTsImage | str | Sequence, device: str = 'cpu') -> Tensor (B, *spatial, dim)`
- `export_ants_displacement_field(disp: Tensor | ndarray, origin=None, spacing=None, direction=None, ref_image=None) -> ANTsImage`
- `export_ants_affine_transform(M_phys: ndarray | Tensor, t_phys: ndarray | Tensor, dim: int, filename: str = None) -> tuple[ANTsTransform, ANTsTransform]`
- `grid_to_physical_affine(T_grid: ndarray | Tensor, fixed: ANTsImage, moving: ANTsImage) -> tuple[ndarray, ndarray]`
- `grid_to_physical_affine_torch(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction) -> tuple[Tensor, Tensor]`
- `physical_to_grid_affine(M_phys, t_phys, fixed_img, moving_img) -> ndarray`
- `get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32) -> Tensor (1, *shape, dim)`
- `physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction) -> Tensor (*shape, dim)`
- `physical_to_normalized_torch_cached(phys_coords, shape_t, spacing_t, origin_t, direction_t) -> Tensor (*shape, dim)`
- `get_identity_grid_torch(target_shape, device='cpu', dtype=torch.float32) -> Tensor (1, *shape, dim)`
- `compute_autograd_physical_scale(shape_t, spacing_t, device=None, dtype=None) -> Tensor (dim,)`
- `image_to_tensor(img: ANTsImage, device: str = 'cpu', dtype=None) -> Tensor (1, 1, *spatial_zyx)`
- `tensor_to_image(tensor: Tensor | ndarray, ref_image: ANTsImage) -> ANTsImage`
- `reverse_components(arr: Tensor | ndarray) -> Tensor | ndarray`
- `reverse_metadata(spacing, origin, direction) -> tuple[tuple, tuple, ndarray]`
- `itk_shape_to_tensor_shape(shape: tuple) -> tuple`

### `syntx.spatial` ↔ `syntx.transform.SyNToTransform`
- `SyNToTransform` delegates all coordinate grid normalization to `syntx.spatial.physical_to_normalized_torch`
- Identity grid creation delegates to `syntx.spatial.get_identity_grid_torch`
- `_to_physical_displacement` directly calls `syntx.spatial.disp_tensor_to_itk` without redundant `[..., ::-1]` flips
- Affine transform file export delegates to `syntx.spatial.export_ants_affine_transform`

## Code Layout
- `src/syntx/spatial.py`: Single source of truth for all spatial, coordinate, and displacement field domain bridges.
- `src/syntx/transform.py`: High-level `SyNToTransform` class and re-exports of spatial primitives.
- `src/syntx/core/grid.py`: Core grid functions re-exporting from `syntx.spatial`.
- `src/syntx/core/affine.py`: Affine utility functions re-exporting from `syntx.spatial`.
- `src/syntx/syn.py`: Primary SyN registration engine.
- `src/syntx/tvf.py`: Time-varying velocity field registration engine.
- `src/syntx/syngs.py`: Geodesic shooting registration engine.
- `src/syntx/robust_affine.py`: Multi-stage robust affine registration engine.
- `src/syntx/scattered/mapping.py`: Scattered coordinate transformations.
- `src/syntx/scattered/transport.py`: Scattered coordinate transport.
- `src/syntx/__init__.py`: Package entry point exposing `spatial`.
- `tests/test_spatial.py`: Spatial unit tests and roundtrip fidelity tests.
- `tests/test_spatial_roundtrip.py`: Dedicated multi-geometry roundtrip invariance test suite ($L_\infty < 10^{-6}$).
- `tests/test_spatial_centralization.py`: Centralization, aliases, and asymmetric affine regression tests.
- `tests/test_adversarial_spatial_primitives.py`: Adversarial affine and grid transformation tests.
- `tests/test_adversarial_spatial_m1.py`: Adversarial displacement field roundtrip tests.
