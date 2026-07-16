# Project: 3D Registration Parity with Native Physical Space & Affine Composition

## Architecture
- **Coordinate Transformations (`src/syntx/transform.py`)**: Interoperability between normalized coordinates and physical space coordinates.
- **PyTorch Registration (`src/syntx/syn.py`)**: Optimization loop, SyN displacement updates, composition, and physical space transformations in PyTorch.
- **JAX Registration (`src/syntx/syn_jax.py`)**: Optimization loop, JAX-based operations parallel to PyTorch, coordinate grids, and transformations.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Investigation & Planning | Explore current coordinate-mapping logic, PyTorch/JAX SyN optimization steps, and identify discrepancies with ANTs ITK. | none | DONE |
| 2 | Native Physical Space Optimization | Rewrite displacement fields and update steps in PyTorch & JAX to operate in physical mm coordinates. | M1 | IN_PROGRESS |
| 3 | Affine Coordinate Composition | Implement coordinate composition strictly as `y = A(phi_2_inv(phi_1(x)))` in PyTorch & JAX while keeping single interpolation. | M2 | IN_PROGRESS |
| 4 | Verification & Profiling | Re-run 2D and 3D parity tests (including DKT overlaps) and profile execution time. Generate final dashboard/reports. | M3 | PLANNED |

## Interface Contracts
- **Coordinate Mapping**: Displacement fields must be represented in physical mm coordinates.
- **Composition**: Composition must follow `y = A(phi_2_inv(phi_1(x)))`.
- **Single Interpolation**: No intermediate pre-warped inputs allowed. All transforms (affine and SyN) must be composed and applied in a single step to native-space images.
- **Nearest Neighbor**: Evaluation against discrete segmentations must use nearest neighbor interpolation and `ants.label_overlap_measures`.
