# Project: Generalized Scattered Data Diffeomorphic Registration in syntx

## Architecture
- Subpackage: `src/syntx/scattered/`
  - `projection.py`: Differentiable Nadaraya-Watson kernel regression, `project_scattered_to_grid`, `ScatteredProjector`, `ProjectionConfig`, and backward-compatible `differentiable_grid_projection`.
  - `mapping.py`: Differentiable field evaluation at scattered coordinates, forward/backward coordinate warping (`warp_scattered_coordinates`, `evaluate_field_at_scattered`).
  - `transport.py`: Feature pullback (`pullback_grid_to_scattered`), pushforward (`pushforward_scattered_to_grid`), and point-to-point transport (`transport_scattered_to_scattered`).
  - `solver.py`: `SyNScattered(nn.Module)` and `syn_scattered` functional solver integrating Anderson acceleration, fluid regularisation, CFL step bounding, and analytical LNCC pseudo-gradients.
  - `__init__.py`: Public exports for `syntx.scattered` subpackage and top-level registration in `src/syntx/__init__.py`.
- Reused Core Modules:
  - `src/syntx/core/inverse.py`: `update_inverse_field_nd_anderson`
  - `src/syntx/core/smoothing.py`: `apply_dsti_green_operator`, `separable_gaussian_filter`, `apply_sobolev_green_operator`
  - `src/syntx/core/losses.py`: `AnalyticalLNCC`, `ANTsPseudoLNCC`, `local_ncc_loss_nd`
  - `src/syntx/core/jacobian.py`: `compute_physical_jacobian_determinant`
  - `src/syntx/core/optimizers.py`: `RegAdam`, `LARS`, CFL step scaling

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1: d-D Nadaraya-Watson Regression | Differentiable projection mapping points $\{x_i\} \subset \mathbb{R}^d$ with features $\{f_i\} \subset \mathbb{R}^C$ onto Eulerian grid lattice | M1 | survey_2 / R1 |
| 2 | F2: Vectorized GEMM & Auto-Chunking | $D^2 = N_Y - 2\tilde{Y}\tilde{X}^T + N_X$ without 3D tensor allocations; auto-chunking memory ceiling $\le 256$ MB | M1 | survey_2 / R1 |
| 3 | F3: Arbitrary Domain Bounds | Support for arbitrary bounding boxes $[b_{\min}, b_{\max}]$, unit domain $[-1, 1]$, and auto-bounding with $3\sigma$ padding | M1 | survey_2 / R1 |
| 4 | F4: Eulerian Masking & Weights | Continuous/binary domain mask integration and point confidence weights | M1 | survey_2 / R1 |
| 5 | F5: Autograd Differentiability | Analytical gradient propagation to $X$ and $F$ without NaNs or numerical explosion | M1 | survey_2 / R1 |
| 6 | F6: Cached ScatteredProjector | Reusable PyTorch module caching Eulerian grid geometry and norms for iterative SyN loops | M1 | survey_2 / R1 |
| 7 | F7: Drop-in Compatibility Alias | Drop-in wrapper `differentiable_grid_projection` matching consumer signature | M1 | survey_2 / R1 |
| 8 | F8: Field Evaluation at Points | Differentiable Eulerian displacement field evaluation at scattered points via singleton grid query | M2 | survey_3 / R3 |
| 9 | F9: Bidirectional Coordinate Warping | Forward $\phi(x) = x + u(x)$ and backward $\phi^{-1}(y) = y + v(y)$ coordinate warping | M2 | survey_3 / R3 |
| 10 | F10: Grid-to-Point Pullback | Differentiable pullback $\Phi^* G_B(x) = G_B(x + u(x))$ of Eulerian grid features to points | M2 | survey_3 / R3 |
| 11 | F11: Point-to-Grid Pushforward | Differentiable pushforward $\Phi_* F_A$ of scattered features to Eulerian grid via warped Nadaraya-Watson | M2 | survey_3 / R3 |
| 12 | F12: Point-to-Point Transport | Direct Lagrangian transport of features between two scattered point sets through diffeomorphism | M2 | survey_3 / R3 |
| 13 | F13: SyNScattered Solver | Symmetric diffeomorphic registration solver for scattered-to-scattered and scattered-to-grid | M3 | survey_1 / R2 |
| 14 | F14: Anderson Inversion Integration | In-loop Anderson fixed-point acceleration (`update_inverse_field_nd_anderson`) | M3 | survey_1 / R2 |
| 15 | F15: Fluid Velocity Regularization | Velocity smoothing using DST-I Green operator (`apply_dsti_green_operator`), Sobolev, or Gaussian | M3 | survey_1 / R2 |
| 16 | F16: Total Field Composition & CFL | Lagrangian pullback step composition and CFL step bounding preventing grid folding | M3 | survey_1 / R2 |
| 17 | F17: Analytical LNCC Pseudo-Gradients | Integration with `AnalyticalLNCC` / `ANTsPseudoLNCC` for $O(1)$ memory metric computation | M3 | survey_1 / R2 |
| 18 | F18: Multi-Resolution Pyramids | Coarse-to-fine multi-resolution grid hierarchy support | M3 | survey_1 / R2 |
| 19 | F19: Unit Tests for Projection | `tests/test_scattered_projection.py` verifying shapes, autograd gradcheck, masks, bounds | M1 | survey_3 / R4 |
| 20 | F20: Unit Tests for Mapping | `tests/test_scattered_mapping.py` verifying warping, field evaluation, and Anderson consistency ($< 10^{-3}$) | M2 | survey_3 / R4 |
| 21 | F21: Unit Tests for Transport | `tests/test_scattered_pullback_pushforward.py` verifying pullback, pushforward, and cycle consistency | M2 | survey_3 / R4 |
| 22 | F22: SyN Solver & Benchmark Tests | `tests/test_scattered_syn.py` and `tests/test_scattered_benchmarks.py` verifying convergence, folding $< 0.1\%$ | M3/M4 | survey_3 / R4 |
| 23 | F23: Zero Regression Suite | Full suite execution (`pytest tests/`) with 0 failures across existing 423 tests | M4 | survey_3 / R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Differentiable Scattered-to-Grid Projection | `src/syntx/scattered/projection.py`, `tests/test_scattered_projection.py` (F1-F7, F19) | none | DONE |
| M2 | Bidirectional Coordinate Mapping & Feature Transport | `src/syntx/scattered/mapping.py`, `src/syntx/scattered/transport.py`, `tests/test_scattered_mapping.py`, `tests/test_scattered_pullback_pushforward.py`, `tests/test_scattered_batch_challenge.py` (F8-F12, F20-F21) | M1 | DONE |
| M3 | Scattered Diffeomorphic SyN Registration Solver | `src/syntx/scattered/solver.py`, `src/syntx/scattered/__init__.py`, `src/syntx/__init__.py`, `tests/test_scattered_syn.py` (F13-F18, F22-solver) | M1, M2 | DONE |
| M4 | Comprehensive Verification Suite & Non-Regression | `tests/test_scattered_benchmarks.py`, full suite regression `pytest tests/` (F22-benchmarks, F23) | M1, M2, M3 | DONE |
| M-Remedy | Victory Audit Remediation & Zero Regression Certification | `src/syntx/core/smoothing.py`, `src/syntx/transform.py`, `tests/test_bulletproof_ants_parity.py`, `tests/test_reproducibility_fast.py` | M4 | DONE |

## Interface Contracts
### `syntx.scattered.projection`
- `project_scattered_to_grid(points, values, grid_shape=64, domain_bounds=(-1.0, 1.0), sigma=0.03, mask=None, point_weights=None, epsilon=1e-8, chunk_size=0, target_memory_mb=256.0, coord_convention='xyz', fill_value=0.0, return_density=False, config=None) -> Tensor | (Tensor, Tensor)`
- `ScatteredProjector(grid_shape=64, domain_bounds=(-1.0, 1.0), sigma=0.03, mask=None, coord_convention='xyz', device=None, dtype=torch.float32, config=None)`: `forward(points, values, point_weights=None, return_density=False)`
- `ProjectionConfig`: Dataclass containing grid configuration parameters.
- `differentiable_grid_projection(u_coords, values, grid_res=64, sigma=0.03, epsilon=1e-8, nw_chunk_size=0, grid_bounds=(-1.0, 1.0), return_density=False) -> Tensor | (Tensor, Tensor)`

### `syntx.scattered.mapping`
- `evaluate_field_at_scattered(field: Tensor, coords: Tensor, mode='bilinear', padding_mode='border', align_corners=True) -> Tensor`
- `warp_scattered_coordinates(coords: Tensor, displacement_field: Tensor, direction='forward', mode='bilinear', domain_bounds=None) -> Tensor`

### `syntx.scattered.transport`
- `pullback_grid_to_scattered(grid_features: Tensor, coords: Tensor, displacement_field: Tensor, mode='bilinear', align_corners=True) -> Tensor`
- `pushforward_scattered_to_grid(coords: Tensor, features: Tensor, displacement_field: Tensor, grid_shape: tuple, sigma=0.03, domain_mask=None, domain_bounds=None) -> Tensor`
- `transport_scattered_to_scattered(coords_source: Tensor, features_source: Tensor, coords_target: Tensor, displacement_field: Tensor, sigma=0.03) -> Tensor`

### `syntx.scattered.solver`
- `SyNScattered(dim=2, grid_res=128, domain_bounds=(-1.0, 1.0), sigma=0.03, fluid_sigma=1.5, elastic_sigma=0.0, regularizer='dsti1', optimizer_type='rprop', in_loop_inv_steps=5, inverse_steps=20, inverse_method='anderson', cfl_voxels=0.25, use_analytical_gradients=False)`
- `syn_scattered(fixed_points, fixed_features, moving_points, moving_features, config=None, **kwargs) -> ScatteredRegistrationResult`

## Code Layout
- `src/syntx/scattered/`
  - `__init__.py`
  - `projection.py`
  - `mapping.py`
  - `transport.py`
  - `solver.py`
- `tests/`
  - `test_scattered_projection.py`
  - `test_scattered_mapping.py`
  - `test_scattered_pullback_pushforward.py`
  - `test_scattered_syn.py`
  - `test_scattered_benchmarks.py`
