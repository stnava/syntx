# Project: syntx Registration Optimization and Validation Benchmark

## Architecture
- **Compute Backends**: PyTorch (`src/syntx/tvf.py`, `src/syntx/shooting.py`, `src/syntx/syn.py`) and JAX (`src/syntx/tvf_jax.py`, `src/syntx/shooting_jax.py`, `src/syntx/syn_jax.py`) compute engines executing TVF, Geodesic Shooting, and SyN registration algorithms with mathematical parity.
- **Central Spatial Conversion Suite**: `src/syntx/spatial.py` converting physical ANTs/ITK space coordinates and displacement fields to/from model tensor space (`disp_tensor_to_itk`, `disp_itk_to_tensor`, `jacobian_determinant`).
- **Standardized Visualization & Reporting**: `src/syntx/visualization.py` and `src/syntx/reporting.py` executing `render_standard_4panel` figures matching `ants.plot` standard display orientation and `create_registration_report` producing interactive HTML benchmark reports.
- **Benchmarking Suite**: Comprehensive 2D (`r16`/`r64`) and 3D (Mindboggle pair 08 `NKI-TRT-20-2` to `MMRR-21-2`) execution validating MSE discrepancy $\le 0.001$, Cortical Dice, Jacobian det(J) range, and execution runtime.

## Code Layout
- `src/syntx/spatial.py`: Central ITK/ANTs physical coordinate and displacement field conversion suite.
- `src/syntx/visualization.py`: 2D/3D visualization functions adhering to `ants.plot` orientation invariants.
- `src/syntx/reporting.py`: HTML report generation and 4-panel image rendering (`create_registration_report`).
- `src/syntx/syn.py` & `src/syntx/syn_jax.py`: PyTorch and JAX SyN registration optimization.
- `src/syntx/tvf.py` & `src/syntx/tvf_jax.py`: PyTorch and JAX Time-Varying Velocity Field (TVF) registration.
- `src/syntx/shooting.py` & `src/syntx/shooting_jax.py`: PyTorch and JAX Geodesic Shooting registration models.
- `examples/`: Benchmark scripts and parameter exploration drivers (e.g. `examples/run_benchmark_2d.py`, `examples/run_benchmark_3d.py`).
- `docs/reports/`: Generated standalone interactive HTML registration reports.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Visualization & Orientation Parity | Ensure 2D/3D images, grids, and Jacobians match `ants.plot` standard display orientation without axis transposition or component inversion. | M1 | survey (explorer_1) |
| 2 | Rule 13 Displacement Export Fix | Delegate displacement field NIfTI export in `src/syntx/syn.py` to `syntx.spatial.disp_tensor_to_itk` instead of ad-hoc inline `[..., ::-1]`. | M1 | survey (explorer_1) |
| 3 | TVF Border Padding (Rule 8) | Enforce `padding_mode='border'` in PyTorch `TVFModel.integrate()` for displacement fields to prevent zero-clamping boundary vectors. | M2 | survey (explorer_2) |
| 4 | Antisymmetric Velocity Projection (Rule 11) | Apply `0.5 * (grad - flip(grad))` antisymmetric projection in PyTorch `TVFModel.fit()` before fluid smoothing to eliminate velocity drift. | M2 | survey (explorer_2) |
| 5 | JAX GeodesicShootingModel | Implement JAX `GeodesicShootingModelJAX` in `src/syntx/shooting_jax.py` matching PyTorch `GeodesicShootingModel` algorithmically. | M2 | survey (explorer_2) |
| 6 | JAX ODE Keyframe Interpolation Fix | Pre-upsample keyframes before RK4/Euler integration loop in `TVFModelJAX.integrate()` to eliminate redundant interpolations. | M2 | survey (explorer_2) |
| 7 | PyTorch & JAX Parity Verification | Enforce $\le 0.001$ MSE discrepancy and loss parity across PyTorch and JAX backends for `TVFModel` and `GeodesicShootingModel`. | M2 | survey (explorer_2) |
| 8 | CFL-like Step Condition | Implement voxel spacing multiplier `step * spacing` per axis (Rule 6) for gradient updates in TVF and Geodesic Shooting models. | M3 | survey (explorer_3) |
| 9 | Zero Regularization Weight (`reg_weight=0.0`) | Enforce `reg_weight=0.0` across similarity-driven TVF and Geodesic Shooting models under fixed 3-level pyramid `[4, 2, 1]` and LNCC loss. | M3 | survey (explorer_3) |
| 10 | Optimizer & Sigma Space Sweeps | Evaluate LARS vs Adam vs SGD optimizers and compare voxel space vs physical space Gaussian smoothing sigmas ($\sigma = \sqrt{\text{variance}}$). | M3 | survey (explorer_3) |
| 11 | 2D Benchmarking (`r16`/`r64`) | Run registration benchmark on 2D `r16` and `r64` datasets, demonstrating equal/superior performance vs `syntx.syn` baseline. | M4 | survey (explorer_3) |
| 12 | 3D Benchmarking (Mindboggle Pair 08) | Run registration benchmark on 3D Mindboggle pair 08 (`NKI-TRT-20-2` to `MMRR-21-2`), verifying Cortical Dice and det(J) > 0. | M4 | survey (explorer_3) |
| 13 | Automated HTML Report Generation | Generate interactive HTML reports for all benchmark runs using `syntx.reporting.create_registration_report`. | M4 | survey (explorer_3) |
| 14 | Forensic Audit & Release Pipeline | Perform forensic integrity audit via `teamwork_preview_auditor`, commit git checkpoints, bump major version, and tag release. | M5 | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Visualization Parity & Export Fix | Verify `ants.plot` display orientation compliance; replace ad-hoc `[..., ::-1]` in `syn.py` with `disp_tensor_to_itk`. | none | DONE |
| 2 | PyTorch & JAX Backend Synchronization | Fix PyTorch border padding & antisymmetric projection; implement `GeodesicShootingModelJAX`; achieve $\le 0.001$ MSE backend parity. | M1 | DONE |
| 3 | Parameter Sweeps & CFL Optimization | Implement `reg_weight=0.0`, CFL `step * spacing` condition, optimizer sweeps (LARS/Adam/SGD), and voxel vs physical sigma space evaluation. | M2 | DONE |
| 4 | 2D & 3D Registration Benchmarks | Run benchmarks on 2D `r16`/`r64` and 3D Mindboggle pair 08; generate HTML reports via `create_registration_report`. | M3 | IN_PROGRESS |
| 5 | Forensic Audit & Major Version Release | Execute `teamwork_preview_auditor`, commit git checkpoints, bump major version, tag release, and deliver final completion report. | M4 | PLANNED |

## Interface Contracts
- **`syntx.spatial.disp_tensor_to_itk` & `disp_itk_to_tensor`**: Must be used for all tensor-to-ITK and ITK-to-tensor displacement field conversions. Inline component swapping (`[..., ::-1]`) is strictly forbidden (Rule 13).
- **Display Orientation**: 2D images use `arr.T`, 2D vector fields use `np.transpose(arr[..., :2], (1, 0, 2))`, 3D volumes reorient to LAI.
- **Backend Parity**: PyTorch and JAX implementations of TVF and Geodesic Shooting models must match within $\le 0.001$ MSE discrepancy on identical inputs.
- **LNCC Variance Floor & Cauchy-Schwarz**: Enforce $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$ and `clamp(cc, -1.0, 1.0)` across all backends.
- **CFL Step Scaling**: ITK gradient steps must be scaled in voxel space by physical spacing (`step * spacing`).
