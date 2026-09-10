# Project: Performance and Memory Optimization across `syntx` and `ANTsTorch`

## Architecture
- **Multi-Workspace Scope**:
  - `ANTsTorch` (`/Users/stnava/code/ANTsTorch`): High-performance GPU-accelerated extrinsic differential geometric curvature computation (`antstorch.weingarten_image_curvature`).
  - `syntx` (`/Users/stnava/code/syntx`): PyTorch/JAX symmetric diffeomorphic and geodesic registration algorithms (`syntx.scattered.solver`, `syntx.syngs`, `syntx.syn`, `syntx.tvf`, `syntx.robust_affine`).
- **Optimization Philosophy**:
  - Zero-Regression Invariant: Exact float32 or bitwise mathematical parity ($r > 0.9999, \Delta \mathcal{L} \le 10^{-7}$). Zero regression in registration accuracy (DICE, folding rates, inverse error).
  - In-place tensor updates and scalar folding into step size to eliminate ephemeral tensor allocations and allocator churn.
  - Elimination of CPU-GPU synchronization stalls (`.item()`, `.cpu().numpy()` inside loops).
  - Pre-allocation and scattering on device; pre-caching constant geometric operators.
  - Non-efficiency issues (algorithmic, mathematical, structural) logged into an audit tracking table without code modifications.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Weingarten Curvature Short-Circuiting | Short-circuit fundamental coefficients $b, c$, Gaussian curvature $K$, and 8-class categorization when `opt='mean'` | M1 | Survey (R1.1) |
| 2 | Weingarten GPU-to-CPU Sync Elimination | Eliminate 57 per-chunk `.cpu().numpy()` synchronization barriers by pre-allocating output tensor on device and scattering in-place | M1 | Survey (R1.1) |
| 3 | Weingarten Vectorized Normal & Grid Slicing | Move constant tensors outside chunk loop and slice $D^\dagger[1:3]$ to eliminate 33% of shape matrix FLOPs | M1 | Survey (R1.1) |
| 4 | Scattered Adam In-Place Operations | Use `mul_().add_()`, `mul_().addcmul_()`, and factor scalar bias correction into step size to eliminate 24+ temporary allocations/epoch | M2 | Survey (R1.2) |
| 5 | Scattered Lagrangian Restride Elimination | Remove redundant `.contiguous()` on non-contiguous views in `F.grid_sample` and in-place `.sub_()`, eliminating 6 full-volume copies/iteration | M2 | Survey (R1.2) |
| 6 | Scattered Anderson Frequency Parameter | Add `in_loop_inv_interval` parameter to `ScatteredRegistrationConfig` and `SyNScattered` to control in-loop inversion frequency | M2 | Survey (R1.2) |
| 7 | SyNGS Velocity Integration Streamlining | Pre-fuse affine mapping matrix absorbing `torch.flip`, cache normalized identity grid, and eliminate duplicate interpolation in `shoot` | M3 | Survey (R1.3) |
| 8 | Systematic Codebase-Wide Audit | Catalog memory churn, un-cached operators, and synchronization stalls across `syn`, `tvf`, `robust_affine`, `spatial`, `smoothing`, `features`, `losses` | M4 | Survey (R2) |
| 9 | Non-Efficiency Issues Catalog | Log 20+ algorithmic, mathematical, and structural issues into tracking table without code modifications | M4 | Survey (R2) |
| 10 | Structured Efficiency Report | Author `docs/compute_and_memory_efficiency_audit.md` documenting before/after benchmarks, hotspot speedups, and audit findings | M4 | Survey (R2) |
| 11 | Comprehensive Zero-Regression Full Suite Verification | Execute full test suites (`pytest tests/`) across both `syntx` and `ANTsTorch`, ensuring 100% test pass and zero degradation | M5 | Survey (R3) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | Survey & Baseline Profiling | Map codebase bottlenecks, benchmark baselines, design implementation strategies | none | DONE |
| 1 | M1: ANTsTorch Weingarten Curvature Optimization | Implement short-circuiting, GPU pre-allocation, in-place scattering in `weingarten_image_curvature.py`; verify >=20% speedup, $r > 0.9999$, 10/10 tests | M0 | DONE |
| 2 | M2: Scattered Solver Adam & Anderson Optimization | In-place Adam/RegAdam ops, scalar bias step size, Lagrangian movedim restride elimination, `in_loop_inv_interval` in `src/syntx/scattered/solver.py` | M0 | DONE |
| 3 | M3: SyNGS Velocity Integration Streamlining | Pre-fuse coordinate affine map, cache normalized identity grid, eliminate duplicate trilinear sampling in `src/syntx/syngs.py` | M0 | DONE |
| 4 | M4: Codebase Audit & Efficiency Report Delivery | Publish comprehensive `docs/compute_and_memory_efficiency_audit.md` with before/after benchmarks, profiling logs, and non-efficiency tracking table | M1, M2, M3 | DONE |
| 5 | M5: Full Suite Zero-Regression Verification | Run full test suite across `syntx` and `ANTsTorch`, verify zero regressions and complete deliverable checklist | M1, M2, M3, M4 | DONE |

## Interface Contracts
### `antstorch.weingarten_image_curvature`
- `weingarten_image_curvature(image, sigma=1.5, opt='mean', mask=None, chunk_size=100000, device=None) -> ANTsImage`
- `opt='mean'`: Returns scalar mean curvature image $H$. Gaussian curvature $K$ and 8-class categorization are short-circuited.
- `opt='gaussian'`: Returns scalar Gaussian curvature image $K$.
- `opt='characterize'`: Returns integer classification image ($0\dots 8$).
- Preserves exact argument signature, return types, and physical metadata inheritance.

### `syntx.scattered.solver`
- `ScatteredRegistrationConfig.in_loop_inv_interval: int = 1` (default 1 preserves 100% historical parity; values > 1 accelerate multi-secant solving).
- `SyNScattered.__init__(..., in_loop_inv_interval: int = 1, ...)`
- Adam / RegAdam updates in `SyNScattered.fit` mutate pre-allocated `adam_m_l`, `adam_v_l`, `adam_m_r`, `adam_v_r` in-place using `.mul_()`, `.add_()`, `.addcmul_()`.
- Lagrangian step composition in `SyNScattered.fit` operates on non-contiguous grid sampled views directly via `self.warp_l2r.sub_()`.

### `syntx.syngs`
- `GeodesicShootingModel.shoot` retains identical input signature and return type (`Tensor` displacement field).
- Internal ODE integration calculates normalized grid via linear projection $u_{\text{id}} + (\Delta \cdot M_{\text{norm}})$ avoiding repeated calls to `physical_to_normalized_torch_cached` inside intermediate RK substeps.

## Code Layout
- ANTsTorch:
  - `/Users/stnava/code/ANTsTorch/antstorch/utilities/weingarten_image_curvature.py`: Core Weingarten curvature implementation.
  - `/Users/stnava/code/ANTsTorch/tests/test_weingarten_image_curvature.py`: 10 unit tests for Weingarten curvature.
- syntx:
  - `src/syntx/scattered/solver.py`: `SyNScattered` solver and registration config.
  - `src/syntx/syngs.py`: `GeodesicShootingModel` and geodesic velocity integration.
  - `docs/compute_and_memory_efficiency_audit.md`: Efficiency report deliverable.
  - `tests/test_scattered*.py`: Scattered registration test suite (125+ tests).
  - `tests/test_syngs*.py`: SyNGS registration test suite.
  - `tests/test_reproducibility_fast.py`: PyTorch/JAX reproducibility tests.
