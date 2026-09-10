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

## Follow-up — 2026-09-10T21:48:33Z

Address identified compute and memory bottlenecks in `antstorch.weingarten_image_curvature`, `syntx.scattered.solver`, and `syntx.syngs`, profile the rest of the `syntx` codebase for memory churn and compute bottlenecks, and document all findings in a structured efficiency report with strict zero-regression guarantees.

Working directory: `/Users/stnava/code/syntx` (with `/Users/stnava/code/ANTsTorch` for Weingarten module)
Integrity mode: development

## Requirements

### R1. Implement High-Priority Bottleneck Fixes
Address the immediate compute and memory bottlenecks identified in the recent commits:
1. **`antstorch.weingarten_image_curvature`** (`/Users/stnava/code/ANTsTorch`):
   - In `weingarten_image_curvature.py`: Short-circuit Gaussian curvature ($K$), fundamental coefficients ($b, c$), and 8-class topographic categorization when `opt='mean'`.
   - Eliminate per-chunk GPU-to-CPU synchronization barriers (`.cpu().numpy()` inside chunk loop) by pre-allocating the output tensor directly on the GPU device and scattering results in-place.
2. **`syntx.scattered.solver`** (`/Users/stnava/code/syntx`):
   - In `SyNScattered.fit`: Eliminate temporary tensor allocations in the Adam/RegAdam step by using in-place operations (`adam_m_l.mul_().add_()`, `adam_v_l.mul_().addcmul_()`) and factoring scalar bias correction into the step size (removing `m_hat` and `v_hat` allocations).
   - Eliminate redundant `.movedim(-1, 1).contiguous()` restriding in Lagrangian step composition.
   - Optimize in-loop Anderson acceleration frequency (add an interval parameter `in_loop_inv_interval` rather than executing 5 multi-secant steps on both warps every single epoch).
3. **`syntx.syngs`** (`/Users/stnava/code/syntx`):
   - In `GeodesicShootingModel.integrate_velocity`: Streamline numerical integration by avoiding repeated physical-to-normalized coordinate transformations (`physical_to_normalized_torch_cached`) on every intermediate ODE substep and minimizing vector channel permutations.

### R2. Systematic Codebase-Wide Memory & Compute Audit
Conduct a systematic audit of the remaining `syntx` modules (`syn`, `tvf`, `robust_affine`, `spatial`, `smoothing`, `features`, `losses`):
- Identify redundant tensor allocations, unnecessary `.clone()`, `.contiguous()`, or `.movedim()` copies in inner optimization loops.
- Identify un-cached filters or repeated Fourier / Green operator setups.
- Identify hidden CPU-GPU synchronization stalls (`.item()`, `.cpu()`, `np.asarray()` inside hot training loops).
- **Rule**: If other issues (algorithmic, mathematical, or structural) are identified during the review, log them into an efficiency audit tracking table, but **do not address them** in this pass.

### R3. Strict Zero-Regression Invariant
- **CRITICAL**: Do NOT make any changes that lead to regressions in registration accuracy (DICE, folding rates, inverse consistency error).
- All changes must be strictly backward compatible, functionally identical (or mathematically superior within numerical precision), and focused solely on memory and speed efficiency.

## Acceptance Criteria

### Hotspot Optimization Verification
- [ ] `weingarten_image_curvature` executes with $\ge 20\%$ faster chunk processing in `opt='mean'` mode with bitwise or near-identical numerical parity ($r > 0.9999$).
- [ ] All 10 unit tests in `ANTsTorch/tests/test_weingarten_image_curvature.py` pass.
- [ ] `SyNScattered` tests in `tests/test_scattered*.py` pass with reduced memory allocations and zero loss degradation.
- [ ] `SyNGS` reproducibility tests in `tests/test_reproducibility_fast.py` and standard `tests/test_syngs*.py` pass with identical output.

### Codebase Audit Deliverables
- [ ] A structured markdown report (`docs/compute_and_memory_efficiency_audit.md` or artifact) detailing:
  - Benchmark timings before and after hotspot optimizations (runtime, peak memory).
  - Codebase audit cataloging memory churn, synchronization barriers, and cache opportunities across `syn`, `tvf`, `robust_affine`, and `spatial`.
  - Log of identified non-efficiency issues for future tracking without code modifications.
- [ ] Overall test suite passes (`pytest tests/`) with zero regressions.
