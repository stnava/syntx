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

## Follow-up — 2026-09-11T02:38:25Z

Remediate memory misuse, ephemeral allocation churn, CPU-GPU synchronization stalls, and operator recalculations across core registration engines (`syntx.syn`, `syntx.tvf`, `syntx.core.smoothing`, `syntx.core.inverse`), guaranteeing zero registration accuracy regressions.

Working directory: `/Users/stnava/code/syntx`
Integrity mode: development

## Requirements

### R1. Remediate Memory Churn & Ephemeral Allocations in `syntx.syn` and `syntx.tvf`
- In `syntx.syn`: Convert Adam and RegAdam moment updates to in-place operations (`adam_m.mul_().add_()`, `adam_v.mul_().addcmul_()`) and factor scalar bias correction into the effective step size, eliminating ephemeral `m_hat` and `v_hat` tensor allocations in the primary registration loops.
- In `syntx.syn` and `syntx.tvf`: Eliminate redundant full-volume layout restriding copies (`.movedim(-1, 1).contiguous()`) during Eulerian composition and velocity pullback transformations.
- Eliminate any un-detached tensor retention or graph leaks across optimization loops and history tracking.

### R2. Eliminate CPU-GPU Synchronization Barriers in `syntx.core.inverse` and `syntx.tvf`
- In `syntx.core.inverse` (`update_inverse_field_nd_anderson`): Replace blocking per-iteration scalar conversions (`float(scaled_norm.max())`) with on-device tensor reductions or decoupled error check intervals to prevent GPU pipeline serialization stalls during in-loop inversion.
- In `syntx.tvf` (`TVFModel.integrate`): Eliminate blocking `.item()` queries in the inner integration loop for CFL conditions, evaluating bounds upfront or via on-device operations.

### R3. Stationary Operator Pre-Computation & Caching in `syntx.core.smoothing`
- In `syntx.core.smoothing`: Implement thread-safe LRU/dictionary caching for Discrete Sine Transform (DST-I) Green operator eigenvalues ($K_{\text{dst}}$) keyed by spatial grid shape, spacing, device, and dtype, avoiding dynamic trigonometric meshgrid re-allocations on every smoothing pass.

### R4. Strict Zero-Regression Invariant
- **CRITICAL**: Do NOT make any changes that lead to regressions in registration accuracy (DICE scores, topological folding percentages $\det(J) \le 0$, and real physical inverse consistency error).
- All changes must be strictly backward compatible and focused solely on memory and compute efficiency.
- Any non-efficiency algorithmic or structural issues encountered during review must be recorded in an audit tracking log without modifying the surrounding logic.

## Acceptance Criteria

### Performance & Memory Optimization Verification
- [ ] Adam moment updates in `syntx.syn` execute in-place with zero ephemeral `m_hat` and `v_hat` allocations, demonstrating measured memory allocation reduction in 3D registration loops.
- [ ] `update_inverse_field_nd_anderson` in `syntx.core.inverse` executes with reduced host-accelerator synchronization stalls while maintaining identical inverse convergence and stopping behavior.
- [ ] Green operator smoothing in `syntx.core.smoothing` reuses cached eigenvalue tensors across successive calls with matching geometry, eliminating repetitive trigonometric grid allocations.
- [ ] TVF velocity field integration avoids blocking `.item()` calls inside inner numerical integration loops.

### Robustness & Regression Verification
- [ ] Bitwise or near-bitwise numerical parity ($L_\infty < 10^{-6}$, relative error $< 10^{-5}$) verified against baseline implementations for all modified routines.
- [ ] All existing regression and reproducibility tests (`tests/test_reproducibility_fast.py`, `tests/test_scattered_syn.py`, `tests/test_syngs_parity.py`) pass with 100% success.
- [ ] New dedicated adversarial tests verify zero accuracy regressions in deformation fields, Jacobian determinants, and inverse error.
- [ ] Non-efficiency observations are appended to `docs/compute_and_memory_efficiency_audit.md` tracking log.



## 2026-10-03T20:02:56Z

An orchestrator coordinating a software engineer, a minimalist, and a Brian Avants / style consultant.

Exploratory planning to rigorously evaluate, prune, and refine future registration robustness improvements for the `syntx` library, synthesizing a lean, non-duplicative, and mathematically sound architectural plan.

Working directory: /Users/stnava/data/repos/syntx
Integrity mode: development

## Requirements

### R1. Persona-Driven Exploratory Review of Robustness Candidates
Review the proposed registration robustness extensions from four distinct perspectives:
- **Orchestrator**: Synthesizes trade-offs, drives consensus, and structures the final actionable roadmap.
- **Software Engineer**: Evaluates architectural fit, maintainability, testability, code duplication, and runtime performance in PyTorch/JAX.
- **Minimalist**: Actively prunes non-essential bloat, eliminates redundant mechanisms (e.g. scrutinizing whether step-size bounding or QC duplicate existing smoothing, velocity clamps, or deformation metrics), and keeps the core registration engine lean.
- **Brian Avants / style consultant**: Enforces ANTs/ITK registration theory, physical space invariants (ITK LPS coordinates, direction matrices), metric physics (Mattes MI, CC2, MIND), diffeomorphism properties, and ANTs idioms.

### R2. Critical Evaluation of Specific Robustness Proposals
Specifically evaluate:
1. **Multi-Modal Metric Parity in `syntx.greedy` (`mattes_mi`)**: Integrating `mattes_mi_loss_nd` into `GreedyRegistrationModel` to support cross-modal and inverted-contrast registrations.
2. **Dense MIND / Self-Similarity Loss**: Adding 3D dense MIND-SSC to `core/losses.py` vs relying on multi-channel / landmark guidance.
3. **Spatial Masking Support (`fixed_mask`, `moving_mask`)**: Implementing true masked similarity across pyramid levels to isolate anatomy of interest.
4. **Principal Axes / Orientation Initialization**: Analyzing failure modes and fragility of spatial moment tensors vs discrete cone angles / header permutations, establishing safe guardrails.
5. **CFL / Diffeomorphism Step Bounding**: Determining if explicit CFL step bounding is redundant with existing velocity/total smoothing sigmas (`flow_sigma`, `total_sigma`) or if a minimal velocity norm cap is needed.
6. **Automated QC & Self-Healing**: Evaluating whether post-hoc QC duplicates existing evaluation utilities (`deformation_metrics.py`, `image_compare.py`) or should be a lightweight diagnostic wrapper.

### R3. Output Deliverable: Unified Actionable Plan
Produce a consolidated, prioritized `/plan` markdown document detailing what should be built, what must be rejected as bloat or duplicative, and exact technical specifications for approved features.

## Acceptance Criteria

### Architectural Integrity & Non-Duplication
- [ ] Clear justification for every accepted feature showing it does not duplicate existing functionality in `syntx`.
- [ ] Explicit rejection or scoping down of features identified as bloat, redundant, or overly fragile.
- [ ] Strict compliance with ANTs physical space standards and ITK LPS invariants.

### Concrete Implementation Specifications
- [ ] Exact mathematical formulation and API signatures for all approved features.
- [ ] Test strategy and failure-mode analysis for delicate components (e.g. orientation handling).
- [ ] Prioritized implementation phases with estimated complexity.


## 2026-10-04T19:59:37Z

Use a team of specialized agents per loss family.

Centralize all image similarity loss implementations in syntx into a canonical, high-performance losses core (`syntx.core.losses`), eliminating ad-hoc reimplementations across syn, greedy, robust_affine, syngs, and tvf, while maximizing numerical soundness, gradient correctness, and GPU/MPS compute efficiency.

Working directory: /Users/stnava/data/repos/syntx
Integrity mode: development

## Requirements

### R1. Centralize All Loss Functionals in `syntx.core.losses`
Consolidate all similarity metric families into a single canonical source of truth in `syntx.core.losses`:
- **Cross-Correlation Family**: `local_ncc_loss_nd`, `box_lncc_loss_nd`, `box_cc2_loss_nd`, `AnalyticalLNCC`, `ANTsPseudoLNCC`.
- **Mutual Information Family**: `CanonicalMattesMIFunction`, `mattes_mi_loss_nd`, `mattes_mi_loss_core`, `parzen_weights`.
- **Pairwise & Pointwise Family**: Standardized `mse_loss_nd`, `l2_loss_nd`, and `mae_loss_nd` with uniform masked reduction.
- **Structural & Label Family**: `soft_dice_loss_nd`, `compute_soft_distance_transform`, `distance_transform_loss`.
- **Deep Feature Family**: Unified `FeatureSpaceLoss` interfacing cleanly with the canonical loss dispatcher.
Remove all duplicate loss math, inline metric computations, or diverging alias parsing in `syn.py`, `greedy.py`, `robust_affine.py`, `syngs.py`, and `tvf.py`.

### R2. Numerical Soundness & GPU/MPS Memory Efficiency
Every canonical loss functional must be computationally optimized for full-scale 3D medical volumes (e.g. $176 \times 256 \times 256 \approx 11.5\text{M}$ voxels):
- Eliminate multi-gigabyte intermediate tensor allocations and memory leaks.
- Guarantee bitwise deterministic accumulation across memory allocations on Apple Silicon MPS, CUDA, and CPU (per GEMINI.md Rules 1 & 4).
- Protect against float16 AMP overflow (max 65,504) and enforce strict variance floors ($\ge 10^{-6}$ for LNCC) to prevent NaNs.

### R3. Gradient Accuracy & Verification
Every loss calculation must support accurate, robust backpropagation:
- Analytical pointwise gradients or custom `torch.autograd.Function` backprop must match numerical differentiation via `torch.autograd.gradcheck`.
- Provide gradient cosine similarity $\ge 0.999999$ against reference autograd.
- Zero loss of gradient flow through coordinate sampling grids or image detachment boundaries.

### R4. Backend Parity & Driver Parity
Maintain strict algorithmic and parameter parity across compute backends (PyTorch and JAX) per GEMINI.md Rule 2:
- Uniform alias parsing for all similarity metric strings across all drivers (`'cc2'`, `'lncc'`, `'mattes'`, `'mattes_mi'`, `'mse'`, `'soft_dice'`).
- JAX implementations in `tvf_jax.py` / `syn_jax.py` must either wrap the canonical mathematical formulations or maintain exact parameter parity with the PyTorch core.

## Acceptance Criteria

### Canonical Consolidation
- [ ] No registration driver (`src/syntx/syn.py`, `src/syntx/greedy.py`, `src/syntx/robust_affine.py`, `src/syntx/syngs.py`, `src/syntx/tvf.py`) contains private, duplicated loss calculation loops; all call `syntx.core.losses`.
- [ ] All registration drivers uniformly parse metric aliases, bin counts, and kernel radii using a shared loss parser in `syntx.core.losses`.

### Numerical Accuracy & Verification
- [ ] Every similarity metric passes `torch.autograd.gradcheck` with float64.
- [ ] Autograd and analytical gradients achieve $\ge 0.999999$ cosine similarity against float64 numerical references.
- [ ] AMP autocast stress test confirms no NaN or infinite gradients occur on large ($> 10^6$ voxel) tensors.

### Performance & Memory Guardrails
- [ ] Peak transient memory allocation for 3D volumes ($176 \times 256 \times 256$) does not exceed $100\text{ MB}$ during similarity loss evaluation.
- [ ] Joint histogram and convolution pooling operations remain bitwise deterministic across repeated runs and allocations on MPS.

### Test Suite Integrity
- [ ] Full pytest test suite across `tests/` passes with 100% success and zero regressions.
