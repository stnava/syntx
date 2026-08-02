# Project: syntx Registration Parameter Tuning and Backend Parity

## Architecture
- **PyTorch Registration Engines**: `syntx.syn` (`SyNTo`), `syntx.tvf` (`TVFModel`), `syntx.syngs` (`GeodesicShootingModel`) in `src/syntx/`.
- **JAX Registration Engines**: `syntx.syn_jax` (`SyNToJAX`), `syntx.tvf_jax` (`TVFModelJAX`), `syntx.syngs_jax` (`GeodesicShootingModelJAX`) in `src/syntx/`.
- **Similarity Losses & Guardrails**: Standardized LNCC with variance floor ($\text{Var}_{\text{safe}} = \max(\text{Var}, 10^{-6})$) and Cauchy-Schwarz clamping ($\text{clamp}(\text{cc}, -1.0, 1.0)$), Triplanar VGG 3D Layer 4 LNCC, and Mattes MI in `src/syntx/syn.py` & `src/syntx/syn_jax.py`.
- **Diffeomorphic Solvers & Optimizers**: Sobolev Green's operator frequency preconditioning $K(k)$, LARS scale-invariant trust-ratio velocity field optimizer, EPDiff Euler/RK4 solvers, antisymmetric velocity projection, and bidirectional inverse composition penalties.
- **Diagnostics & Physical Mapping**: Analytical 2D/3D physical Jacobian determinant ($\min \det(J)$) calculation and boundary-masked physical inverse identity error (mm) computation in `src/syntx/syn.py` & `src/syntx/transform.py`.
- **Benchmarking & Reporting Suite**: Benchmark scripts (`examples/debug_2d.py`, `examples/compare_registration_backends_3d.py`, `examples/generate_ants_3d_comparison_report.py`, `examples/benchmark_suite.py`) rendering standard 4-panel visual reports (`render_standard_4panel()`) per GEMINI.md.

## Code Layout
- `src/syntx/syn.py` & `src/syntx/syn_jax.py`: SyN registration engines, loss functions, physical coordinate mapping, analytical Jacobian det, inverse error calculator, and `render_standard_4panel()`.
- `src/syntx/tvf.py` & `src/syntx/tvf_jax.py`: Time-Varying Velocity Field (TVF) registration models, LARS optimizer, B-spline temporal velocity interpolation, and ODE trajectory integrators.
- `src/syntx/syngs.py` & `src/syntx/syngs_jax.py`: SyNGS Geodesic Shooting registration models, dual momentum fields, EPDiff solvers, and Sobolev Green's operator frequency preconditioning.
- `src/syntx/transform.py`: `SyNToTransform` physical-to-normalized warp adapter, Jacobian det generator, and ITK composite NIfTI exporter.
- `src/syntx/image_compare.py`: Unified metric evaluation module supporting 2D/3D inputs.
- `src/syntx/reporting.py`: Visual HTML report generation infrastructure.
- `examples/`: Benchmark scripts (`debug_2d.py`, `compare_registration_backends_3d.py`, `generate_ants_3d_comparison_report.py`, `benchmark_suite.py`, `pairs.csv`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | PyTorch Multi-Model Hyperparameter Tuning | Comprehensive tuning (sigmas, Sobolev alpha/scale, CFL steps, momentum, inverse penalty) for `syn`, `tvf`, `syngs` | M1 | survey 1 |
| 2 | 2D Brain Tissue Dice Alignment | Achieve $\ge 0.8400$ tissue Dice across 2D benchmarks | M1 | survey 3 |
| 3 | 3D Cortical DKT31 Dice Alignment | Achieve $\ge 0.5800$ mean DKT31 label Dice across 3D Mindboggle pairs | M1 | survey 3 |
| 4 | Strict 0.0000% Grid Folding Guardrail | Ensure $\min \det(J) > 0.0$ strictly across all 2D/3D registration models | M1 | survey 1, 3 |
| 5 | Physical Inverse Identity Error Minimization | Keep max inverse identity error $< 0.10$ mm (R1) / $< 0.15$ mm (Acceptance) | M1 | survey 1, 3 |
| 6 | JAX Sobolev Green's Operator Synchronization | Port FFT Sobolev Green's Operator $K(k)$ to `syngs_jax.py`, `tvf_jax.py`, `syn_jax.py` | M2 | survey 2 |
| 7 | JAX Cosine Dirichlet Boundary Tapering | Port `_create_boundary_mask` to JAX velocity fields and gradients | M2 | survey 2 |
| 8 | Fix `vel_spacing` NameError Bug | Fix unassigned `vel_spacing` reference in `syngs_jax.py:468` | M2 | survey 2 |
| 9 | TVF Max CFL Step Cap Parity | Synchronize TVF max CFL step cap to `0.10` in `tvf_jax.py` | M2 | survey 2 |
| 10 | EPDiff RK4 Solver & Velocity Bound Parity | Port RK4 EPDiff ODE solver and spacing-scaled `max_v_phys` to `syngs_jax.py` | M2 | survey 2 |
| 11 | TVF Inverse Identity Penalty Synchronization | Add `inv_id_loss` bidirectional warp penalty to `tvf_jax.py` | M2 | survey 2 |
| 12 | Direction Matrix Inversion Fix | Fix `diff @ inverse(direction_t.t())` in `syn_jax.py:1867` | M2 | survey 2 |
| 13 | JAX LARS Trust-Ratio Optimizer Port | Port LARS optimizer (`LARSJAX` / optax wrapper) to JAX models | M2 | survey 2 |
| 14 | JAX Velocity Parameter Clamping | Port post-step `jnp.clip` velocity clamping to `syngs_jax.py` | M2 | survey 2 |
| 15 | JAX Gaussian Filter Spacing Harmonization | Fix physical spacing and `sigma_mode` parameters in `syngs_jax.py` | M2 | survey 2 |
| 16 | JAX Velocity Grid Resizing Interpolation Parity | Align `align_corners=True` interpolation in velocity grid resizing | M2 | survey 2 |
| 17 | Spatial Gradient Axis & Mask Harmonization | Harmonize image gradient axis ordering and spatial padding mask evaluation | M2 | survey 2 |
| 18 | JAX/PyTorch Delta Dice Parity Verification | Guarantee $| \text{Dice}_{\text{PT}} - \text{Dice}_{\text{JAX}} | \le 0.001$ across backends | M2 | survey 3 |
| 19 | Sobolev-LARS Trust-Ratio Optimizer Tuning | Advanced Sobolev-LARS trust-ratio parameter updates for TVF/SyNGS | M3 | survey 1, 3 |
| 20 | Triplanar Feature Preconditioning | Integrate and tune 3D triplanar VGG Layer 4 LNCC feature preconditioning | M3 | survey 1 |
| 21 | Adaptive CFL Step Controllers | Implement and tune adaptive CFL step size controllers for dynamic speedup | M3 | survey 1 |
| 22 | Standard 4-Panel HTML Visual Reporting | Render `render_standard_4panel()` visual HTML reports for all benchmarks | M4 | survey 3 |
| 23 | E2E Benchmark Suite Execution | Run full 2D and 3D Mindboggle benchmark suite (including `hard_pair_00`) | M4 | survey 1, 3 |
| 24 | Forensic Integrity Audit & Final Acceptance | Independent audit of Single Interpolation Policy, no cheating, and full criteria | M5 | survey 3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | PyTorch Multi-Model Parameter Tuning & Alignment | Comprehensive hyperparameter tuning across `syn`, `tvf`, `syngs` on PyTorch targeting 2D Tissue Dice $\ge 0.8400$, 3D Cortical DKT31 Dice $\ge 0.5800$, $\min \det(J) > 0.0$, max inverse error $< 0.10$ mm, and inter-model Dice parity $\le 1\%$. | none | IN_PROGRESS |
| 2 | Algorithmic & Numerical JAX Backend Parity | Synchronize all 13 audited discrepancies across `syn_jax`, `tvf_jax`, `syngs_jax` targeting $\Delta \text{Dice} \le 0.001$. | M1 | PLANNED |
| 3 | Advanced SOTA Optimization & Final Re-Tuning | Integrate and tune Sobolev-LARS trust-ratio updates, triplanar feature preconditioning, and adaptive CFL step controllers across PyTorch and JAX engines. | M2 | PLANNED |
| 4 | Verification, Benchmarking & HTML Visual Reports | Execute 2D and 3D Mindboggle benchmark suites across PyTorch and JAX backends, generating standard 4-panel visual HTML reports per GEMINI.md. | M3 | PLANNED |
| 5 | Forensic Integrity Audit & Final Verification | Execute independent Forensic Integrity Audit verifying Single Interpolation Policy, no hardcoding, and complete acceptance criteria. | M4 | PLANNED |

## Interface Contracts
- **Single Interpolation Policy**: No intermediate pre-warping of images or segmentations prior to optimization. Multiple transforms must be composed and applied directly to native-space images in a single step.
- **Discrete Label Evaluation**: Label maps must be transformed using nearest-neighbor interpolation (`interpolator='nearestNeighbor'`) and evaluated using `ants.label_overlap_measures()`.
- **LNCC Safeguards**: All LNCC loss implementations must enforce variance floor $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$ and Cauchy-Schwarz clamping $\text{clamp}(\text{cc}, -1.0, 1.0)$.
- **Standard 4-Panel Visual Reports**: All benchmark reports MUST render visual panels using `render_standard_4panel()` from `syntx.syn` (Panel A: Deformed Mesh Grid, Panel B: Divergent Jacobian Det Map, Panel C: Inverse Identity Error Map, Panel D: Canny Edge Alignment Overlap).
- **Backend Parity**: JAX and PyTorch registration results must match within $\Delta \text{Dice} \le 0.001$.
