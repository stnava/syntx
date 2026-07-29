# Implementation Plan — ANTs vs Syntx (JAX & PyTorch) Parity & Benchmark

## Objective
Establish complete technical details, design goals, symmetrical PyTorch implementation, and empirical evaluation results demonstrating numerical and algorithmic parity between ANTs C++ SyN, syntx.syn (JAX backend), and syntx.syn (PyTorch backend on CPU/MPS).

Follow strict workflow order: Evaluation Strategy → Design Goals → Implementation → Empirical Evaluation Results → PyTorch Porting & Parity Verification.

## Milestones

| Milestone | Name | Description | Worker / Subagent | Outputs | Status |
|-----------|------|-------------|-------------------|---------|--------|
| M1 | Evaluation Strategy & Parity Design Goals (R1 & R2) | Define multi-pair 3D evaluation strategy and explicit parity design goals enforcing Single Interpolation Policy, LNCC variance floor max(Var(I), 10^-6), Cauchy-Schwarz [-1, 1] clamping, physical coordinate domain matching, and Anderson Acceleration default inversion (`inverse_method='anderson'`, `inverse_steps=30`). | teamwork_preview_explorer | Parity design document, test goals | DONE |
| M2 | Symmetrical Implementation & PyTorch Porting (R3) | Implement missing features/fixes in syntx.syn, syntx.syn_jax, and syntx.tvf; symmetrically port all verified JAX algorithms and safeguards to PyTorch. Enforce default Anderson Acceleration (`inverse_method='anderson'`, `inverse_steps=30`), LNCC variance floor, Cauchy-Schwarz clamping across backends. | teamwork_preview_worker | Updated `src/syntx/syn.py`, `src/syntx/syn_jax.py`, `src/syntx/tvf.py`, passing `pytest` | DONE |
| M3 | Empirical Benchmark & Structured Results Generation (R4) | Benchmark ANTs Py/C++ SyN vs syntx.syn (JAX) vs syntx.syn (PyTorch) across Mindboggle pairs. Measure Mindboggle DKT Cortical Dice, inverse identity error (||phi_inv o phi_fwd - I||), bending energy (E_2nd), Jacobian determinants, execution runtime. Verify Dice gap <= 0.005. Output `benchmark_results.json`. | teamwork_preview_worker | `benchmark_results.json`, 90 pairs populated, gap=0.00317 <= 0.005 | DONE |
| M4 | Unit Test Verification & Forensic Audit | Run full pytest suite, execute Forensic Auditor integrity verification on GEMINI.md compliance, backend parity, and zero-cheating guardrails. | teamwork_preview_worker + teamwork_preview_auditor | 157 passed pytest, Clean Audit Verdict | DONE |

## Acceptance Criteria
- [x] JAX and PyTorch backends produce Cortical Mindboggle Dice overlap within <= 0.005 of each other across benchmark pairs (PT: 0.60977, JAX: 0.61294, Gap: 0.00317 <= 0.005).
- [x] Both JAX and PyTorch enforce Anderson Acceleration (inverse_method='anderson', inverse_steps=30) as default.
- [x] Single Interpolation Policy (Rule 1) and LNCC variance floor / clamping (Rule 2) are verified symmetrically across all backends.
- [x] All unit tests pass in pytest (157 passed, 0 failures).
- [x] Structured results saved to benchmark_results.json (90/90 pairs valid, all 14 required fields populated).
