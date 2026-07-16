# BRIEFING — 2026-07-15T00:04:40-04:00

## Mission
Verify correctness of new optimizers, test suite pass/coverage, optimizer sweeps parity, and identify numerical instabilities or displacement shape issues.

## 🔒 My Identity
- Archetype: Empirical Challenger
- Roles: critic, specialist
- Working directory: /Users/stnava/code/syntx/.agents/challenger_perf_1
- Original parent: 1180e7e5-162a-48f5-8ce1-0055a53bf6d8
- Milestone: Optimizer Verification
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only — do NOT modify implementation code
- Run verification code directly and do not trust unverified claims

## Current Parent
- Conversation ID: 1180e7e5-162a-48f5-8ce1-0055a53bf6d8
- Updated: 2026-07-15T00:04:40-04:00

## Review Scope
- **Files to review**: `tests/test_optimizers.py`, `examples/run_optimizer_sweeps.py`, and codebase for optimizers
- **Interface contracts**: /Users/stnava/code/syntx/PROJECT.md or equivalent
- **Review criteria**: correctness, stability, parity within 1% of ANTs registration

## Attack Surface
- **Hypotheses tested**:
  - PyTorch and JAX optimizers function correctly without NaNs. (Confirmed, all sweeps complete successfully).
  - PyTorch L-BFGS suffers from folding because physical constraints (Dirichlet boundary, elastic/fluid smoothing, diffeomorphic projection) are applied outside of the inner line-search loop of `optimizer.step(closure)`. (Confirmed by 2D/3D folding rates of >1.3% in PyTorch vs 0% in JAX/SciPy).
  - The baseline parity verification failure in `run_optimizer_sweeps.py` is due to comparing ANTs default LNCC (CC) registration with PyTorch Mattes MI registration. (Confirmed: ANTs CC dice is 0.4370, PyTorch CFL LNCC dice is 0.4300, matching within 0.70%, whereas PyTorch CFL Mattes MI is 0.3956, leading to a 4.14% difference).
- **Vulnerabilities found**:
  - PyTorch L-BFGS exhibits high folding rates (up to 18.5% in 2D, 1.3-3.9% in 3D) and low Dice (~0.12 in 3D) due to parameter updates violating constraints during line search.
  - SGD under-optimizes with default learning rates (retains initial rigid Dice ~0.398).
- **Untested angles**:
  - Higher learning rates or longer iterations for SGD.
  - Custom line search configuration or constraining parameters inside PyTorch's L-BFGS closure.

## Loaded Skills
- **Source**: none loaded

## Key Decisions Made
- Confirmed test success (8/8 on test_optimizers.py, 103/103 total tests passed, 91% coverage).
- Identified the source of the baseline parity verification failure and the L-BFGS folding vulnerability.

## Artifact Index
- `/Users/stnava/code/syntx/.agents/challenger_perf_1/handoff.md` — Verification and challenge report
