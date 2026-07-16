# BRIEFING — 2026-07-15T04:06:23Z

## Mission
Verify the correctness of new optimizers and check for baseline parity with ANTs registration within 1%.

## 🔒 My Identity
- Archetype: Empirical Challenger (critic)
- Roles: critic, specialist
- Working directory: /Users/stnava/code/syntx/.agents/challenger_perf_2/
- Original parent: 1180e7e5-162a-48f5-8ce1-0055a53bf6d8
- Milestone: Optimizer Verification
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only — do NOT modify implementation code
- CODE_ONLY network mode
- Single Interpolation Policy (from GEMINI.md)
- Similarity Metric & VGG Feature Space Guidelines (from GEMINI.md)

## Current Parent
- Conversation ID: 1180e7e5-162a-48f5-8ce1-0055a53bf6d8
- Updated: not yet

## Review Scope
- **Files to review**: `tests/test_optimizers.py`, `examples/run_optimizer_sweeps.py`
- **Interface contracts**: /Users/stnava/code/syntx/GEMINI.md
- **Review criteria**: correctness, numerical stability, shape conformance, ANTs baseline parity within 1%.

## Key Decisions Made
- Executed unit tests (`pytest tests/test_optimizers.py`), complete suite (`pytest --cov=src`), and parameter sweeps (`python examples/run_optimizer_sweeps.py`).
- Isolated PyTorch L-BFGS instability to regularisation exclusion in the line-search closure.
- Identified that static learning rate `1e-2` leaves SGD completely inactive.

## Attack Surface
- **Hypotheses tested**: 
  - Hypothesis: Syntx PyTorch CFL Mattes MI matches ANTs Mattes MI baseline within 1%. (Result: FAILED. 3.67% difference).
  - Hypothesis: PyTorch and JAX L-BFGS optimizers perform identically. (Result: FAILED. PyTorch L-BFGS collapses to Dice ~0.12 and high folding, JAX L-BFGS is stable).
- **Vulnerabilities found**:
  - Missing Gaussian smoothing in image downsampling for PyTorch/JAX image pyramids.
  - PyTorch L-BFGS closure does not perform elastic smoothing or double-inversion, causing collapse during line search.
- **Untested angles**: Tuning alternative learning rates or parameters for SGD/Adam.

## Loaded Skills
- None

## Artifact Index
- /Users/stnava/code/syntx/.agents/challenger_perf_2/handoff.md — Verification report
