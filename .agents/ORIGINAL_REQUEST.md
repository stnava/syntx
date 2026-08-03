# Original User Request

## Initial Request — 2026-08-03T13:18:21Z

Systematically investigate, debug, and optimize syntx.tvf to achieve peak accuracy parity with syntx.syn (>=0.8800 Cortical Label 3 Dice under ants.label_overlap_measures) with 100% Diffeomorphic Safety (0.0000% Folding, min det(J) > 0.0), without regressing any pre-existing project utilities or unit tests.

Working directory: /Users/stnava/data/syntx
Integrity mode: development

## Requirements

### R1. Root Cause Identification & Utility Regression Safeguards
Investigate every algorithmic difference between syntx.syn (>=0.8800 Dice, 0.0000% Folding) and syntx.tvf. Identify why syntx.tvf failed to match syntx.syn accuracy without grid folding, drawing inspiration from both syntx.syn's symmetric midpoint deformation and main branch's high alignment mechanisms, while guaranteeing zero regressions across all core utilities (syntx.viz, syntx.robust_affine, syntx.syn, etc.).

### R2. Strict Diffeomorphic Optimization & Comprehensive Verification
Implement the algorithmic fixes in src/syntx/tvf.py and src/syntx/tvf_jax.py to achieve >=0.8800 Cortical Label 3 Dice on Mindboggle DKT benchmark pairs while strictly enforcing 0.0000% grid folding (min det(J) > 0.0), sub-0.01mm mean inverse identity error, and 100% unit test suite pass rate across the entire repository.

## Acceptance Criteria

### Registration Accuracy, Safety & Utility Preservation
- [ ] syntx.tvf Cortical Label 3 Dice >= 0.8800 evaluated strictly via ants.label_overlap_measures().
- [ ] Grid Folding strictly 0.0000% (min det(J) > 0.0).
- [ ] Mean Inverse Identity Error <= 0.01 mm.
- [ ] Deformable execution time <= 20.0 seconds.
- [ ] 100% Pass Rate across full unit test suite (pytest tests/).
