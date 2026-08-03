# Implementation Plan — syntx.tvf Optimization & Safety

## Objective
Systematically investigate, debug, and optimize `syntx.tvf` (PyTorch `tvf.py` and JAX `tvf_jax.py`) to achieve peak accuracy parity with `syntx.syn` (Cortical Label 3 Dice >= 0.8800 under `ants.label_overlap_measures()`) with 100% Diffeomorphic Safety (0.0000% Folding, min det(J) > 0.0), sub-0.01mm mean inverse identity error, execution time <= 20.0s, and 100% pass rate across the full pytest unit test suite without regressing core project utilities.

## Milestones

| Milestone | Name | Description | Target Files | Status |
|-----------|------|-------------|--------------|--------|
| M0 | Survey & Algorithmic Audit | Parallel exploration of `syntx.syn` vs `syntx.tvf` (PyTorch & JAX). Identify velocity integration, geodesic anchoring, pyramid grid sizing, padding mode, LARS/Adam optimizer, and loss formulation differences. | `src/syntx/syn.py`, `src/syntx/tvf.py`, `src/syntx/tvf_jax.py` | IN_PROGRESS |
| M1 | Root Cause Identification & TVF Optimization Design | Reconcile findings from 3 Explorers. Enumerate exact algorithmic fixes needed in `tvf.py` and `tvf_jax.py` to match `syntx.syn` high alignment mechanisms while preserving 100% diffeomorphic safety and zero utility regressions. | `src/syntx/tvf.py`, `src/syntx/tvf_jax.py` | PLANNED |
| M2 | Strict Diffeomorphic Optimization Implementation | Implement TVF algorithmic fixes in `src/syntx/tvf.py` (PyTorch) and `src/syntx/tvf_jax.py` (JAX). Enforce pyramid velocity grid sizing, LARS optimizer, antisymmetric velocity projection/geodesic anchoring, LNCC variance floor, padding_mode='border', and inverse identity bounds. | `src/syntx/tvf.py`, `src/syntx/tvf_jax.py` | PLANNED |
| M3 | Comprehensive Verification, Regression Testing & Forensic Audit | Verify >=0.8800 Cortical Label 3 Dice on Mindboggle benchmarks, 0.0000% folding, min det(J) > 0.0, <=0.01mm inverse error, runtime <=20.0s, 100% unit test suite pass rate, and CLEAN Forensic Audit verdict. | Entire codebase & test suite | PLANNED |

## Verification Criteria
1. Cortical Label 3 Dice >= 0.8800 evaluated strictly via `ants.label_overlap_measures()`.
2. Grid Folding strictly 0.0000% (min det(J) > 0.0).
3. Mean Inverse Identity Error <= 0.01 mm.
4. Deformable execution time <= 20.0 seconds.
5. 100% Pass Rate across full unit test suite (`pytest tests/`).
6. Forensic Auditor reports CLEAN verdict (zero integrity violations/cheating).
