# BRIEFING — 2026-07-15T11:36:30-04:00

## Mission
Investigate and plan the implementation of 3D registration parity between syntx and ants.registration, focusing on coordinate mapping, SyN optimization loop, displacement fields, affine composition, and evaluating accuracy.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, analyst, plan-writer
- Working directory: /Users/stnava/code/syntx/.agents/explorer_investigation
- Original parent: 97b990be-00c5-417a-9176-96f8949beb69
- Milestone: Registration Parity Analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Single interpolation policy: No intermediate file-based pre-warping; compose all transforms and apply directly.
- Standardized Returns: Lower metric score strictly means higher similarity.
- Physical space awareness in optimization pipelines mapping physical space differences to normalized grid space.
- Lie Algebra gradients must be preserved (avoid non-differentiable conditionals at zero).

## Current Parent
- Conversation ID: 97b990be-00c5-417a-9176-96f8949beb69
- Updated: 2026-07-15T11:35:15-04:00

## Investigation State
- **Explored paths**:
  - `src/syntx/syn.py` (PyTorch implementation of the SyN loops and coordinate conversion functions)
  - `src/syntx/syn_jax.py` (JAX implementation of the SyN loops and coordinate conversion functions)
  - `src/syntx/transform.py` (transform structures and composition helpers)
- **Key findings**:
  - Currently, `syntx` uses normalized coordinate grids in `[-1, 1]` for displacement fields and composition during optimization.
  - The affine composition logic `compose_grids(identity_full + w_r2l_inv, identity_full + w_l2r)` currently uses $\phi_2 \circ \phi_1^{-1}$ which is dimensionally incorrect (evaluates moving-domain warp at fixed domain coordinates) and differs from standard SyN $\phi_2^{-1} \circ \phi_1$.
  - Explicitly mapping physical coordinates using metadata (spacing, origin, direction) yields a clean composition $y = A(\phi_2^{-1}(\phi_1(x)))$ using a single interpolation of the moving image.
- **Unexplored areas**:
  - GPU benchmarking speed of JAX vs PyTorch under the new physical mm coordinate grid calculation.

## Key Decisions Made
- Confirmed coordinate mapping formulas for fixed-space physical grids and physical-to-normalized conversions.
- Proposed a physical mm update strategy for displacement fields and a unified affine grid composition method.
- Designed `scratch/test_internal_dice.py` test suite.

## Artifact Index
- /Users/stnava/code/syntx/.agents/explorer_investigation/ORIGINAL_REQUEST.md — Initial task request
- /Users/stnava/code/syntx/.agents/explorer_investigation/BRIEFING.md — Current status and constraints index
- /Users/stnava/code/syntx/.agents/explorer_investigation/handoff.md — Detailed analysis report
