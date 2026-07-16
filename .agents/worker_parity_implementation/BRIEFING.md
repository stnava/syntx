# BRIEFING — 2026-07-15T11:36:52-04:00

## Mission
Implement native physical space optimization and correct affine coordinate composition in PyTorch and JAX to achieve 3D registration parity.

## 🔒 My Identity
- Archetype: teamwork_preview_worker
- Roles: implementer, qa, specialist
- Working directory: /Users/stnava/code/syntx/.agents/worker_parity_implementation
- Original parent: 97b990be-00c5-417a-9176-96f8949beb69
- Milestone: 3D registration parity

## 🔒 Key Constraints
- Native physical space optimization (displacement fields in mm).
- Affine coordinate composition must be strictly composed as y = A(phi_2_inv(phi_1(x))).
- Single interpolation policy: DO NOT pre-warp images/labels before optimization.
- Target: DICE score >= 0.999 on `scratch/test_internal_dice.py`.
- Run pytest and ensure all tests pass.
- Preserve 2D parity.

## Current Parent
- Conversation ID: 97b990be-00c5-417a-9176-96f8949beb69
- Updated: not yet

## Task Summary
- **What to build**: Native physical space optimization and correct affine coordinate composition in both PyTorch (`syn.py`) and JAX (`syn_jax.py`).
- **Success criteria**:
  - `scratch/test_internal_dice.py` passes with DICE >= 0.999.
  - No pre-warping of images/labels prior to optimization.
  - Standard unit tests in the repo pass successfully.
  - 2D registration parity preserved and 3D registration runs successfully.
- **Interface contracts**: /Users/stnava/code/syntx/.agents/orchestrator_3d_parity_1/PROJECT.md / GEMINI.md
- **Code layout**: src/syntx/

## Key Decisions Made
- [TBD]

## Artifact Index
- [TBD]

## Change Tracker
- **Files modified**: None
- **Build status**: TBD
- **Pending issues**: None

## Quality Status
- **Build/test result**: TBD
- **Lint status**: TBD
- **Tests added/modified**: None

## Loaded Skills
- None
