# BRIEFING — 2026-07-15T12:00:03-04:00

## Mission
Implement native physical space optimization and correct affine coordinate composition in PyTorch and JAX to achieve 3D registration parity.

## 🔒 My Identity
- Archetype: teamwork_preview_worker
- Roles: implementer, qa, specialist
- Working directory: /Users/stnava/code/syntx/.agents/worker_parity_implementation_gen2
- Original parent: 97b990be-00c5-417a-9176-96f8949beb69
- Milestone: 3D Registration Parity

## 🔒 Key Constraints
- Native physical space optimization (displacement fields in mm)
- Correct affine coordinate composition: y = A(phi_2_inv(phi_1(x)))
- Single interpolation policy (no pre-warped images or labels prior to optimization)
- Pass scratch/test_internal_dice.py with DICE >= 0.999
- Run pytest and pass all unit tests
- Keep existing 2D parity preserved and 3D registration running successfully

## Current Parent
- Conversation ID: 97b990be-00c5-417a-9176-96f8949beb69
- Updated: not yet

## Task Summary
- **What to build**: Physical space optimization & composition mapping logic in `syn.py` and `syn_jax.py`
- **Success criteria**: scratch/test_internal_dice.py DICE >= 0.999 and pytest green
- **Interface contracts**: /Users/stnava/code/syntx/.agents/orchestrator_3d_parity_1/PROJECT.md, GEMINI.md
- **Code layout**: src/syntx/syn.py and src/syntx/syn_jax.py

## Key Decisions Made
- None yet

## Change Tracker
- **Files modified**: None
- **Build status**: TBD
- **Pending issues**: TBD

## Quality Status
- **Build/test result**: TBD
- **Lint status**: TBD
- **Tests added/modified**: TBD

## Loaded Skills
For each loaded Antigravity skill, record:
- **Source**: /Users/stnava/code/syntx/.agents/skills/release/SKILL.md
  - **Local copy**: /Users/stnava/code/syntx/.agents/worker_parity_implementation_gen2/skills/release_SKILL.md
  - **Core methodology**: Automate the release process (git tag/version bump)
- **Source**: /Users/stnava/.gemini/antigravity-cli/builtin/skills/antigravity_guide/SKILL.md
  - **Local copy**: /Users/stnava/code/syntx/.agents/worker_parity_implementation_gen2/skills/antigravity_guide_SKILL.md
  - **Core methodology**: Provides a guide and sitemap for Google Antigravity platforms and surfaces

## Artifact Index
- None yet
