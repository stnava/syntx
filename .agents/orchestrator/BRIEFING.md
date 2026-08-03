# BRIEFING — 2026-08-03T09:21:30-04:00

## Mission
Systematically investigate, debug, and optimize syntx.tvf to achieve peak accuracy parity with syntx.syn (>=0.8800 Cortical Label 3 Dice under ants.label_overlap_measures) with 100% Diffeomorphic Safety (0.0000% Folding, min det(J) > 0.0), without regressing any pre-existing project utilities or unit tests.

## 🔒 My Identity
- Archetype: teamwork_preview_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: /Users/stnava/data/syntx/.agents/orchestrator
- Original parent: parent
- Original parent conversation ID: b8698401-1f92-4fc9-9a2e-fc1ca77710bc

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: /Users/stnava/data/syntx/.agents/orchestrator/PROJECT.md
1. **Decompose**:
   - Survey Phase: Dispatch 3 Explorers (in parallel) to map differences between syntx.syn and syntx.tvf, investigate src/syntx/tvf.py and tvf_jax.py, identify root causes of accuracy gaps & folding.
   - Decompose into Milestones (M1: Root Cause Analysis & Architecture Strategy Formulation, M2: Strict Diffeomorphic Optimization Implementation in PyTorch & JAX, M3: Comprehensive Verification, Utility Safeguards & Forensic Audit).
2. **Dispatch & Execute**:
   - Direct iteration loop per milestone (Explorer -> Worker -> Reviewer -> Challenger -> Forensic Auditor -> Gate).
3. **On failure**: Retry -> Replace -> Skip -> Redistribute -> Redesign -> Escalate.
4. **Succession**: Self-succeed at spawn count >= 20. Write handoff.md, spawn successor, exit.
- **Work items**:
  1. Survey Phase: 3 Explorers inspecting syntx.syn vs syntx.tvf [in-progress]
  2. M1: Root Cause Analysis & Strategy Formulation [pending]
  3. M2: Strict Diffeomorphic Optimization Implementation (tvf.py & tvf_jax.py) [pending]
  4. M3: Comprehensive Verification & Forensic Audit [pending]
- **Current phase**: 1 (Survey & Root Cause Analysis)
- **Current focus**: Parallel Exploration of syntx.tvf vs syntx.syn codebase

## 🔒 Key Constraints
- User request: /Users/stnava/data/syntx/.agents/ORIGINAL_REQUEST.md
- Cortical Label 3 Dice >= 0.8800 (ants.label_overlap_measures)
- Grid Folding strictly 0.0000% (min det(J) > 0.0)
- Mean Inverse Identity Error <= 0.01 mm
- Execution time <= 20.0 seconds
- 100% unit test pass rate (pytest tests/)
- Single Interpolation Policy (Rule 1 in GEMINI.md)
- LNCC Variance Floor var_safe = max(var, 10^-6) & clamp(cc, -1.0, 1.0)
- Physical Spacing Anisotropy & ITK CFL Spacing Multiplier
- Lie Algebra infinitesimal expansion
- TVF LARS Optimizer / Pyramid-Proportional Velocity Grids / padding_mode='border'
- Zero tolerance for cheating: Forensic auditor must deliver CLEAN verdict
- Never reuse a subagent after handoff

## Current Parent
- Conversation ID: b8698401-1f92-4fc9-9a2e-fc1ca77710bc
- Updated: 2026-08-03

## Key Decisions Made
- Scheduled heartbeat cron task af276618-ee17-4e9e-b1ac-118e2b78174b/task-17.
- Initialized BRIEFING.md, plan.md, progress.md, context.md, PROJECT.md, GATE_STATUS.md.
- Dispatched 3 parallel Explorers for Survey Phase.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_survey_1 | teamwork_preview_explorer | PyTorch TVF vs SyN Investigation | in-progress | 8a5d3859-29ba-4cb7-aba7-ece613d4011e |
| explorer_survey_2 | teamwork_preview_explorer | JAX TVF vs SyN & Parity Investigation | in-progress | e655ebb4-4c44-4172-a622-527f451de67b |
| explorer_survey_3 | teamwork_preview_explorer | Benchmarks, Safety & Utility Preserver | in-progress | b27ac69d-9039-465e-8782-6234f722f87d |

## Succession Status
- Succession required: no
- Spawn count: 3 / 20
- Pending subagents: 8a5d3859-29ba-4cb7-aba7-ece613d4011e, e655ebb4-4c44-4172-a622-527f451de67b, b27ac69d-9039-465e-8782-6234f722f87d
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: af276618-ee17-4e9e-b1ac-118e2b78174b/task-17
- Safety timer: none
- On succession: kill all timers before spawning successor
- On context truncation: run `manage_task(Action="list")` — re-create if missing

## Artifact Index
- /Users/stnava/data/syntx/.agents/ORIGINAL_REQUEST.md — Verbatim original user request
- /Users/stnava/data/syntx/.agents/orchestrator/BRIEFING.md — Working memory index
- /Users/stnava/data/syntx/.agents/orchestrator/plan.md — Detailed execution plan
- /Users/stnava/data/syntx/.agents/orchestrator/progress.md — Progress log and heartbeat
- /Users/stnava/data/syntx/.agents/orchestrator/context.md — Mission context & constraints
- /Users/stnava/data/syntx/.agents/orchestrator/PROJECT.md — Architecture & Milestones
