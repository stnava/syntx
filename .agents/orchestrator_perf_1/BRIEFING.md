# BRIEFING — 2026-07-14T18:50:31Z

## Mission
Perform a performance audit of the `syntx` registration pipeline (JAX and PyTorch backends) to identify bottlenecks, reduce redundant operations, and optimize registration speed and memory utilization.

## 🔒 My Identity
- Archetype: teamwork_preview_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: /Users/stnava/code/syntx/.agents/orchestrator_perf_1
- Original parent: parent
- Original parent conversation ID: ecfa969b-d232-4cfb-9477-a3e3d97d02d2

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: /Users/stnava/code/syntx/PROJECT.md
1. **Decompose**: Decompose the performance audit into parallel evaluation/profiling track (E2E/benchmarking) and implementation track (optimizations).
2. **Dispatch & Execute**:
   - **Delegate (sub-orchestrator)**: Spawn sub-orchestrators for E2E benchmarking and performance optimizations.
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (sub-orchestrators only, last resort)
4. **Succession**: Self-succeed at 16 spawns. Spawn successor via teamwork_preview_orchestrator.
- **Work items**:
  1. Initialize PROJECT.md and decomposition [done]
  2. Spawn E2E Testing Track/Explorer Agent to run baseline pytest and profiling [done]
  3. Spawn Worker Agent to implement optimizations [done]
- **Current phase**: 4
- **Current focus**: Final handoff and synthesis

## 🔒 Key Constraints
- Avoid spatial blurring / loss of high-frequency boundary info (from GEMINI.md). No pre-warping of images/segmentations prior to optimization.
- Comcompose and apply multiple transforms directly to the native-space images in a single step (GEMINI.md).
- Accuracy thresholds: For cortical label maps, drop in Mean DICE score of >= 0.01 is unacceptable.
- Only VGG 3D LNCC with Layer 4 meets performance level of standard intensity LNCC (no VGG 2D or coarser layers like Layer 8).
- Any HTML or artifact reports must display structural/spatial images, including: edge/region overlap, deformed grids, Jacobian determinant maps, warped/deformed images side-by-side next to target/fixed images.
- Never write, modify, or create source code files directly as the orchestrator.
- Never run build/test commands yourself.
- Never reuse a subagent after it has delivered its handoff.

## Current Parent
- Conversation ID: ecfa969b-d232-4cfb-9477-a3e3d97d02d2
- Updated: not yet

## Key Decisions Made
- Perform performance audit using the real multi-modal datasets and existing codebase setups.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| Explorer | teamwork_preview_explorer | Baseline Profiling & Diagnostic Run | completed | 79235205-cc3e-4d53-9e27-ec080cc3beed |
| Worker | teamwork_preview_worker | Performance Optimization | completed | 29dc8dd7-92dc-499e-9db4-be2b3ab2f06b |
| Auditor | teamwork_preview_auditor | Forensic Integrity Audit | completed | 57a6dbe5-013e-405c-8858-fff8edde6fc1 |

## Succession Status
- Succession required: no
- Spawn count: 3 / 16
- Pending subagents: none
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: cancelled
- Safety timer: none

## Artifact Index
- /Users/stnava/code/syntx/.agents/orchestrator_perf_1/ORIGINAL_REQUEST.md — Original request verbatim
