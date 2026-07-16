# BRIEFING — 2026-07-14T16:06:00-04:00

## Mission
Implement 2D and 3D registration benchmarks (R1, R2, R3) and export a visual dashboard report to docs/benchmarks.html.

## 🔒 My Identity
- Archetype: Performance & Benchmarking Worker
- Roles: implementer, qa, specialist
- Working directory: /Users/stnava/code/syntx/.agents/worker_benchmarking
- Original parent: 082e9530-4ddf-4d1a-a89f-339fd391a8ac
- Milestone: Benchmark and Verification

## 🔒 Key Constraints
- Single Interpolation Policy (no intermediate resampling/pre-warping).
- VGG 3D Mode Requirement: only vgg_mode='lncc_3d' with layers=[4] is acceptable for accuracy benchmarks.
- No dummy/facade or hardcoded test results.

## Current Parent
- Conversation ID: 082e9530-4ddf-4d1a-a89f-339fd391a8ac
- Updated: yes

## Task Summary
- **What to build**: A benchmark runner script `examples/run_benchmarks.py` comparing VGG19, DINOv2, ResNet-10, and standard ants.registration for 2D phantoms and 3D T1w scans (graded via `antspyt1w.resnet_grader`), generating an HTML dashboard report in `docs/benchmarks.html`.
- **Success criteria**: Successful run of both 2D and 3D benchmarks, saving outputs/visualizations in `outputs_comparison/`, compiling the dashboard report showing DICE, runtime, folding rates, correlation of grader score with DICE, grid warp, edge overlap, Jacobian determinant, and side-by-side warped vs target visual maps.
- **Interface contracts**: /Users/stnava/code/syntx/GEMINI.md
- **Code layout**: Source in `syntx/`, benchmarks in `examples/run_benchmarks.py`, outputs in `outputs_comparison/` and `docs/benchmarks.html`.

## Key Decisions Made
- Deletion bug fixed in `examples/run_benchmarks.py`: rigid transform file is copied to stable cached path and kept until all metrics for a scan complete.
- Axes swap bug fixed in `src/syntx/syn.py`: component mapping `c_idx = k` instead of `dim - 1 - k` to scale and stack components in correct coordinate order (X, Y, Z).
- Skipping affine pre-alignment in PyTorch since the initial rigid pre-alignment from ANTs is already extremely accurate (correlation > 0.92) and prevents incremental vs absolute affine space mismatch.

## Artifact Index
- None yet

## Change Tracker
- **Files modified**: `src/syntx/syn.py`
- **Build status**: Pass (All unit tests pass)
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pass
- **Lint status**: Pass
- **Tests added/modified**: None

## Loaded Skills
- **Source**: release (/Users/stnava/code/syntx/.agents/skills/release/SKILL.md)
  - **Local copy**: /Users/stnava/code/syntx/.agents/worker_benchmarking/skills/release/SKILL.md
  - **Core methodology**: Release bump version, commit, and tag.
