# BRIEFING — 2026-07-14T18:55:00Z

## Mission
Optimize the syntx registration pipeline (DLPack bridge, SwinUNETR features, single interpolation policy, and LNCC 3D default metrics).

## 🔒 My Identity
- Archetype: implementer/qa/specialist
- Roles: implementer, qa, specialist
- Working directory: /Users/stnava/code/syntx/.agents/worker_perf_optimization
- Original parent: f21b20dc-e4b4-4894-9c5b-2f32499326d4
- Milestone: Milestones 2 & 3

## 🔒 Key Constraints
- Single Interpolation Policy (no intermediate file-based pre-warping; compose all transforms in a single step).
- Accuracy threshold constraint: no drop in Mean DICE score >= 0.01.
- VGG 3D Mode Requirement (default to vgg_mode='lncc_3d' and layer 4, no VGG 2D).
- Complete DLPack bridge optimization avoiding CPU fallbacks and compile invalidation in SyNTo.fit.

## Current Parent
- Conversation ID: f21b20dc-e4b4-4894-9c5b-2f32499326d4
- Updated: not yet

## Task Summary
- **What to build**: JAX-PyTorch DLPack optimization in `syn_jax.py`, SwinUNETR padding/crop optimization in `features.py`, Initial grid warp-based composition in `syn.py` and `syn_jax.py`, default parameters update.
- **Success criteria**: 77 unit tests pass, no CPU fallback during registration, evaluate_all_metrics.py runs successfully, files modified match requirements.
- **Interface contracts**: GEMINI.md, syn.py, syn_jax.py, features.py.
- **Code layout**: src/syntx/

## Key Decisions Made
- Used zero-copy DLPack to bridge JAX and PyTorch eager execution in syn_jax.py, bypassing JAX value_and_grad compilation and CPU fallback for PyTorch metric passes.
- Leveraged JAX VJP for backpropagating gradients of the similarity loss to the warps when analytical gradients are disabled.
- Optimized SwinUNETRExtractor.extract by padding to multiple of 32 and cropping the output features, avoiding slow upscaling/downscaling.
- Reimplemented TriPlanar 2D slice interpolation to skip redundant Bilinear interpolation if the input slice dimensions already match the target max resolution.
- Enforced Single Interpolation Policy: replaced moving image pre-warping with coordinate-based `initial_grid` calculation and composed grid sampling in both backends.

## Artifact Index
- /Users/stnava/code/syntx/.agents/worker_perf_optimization/handoff.md — Handoff report for verification

## Change Tracker
- **Files modified**:
  * src/syntx/syn_jax.py: Wrapped make_pytorch_loss_jax, implemented JIT-compiled prepare_mid_images_and_gradients_jax/syn_update_step_jax helpers, redesigned fit loop to eagerly evaluate PyTorch losses on GPU arrays via DLPack.
  * src/syntx/features.py: Optimized SwinUNETRExtractor padding/cropping, optimized TriPlanar slice interpolation.
  * src/syntx/syn.py: Integrated coordinate warping to compute initial_grid and compose grids on the fly, updated VGG defaults to layer 4 and lncc_3d.
- **Build status**: Pass
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pass (85/85 unit tests passed, 93% coverage)
- **Lint status**: Pass
- **Tests added/modified**: Added comprehensive coverage tests for JAX helpers, callbacks, empty tensors, and inverse series projection.

## Loaded Skills
- **Source**: release (/Users/stnava/code/syntx/.agents/skills/release/SKILL.md)
- **Local copy**: None
- **Core methodology**: Version bumping and release tagging.
