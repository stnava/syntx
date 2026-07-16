# BRIEFING — 2026-07-14T14:54:00-04:00

## Mission
Profile the baseline execution, identify performance bottlenecks (JIT compilation, host-device transfers, VJP callbacks, interpolations), inspect GEMINI.md compliance, and document the findings.

## 🔒 My Identity
- Archetype: explorer
- Roles: Teamwork explorer
- Working directory: /Users/stnava/code/syntx/.agents/teamwork_preview_explorer_perf_baseline
- Original parent: f21b20dc-e4b4-4894-9c5b-2f32499326d4
- Milestone: Milestone 1 (Baseline Profiling & Diagnostics)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- CODE_ONLY network mode: no external HTTP/HTTPS clients

## Current Parent
- Conversation ID: f21b20dc-e4b4-4894-9c5b-2f32499326d4
- Updated: 2026-07-14T14:54:00-04:00

## Investigation State
- **Explored paths**:
  - `src/syntx/syn_jax.py` (JAX-PyTorch bridge, custom VJP, and compile patterns)
  - `src/syntx/features.py` (Interpolation and extraction logic)
  - `src/syntx/syn.py` (High-level registration API and alignment constraints)
  - `GEMINI.md` (Project constraints and guidelines)
  - `examples/evaluate_all_metrics.py` (Benchmark script)
- **Key findings**:
  - All 77 unit tests passed successfully (6 skipped).
  - XLA JAX JIT compilation dominates registration runtime (~65%). Recompilation is triggered per-registration call because the loss functions are dynamically instantiated/wrapped via `make_pytorch_loss_jax` inside `fit`, creating a new function object that invalidates the JIT cache key.
  - The DLPack JAX-PyTorch bridge is bypassed during JIT tracing; JAX tracers trigger a `jax.pure_callback` fallback, causing CPU host-memory copies (using `np.asarray` and `torch.from_numpy` on CPU, then copying back to device).
  - Redundant 2D interpolations exist in `FeatureSpaceLoss._forward_2d_triplanar` (which forces slice scaling to target_size).
  - Redundant 3D interpolations exist in `SwinUNETRExtractor` (forces 3D input scaling to 96x96x96 and scales down feature maps).
  - A violation of the Single Interpolation Policy exists in `syn.py` (using `ants.apply_transforms` to pre-warp input images if `initial_transform` is provided).
  - A signature configuration mismatch exists in `syn.py` defaults (defaults to `vgg_layers=[8]` instead of the recommended `[4]`).
- **Unexplored areas**: None (Milestone 1 is complete).

## Key Decisions Made
- Completed profiling using cProfile and analyzed trace for JAX/PyTorch interactions.
- Performed detailed review of source logic against GEMINI.md requirements.

## Artifact Index
- /Users/stnava/code/syntx/.agents/teamwork_preview_explorer_perf_baseline/handoff.md — Detailed analysis and handoff report
