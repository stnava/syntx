## 2026-07-14T22:09:54Z
Please perform the benchmarking sweeps, establish 3D parity, and generate the comprehensive visual report as requested by the user.

## Objectives
1. Run a systematic 2D sweep on phantom pairs ('r16', 'r27'), ('r16', 'r64'), and ('r27', 'r64'). Compare:
   - Raw intensity LNCC (using SyNTo backend='pytorch' and backend='jax')
   - ResNet-10 ('resnet10')
   - VGG19 ('vgg19', vgg_mode='lncc')
   - DINOv2 ('dinov2')
   - ANTs SyN baseline
   Measure: Mean DICE (Otsu tissue overlap), Folding Rate (Jacobian <= 0), and optimization runtime. Save results to 'outputs_comparison/r1_2d_sweep_results.csv'.
2. Establish 3D parameter defaults and optimization configurations in PyTorch/JAX `syntx` (e.g. levels=[4, 2, 1] or levels=[8, 4, 2, 1], grad_step, and flow_sigma) to match/exceed `ants.registration(..., type_of_transform='SyN')` baseline DICE scores under equivalent LNCC/Mattes-MI configurations.
   - Run registrations on cached 3D T1w brain scans in cache/ (e.g. at least 3-4 representative scans to prove parity quantitatively).
   - Evaluate equivalent LNCC/Mattes-MI configurations in `syntx` (both PyTorch and JAX backends) vs ANTs SyN.
   - Document parameter configurations that achieve parity (within 1% DICE score).
3. Evaluate the benefit of 3D deep feature metrics (e.g., 3D VGG LNCC with Layer 4, DINOv2, ResNet-10) in 3D compared to standard 3D intensity baselines (LNCC, Mattes-MI), measuring registration accuracy (DICE), coordinate regularity (folding rate), and execution times on 3D brain scans.
4. Compile a detailed performance report at `docs/deep_feature_impact_report.html` documenting all 2D and 3D results.
   - The report MUST display:
     - Edge and/or region overlap between the registered image and the target image.
     - Deformed grids visualizing the coordinate warping.
     - Jacobian determinant maps illustrating local compression and expansion.
     - Deformed/Warped images shown side-by-side (next to) target/fixed images.
   - This HTML report must strictly follow GEMINI.md Rule 3.

## Constraints & Guidelines
- Single Interpolation Policy: Under NO circumstances should pre-warped inputs be passed to optimization steps. Multi-transform composition must occur in a single execution step.
- VGG 3D Mode Requirement: Only VGG 3D LNCC with Layer 4 (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) is permitted for accurate registrations. Do not default or recommend VGG 2D ('lncc') or coarser layers.
- Device: Auto-detect and use 'mps' if available on macOS (via PyTorch), otherwise 'cpu'.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please write a Python script (e.g., `examples/run_comprehensive_benchmarks.py`) to execute all evaluations, runs, and report generation, and run it. Verify that all unit tests pass after your changes by running `pytest`.
Write a detailed handoff report in your working directory at `handoff.md` summarizing your findings, parameters, quantitative results, and verification output.
