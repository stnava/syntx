# Original User Request

## Follow-up — 2026-07-15T03:16:22Z

Re-evaluate deep features in 2D and 3D, re-confirm 3D baseline parity with classical ANTs, and systematically sweep and analyze the impact of different optimizers (specifically Adam, SGD, L-BFGS, and the current step-based/CFL update) on registration accuracy (DICE), coordinate regularity (folding rate), and convergence speed in 2D and 3D.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. Deep Features Analysis (2D/3D)
Perform a detailed, systematic analysis comparing LNCC against deep feature extractors (VGG19, ResNet-10, DINOv2) in both 2D and 3D. Document DICE scores, folding rates (Jacobian determinant <= 0), and optimization speeds.

### R2. 3D Parity Verification
Re-confirm registration parity with `ants.registration` across 3D brain benchmarks under intensity LNCC and Mattes Mutual Information configurations.

### R3. Optimizer Sweep (2D/3D)
Perform a comparative analysis of optimizer choices (specifically Adam vs SGD vs L-BFGS vs step-based updates) under both intensity metrics and deep feature metrics in both 2D and 3D. Measure loss convergence, final overlap (DICE), runtimes, and folding rates.

### R4. Rich HTML Performance Dashboard
Compile a detailed performance report in HTML format at `docs/optimizer_and_deep_feature_report.html` containing structural overlays, warp grids, Jacobian maps, convergence plots, and side-by-side deformed/target comparisons.

## Acceptance Criteria

### Parity & Accuracy
- [ ] Measure and document the impact of optimizer choice on registration quality (DICE, runtime, folding) in both 2D and 3D.
- [ ] Re-confirm 3D baseline parity (within 1%) with `ants.registration` across standard 3D brain benchmarks.
- [ ] Generate the HTML report at `docs/optimizer_and_deep_feature_report.html` with all required structural images, convergence plots, and warp/Jacobian grids.
- [ ] All unit tests in the repository must pass successfully.
