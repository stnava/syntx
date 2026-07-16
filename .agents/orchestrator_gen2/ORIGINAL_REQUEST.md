# Original User Request

## Follow-up — 2026-07-14T21:17:09Z

Establish baseline parameters for `syntx` to achieve parity with `ants.registration` on 2D images, then design and evaluate an experimental mechanism to trigger deep feature metrics only when they are non-degenerate.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. Baselines & Parity Audit
Establish default parameters for the `syntx` registration pipeline (pyramid levels, iterations, smoothing, step size) to match or exceed `ants.registration` baseline DICE scores under equivalent LNCC/Mattes-MI configurations in 2D.

### R2. Deep Feature Degeneracy Triggering
Experiment with heuristics (such as hardcoded resolution limits, spatial variance, or gradient norms) to determine when deep feature extractors become degenerate during coarse resolution stages. Implement a mechanism in `syntx` that dynamically decides when to activate deep feature-space metrics (ResNet-10/VGG19) versus falling back to raw intensity metrics based on your findings.

### R3. Visual Comparison & Inference Reporting
Generate a detailed performance report with rich imagery showing edge overlap, deformed grids, Jacobian determinants, and side-by-side warped vs target images. Save this report to `docs/parity_report.html`.

## Acceptance Criteria

### Parity & Accuracy
- [ ] Achieve mean DICE score parity (within 1%) with `ants.registration` across 2D phantoms `r16`, `r27`, `r64`.
- [ ] Verify that the triggering mechanism successfully deactivates deep feature metrics at degenerate resolution levels (improving DICE and reducing folding compared to always-on baseline).
- [ ] Export the generated HTML performance report to `docs/parity_report.html`.
- [ ] All 78 unit tests must pass successfully.
