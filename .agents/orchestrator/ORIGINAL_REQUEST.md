# Original User Request

## 2026-07-14T22:08:23Z

Evaluate the impact of deep features in 2D, achieve registration parity with classical ANTs in 3D using cached 3D T1w brain scans, and evaluate the benefit of deep feature metrics in 3D.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. 2D Deep Features Impact Analysis
Perform a systematic sweep in 2D comparing raw intensity LNCC against deep feature metrics (ResNet-10, VGG19, DINOv2) across the 2D phantoms benchmark suite. Measure DICE scores, folding rates (percentage of Jacobian determinant <= 0), and optimization speeds.

### R2. 3D Parity Achievement
Establish parameter defaults and optimization configurations in 3D for PyTorch/JAX `syntx` (e.g. `levels=[4, 2, 1]`, optimized step sizes, and smoothing parameters) to match or exceed `ants.registration` baseline DICE scores under equivalent LNCC/Mattes-MI configurations using cached 3D T1w brain scans.

### R3. 3D Deep Features Impact Analysis
Evaluate the benefit of deep feature metrics (e.g., 3D VGG LNCC with Layer 4, DINOv2, ResNet-10) in 3D compared to standard 3D intensity baselines, measuring registration accuracy (DICE), coordinate regularity (folding rate), and execution times on 3D brain scans.

### R4. Comprehensive Parity & Deep Feature Report
Compile a detailed performance report at `docs/deep_feature_impact_report.html` documenting all 2D and 3D results, including structural overlaps, Jacobian determinant maps, warped grids, and side-by-side visualization pairs.

## Acceptance Criteria

### Parity & Accuracy
- [ ] Establish 3D parameter configurations that achieve DICE score parity (within 1%) with `ants.registration(..., type_of_transform='SyN')` on 3D brain registration benchmarks.
- [ ] Measure and document the impact of deep features (ResNet-10, VGG19, DINOv2) in 2D and 3D.
- [ ] Generate the HTML report at `docs/deep_feature_impact_report.html` with all required structural images, grid warps, and Jacobian maps.
- [ ] All unit tests in the repository must pass successfully.
