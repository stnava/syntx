# Plan: syntx Deep Feature Benchmarking & 3D Parity Validation

This plan outlines the steps to evaluate the impact of deep features in 2D and 3D, achieve 3D registration parity with classic ANTs, and generate a comprehensive visual performance report.

## Milestones

### Milestone 1: Exploration and Baseline Verification
- Dispatch an Explorer subagent to:
  - Verify access to 2D phantoms (`r16`, `r27`, `r64`).
  - Verify access to cached 3D scans (`cache/28364-00000000-T1w-00_brain.nii.gz` to `cache/28575-00000000-T1w-07_brain.nii.gz` and their corresponding DKT label maps).
  - Inspect existing registration and evaluation functions in `src/syntx/syn.py` and `src/syntx/syn_jax.py`.
  - Propose specific registration parameter configurations (step sizes, iterations, smoothing) to run for 2D sweeps and 3D parity checks.

### Milestone 2: 2D Systematic Sweep (R1)
- Dispatch a Worker subagent to:
  - Run 2D registration sweeps between phantom pairs: ('r16', 'r27'), ('r16', 'r64'), and ('r27', 'r64').
  - Compare raw intensity LNCC against deep feature metrics: ResNet-10, VGG19 (in 2D LNCC mode), and DINOv2.
  - Measure Otsu-based tissue overlap DICE scores, coordinate folding rates (percentage of Jacobian determinant <= 0), and execution/optimization times.
  - Save results in a structured CSV (`outputs_comparison/r1_2d_sweep_results.csv`).

### Milestone 3: 3D Parity Configuration & Evaluation (R2 & R3)
- Dispatch a Worker subagent to:
  - Establish 3D parameter configurations (e.g., levels, affine/deformable iterations, step sizes, and smoothing) for PyTorch and JAX backends of `syntx` to achieve parity with `ants.registration(..., type_of_transform='SyN')`.
  - Evaluate registration accuracy (DKT label map DICE), coordinate regularity (folding rate), and execution time on cached 3D T1w brain scans.
  - Benchmark equivalent standard intensity metrics (LNCC, Mattes-MI) between `syntx` and `ants.registration` to prove parity (within 1% DICE).
  - Benchmark 3D deep feature metrics (VGG 3D LNCC with Layer 4, DINOv2, ResNet-10) compared to the standard intensity baselines on the same 3D scans.
  - Save results in a structured CSV (`outputs_comparison/r2_r3_3d_sweep_results.csv`).

### Milestone 4: Comprehensive Report & Visualizations (R4)
- Dispatch a Worker subagent to:
  - Generate the required diagnostic visualizations for the HTML report:
    - Region/label overlaps and edge overlays between warped and fixed/target images.
    - Warped/deformed coordinate grids.
    - Jacobian determinant maps (highlighting folds).
    - Side-by-side warped vs. fixed/target images.
  - Compile all results (2D sweeps, 3D parity, 3D deep features) into a comprehensive HTML dashboard report at `docs/deep_feature_impact_report.html`.
  - The report must strictly follow `GEMINI.md` Rule 3 for reporting/visualization.

### Milestone 5: Verification & Forensic Audit
- Dispatch Reviewer, Challenger, and Forensic Auditor subagents to:
  - Verify that the generated HTML report contains all required visualizations and tables.
  - Verify all unit tests in the repository pass.
  - Run the Forensic Auditor to check compliance with the Single Interpolation Policy and VGG 3D LNCC Layer 4 guidelines.
