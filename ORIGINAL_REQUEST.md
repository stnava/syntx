# Original User Request

## Initial Request — 2026-07-14T19:33:04Z

Develop a detailed performance comparison across all feature-space metrics (VGG19, DINOv2, ResNet-10, Swin UNETR) and standard baselines in 2D and 3D. Analyze the optimization landscape, incorporate image quality grading using `resnet_grader`, and compute cortical/DKT overlap scores.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. 2D Comparative Benchmarks
Run registrations between 2D images (`r16`, `r27`, `r64`) using all feature-space metrics (VGG19, DINOv2, ResNet-10) and compare them to the baseline `ants.registration(..., "SyN")`. Characterize optimization parameters (learning rates, iterations, convergence).

### R2. 3D Characterization on Low-Quality Scans (Native Resolution)
Grade 3D T1w scans (`28364-00000000-T1w-00` through `28575-00000000-T1w-07`) using `antspyt1w.resnet_grader`. Conduct 3D deformable registrations to a fixed template at **native resolution** using all 3D feature metrics, and evaluate cortical label map overlap (DICE score on DKT labels) against the `ants.registration` baseline.

### R3. Visual Performance Reporting
Export a detailed comparison report containing:
- Summary tables of DICE scores, runtimes, and folding rates.
- Correlation plots between scan quality (resnet_grader score) and registration DICE.
- Grid warp, edge overlap, Jacobian determinant, and side-by-side warped vs target visual maps.

## Acceptance Criteria

### Execution & Accuracy
- [ ] Complete all 2D and 3D native resolution benchmark runs successfully.
- [ ] Export the visual maps for top-performing configurations to the output folder.
- [ ] Save the comprehensive benchmark report as an HTML dashboard under `docs/benchmarks.html` in the repository.

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


## 2026-07-15T04:06:43Z

Fix the parity comparison logic in `examples/run_optimizer_sweeps.py` and re-run the sweep.

Please do the following:
1. Open `examples/run_optimizer_sweeps.py`.
2. Locate the verification section around lines 383-397.
3. Currently, the script compares the ANTs CC baseline (`ants_mean_dice` from `type_of_transform='SyNOnly'`) with the PyTorch CFL `mattes_mi` run. This is a metric mismatch.
4. Modify the script to compare the ANTs CC baseline with the PyTorch CFL `lncc` run (which both use the Cross-Correlation metric).
5. Correct the print labels to read `ANTs SyN (LNCC) Dice` and `PyTorch SyNTo LNCC CFL Dice`.
6. Run `python examples/run_optimizer_sweeps.py` to ensure it completes successfully, outputs a clean `VERIFICATION SUCCESS: 3D baseline parity within 1% met!` message, and updates `docs/optimizer_and_deep_feature_report.html` and `outputs_comparison/optimizer_sweep_results.csv`.
7. Run `pytest` to ensure all tests still pass.

## Follow-up — 2026-07-15T13:13:00Z

Implement a comprehensive suite of at least 64 image comparison metrics in `syntx.image_compare` for assessing registration quality, and systematically evaluate them against a 2D generative cross-product space of known intensity and shape changes (with Grenander's metric deformation as ground truth).

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. Metric Suite Implementation
Implement a comprehensive suite of at least 64 distinct image comparison metrics accessible via a standardized API: `syntx.image_compare(a, b, metricname)`. The return value must always be standardized such that a lower score indicates better similarity. Metrics should include classical ones (MSE, PSNR, GMSD, LNCC, Mattes MI), novel metrics, and deep feature metrics (restricted to PyTorch/Torchvision to avoid heavy dependencies).

### R2. Generative Simulation Space
Create a 2D generative cross-product space of known transformations. It must include intensity changes across categories (noise, bias, inhomogeneity, modality change, step function, missing data) and shape changes (translation, rotation, affine, deformation). The ground truth shape change magnitude should be defined by the L2 norm of the displacement field (Grenander's metric deformation). Ensure at least 80% overlap is maintained for all generated image pairs.

### R3. Evaluation and Reporting
Evaluate the metrics systematically against the generative simulation space. Produce an HTML or markdown artifact report summarizing the performance of the metrics. Per `syntx` project rules, the report must include visual inspections of registration/comparison quality (structural/spatial images, deformed grids, Jacobian determinant maps, and side-by-side deformed/warped images).

### R4. Intelligible Documentation
Write documentation and examples that are highly intelligible to non-experts ("naive users"). The documentation should clearly explain the behavior, usage, and expected outcomes of the metric suite. Includes roles like engineers, critics, and naive users to make sure it is intelligible.

## Acceptance Criteria

### Metric Suite & API
- [ ] A programmatic test verifies that `syntx.image_compare` can be called with at least 64 unique, valid `metricname` strings without erroring.
- [ ] A programmatic test verifies that for identical images, all metrics return a score of 0 (or their minimum possible value), and scores strictly increase as divergence increases.

### Generative Space
- [ ] A test asserts that the generative pipeline outputs a cross-product of the specified intensity and shape changes, and that every generated pair maintains >= 80% spatial overlap.
- [ ] Ground truth magnitudes (e.g., L2 norm of the displacement field) are explicitly returned for each generated pair.

### Evaluation & Documentation
- [ ] An evaluation script successfully runs a subset of metrics against the generative space and outputs a visual report.
- [ ] The report contains at least one example of a deformed grid, Jacobian determinant map, and side-by-side image comparison.
- [ ] Documentation includes a clear, runnable example script geared towards a non-expert user.

## Follow-up — 2026-07-15T13:13:10Z

The user just added an important note to the requirements: "these should all work in 2D and 3D although it is permitted to make 3d versions a '3D extenstion' of a 2D model as we dow with vgg19". Please ensure the metrics suite supports both 2D and 3D images according to this instruction.

## 2026-07-15T15:34:06Z

Achieve 3D registration parity (including DKT label overlap) between `syntx` and `ants.registration` for the Mindboggle dataset, while maintaining the existing 2D parity. The fix must ensure that the SyN optimization operates correctly in fixed physical space and composes with the affine mapping properly, strictly adhering to the single interpolation policy.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. Native Physical Space Optimization
Rewrite the PyTorch and JAX SyN optimization loops so that the forward (`phi_1`) and inverse (`phi_2`) displacement fields operate natively in physical millimeter coordinates within the fixed image's domain, mirroring the ITK C++ reference implementation.

### R2. GPU Performance Balance
The implementation must remain blazing fast. Strike a balance between mathematical correctness in physical space and the efficiency of PyTorch/JAX tensor operations, avoiding heavy per-iteration overhead or CPU round-trips.

### R3. Affine Coordinate Composition
Correctly compose the initial affine transform to account for disparate fixed/moving physical spaces. The mapping to moving space must be strictly implemented as `y = A(phi_2_inv(phi_1(x)))`.

## Verification Resources
- Use `scratch/test_internal_dice.py` to evaluate the raw coordinate-mapping accuracy internally.
- `ants.label_overlap_measures` against the Mindboggle DKT manual labels.

## Acceptance Criteria

### Parity & Accuracy
- [ ] Programmatic DICE Verification: Running `synto` with an equivalent set of parameters must yield a label overlap metric (DICE score) that meets, exceeds, or is at least within 0.5% (0.005) of the standard `ants.registration` C++ baseline in both the 2D and 3D parity tests.

### Runtime Profiling
- [ ] Profiling Report: The agent must generate a profiling breakdown report confirming that the newly implemented physical space conversions do not dominate the GPU runtime during the optimization loop.



