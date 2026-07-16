# Original User Request

## 2026-07-15T13:13:21Z

You are the Project Orchestrator for the Image Comparison Metrics Suite project.
Your working directory is `/Users/stnava/code/syntx/.agents/orchestrator_metrics_1`.

### Task Overview
Implement a comprehensive suite of at least 64 image comparison metrics in `syntx.image_compare` for assessing registration quality, and systematically evaluate them against a 2D generative cross-product space of known intensity and shape changes (with Grenander's metric deformation as ground truth).

### 2D and 3D Support Requirement
All 64 metrics must support both 2D and 3D images. As per the user's note: "these should all work in 2D and 3D although it is permitted to make 3d versions a '3D extenstion' of a 2D model as we dow with vgg19". Please ensure the metrics suite supports both 2D and 3D images accordingly.

### Project Guardrails and Constraints (from GEMINI.md)
You MUST strictly adhere to the following project rules:
1. **Single Interpolation Policy:** To prevent spatial blurring and loss of high-frequency boundary information, all registration workflows in `syntx` must avoid pre-warping images or intermediate segmentations prior to optimization. Compose and apply all transforms in a single step (e.g. passing the list to `ants.apply_transforms`).
2. **Similarity Metric & VGG Feature Space Guidelines:**
   - Cortical label map accuracy drop of >= 0.01 (1%) is unacceptable.
   - VGG 2D orthogonal slice LNCC (`vgg_mode='lncc'`) is NOT an acceptable substitute for standard intensity-based LNCC.
   - Only VGG 3D LNCC with Layer 4 (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) meets standard intensity LNCC performance. Do not default to VGG 2D or coarser layers when accuracy is the target.
3. **Reporting and Visualization Guidelines:** Any HTML or artifact reports summarizing registration performance comparisons must always display structural/spatial images to visually inspect registration quality, including:
   - Edge and/or region overlap between the registered image and the target image.
   - Deformed grids visualizing the coordinate warping.
   - Jacobian determinant maps illustrating local compression and expansion.
   - Deformed/Warped images shown side-by-side (next to) target/fixed images.

### Acceptance Criteria
#### Metric Suite & API
- A programmatic test verifies that `syntx.image_compare` can be called with at least 64 unique, valid `metricname` strings without erroring.
- A programmatic test verifies that for identical images, all metrics return a score of 0 (or their minimum possible value), and scores strictly increase as divergence increases.

#### Generative Space
- A test asserts that the generative pipeline outputs a cross-product of the specified intensity (noise, bias, inhomogeneity, modality change, step function, missing data) and shape changes (translation, rotation, affine, deformation), and that every generated pair maintains >= 80% spatial overlap.
- Ground truth magnitudes (e.g., L2 norm of the displacement field) are explicitly returned for each generated pair.

#### Evaluation & Documentation
- An evaluation script successfully runs a subset of metrics against the generative space and outputs a visual report.
- The report contains at least one example of a deformed grid, Jacobian determinant map, and side-by-side image comparison, complying with the visualization guidelines in GEMINI.md.
- Documentation includes a clear, runnable example script geared towards a non-expert user.

### Your Subagent Coordination
- You may spawn worker and reviewer subagents as needed to divide the work, but they must all write their plans, progress, and handoffs in their own separate folders under `.agents/`.
- Ensure all coordinate grids, Jacobian maps, and structural side-by-sides are exported and verified.
- Write your main coordination plans, milestones, and daily progress to `/Users/stnava/code/syntx/.agents/orchestrator_metrics_1/plan.md` and `/Users/stnava/code/syntx/.agents/orchestrator_metrics_1/progress.md`.
- When all milestones are complete, send a message back to me (the Sentinel) claiming completion.
