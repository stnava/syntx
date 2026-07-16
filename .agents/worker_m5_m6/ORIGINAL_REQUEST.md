## 2026-07-15T13:25:49Z
Implement the evaluation script, HTML visual report, and user documentation/tutorial script for the Image Comparison Metrics Suite.
Specifically:
1. Create `examples/evaluate_metrics_generative.py`. This script must:
   - Instantiate `CrossProductGenerator` to generate image pairs covering combinations of the 6 intensity changes and 4 shape changes.
   - Run a representative subset of metrics (covering classical, spatial, and deep feature space models: VGG19, DINOv2, ResNet10, SwinUNETR) against the generative space.
   - Save the resulting comparison metrics to `outputs_comparison/generative_evaluation_results.csv`.
   - Run a short registration using `SyNTo` or registration helpers on one of the deformed pairs, extracting the warped/registered image, coordinate displacement grid, and Jacobian determinant map.
   - Generate a high-quality visual HTML report saved at `docs/registration_report.html` to summarize the metric evaluation.
2. The HTML report `docs/registration_report.html` MUST comply with the GEMINI.md visualization guidelines, displaying the following structural/spatial images:
   - Edge and/or region overlap between the registered/warped image and the target/fixed image (e.g. Canny or Sobel edges overlaid).
   - Deformed coordinate grids visualizing the coordinate warping.
   - Jacobian determinant maps illustrating local compression and expansion.
   - Deformed/Warped images shown side-by-side (next to) target/fixed images.
   Ensure all visual assets/figures are embedded as base64 PNGs in the HTML so the report is fully self-contained.
3. Write a clear, runnable tutorial example script under `examples/compare_metrics_tutorial.py` geared towards a non-expert user. It should explain:
   - How to use `syntx.image_compare` with classical, spatial, and deep feature metrics.
   - How to use the `CrossProductGenerator` to simulate transformations.
   - How to interpret the returned scores (e.g., lower score indicates better similarity).
4. Run `pytest` to make sure the entire project test suite passes.
5. Write your plans, progress, and handoffs in `/Users/stnava/code/syntx/.agents/worker_m5_m6/`.
