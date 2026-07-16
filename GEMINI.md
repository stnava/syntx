# Syntx Registration Guardrails

## 1. Single Interpolation Policy
To prevent spatial blurring and loss of high-frequency boundary information, all registration workflows in `syntx` must avoid pre-warping images or intermediate segmentations prior to optimization.

* **Constraint:** No intermediate file-based pre-warping (e.g., calling `ants.apply_transforms` to generate a pre-aligned image for optimization inputs).
* **Composition:** If multiple transforms are required (such as an initial translation and learned affine/deformable warps), they must be composed and applied directly to the native-space images in a single step (e.g., passing the list `[deformable, affine, initial_translation]` to a single `ants.apply_transforms` call).
* **Initialization:** Initial alignments (such as center-of-mass matching) should be optimized or initialized directly on the transformation grid parameters in PyTorch/JAX without altering the input image arrays.

## 2. Similarity Metric & VGG Feature Space Guidelines
* **Accuracy Thresholds:** For registration tasks targeting cortical label maps, a drop in Mean DICE score of $\ge 0.01$ (1%) is considered a massive, unacceptable regression.
* **VGG 2D Mode Limitation:** VGG 2D orthogonal slice LNCC (`vgg_mode='lncc'`) is **not** an acceptable substitute for standard intensity-based LNCC ($5 \times 5 \times 5$ window) when high accuracy is required, as it incurs a major drop in DICE (e.g., from `0.476` to `0.438`, or ~4%).
* **VGG 3D Mode Requirement:** Only **VGG 3D LNCC with Layer 4** (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) meets the performance level of standard intensity LNCC (`0.4746` vs `0.4761`), while significantly regularizing grid folds (from `0.096%` to `0.003%`). Do not recommend or default to VGG 2D or coarser layers (like Layer 8) when accuracy is the target.
* **Deep Feature Registration Metrics (SyNTo):** 
  - Use `similarity_metric='dino_2_lncc'` for general robust deep registration (resilient to noise, bias, and missing data).
  - Use `similarity_metric='vgg_4_lncc'` specifically for massive modality inversions or intensity shuffling, as VGG preserves structural high-frequency edges much better than semantic patches during contrast inversion.
* **Baseline Metrics:** `ssim` and `ncc` are fast, highly reliable, fully differentiable baseline metrics natively supported in the `SyNTo` engine.

## 3. Reporting and Visualization Guidelines
* **Required Report Visualizations:** Any HTML or artifact reports summarizing registration performance comparisons must always display structural/spatial images to visually inspect registration quality, including:
  - **Edge and/or region overlap** between the registered image and the target image.
  - **Deformed grids** visualizing the coordinate warping.
  - **Jacobian determinant maps** illustrating local compression and expansion.
  - **Deformed/Warped images** shown side-by-side (next to) target/fixed images.

## 4. Label Evaluation Constraints
To ensure accurate and standardized registration benchmarking against ground-truth segmentations (e.g., Mindboggle DKT labels):
* **Interpolation:** When applying transforms to discrete/integer label maps, you **must** use nearest neighbor interpolation (e.g., `interpolator='nearestNeighbor'` in `ants.apply_transforms`). Never use linear or b-spline interpolation on segmentations.
* **Overlap Metrics:** Use `ants.label_overlap_measures` to systematically compute structural DICE scores (TargetOverlap) when assessing registration quality.

## 5. Image Comparison Metric Guidelines (`syntx.image_compare`)
To maintain a unified API and consistent cross-dimensional support:
* **Standardized Returns (Lower is Better):** All metrics evaluated through `image_compare` must return scores where a lower value strictly indicates higher similarity. For metrics traditionally maximized (e.g., PSNR, NCC), return the inverted or negative value (e.g., `-PSNR` or `1 - NCC`).
* **2D and 3D Dimensionality:** All metrics must support both 2D and 3D inputs. When integrating 2D-native deep feature models (like VGG19), it is standard and permitted to implement a "3D extension" (such as a triplanar ensemble) to support 3D images, rather than restricting to native 3D architectures.

## 6. Registration Optimization & Initialization Constraints
* **Physical Space Awareness:** Optimization pipelines using PyTorch/JAX normalized `[-1, 1]` grids must explicitly map physical space differences (origin, spacing, direction) to the grid space. Do not assume normalized grids naturally align images from different physical scanner spaces.
* **CoM Initialization Selection:** For affine alignments, dynamically select the best initialization by testing both Field of View (FOV) and Foreground (intensity-weighted) Center of Mass physical translations via a fast Mutual Information evaluation (e.g., downsampled `mattes_mi_loss_nd`).
* **Preserving Gradients in Lie Algebra:** When parameterizing spatial rotations via Lie Algebra, avoid non-differentiable conditionals at zero angles (e.g., `torch.where(omega == 0, I, R)`) that lock gradients to zero. Always implement a first-order Taylor expansion (`I + K_raw`) for infinitesimally small angles to ensure continuous gradient flow at identity initialization.

## 7. Modality Simulation & Metric Evaluation
* **Generative Disparity Spaces:** When evaluating image similarity metrics via generative shape and intensity transformations, you must use a continuous/uniform distribution (e.g., `np.linspace(0.1, 6.0)`) across magnitude multipliers. Do not use discrete rigid buckets (small, medium, large), as this creates horizontal gaps and clustered artifacts in scatter plots.
* **Modality Simulation (Intensity Shuffling):** When simulating modality differences (e.g. T1 vs T2), use a multi-level piecewise intensity shuffling strategy (e.g., swapping intensity ranges `[0.0, 0.6, 1.0]` non-linearly) to create massive contrast inversions that properly test a metric's structural invariance.
