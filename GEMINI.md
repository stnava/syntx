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

## 3. Reporting and Visualization Guidelines
* **Required Report Visualizations:** Any HTML or artifact reports summarizing registration performance comparisons must always display structural/spatial images to visually inspect registration quality, including:
  - **Edge and/or region overlap** between the registered image and the target image.
  - **Deformed grids** visualizing the coordinate warping.
  - **Jacobian determinant maps** illustrating local compression and expansion.
  - **Deformed/Warped images** shown side-by-side (next to) target/fixed images.
