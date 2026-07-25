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
* **LNCC Autograd Derivative Singularity & Variance Floor**: In flat intensity regions (e.g., background zero-padding or uniform white matter), $\text{Var}(I) \rightarrow 0$. Because $\frac{\partial \text{LNCC}}{\partial I}$ contains $\frac{1}{\text{Var}(I)}$ in the analytical autograd derivative, un-floored variance causes derivative spikes that drive local grid folding. All LNCC loss implementations (PyTorch, JAX, etc.) **must** enforce a variance floor:
  $$\text{Var}_{\text{safe}}(I) = \max\left(\text{Var}(I), 10^{-6}\right)$$
* **LNCC Cauchy-Schwarz [-1.0, 1.0] Clamping**: 32-bit floating-point roundoff errors in spatial box filtering near sharp image edges can cause cross-correlation magnitudes $|r| > 1.0$ (e.g., $r = 1.0000004$). Always apply `clamp(cc, -1.0, 1.0)` to strictly enforce Cauchy-Schwarz bounds and prevent non-physical derivative forces.

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
* **ANTs Affine Center of Rotation:** When parsing an ANTs affine transform to a standard $4 \times 4$ homogeneous matrix $y = Ax + t_{new}$, you **must** account for the center of rotation $C$ (stored in `tx.fixed_parameters`). The translation vector must be explicitly updated as $t_{new} = t + C - A \cdot C$. Ignoring $C$ results in massive physical coordinate misalignments.
* **ITK CFL Gradient Step (Voxel Space):** In ITK, `gradientStep` (used in SyN/Demons CFL optimization) is scaled in **voxel space**, not absolute physical space. When normalizing the gradient field ($\Delta = \text{step} \cdot \frac{\nabla}{||\nabla||_{max}}$), you **must** multiply the step size by the grid's current physical spacing (`step * spacing`). This ensures that a step of $0.1$ voxels translates to a proportionately larger physical step (e.g. $0.4$ mm) at coarser pyramid levels (e.g. downsampled by $4\times$). Without this spacing multiplier, optimization will severely stall at coarse levels.
* **Coordinate Domain Matching for `grid_sample`:** When composing spatial grids (e.g., evaluating $G_1(X)$ where $G_1$ maps Fixed $\rightarrow$ Moving), the lookup coordinates $X$ **must** be normalized relative to the **domain** of $G_1$ (the Fixed space). Passing Moving-space normalized coordinates to sample a Fixed-space grid results in completely invalid coordinate mapping.
* **Affine Parameter Post-Step Clamping:** For affine optimizations in PyTorch and JAX, parameters must be clamped post-step (`scale` and `anisotropic_scale` $\in [0.05, 20.0]$, `shear` $\in [-5.0, 5.0]$, `omega` $\in [-\pi, \pi]$). Never place `clip`/`clamp` functions inside the forward loss autograd function, as zeroing gradients outside bounds causes Adam momentum wind-up.

## 7. Modality Simulation & Metric Evaluation
* **Generative Disparity Spaces:** When evaluating image similarity metrics via generative shape and intensity transformations, you must use a continuous/uniform distribution (e.g., `np.linspace(0.1, 6.0)`) across magnitude multipliers. Do not use discrete rigid buckets (small, medium, large), as this creates horizontal gaps and clustered artifacts in scatter plots.
* **Modality Simulation (Intensity Shuffling):** When simulating modality differences (e.g. T1 vs T2), use a multi-level piecewise intensity shuffling strategy (e.g., swapping intensity ranges `[0.0, 0.6, 1.0]` non-linearly) to create massive contrast inversions that properly test a metric's structural invariance.

## 8. Inverse Displacement Field Evaluation (SyN)
* **Algorithmic Inversion vs. Composition:** Do NOT use a numerical fixed-point solver (like ITK's `InvertDisplacementFieldImageFilter`) to invert a fully composed, heavily-deformed mapping at the end of registration. While fixed-point inversion is necessary *during* the SyN iterative loop (where incremental deformation steps are small and bounded by `max_error_threshold`), applying it from scratch to a massively deformed field will diverge or stall.
* **Algebraic Composition:** The final inverse mapping $M \rightarrow F$ must be constructed algebraically by composing the true intermediate inverses maintained symmetrically during the optimization loop (i.e., $\phi_{inv} = \phi_{l2r}^{-1} \circ \phi_{r2l}$). This guarantees perfect symmetry bounded only by interpolation precision.
* **ITK Fixed-Point Continuation Condition:** Fixed-point inverse field solvers must use `logical_or(max > max_threshold, mean > mean_threshold)` (matching ITK `InvertDisplacementFieldImageFilter`'s `while(max > thresh || mean > thresh)`). Stopping when `mean <= mean_threshold` alone leaves local boundary max errors un-converged.
* **In-Loop Fixed-Point Step Bounding:** Restrict inner-loop fixed-point inverse updates to `in_loop_inv_steps = min(3, inverse_steps)` per epoch during SyN optimization to avoid over-inverting intermediate deformation noise.
* **Displacement Field vs. Intensity Image Padding Modes**:
  - **Intensity Images**: Must use `padding_mode='zeros'` so out-of-bounds coordinates sample $0.0$ intensity without creating artificial edge-color stripes that pull background grid vectors.
  - **Displacement Fields**: Must use `padding_mode='border'` during fixed-point inversion and algebraic composition ($\phi_{inv} = \phi_{r2l} \circ \phi_{l2r}^{-1}$). Displacement fields represent physical coordinate offsets; zero-clamping out-of-bounds offsets corrupts boundary displacement vectors and inflates inverse identity errors.

## 9. Backend Parity Requirements
JAX, PyTorch, and C++ (ANTs/ITK) are compute engines — not algorithmic variants. All backend implementations must be strictly synchronized algorithmically across every pipeline stage (parameter initialization, optimizer updates, parameter clamping, in-loop inverse step bounds, and end-of-fit algebraic warp compositions). When adding a fix, safeguard, or feature to one backend (e.g., PyTorch), you MUST implement the exact same algorithmic logic symmetrically in all other backends (e.g., JAX). Results across backends must match within floating-point tolerance (~0.001 Dice). Any larger discrepancy (e.g., ≥0.01 Dice) indicates an implementation bug, not an inherent backend limitation. Never rationalize quality differences between backends as "expected numerical behavior." Instead, systematically diff the code paths to find the algorithmic mismatch.
* **No Ad-Hoc Per-Voxel Gradient Clamping**: Never introduce per-voxel gradient magnitude clamping (e.g. `8.0 * grad_ref`) or asymmetric CFL step capping in one backend without symmetrical inclusion across all backends. Asymmetric velocity clamping breaks mathematical symmetry between forward ($v_{l2r}$) and backward ($v_{r2l}$) velocity updates, causing inverse solvers to diverge.

## 10. Gaussian Smoothing Space and Unit Conventions (ITK Parity)
* **Variance-to-Sigma Unit Conversion**: ANTs/ITK registration parameters for update and total field smoothing (`flow_sigma` / `total_sigma`) represent **variance** ($\sigma^2$), not standard deviation ($\sigma$). Always convert these parameters using $\sigma = \sqrt{\text{variance}}$ before performing Gaussian convolution. Failing to do so results in massive over-smoothing (e.g., $1.73\times$ over-smoothing when variance = 3.0).
* **Voxel Index Space Smoothing**: In ITK, `GaussianOperator` performs convolution in **voxel units**, not physical units. Do not pass spacing vectors to Gaussian filters in PyTorch/JAX to scale $\sigma$. Keep the smoothing isotropic in voxel space at all multi-resolution/downsampled levels to ensure mathematical parity between backends.
