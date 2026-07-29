# Syntx Registration Guardrails

## 1. Single Interpolation Policy
To prevent spatial blurring and loss of high-frequency boundary information, all registration workflows in `syntx` must avoid pre-warping images or intermediate segmentations prior to optimization.

* **Constraint:** No intermediate file-based pre-warping (e.g., calling `ants.apply_transforms` to generate a pre-aligned image for optimization inputs).
* **Composition:** If multiple transforms are required (such as an initial translation and learned affine/deformable warps), they must be composed and applied directly to the native-space images in a single step (e.g., passing the list `[deformable, affine, initial_translation]` to a single `ants.apply_transforms` call).
* **Midpoint Image Export:** Exporting deformed midpoint images (e.g., `midpoint_moving`) must strictly compose the non-linear midpoint warp and affine transform in a single step (`transformlist=[inv_midpoint_warp, affine_file]`) directly on native-space images (`moving=moving`). Never perform intermediate 2-step pre-warping calls.
* **Initialization:** Initial alignments (such as center-of-mass matching) should be optimized or initialized directly on the transformation grid parameters in PyTorch/JAX without altering the input image arrays.

## 2. Similarity Metric & VGG Feature Space Guidelines
* **Accuracy Thresholds:** For registration tasks targeting cortical label maps, a drop in Mean DICE score of $\ge 0.01$ (1%) is considered a massive, unacceptable regression.
* **Hybrid Loss Superiority:** Combining sharp intensity LNCC ($5 \times 5 \times 5$ window) with deep feature LNCC (e.g., $0.5 \cdot \text{LNCC} + 0.5 \cdot \text{VGG\_4\_LNCC}$) outperforms standalone metrics (yielding $+1.21\%$ Cortical Dice gain over standard intensity LNCC). Intensity LNCC aligns high-frequency cortical boundaries while deep feature LNCC regularizes global shape alignment, preventing optimization from getting trapped in local sulcal minima.
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
* **Standardized Visualization Routines**: Always use standardized `syntx.visualization` display functions (`plot_comparison`, `plot_edge_overlay`, `plot_deformation_grid`, `render_standard_4panel`) for all report figures. Never write custom or ad-hoc matplotlib rendering logic inside report generator scripts.
* **Visualization Modularization**: Any new visualization capability or layout mode must be developed as a reusable tool in `syntx.visualization`, backed by unit tests, packaged cleanly, and support **2D, 3D, and 4D** inputs seamlessly.
* **ANTsPy `ants.plot` Display Orientation Invariant:**
  - All 2D slice extraction routines in `syntx.visualization` (such as `extract_2d_slice`) MUST strictly adhere to the standard `ants.plot` display orientation convention.
  - Specifically, 2D scalar images MUST be formatted via `arr.T` (matching `ants.plotting.plot.rotate90_matrix`) and 2D vector fields MUST be transposed via `np.transpose(arr[..., :2], (1, 0, 2))`.
  - Deformation mesh grid coordinates (`plot_deformation_grid` and `render_standard_4panel`) MUST align `disp_x` (component 0) with horizontal grid coordinates (`grid_x`) and `disp_y` (component 1) with vertical grid coordinates (`grid_y`) on the transposed display grid.
* **Table & Manuscript Formatting Invariants:**
  - **Formatted Tables:** Always format tables as clean Markdown (never raw ASCII boxes like `+---+`). Limit table width to 5–6 columns to prevent right-margin truncation in Pandoc XeLaTeX PDF rendering.
  - **Sequential Automated Figure Numbering:** Index all figures in strict 1-to-N sequential order in text flow, ensuring figure captions match all text cross-references.
  - **Clean Reference Management:** Never wrap BibTeX code in raw Markdown code blocks (```bibtex```). Store entries in a standalone `references.bib` file and render Section 8 as a clean numbered bibliography list.

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
* **Affine Parameter Post-Step Clamping**: For affine optimizations in PyTorch and JAX, parameters must be clamped post-step (`scale` and `anisotropic_scale` $\in [0.05, 20.0]$, `shear` $\in [-5.0, 5.0]$, `omega` $\in [-\pi, \pi]$). Never place `clip`/`clamp` functions inside the forward loss autograd function, as zeroing gradients outside bounds causes Adam momentum wind-up.
* **LARS Optimizer for Time-Varying Velocity Fields (TVF)**:
  - **Scale-Invariant Momentum vs. Adam Stalling**: Standard Adam updates parameters via unscaled moment ratios ($m_t / \sqrt{v_t}$). In smooth, low-gradient similarity loss plateaus, $v_t$ shrinks, causing Adam step sizes to stall before resolving high-frequency sulcal boundaries.
  - **Layer-wise Trust Ratio Scaling**: LARS rescales velocity updates per keyframe tensor $v(t_k)$ using the trust ratio $\text{trust\_ratio} = \eta \cdot \frac{\|v(t_k)\|}{\|g(t_k)\| + \epsilon}$, allowing high global learning rates ($lr \in [0.50, 1.20]$) while maintaining scale-invariant optimization momentum and preserving diffeomorphic invertibility ($\det(J) > 0$).

## 7. Modality Simulation & Metric Evaluation
* **Generative Disparity Spaces:** When evaluating image similarity metrics via generative shape and intensity transformations, you must use a continuous/uniform distribution (e.g., `np.linspace(0.1, 6.0)`) across magnitude multipliers. Do not use discrete rigid buckets (small, medium, large), as this creates horizontal gaps and clustered artifacts in scatter plots.
* **Modality Simulation (Intensity Shuffling):** When simulating modality differences (e.g. T1 vs T2), use a multi-level piecewise intensity shuffling strategy (e.g., swapping intensity ranges `[0.0, 0.6, 1.0]` non-linearly) to create massive contrast inversions that properly test a metric's structural invariance.

## 8. Inverse Displacement Field Evaluation (SyN)
* **Algorithmic Inversion vs. Composition:** Do NOT use a numerical fixed-point solver (like ITK's `InvertDisplacementFieldImageFilter`) to invert a fully composed, heavily-deformed mapping at the end of registration. While fixed-point inversion is necessary *during* the SyN iterative loop (where incremental deformation steps are small and bounded by `max_error_threshold`), applying it from scratch to a massively deformed field will diverge or stall.
* **Newton Iteration vs. Fixed-Point Inversion Scope:** While Newton's method ($[I + \nabla \mathbf{u}]^{-1} \cdot \text{error}$) achieves $15\times$ faster convergence on small-to-moderate deformations ($\le 2\text{ mm}$ displacement), steep spatial gradients under large deformations ($\ge 8 - 10\text{ mm}$) cause $[I + \nabla \mathbf{u}]$ to become ill-conditioned ($\det \le 0$), causing Newton's method to diverge locally. ITK Fixed-Point iteration with norm clipping and relaxation ($\epsilon = 0.5$) combined with algebraic composition ($\phi_{\text{inv}} = \phi_{l2r}^{-1} \circ \phi_{r2l}$) is required for global stability in 3D neuroimaging registration.
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

## 11. Midpoint Warp Field Preservation & Geodesic Midpoint Anchoring
* **Half-Warp Preservation:** At end-of-fit in SyN, the half-warp displacement fields (`w_l2r`, `w_r2l`) that define the geodesic midpoint must be saved as separate model attributes (`self.midpoint_warp_l2r`, `self.midpoint_warp_r2l`) **before** the full-geodesic composition overwrites `self.warp_l2r`, `self.warp_l2r_inv`, `self.warp_r2l`, and `self.warp_r2l_inv`. The `registration()` function must read the preserved half-warps for midpoint image export, never the fully-composed fields.
* **Antisymmetric Velocity Projection:** To enforce geodesic smoothness at the midpoint and anchor it at the Fréchet mean, the CFL velocity updates $(\delta_l, \delta_r)$ must be projected onto the antisymmetric subspace by removing the common-mode (symmetric) drift component:
  $$e_0 = \delta_l + \delta_r$$
  $$\delta_l \leftarrow \delta_l - 0.5 \cdot e_0$$
  $$\delta_r \leftarrow \delta_r - 0.5 \cdot e_0$$
  This guarantees $\delta_l + \delta_r = 0$ exactly, requires zero hyperparameters, costs only two additions, and preserves the CFL bound. Do **not** use Charbonnier MCR (Midpoint Continuity Regularization) — it requires Laplacian convolutions, two hyperparameters, and historically contained a sign error that amplified rather than reduced midpoint drift.

## 12. TVF Model & Velocity Field Optimization Guardrails
To ensure high accuracy and computational efficiency in Time-Varying Velocity Field (TVF) registrations:
* **Pyramid-Proportional Velocity Grids**: Velocity field parameter tensors MUST be sized proportionally to the active image pyramid level:
  $$\text{vel\_shape}_{\text{level}} = \max\left(8, \left\lfloor \frac{\text{max\_vel\_shape}}{\text{level}} \right\rfloor \right)$$
  When transitioning between pyramid levels in `fit()`, velocity parameters must be resized using trilinear/bilinear interpolation (`_resize_velocity`) to preserve learned deformations. Never maintain a fixed high-resolution grid (e.g., $[96^3]$) at coarse pyramid levels, as it causes massive over-parameterization and redundant `grid_sample` calls.
* **Efficient TVF Solver Defaults**: When paired with pyramid-proportional velocity grids, the Euler ODE solver (`solver='euler'`) with $T=4$ keyframes and $1$ integration step per interval achieves accuracy parity with RK4 ($T=8$, substeps=2) while reducing `grid_sample` kernel launches by $16\times$.
* **LNCC Window Size for Cortical Regions**: For TVF similarity evaluation targeting cortical label maps, set `lncc_radius=2` (`window_size=5`). Using window size 9 over-smooths local gradients.
* **Anti-Aliasing Image Pyramid Smoothing**: Before downsampling image tensors across multi-resolution pyramid levels, apply Gaussian anti-aliasing pre-smoothing with $\sigma = \log_2(\text{level})$ to eliminate aliasing noise in spatial image gradients.

## 13. Centralized Spatial Conversion Suite (`syntx.spatial`)
All coordinate and displacement field conversions between ITK/ANTs physical space and PyTorch/JAX tensor space **must** use the centralized `syntx.spatial` module. Never introduce ad-hoc inline `[..., ::-1]` or `.transpose()` calls for spatial domain conversion.

* **ANTs C-Contiguous Axis-Aligned Convention**: When `ants.image_read().numpy()` returns a multi-component displacement field, **component `i` corresponds to displacement along spatial axis `i`** of the C-contiguous array. For 2D `(H, W, 2)`: comp 0 = displacement along rows (Y), comp 1 = along columns (X). For 3D `(Z, Y, X, 3)`: comp 0 = along Z, comp 1 = along Y, comp 2 = along X. This is NOT the same as physical `(dx, dy, dz)` ordering.
* **Tensor-to-ITK Conversion**: Converting a model tensor (e.g., `model.warp_l2r`) to an ANTs displacement image requires **two** operations via `syntx.spatial.disp_tensor_to_itk(disp, ref_image)`:
  1. **Spatial axis transposition**: tensor `(Z_t, Y_t, X_t)` → ANTs `(X, Y, Z)` via `transpose(2, 1, 0, 3)` (3D) or `transpose(1, 0, 2)` (2D).
  2. **Component reversal**: tensor `(dz, dy, dx)` → ITK axis-aligned via `[..., ::-1]`.
* **ITK-to-Tensor Conversion**: The inverse via `syntx.spatial.disp_itk_to_tensor(disp_img)` performs the same two operations in reverse order.
* **Jacobian Determinant**: Always use `syntx.spatial.jacobian_determinant(disp, ref_image=fixed)` which implements the axis-aligned convention: $J[i,i] = 1 + \partial u_i / \partial \text{axis}_i$, with spacing per axis: axis 0 uses `spacing[dim-1]`, axis 1 uses `spacing[dim-2]`, etc. Validated against ANTs C++ ITK reference with Pearson $r > 0.999$.
* **No Inline Conversions**: When adding new registration backends or transform export paths, always delegate to `syntx.spatial.*` functions rather than reimplementing conversions. The spatial module handles all edge cases (batch dimensions, component-first format, tensor auto-detection) consistently.

## 14. Standardized Registration Pipeline Workflow
When building or evaluating image registration pipelines in `syntx`:
1. **Pre-Registration Standardization**: Always standardize input images before optimization using `syntx` utilities (e.g., intensity normalization to `[0, 1]` via `normalize_tensor`).
2. **Initial Affine Alignment**: Use ANTsPy `ants.registration` affine (or `syntx`'s `HierarchicalAffine`) to compute initial rigid/affine spatial alignment.
3. **Deformable Registration**: Delegate non-linear deformable warping to `syntx.syn` or `TVFModel`.
4. **Standard Metrics & Physical Space Preservation**: Always evaluate registration performance using centralized `syntx.reporting` and `syntx.spatial` metrics, strictly preserving physical scanner coordinates (origin, spacing, direction).
5. **Unified Reporting**: All execution provenance, deformation metrics, and visual figures must be generated via standard `syntx.reporting.create_registration_report` and `syntx.visualization` utilities.

