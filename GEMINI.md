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
* **Dedicated Visualization Sub-Package (`syntx.viz`):** All figure generators (`render_input_pair_figure`, `render_standard_4panel`, `render_label_alignment_figure`, `plot_deformation_grid`, `plot_edge_overlay`), statistical displays (`plot_label_overlap_stats`, `plot_jacobian_distribution`), gallery builders (`create_visualization_gallery`), and interactive HTML report tools (`create_registration_report`, `build_engine_provenance`) MUST reside in and be systematically accessible via `syntx.viz`.
* **Anatomical Segmentation & Statistical Display Standards (`render_label_alignment_figure`, `plot_label_overlap_stats`, `plot_jacobian_distribution`):**
  - **Label Overlays**: Anatomical segmentations (Mindboggle DKT labels) must be rendered in 2x3 tri-planar layouts in canonical LPI space with physical anisotropy scaling and discrete qualitative colormapping (`gist_ncar` / `tab20`).
  - **Statistical Quality Summaries**: Mindboggle benchmark evaluations must include per-region Cortical DKT Dice bar charts, symmetric space ($\text{Dice}_{\text{fixed}}$ vs $\text{Dice}_{\text{moving}}$) boxplots, and Jacobian determinant $\det(J)$ singularity histograms.
* **Standard Reporting Infrastructure Requirement (`render_standard_4panel`, `render_input_pair_figure`):** All registration reports, benchmarks, and comparison artifacts (2D and 3D) MUST render visual panels using `render_standard_4panel()` and `render_input_pair_figure()` from `syntx.viz`.
* **Standard Figure 1 Layout Invariant (`render_input_pair_figure`):**
  - **3D Volume Inputs**: Rendered as a $2 \times 3$ panel layout within one single figure panel: **Fixed Image at top** (Axial, Coronal, Sagittal views) and **Moving Image at bottom** (Axial, Coronal, Sagittal views).
  - **2D Image Inputs**: Rendered as a $1 \times 2$ panel layout within one single figure panel: **Fixed Image on Left** and **Moving Image on Right**.
  - **Colorbar Invariant**: Exactly **1 colorbar per image** (1 shared row colorbar for the Fixed Image, 1 shared row colorbar for the Moving Image). Never render separate colorbars on every individual subplot panel.
* **Anatomical Orientation Invariants (`render_input_pair_figure`, `render_standard_4panel`):**
  - All 3D orthographic slice visualizations MUST be reoriented into canonical LPI anatomical space (`reorient=True`).
  - **Axial View**: Rendered with **Anterior (Front of Head) UP** and Posterior (Back of Head) DOWN.
  - **Coronal View**: Rendered with **Superior (Top of Head) UP** and Inferior DOWN.
  - **Sagittal View**: Rendered with **Superior (Top of Head) UP** and Anterior RIGHT.
* **2D & 3D `ants.plot` Orientation & Metadata Inheritance Invariants (`syntx.viz.core`, `syntx.viz.figures`, `syntx.transform`):**
  - **PyTorch/NumPy ZYX $\rightarrow$ XYZ Array Order Parity**: PyTorch tensors and raw NumPy arrays are indexed in matrix ZYX order, whereas ANTsPy's `ants.from_numpy(arr)` strictly expects ITK XYZ array ordering. Any raw 2D/3D arrays (scalar maps or displacement fields) MUST have their spatial axes transposed BEFORE passing to `ants.from_numpy`.
  - **3D Displacement Field Transpose**: When exporting PyTorch 3D deformation grids `[batch, Z, Y, X, 3]`, you MUST transpose the spatial axes via `transpose(2, 1, 0, 3)` so the array is ordered `[X, Y, Z, 3]` before `ants.from_numpy`. Reversing only the vector components (from `v_z, v_y, v_x` to `v_x, v_y, v_z`) is NOT enough; the spatial layout will still be read inverted (X read as Z).
  - **2D Scalar Map Transpose**: Any raw 2D array MUST be transposed (`arr.T`) BEFORE passing to `ants.from_numpy(arr.T, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)` so that scalar maps align with target ANTsImages.
  - **Automatic ANTsImage Metadata Inheritance**: In `render_standard_4panel()`, any raw NumPy array inputs (`detJ`, `inv_err_map`, `warped`, `moving`) MUST be automatically transposed (`arr.T`) and wrapped into `ants.ANTsImage` using `ants.from_numpy(arr.T, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)` prior to slice extraction to guarantee 100% spatial grid and orientation alignment with `fixed`.
  - **2D Image Matrix Transpose (`ants.plot` Parity)**: All 2D scalar images (`fixed`, `moving`, `warped`, `detJ`, `inv_err_map`) MUST be extracted via `AnatomicalVisualizer.extract_slice()`, which applies matrix transpose (`arr.T`) matching ANTsPy's `rotate90_matrix(x) = x.T`. Never apply ad-hoc `np.rot90` calls in individual plotting functions.
  - **2D Vector Field Transpose Parity**: 2D displacement fields (`warp` of shape `[H, W, 2]`) MUST transpose spatial dimensions and swap vector channels as `np.transpose(warp, (1, 0, 2))[..., [1, 0]]` to guarantee exact spatial alignment with scalar maps ($\det J$, inverse error).
  - **No Double Rotations**: Figure generators (`render_input_pair_figure`, `render_standard_4panel`) MUST consume the sliced array returned by `extract_oriented_slice()` directly without secondary rotation or flipping.
* **Physical Voxel Spacing Anisotropy Scaling**:
  - All slice visualizations MUST set `imshow(sl, aspect=aspect)` according to physical voxel spacing ratios ($\frac{s_y}{s_x}$ for Axial, $\frac{s_z}{s_x}$ for Coronal, $\frac{s_z}{s_y}$ for Sagittal) to prevent physical distortion when voxel acquisitions are non-isotropic.
* **Standard Figure 2 Layout Invariant (`render_standard_4panel`):**
  - **Header Row (Far Top)**: **Fixed Image Input at Far Top Left** and **Moving / Warped Image Input at Far Top Right**.
  - **Panel A**: Standard Deformed Mesh Grid (Cyan grid lines overlay)
  - **Panel B**: Standard Divergent Jacobian Determinant Map (`seismic` colormap centered at 1.0)
  - **Panel C**: Standardized Inverse Identity Error Map (mm) (`inferno` / `hot` colormap)
  - **Panel D**: High-Contrast Canny Edge Alignment Overlap (Cyan/Magenta or Red contours)
* **Standard 5-Figure Visual Suite Requirement for HTML Reports:** All 2D and 3D standard registration reports MUST embed and display the complete 5-figure visual verification suite:
  - **Figure 1**: Original Fixed Target and Moving Source input pair (`render_input_pair_figure`).
  - **Figure 2**: Standard 4-Panel Diagnostic Report (`render_standard_4panel`: Panel A Mesh Grid, Panel B Seismic Log-$\det(J)$ Map, Panel C Real Physical Inverse Identity Error Map in mm, Panel D Canny Edge Alignment Overlap).
  - **Figure 3**: Standard Time-Varying Velocity Field Keyframe Flow Visualization (`plot_time_varying_velocity_grid`: magnitude heatmaps overlaid with Cyan flow quiver vectors).
  - **Figure 4**: Multi-Resolution Similarity Loss Convergence Curves (Epoch-by-epoch LNCC loss progression across pyramid levels).
  - **Figure 5**: Segmentation & Cortical Dice Overlap Curves (Epoch-by-epoch progression for Fixed, Moving, and Symmetric Mean Dice).
* **Standard Quantitative Deformation Metrics Suite:** All registration reports MUST report the complete suite of utility-computed metrics:
  - **Bidirectional Dice Scores**: Fixed Space Dice, Moving Space Dice, and Symmetric Mean Dice ($\text{Dice}_{\text{sym}}$).
  - **Real Physical Inverse Identity Error Map (mm)**: $\mathbf{e}(x) = \|\phi_{\text{inv}}(x + \phi_{\text{fwd}}(x)) + \phi_{\text{fwd}}(x)\|_2$ (Mean, 95th Percentile, and Peak Max Error).
  - **Manifold Regularity**: Grid Folding Percentage ($\det(J) \le 0$) and Minimum Jacobian Determinant ($\min \det(J)$).
  - **Compute Runtime**: Total execution time in seconds.
* **ANTsPy Jacobian Determinant Log Parameter Invariant (`do_log=False` / `dolog=False`):**
  - ANTsPy's `ants.create_jacobian_determinant_image` (and `ants.create_jacobian`) returns log-Jacobian values ($\ln \det(J)$) by default unless `do_log=False` (or `dolog=False`) is explicitly set.
  - When computing raw physical Jacobian determinant maps ($\det(J)$) for grid folding percentage ($\det(J) \le 0$) or minimum determinant ($\min \det(J)$) metrics, functions MUST explicitly pass `do_log=False` (or `dolog=False`) or exponentiate log-Jacobian outputs (`np.exp(log_jac)`).
* **2D Otsu Segmentation Guidelines (`r16` / `r64` Benchmarks):**
  - **Cortical Gray Matter (Class 2)**: Isolated via `ants.threshold_image(img, "Otsu", 3).threshold_image(2, 2)`.
  - **Parenchymal Brain Tissue (Class 2+3)**: Isolated via `ants.threshold_image(img, "Otsu", 3).threshold_image(2, 3)`.
* **SyN Provenance Parameter Invariants (`syntx.syn`):**
  - `flow_sigma = 3.0` (fluid velocity smoothing variance $\sigma^2 = 3.0$)
  - `total_sigma = 0.0` (pure fluid deformation without total elastic field smoothing)
  - `grad_step = 0.25`
  - `in_loop_inv_steps = 10` (compute inverse fixed-point update at every iteration inside the optimization loop)
  - `initial_transform` from `syntx.robust_affine(mode='pytorch')`
* **Required Report Visualizations:** Any HTML or artifact reports summarizing registration performance comparisons must always display structural/spatial images to visually inspect registration quality.
  - **Edge and/or region overlap** between the registered image and the target image.
  - **Deformed grids** visualizing the coordinate warping.
  - **Jacobian determinant maps** illustrating local compression and expansion.
  - **Deformed/Warped images** shown side-by-side (next to) target/fixed images.
* **Table & Manuscript Formatting Invariants:**
  - **Formatted Tables:** Always format tables as clean Markdown (never raw ASCII boxes like `+---+`). Limit table width to 5–6 columns to prevent right-margin truncation in Pandoc XeLaTeX PDF rendering.
  - **Sequential Automated Figure Numbering:** Index all figures in strict 1-to-N sequential order in text flow, ensuring figure captions match all text cross-references.
  - **Clean Reference Management:** Never wrap BibTeX code in raw Markdown code blocks (```bibtex```). Store entries in a standalone `references.bib` file and render Section 8 as a clean numbered bibliography list.
* **TVF Keyframe Velocity Grid & Bending Energy Invariants (`plot_time_varying_velocity_grid`):**
  - **Real Thin-Plate Bending Energy (`Bnd`)**: The `Bnd` metric in keyframe figure titles MUST compute the exact domain-wide thin-plate bending energy across all spatial dimensions:
    $$\text{Bnd}(v) = \frac{1}{|\Omega|} \int_{\Omega} \left( \|\nabla^2 v_x\|_F^2 + \|\nabla^2 v_y\|_F^2 \right) dx dy$$
    Formatted with 3 significant digits in scientific notation (e.g. `Bnd=3.842e-03`). Never compute `Bnd` from single corner voxels or outer boundary edge arrays.
  - **Matplotlib Quiver Arrow Amplification (`scale=0.008`)**: In matplotlib `ax.quiver(..., scale_units='xy', scale=scale)`, `scale` is an inverse scaling denominator. Velocity vector arrow visualization MUST set `scale \le 0.010` (default `scale=0.008` for $125\times$ length amplification) so flow arrows along sulcal and cortical boundaries are long, crisp, and clearly visible.
  - **Dynamic Local Heatmap Scaling (`vmax=max_v_mag`)**: Continuous magnitude velocity heatmaps MUST set `vmax = max_v_mag` per keyframe to maximize dynamic range across `plasma` colormapping.
  - **Standard 4-Figure TVF Report Suite**: All TVF HTML benchmark reports MUST generate and display 4 dedicated visual figures:
    - **Figure 1**: Input Pair (`render_input_pair_figure`)
    - **Figure 2**: Standard 4-Panel Diagnostic (`render_standard_4panel`, 2x2 grid, 6 significant digits)
    - **Figure 3**: Keyframe Velocity Fields (`plot_time_varying_velocity_grid`, $125\times$ quiver arrows, real domain `Bnd`)
    - **Figure 4**: Multi-Resolution Loss Convergence Curves (Epoch-by-epoch LNCC loss progression across pyramid levels)
* **TVF Peak Provenance Parameter Invariants (`syntx.tvf`)**:
  - `multipoint_loss = [0.0, 0.5, 1.0]` (evaluate LNCC similarity at trajectory start t=0.0, midpoint t=0.5, and endpoint t=1.0)
  - `flow_sigma = 0.4` (fluid velocity smoothing)
  - `total_sigma = 0.05` (elastic grid smoothing)
  - `grad_step = 0.45`
  - `cfl_momentum = 0.95` (scale-invariant velocity momentum)
  - `n_time_steps = 3`
  - `use_analytical_gradients = True`
  - `constant_speed = True` (`constant_speed_relaxation = 0.10`)
  - `reg_iterations = [100, 100, 20]` (or `[200, 200, 40]`)
  - `initial_transform` from `syntx.robust_affine(mode='pytorch')`
* **Systematic Provenance Persistence (`docs/provenance/best_parameters.json`)**:
  - Whenever optimization, parameter sweeps, or benchmark experiments discover new peak performance configurations, the agent MUST immediately persist the complete algorithm parameters and full provenance dictionary (`ret['provenance']`) to `docs/provenance/best_parameters.json`.
  - The file MUST maintain structured JSON records per algorithm (`syntx.syn`, `syntx.tvf`, `syntx.syngs`, `syntx.robust_affine`) containing exact parameter values, dataset pair metadata, hardware compute device, and benchmark metrics.



## 4. Label Evaluation Constraints
To ensure accurate and standardized registration benchmarking against ground-truth segmentations (e.g., Mindboggle DKT labels):
* **Interpolation:** When applying transforms to discrete/integer label maps, you **must** use nearest neighbor interpolation (e.g., `interpolator='nearestNeighbor'` in `ants.apply_transforms`). Never use linear or b-spline interpolation on segmentations.
* **Overlap Metrics:** Use `ants.label_overlap_measures` to systematically compute structural DICE scores (TargetOverlap) when assessing registration quality.
* **Bidirectional Fixed & Moving Space Evaluation**:
  - In all Mindboggle benchmarks, Cortical DKT31 Dice MUST be evaluated **symmetrically in both image spaces**:
    - **Fixed Space**: Warp moving labels to fixed space (`fwdtransforms`, `interpolator='nearestNeighbor'`) and compare with fixed labels.
    - **Moving Space**: Warp fixed labels to moving space (`invtransforms`, `interpolator='nearestNeighbor'`) and compare with moving labels.
  - Report both directional Dice scores ($\text{Dice}_{\text{fixed}}$, $\text{Dice}_{\text{moving}}$) and their symmetric mean $\text{Dice}_{\text{sym}} = 0.5 \cdot (\text{Dice}_{\text{fixed}} + \text{Dice}_{\text{moving}})$.

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
* **Coordinate Domain Matching for `grid_sample`:** When composing spatial grids (e.g., evaluating $G_1(X)$ where $G_1$ maps Fixed $\rightarrow$ Moving), the lookup coordinates $X$ **must** be normalized relative to the **domain** of the tensor you are sampling. Passing Moving-space normalized coordinates to sample a Fixed-space grid results in completely invalid coordinate mapping.
  - **Physical to Normalized Target Mapping**: When normalizing physical coordinates to sample a target image tensor (e.g., mapping moving physical coordinates into `[-1, 1]` for `grid_sample(moving_image, ...)`), you **must** use the physical metadata (shape, spacing, origin, direction) of the **target image you are sampling** (the moving image), NOT the metadata of the grid you are coming from. Using fixed image metadata to normalize physical coordinates meant for sampling the moving image will cause massive misalignments on heterogeneous datasets.
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
  - **Displacement Fields & Velocity Fields**: Must use `padding_mode='border'` during ODE trajectory integration (`tvf.py`, `syngs.py`), fixed-point inversion, and algebraic composition ($\phi_{inv} = \phi_{r2l} \circ \phi_{l2r}^{-1}$). Omitting `padding_mode` causes PyTorch to default to `padding_mode='zeros'`, zero-clamping boundary velocity vectors, creating hard boundary discontinuities, and inflating max inverse identity errors.

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
* **Cubic B-Spline Temporal Velocity Interpolation**:
  - In Time-Varying Velocity Field (TVF) registration, velocity fields $\mathbf{v}(t_k)$ are parameterized at discrete keyframe timepoints $t_k \in [0, 1]$ (e.g., $T=3$ or $T=4$).
  - Dense ODE time integration along $t \in [0, 1]$ uses cubic B-spline temporal interpolation across stored keyframes.
  - Similarity losses are evaluated strictly at stored keyframe volumes (e.g., $t=0.0, 0.5, 1.0$), while ODE trajectories integrate densely through continuous time without requiring extra stored intermediate velocity volumes.
* **Fast Gradient Smoothing (`fast_smooth=True`)**:
  - Downsamples velocity field gradients by $2\times$ prior to 3D separable Gaussian filtering, accelerating the primary smoothing bottleneck by $9.4\times$ ($547\text{ ms} \rightarrow 58\text{ ms}$) without degrading convergence accuracy.
* **Optimal Pyramid Schedule & CFL Momentum Defaults:**
  - High-accuracy TVF registration MUST default to `reg_iterations = [100, 100, 20]` with `cfl_momentum = 0.95` and soft constant speed relaxation (`constant_speed_relaxation = 0.05 - 0.10`). This configuration yields peak Cortical Label 3 Dice ($\ge 0.8917$) in under 8 seconds.
* **Elastic Total Field Smoothing Sweet Spot (`total_sigma = 0.05`)**:
  - Setting `total_sigma = 0.05` (with `flow_sigma = 0.5`) in `tvf_registration()` provides the optimal elastic total field smoothing parameter.
  - It strictly eliminates negative Jacobians (**0.0000% grid folding**, $\min \det(J) > 0.0$) and reduces inverse identity mapping error by **$10\times$** (sub-0.03 mm) while preserving peak Cortical Dice alignment ($\ge 0.8860$).
* **Multi-Dimensional Image Shape Guard for Identity Checks**:
  - In `TVFModel.fit()`, identity short-circuit checks MUST verify array shape equality BEFORE calling `torch.allclose`:
    `if fixed_image.shape == moving_image.shape and torch.allclose(fixed_image, moving_image, atol=1e-5):`
  - Direct `torch.allclose` evaluation on images with differing spatial shapes (e.g., 3D brain volumes from different subjects) raises a PyTorch broadcast `RuntimeError`.


* **Sobolev Gradient Preconditioning vs. Parameter Dampening**:
  - In EPDiff geodesic shooting (`syngs.py`, `syngs_jax.py`), Sobolev Green's operator smoothing ($\widehat{K}(\mathbf{k}) = \frac{1}{(1 + \alpha k_{\text{sq}})^s}$) MUST be applied **strictly to parameter gradients** ($\nabla m_0 \leftarrow K \nabla_{m_0} L$). Never apply post-step parameter dampening ($m_0 \leftarrow K m_0$) on initial momentum $m_0$ at every epoch, as $m_0$ generates the entire geodesic path and post-step dampening chokes deformation energy.
* **Sobolev Green's Operator Frequency Calibration in 3D TVF**:
  - In 3D physical coordinate space ($256 \times 256 \times 256$), spatial frequency norm scales across 3 dimensions ($k^2 = k_x^2 + k_y^2 + k_z^2$).
  - Never apply 2D parameter defaults (`sobolev_alpha = 0.5`, `grad_step = 0.30`) to 3D TVF, as high-frequency accumulator forces cause coordinate boundary overshooting.
  - In 3D TVF, calibrate Sobolev Green's operator frequency decay to `sobolev_alpha = 2.5` with `grad_step = 0.18` and low fluid smoothing `fluid_sigma = 0.8`. This achieves SOTA Cortical DKT31 Dice (`0.5853`) while maintaining strict `0.0000%` grid folding.
* **Dense Trajectory Multi-point Loss Alignment**:
  - For complex 3D cortical registrations, set `multipoint_loss = [0.0, 0.5, 1.0]`. Evaluating loss at the Fréchet midpoint ($t=0.5$) alongside endpoints ($t=0.0, 1.0$) provides continuous gradient feedback along the ODE trajectory, driving fine sulcal alignment.
* **Velocity Field vs. Accumulated Field Kinematic Regularization**:
  - **Velocity Regularization (`flow_sigma` / Sobolev)**: Applied to velocity gradients ($\nabla_{\mathbf{v}} \mathcal{L}$) *before* parameter updates. Preserves the Lie group manifold $\text{Diff}(\Omega)$, geodesic momentum conservation, and bi-diffeomorphic invertibility.
  - **Accumulated Field Regularization (`total_sigma` / `elastic_sigma`)**: Extrinsically distorts geodesic momentum. Must default to `total_sigma = 0.0` in pure SyN/TVF registration pipelines.

## 13. Mindboggle Benchmark Pair Conventions & Hard Pairs
* **Hard Pair 00 (`hard_pair_00`)**: Defined as the inter-cohort 3D Mindboggle registration pair:
  - **Fixed Subject**: `NKI-TRT-20-2` (Cohort: `NKI-TRT-20`, Origin: `[0, 0, 0]`, Spacing: `[1.0, 1.0, 1.0]`)
  - **Moving Subject**: `MMRR-21-2` (Cohort: `MMRR-21`, Origin: `[202.8, 0, 0]`, Spacing: `[1.2, 1.0, 1.0]`)
  - **CSV Index**: Pair 45 (Line 46 in `examples/pairs.csv`).
  - **Benchmark Significance**: Canonical inter-cohort stress-test pair evaluating physical coordinate mapping across scanner origins, anisotropic voxel spacing, and SyN backend parity (ANTs C++, PyTorch MPS, JAX CPU).

## 14. TVF Temporal Anti-Symmetry & Constant Speed Parameterization Invariants
* **Vector Channel Standardization**: Vector component channels (e.g. displacement fields of shape `(D, H, W, 3)`) in `syntx` are standardized natively across `syntx.spatial`, `syntx.syn`, `syntx.tvf`, and `syntx.transform`. Never apply ad-hoc component channel permutations (such as `[2, 1, 0]`) when exporting displacement tensors to ANTs NIfTI images (`ants.from_numpy(..., has_components=True)`).
* **TVF Constant Speed Parameterization (`constant_speed=True`) vs. Midpoint Zeroing**:
  - **Never force zero midpoint velocity ($\mathbf{v}(0.5) = \mathbf{0}$)** during standard 3D brain registration. Forcing anti-symmetry (`antisymmetric=True`) creates a severe flow stagnation bottleneck at $t=0.5$ that degrades Cortical Dice by >14% (`0.3324` vs `0.4743`).
  - **Always prefer Constant Speed Parameterization (`constant_speed=True`)**: Enforces uniform keyframe kinetic energy ($\|\mathbf{v}_k\|_V = E_{\text{mean}}$) across $t \in [0, 1]$, maintaining steady flow momentum throughout integration trajectory without zero-velocity dead zones.
* **CFL Momentum Floor ($\beta \ge 0.90$)**:
  - All TVF LARS/CFL velocity optimizers MUST maintain a momentum buffer $\beta \ge 0.90$ (`cfl_momentum=0.90` to `0.95`) to prevent gradient stalling on smooth LNCC similarity loss plateaus (+5.27% Cortical Dice gain over $\beta=0.0$).
* **Analytical Gradients ODE Trajectory Backpropagation Invariant**:
  - Analytical gradient calculations (`use_analytical_gradients=True`) MUST route spatial similarity derivative forces ($\nabla_{\mathbf{x}} I \cdot \frac{\partial \text{LNCC}}{\partial I}$) continuously back through the ODE trajectory forward pass (`self.forward()`).
  - Never assign static un-integrated spatial gradients from a single $t=0.5$ midpoint to keyframe velocity parameters, as bypassing ODE flow trajectory integration causes severe coordinate overshooting and 49.68% grid folding.
* **User Commit Authorization Guardrail**:
  - The AI assistant MUST NEVER execute `git commit` or `git push` without explicit, prior user authorization in the current prompt conversation turn.
* **TVF Optimal Triplet Multi-point Loss Default (`multipoint_loss = [0.0, 0.5, 1.0]`)**: The optimal multi-point loss configuration for TVF registration is `multipoint_loss = [0.0, 0.5, 1.0]` (triplet loss evaluating similarity simultaneously at endpoints $t=0.0, 1.0$ and Fréchet midpoint $t=0.5$). Triplet loss provides continuous gradient feedback along the entire ODE trajectory while anchoring direct endpoint boundaries, maximizing Cortical Dice overlap.
* **Asymmetric Topologies (Forward-Only Shooting)**: For highly asymmetric shape transformations (e.g. Half-C to Full-C expansion), use **forward-only (non-symmetric) EPDiff shooting**. Forced geodesic midpoint symmetry constrains single-direction topological expansion.
* **High-Resolution Grid Nyquist Bounds**: Higher spatial grid resolutions ($128 \times 128$, $256 \times 256$) expand Fourier frequency Nyquist bounds for FFT spectral derivatives ($\widehat{\nabla v} = 2\pi i \mathbf{k} \hat{v}$), suppressing spatial boundary aliasing and guaranteeing **strict 0.0000% grid folding** ($\min \det(J) > 0.0$).

## 15. Apple Silicon MPS Scheduling Constraints
* **Unified Memory Bandwidth**: On Apple Silicon, CPU and MPS GPU share the same memory subsystem. Running CPU-intensive workloads (e.g., ANTs C++ registration with 4+ threads, JAX CPU computations) concurrently with MPS GPU workloads causes severe memory bandwidth contention, degrading MPS performance by 10–15×.
* **Benchmark Scheduling**: When benchmarking MPS-accelerated methods alongside CPU methods, always run MPS tasks **first** (with CPU idle), then run CPU tasks **after** MPS tasks complete. Never launch CPU-heavy threads in parallel with active MPS computation.

## 16. Benchmark & Pipeline Design: Affine Initialization & `syntx.robust_affine`
* **Mandatory Affine Pre-Alignment**: Every registration pipeline (`syntx.tvf`, `syntx.syn`, `syntx.syngs`, `ants.registration`) **MUST** perform affine pre-alignment prior to non-linear deformable optimization. Never run deformable registration from unaligned native coordinates without affine alignment.
* **Standardized Multi-Start Initial Alignment (`syntx.robust_affine`)**:
  - Use `syntx.robust_affine` (supporting `mode='pytorch'`, `'auto'`, `'ants_fast'`, or `'com_only'`) to compute a ultra-fast, fail-safe initial affine transformation.
  - `robust_affine` utilizes PyTorch Lie Algebra rotation parameterization with continuous Taylor expansion and intensity-weighted Center-of-Mass matching, guaranteeing robust convergence even under massive spatial translation or rotation offsets.
* **Fair Benchmark Comparison**: When isolating non-linear deformable registration quality across algorithms, initialize all methods with the same `robust_affine` transform computed once per image pair.
* **Internal Affine Refinement Permitted**: Pre-seeding an initial affine transform does **NOT** replace or forbid internal affine optimization; each method is encouraged to refine the affine parameters with its internal optimizer (e.g., `affine_epochs > 0` in `syntx.tvf` or `affine_iterations` in `syntx.syn`).

## 17. GPU Memory Management & Garbage Collection Guardrails
* **In-Loop GPU Cache Clearing**: In sequential batch processing loops (e.g., Mindboggle benchmark pairs), PyTorch's internal `CachingAllocator` retains allocated memory buffers across iterations, leading to memory fragmentation over large 3D volume runs. Call `torch.mps.empty_cache()` (Apple Silicon MPS) or `torch.cuda.empty_cache()` (NVIDIA CUDA) accompanied by `gc.collect()` at the end of every registration pair loop.
* **Process Isolation for Batch Benchmarks**: For long-running multi-pair benchmark suites, execute each registration pair in an isolated subprocess (`multiprocessing` with `spawn` context). OS-level process termination guarantees 100% memory pool teardown and eliminates autograd or Metal/CUDA state leakage.

## 19. TVF Velocity Resizing, Fluid Increment Regularization, & Pyramidal Flow Decay
* **Pyramid Velocity Resizing Tensor Ordering Invariant**:
  - Velocity parameter tensors in PyTorch have shape `(B, T, *spatial, dim)`.
  - When resizing velocity grids across multi-resolution pyramid levels in `_resize_velocity()`, spatial interpolation MUST reshape `(B * T, *spatial, dim)` prior to `F.interpolate()` and reshape back to `(B, T, *new_shape, dim)` post-interpolation. Squeezing dimension 1 transposes batch `B` and keyframe `T` dimensions, scrambling temporal keyframe ordering across levels.
* **Fluid Increment Update vs. Total State Smoothing**:
  - Green's regularization operator $\mathcal{R}_{\text{fluid}}$ MUST filter the velocity update increment $\delta \mathbf{v}_k = \eta \cdot \nabla L$ per iteration step.
  - Never apply heavy Green's operators repeatedly to the total accumulated velocity parameter $\mathbf{v}_{\text{old}}$ every iteration step, as $O(N^2)$ repeated low-pass filtering multiplies frequency components by $(\hat{K}_{\text{DSTI}})^N \approx 0$, collapsing registration accuracy and driving un-damped DC grid folding.
  - Mild post-step elastic Gaussian smoothing (`total_sigma`) may be applied to $\mathbf{v}(x, t)$ to dampen discrete derivative noise without destroying high-frequency sulcal detail.
* **Pyramidal Flow Sigma Decay**:
  - For multi-resolution TVF registrations, `flow_sigma` should decay across pyramid levels (e.g., `flow_sigma = [2.0, 1.0, 0.4, 0.08]`), applying strong fluid regularization at coarse scales for global shape alignment and fine fluid smoothing at full resolution ($1\times$) for sulcal boundaries.

## 20. Framing of Outliers and Registration Failures
* **MPS Transient Instability vs. Algorithmic Divergence**: Never frame catastrophic registration failures (e.g., massive Dice drops $\le 0.10$, total geometric distortion) as theoretical flaws or algorithmic "collapse" of the underlying methods (e.g., SyN, TVF, DSTI). 
* **Attribution**: These failures must be correctly attributed as **transient numerical/compute instabilities** specifically tied to Apple Silicon MPS float32 tensor cores struggling with aggressive gradient step sizes or LARS momentum parameters.
* **Communication Integrity**: Preserve the integrity of the mathematical methods. Explicitly state that these outliers do not invalidate the underlying theory (such as DSTI's spectral superiority) but merely reflect hardware-specific stochasticity requiring threshold trimming or parameter tuning (e.g., reducing `grad_step`).
