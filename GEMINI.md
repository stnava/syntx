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
* **LNCC Autograd vs Center-of-Window Approximation**: Analytical autograd backpropagation through standard LNCC ($CC = \frac{s_{FM}}{\sqrt{s_{FF} \cdot s_{MM}}}$) yields superior spatial descent directions and $+1.08\%$ higher Symmetric Dice accuracy compared to ITK C++ center-of-window approximation ($CC^2 = \frac{s_{FM}^2}{s_{FF} \cdot s_{MM}}$).
* **Autograd Physical Scaling & Vector Channel Alignment**: When backpropagating through coordinate grid samplers, the gradient $\frac{\partial \mathcal{L}}{\partial \phi}$ must be converted from normalized grid units $[-1, 1]$ to physical displacement units ($1/\text{mm}$) by multiplying by $\frac{(N - 1) \cdot s}{2}$. Because spatial dimensions in PyTorch are ordered $(Z, Y, X)$ while displacement vector channels are ordered $(x, y, z)$, the scaling vector MUST be flipped along dimension 0 via `torch.flip(scale, dims=[0])`:
  $$\mathbf{s}_{\text{phys}} = \text{flip}\left(\frac{(\mathbf{N} - 1) \odot \mathbf{s}}{2}, \text{dim}=0\right)$$
  Failing to flip causes directional cross-axis gradient scaling errors on anisotropic acquisitions (e.g. scaling $x$-displacements by $z$-dimensions).
* **Antithetic Bootstrapped Gradient Estimation Invariant (`syntx.syn`)**:
  To prevent localized gradient aliasing and high-frequency grid folding caused by discrete coordinate discretization at sharp anatomical boundaries, `syntx.syn` supports unbiased **Antithetic Bootstrapping** (`bootstrap_mode='antithetic'`, `bootstrap_orig_weight=0.50`, `bootstrap_jitter_scale=0.25`). The effective descent direction is formulated as an unbiased coordinate-centered triplet:
  $$\bar{\mathbf{g}} = w_0 \mathbf{g}(\mathbf{X}) + \frac{1 - w_0}{2} \left[ \mathbf{g}(\mathbf{X} + \boldsymbol{\delta}) + \mathbf{g}(\mathbf{X} - \boldsymbol{\delta}) \right] \quad \text{where } \boldsymbol{\delta} \sim \mathcal{U}(-0.25, 0.25) \odot \mathbf{s}_{\text{phys}}$$
  Because $\mathbb{E}[\boldsymbol{\delta}] = \mathbf{0}$, this guarantees zero spatial directional bias while destructively cancelling discrete interpolation noise, reducing grid folds by $6\times$ to $25\times$ and achieving $0.00000\%$ folding with lower harmonic deformation energy.
* **Autograd + Gaussian Kernel Peak Standard (`use_analytical_gradients=False`, `kernel_type='gaussian'`)**: Full autograd backpropagation through sliding box-filter LNCC coupled with the ITK truncated sampled Gaussian kernel represents the verified peak standard for 3D SyN registration, achieving a 6/6 win sweep over ANTs C++ SyN (Mean Symmetric Dice $0.6476$ vs $0.6236$, $0.0005\%$ folding, $0.027\text{ mm}$ inverse error, and $4.35\times$ GPU speedup).
* **Foreground 2nd–98th Percentile Intensity Normalization Policy**:
  To prevent gradient stalling and Mutual Information compression caused by high-intensity acquisition outliers (e.g. vascular or reconstruction spikes up to 3000+), all input images to registration optimization (both affine initialization and deformable SyN) MUST be truncated and scaled using foreground non-zero 2nd-to-98th percentiles:
  $$I_{\text{norm}} = \text{clamp}\left(\frac{I - p_{02}(I_{>0})}{p_{98}(I_{>0}) - p_{02}(I_{>0}) + 10^{-6}}, 0.0, 1.0\right)$$
  When $p_{98} \le p_{02} + 10^{-4}$ (e.g. binary masks or flat regions), the normalizer must gracefully fall back to positive range $[0.0, \max(I_{>0})]$ to prevent zero-array collapse.
* **Mattes Mutual Information Foreground Masking Invariant**:
  When evaluating Mutual Information (MI) on 2D/3D images, background zero padding voxels dominate joint histogram distributions. All MI optimization loss calculations and multi-start candidate selections MUST apply foreground union masking:
  $$\text{mask} = (I > 0.01) \mid (J > 0.01)$$
* **Deterministic Affine Multi-Start Invariant (`syntx.robust_affine`)**:
  To prevent affine local basin entrapment and stochastic noise across serial benchmark evaluations, `syntx.robust_affine` MUST use deterministic regular uniform grid sampling (`sampling_strategy='regular'`) and foreground union-masked Mutual Information candidate scoring (`mask=(I > 0.01) | (J > 0.01)`). All affine population evaluations MUST render standardized interactive HTML reports via `syntx.viz.create_affine_benchmark_report()`.
* **Top-Level Wrapper `fit_kwargs` Forwarding Invariant**:
  All top-level registration wrappers (`syntx.syn()`, `syntx.tvf()`, `syntx.robust_affine()`) MUST explicitly forward all non-signature keyword arguments (`**fit_kwargs`) into underlying `model.fit()` and optimization routines. Never drop `**kwargs` at wrapper interfaces.

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
* **SyN Peak Eulerian Provenance Parameter Invariants (`syntx.syn`):**
  - `formulation = 'eulerian'` (Strictly superior to Lagrangian; yields ~0.688 Dice vs ANTs 0.671 on Mindboggle fine resolution).
  - `inverse_method = 'anderson'` (Mandatory for stability in PyTorch Eulerian composition; `fixed_point` will diverge).
  - `use_analytical_gradients = True` (The ITK `CC²` pseudo-derivative is optimal with Eulerian).
  - `grad_step = 0.25` (Produces peak accuracy with functionally negligible folding ~0.01%).
  - `flow_sigma = 3.0` (ITK variance convention: $\sigma^2 = 3.0$, equivalent to `syntx` `flow_sigma = sqrt(3.0) ≈ 1.732` std dev)
  - `total_sigma = 0.0` (pure fluid deformation without total elastic field smoothing)
  - `in_loop_inv_steps = 10` (compute inverse update at every iteration inside the optimization loop)
  - `initial_transform` from `syntx.robust_affine(mode='pytorch')`
* **Lagrangian SyN Provenance Parameter Invariants (Legacy / Fallback):**
  - **Note:** Lagrangian is now considered a fallback. PyTorch Eulerian composition with the `shrink_ratio` scale fix significantly outperforms Lagrangian in both accuracy and folding stability.
  - **Constraint:** Lagrangian transformation updates must use subtraction ($\phi_{\text{new}} = \phi_{\text{old}} - u \circ (\text{Id} + \phi_{\text{old}})$) to enforce correct velocity field pullback direction (gradient descent). Using addition produces gradient ASCENT and causes optimization divergence.
  - **Folding Behavior & Sensitivity:** Lagrangian velocity integration is extremely sensitive to folding when using the ANTs C++ default variance (`flow_sigma=1.732`). Unlike Eulerian (which remains fold-free up to `grad_step=0.05`), Lagrangian begins folding immediately at `grad_step=0.04`. To prevent Lagrangian folding while maximizing accuracy, you MUST heavily smooth the gradients (e.g., `flow_sigma=3.0`, equivalent to ITK variance 9.0) and use low step sizes (`grad_step=0.10`).
  - **Optimal Provenance Parameters:** `grad_step = 0.10`, `flow_sigma = 3.0` (yields $0.7667$ Symmetric Dice with $0.029\%$ folding, closely matching ANTs C++ SyN $0.764$ Dice $0.000\%$ folding).
  - **Analytical vs Autograd:** Analytical gradients (`use_analytical_gradients=True`) produce ~1–2% higher Dice but ~2–10× more folding than autograd due to sharper spatial gradient approximation. Use autograd for topology-preserving registration.
  - **Deformed-Space Smoothing:** For Eulerian formulation, `smooth_in_deformed_space=True` reduces folding by ~50% at the cost of ~1% Dice and ~15% compute time. Not applicable to Lagrangian.
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
* **TVF Peak Provenance Parameter Invariants (`syntx.tvf`, `syntx.core.optimizers`)**:
  - `optimizer = 'reg_adam'` (with `optimizer_lr = 1.2`, `max_step_norm = 0.50`)
  - `max_step_norm = 0.50` (Optimal Courant-Friedrichs-Lewy displacement step limit in voxels; yields $+1.12\%$ Cortical DICE boost over $0.35$ without grid folding)
  - **Peak Accuracy Configuration (`regularizer='gaussian'`, `fast_smooth=False`)**:
    * `flow_sigma = 3.0` (ITK variance 3.0, std dev $\approx 1.732\text{ mm}$ fluid smoothing)
    * `total_sigma = 0.0` (pure fluid deformation without post-step elastic over-stiffening)
    * `gaussian_sigma = 1.5` (RegAdam quotient step filter)
    * Achieves peak Mindboggle accuracy (Mean Symmetric DICE $0.6345$ on `mbhard`) by eliminating periodic Fourier boundary reflections.
  - **Peak Speed & Strict Topology Configuration (`regularizer='sobolev'`, `fast_smooth=True`)**:
    * `flow_sigma = 1.0` (fluid velocity smoothing for sharp sulcal guidance)
    * `total_sigma = 0.035` (calibrated Sobolev elastic velocity smoothing; guarantees 0.00% folding)
    * `sobolev_alpha = 0.035` (dimension-aware physical frequency damping in $\text{mm}^{-1}$)
    * `fast_smooth = True` (utilizing `_SOBOLEV_FILTER_CACHE` and composite radix-2 dimensions for $1.77\times$ speedup, $163\text{ s}$, $0.0007\%$ folds)
  - **Exact Homogeneous Dirichlet Zero-Boundary Configuration (`regularizer='dsti1'`)**:
    * `regularizer = 'dsti1'` (Separable Discrete Sine Transform Type-I Green operator)
    * `dsti_alpha = 0.035`, `flow_sigma = 1.0`, `total_sigma = 0.035`
    * Analytically enforces $v(x \in \partial \Omega) \equiv 0$, guaranteeing strictly positive Jacobian determinants ($\min \det(J) = +0.0039 > 0$) and $0.0000\%$ folding across the entire volume.
  - `multipoint_loss = [0.0, 0.5, 1.0]` (evaluate LNCC similarity at trajectory start t=0.0, midpoint t=0.5, and endpoint t=1.0)
  - `antisymmetric = False` (explicit 3-point loss control without automatic timepoint injection)
  - `solver = 'euler'` (35% faster than RK4 with identical accuracy)
  - `cfl_momentum = 0.9`
  - `n_time_steps = 3`
  - `use_analytical_gradients = False`
  - `constant_speed = True` (`constant_speed_relaxation = 0.10`)
  - `reg_iterations = [100, 50, 10]` (Peak Full Schedule: 100 coarse, 50 medium, 10 native iters for >0.62–0.64 DICE)
  - `reg_iterations = [100, 40, 0]` (Ultra-Fast 35s Schedule for real-time fold-free registration)
  - `initial_transform` from `syntx.robust_affine(mode='auto')`
  - **Elastic Over-Stiffening Invariant**: Never apply post-step Gaussian elastic smoothing (`total_sigma > 0` with Gaussian); it acts as an overly stiff global spring, causing $-4.6\%$ DICE collapse. All elastic regularization MUST use the physical Sobolev Green operator.
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
* **Strict Mandate Against Ad-Hoc Benchmarking Scripts:** Never write scratch scripts (e.g., `test_bench.py`) or ad-hoc `ants.label_overlap_measures` loops to evaluate registration performance. Ad-hoc scripts are prone to incorrect initializations (e.g., using `AffineFast` instead of `syntx.robust_affine`) and metric misinterpretations (e.g., indexing `MeanOverlap[0]` for single-class metrics instead of multi-class averages), which lead to massive false regression debugging.
  - **Required Action:** ALWAYS use the validated, high-level evaluation functions (e.g., `compute_bidirectional_dice` from `syntx.benchmark.worker`) and the established benchmark runner scripts (`run_r16_r64.py`, `run_mbhard.py`, `run_full_benchmarks.py`) to guarantee apples-to-apples baseline comparisons.
* **Subprocess Isolation Requirement for Multi-Pair Benchmarks**: When running serial registration benchmark suites on Apple Silicon GPU (`mps`), each pair MUST be executed in a dedicated, isolated Python subprocess (`subprocess.run([sys.executable, "-u", "scripts/run_single_pair_eval.py", str(pair_idx)])`). Running multiple large 3D registrations sequentially inside a single Python process causes GPU memory retention, graph caching leaks, and MPS allocator corruption.

## 5. Image Comparison Metric Guidelines (`syntx.image_compare`)
To maintain a unified API and consistent cross-dimensional support:
* **Standardized Returns (Lower is Better):** All metrics evaluated through `image_compare` must return scores where a lower value strictly indicates higher similarity. For metrics traditionally maximized (e.g., PSNR, NCC), return the inverted or negative value (e.g., `-PSNR` or `1 - NCC`).
* **2D and 3D Dimensionality:** All metrics must support both 2D and 3D inputs. When integrating 2D-native deep feature models (like VGG19), it is standard and permitted to implement a "3D extension" (such as a triplanar ensemble) to support 3D images, rather than restricting to native 3D architectures.

## 6. Registration Optimization & Initialization Constraints
* **Achieving Exact 0.000% Folding (Topological Parity with ANTs C++):**
  - **Context:** Following the multi-resolution `shrink_ratio` fix, `syntx` Eulerian composition correctly maintains a constant maximal physical step size. It no longer requires extreme step clipping to prevent catastrophic tearing.
  - **Rule:** To strictly enforce exactly `0.0000%` grid folding parity with ANTs C++, use `formulation='eulerian'` with `grad_step=0.10`. This maintains perfect topology while effectively matching ANTs C++ peak accuracy (~0.670 Dice). 
  - **Peak Accuracy Alternative:** For maximum possible accuracy (~0.688 Dice, crushing ANTs), use `grad_step=0.25`, which produces functionally negligible trace folding (~0.01%).
* **Physical Space Awareness:** Optimization pipelines using PyTorch/JAX normalized `[-1, 1]` grids must explicitly map physical space differences (origin, spacing, direction) to the grid space. Do not assume normalized grids naturally align images from different physical scanner spaces.
* **CoM Initialization Selection:** For affine alignments, dynamically select the best initialization by testing both Field of View (FOV) and Foreground (intensity-weighted) Center of Mass physical translations via a fast Mutual Information evaluation (e.g., downsampled `mattes_mi_loss_nd`).
* **Preserving Gradients in Lie Algebra:** When parameterizing spatial rotations via Lie Algebra, avoid non-differentiable conditionals at zero angles (e.g., `torch.where(omega == 0, I, R)`) that lock gradients to zero. Always implement a first-order Taylor expansion (`I + K_raw`) for infinitesimally small angles to ensure continuous gradient flow at identity initialization.
* **ANTs Affine Center of Rotation:** When parsing an ANTs affine transform to a standard $4 \times 4$ homogeneous matrix $y = Ax + t_{new}$, you **must** account for the center of rotation $C$ (stored in `tx.fixed_parameters`). The translation vector must be explicitly updated as $t_{new} = t + C - A \cdot C$. Ignoring $C$ results in massive physical coordinate misalignments.
* **ITK CFL Gradient Step (Voxel Space):** In ITK, `gradientStep` (used in SyN/Demons CFL optimization) is scaled in **voxel space**, not absolute physical space. When normalizing the gradient field ($\Delta = \text{step} \cdot \frac{\nabla}{||\nabla||_{max}}$), you **must** multiply the step size by the grid's current physical spacing (`step * spacing`). This ensures that a step of $0.1$ voxels translates to a proportionately larger physical step (e.g. $0.4$ mm) at coarser pyramid levels (e.g. downsampled by $4\times$). Without this spacing multiplier, optimization will severely stall at coarse levels.
* **ITK↔syntx Gaussian Sigma Convention (SETTLED — Do Not Re-Investigate):**
  - ITK C++ `GaussianOperator::SetVariance(v)` takes **variance** $\sigma^2 = v$, so `flow_sigma=3.0` in ANTs C++ means $\sigma = \sqrt{3.0} \approx 1.732$ voxels.
  - `syntx` `separable_gaussian_filter(field, sigma)` takes **standard deviation** $\sigma$ directly, so `flow_sigma=3.0` in `syntx` means $\sigma = 3.0$ voxels.
  - **Conversion**: To match ITK C++ smoothing with parameter value $v$, pass `sigma = sqrt(v)` to `syntx`. For example, ITK `flow_sigma=3.0` → syntx `flow_sigma=sqrt(3.0)=1.732`.
  - **This is a settled, permanent fact. Never re-derive, re-investigate, or re-verify this relationship.**
* **Coordinate Domain Matching for `grid_sample`:** When composing spatial grids (e.g., evaluating $G_1(X)$ where $G_1$ maps Fixed $\rightarrow$ Moving), the lookup coordinates $X$ **must** be normalized relative to the **domain** of the tensor you are sampling. Passing Moving-space normalized coordinates to sample a Fixed-space grid results in completely invalid coordinate mapping.
  - **Physical to Normalized Target Mapping**: When normalizing physical coordinates to sample a target image tensor (e.g., mapping moving physical coordinates into `[-1, 1]` for `grid_sample(moving_image, ...)`), you **must** use the physical metadata (shape, spacing, origin, direction) of the **target image you are sampling** (the moving image), NOT the metadata of the grid you are coming from. Using fixed image metadata to normalize physical coordinates meant for sampling the moving image will cause massive misalignments on heterogeneous datasets.
* **Affine Parameter Post-Step Clamping**: For affine optimizations in PyTorch and JAX, parameters must be clamped post-step (`scale` and `anisotropic_scale` $\in [0.05, 20.0]$, `shear` $\in [-5.0, 5.0]$, `omega` $\in [-\pi, \pi]$). Never place `clip`/`clamp` functions inside the forward loss autograd function, as zeroing gradients outside bounds causes Adam momentum wind-up.
* **Spatial Coordinate Centralization Invariants (`syntx.spatial`):**
  - **Single Source of Truth (`syntx.spatial`):** All coordinate transformations, displacement field domain conversions, metadata reversals, and ITK $\leftrightarrow$ PyTorch/JAX conversions MUST reside in and be handled exclusively by `syntx.spatial`.
  - **No Ad-Hoc Local Transposes or Axis Reversals:** Never insert ad-hoc matrix reversals (e.g. `[::-1, ::-1]`) or uncoordinated array transposes inside optimization algorithms (`robust_affine.py`, `syn.py`, `tvf.py`). All spatial conversions must use centralized helpers:
    - `syntx.spatial.reverse_metadata(spacing, origin, direction)` for converting ITK $(x,y,z)$ physical metadata to tensor $(z,y,x)$ order.
    - `syntx.spatial.image_to_tensor(img, device)` for converting ANTsImages into PyTorch tensors.
    - `syntx.spatial.disp_tensor_to_itk` and `syntx.spatial.disp_itk_to_tensor` for displacement field domain conversions.
  - **PyTorch 3D `grid_sample` Channel Mapping Invariant:**
    - In PyTorch 3D `grid_sample(input, grid)` where `input` has shape `(N, C, D, H, W)`: `grid[..., 0]` maps to `W` ($X$), `grid[..., 1]` maps to `H` ($Y$), and `grid[..., 2]` maps to `D` ($Z$).
    - When sampling physical coordinates, grid channels MUST be arranged in physical $[x_{norm}, y_{norm}, z_{norm}]$ order matching the physical metadata provided by `syntx.spatial`.
* **LARS Optimizer for Time-Varying Velocity Fields (TVF)**:
  - **Scale-Invariant Momentum vs. Adam Stalling**: Standard Adam updates parameters via unscaled moment ratios ($m_t / \sqrt{v_t}$). In smooth, low-gradient similarity loss plateaus, $v_t$ shrinks, causing Adam step sizes to stall before resolving high-frequency sulcal boundaries.
  - **Layer-wise Trust Ratio Scaling**: LARS rescales velocity updates per keyframe tensor $v(t_k)$ using the trust ratio $\text{trust\_ratio} = \eta \cdot \frac{\|v(t_k)\|}{\|g(t_k)\| + \epsilon}$, allowing high global learning rates ($lr \in [0.50, 1.20]$) while maintaining scale-invariant optimization momentum and preserving diffeomorphic invertibility ($\det(J) > 0$).
* **TVF Sigma Single-Conversion Invariant (`syntx.tvf`):**
  - `tvf_registration()` converts user-facing ITK variance convention to standard deviation via `sqrt()`. The converted values are passed directly to `TVFModel.fit()`.
  - `fit()` MUST NOT apply any further `sqrt()` or other conversion to `fluid_sigma` or `elastic_sigma`. Applying `sqrt()` twice creates a quartic root ($\sigma^{0.25}$) that makes parameters physically uninterpretable.
  - This bug is TVF-specific; `syntx.syn` correctly converts once in `registration()` and passes directly to `SyNTo.fit()`.
* **TVF Antisymmetric Gradient Averaging (`syntx.tvf`):**
  - The `antisymmetric=True` flag in TVF ensures `multipoint_loss` includes both `t=0.0` (fixed-side gradient) and `t=1.0` (moving-side gradient).
  - Autograd naturally computes $\partial L / \partial v = (\partial L / \partial I_{\text{warped}})(\partial I_{\text{warped}} / \partial v) + (\partial L / \partial J_{\text{warped}})(\partial J_{\text{warped}} / \partial v)$. Dividing by `len(eval_points)` averages the fixed-side and moving-side contributions. This is the exact TVF generalization of SyN's `delta_l`/`delta_r` averaging.
  - Do NOT manipulate gradients or velocity parameters directly for antisymmetry. Gradient-space projection ($g - g_{\text{flip}}$) kills optimizer signal at initialization ($g(t_k) \approx g(t_{T-1-k})$ when $v=0$). Velocity-space projection zeros out the center keyframe with odd $T$.
* **TVF Forward Pass Compute Invariants (`syntx.tvf`):**
  - **Identity Short-Circuit:** `integrate(t, t)` MUST return zero displacement immediately without entering the ODE solver loop.
  - **Velocity Upsample Caching:** Pre-upsampled velocity keyframes (`_cached_velocity_fine_cf`) MUST be computed once per `forward()` call and shared across all `integrate()` calls within that pass. Never re-upsample per `integrate()` call.
  - **Boundary Mask Caching:** The cosine taper boundary mask MUST be cached per pyramid level and reused across epochs. Never recreate per epoch.
* **TVF Folding Behavior & Regularization Rule:**
  - TVF folding originates from accumulated sharp spatial features in the velocity field over many optimization epochs, NOT from ODE integration error.
  - **Strict TVF Regularization Rule:** TVF MUST use `flow_sigma = 0.0` (zero fluid gradient smoothing). Using `flow_sigma > 0` degrades Cortical Dice by 2.5–3.5% across all elasticity levels and doubles compute time due to per-epoch 3D convolution overhead.
  - **Sole Regularization Lever (`total_sigma`):** All velocity field regularization MUST be handled exclusively via `total_sigma` (elastic post-step smoothing of velocity field parameters). `total_sigma` monotonically controls the Dice vs. folding Pareto frontier (e.g. `0.0` → 0.815 Dice / 1.87% fold; `0.2` → 0.774 Dice / 0.01% fold; `0.5` → 0.743 Dice / 0.00% fold).
  - The Euler solver produces marginally more folding than RK4 (0.15% vs 0.13%) but is 35% faster with identical Dice.

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

## 10. Gaussian Smoothing Space and Unit Conventions (ANTs/ITK Parity)
* **Physical Standard Deviation Standard ($\sigma$ in mm)**: In all `syntx` registration interfaces (`syntx.syn`, `syntx.tvf`, `syntx.syngs`), `flow_sigma` and `total_sigma` represent standard deviations in physical space ($\sigma$ in mm), matching ANTsPy `ants.registration(flow_sigma=...)`.
* **No `sqrt` Variance Conversion**: Gaussian and Green operator smoothing filters (`separable_gaussian_filter`, Sobolev, DST-I1) MUST consume $\sigma$ directly without applying `math.sqrt` (`fluid_sigma_actual = float(flow_sigma)`). Applying `math.sqrt` halves the effective smoothing bandwidth (e.g. $\sigma = 1.732\text{ mm}$ instead of $3.0\text{ mm}$), causing gradient instability, uncalibrated kinetic spikes, and local grid folding.
* **Voxel Index Space Smoothing**: In ITK, `GaussianOperator` performs convolution in **voxel units**, not physical units. Do not pass spacing vectors to Gaussian filters in PyTorch/JAX to scale $\sigma$. Keep the smoothing isotropic in voxel space at all multi-resolution/downsampled levels to ensure mathematical parity between backends.

## 11. Geodesic Shooting & Momentum Invariants (`syntx.syngs`)
* **EPDiff Geodesic Evolution Invariant**:
  In Geodesic Shooting, the entire continuous diffeomorphism $\boldsymbol{\phi}_{0 \to 1}$ is parameterized uniquely by the initial velocity / momentum vector field $\mathbf{v}_0 \in V = H^s(\Omega)$ at $t=0$. Evolution follows the Euler-Poincaré differential equation (EPDiff):
  $$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla) \mathbf{v} + \text{ad}_{\mathbf{v}}^* \mathbf{v} = 0 \quad \text{where } \mathbf{v}(t) = K(\mathbf{m}(t))$$
* **Pareto Regularization Frontier (Sobolev & DST-I)**:
  1. **Strict Topology Shield (100% Zero Folding Guaranteed)**:
     - Configuration: `regularizer='sobolev'`, `sobolev_alpha=0.35`, `max_step_norm=0.22`, `similarity_metric='cc2'`, `reg_iterations=[100, 100, 30]`
     - Metrics: Achieves strictly **`0.00000%` grid folding** with $\min \det(J) \ge +0.0417 > 0$ and $0.6320$ Symmetric DICE.
  2. **Balanced Diffeomorphic Regime**:
     - Configuration: `regularizer='sobolev'`, `sobolev_alpha=0.28`, `max_step_norm=0.25`, `similarity_metric='cc2'`
     - Metrics: Achieves **`0.6378` Symmetric DICE** with negligible **`0.0020%` folding** (2 voxels per 100,000).
  3. **Peak Accuracy Regime (Outperforming ANTs C++ and `syntx.syn`)**:
     - Configuration: `regularizer='sobolev'`, `sobolev_alpha=0.22`, `max_step_norm=0.35`, `similarity_metric='cc2'`, `bootstrap_mode='antithetic'`, `bootstrap_jitter_scale=0.25`
     - Metrics: Achieves peak **`0.6478` Symmetric DICE** (`0.6750` Fixed DICE, `0.6206` Moving DICE), outperforming ANTs C++ SyN by **$+0.8522\%$** and `syntx.syn` by **$+0.5456\%$**.
* **Velocity Transport vs. Compounding In-Loop Re-Filtering Invariant (`transport_mode='transport'`)**:
  In Geodesic Shooting, applying smoothing filters (Gaussian or Sobolev) recursively inside the ODE integration loop creates an exponential compounding filter $K^N(\mathbf{k}) = e^{-\frac{N}{2} \sigma^2 \|\mathbf{k}\|^2}$ that freezes deformation magnitude and collapses DICE. Geodesic Shooting MUST use **Velocity Transport** (`transport_mode='transport'`), where the initial velocity $\mathbf{v}_0 = K(\mathbf{m}_0)$ is smoothed ONCE at $t=0$ and transported directly along the trajectory:
  $$\mathbf{v}(t_k, \mathbf{x}) = \mathbf{v}_0(\boldsymbol{\phi}_k(\mathbf{x})) = \text{grid\_sample}(\mathbf{v}_0, \boldsymbol{\phi}_{\text{norm}})$$
  This preserves 100% of the kinetic spectral bandwidth, executes $2\times$ faster per epoch, and establishes a clean, monotonic Pareto scaling across both 2D and 3D.
* **Optimization Method Invariant (`optimizer='reg_adam'`)**:
  Momentum parameters in Geodesic Shooting require scale-bounded adaptive step preconditioning. Across 2D and 3D benchmarks:
  - `optimizer='reg_adam'`: Peak **`0.6479` - `0.6651` Symmetric DICE** ($+0.86\%$ to $+2.58\%$ over ANTs C++).
  - `optimizer='adam'` / `'adamw'`: `0.6131` Symmetric DICE ($-2.61\%$ drop due to coordinate tearing from unbounded coordinate descent).
  - `optimizer='sgd'`: `0.6028` 2D DICE (stalls near affine initialization).
* **Comprehensive 3-Way Regularization Operating Frontier**:
  1. **Gaussian Velocity Transport (`regularizer='gaussian'`, `transport_mode='transport'`)**:
     - Peak Accuracy ($\sigma = 0.8\text{ mm}$): **`0.6651` Symmetric DICE** ($+2.58\%$ over ANTs, $0.818\%$ folds).
     - Balanced Parity ($\sigma = 1.2\text{ mm}$): **`0.6479` Symmetric DICE** ($+0.86\%$ over ANTs, $0.0366\%$ folds).
     - Strict Topology Shield ($\sigma = 1.4\text{ mm}$): **`0.6334` Symmetric DICE**, strictly **`0.00000%` folds**, $\min \det(J) = \mathbf{+0.0044 > 0}$.
  2. **Fourier Sobolev Velocity Transport (`regularizer='sobolev'`, `transport_mode='transport'`)**:
     - Peak Accuracy ($\alpha = 0.28, \text{cfl} = 0.28$): **`0.6551` Symmetric DICE** ($+1.58\%$ over ANTs, $0.179\%$ folds).
     - Strict Topology Shield ($\alpha = 0.35, \text{cfl} = 0.22$): **`0.6528` Symmetric DICE** ($+1.35\%$ over ANTs, **`0.0474%` folds**).
  3. **Discrete Sine Transform Type-I (`regularizer='dsti1'`) (Exact Dirichlet Zero Boundary)**:
     - Exact Dirichlet ($\alpha = 0.035, \text{cfl} = 0.30$): **`0.6289` Symmetric DICE**, strictly **`0.00000%` folds**, $\min \det(J) = \mathbf{+0.0746 > 0}$ (100% boundary-clamped diffeomorphic regularity).
* **Antithetic Bootstrapped Momentum Estimation Invariant**:
  To prevent localized gradient aliasing and high-frequency momentum singularities caused by discrete coordinate discretization at sharp cortical boundaries, `syntx.syngs` evaluates coordinate-centered antithetic triplets:
  $$\bar{\mathcal{L}} = w_0 \mathcal{L}(\mathbf{X}) + \frac{1 - w_0}{2} \left[ \mathcal{L}(\mathbf{X} + \boldsymbol{\delta}) + \mathcal{L}(\mathbf{X} - \boldsymbol{\delta}) \right] \quad \text{where } \boldsymbol{\delta} \sim \mathcal{U}(-0.25, 0.25) \odot \mathbf{s}_{\text{phys}}$$
  Because $\mathbb{E}[\boldsymbol{\delta}] = \mathbf{0}$, this guarantees zero spatial directional bias while destructively cancelling discrete interpolation noise, reducing grid folds by over $10\times$.
* **Momentum Integration Tooling (`syntx.integrate_momentum` / `syntx.shoot_geodesic`)**:
  Initial momentum vector fields $\mathbf{v}_0$ exported in `reg['fwd_momentum']` and `reg['inv_momentum']` can be seamlessly integrated into physical displacement fields or continuous temporal keyframe trajectories via EPDiff geodesic evolution.

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
* **Native Data Requirement**: All experiments and benchmarking must be conducted on raw Native space data (`t1weighted_brain.nii.gz` and `labels.DKT31.manual.nii.gz`), NEVER the `MNI152` pre-aligned versions. 
* **Affine Initializers for Native Space**: When evaluating pure deformable models without internal affine optimizers (e.g., `TVF NoAff`) on native data, they must be initialized with a shared ANTs affine transform (`initial_transform=affine_tx`) to bridge the massive scanner space misalignments.

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
* **ITK Affine Transform Offset Parameter Parity**:
  In ITK `AffineTransform`, the forward mapping from fixed physical point $x_f$ to moving point $x_m$ around center of rotation $C$ is parameterized as:
  $$\mathbf{x}_m = \mathbf{A}(\mathbf{x}_f - \mathbf{C}) + \mathbf{C} + \mathbf{t}_{\text{offset}}$$
  When exporting transforms to ANTs/ITK format with `tx.set_fixed_parameters(C)` (where $C = \text{com}_f$), `tx.set_parameters` MUST take $\mathbf{t}_{\text{offset}} = \mathbf{t}_{\text{final}}$ (the displacement between centers). Never pass $\mathbf{t}_{\text{final}} + \mathbf{C} - \mathbf{A}\mathbf{C}$ when fixed parameters $C$ are non-zero, as ITK will add the centering term $(\mathbf{I} - \mathbf{A})\mathbf{C}$ twice.
* **Fast End-to-End Reproducibility Standard**:
  To guarantee exact reproducibility across all optimization solvers, maintain lightweight, fast ($< 5\text{s}$) 2D and 3D end-to-end regression tests on synthetic data (`test_reproducibility_fast.py`) to verify exact float identity ($\Delta < 10^{-6}$) and zero stochastic drift across consecutive runs.

## 17. GPU Memory Management & Garbage Collection Guardrails
* **In-Loop GPU Cache Clearing**: In sequential batch processing loops (e.g., Mindboggle benchmark pairs), PyTorch's internal `CachingAllocator` retains allocated memory buffers across iterations, leading to memory fragmentation over large 3D volume runs. Call `torch.mps.empty_cache()` (Apple Silicon MPS) or `torch.cuda.empty_cache()` (NVIDIA CUDA) accompanied by `gc.collect()` at the end of every registration pair loop.
* **Process Isolation for Batch Benchmarks**: For long-running multi-pair benchmark suites, execute each registration pair in an isolated subprocess (`multiprocessing` with `spawn` context). OS-level process termination guarantees 100% memory pool teardown and eliminates autograd or Metal/CUDA state leakage.
* **Strict Sequential Execution on Apple Silicon MPS (No Concurrent MPS Jobs)**:
  - **Constraint**: Resource contention and unified memory bandwidth limits on macOS Metal Performance Shaders (MPS) prevent running parallel or concurrent registration processes on the GPU simultaneously.
  - **Failure Mode**: Concurrent processes accessing MPS cause Metal command buffer execution errors (`kIOGPUCommandBufferCallbackErrorInnocentVictim`), memory allocation deadlocks, and GPU recovery resets.
  - **Mandate**: All MPS registration tasks, multi-pair benchmark sweeps, and evaluations MUST be executed strictly sequentially (one active MPS process at a time).
  - **Concurrency Exception**: Parallel execution across multiple processes is permitted only for CPU-bound tasks (`device='cpu'`) or on multi-GPU CUDA clusters with isolated device IDs.

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

## 9. Epistemic Integrity and Technical Explanations
* **No Unfounded Guesses:** Never invent, hallucinate, or present unverified hypotheses as factual explanations for technical, mathematical, or algorithmic behaviors.
* **Differentiate Fact from Hypothesis:** If the root cause of an observed behavior (e.g., performance discrepancies, gradient folding, numerical instability) is unknown, explicitly state: "I do not know the exact cause." 
* **Verify Before Explaining:** Instead of guessing why a mathematical operation behaves a certain way, formulate a hypothesis and immediately write an experiment, test, or code inspection to prove or disprove it. Only present the conclusion after empirical verification.

* **Eulerian vs Lagrangian Formulation Parity**: Both Eulerian and Lagrangian formulations achieve strict mathematical diffeomorphism (`Fold% = 0.000000%`) when implemented correctly. The Eulerian formulation (`formulation='eulerian'`) exactly matches ANTs C++ SyN performance (`0.7660` Dice) using standard parameters (`flow_sigma=2.0` or `3.0`, `grad_step=0.25` or `0.45`).
* **Strict Ban on Manual Boundary Discontinuities**: NEVER manually enforce exact zero Dirichlet boundary conditions (e.g., `warp.mul_(b_mask)`) directly on the displacement field *inside* the optimization loop unless followed immediately by a heavy elastic smoothing pass. Multiplying a continuous vector field by a binary mask creates massive spatial discontinuities that artificially explode PyTorch gradients and cause severe grid folding (`det(J) < 0`). Let the PyTorch `grid_sample` (which defaults to `border` padding) naturally clamp out-of-bounds coordinates instead.
* **Inverse Tolerance Scaling (`inv_tolerance`)**: When updating inverse fields iteratively via fixed-point/Anderson acceleration, the exit tolerance must be scaled in physical voxel units (e.g., `inv_tolerance = 0.1 * min(spacing)`). Never use unscaled massive physical limits (like `2.8` mm), which cause the solver to exit after 1 iteration, destroying inverse consistency and bidirectional Dice scores.
* **Background Label Ban in Pandas DataFrames**: When evaluating Dice scores using `ants.label_overlap_measures`, NEVER index the first row (`iloc[0]`) to extract the Mean Overlap. The first row (Label `0.0`) is always the background class, which artificially inflates the Dice score (e.g., `0.787` background vs `0.650` cortex).
* **Required Action:** ALWAYS use the validated, high-level evaluation functions (e.g., `compute_bidirectional_dice` from `syntx.benchmark.worker`) which explicitly filters out the background label and returns the true mean cortical overlap.
