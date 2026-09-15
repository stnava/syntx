# syntx — detailed findings, benchmarks and provenance (historical record)

This is the full, unabridged record that used to live in `GEMINI.md`. It contains every measured result,
parameter value and dated finding. The agent-facing rule set is now the short `GEMINI.md` at the repository root,
which points here by section number when a rule needs its evidence.

---

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
* **Adaptive Non-Local Means Denoising Policy (`antstorch.denoise_image`)**:
  To suppress high-frequency thermal and Rician acquisition noise without blurring sulcal anatomical boundaries, standard preprocessing for all 3D volumetric registrations MUST apply adaptive non-local means denoising after N4 bias field correction and immediately prior to foreground 2nd–98th percentile normalization:
  ```python
  if use_denoise:
      try:
          import antstorch
          img = antstorch.denoise_image(img, shrink_factor=2, p=1, r=1, noise_model="Rician")
      except Exception as e:
          logger.warning(f"Denoising fallback: {e}")
  ```
  Empirically across 210 benchmark evaluations (10-pair cohort), non-local means denoising consistently improves Mean Symmetric Dice by **+0.40% to +0.90%** across every registration architecture (Sobolev SyN rising from $0.6278 \to 0.6351$, achieving the #1 highest accuracy and 100% win rate over ANTs C++) while reducing grid folding by up to $50\%$.
* **Foreground 2nd–98th Percentile Intensity Normalization Policy**:
  To prevent gradient stalling and Mutual Information compression caused by high-intensity acquisition outliers (e.g. vascular or reconstruction spikes up to 3000+), all input images to registration optimization (both affine initialization and deformable SyN) MUST be truncated and scaled using foreground non-zero 2nd-to-98th percentiles:
  $$I_{\text{norm}} = \text{clamp}\left(\frac{I - p_{02}(I_{>0})}{p_{98}(I_{>0}) - p_{02}(I_{>0}) + 10^{-6}}, 0.0, 1.0\right)$$
  When $p_{98} \le p_{02} + 10^{-4}$ (e.g. binary masks or flat regions), the normalizer must gracefully fall back to positive range $[0.0, \max(I_{>0})]$ to prevent zero-array collapse.
* **Universal Default Similarity Metric (`similarity_metric='cc2'`)**:
  All SyN registration modules (`syntx.syn`, `syntx.syngs`, `syntx.tvf`, `syntx.greedy`, and CLI) MUST default to squared cross-correlation (`similarity_metric='cc2'`). In head-to-head 10-pair evaluation, `cc2` achieves $0.6220$ Dice ($80\%$ win rate vs ANTs C++), outperforming linear unsquared `lncc` ($0.6203$, $60\%$ win rate) and autograd `box_lncc` ($0.6210$). The $CC^2$ analytical pseudo-gradient scale is modulated by $CC \cdot \nabla I$, naturally damping deformation forces in areas of poor correlation and preventing boundary gradient spikes.
* **Mattes Mutual Information Standard & Boundary-Padding Invariant (`similarity_metric='mattes_mi'`)**:
  - **Boundary-Padded Partition of Unity**: All Parzen windowing implementations using cubic B-splines MUST enforce a boundary margin of $\text{pad} = 2.0$ bins, mapping dynamic range $x \in [-1, 1]$ into index space $u \in [2.0, (K-1) - 2.0]$. This guarantees that the sum of Parzen weights identically equals 1.000000 across the entire dynamic range ($\sum_k w_k(x) \equiv 1$) and prevents phantom boundary forces ($\frac{d}{dx} \sum w_k(x) = 0$).
  - **Precision & Memory Isolation**: Parzen density accumulation MUST execute in explicit `float32` (isolating from half-precision AMP overflow and `NaN` log-ratios) and extract non-zero foreground voxels prior to dynamic range scaling.
  - **Application Scope**: Mattes MI achieves $0.6108$ Dice ($40\%$ win rate vs ANTs CC2, $-0.48\%$ difference) on intra-modality T1-to-T1 registration where local windowed metrics excel. Mattes MI is designated as the primary default metric for multi-modal (e.g. T1-to-T2, T1-to-FLAIR, MRI-to-CT) and contrast-inverted registrations where non-monotonic joint intensity mappings require entropy maximization.
* **Mattes Mutual Information Masking (revised 2026-09-14, `scripts/affine_repro_harness.py`, `results/affine_baseline/sweep_mask_mode.json`)**:
  - For the **affine** objective, sample the **whole fixed domain including background** (`mask_mode='none'`, as ANTs/ITK do without masks). On mbhard this reaches Dice 0.3227–0.3265 (ANTs affine 0.3255). Fixed-foreground-only masking plateaus at 0.31 because a moving brain spilling into fixed background is never penalised — longer optimisation then *lowers* Dice while lowering the loss. The ANTs affine itself scores best under the whole-domain objective (−0.2715) and poorly under the foreground-only one (−0.2276).
  - The former union-mask rule `(I > 0.01) | (J > 0.01)` is a transform-dependent sample set; it neither matched ANTs nor reached its Dice (≤ 0.318) and triggers a pathological variable-shape path on MPS (900 s). Do not use it for optimisation; foreground-only masking remains appropriate for *reporting* similarity of already-aligned images.
* **Universal Default Spatial Regularizer (`regularizer='sobolev'`, $\alpha=1.5$)**:
  All SyN Eulerian registration interfaces MUST default to Sobolev space smoothing (`regularizer='sobolev'`, `sobolev_alpha=1.5`) rather than isotropic Gaussian filter smoothing:
  - **Accuracy & Pareto Superiority**: Sobolev SyN achieves $0.6278$ Dice ($100\%$ win rate over ANTs C++) vs $0.6220$ for Gaussian ($\sigma=3.0$, $80\%$ win rate) and $0.6197$ for Gaussian ($\sigma=2.0$). With preprocessing denoising, Sobolev SyN reaches $0.6351$ Dice (highest in 210-run sweep).
  - **Topological Invariant**: Sobolev FFT spectral decay ($\frac{1}{1 + \alpha \|\mathbf{k}\|^2}$) completely eliminates grid folding on intra-study pairs ($0.0000\%$ folds) and yields an overall folding rate of only $0.0078\%$ ($2.5\times$ lower than Gaussian's $0.0199\%$), while running faster ($54.8\text{ s}$ vs $55.9\text{ s}$).
* **Affine Reproducibility Harness & ANTs Baseline (2026-09-14)**:
  - `scripts/affine_repro_harness.py` records, per solver/device, repeated-run parameter spread (0.0 = bitwise), Dice, image similarity and runtime, against the ANTs C++ affine baseline in `results/affine_baseline/mbhard_ants_affine_baseline.json` (best: `Affine` default, symmetric DKT31 Dice 0.3255, ~43 s; ANTs itself is not bitwise repeatable run-to-run, max |Δparam| up to 0.33).
  - Acceptance for any PyTorch affine change: same-device Δparams == 0, Dice ≥ best ANTs − 0.005, runtime < ANTs (`tests/test_affine_reproducibility.py`).
  - Baseline state before the affine commit series: `mode='pytorch'` Dice 0.304 on CPU (bitwise reproducible, 48 s) but 0.215–0.224 and non-reproducible on MPS.
* **Pyramid Geometry Invariant (any avg-pooled multi-resolution registration)**:
  - A pooled voxel `i` at downsampling factor `L` averages full-resolution voxels `[L·i, L·i+L−1]`; its centre is `L·i + (L−1)/2`, and a pooled moving tensor must be addressed in *pooled* index space normalised by the pooled shape. Mapping `i → L·i` and normalising with the full-resolution shape (the old `robust_affine` pyramid) biases every coarse level by `(L−1)/2` voxels plus `~(L−1)/n` scale — 1.5 vox at level 4 — which the short fine stage never fully undoes. Fixing this alone moved the 6-pair affine result from −0.0039 to parity with ANTs and made a dense level-3 stage viable.
* **PyTorch Affine Solver Defaults & Presets (`robust_affine(mode='pytorch')`, 2026-09-14)**:
  - One schedule-driven optimisation path (`_AffinePath`: `A = R(ω)·B·diag(eˢ)·Shear`), seeded coarse-level candidates (identity at CoM, ±4/8/12° single-axis rotations, optional `initial_transform` re-centred as a candidate; a landmark affine from `syntx.landmarks` is a valid seed), best-of-`n_starts` (3) selection by exact MI.
  - Objective: Mattes MI, **32** bins, fixed bounds (0, 1), **whole fixed domain** (`mask_mode='none'`). Dense stages evaluate the exact voxel grid (fixed-image Parzen weights cached); point-sampled stages use a fixed seeded 100k-voxel sample. Never gradient-weighted sampling or transform ensembles (both tested, both worse).
  - `preset='default'`: L4 rigid exact → L3 affine exact (100 it, select) → L2/L1 point-sampled — ties/beats ANTs C++ Affine on 6/6 sweep pairs (+0.0002 mean Dice) in ~11 s on MPS. `preset='accurate'`: L4 → L2 affine on a regular 50 % grid (100 it) → L1 — +0.0021 mean, 6/6 wins, ~20 s. `preset='fast'`: all point-sampled, ~5 s. Cohort validation: `results/affine_baseline/cohort10_v2_affine.json`, `cohort10_accurate_affine.json`, `docs/AFFINE_GUIDE.md`.
  - Bitwise reproducible per device (same process and fresh interpreter); CPU and MPS may settle in neighbouring optima (Dice within ~0.005). ANTs C++ Affine is *not* run-to-run reproducible.
* **MPS Large-K Matmul Ban for Parzen Histograms**:
  - On Apple MPS (torch 2.13) `w_xᵀ @ w_y` with N ≳ 1e5 rows is non-deterministic across allocations and, for concentrated histograms, wrong by up to ~15 %. This corrupted every Mattes MI evaluation on the GPU and hence the affine solve.
  - Joint histograms MUST be accumulated with `syntx.core.losses._parzen_joint_histogram` (blocked `bmm`, K = 4096, fixed-order sum), which is bitwise deterministic and ~30× closer to a float64 reference. Subsampled vectors MUST be made contiguous before feeding MPS kernels.
  - MI used for optimisation or candidate scoring MUST use fixed histogram bounds (`fixed_range=(0.0, 1.0)` on foreground-normalised images) and the fixed-image foreground mask, so values are comparable across transforms and candidates.
* **Deterministic Affine Multi-Start Invariant (`syntx.robust_affine`)**:
  To eliminate stochastic sampling risk and moving baselines across serial benchmark evaluations, `syntx.robust_affine` MUST enforce 100% bit-for-bit determinism:
  - **Thread & RNG Isolation**: Enforces `os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"` and invokes `ants.config.set_ants_deterministic(True, seed)` (defaulting to `seed=42`) to eliminate multi-threaded floating-point summation race conditions and guarantee `--random-seed` is explicitly passed to ITK C++.
  - **Deterministic Multi-Start Search**: Candidate evaluation strictly scores explicit `.mat` initial transforms using foreground union-masked Mutual Information (`mask=(I > 0.01) | (J > 0.01)`). Eliminates unseeded, stochastic internal registration calls (such as `AffineFast`) during candidate scoring.
  - **Regular Sampling Invariant**: Affine registration optimization strictly defaults to deterministic regular uniform grid sampling (`aff_random_sampling_rate=0.25`, `sampling_strategy='regular'`).
  - All affine population evaluations MUST render standardized interactive HTML reports via `syntx.viz.create_affine_benchmark_report()`.
* **Top-Level Wrapper `fit_kwargs` Forwarding Invariant**:
  All top-level registration wrappers (`syntx.syn()`, `syntx.tvf()`, `syntx.robust_affine()`) MUST explicitly forward all non-signature keyword arguments (`**fit_kwargs`) into underlying `model.fit()` and optimization routines. Never drop `**kwargs` at wrapper interfaces.
* **Standard Preprocessing & Affine Benchmark Invariant (`syntx.robust_affine`)**:
  - All 3D registration benchmarks and evaluations MUST apply adaptive non-local means denoising via `antstorch.denoise_image` followed by foreground 2nd–98th percentile intensity normalization via `normalize_intensity()` prior to affine initialization and deformable optimization.
  - When evaluating 3D brain registration against standard benchmarks (e.g. Mindboggle `mbhard` or 90-pair cohort), affine initialization MUST use `syntx.robust_affine(fi, mi, mode='auto')`. Never use experimental `mode='pytorch'` or unmasked initializers for official benchmark reporting.
* **B-Spline SyN Regularization & Knot Spacing Invariants (`regularizer='bspline'` / `BSplineSyN`)**:
  - **Cortical Knot Spacing Scale**: Macro-knot spacing (`spline_distance >= 20.0 mm`) restricts deformation to low-frequency global warps (~10 control points across a whole brain volume) and is inadequate for resolving 2–4 mm cortical ribbon anatomy. For high-resolution cortical registration, `spline_distance` MUST be parameterized in the fine range (typically $3.0\text{ mm} \le \text{dist} \le 6.0\text{ mm}$), or structured across multi-resolution pyramid levels.
  - **Non-Mutating Keyword Arguments Invariant**: Operator dispatchers (including `_apply_bspline_operator`, `_apply_sobolev_green_operator`, `_apply_dsti1_green_operator`) MUST NEVER mutate `**kwargs` in-place using `kwargs.pop()`. They must use `kwargs.get()` and filter dictionary subsets to ensure parameters persist uniformly across all iterations and forward/reverse gradient updates.
* **Weingarten Curvature & Sulcal Ridge Guidance Invariants (`antstorch.weingarten_image_curvature`)**:
  - **GPU Acceleration Requirement**: When extracting extrinsic differential geometric level-set curvatures for 2D or 3D scalar images, registration workflows should utilize `antstorch.weingarten_image_curvature` on GPU/MPS backends, achieving over $20\times$ speedup over ITK C++ ($0.41\text{ s}$ vs $9.12\text{ s}$ on full MNI brain, $r = 0.999734$).
  - **Mean Curvature Invariant**: Cortical registration workflows targeting sulcal ridge guidance MUST strictly use **Mean Curvature** ($H = \frac{1}{2}(\kappa_1 + \kappa_2) = -\frac{1}{2}\nabla \cdot \left(\frac{\nabla I_\sigma}{\|\nabla I_\sigma\|}\right)$) rather than Gaussian Curvature ($K = \kappa_1 \kappa_2 = \det(\mathcal{W})$). Because cortical sulcal folds are developable/cylindrical surfaces ($\kappa_1 \gg 0, \kappa_2 \approx 0$), Gaussian curvature vanishes along sulcal troughs ($K \approx 0$) and drops DICE by $\ge 0.7\%$, whereas Mean Curvature maintains continuous tangential tracking along the entire length of the sulcal trench.
  - **Aperture Problem Resolution & Deformation Regularization**: Intensity gradients in the uniform ~2.5 mm gray matter ribbon have near-zero tangential component ($\nabla_\parallel I \approx 0$). Blending Mean Curvature into registration targets ($I_{\text{guided}} = I + \alpha \cdot H$, $\alpha \approx 0.10, \sigma = 1.5\text{ mm}$) provides dense, monotonic tangential landmarks (sulcal fundi $< 0$, gyral crests $> 0$). This anchors opposing sulcal banks, suppresses cross-bank spatial shear strain ($\frac{\partial v_\parallel}{\partial x_\perp}$), and elevates $\min \det(J)$, eliminating coordinate grid folding ($0.00000\%$ folds on intra-study pairs).
  - **Intra-Study vs Inter-Study Guidance Policy**:
    * **Intra-Study / Intra-Cohort**: Use fine-scale curvature ($\alpha = 0.10, \sigma = 1.5\text{ mm}$) for maximal cortical Dice gain (+0.16% to +0.58%) with zero folding.
    * **Inter-Study / Inter-Cohort**: Due to divergent tertiary sulcal branching across individuals and cross-site acquisition differences, fine-scale curvature can over-constrain non-homologous folds. Inter-study registration should use coarser scales ($\sigma \ge 3.0\text{ mm}$) or apply multi-channel separate weighting ($w_I \cdot \text{CC2}_I + w_H \cdot \text{CC2}_H$) rather than scalar blending.
* **Sulcal Soft Dice Guided Registration Policy (`syntx.surface`, `similarity_metric=['cc2', 'dice']`)**:
  - **Representation Superiority**: Continuous distance potentials ($\exp(-D/\tau)$) bleed across the 2.5 mm gray matter ribbon and degrade cortical boundaries (-0.44% to -1.47% DICE). In contrast, multi-metric SyN optimizing T1 intensity CC2 alongside a localized sulcal fundus probability map with Soft Dice enforces topological fundus ridge overlap without cross-bank distortion.
  - **Sharp Probability Bandwidth**: Fundus probability maps MUST be extracted via `compute_surface_classes(img, sigma=1.5, grouping='gyral_sulcal')` and smoothed with sharp bandwidth ($\sigma = 1.0\text{ mm}$) via `generate_surface_channels(cls, mode='soft_prob', smoothing_sigma=1.0)`. Sharper bandwidths ($\sigma=1.0\text{ mm}$) strictly outperform broader bandwidths ($\sigma=2.0\text{ mm}$) by preventing gradient bleed across opposing sulcal banks.
  - **Inter-Study / Cross-Cohort Guidance Weighting**: On inter-study pairs exhibiting cross-scanner contrast discrepancies and acquisition differences, sulcal Soft Dice guidance MUST be weighted dominantly at $w = [0.30, 0.70]$ (`similarity_metric=['cc2', 'dice']`, `metric_weights=[0.30, 0.70]`). This achieves **+0.63% to +0.96% Cortical DICE gains** (Pair 53: $0.6182 \to 0.6278$, Pair 42: $0.6380 \to 0.6443$) by breaking out of local intensity minima.
  - **Intra-Study Guidance Weighting**: On intra-study pairs where acquisition parameters and scanner contrast are identical, intensity CC2 must remain primary. Use $w = [0.80, 0.20]$ (`metric_weights=[0.80, 0.20]`) to preserve peak intensity overlap while reducing coordinate grid folds by **$4\times$** ($0.0264\% \to 0.0065\%$).
* **Mindboggle Benchmark Cohort Nomenclature Invariant (`intra` vs `inter`)**:
  In Mindboggle-101 benchmark reports and evaluations, all pairs represent cross-subject registrations between distinct individuals. The category designations strictly specify dataset acquisition provenance:
  - **`intra` = Intra-Study / Intra-Cohort**: Fixed and moving images originate from the same acquisition cohort and scanning site (e.g., both from `OASIS-TRT-20`, `NKI-RS-22`, or `MMRR-21`). They share acquisition protocols, RF coils, contrast weighting, and voxel spacing.
  - **`inter` = Inter-Study / Inter-Cohort**: Fixed and moving images originate from different acquisition cohorts and scanning sites (e.g., `OASIS-TRT-20` registered to `MMRR-21` or `NKI-TRT-20`). They exhibit cross-site protocol divergence, varying contrast profiles, and anisotropic resolution discrepancies.
  - **Forbidden Nomenclature**: Never refer to `intra` as "intra-subject" or "longitudinal". It is strictly **intra-study / intra-cohort**.

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
* **Overlap Metrics:** Use `ants.label_overlap_measures` to systematically compute structural DICE scores. You MUST strictly extract and report the **Sørensen-Dice** coefficient (`MeanOverlap` in ITK/ANTsPy: $2 |A \cap B| / (|A| + |B|)$). Under no circumstance should Target Overlap (`TotalOrTargetOverlap` in ANTsPy: $|A \cap B| / |A|$) or Union Overlap be reported as Dice.
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

## 21. Autonomous Diagnostic Engine & Decathlon Multi-Task Registration Guardrails (`syntx.diagnose`, `syntx.policy`, `syntx.classifier`)
* **Two-Tier Diagnostic Architecture (`syntx.diagnose`, `syntx.classifier`)**:
  All autonomous registrations in `syntx.auto_reg()` MUST route through the intelligent diagnostic engine:
  - **Tier 1 (Fast Physical & Statistical Heuristics, $<15\text{ ms}$)**: Instant Hounsfield thresholding (CT $[-1000, 2000]\text{ HU}$), internal lung cavity vs abdominal soft-tissue ratios, physical transverse field of view, slice thickness anisotropy, and percentile dynamic range tail ratios.
  - **Tier 2 (Deep 3D ResNet-10 Multi-Task Classification)**: When `fast=False` or ML inference is requested, 3D ResNet-10 (`resnet10_3d()`) executes multi-task classification across Anatomy (`BRAIN`, `THORAX`, `ABDOMEN`, `PELVIS`, `HEART`) and Modality (`CT`, `MRI_T1`, `MRI_T2`, `MRI_FLAIR`, `MRI_ADC`) in $<25\text{ ms}$ on GPU/MPS.
* **Sub-Structural ROI Crop Sulcal Guidance Invariant**:
  Sulcal Soft Dice geometric guidance (`guided='sulcal'`) is mathematically designed for whole-brain cortical ribbon alignment. It MUST NEVER be applied to cropped sub-structural ROIs (such as Hippocampus, Basal Ganglia, Amygdala, where physical dimensions $< 90\text{ mm}$), as internal structures lack cortical gyri/sulci. `syntx.policy` strictly enforces `guided=None` and translation-only pre-alignment for sub-structural ROI crops.
* **Thick-Slice Anisotropic Pelvic Slab Invariant (`Task05_Prostate`)**:
  For thick-slice 2D axial acquisitions (e.g. Prostate MRI with $\ge 2.5\times$ anisotropy, $4\text{ mm}$ slice thickness vs $0.6\text{ mm}$ in-plane, $Z$-slices $\le 35$):
  - Standard 12-parameter affine gradient descent induces severe out-of-plane shear and spatial degradation.
  - Registration policy MUST enforce pure Center-of-Mass physical translation initialization (`robust_affine='translation_only'`) and strictly zero out gradient affine iterations (`affine_iterations=[0, 0]`). This preserves spatial slab integrity and yields $+39.72\%$ DICE gains.
* **Abdominal Soft-Tissue CT Windowing Invariant (`Task09_Spleen`)**:
  For CT scans of solid abdominal parenchymal organs (Spleen, Liver, Pancreas), input intensities MUST be clamped and normalized to standardized abdominal soft-tissue Hounsfield windows (`ct_window=[-150.0, 250.0]`) prior to Eulerian Sobolev SyN optimization.
* **Cardiac MRI Translation Initialization Invariant (`Task02_Heart`)**:
  Cardiac chest MRI volumes ($400 \times 400\text{ mm}$ FOV) exhibit localized non-rigid myocardial contraction within a broad static chest cavity. Initial alignment MUST enforce center-of-mass translation initialization (`robust_affine='translation_only'`) coupled with Eulerian Sobolev SyN and adaptive Rician denoising, achieving over $+33\%$ to $+45\%$ DICE gains on the Left Atrium.
* **Bidirectional Label DICE Inversion Flag Invariant (`compute_bidirectional_dice`)**:
  When evaluating symmetric label DICE, the moving-space evaluation pulls the fixed ground truth label `fl` back into moving space. The inverse transform list MUST strictly pass `whichtoinvert_inv=[True]` (or `[True, False]` for composite warp+affine). NEVER pass `whichtoinvert_inv=[False]`, as applying un-inverted forward matrices to fixed labels corrupts moving-space DICE down to $\approx 0.05$.

## 22. ANTsPy API & Transform Invariants
* **Strict Ban on `ants.reorient_image` & `ants.reorient_image2`**:
  - In ANTsPy, `ants.reorient_image` is a Python module (`ants.ops.reorient_image`), NOT a callable function. Calling `ants.reorient_image(image, ...)` raises `TypeError: 'module' object is not callable`.
  - NEVER call `ants.reorient_image` or `ants.reorient_image2`.
  - Reorienting arrays for display is forbidden. Keep image arrays in their native space and compute physical display coordinates, slice extractions, and anatomical axis labels (L/R, P/A, I/S) strictly via `syntx.landmarks.spatial`.
* **Transform List & `whichtoinvert` Length Equality Invariant**:
  - In all calls to `ants.apply_transforms`, `whichtoinvert` MUST have the exact same number of boolean elements as `transformlist` has entries (`len(whichtoinvert) == len(transformlist)`).
  - Passing `whichtoinvert=[False]` when `transformlist=[warp, affine]` is an error.
  - Use `syntx.landmarks.spatial.safe_whichtoinvert(transformlist, whichtoinvert)` to guard and pad/truncate boolean flags automatically.

## 23. Physical Space & Display Framework Invariants (`syntx.landmarks.spatial`)
* **Single Module Invariant for Spatial Math**:
  - ALL coordinate conversions (physical mm $\leftrightarrow$ voxel indices), affine matrix extractions, orthographic slice extractions, and landmark projection overlays MUST reside in `syntx.landmarks.spatial`.
  - NEVER duplicate ad-hoc voxel-to-physical arithmetic like `x_mm = w * sx` in scripts, detectors, or plotting routines, as it ignores origin and direction cosines.
* **ANTs Physical Coordinate Convention**:
  $$\mathbf{x}_{\text{phys}} = \mathbf{o} + \mathbf{D} (\mathbf{i}_{\text{XYZ}} \odot \mathbf{s})$$
  $$\mathbf{i}_{\text{XYZ}} = \mathbf{D}^{-1} (\mathbf{x}_{\text{phys}} - \mathbf{o}) \oslash \mathbf{s}$$
  where $\mathbf{o}$ is origin, $\mathbf{s}$ is voxel spacing, and $\mathbf{D}$ is the direction cosine matrix.
* **Canonical API**:
  - `get_image_affine(image)` $\rightarrow$ `(origin, spacing, direction)`
  - `vox_to_physical(image, indices_xyz)` $\rightarrow$ `[N, 3]` physical mm
  - `physical_to_vox(image, points_mm)` $\rightarrow$ `[N, 3]` continuous voxel XYZ indices
  - `physical_offset_to_voxel(image, offsets_mm)` / `voxel_gradient_to_physical(image, g_axes)` for frame-independent neighbourhoods and gradients
  - `image_to_tensor(image)` $\rightarrow$ `[1, 1, nx, ny, nz]` (XYZ layout preserved); `sample_tensor_at_physical(vol, image, points_mm)` is the only permitted `grid_sample` wrapper
  - `extract_ortho_slices(image, center_mm=None, convention='radiological')` $\rightarrow$ display slices plus `views` specs; `project_to_slice(image, points_mm, view=spec)` $\rightarrow$ `(u, v, mask)`
  - `vox_zyx_to_physical` exists only for tensors explicitly transposed to `(nz, ny, nx)`; it must never be applied to `argwhere` output of `image_to_tensor` volumes
* **Tensor Layout Invariant (XYZ, not ZYX)**:
  - `ants.ANTsImage.numpy()` is `arr[ix, iy, iz]`; `torch.from_numpy(arr)[None, None]` keeps that order, so torch dims (2, 3, 4) are `(ix, iy, iz)`.
  - Treating such tensors as `(D, H, W) = (iz, iy, ix)` and reversing `argwhere` indices swaps $x$ and $z$ (the `mbhard` "points float in the padding" regression). All detectors call `vox_to_physical` directly on `argwhere` output.
  - `F.grid_sample` 5-D grids are ordered `(W, H, D)`; for XYZ tensors the normalised grid is `(iz_n, iy_n, ix_n)` with each axis normalised by its own size.
* **Physical-Scale & Frame-Independence Invariant**:
  - Detector scales (`sigma_min`, `sigma_max`) and MIND `offset_distance` are in mm; per-axis voxel sigmas / integer shifts are derived from spacing and direction. Laplacians are scaled by $1/s_a^2$ per axis.
  - SIFT3D gradients are rotated into LPS axes ($\mathbf{g}_{\text{phys}} = \mathbf{D}(\mathbf{g}_{\text{idx}} \oslash \mathbf{s})$) and descriptor windows are sampled on a mm lattice, so the same anatomy stored as LAS vs RPS arrays yields identical descriptors (verified exactly on siq phantoms).
  - Candidate keypoints must lie in the foreground (`normalized > 0.05`) and NMS must rank by response magnitude, never by array order.
* **MPS `F.pad` Ban for 5-D Volumes**:
  - On Apple MPS (torch 2.13) `torch.nn.functional.pad` applied to a 5-D tensor whose trailing `H*W >= 65536` (any 256×256 slice) silently returns corrupted data along the padded depth dim. It zeroed the x-gradient of full-size brain volumes and corrupted the LoG, while all small phantom tests passed.
  - Border handling in `syntx.landmarks` MUST use slicing + `torch.cat` (`blob._shift_pad`); never `F.pad`. `conv3d`, `max_pool3d` and `grid_sample` were verified correct at these sizes. Regression test: `tests/test_landmarks_simulated.py::test_border_ops_match_cpu_on_mps_large_slices`.
* **Global Rotation Search Invariant (`syntx.landmarks.orient`)**:
  - The axis-aligned SIFT3D descriptor tolerates ~30° of relative rotation (same-subject 30°: 306 rigid inliers). Beyond that use `match_sift3d_with_rotation_search`: PCA of the two within-subject landmark clouds seeds iteration 0 only; then iterate descriptors-in-frame → match → rigid RANSAC → rotation, coarse-to-fine over keypoint strength (25% → 50% → 100%), and pick the converged hypothesis with most inliers. Frame voting from per-keypoint structure-tensor frames is *not* reliable on real brains (cortical frames are correlated, wrong matches vote coherently).
  - `frame_rotation` semantics: to compare against the fixed image's axis-aligned descriptors, build the moving descriptors with the fixed→moving direction rotation (= linear part of the fixed→moving point transform).
* **Landmark Guidance Finding (mbhard, 2026-09-14, `scripts/benchmark_mbhard_landmark_guidance.py`)**:
  - A least-squares affine on SIFT3D RANSAC-inlier landmarks (`match_sift3d_with_rotation_search`) matched the standard `robust_affine` (Dice 0.3259 vs 0.3244, foreground NCC 0.551 vs 0.535) in ~1/4 of the time; re-running the ANTs affine stage from it made it worse (0.3237).
  - Sobolev SyN with K=8 landmark-cluster soft-Dice channels at weight 0.2 (`['cc2'] + ['dice']*8`, weights `[0.8] + [0.025]*8`) improved symmetric Dice by +0.0096 (0.6025 → 0.6121) with folding < 0.02 %; weight 0.5 over-constrains (+0.003); restricting to label-agreeing landmarks reduced the gain (+0.006). Single pair — validate on the cohort before changing defaults.
* **Display Orientation Invariant**:
  - Anatomical axes come from `dominant_axes(direction)` (argmax of $|\mathbf{D}|$ per physical axis), handling any axis permutation and flip; never from `D[i, i]` signs alone.
  - ITK LPS: $+x$ = Left, $+y$ = Posterior, $+z$ = Superior. Axial shows Anterior UP, coronal/sagittal Superior UP, sagittal Anterior on the RIGHT. Left/Right follows `convention` (`'radiological'` default: patient Left on the viewer's RIGHT; `'neurological'`: Left on Left).
  - Overlays must use the `view` spec returned with the slice so flips agree by construction.
* **Strict Ban on SIFT2D for 3D Volumetric Data**:
  - NEVER use `sift2d` (or slice-wise 2D back-projection) on 3D volumetric data. Slicing 3D volumes into 2D planar slices loses out-of-plane gradient continuity, introduces slice-sampling bias, and degrades cross-subject landmark matching.
  - For all 3D volumetric registrations, benchmarks, and evaluations, strictly use **true 3D volumetric detectors**:
    - `detect_sift3d` (true 3D DoG scale-space extrema + 3D spherical gradient histogram descriptors)
    - `detect_blobs_log` (3D Laplacian-of-Gaussian scale-space extrema)
    - `detect_blobs_dog` (3D Difference-of-Gaussians scale-space extrema)
    - Volumetric self-similarity descriptors (`extract_mind_at_points`).

