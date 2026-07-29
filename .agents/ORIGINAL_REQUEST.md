# Original User Request

## 2026-07-25T14:24:28Z

Collaborative enhancement of manuscript_report.md bringing together a Statistician, Visualization Expert, Educator, and Computer Vision Scientist to add inferential statistics, high-resolution data plots, educational callouts, and future research directions.

Working directory: /Users/stnava/code/syntx/docs/manuscript
Integrity mode: development

## Requirements

### R1. Statistical Rigor & Inferential Hypotheses (Statistician)
- Perform and document formal inferential statistical tests across the 90 Mindboggle benchmark pairs comparing Syntx JAX, Syntx PyTorch, and ANTs C++ baseline:
  - Paired two-sample t-tests (t-statistic, p-value, degrees of freedom).
  - Non-parametric Wilcoxon signed-rank tests (W-statistic, p-value).
  - Cohen's d effect size calculations and 95% Confidence Intervals (CI_95%) for Cortical Dice overlap.
  - Per-lobe and per-region statistical significance testing.

### R2. Data Visualization & Quantitative Plots (Visualization Expert)
- Create publication-quality data visualization plots and embed them in the manuscript:
  - fig6_dice_distribution_violin.png: Violin/Box plot comparing Cortical Dice distributions across all 90 pairs for JAX, PyTorch, and ANTs C++.
  - fig7_regional_dkt31_heatmap.png: Regional heatmap of DKT31 cortical Dice overlap across all 31 individual structures.
  - fig8_runtime_versus_accuracy.png: Scatter plot comparing 3D Volume Registration Speed (seconds) vs Median Cortical Dice.

### R3. Educational Conceptual Illustrations & Callout Boxes (Educator)
- Generate educational conceptual illustrations and clear callout boxes explaining key concepts for readers:
  - fig9_diffeomorphic_invertibility_concept.png: Conceptual illustration explaining Diffeomorphic Invertibility (J(x) > 0) vs Non-diffeomorphic Grid Folding (J(x) <= 0).
  - Educational callout boxes detailing the LNCC Variance Floor (Var_safe = max(Var(I), 1e-6)), Lie Algebra so(3) Exponential Map, and Single Interpolation Policy.

### R4. Scientist-Led Discussion of Next Steps & Further Improvements (Scientist)
- Write a dedicated "7. Future Directions & Next Steps" section covering:
  - Integration of continuous geodesic shooting and Stationary Velocity Fields (SVF).
  - Incorporation of multi-modal deep feature metrics (dino_2_lncc, vgg_4_lncc) for cross-modality registration.
  - Multi-GPU distributed parallelization via JAX vmap/pmap and PyTorch Distributed Data Parallel (DDP).
  - Surface-constrained cortical registration integrating Freesurfer/Mindboggle surface meshes.

## Acceptance Criteria

### Publication Rigor & Completeness
- [ ] Manuscript at /Users/stnava/code/syntx/docs/manuscript/manuscript_report.md includes formal inferential statistical test results (t, p, W, Cohen's d, CI_95%).
- [ ] High-resolution data plots (fig6_dice_distribution_violin.png, fig7_regional_dkt31_heatmap.png, fig8_runtime_versus_accuracy.png) are generated and embedded.
- [ ] Educational illustrations (fig9_diffeomorphic_invertibility_concept.png) and callout boxes are embedded.
- [ ] Dedicated Section 7 detailing future research directions (geodesic shooting, deep feature metrics, multi-GPU scaling, surface constraints) is included.
- [ ] Markdown, standalone HTML (manuscript_report.html), and PDF (manuscript_report.pdf) formats are updated and compiled cleanly.

## 2026-07-26T23:36:01Z

Establish complete technical details, design goals, and rigorous evaluation results demonstrating numerical and algorithmic parity between ANTs C++ SyN, syntx.syn (JAX backend), and syntx.syn (PyTorch backend). Extend the evaluation strategy and port verified algorithmic features symmetrically from JAX to PyTorch, following a strict workflow order: Evaluation Strategy → Design Goals → Implementation → Empirical Evaluation Results → PyTorch Porting & Parity Verification.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. Evaluation Strategy & Benchmark Design
Develop a comprehensive, multi-pair 3D evaluation strategy for ANTs Py/C++ SyN vs syntx.syn (JAX) vs syntx.syn (PyTorch on MPS/CPU) including TVF integration, measuring Mindboggle DKT Cortical Dice overlap, inverse identity error (||phi_inv o phi_fwd - I||), bending energy (E_2nd), Jacobian determinants, and execution runtime.

### R2. Algorithmic Parity Design Goals (JAX & PyTorch)
Define explicit design goals enforcing strict algorithmic parity across PyTorch and JAX backends per Syntx registration guardrails (Single Interpolation Policy, LNCC variance floor max(Var(I), 10^-6), Cauchy-Schwarz [-1, 1] clamping, physical coordinate domain mapping, and Anderson Acceleration default inversion).

### R3. Implementation & Symmetrical Porting to PyTorch
Implement any missing features or fixes in syntx.syn / syntx.syn_jax / syntx.tvf, and symmetrically port all verified JAX algorithms and safeguards to PyTorch.

### R4. Empirical Verification & Results Artifacts
Gather empirical runtime and overlap verification across Mindboggle subject pairs, generate comparative tables and figures, and record structured results in benchmark_results.json.

## Acceptance Criteria

### Algorithmic Parity & Accuracy
- [ ] JAX and PyTorch backends produce Cortical Mindboggle Dice overlap within <= 0.005 of each other across benchmark pairs.
- [ ] Both JAX and PyTorch enforce Anderson Acceleration (inverse_method='anderson', inverse_steps=30) as default.
- [ ] Single Interpolation Policy (Rule 1) and LNCC variance floor / clamping (Rule 2) are verified symmetrically across all backends.

### Verification & Documentation
- [ ] All unit tests pass in pytest.
- [ ] Structured results saved to benchmark_results.json.

## 2026-07-27T13:53:07Z

Fix velocity gradient smoothing tensor layout in syntx/tvf.py, correct axial slice orientation in docs/tvf_guide.html matching ants.plot, and regenerate clean diffeomorphic figures.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. TVF Velocity Gradient Smoothing Fix (syntx/tvf.py)
Fix tensor layout passed to separable_gaussian_filter in TVFModel.fit() so it receives channel-last tensors (1, *spatial, dim) rather than channel-first (1, dim, *spatial). Ensure fluid regularization smooths spatial gradients correctly without permuting spatial dimensions.

### R2. Fold-Free Diffeomorphic Registration
Verify that 3D TVF registration on test brain volumes (OASIS-TRT-20) produces smooth, diffeomorphic displacement fields with min det J(x) > 0 and no grid folding.

### R3. docs/tvf_guide.html Figure Orientation & Integrity
Regenerate Figure 2 (geodesic trajectory) and Figure 3 (grid + Jacobian map) so that:
- Axial slice orientation matches ants.plot (origin='lower', Anterior at bottom).
- Deformation grids overlay correctly without folds or cross-axis inversions.
- MathJax 3 LaTeX rendering in docs/tvf_guide.html remains clean with no corrupted escape characters.

## Acceptance Criteria

### Code & Math Verification
- [ ] pytest tests/test_tvf.py passes cleanly.
- [ ] min det J(x) > 0 across the full deformation grid for TVF registrations.

### Figure & Documentation Verification
- [ ] docs/assets/tvf_geodesic_trajectory.png and docs/assets/tvf_grid_and_jacobian.png show smooth, fold-free grid lines in ants.plot axial orientation.
- [ ] docs/tvf_guide.html opens cleanly with rendered math and aligned figures.

## 2026-07-27T13:54:00Z

REQUIREMENT UPDATE: Implement complete PyTorch <=> JAX parity for TVF (GEMINI.md Rule 9).

1. Implement TVFModelJAX in src/syntx/tvf_jax.py (or src/syntx/syn_jax.py) mirroring PyTorch TVFModel algorithmically across all pipeline stages (RK4/Euler integration, midpoint-symmetric LNCC loss, fluid gradient smoothing, and multi-res pyramid fit).
2. Export TVFModelJAX and TVFModel in syntx.__init__.py.
3. Add comparative parity unit tests in tests/test_tvf.py verifying that PyTorch and JAX TVF outputs match within floating point tolerance (<= 0.001).

## 2026-07-27T23:27:22Z

Achieve 3D Cortical DICE Parity (DICE_TVF > DICE_SyN ≈ 0.5975) on Mindboggle Pair 87 using time-varying velocity field (TVF) registration in syntx. Generate an interactive, publication-quality HTML report (docs/pareto_3d_mindboggle_report.html) containing 4-panel registration visual figures and a complete quantitative metrics table across all parameter sets, modeled after file:///Users/stnava/code/syntx/docs/pareto_diffeo_report.html.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. 3D Cortical Registration Parity Goal
Systematically optimize TVF registration on 3D Mindboggle Pair 87 (NKI-TRT-20-19 vs MMRR-21-12) to exceed the PyTorch SyN Cortical DICE baseline (> 0.5975) while enforcing strict diffeomorphic invertibility (det(J) > 0.0 everywhere, zero grid folding, mean inverse identity error < 0.5 mm).

### R2. Systematic Parameter Sweeps & Invariant Similarity Metric
Sweep non-loss TVF parameters (velocity grid resolution downsampling, LARS learning rates, progressive keyframe schedules, fluid regularization sigma = sqrt(3.0), RK4 vs Euler integration substeps). The similarity loss MUST be strictly fixed to standard 3D Intensity LNCC (window_size = 9).

### R3. Standalone Interactive HTML Pareto & Diffeomorphic Report
Generate a complete, publication-grade HTML report (docs/pareto_3d_mindboggle_report.html) modeled after docs/pareto_diffeo_report.html. The report must include summary metric cards, an interactive visual tab viewer with 4-panel registration figures (Panel A: Deformed Mesh Grid, Panel B: Divergent Jacobian Map, Panel C: Inverse Error Map in mm, Panel D: Edge Overlap), and a comprehensive multi-column quantitative comparison table.

## Acceptance Criteria

### Performance & Diffeomorphism
- [ ] TVF 3D Cortical DICE on Pair 87 exceeds PyTorch SyN Baseline (> 0.5975) under fixed 3D LNCC loss.
- [ ] 0.00% grid folding (min det(J) > 0.0 everywhere).
- [ ] Mean inverse identity error < 0.5 mm.

### Report Deliverables
- [ ] HTML report generated at docs/pareto_3d_mindboggle_report.html.
- [ ] Standardized 4-panel PNG figures generated for each experiment in docs/assets/.
- [ ] Complete quantitative metrics table including LNCC, DICE, Inverse Error (Max & Mean), det(J) range, Folding %, and Runtime.
