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
