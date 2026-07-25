# Original User Request

## 2026-07-25T13:17:12Z

Drafting a comprehensive manuscript report detailing the Syntx 90-pair Mindboggle registration evaluation results, backend optimizations, and regional DKT31 performance.

Working directory: /Users/stnava/code/syntx/docs/manuscript
Integrity mode: development

## Requirements

### R1. Comprehensive Manuscript Document
- Generate a publication-ready Markdown manuscript (`manuscript_report.md` located at `/Users/stnava/code/syntx/docs/manuscript/manuscript_report.md`) detailing the `syntx` package design, mathematical formulation, backend parity engineering, performance optimizations, and full 90-pair Mindboggle benchmark results.

### R2. Empirical Benchmarking & Outlier Analysis
- Present the 100% verified, outlier-corrected summary statistics across all 90 Mindboggle subject pairs comparing Syntx JAX, Syntx PyTorch, and the C++ ANTs SyN baseline:
  - **Syntx JAX**: **`0.5676` Mean / `0.5978` Median Cortical Dice** (+0.0068 / +0.0091 vs ANTs), **`45.5s` per pair** ($6.6\times$ speedup), **`0.00000%` folding rate**.
  - **Syntx PyTorch**: **`0.5593` Mean / `0.5913` Median Cortical Dice** (+0.0026 Median vs ANTs), **`14.1s` per pair** ($21.3\times$ speedup), **`0.00000%` folding rate**.
  - **ANTs C++ Baseline**: **`0.5608` Mean / `0.5887` Median Cortical Dice**, `301.5s` per pair, `0.00000%` folding rate.
- Include a dedicated section analyzing the 5 raw dataset orientational outliers (Pairs 14, 41, 44, 53, 55) caused by $180^\circ$ NIfTI header flips, and show how rotational initialization (`search_factor=30`, `radian_fraction=0.8`) resolves alignment (Pair 55: JAX `0.6113` / PyTorch `0.5998` vs ANTs `0.4819`).

### R3. Regional DKT31 Cortical Breakdown
- Provide detailed, individual region-level DKT31 cortical Dice tables and analysis across major neuroanatomical structures (precentral, postcentral, superior frontal, superior temporal, cingulate, insula, occipital, parietal).

### R4. Core System & Mathematical Insights
- Detail everything learned during system development:
  - Single Interpolation Policy (no intermediate pre-warping)
  - LNCC autograd derivative variance floor ($\text{Var}_{\text{safe}} = \max(\text{Var}(I), 10^{-6})$) & Cauchy-Schwarz $[-1, 1]$ clamping
  - Gradient preservation in Lie Algebra rotation parameterization
  - ITK CFL gradient step physical spacing multiplier ($\text{step} \cdot \text{spacing}$)
  - Zero-permute 3D depthwise separable Conv3D kernel optimization (`F.conv3d(..., groups=C)`)
  - XLA Eigen multi-threading in JAX CPU (`XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"`).

## Acceptance Criteria

### Publication Integrity & Completeness
- [ ] Manuscript is created at `/Users/stnava/code/syntx/docs/manuscript/manuscript_report.md` and includes Abstract, Introduction, Mathematical & Backend Parity Methods, 90-Pair Outlier-Corrected Benchmark Results Table, Regional DKT31 Cortical Breakdown Table, Orientational Outliers Case Study, and Discussion sections.
- [ ] Incorporates 100% verified 90-pair metrics: JAX Mean/Median Dice (`0.5676` / `0.5978`), PyTorch Mean/Median Dice (`0.5593` / `0.5913`), ANTs Baseline (`0.5608` / `0.5887`), $21.3\times$ PyTorch speedup (`14.1s` vs `301.5s`), $6.6\times$ JAX speedup (`45.5s`), and `0.00000%` folding rate.
- [ ] Includes detailed mathematical formulations and code patterns for all optimizations and parity guardrails.
