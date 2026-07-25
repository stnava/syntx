## 2026-07-25T14:25:01Z
Role: Statistician Specialist
Working directory: /Users/stnava/code/syntx/.agents/teamwork_preview_worker_m1_1

Objective:
Perform and document formal inferential statistical tests across the 90 Mindboggle benchmark pairs comparing Syntx JAX, Syntx PyTorch, and ANTs C++ baseline, fulfilling requirement R1.

Tasks:
1. Examine existing benchmark results in /Users/stnava/code/syntx/outputs_comparison/ (such as r2_3d_results.csv, r2_3d_sweep_results.csv, pairs.csv, etc.) and/or write a Python statistical calculation script to compute:
   - Paired two-sample t-tests (t-statistic, p-value, degrees of freedom df) comparing JAX vs ANTs C++, PyTorch vs ANTs C++, JAX vs PyTorch.
   - Non-parametric Wilcoxon signed-rank tests (W-statistic, p-value) for the same comparisons.
   - Cohen's d effect size calculations and 95% Confidence Intervals (CI_95%) for Cortical Dice overlap across all 90 benchmark pairs.
   - Per-lobe (Frontal, Parietal, Temporal, Occipital, Cingulate/Insular) and per-region (31 DKT structures) statistical significance testing (t, p, W, Cohen's d).
2. Write a comprehensive markdown snippet file at /Users/stnava/code/syntx/docs/manuscript/r1_stat_rigor.md detailing all formal inferential statistical findings, complete with data tables and statistical interpretations.
3. Integrate/update the statistical test results into /Users/stnava/code/syntx/docs/manuscript/manuscript_report.md under Section 3.2, 3.3, 4.1, and 4.2.
4. Verify all math and numbers, execute tests/scripts, record results and commands, and write a comprehensive handoff report at /Users/stnava/code/syntx/.agents/teamwork_preview_worker_m1_1/handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
