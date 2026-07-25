# Implementation Plan — manuscript_report.md Enhancement

## Objective
Collaborative enhancement of `manuscript_report.md` bringing together Statistician, Visualization Expert, Educator, and Computer Vision Scientist expertise to fulfill requirements R1-R4, embed figures fig6-fig9, add educational callout boxes, include Section 7 (Future Directions), update compiled HTML/PDF artifacts, and verify complete compliance with GEMINI.md user rules.

## Milestones

| Milestone | Name | Description | Worker Archetype | Outputs | Status |
|-----------|------|-------------|------------------|---------|--------|
| M1 | Statistical Rigor & Hypotheses (R1) | Perform paired t-tests, Wilcoxon signed-rank tests, Cohen's d, 95% CIs across 90 Mindboggle pairs (JAX, PyTorch, ANTs C++). Per-lobe & per-region tests. Insert statistical analysis tables & narrative into manuscript. | teamwork_preview_worker (Statistician) | Statistical tables & text in manuscript_report.md | DONE |
| M2 | Data Visualization & Quantitative Plots (R2) | Generate publication-quality plots: fig6_dice_distribution_violin.png, fig7_regional_dkt31_heatmap.png, fig8_runtime_versus_accuracy.png. Embed in manuscript. | teamwork_preview_worker (Visualization Expert) | fig6, fig7, fig8 in figures/, embedded in manuscript_report.md | DONE |
| M3 | Educational Conceptual Illustrations & Callout Boxes (R3) | Generate fig9_diffeomorphic_invertibility_concept.png and insert clear callout boxes for LNCC Variance Floor, Lie Algebra so(3) Exponential Map, and Single Interpolation Policy. | teamwork_preview_worker (Educator) | fig9 in figures/, callout boxes in manuscript_report.md | DONE |
| M4 | Scientist-Led Future Directions (R4) | Draft and integrate Section 7 ("Future Directions & Next Steps") covering geodesic shooting, deep feature metrics, multi-GPU scaling, and surface constraints. | teamwork_preview_worker (Scientist) | Section 7 in manuscript_report.md | DONE |
| M5 | Compilation, Review & Forensic Audit | Verify all manuscript content, compile manuscript_report.html and manuscript_report.pdf, verify zero integrity violations via Forensic Auditor. | teamwork_preview_worker + teamwork_preview_auditor | manuscript_report.html, manuscript_report.pdf, Clean Audit Verdict | DONE |

## Verification Criteria
1. manuscript_report.md includes formal statistical test results (t, p, W, Cohen's d, CI_95%). [VERIFIED]
2. High-resolution plots fig6, fig7, fig8 generated and embedded in manuscript_report.md. [VERIFIED]
3. Educational illustration fig9 and callouts (Var_safe, Lie Algebra so(3), Single Interpolation) embedded. [VERIFIED]
4. Section 7 detailing future directions integrated into manuscript_report.md. [VERIFIED]
5. HTML and PDF compiled cleanly without errors. [VERIFIED]
6. Forensic Auditor reports CLEAN verdict with zero integrity violations. [VERIFIED]
