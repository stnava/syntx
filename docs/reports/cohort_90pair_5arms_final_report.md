# Full Population 90-Pair 5-Arm Mindboggle-101 Benchmark Report

**Generated:** 2026-09-20 09:24:12
**Total Pairs:** 90 (40 intra-study, 50 inter-study)
**Seeding:** Single canonical affine seed (`syntx.robust_affine`, mode='auto') shared across all methods.

## Aggregate Results Across Full Cohort (N = 90)

| Method | Mean Dice | Median Dice | Std Dev | Win vs ANTs | Win Rate | Mean Time (s) | Mean Folding % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Robust Affine Seed** | 0.3448 | 0.3448 | 0.0230 | — | — | 5.3s | — |
| **ANTs C++ SyN (Live)** | 0.6097 | 0.6090 | 0.0220 | — | — | 124.3s | 0.0000% |
| **syntx.syn (Sobolev)** | **0.6350** | **0.6343** | 0.0215 | 90/90 | **100.0%** | **51.6s** | 0.0066% |
| **syntx.tvf (DSTI-1)** | **0.6387** | **0.6384** | 0.0219 | 90/90 | **100.0%** | 250.4s | **0.00246%** |
| **syntx.syngs (Shooting)** | **0.6187** | **0.6182** | 0.0227 | 78/90 | **86.7%** | 115.9s | 0.0019% |
| **syntx.greedy** | **0.6071** | **0.6074** | 0.0250 | 33/90 | **36.7%** | 12.9s | 0.0008% |
