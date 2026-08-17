# Head-to-Head Ablation: Sobolev SyN vs. Gaussian SyN in Syntx (Subprocess Isolated)
**Generated**: 2026-08-17 15:03:58 UTC
**Test Suite**: 6 Probe Pairs (3 Intra-Cohort: 0, 1, 2 | 3 Inter-Cohort: 45, 67, 82)

## Summary Results Table
| Model Variant | Mean Symmetric Dice | vs ANTs Baseline | Mean Grid Folding % | Mean Time (s) | Speedup vs ANTs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Syntx Sobolev SyN** | **0.6378** | **+1.60%** | **0.0001%** | **68.4s** | **2.47x** |
| **Syntx Gaussian SyN** | 0.6399 | +1.80% | 0.0002% | 62.9s | 2.68x |
| **ANTs C++ Baseline** | 0.6218 | Baseline | 0.0000% | 168.8s | 1.00x |

## Detailed Per-Pair Results
| Pair | Cohort | Fixed ID | Moving ID | Sobolev Dice | Gaussian Dice | ANTs Baseline | Sobolev vs Gaussian | Sobolev vs ANTs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #00 | INTRA | `OASIS-TRT-20-17` | `OASIS-TRT-20-16` | **0.6474** | 0.6538 | 0.6329 | **-0.64%** | **+1.45%** |
| #01 | INTRA | `NKI-RS-22-13` | `NKI-RS-22-15` | **0.6594** | 0.6509 | 0.6067 | **+0.84%** | **+5.26%** |
| #02 | INTRA | `NKI-TRT-20-10` | `NKI-TRT-20-4` | **0.6882** | 0.6900 | 0.6737 | **-0.18%** | **+1.45%** |
| #45 | INTER | `MMRR-21-17` | `NKI-TRT-20-7` | **0.6106** | 0.6149 | 0.6038 | **-0.43%** | **+0.68%** |
| #67 | INTER | `OASIS-TRT-20-17` | `NKI-RS-22-22` | **0.6125** | 0.6196 | 0.6152 | **-0.71%** | **-0.26%** |
| #82 | INTER | `OASIS-TRT-20-19` | `MMRR-21-11` | **0.6087** | 0.6099 | 0.5986 | **-0.12%** | **+1.01%** |
