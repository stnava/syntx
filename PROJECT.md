# Project: syntx Registration Reconstruction Study

## Architecture
- Target project: `syntx` medical image registration framework.
- Execution domain: PyTorch / ANTsPy 3D registration benchmarking on native brain volumes.
- Dataset: Native Pair 0 (`NKI-TRT-20-3` -> `NKI-RS-22-22`, 192x256x256, 1.0mm isotropic).
- Verification & Reports: Standard 5-Figure Visual Suite via `syntx.viz.create_registration_report`, Sym Dice (`compute_bidirectional_dice`), Grid Folding % (`det(J) <= 0` with `do_log=False`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1 Exploit Baseline Benchmark | Execute commit 01d74b0 baseline on Pair 0 (padding_mode='border', fast_smooth=True, in_loop_inv_steps=6) | M1 | ORIGINAL_REQUEST §R1 |
| 2 | R1 Baseline HTML Report | Generate interactive HTML report with Standard 5-Figure Visual Suite for Baseline | M1 | ORIGINAL_REQUEST §R1 |
| 3 | R2.a Fix 1 LNCC Metric Padding | Apply padding_mode='zeros' fix and benchmark Sym Dice & Grid Folding % | M2 | ORIGINAL_REQUEST §R2.a |
| 4 | R2.a Fix 1 HTML Report | Generate interactive HTML report with Standard 5-Figure Visual Suite for Fix 1 | M2 | ORIGINAL_REQUEST §R2.a |
| 5 | R2.b Fix 2 Elastic Smoothing | Apply fast_smooth=False fix and benchmark Sym Dice & Grid Folding % | M3 | ORIGINAL_REQUEST §R2.b |
| 6 | R2.b Fix 2 HTML Report | Generate interactive HTML report with Standard 5-Figure Visual Suite for Fix 2 | M3 | ORIGINAL_REQUEST §R2.b |
| 7 | R2.c Fix 3 Symmetric Inverse | Apply in_loop_inv_steps=10 fix and benchmark Sym Dice & Grid Folding % | M4 | ORIGINAL_REQUEST §R2.c |
| 8 | R2.c Fix 3 HTML Report | Generate interactive HTML report with Standard 5-Figure Visual Suite for Fix 3 | M4 | ORIGINAL_REQUEST §R2.c |
| 9 | R3 Optimization Mechanics Isolation | Analyze CFL normalization / gradient scaling logic from 01d74b0 vs 0.6095 target | M5 | ORIGINAL_REQUEST §R3 |
| 10| Final Reconstruction Summary Report | Markdown report summarizing step-by-step findings table and final analysis | M5 | ORIGINAL_REQUEST §Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Baseline Exploit Benchmark | Benchmark commit 01d74b0 state on Pair 0 + Baseline HTML Report | None | DONE |
| M2 | Exploit Fix 1 (LNCC zeros) | Apply padding_mode='zeros' fix + Fix 1 HTML Report | M1 | DONE |
| M3 | Exploit Fix 2 (fast_smooth=False) | Apply fast_smooth=False fix + Fix 2 HTML Report | M2 | DONE |
| M4 | Exploit Fix 3 (in_loop_inv_steps=10) | Apply in_loop_inv_steps=10 fix + Fix 3 HTML Report | M3 | DONE |
| M5 | Mechanics Analysis & Final Report | Analyze R3 mechanics + Markdown Summary Report & Table | M4 | DONE |


## Interface Contracts
### Benchmark Harness ↔ Visual Reporting (`syntx.viz`)
- `fi`, `mi`, `fl`, `ml`: ANTsImages in physical space.
- `reg`: Output dictionary containing `fwdtransforms`, `invtransforms`, `warped`, `warp`, `detJ`, `inv_err_map`, `model='SyNModel'`.
- `compute_bidirectional_dice`: returns `(dice_fixed, dice_moving, dice_sym)`.
- `folding_pct`: `np.mean(jac_np[mask] <= 0) * 100.0` with `ants.create_jacobian_determinant_image(..., do_log=False)`.
- `create_registration_report`: generates standalone HTML file + `assets/` directory.

## Code Layout
- `src/syntx/syn.py`: SyN algorithm implementation.
- `src/syntx/viz/`: Visual reporting package.
- `src/syntx/benchmark/`: Benchmark orchestration and worker scripts.
- `/Users/stnava/data/mindboggle/volumes/`: Dataset location.
- `.agents/`: Agent metadata and reports.
