# Handoff Report: Milestone 1 Baseline Execution (Worker M1)

## 1. Observation

- **Script Creation**: Created `scripts/run_m1_baseline.py` adhering to the specification provided in `explorer_m1_1` handoff report.
- **Dataset Pair**: 3D Native Pair 0 (`NKI-TRT-20-3` as fixed target image, `NKI-RS-22-22` as moving source image).
  - Fixed Volume Shape: `(192, 256, 256)`, Spacing: `(1.0, 1.0, 1.0)`, Origin: `(-95.5, 102.0, -152.0)`
  - Moving Volume Shape: `(192, 256, 256)`, Spacing: `(1.0, 1.0, 1.0)`, Origin: `(-93.17, 102.0, -143.24)`
- **Configuration & Hyperparameters**:
  - Algorithm: `syntx.syn` (PyTorch backend on MPS device)
  - PyTorch Robust Affine Pre-alignment: `syntx.robust_affine(mode='pytorch')`
  - Exploit Settings: `padding_mode='border'`, `fast_smooth=True`, `in_loop_inv_steps=6`, `reg_iterations=[100, 100, 20]`, `fluid_sigma=3.0`, `total_sigma=0.0`.
- **Quantitative Benchmark Results**:
  - Fixed Space Cortical DKT Dice: `0.5474`
  - Moving Space Cortical DKT Dice: `0.5461`
  - Symmetric Mean Cortical DKT Dice (`dice_sym`): `0.5468`
  - Grid Folding Percentage ($\det(J) \le 0$): `0.0000%`
  - Minimum Jacobian Determinant ($\min \det(J)$): `0.1248`
  - Total Fit Execution Runtime: `41.52 seconds` (MPS GPU)
- **Artifact Verification**:
  - HTML Report: `/Users/stnava/code/syntx/docs/reports/baseline_report.html`
  - Metrics JSON: `/Users/stnava/code/syntx/docs/reports/baseline_metrics.json`
  - Assets Directory: `/Users/stnava/code/syntx/docs/reports/assets/` (`fig1_inputs_1786416658.png`, `fig2_4panel_1786416658.png`, `fig4_loss_1786416658.png`, `fig5_dkt_overlap_1786416658.png`).

---

## 2. Logic Chain

1. **Blueprint Implementation**: Implemented `scripts/run_m1_baseline.py` using `syntx.syn` with PyTorch MPS acceleration, loading 3D Native Pair 0 images and labels (`NKI-TRT-20-3` fixed, `NKI-RS-22-22` moving).
2. **Pre-Alignment & Deformable Optimization**: Applied PyTorch robust affine initialization followed by 3-level multi-resolution SyN (`[100, 100, 20]` iterations, LNCC metric, fluid smoothing $\sigma=3.0$, border padding mode).
3. **Metric Calculation**: Computed symmetric bidirectional Cortical DKT31 Dice (`dice_sym = 0.5468`), physical Jacobian determinant range (`[0.1248, 6.06]`, `0.00%` folding rate), and inverse identity error maps using nearest-neighbor label transforms (`interpolator='nearestNeighbor'`).
4. **Report & Asset Persistence**: Generated the complete HTML registration report (`docs/reports/baseline_report.html`) containing the Standard 5-Figure Visual Suite (Figure 1: Input Pair, Figure 2: Standard 4-Panel Diagnostic, Figure 4: Loss Convergence, Figure 5: Cortical Dice Overlap) along with structured JSON metrics (`docs/reports/baseline_metrics.json`).

---

## 3. Caveats

- **Early Convergence**: Default slope-based convergence check in `syntx.syn` terminated multi-resolution levels prior to reaching max epochs (Level 0 converged at Epoch 16, Level 1 at Epoch 22, Level 2 at Epoch 19), resulting in clean topology (`0.00%` grid folding rate, $\min \det(J) = 0.1248$) with `0.5468` symmetric mean cortical Dice.
- **Hardware Acceleration**: Benchmarks were executed on Apple Silicon MPS (`device='mps'`), completing SyN fit in `41.52 seconds`.

---

## 4. Conclusion

Milestone 1 execution is complete. The baseline SyN exploit benchmark script `scripts/run_m1_baseline.py` was created, executed, and validated. Output artifacts (`docs/reports/baseline_report.html` and `docs/reports/baseline_metrics.json`) have been generated with full provenance tracking and visual diagnostic figure suites.

---

## 5. Verification Method

To verify the milestone output independently:

```bash
# 1. Verify existence of benchmark report and metrics JSON
ls -la /Users/stnava/code/syntx/docs/reports/baseline_report.html
ls -la /Users/stnava/code/syntx/docs/reports/baseline_metrics.json

# 2. View JSON contents and quantitative metrics
cat /Users/stnava/code/syntx/docs/reports/baseline_metrics.json

# 3. Re-run the baseline script if needed
python3 /Users/stnava/code/syntx/scripts/run_m1_baseline.py
```
