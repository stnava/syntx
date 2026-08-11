# Handoff Report — Milestone 2: Systematic Ablation Fix 1 (LNCC padding_mode='zeros')

## 1. Observation
- Executed `scripts/run_m2_fix1_lncc_zeros.py` on 3D Native Pair 0 (`NKI-TRT-20-3` -> `NKI-RS-22-22`).
- Command: `python3 scripts/run_m2_fix1_lncc_zeros.py`
- Configuration tested:
  - `padding_mode = 'zeros'` (Fix 1 applied)
  - `fast_smooth = True` (baseline preserved)
  - `in_loop_inv_steps = 6` (baseline preserved)
  - `reg_iterations = [100, 100, 20]`
  - `fluid_sigma = 3.0`, `total_sigma = 0.0`
- Execution logs summary:
  ```
  =====================================================================
   MILESTONE 2 FIX 1 RESULTS
  =====================================================================
    Fixed Space Cortical Dice:    0.5466
    Moving Space Cortical Dice:   0.5454
    Symmetric Mean Cortical Dice: 0.5460
    Grid Folding Percentage:       0.0000 %
    Minimum Jacobian Det:          0.1288
    Execution Runtime:             57.33 s
  =====================================================================
  ```
- Generated Artifacts:
  - HTML Report: `/Users/stnava/code/syntx/docs/reports/fix1_lncc_zeros_report.html` (embedded Standard 5-Figure Visual Suite via `syntx.viz.create_registration_report`)
  - Metrics JSON: `/Users/stnava/code/syntx/docs/reports/fix1_lncc_zeros_metrics.json`

## 2. Logic Chain
1. Milestone 1 established the exploit baseline at commit `01d74b0` (`padding_mode='border'`, `fast_smooth=True`, `in_loop_inv_steps=6`) with Sym Dice `0.5468`, Grid Folding `0.0000%`, and min det(J) `0.1239`.
2. Milestone 2 isolates Fix 1 by setting `padding_mode='zeros'` in LNCC while holding `fast_smooth=True` and `in_loop_inv_steps=6` constant.
3. Changing LNCC padding from `'border'` to `'zeros'` prevents synthetic boundary intensity extrapolation during cross-correlation box filtering, slightly altering local gradient forces near image volume borders.
4. The resulting evaluation demonstrates:
   - Fixed Space Dice: 0.5466
   - Moving Space Dice: 0.5454
   - Symmetric Mean Dice: 0.5460 (a slight decrease of ~0.0008 relative to M1 baseline 0.5468)
   - Grid Folding Percentage: 0.0000 % (diffeomorphic topology strictly preserved)
   - Minimum Jacobian Det: 0.1288
   - Runtime: 57.33 seconds
5. All outputs were verified to exist on disk and correctly saved in `docs/reports/`.

## 3. Caveats
- Fast gaussian smoothing (`fast_smooth=True`) and un-enforced symmetric inverse steps (`in_loop_inv_steps=6`) remain in place as baseline controls for M2. They will be isolated sequentially in M3 and M4.
- Execution device was Apple Silicon MPS (`device='mps'`).

## 4. Conclusion
- Milestone 2 execution completed successfully with genuine registration logic and metrics calculations.
- Quantitative comparison:
  - Baseline (M1): Sym Dice = 0.5468, Grid Folding % = 0.0000%, min det(J) = 0.1239, Runtime = 69.03 s
  - Fix 1 (M2): Sym Dice = 0.5460, Grid Folding % = 0.0000%, min det(J) = 0.1288, Runtime = 57.33 s
- Generated report artifacts verified at `docs/reports/fix1_lncc_zeros_report.html` and `docs/reports/fix1_lncc_zeros_metrics.json`.

## 5. Verification Method
1. Re-run `python3 scripts/run_m2_fix1_lncc_zeros.py`.
2. Inspect `docs/reports/fix1_lncc_zeros_metrics.json` and confirm metrics match `dice_sym ~ 0.5460` and `folding_pct == 0.0`.
3. Open `docs/reports/fix1_lncc_zeros_report.html` in browser to visually confirm the Standard 5-Figure Visual Suite.
