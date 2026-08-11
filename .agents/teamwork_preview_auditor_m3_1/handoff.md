# Handoff Report — Forensic Audit M3 (Exploit Fix 2: fast_smooth=False)

## 1. Observation
- **Audited Target Files**:
  - Script: `/Users/stnava/code/syntx/scripts/run_m3_fix2_fast_smooth_false.py`
  - HTML Report: `/Users/stnava/code/syntx/docs/reports/fix2_fast_smooth_false_report.html`
  - Metrics JSON: `/Users/stnava/code/syntx/docs/reports/fix2_fast_smooth_false_metrics.json`
- **Static Analysis Findings**:
  - `scripts/run_m3_fix2_fast_smooth_false.py` imports `syntx`, loads native 3D volumes (`NKI-TRT-20-3` fixed, `NKI-RS-22-22` moving), and executes `syntx.syn(...)` with `fast_smooth=False`, `padding_mode='zeros'`, and `in_loop_inv_steps=6`.
  - Dice metrics are computed dynamically via `compute_bidirectional_dice(...)` using ANTsPy `label_overlap_measures`.
  - Grid folding percentage is computed dynamically via `ants.create_jacobian_determinant_image(..., do_log=False)` with `jac_np[mask] <= 0.0`.
  - No hardcoded metrics, facade functions, or pre-populated stub figures were found.
  - In `src/syntx/syn.py` (lines 2954-2984), `fast_smooth=False` genuinely routes fluid smoothing through `separable_gaussian_filter(..., curr_fluid_sig)` (exact 3D spatial Gaussian filtering) rather than the fast spectral approximation.
- **Independent Test Execution Results**:
  - Command: `python3 scripts/run_m3_fix2_fast_smooth_false.py`
  - Execution completed cleanly in 64.40 s on MPS (`mps`) backend.
  - SyN Level 1 converged at Epoch 62; Level 2 ran for 20 epochs.
  - Fixed Space Cortical Dice: `0.6041`
  - Moving Space Cortical Dice: `0.5971`
  - Symmetric Mean Cortical Dice: `0.6006` (matches worker report of `0.6007`)
  - Grid Folding Percentage: `0.0000 %` (0 folded voxels)
  - Minimum Jacobian Det: `0.0462` (strictly positive, topology preserving)
- **Artifact Verification**:
  - Interactive HTML report generated at `docs/reports/fix2_fast_smooth_false_report.html` embedding the Standard 5-Figure Visual Suite assets (`fig1_inputs_*.png`, `fig2_4panel_*.png`, `fig4_loss_*.png`, `fig5_dkt_overlap_*.png`).
  - Metrics JSON saved at `docs/reports/fix2_fast_smooth_false_metrics.json` matching the dynamic benchmark outputs.

## 2. Logic Chain
1. **Verification of Ground-Truth Constraints**:
   `ORIGINAL_REQUEST.md` requires systematic ablation of Fix 2 (`fast_smooth=False`) in development integrity mode, reporting both Sym Dice and Grid Folding %.
2. **Static Source Verification**:
   Examined `scripts/run_m3_fix2_fast_smooth_false.py` and `src/syntx/syn.py`. Confirmed that `fast_smooth=False` is passed to `syn()` and enforced in the fluid smoothing step via `separable_gaussian_filter`. Metric calculations are fully dynamic without hardcoded constants.
3. **Empirical Reproduction**:
   Ran `python3 scripts/run_m3_fix2_fast_smooth_false.py` independently. The run produced consistent numbers (Sym Dice `0.6006`, Grid Folding `0.0000%`, runtime `64.40 s`), confirming that the worker's results were authentically computed without manipulation or shortcuts.
4. **Artifact Integrity**:
   Checked generated HTML report and metrics JSON. All assets are correctly linked and populated with true execution data.

## 3. Caveats
- Benchmark execution ran on Apple Silicon (`mps`) device. Minor numerical variations in floating-point operations across different GPU backends or PyTorch versions may cause slight variations in runtime (~64s vs ~79s) or minimum Jacobian determinant (~0.046 vs ~0.048), but the overall metric values match within expected tolerance.

## 4. Conclusion
**Verdict: CLEAN**

Milestone 3 work products (`scripts/run_m3_fix2_fast_smooth_false.py`, `docs/reports/fix2_fast_smooth_false_report.html`, and `docs/reports/fix2_fast_smooth_false_metrics.json`) pass all static, behavioral, and forensic integrity checks. The implementation of `fast_smooth=False` is genuine and authentic.

## 5. Verification Method
To independently re-verify this verdict:
1. Run `python3 scripts/run_m3_fix2_fast_smooth_false.py` from `/Users/stnava/code/syntx`.
2. Verify console output reports Symmetric Mean Cortical Dice ~`0.6007` and Grid Folding Percentage `0.0000 %`.
3. Check JSON metrics at `/Users/stnava/code/syntx/docs/reports/fix2_fast_smooth_false_metrics.json`.
4. Open and inspect HTML report at `/Users/stnava/code/syntx/docs/reports/fix2_fast_smooth_false_report.html`.
