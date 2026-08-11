# Milestone 4 Handoff Report: Exploit Fix 3 (in_loop_inv_steps=10)

## 1. Observation

### Execution Logs & Metrics Output
Command executed:
`python3 scripts/run_m4_fix3_inv_steps_10.py`

Verbatim Output:
```
=====================================================================
 Milestone 4: Systematic Ablation Fix 3 (in_loop_inv_steps=10)
 Pair 0: NKI-TRT-20-3 (Fixed) -> NKI-RS-22-22 (Moving)
 Configuration: padding_mode='zeros', fast_smooth=False, in_loop_inv_steps=10, inverse_steps=10
=====================================================================

[1/4] Loading 3D Native Pair 0 Volumes...
  Fixed Image:  (192, 256, 256), Spacing: (1.0, 1.0, 1.0), Origin: (-95.5, 102.0, -152.0)
  Moving Image: (192, 256, 256), Spacing: (1.0, 1.0, 1.0), Origin: (-93.174560546875, 102.0, -143.2403564453125)
  Execution Device: mps

[2/4] Computing Robust Affine Initialization...
  Robust Affine completed in 11.13 s

[3/4] Running SyN Registration (Fix 3: in_loop_inv_steps=10)...
[pytorch-fit] SyN Level 0 Epoch 36: loss=-0.925833 (lncc=-0.925833), warp_l2r max norm=5.4027
[pytorch-fit] SyN Level 0 converged at Epoch 36.
[pytorch-fit] SyN Level 1 Epoch 46: loss=-0.888138 (lncc=-0.888138), warp_l2r max norm=6.7086
[pytorch-fit] SyN Level 1 converged at Epoch 46.
[pytorch-fit] SyN Level 2 Epoch 19: loss=-0.696912 (lncc=-0.696912), warp_l2r max norm=6.6229
  SyN Registration completed in 74.17 s

[4/4] Computing Quantitative Metrics & Generating HTML Report...

=====================================================================
 MILESTONE 4 FIX 3 RESULTS
=====================================================================
  Fixed Space Cortical Dice:  0.6022
  Moving Space Cortical Dice: 0.5958
  Symmetric Mean Cortical Dice: 0.5990
  Grid Folding Percentage:     0.0000 %
  Minimum Jacobian Det:        0.0528
  Execution Runtime:           74.17 s
=====================================================================

HTML Report saved to: /Users/stnava/code/syntx/docs/reports/fix3_inv_steps_10_report.html
Metrics JSON saved to: /Users/stnava/code/syntx/docs/reports/fix3_inv_steps_10_metrics.json
```

### Generated Artifacts
- **Script**: `/Users/stnava/code/syntx/scripts/run_m4_fix3_inv_steps_10.py`
- **HTML Report**: `/Users/stnava/code/syntx/docs/reports/fix3_inv_steps_10_report.html`
- **Metrics JSON**: `/Users/stnava/code/syntx/docs/reports/fix3_inv_steps_10_metrics.json`

---

## 2. Logic Chain

1. **Script Creation**: Created `scripts/run_m4_fix3_inv_steps_10.py` based on `scripts/run_m3_fix2_fast_smooth_false.py`, configuring `syntx.syn` with all 3 isolated exploit fixes active simultaneously:
   - `padding_mode='zeros'` (Fix 1 active: true zero-padded LNCC boundary metric evaluation)
   - `fast_smooth=False` (Fix 2 active: exact 3D spatial Gaussian kernel filtering)
   - `in_loop_inv_steps=10`, `inverse_steps=10` (Fix 3 active: 10 fixed-point inverse field projection steps inside every optimization loop iteration)
   - Parameters: `reg_iterations=[100, 100, 20]`, `fluid_sigma=3.0`, `total_sigma=0.0`.

2. **Benchmark Execution**: Ran `python3 scripts/run_m4_fix3_inv_steps_10.py` on 3D Native Pair 0 (`NKI-TRT-20-3` -> `NKI-RS-22-22`, 192x256x256, 1.0mm isotropic).

3. **Metric Calculation**:
   - Evaluated Mindboggle DKT31 label overlap symmetrically:
     - Fixed Space Dice: `0.6022`
     - Moving Space Dice: `0.5958`
     - **Symmetric Mean Cortical Dice**: `0.5990`
   - Evaluated physical Jacobian determinant $\det(J)$ map (`do_log=False`):
     - **Grid Folding %** ($\det(J) \le 0$): `0.0000 %`
     - **Minimum Jacobian Det**: `0.0528`
   - **Compute Runtime**: `74.17 s`

4. **Synthesis**:
   - Enforcing 10 in-loop inverse fixed-point solver iterations (`in_loop_inv_steps=10`) alongside exact Gaussian filtering (`fast_smooth=False`) and zero boundary padding (`padding_mode='zeros'`) yields a mathematically rigorous, fully symmetric, diffeomorphic SyN registration with zero grid folding (0.0000 %) and positive minimum Jacobian determinant (0.0528).

---

## 3. Caveats

- Benchmark executed on 3D Native Pair 0 (`NKI-TRT-20-3` -> `NKI-RS-22-22`).
- Device used: Apple Silicon PyTorch MPS backend (`device='mps'`).

---

## 4. Conclusion

Milestone 4 (Fix 3: Symmetric Inverse `in_loop_inv_steps=10`) execution completed successfully.

**Quantitative Summary Table**:
| Milestone | State / Fix | Sym Dice | Grid Folding % | Min det(J) | Runtime (s) |
|---|---|---|---|---|---|
| **M4** | **Fix 3 (`in_loop_inv_steps=10`)** | **0.5990** | **0.0000 %** | **0.0528** | **74.17 s** |

- Interactive HTML report embedding the Standard 5-Figure Visual Suite is at `docs/reports/fix3_inv_steps_10_report.html`.
- Quantitative metrics JSON is saved at `docs/reports/fix3_inv_steps_10_metrics.json`.

---

## 5. Verification Method

To independently verify the Milestone 4 results:
1. Execute the benchmark script:
   ```bash
   python3 scripts/run_m4_fix3_inv_steps_10.py
   ```
2. Verify the saved metrics JSON file:
   ```bash
   cat docs/reports/fix3_inv_steps_10_metrics.json
   ```
3. Confirm HTML report existence:
   ```bash
   ls -lh docs/reports/fix3_inv_steps_10_report.html
   ```
