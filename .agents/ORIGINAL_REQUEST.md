# Original User Request

## 2026-08-11T02:26:47Z

Perform a reconstruction study on the `syntx` registration algorithm. Start from the historic 'broken' commit (`01d74b0`) that yielded a 0.65 Dice score, isolate and fix each specific mathematical exploit (padding mode, smoothing type, inverse bounding) one by one, and benchmark the performance degradation after each fix to identify any legitimate optimizations we lost in the modern rebuild.

Working directory: /Users/stnava/code/syntx

Integrity mode: development

## Requirements

### R1. Establish the Exploit Baseline
The team must check out the `syntx` repository at commit `01d74b0`. They must write and execute a benchmark script on the 3D Native Pair 0 (`NKI-TRT-20-3` -> `NKI-RS-22-22`) to verify the baseline Sym Dice score (~0.65) and the baseline Grid Folding percentage ($\det(J) \le 0$).

### R2. Systematic Ablation of Exploits
The team must systematically apply the following three mathematical corrections to `syn.py` ONE at a time:
1. Fix LNCC metric: `padding_mode='zeros'`
2. Fix Elastic Smoothing: `fast_smooth=False`
3. Enforce Symmetric Inverse: `in_loop_inv_steps=10`

### R3. Isolate Legitimate Optimization Mechanics
After all exploits are removed, if the resulting algorithm scores higher than `0.6095`, the team must investigate the original `01d74b0` gradient scaling and step normalization logic (e.g., CFL normalization) to identify exactly what legitimate mathematical mechanic is driving the faster convergence.

## Acceptance Criteria

### Verification Artifact
- [ ] A complete interactive HTML report (via `syntx.viz.create_registration_report`) is generated for each state of the algorithm (Baseline, and after each of the 3 isolated fixes), explicitly containing the Standard 5-Figure Visual Suite.
- [ ] A markdown report is generated containing a step-by-step table summarizing the findings.
- [ ] The table strictly reports both **Sym Dice** and **Grid Folding %** for the Baseline and after each of the 3 isolated fixes.
- [ ] The final analysis explicitly identifies whether the remaining gap to `0.6095` was driven entirely by exploits, or if a specific gradient scaling technique from `01d74b0` was identified and preserved.
