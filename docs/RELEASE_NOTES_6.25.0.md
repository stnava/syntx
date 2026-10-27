# syntx 6.25.0

One optimisation of head motion and acquisition effects over the whole series: `motion_correct_design(..., engine="joint")`.
Evidence: known-motion phantoms (DWI, ASL, slow flow) and the real `motion_cohort` archive (4 ASL, 4 DWI series;
`docs/reports/motion_cohort_gate_2026-10-27_default.html`). Design: `docs/MOTION_JOINT_OPTIMIZATION_PLAN.md`; the review it came
from: `docs/MOTION_COMPLETE_MODEL_REVIEW.md`.

## What changed

- **`engine="joint"` (`motion_design._joint`).** H0: every level to its own mean, the level means linked by an explicit
  rigid registration (the DW-to-b0 link is a measurement). H1: one batched pass over the ORIGINAL frames against the
  corrected mean of their own level in the anchor space, started from the H0 poses, with every design column of the
  timeline in the optimiser (level offsets, l = 2 terms per shell, eddy affine per shell); an opt-in Charbonnier prior on
  the second differences of the head trajectory with its knee at the registration precision floor (0.05 mm; off by
  default: its absolute weight competes with the image term's curvature and displaced a featureless no-motion phantom
  by 0.1 mm at 0.02). Phantoms (true mean FD 0.156 mm): `unified` reported 0.301, `joint` 0.220; FD error DWI
  0.438 -> 0.208, ASL 0.089 -> 0.059, slow flow 0.133 -> 0.068 mm. Real data: FD falls 1-7 % on ASL and 0-16 % on DWI with identical alignment (temporal SD of the
  corrected series), the phase-encode step on the two FPA 3-shell series down 0.14-0.16 mm, the B0-immune readout step
  within 0.025 mm of `unified` in either direction, every frame converged; 1.2-1.5x the time.
- **Solver (`motion_batched`).** `init_pose` (per-frame starting pose) and `schedule_mode="refine"` (L-BFGS then Newton
  at full resolution; the coarse Adam stages walk a good pose away). Per-reference intensity range with per-frame
  references: frames of a dark contrast (high b) keep the full histogram range of their own reference instead of the
  low bins of the brightest one (this alone took the DWI phantom's FD error from 1.56 to 0.21 mm). `frame_levels`:
  per-frame point weights from each level's own reference (`PointSampler.weights_of`), off by default (neutral on the
  phantoms and the cohort). Odd-only phase-encode projection of the encoding gauge: a translation along phase encode
  in the span of the eddy covariates (odd in `g`) is the applied eddy term's; the even part is contrast and stays out
  of the pose (the previous unconditional projection sent it to motion).
- **`joint_design_columns`**: the design matrix of the timeline (level indicators; l = 2 blocks per shell with the
  6-frames-per-column gate; eddy covariates per shell).
- **`benchmarks/motion_cohort_gate.py`**: the real-data gate (step MAD per acquisition DOF, design-locked alternation,
  temporal SD, readout-floor tolerance) with an HTML report.
- Downstream: antsxdwi 0.31, antsxfunctional 0.20 and antsxslowflow 2.10 default to `engine="joint"`; antsxslowflow's
  oscillation GLM orthogonalises motion nuisance columns to the harmonic columns (`motion_orthogonalize=True`).

## Not changed

- `unified` and `dispatch` are unchanged and remain available (`engine=`).
- BOLD (one level): identical to the `bold` preset; the respiratory field term (plan section 8) and the
  motion-specific metric (section 9) are not implemented.
- `tests/test_motion_overlapping_window.py::TestOverlappingWindowBOLD::test_bold_nyquist_sagittal_ripple_debiased_while_nodding_untouched`
  fails on this and the previous release alike (pre-existing).
