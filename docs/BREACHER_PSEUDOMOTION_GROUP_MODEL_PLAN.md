# breacher pseudomotion: a group-level model without DICOM or physio (plan, 2026-10-31)

Status: plan. Nothing is implemented. Companion to `docs/BREACHER_PSEUDOMOTION_REVIEW_PROPOSAL.md` (evidence E1–E4,
D1–D4). Constraint: no DICOMs and no PMU logs. Everything below uses the 48 breacher BOLD runs, plus ExpArt, FPA and
SOCOM as controls.

## What is known and what is open

Known (D2, E3): a breathing-rate displacement of the whole image content, mostly along z, plus pitch and PE, at
0.18–0.25 Hz with TR 2.0 s. Seen in breacher only among the four cohorts. Not the slice interleave, not an
intensity artifact, not a field-induced shift.

Open: which of three mechanisms produces it.

| | mechanism | is the head moving relative to the body? |
|---|---|---|
| H1 | PACE prospective correction moves the FOV once per volume, k = 1 or 2 volumes late, from its own estimate (which a respiratory PE field shift corrupts) | no |
| H2 | the bed/coil assembly slides or flexes with breathing and carries the head | no (moves relative to the gradients) |
| H3 | the head nods with breathing, pivoting where the occiput rests on the coil | yes |

The effect is consistent across the group. That is what makes it identifiable: one scanner, one protocol and one
coil give all runs the same mechanism, so its signatures can be pooled over 48 runs instead of decided per run.

## Three signatures that separate H1, H2 and H3 (all from the images we have)

### S1. Intra-volume clock: volume-locked or continuous? (separates H1 from H2/H3)

Odd-index slices are acquired at 0 to 0.94 s and even-index slices at 0.99 to 1.93 s (interleaved; from the 8 real EPI
sidecars, identical within 2.5 ms). Each frame's two slice sets are registered separately (as in E3, extended to all
runs). Write d_t = pose(set_even) - pose(set_odd).

- H1: PACE changes the FOV between volumes, so both sets of one volume share one FOV. d_t has no breathing-band
  component beyond estimation noise.
- H2/H3: the displacement is continuous in time. The two sets sample it about 1 s apart, so
  d_t ≈ 2 sin(π f Δt)·A_t is coherent with the band component of the full-volume pose and leads it by about 90°.
  At f = 0.3 Hz and Δt = 1 s, the gain is 1.6.

Statistic: the per-run coherence and complex gain of d_t against the full-volume band component, pooled over runs
as a group mean with a bootstrap CI over runs. The noise floor of d_t comes from the yaw and roll axes, which carry
no band power.

### S2. Pivot of the breathing-locked rotation (separates H2 from H3)

For the band-limited pose component, a rotation θ_t about a pivot P with no other translation gives
t_t = θ_t × (c − P), where c is the brain centre. The pivot is estimated per run by least squares from the
band-limited (t, θ) pairs, then expressed relative to the head outline, which is measured on the EPI mean (head mask
plus the posterior-inferior skull surface).

- H3: P lies at the occiput–coil contact, posterior and inferior to c.
- H2 (pure bed slide): no rotation is needed; translation along the bore with pitch ≈ 0. Pitch is clearly present
  (MAD 0.05 to 0.07 mm), so the pivot estimate is meaningful.
- H2 (bed flexing as a cantilever): P lies far outside the head, toward the bed support, at the same scanner location
  in every run.
- H1: P follows from where the scanner centres its rotations. It is not anatomical, so it varies relative to the head
  across runs.

Order-of-magnitude check from the data already measured: θ ≈ 0.07 mm / 50 mm ≈ 1.4 mrad, and t_sl ≈ 0.1 to 0.19 mm,
give a lever arm of 70 to 135 mm. That is about the distance from the brain centre to the occiput. So H3 is not
excluded on magnitude. This is a prediction to test, not a result.

### S3. Event-locked response to isolated jerks (separates H1 from H2/H3 a second way)

Pool every isolated large pose step (> 5 MAD, no other such step within ±5 TR) across all breacher runs. Average the
pose trajectory from -5 to +10 TR around the event, per axis.

- H1: a true jerk at frame t is seen at t, then the scanner removes it at t + k. The average is an impulse that
  returns to baseline after exactly k TR, on every axis at once.
- H2/H3: no feedback. The average is a step, or a decay over several TR.

Control: the same analysis on ExpArt, FPA and SOCOM (no PACE), which should give steps.

### Validation before any real-data reading (required)

For each run, build three synthetic series from its own mean image, noise level and slice timing, with a
breathing-rate trajectory drawn from the run's band amplitude:

- **H1:** a per-volume FOV shift computed from a lag-k estimate. The estimate includes a PE field shift.
- **H2:** a continuous bed translation, sampled at each slice's acquisition time.
- **H3:** a continuous occiput-pivot nod, sampled at each slice's acquisition time.

The S1 to S3 decision rules are fixed in advance and must classify at least 90 % of the synthetic runs correctly
before they are applied to real data. This is the step the D4 coil test skipped: a positive control in real data
then failed.

## Modelling it out: a group-shared component, removed from FD only

Whatever the mechanism, the applied transforms keep the displacement, because the image content moved (pilot
2026-10-27; D2). The model changes only what FD counts. That is a policy, and its form depends on the S1 to S3 outcome:

**M1, if H1 (PACE).** Model the observed pose as x_t = m_t - α·m̂_{t-k} + e_t, with a group-shared α and k. Fit them
from S3's impulse shape and the pooled spectrum, using the control cohorts' motion spectra (aliased to TR 2.0 s) as
the prior on m. Recover the head trajectory m by the inverse filter 1/(1 - α z^-k), regularized near α = 1, and
compute FD from m. Note that under a perfect lag-1 PACE, FD_true = |x_t|, not |Δx_t|.

**M2, if H2 (bed), or if H3 is rejected as a censoring target.** Use a group-shared six-DOF direction u for the
breathing-locked component, in scanner coordinates:
1. Estimate u leave-one-run-out, from pooled PCA of band-limited pose steps in the other runs, so that no run
   contributes to its own direction.
2. In each run, project the band-limited pose onto u, and compare that band power with the mean band power of the
   five orthogonal directions (an anisotropy ratio).
3. Remove from the FD series only the excess above that ratio (Wiener form).

Isotropic white jitter has a ratio of about 1, so nothing is removed. This does not need a spectral peak, which is
the reason `band_limited_fd` leaves 5 of 6 runs untouched (E2). The breathing band per run comes from the
u-projection spectrum, not from a peak test.

**If H3 (real nod).** No change to FD. What remains is the censoring threshold, which is decided by the G5
downstream gate (test-retest connectivity).

## Gates for any FD model

1. **Isotropic white-jitter null:** at least 95 % of FD retained. (`sagittal_nyquist` retained 71 %.)
2. **Synthetic recovery:** on the H1 and H2 generators, the recovered head-motion FD is within 10 % of the true FD.
   On the H3 generator, report what the model removes. This is not a pass/fail gate, because removing real motion is
   a policy choice.
3. **Applied transforms unchanged:** the resampled series is bit-identical.
4. **Downstream:** the G5 test-retest connectivity check is run after the TR fix and a re-run of the 39 affected runs.

## Work packages (none started)

| WP | content | repo | cost |
|---|---|---|---|
| G0 | slice timing for the 39 runs without SliceTiming: assume the 8-sidecar protocol and verify per run with `syntx.acquisition_axes.interleave_scores`; runs that disagree are excluded | syntx benchmarks | small |
| G1 | the three synthetic generators (H1, H2, H3), slice-timed, built from each run's mean and noise | syntx benchmarks | 1 day |
| G2 | S1, S2, S3 estimators plus the pre-set decision rules; validate on G1 (≥ 90 %) | syntx benchmarks | 1 day |
| G3 | run G2 on all 48 breacher runs and 10 control runs (half-set registration about 75 s per run, ≈ 1.5 h total); report with group CIs in `docs/reports/` | syntx | compute |
| G4 | M1 or M2, only for the supported mechanism; gates 1 to 3 | syntx (`band_limited_fd` sibling, reported as a separate FD series, not a default) | 1 day |
| G5 | test-retest connectivity gate | antsxfunctional / antsxmm | after re-run |

## What this cannot decide

- If S1 shows a continuous displacement and S2 places the pivot at the occiput, the data support a real nod (H3),
  and the plan says so rather than modelling it away.
- If S1 is ambiguous (the coherence CI includes both the H1 and H2/H3 predictions), H1 versus H2/H3 stays open.
- The k = 1 versus k = 2 lag in M1 rests on S3 alone.
- Without the DICOMs, "PACE on" is an inference from a signature, not a fact from the protocol.
