# breacher resting BOLD: pseudomotion review and proposal (2026-10-31)

Status: proposal. Nothing below is implemented. Question asked: does the breacher cohort hold misattributed pseudomotion,
and should `syntx.motion_correction` use a different default for it? Input: the `motx.zip` summary (an external
agent's `motx.py` + prompt), the ISMRM 2027 abstract, and new measurements on six breacher runs (scripts listed at the
end; they live in the session scratchpad and are committed only under WP0).

## Answer

No new `syntx` default yet. The high-frequency pose fluctuation that inflates FD appears in every axis, including the
B0-immune readout. It also persists when the odd and even slice sets are registered separately, and the two sets move
together. That rules out the interleaved slice order and a slice-axis field effect as its source. It does **not**
establish head motion: anything that displaces the whole image each TR (the scanner's prospective PACE correction, a
respiratory field shear absorbed as pitch, an intensity-driven registration bias) gives the same result. Corrected
2026-10-31 after review; see "Deep dive" below. Turning on `debias_sagittal_nyquist` would remove 29 % of a true
white-noise jitter and 80 % of a true near-Nyquist nod, so it is not the remedy whatever the cause.

What is actually wrong is upstream and in the record:

1. 39 of 48 breacher BOLD runs carry a field-map sidecar (`SequenceName *fm2d2` / `*fm2d2r`, `RepetitionTime 1`);
   the NIfTI header says 2.0 s. antsxmm trusts the sidecar, so rsfMRI bandpass, ALFF/fALFF and the recorded TR are
   wrong on those runs. Motion and censoring are not affected (below).
2. sub-3315 has been described as "TR 1.0 s" in the pilot, the censoring plan, two docstrings, project memory and the
   ISMRM abstract. It is a 2.0 s run with one of these sidecars.
3. Why breacher still censors 11 to 33 % of frames: the band-limited FD notches only a spectral *peak*. The breacher
   fluctuation is broadband, so on 5 of 6 runs no band is notched and censoring falls back to full FD. That is the
   function behaving as designed. Whether broadband real jitter should be censored is a censoring-policy question for
   the downstream gate (G5), not a motion-solver default.

## Evidence

### E1. Sidecar audit (all 48 breacher BOLD runs, header read directly)

| sidecar `SequenceName` | sidecar TR (s) | header TR (s) | runs | SliceTiming |
|---|---|---|---|---|
| `*fm2d2` (gre_field_mapping) | 1 | 2.0 | 35 | absent |
| `*fm2d2r` | 1 | 2.0 | 4 | absent |
| `epfid2d1_64` / `*epfid2d1_64` (ep2d_pace) | 2 | 2.0 | 8 | present |
| `*epfid2d1_64` | 2.5 | 2.5 | 1 (sub-6223) | present |

All 2.0 s runs are 64×64×36×300 (one run 299), 3.0×3.0×3.66 mm. The real EPI sidecars give interleaved timing: the
odd-index slices are acquired at 0 to 0.94 s, then the even-index slices at 0.99 to 1.93 s. The field-map
sidecars' `PhaseEncodingDirection` (`j-`) matches the EPI ones, so the PE debias direction is right by luck.

Where TR is used:
- `antsxmm/modalities/bridges/functional.py:298-311` (rsfMRI) and `:612-626` (TaskfMRI): the sidecar
  `RepetitionTime` wins over the header with no consistency check, so TR = 1.0 is passed to `run_rsfmri`.
  `bandpass_filter(tr=1.0, 0.03-0.08 Hz)` then passes 0.015 to 0.04 Hz of real signal; ALFF/fALFF bands shift the same
  way.
- `antsxfunctional/rsfmri/pipeline.py:231`: `run_rsfmri` does not pass `tr` to `syntx_motion_correction`, so
  `attach_band_limited_fd` falls back to the header (2.0 s). Censoring therefore uses the right TR. In any case the
  band-limited FD *values* do not depend on TR (the band is chosen in cycles per frame); only the Hz labels change. I
  verified this on all six probed runs, at TR 1 and TR 2.

### E2. Production motion call on six runs (`breacher_moco_probe.py`)

`syntx_motion_correction(..., engine="joint", debias_pe_pseudomotion=True)`, syntx 6.33.0, default compute device,
T = 300 (sub-0102: 299). FD is the result's `fd`, after the PE debias. The step MAD is the median |Δ| per acquisition
axis in mm, with rotations at 50 mm.

| run | sidecar | FD mean (mm) | > 0.5 mm (%) | band notched? | FD_bl mean / > 0.5 (%) | step MAD RO / PE / SL / pitch (mm) | step ac1 RO / SL / pitch |
|---|---|---|---|---|---|---|---|
| sub-0102 | fm2d2 | 0.511 | 33.4 | no (prominence 1.39) | = FD | 0.052 / 0.021 / 0.192 / 0.055 | -0.68 / -0.70 / -0.61 |
| sub-1239 | epfid | 0.300 | 13.4 | no (0.68) | = FD | 0.033 / 0.015 / 0.097 / 0.037 | -0.75 / -0.51 / -0.49 |
| sub-3315 | fm2d2 | 0.546 | 38.1 | yes, 0.18-0.25 Hz (1.94) | 0.284 / 11.4 | 0.028 / 0.026 / 0.104 / 0.072 | -0.62 / -0.84 / -0.66 |
| sub-1268 | fm2d2r | 0.271 | 11.4 | no (0.74) | = FD | 0.027 / 0.017 / 0.079 / 0.046 | -0.76 / -0.54 / -0.66 |
| sub-2026 | epfid | 0.246 | 3.3 | no (0.28) | = FD | 0.014 / 0.014 / 0.130 / 0.023 | -0.38 / -0.63 / -0.26 |
| sub-0509 | fm2d2 | 0.095 | 0.0 | no (0.08) | = FD | 0.004 / 0.012 / 0.025 / 0.014 | -0.31 / -0.41 / -0.00 |

For uncorrelated frame-to-frame noise, the step lag-1 autocorrelation (ac1) is -0.5. Values below -0.5 mean the
pose alternates. On four of six runs the B0-immune readout alternates as strongly as the slice axis does.

sub-3315 reproduces the abstract's BOLD row exactly (FD 0.55 → FD_bl 0.28 mm, censored 38 → 11 %). The notched band
is 0.18 to 0.25 Hz. The abstract's "0.48 Hz" nod is that band labelled at TR 1.0.

Why most runs are not notched: with drift excluded from the denominator, the prominence is still only 1.0 to 1.6.
Above 0.1 Hz the pose spectrum is broad, not a line. The top band holds 53 to 96 % of the step variance, but any
white or blue jitter puts most of its step variance near Nyquist, so that share is not evidence of an oscillation
(`prominence_check.py`). An earlier hypothesis in this session, that slow drift in the denominator suppresses the
prominence, is refuted by that same check.

### E3. Odd and even slice sets registered separately (`halfset_probe.py`)

Each run is split into its two interleaved slice sets (z spacing 7.32 mm, origins exact), and each set is registered
with the same joint engine without the PE debias. If the alternation came from an intensity comb between the sets
(acquired 1 s apart), it could not exist inside one set. If it is a displacement of the head or image, it persists
in both sets and their step series correlate.

| run | t_sl step ac1 full / set0 / set1 | t_sl step MAD full / set0 / set1 (mm) | step corr set0 vs set1: t_sl / pitch / t_ro |
|---|---|---|---|
| sub-0102 | -0.70 / -0.69 / -0.68 | 0.19 / 0.33 / 0.29 | 0.39 / -0.07 / 0.69 |
| sub-3315 | -0.84 / -0.80 / -0.85 | 0.10 / 0.13 / 0.14 | 0.83 / 0.84 / 0.70 |
| sub-1268 | -0.54 / -0.49 / -0.65 | 0.08 / 0.10 / 0.10 | 0.62 / 0.71 / 0.75 |

The alternation persists in each set at full strength. The two independent sets agree (correlations 0.6 to 0.85 on
two runs), even though each half-volume estimate is noisier than the full one. This shows a cause common to both
slice sets (a whole-image displacement or a common intensity effect), not that the head moved. sub-0102 agrees on t_sl and t_ro but
not on pitch. Pitch from a 7.32 mm-spaced half-volume is poorly conditioned, so its independent estimation noise is
large there. This matches the 2026-10-27 pilot: removing the alternation from the applied transforms made edges
15 to 20 % worse.

### E4. Null gate for `sagittal_nyquist` (`null_gate.py`; 20 seeds, T = 300)

FD retained after the debias, PE-only / PE + `sagittal_nyquist`:

| true trajectory | PE only | + sagittal_nyquist |
|---|---|---|
| isotropic white jitter, 0.05 or 0.12 mm per axis | 99.8 % | 70.9 % |
| real nod at 0.4 cycles/TR, 0.15 mm (slice + pitch) + 0.02 mm white | 99.8 % | 19.5 % |

The option cannot separate a real aliased nod from pseudomotion: their pose signatures are identical. On sub-3315,
the one run where the band is notched, it gives a higher FD than the band-limited FD (0.310 vs 0.284 mm).

## Deep dive (2026-10-31): what moves the image?

The user's position is that the breacher alternation is not head motion. These tests were run to find its cause. Unless
stated otherwise, they cover sub-3315, sub-0102 and sub-1268 (all TR 2.0 s, 300 frames). Scripts are in the scratchpad
(listed at the end).

### D1. Where the alternation lives in a free affine (`affine_band_probe.py`; ANTs 12-DOF per frame, Mattes MI)

Every entry of the per-frame affine, written in the acquisition frame, carries the breathing band: 0.12 to 0.25 Hz
holds 72 to 99 % of each entry's step variance. The largest entries are t_pe, t_sl and the PE-SL block. In the
PE-SL block the two off-diagonal steps are anti-correlated (-0.95 and -0.79), so the block behaves mostly as a
rotation (pitch) rather than as a field shear along PE (antisymmetric part step MAD 0.058 / 0.034 mm; symmetric part
0.016 / 0.016 mm). A respiratory B0 change can only move the PE row, but the RO and SL rows alternate as well
(t_ro ac1 -0.61 / -0.69). **A PE field effect explains t_pe, not t_sl, pitch or t_ro.**

### D2. Shift or intensity change? (`shift_vs_tilt.py`, registration-free, 5 runs)

Each frame's difference from the mean is regressed on shift bases (-dI/dx_i for each axis) and intensity bases
(global gain, z-tilt, even/odd slice comb, PE tilt). The z-shift coefficient reproduces the registration's t_sl
(step correlation 0.98 to 1.00). Adding the intensity bases does not change it (step MAD 0.124 → 0.120 mm on
sub-3315). The intensity terms are small and absorb little variance: R² of frame differences 0.107 → 0.137 on
sub-3315. The slice-mean S-I intensity slope correlates with t_sl at 0.97 to 0.99, but that is what a z-shift of a
brain with an intensity profile produces inside a fixed mask. **The image content is displaced along z, edges
included. This is not an intensity tilt that the registration misreads.**

### D3. Field-induced slice shift is too small (computation, sidecar values)

A frequency offset Δf shifts the excited slice by Δf / RF bandwidth × thickness, and shifts the EPI image along PE by
Δf / 85.2 Hz × 3 mm (BandwidthPerPixelPhaseEncode = 85.2 Hz). For any plausible RF bandwidth (1 to 3 kHz, not in
the sidecar), the slice shift is 3 to 8 % of the PE shift. The observed t_sl step MAD is 0.6 to 1.6 times the
observed t_pe. **A field offset cannot produce the t_sl alternation.**

### D4. Does the anatomy move relative to the receive-coil shading? (`coil_ratio_test.py`) — inconclusive

With I = S·A, where S is the coil shading (N4 bias field of the mean), a head moving inside the coil adds a term
I·∂z ln S whose coefficient equals the shift. A shift of the whole grid adds no such term. The test was calibrated
on synthetic series built from each run's own mean, with a respiratory global-gain confound. The ratio of
coil-term coefficient to shift coefficient was 1.03 to 1.09 for head-in-coil and 0.21 to 0.27 for a grid shift
(interpolation leakage). Breacher real data: 0.05 to 0.25 on five runs (95 % CIs exclude 1), which reads as a grid
shift. **But the positive control fails.** ExpArt sub-EAS024 (Prisma, no PACE) reads 0.55 against its own
calibration of 0.51 (grid) / 1.30 (head-in-coil); EAS107 0.71, FPA BLAST071 0.59, SOCOM Blast-01 0.61 (not
individually calibrated). No cohort uses Prescan Normalize, so coil shading is present everywhere. Either the N4
field is not the receive profile closely enough for this test, or z-motion moves with the coil in every cohort. The
test does not separate breacher from the others and is not evidence for either position.

### What is established, and what is not

Established: the alternation is a displacement of the whole image content along z (D2). It is not created by the
interleaved slice order (E3), not an intensity change misread by the registration (D2), and not a field-induced slice
or PE shift (D1, D3). It has respiratory periodicity (0.18 to 0.25 Hz at TR 2.0 s, with slowly decaying alternating
autocorrelation; sub-3315 t_sl steps: -0.83, +0.57, -0.39, +0.28). It is specific to breacher among these four cohorts:
Biograph mMR, HeadNeck_MRPET coil, `ep2d_pace_physio` sequence, TR 2.0 s. Pose excursions are small (≤ 1.5 mm in
slice and ≤ 0.31 mm in readout over 10 min).

Not established: whether the head moved relative to the scanner. Three candidates remain, and the images alone
cannot separate them:

1. **PACE prospective correction active.** The scanner moves the FOV every TR from its own motion estimate, one TR
   late. A one-TR-late correction multiplies a near-Nyquist component by |1 - e^(-i2πf·TR)| = 2 sin(πf·TR): 1.9 at
   0.4 cycles/TR. It would also turn any error in the scanner's estimate (for example a respiratory PE shift read as
   motion) into real FOV moves on all axes. The FOV moves are real image displacements, and the head did not move.
   The small slow drift fits PACE tracking it away, but it also fits a still subject.
2. **The patient table or coil assembly moving with breathing** (the mMR bed is cantilevered). This is a real
   displacement of the head relative to the gradients, but not motion of the head relative to the body. It would be
   phase-locked to breathing with no one-TR lag.
3. **Real respiratory head motion.** Not excluded by any test here.

Data that would decide it (not on this machine; barncreek):
- **The DICOMs.** First, the PACE setting in the series protocol (CSA header). Second, the per-volume
  ImagePositionPatient / ImageOrientationPatient: when PACE moves the FOV these change across volumes, and dcm2niix
  keeps only the first.
- **The PMU respiratory logs** (`ep2d_pace_physio` records physio). The phase lag between the respiration trace and
  the displacement is 0 for table/head motion and one TR for a PACE feedback.

### Consequence for the defaults, whichever candidate holds

If the cause is (1) or (2), FD should not count this component, but the correction must keep it, because the image
content did move (pilot 2026-10-27: removing it from the applied transforms worsened edges). That is the split
`band_limited_fd` already makes when it detects the band. The open defect is that its detector requires a spectral
peak, and on 5 of 6 runs the breathing component is broadband. A breathing-locked estimate (from the PMU trace if it
exists, else from the PE-row B0 signal, which is a field effect and therefore a breathing reference independent of
motion) would make the notch independent of peak shape. That is a proposal, not a measured result.

## Claims in the motx summary: accepted, rejected

| claim | verdict | basis |
|---|---|---|
| 39/48 runs carry field-map sidecars with TR 1; antsxmm trusts them | accepted | E1 |
| `run_rsfmri` does not pass `tr` to the motion call | accepted, harmless today | header fallback is correct; FD_bl is TR-invariant (E1) |
| `sagittal_nyquist`'s ac1 gate fires on real white jitter | accepted | E4 |
| breacher t_sl / pitch alternation is respiratory ΔB0 pseudomotion along the slice axis | rejected | a field offset shifts the slice profile by Δf / RF bandwidth × thickness, about 0.01 mm for Δf ≤ 5 Hz and ≥ 1 kHz bandwidth (assumed, not measured); readout alternates too (E2); the alternation persists within one slice set (E3) |
| the interleaved odd/even comb creates the t_sl alternation | rejected | E3 |
| leaving t_sl in `fwdtransforms` is a resampling bug | rejected | E3; 2026-10-27 pilot (removing it from the applied transforms worsened edges) |
| a gate on EPI-vs-immune anisotropy would identify pseudomotion | rejected as an identifier | thick, gapped slices make through-slice estimation intrinsically noisier, and a real respiratory nod is sagittal-plane anisotropic; anisotropy appears with both causes |
| even/odd slice intensity steps anti-correlate in raw pixels (r = -0.44) | not re-measured | consistent with real through-plane motion in an interleaved acquisition (spin history), so not evidence of pseudomotion |

## Work packages

**WP0 — record the evidence (syntx).** Commit the four probes as `benchmarks/breacher_moco_probe.py`,
`benchmarks/breacher_halfset_probe.py`, `benchmarks/sagittal_nyquist_null_gate.py` and
`benchmarks/band_prominence_check.py`, with a dated report in `docs/reports/`. Verification: the scripts reproduce
the E2 to E4 tables.

**WP1 — TR guard (antsxmm; required).** At both TR sites (`functional.py` rsfMRI and TaskfMRI), compare the sidecar
`RepetitionTime` with header `pixdim4`. Also check that the sidecar describes a functional series: no `fm2d`
`SequenceName`, no `BidsGuess[0] == "fmap"`. On any mismatch above 1 %, warn and use the header TR, unless the
caller passes `tr=` explicitly. Record `tr_source` and both values in provenance either way. (Decided: warn and use the header, not raise.) Tests: a
synthetic BIDS pair with a field-map sidecar raises; `tr_policy="header"` uses 2.0; a consistent pair is unchanged.
`python -m antsxqc.lint .` must report 0 findings.

**WP2 — plumb `tr` (antsxfunctional; small).** Pass `tr=tr` from `run_rsfmri` into `syntx_motion_correction`, so the
band labels and provenance use the same TR as the bandpass. FD values do not change (E1). Test: the provenance
`fd_band.tr_s` equals the `run_rsfmri` TR.

**WP3 — re-run breacher rsfMRI (operations; after WP1 and WP2).** Re-run the 39 affected runs. Expect changes in
bandpass-derived measures (connectivity, ALFF/fALFF), not in FD or censoring. Do not touch running batch jobs.
Verification: the wide-table `rsf_tr` equals 2.0 on all 47 runs and 2.5 on sub-6223; FD columns are identical before
and after.

**WP4 — correct the record (syntx docs, antsxfunctional docstring, memory, abstract).**
- `docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md`: sub-3315 is TR 2.0 s.
- `syntx.motion.band_limited_fd` docstring and `antsxfunctional.diagnostics.fd_for_censoring` docstring:
  the breacher examples are unchanged in value; fix any TR or Hz labels.
- `docs/abstracts/ISMRM_2027_motion_solver.tex`: "TR 1.0 s" → "TR 2.0 s" (Datasets, Results, both table rows,
  Figure 0 caption "1 s BOLD", Methods §3 "1 s TR BOLD"); "0.48 Hz" → the band at TR 2.0 (0.18 to 0.25 Hz; peak
  0.24 Hz). The FD and censoring numbers stand. The §4 rationale ("differencing a steady ±0.20 mm breathing wave every
  1 s") must be rewritten for 2 s. Regenerate the table rows with `benchmarks/ismrm_table.py` after relabelling the
  case in `benchmarks/ismrm_cases.py`; never hand-edit them.
- Project memory `breacher-fd-alternation-is-real`: TR correction.

**WP5 — `sagittal_nyquist` (syntx docs only).** Keep it off. Add the E4 null-gate numbers to its docstring and mark it
deprecated for censoring use. Do not redesign its gate. E3 says the target is real motion, so no gate can make it
correct.

**WP6 — split-half precision floor (syntx; measurement, optional).** E3 gives a non-circular way to split
high-frequency pose variance into a part shared by the two slice sets (displacement) and a part that is independent
(estimation noise). Run it on all 48 runs and report per run: the shared and independent step variance per axis, and
the FD that would remain if the independent part were removed. This is evidence for the G5 decision, not a new default.
Cost: two extra registrations per run (about 25 s each on this machine).

**WP7 — censoring policy (decision; blocked on G5).** The open question is policy, not physics. Should broadband,
real, sub-voxel high-frequency displacement be censored at 0.5 mm? The options are: (a) the current rule; (b) FD with
WP6's independent part removed; (c) a different threshold from `antsxqc.metrics.FD_SCRUBBING_THRESHOLD_MM`. The
existing plan's G5 gate (test-retest connectivity, censoring keeps ≥ 90 % of frames) decides. G5 has not been run, and
it should run after WP3, because connectivity today is computed with the wrong bandpass on 39 runs.

## Decisions for you

1. WP1 behaviour on a mismatch: **decided 2026-10-31: warn and take the header TR** (recorded in provenance as
   `tr_source="nifti_header_pixdim4 (sidecar mismatch)"` with both values).
2. Repair the 39 sidecars in the BIDS tree: first decided yes, then **withdrawn** ("forget about that"). Finding kept for
   the record: antsxbids did not write these JSONs (all 48 bold JSONs dated 2026-09-30 over 2018 NIfTIs, no
   `SourceDICOMDirectory` key, which antsxbids always writes). A separate backfill step most likely matched "RESTING"
   in the field-map series description `gre_field_mapping_RESTING_v2` and kept the last match.
3. Before any censoring decision: **decided: deeper dive on the cause of the pseudomotion**. The user's position is
   that it is not head motion (see "Deep dive").

## Limits of this review

Six runs, one session each; E3 on three runs. One device and one solver version (6.33.0). The cohort-level medians in
the motx summary (tSNR, DVARS, 12.9 % high-motion fraction) come from `antsxmm_clean_v202610`, which is not on this
machine, and were not re-measured. The ≈ 0.01 mm slice-select estimate uses assumed Δf and RF bandwidth, not
measured values.

## Probe scripts (session scratchpad)

`/private/tmp/claude-501/-Users-stnava-data-repos-syntx/bd7342f1-d24a-46cc-a87f-cc8030bd6349/scratchpad/scripts/`:
`breacher_moco_probe.py`, `halfset_probe.py`, `null_gate.py`, `prominence_check.py`, `pose_signatures.py`,
`affine_band_probe.py`, `shift_vs_tilt.py`, `coil_frame_calibration.py`, `coil_ratio_test.py`. Their outputs
(`probe.json`, `halfset.json`, `affine.json`) are in the parent directory.

## Implemented 2026-10-31 (antsxmm, uncommitted)

WP1, the user's decision version: `antsxmm/modalities/bridges/functional.py` `_resolve_bold_tr` (rsfMRI and TaskfMRI).
A sidecar RepetitionTime that differs from header pixdim4 by more than 1 % warns and the header TR is used; both
values and `tr_mismatch` go to provenance. TaskfMRI also gets `_sidecar_with_header_tr`, because
`antsxfunctional.task.run_task_fmri` re-reads RepetitionTime from each sidecar path; a mismatched sidecar is replaced
by a corrected copy in the output directory. Tests: `tests/test_bold_tr_resolution.py` (6). Fixtures:
`tests/pe_helpers.write_nifti` now writes header TR 2.0 for 4-D images, matching its sidecars, and the
`test_rsfmri_dispatch_success` header was set to its sidecar's 2.5 s. Suite: the same 4 failures + 1 collection error
as before the change (`test_device_cross_repo` x3, `test_modality_registry_contract`, `test_modalities` import);
`antsxqc.lint` 0 findings. The BIDS sidecars are not modified (the user withdrew the repair).
