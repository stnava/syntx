# BOLD (and ASL) motion correction: the `bold` preset

Measured 2026-10-14 on real data. Scripts and logs: `/Users/stnava/data/handoff/motionbias/` (`bold_bench.py`,
`bold_sched.py`, `asl_check.py`; not committed).

## Question

The rsfMRI and ASL pipelines (antsxfunctional) correct motion through one shared wrapper that used
`pytorch_batched_adaptive` against the mean image: a cheap registration at 8 mm and 4 mm resolution caps, refined only for
frames above an FD gate. For DWI that backend was found biased (docs/MOTION_DWI.md). Is it biased for BOLD?

## Data

SRPBS travelling subjects, `traveling_subjects/` (9 subjects x 12 sites, 411 resting-state runs, TR 2.5 s,
3.3 x 3.3 x 4 mm, 240-250 volumes), with a QC table (`QC/group_bold.tsv`, MRIQC-style: `fd_mean`, `fd_perc`, DVARS, tSNR) from
an independent estimator. Nine runs chosen from the QC table: 3 quiet (QC FD 0.046-0.047 mm), 3 mid (0.122-0.123), 3 high
(0.399-0.411), on several scanners. PTBP, breacher and SOCOM ASL for the ASL checks (see the ASL section).

## Results

Power FD (50 mm radius, mean over frames), our estimate against the QC value:

| QC FD | current adaptive | full precision, narrow seeds | full precision, wide seeds |
|---|---|---|---|
| 0.046 / 0.047 / 0.047 (quiet) | 0.155 / 0.186 / 0.199 | 0.099 / 0.070 / 0.079 | identical |
| 0.122 / 0.123 / 0.123 (mid) | 0.327 / 0.322 / 0.303 | 0.147 / 0.167 / 0.196 | identical |
| 0.399 / 0.402 / 0.411 (high) | 0.593 / 0.595 / 0.572 | 0.523 / 0.542 / 0.518 | identical |

* Against the QC FD over the nine runs: current backend Pearson 0.987, Spearman 0.883, fit 1.09 x QC + 0.153 mm;
  full precision Pearson 0.996, Spearman 0.900, fit 1.25 x QC + 0.021 mm. The current backend carries a constant 0.15 mm
  offset (spurious motion on a still head), as in DWI. The slope of 1.25 is probably a different estimator or radius
  convention in the QC pipeline; not investigated.
* Known motion (smooth drift plus six spikes injected into a quiet run; true Jenkinson FD 0.178 mm): relative-motion error
  of the current backend 0.083 mm (FD estimate 0.212, +19%); narrow seeds 0.041 mm (0.192, +8%); wide seeds 0.042 mm (same FD).
* Narrow seeds equal wide seeds on every run at about 70% of the time. Time per 240-volume run: adaptive 4.8 s, full
  precision 24-33 s.
* Schedule ablation (3 quiet + 3 high runs + the injected case; time per run, injected error, quiet-run FD):

  | schedule | time | injected error | quiet FD |
  |---|---|---|---|
  | `full` | 28.3 s | 0.041 mm | 0.083 |
  | polish at 50% sampling | 21.4 s | 0.043 | 0.086 |
  | polish at 30% | 19.1 s | 0.045 | 0.085 |
  | **no middle L-BFGS stages, polish 50% (`medium`)** | **16.9 s** | **0.045** | **0.084** |
  | same, polish 30% | 14.0 s | 0.047 | 0.096 |
  | same, Adam 40/20 steps | 10.1 s | 0.063 | 0.112 |

  Profile of the full schedule: 22.1 of 23.5 s is the single batched registration call, 15.4 s of it in L-BFGS closure
  evaluations; resampling and bookkeeping are under 1 s. `medium` was chosen as the fastest setting with no accuracy loss.

## The preset

`motion_correction(image, preset="bold")` = `backend="pytorch_batched"`, `seed_mode="narrow"`, `schedule_mode="medium"`
(`syntx.motion_presets.BOLD_MOTION_PRESET`); a conflicting `backend` or option raises. Power FD stays primary for
BOLD (the 0.5 mm scrubbing threshold of Power et al. 2012 is defined for it); Jenkinson FD is returned alongside.

## ASL: same preset, better than the current backend everywhere, still less accurate than BOLD

Five ASL datasets, injected known motion (smooth drift plus spikes; true Jenkinson FD) and real series:

| series (slice thickness, volumes) | true FD | current backend: relative-motion error / FD estimate | `bold` preset: error / FD estimate |
|---|---|---|---|
| PTBP PCASL 64 x 64 x 24, 3.1 x 3.1 x 6 mm (80) | 0.301 | 0.411 mm / 0.558 (+85%) | 0.345 / 0.521 (+73%) |
| breacher sub-9162 follow-up 96 x 96 x 30, 2.5 x 2.5 x 4 mm (40) | 0.360 | 0.603 / 0.773 (+115%) | 0.476 / 0.688 (+91%) |
| breacher sub-9162 initial 96 x 96 x 50, 2.5 x 2.5 x 3 mm (60) | 0.301 | 0.236 / 0.451 (+50%) | 0.093 / 0.346 (+15%) |
| SOCOM Blast-01 128 x 128 x 36, 1.9 x 1.9 x 4 mm (25) | 0.692 | 0.568 / 1.064 (+54%) | 0.193 / 0.807 (+17%) |
| SOCOM Blast-02 (same) | 0.692 | 0.768 / 1.215 (+76%) | 0.318 / 0.840 (+21%) |

Real series (Power FD, current vs preset): PTBP six series 1.180/1.124, 0.645/0.654, 0.397/0.310, 0.234/0.168,
0.804/0.721, 0.274/0.217 mm; breacher 0.902/0.815 and 0.398/0.161; SOCOM 0.719/0.309 and 1.002/0.575. The preset reads
lower on all ten series except one tie (0.645 vs 0.654), by 10-60%.

The preset is better than the current backend in all five truth tests and costs 6-11 s per series instead of about 2 s.
It is least accurate on the thick-slice, few-slice or short series (PTBP 6 mm, breacher follow-up 40 volumes); a role for
slice thickness and coverage is plausible but not tested (five datasets differing in several ways at once). ASL
estimates remain 15-90% above the injected truth, against 8% for BOLD, so ASL motion is less well determined than BOLD
motion by this method.

## Not established
* The 1.25 slope against the QC FD; DVARS and other QC columns were not compared; nine runs in three motion clusters is a
  small sample, so the correlations describe level and ordering, not a precise calibration.
* The effect on downstream rsfMRI results (censoring with `fd_censor_threshold_mm`, connectivity) was not measured: a
  quiet run read 0.15-0.2 mm under the old backend, near typical censoring thresholds, so frames may have been dropped
  spuriously. Not quantified.

## Band-limited FD for censoring (v6.26, `syntx.band_limited_fd`)

A sub-voxel physiological oscillation faster than the sampling (respiration aliased at a long TR, cardiac pulsation at a
short one) is a real frame-to-frame displacement of the image content. On the `breacher` resting series (TR 2.5 and 1.0 s)
a joint group-wise objective aligned the frames slightly better than the `bold` preset while reporting more FD, and
removing the oscillation from the poses before resampling made tissue edges 15 to 20 % worse
(`benchmarks/pilot_bold_resting_joint.py`, `docs/reports/pilot_bold_resting_joint_2026-10-27.html`). The oscillation must
be applied as a correction. What it must not do is count as motion for censoring: a per-step sum counts it twice per cycle
and inflates mean FD three to four fold.

`band_limited_fd(translations_mm, rotations_deg, tr_s, band_hz=None)` removes one band from the pose series and
recomputes Power FD. The band is given (a recorded respiratory trace is the right source) or found from the pose
spectrum: the peak above 0.1 Hz, widened to the contiguous region above a quarter of the hump's height over the white
floor and to at least a quarter of the Nyquist frequency on either side. It is applied only when the band holds at
least 1.5 times the power its width alone would hold; otherwise `band_source` is `'none'` and the series is the plain
FD (white jitter and drift-plus-jump series are untouched, `tests/test_band_limited_fd.py`). Every result of
`motion_correction`, `motion_correct_grouped` and `motion_correct_design` carries `fd_bandlimited` and `fd_band`
(`band_hz`, `band_source`, `alternation_hz`, `alternation_amplitude_mm`, `band_prominence`, `band_power_fraction`), the
TR taken from the 4-D time spacing or the `tr` keyword.

| series | FD raw | FD band-limited | band (Hz) | frames above 0.5 mm raw | band-limited |
|---|---|---|---|---|---|
| breacher sub-6223, TR 2.5 | 0.426 | 0.246 | 0.146 to 0.200 | 27 % | 10 % |
| breacher sub-3315, TR 1.0 | 0.622 | 0.328 | 0.355 to 0.500 | 48 % | 16 % |

The consumers censor on `fd_bandlimited` and keep the full poses as the correction and as nuisance regressors
(`docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md`, section 4). Precedent: Power et al. 2019, Fair et al. 2020, Gratton et al.
2020.
