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
(0.399-0.411), on several scanners. PTBP PCASL for the ASL check (see the ASL section).

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

## ASL: checked on one dataset only; not solved there

Data: PTBP pediatric PCASL, 64 x 64 x 24 at 3.1 x 3.1 x **6.0** mm (3.4 mm in-plane for the first series), 80 volumes
(`Data/PTBP/images/PEDSxxx/<date>/PCASL/`). Other ASL in the repositories has different grids and was NOT tested: breacher
sub-9162 96 x 96 x 30 at 2.5 x 2.5 x 4.0 mm (40 volumes) and 96 x 96 x 50 at 2.5 x 2.5 x 3.0 mm (60 volumes); SOCOM
128 x 128 x 36 at 1.9 x 1.9 x 4.0 mm (25 volumes).

On six real PTBP PCASL series the `bold` preset reads lower Power FD than the current backend (0.168 vs 0.234, 0.217 vs
0.274, 0.310 vs 0.397, 0.721 vs 0.804 mm; equal within 5% on the two noisiest series, 0.65 and 1.2 mm). With injected known
motion on one series (true Jenkinson FD 0.301 mm) both are far from the truth: relative-motion error 0.411 mm (current)
and 0.345 mm (preset), FD estimates 0.558 and 0.521 mm, about ten times the BOLD error. Candidate causes, none tested: the
thick (6 mm) slices of this particular acquisition, low-SNR label/control images, the label/control alternation. Whether
the result holds for the thinner-slice ASL above is unknown. ASL therefore stays on the current setting in antsxfunctional
until it is investigated.

## Not established
* The 1.25 slope against the QC FD; DVARS and other QC columns were not compared; nine runs in three motion clusters is a
  small sample, so the correlations describe level and ordering, not a precise calibration.
* The effect on downstream rsfMRI results (censoring with `fd_censor_threshold_mm`, connectivity) was not measured: a
  quiet run read 0.15-0.2 mm under the old backend, near typical censoring thresholds, so frames may have been dropped
  spuriously. Not quantified.
