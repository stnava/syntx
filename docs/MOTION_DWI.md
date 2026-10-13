# DWI motion correction: bias in the generic estimator and the `dwi` preset

Measured 2026-10-13/14 on real data. Scripts and logs: `/Users/stnava/data/handoff/motionbias/` (not committed).

## Problem

On a head that does not move, the DWI pipeline reported 1-1.7 mm of motion per diffusion volume and lowered
white-matter FA by about 19%. Test data with known truth: the multicenter travelling-subject dataset (Tong, He et
al.; CC BY 4.0), whose DWI the authors had already corrected with topup and eddy (1.5 mm, 96 volumes, b=0/1000/2000/
3000), so the true residual motion is about zero.

Baseline (pipeline default `pytorch_batched_adaptive`, each DW volume against the mean of the DW volumes, grouped
design, Power FD per group):

| case | b0 FD | DW FD mean | DW FD max | DW translation | DW rotation |
|---|---|---|---|---|---|
| traveler c01r1 (still) | 0.23 mm | 5.36 mm | 15.9 mm | 1.51 mm | 0.75 deg |
| traveler c05r1 (still) | 0.34 | 2.32 | 7.18 | 0.71 | 0.34 |
| traveler c10r1 (still) | 0.25 | 2.60 | 5.47 | 0.87 | 0.38 |
| PTBP PEDS022 0021 (subtle motion) | - | 1.77 | 4.25 | 0.79 | 0.69 |
| PTBP PEDS120 0015 (motion + slice dropout) | - | 3.00 | 14.57 | 1.97 | 1.39 |

Effect on the data (c01r1, same tensor fit and mask, b <= 1500): WM mean FA 0.468 (authors' data) -> 0.380 after the
pipeline's motion stage (-19%), voxel FA correlation 0.80.

## Causes (12 volumes of c01r1, b = 1000/2000/3000, 4 each; truth = 0 motion)

| registration | mean translation | max | mean rotation |
|---|---|---|---|
| `pytorch_batched_adaptive`, DW-mean target (pipeline default) | 1.33 mm | 2.77 | 0.33 deg |
| `pytorch_batched` (full precision), DW-mean target | 0.24 | 0.66 | 0.06 |
| ANTs rigid (mean squares), DW-mean target | 0.17 | 0.31 | 0.08 |
| ANTs rigid, same-shell mean | 0.04 | 0.20 | 0.04 |
| ANTs rigid, leave-one-out spherical-harmonic prediction | 0.02 | 0.04 | 0.02 |
| `pytorch_batched`, leave-one-out Gaussian-process target (the preset) | 0.049 | 0.076 | 0.074 |

1. The adaptive backend registers at 8 then 4 mm resolution caps with a narrow search and a motion gate (tuned for
   PET-like series); on 1.5 mm data it cannot resolve sub-millimetre motion. This is about 1.1 mm of the 1.33 mm.
2. One isotropic mean is a poor target for volumes whose contrast depends on gradient direction (about 0.2 mm).
   Registering each volume to a model-free prediction of its own direction is the approach of FSL eddy
   (Andersson & Sotiropoulos 2016, NeuroImage 125:1063).

## Recovery of known motion

All 30 b=1000 volumes of c01r1 displaced by random rigid motion (sd 1.5 mm / 1.5 deg, mean max probe displacement
2.54 mm); error is the probe displacement difference after removing the group's common offset (the target defines an
arbitrary common pose):

| method | mean error | max error |
|---|---|---|
| `pytorch_batched`, mean target | 0.073 mm | 0.124 |
| + leave-one-out target, 1 round | 0.062 | 0.099 |
| + leave-one-out target, 2 rounds | 0.063 | 0.120 |

A second round adds nothing, so the preset uses one round after the first pass. The floor of about 0.06 mm includes
the linear interpolation used to inject the motion.

## Interpolation

A random 0.25 mm / 0.15 deg resample of the still scan (no correction applied) changed WM FA by: linear -5.1%,
b-spline -1.1%, windowed sinc -1.3%. The preset therefore resamples with b-spline, and group-B frames are resampled
once from the original through the composed chain (cross transform, then own transform) instead of a second time from
an already resampled frame.

## Framewise displacement

`calculate_framewise_displacement` was checked numerically: Power FD equals the hand calculation to 1e-14; Jenkinson
FD matches an independent Monte-Carlo average over a 50 mm ball to 0.16%. Power FD (Power et al. 2012,
NeuroImage 59:2142; scrubbing at 0.5 mm) sums six absolute differences between consecutive frames, so independent
per-volume estimation error is amplified: with true motion zero, 0.15 mm / 0.10 deg of error per volume reads as Power
FD 0.82 mm and Jenkinson FD 0.38 mm. In DWI consecutive frames are different gradient directions, so the errors are
independent.

Two inconsistencies in the generic grouped path: FD was computed within each group (a DW volume against the previous DW
volume, b0 transitions never counted, first frame of each group 0), and summaries averaged over a different set of
frames than the BOLD summary. `motion_correct_grouped` now returns `fd` as the series-level Jenkinson FD in
acquisition order in group A's space (`jenkinson_framewise_displacement`, `syntx.motion`), and keeps the old per-group
Power FD as `fd_group_power`. BOLD and ASL keep Power FD as primary (the 0.5 mm scrubbing threshold is defined for it)
and report Jenkinson alongside.

## The preset (`syntx.motion_presets.DWI_MOTION_PRESET`)

`motion_correct_grouped(image, group_is_reference, preset="dwi", bvals=..., bvecs=...)`: full-precision
`pytorch_batched` backend; group B registered first to its mean, then once to the per-direction leave-one-out Gaussian
process prediction (Watson-type kernel exp(kappa (cos^2 - 1)), kappa 6, nugget 0.05, per b-shell); b-spline
resampling; one cross transform. Per-frame references are also available to any caller:
`motion_correction(reference=[image per frame], backend="pytorch_batched")`.

## Final-code results

RESULTS_TABLE

## Not established

* Thresholds for Jenkinson FD in DWI are not literature-calibrated here; `FD_THRESHOLD` (0.5 mm) in antsxdwi remains
  a QC flag only (counted and plotted, never used to censor).
* Slice-wise dropout (PTBP PEDS120: slices at 27-30% of the median signal in volumes 1, 2, 3, 7, 8) is not handled
  by the preset; the metric is mutual information plus correlation, not a robust loss. MSE, MAE and Huber were not
  compared.
* One subject's scanner for the bias numbers beyond c01r1/c05r1/c10r1; the injected-motion recovery used c01r1 only.
