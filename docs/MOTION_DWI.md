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
`motion_correction(reference=[image per frame], backend="pytorch_batched")`, and
`batched_rigid_register_pass([one reference per frame], frames, seed_mode="narrow")` registers all frames in one batched
call (a per-frame intensity range is used for the mutual-information binning).

## Final-code results

Final code (preset `dwi`, b-spline single resample, series-level Jenkinson FD in acquisition order; `fd` is now a
different quantity from the baseline table's Power FD, so the FD columns are not directly comparable; the translation
and rotation columns are):

| case | series Jenkinson FD mean | max | DW translation | DW rotation | baseline translation (before) |
|---|---|---|---|---|---|
| traveler c01r1 (still) | 0.44 mm | 0.90 | 0.27 mm | 0.13 deg | 1.51 mm |
| traveler c05r1 (still) | 0.33 | 0.60 | 0.21 | 0.18 deg | 0.71 |
| traveler c10r1 (still) | 0.39 | 0.67 | 0.25 | 0.17 deg | 0.87 |
| PTBP PEDS022 0021 (subtle motion) | 0.45 | 1.78 | 0.59 | 0.71 deg | 0.79 |
| PTBP PEDS120 0015 (motion + dropout) | 1.03 | 6.93 | 1.50 | 1.04 deg | 1.97 |

Downstream effect on the still scan c01r1 (same tensor fit and mask as the baseline): WM mean FA 0.468 (authors'
data) -> 0.458 (ratio 0.977, voxel correlation 0.964); the baseline pipeline gave 0.380 (0.81, 0.80). The remaining
2% is of the order of the interpolation floor above. FA ratios were not computed for c05r1/c10r1.

Run time of the v6.7.0 design on this machine: 8-25 minutes per 96-volume session (one registration call per volume;
see "Speed work" for what replaced it).

The two PTBP scans still show estimated motion (about 0.6 and 1.5 mm); there is no ground truth for them, and the
preset's PEDS120 estimates are affected by its dropout slices (see below).

## Speed work (6.7.1)

Owner budget: about 2 minutes per 96-volume 1.5 mm session; accepted result: **about 3 minutes on this machine**, accuracy
unchanged within noise. Measured on c01r1 (96 volumes, 1.5 mm), machine otherwise idle (timings here vary up to 10x
with other load; one run stalled with 0% CPU):

| stage | v6.7.0 | 6.7.1 |
|---|---|---|
| b0 group, references, cross registration | 20 s | 21 s |
| pass 1 (90 volumes against the group mean, one batched call) | 205 s at 1.5 mm | 69 s (2 mm registration) |
| leave-one-out targets | 164 s (per-volume loop) | 1 s (one matrix product) |
| per-direction pass (90 volumes) | 90 single-frame calls, about 150 s | 60 s, one batched call, narrow seeds, 2 mm |
| resampling, FD, output | 46 s | about 30-45 s |
| **total** | **8-25 min** | **186 s (3.1 min)** |

Still-scan accuracy at 6.7.1: WM FA ratio 0.982 (authors' data 0.468 -> 0.459), voxel correlation 0.968 (v6.7.0: 0.977 /
0.964). Corrupt scan PTBP PEDS120, 31 volumes: 85 s (reference setting 193 s); tensor-fit residual 0.290 (uncorrected 0.303,
old default 0.296, slow reference 0.285).

What made it faster, each measured on its own:
* **Registration at 2 mm** (`resolution_cap_mm`, the existing `cap_resolution_for_registration`): same still-scan error as
  native 1.5 mm (0.052 vs 0.048 mm); 2.5 and 3 mm were worse (0.077, 0.069 mm). Transforms stay valid at native resolution.
* **Narrow seeds in the per-direction pass**: agree with the wide search to 0.014 mm on the high-motion scan.
* **One batched call with one reference per frame** (`motion_batched`): 0.89, 0.38, 0.17, 0.15 s per frame at batch sizes
  1, 4, 16, 64 (MPS, 2 mm). Needs a per-frame intensity range; with the maximum over the batch, dim frames (slice
  dropout, low-signal directions) use few bins and are mis-estimated.
* **Vectorised targets**: the leave-one-out prediction is linear, so all targets are one matrix product with the stacked
  volumes; identical to the per-volume formula (test).

What did not work (and is not in the code): caching tiny per-step tensors (the 40 s attributed to `torch.tensor` was GPU
time at a synchronisation point; no change in total), the `fast` schedule or a trimmed `medium` schedule (up to 1.4 mm /
2.5 deg off the reference on PEDS120), an Adam-only schedule (same disagreement), smaller L-BFGS chunks (4 frames per call
already differ), cheaper pass 1 (narrow seeds, fast schedule: 112 s but FA ratio 0.957 and correlation 0.912, rejected),
threading the resampling (no gain: 0.11 s per frame).

Caveats:
* Batched L-BFGS shares one line search across frames, so on a scan with slice dropout (PEDS120) the per-volume parameters
  differ from a one-frame-per-call run by 0.14 mm on average (up to about 0.7 mm). `loo_batch_frames: 1` in the preset
  restores one-frame-per-call exactly (it matched the reference to 0.004 mm) at about twice the time. There is no ground
  truth for that scan; the tensor-fit residual is not a clean arbiter either (it also rewarded the biased old default on
  clean volumes), so it is reported only as a sanity check.
* The still-scan "mean translation" printed by the scratch scripts (0.27-0.35 mm) is the pose relative to the group's
  reference pose, including a common offset that is not motion; FA ratio and series FD are the accuracy measures used.
* Remaining time: about 70 s of pass 1 and 60 s per-direction pass (registration, MPS), and a plumbing tail (resampling
  the 90 frames three times, 4D container building, DVARS) of about 40 s that could be trimmed further.
* A 2-minute target was not reached; 3 minutes was accepted by the owner.

## Not established

* Thresholds for Jenkinson FD in DWI are not literature-calibrated here; `FD_THRESHOLD` (0.5 mm) in antsxdwi remains
  a QC flag only (counted and plotted, never used to censor).
* Slice-wise dropout (PTBP PEDS120: slices at 27-30% of the median signal in volumes 1, 2, 3, 7, 8) is not handled
  by the preset; the metric is mutual information plus correlation, not a robust loss. MSE, MAE and Huber were not
  compared.
* The injected-motion recovery and the interpolation and FA checks used c01r1 only; c05r1 and c10r1 report motion only.
