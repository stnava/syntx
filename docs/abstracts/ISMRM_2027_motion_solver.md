# One motion solver for every MRI time series: the acquisition design inside the objective, and an account of what it explains

Author: Brian B. Avants. Affiliation: University of Virginia.

## Synopsis (100 words)

Head-motion estimates in MRI time series absorb whatever changes the images in step with the acquisition: ASL label
alternation, diffusion shell and direction contrast, eddy currents, the physiological alternation of fast resting BOLD.
We present one solver that registers all frames of a series in a single optimisation with the acquisition design
inside its objective, applies the terms physics calls displacements, removes the terms physics calls contrast, and
reports the share of every term. On resting BOLD, ASL and two multi-shell DWI series from four clinical cohorts, the
acquisition terms carry 39 to 75 % of the frame-to-frame displacement along the readout and phase-encode axes, and the
residual after correction no longer follows motion. Open software: syntx and the antsx pipelines.

## Introduction

Head-motion estimates in MRI time series absorb whatever changes the images in step with the acquisition: ASL
label alternation, diffusion shell and direction contrast and eddy currents, the near-Nyquist physiological
alternation of fast resting BOLD. Single-reference registration assigns all of it to motion [1,2]. We present one solver with the acquisition
design inside its objective for every modality, reporting every term's share.

## Methods

**Objective.** All T frames are registered in one optimisation. The pose of frame t is m_t + D_t beta: m_t the six
rigid head parameters, D_t the known acquisition design, beta its coefficients. The image term is a robust (Huber)
match of every frame to its own level's reference at one shared sample point set; the head trajectory carries a
Charbonnier prior on its second differences (knee 0.05 mm) in the units of the image log-likelihood; L-BFGS then a
coupled Newton stage solve all terms together. Terms that physics calls
displacements are applied (the eddy affine along phase encode, linear in the gradient [3]); contrast terms (level
offsets, l = 2 direction terms) are removed from the reported pose and never applied. One chain, one interpolation per frame.

**Design.** Resting BOLD: one level. ASL: one level per label state. Diffusion: one level per shell with its l = 2
terms and eddy affine.

**Band-limited FD.** For censoring, the near-Nyquist alternation (respiration aliased at a long TR, cardiac at a short
one) is removed from the pose series before FD is computed [2,4], from the pose spectrum or a recorded trace. The full
poses remain the correction: notching the alternation out of them raised tissue-edge temporal variance 15-20 %.

**Account of the model.** Pose domain: along each acquisition axis the applied displacement is head + applied term +
removed design-locked term; the share of the frame-to-frame step variance of each is reported, the readout axis being
B0-immune. Image domain: sequential sums of squares of per-voxel time series against the design (level intercepts,
l = 2 and l = 4 terms), then motion regressors, then the residual, raw and corrected, plus the residual-FD correlation.

**Data.** Four clinical cohorts: resting BOLD (TR 1.0 s, 300 volumes), pseudo-continuous ASL (control/label with M0,
25 volumes), two multi-shell DWI series (b = 1500/3000, 1.5 mm, 99-100 volumes), one with large motion, one at low SNR;
13-95 s per series on a laptop GPU. Simulated
series with known motion: head-FD error 0.319/0.210 mm (per-level registration/this solver) for DWI with a planted
eddy shift, 0.092/0.055 mm ASL, 0.133/0.068 mm slow flow.

**Software.** syntx 6.26 (`motion_correct_design`, `band_limited_fd`, `motion_decomposition`), antsxdwi 0.32,
antsxfunctional 0.22, antsxqc 0.4; one call per modality.

## Results

Table 1 and Figure 1. On both diffusion series the eddy term holds 71-75 % of the phase-encode step variance and the
design-locked term 39-45 % of the readout step variance; FD reports the remainder (0.57 and 0.52 mm against 0.91 and
0.86 mm for the full applied pose). After correction the design explains 95-96 % of the voxel variance, motion after
the design falls from 0.8 to 0.5 % and from 2.0 to 0.9 %, the residual from 4.8 to 3.3 % and from 7.4 to 3.7 %, and
the residual is uncorrelated with FD (r = 0.01, -0.12). Unmodelled l = 4 terms explain 0.8-1.0 %, as much as motion after the design. On the resting series band-limited FD lowers the mean from 0.62 to 0.33 mm and the censored
fraction from 48 to 16 %. On the 25-frame ASL series (FD 0.52 mm) the label and M0 levels explain 98.4 % of the variance after
correction, motion after the design falls from 0.34 to 0.17 % and the residual from 2.6 to 1.4 %. Every frame converged.

## Discussion

The misattribution of single-reference registration is now a number: on real multi-shell DWI, 39-45 % of the
displacement along the axis the scanner cannot distort was acquisition-locked and three quarters along phase encode
was eddy; reported as motion it would have doubled FD. The
decomposition is the model's own audit and shows its limits: a quarter of the resting-BOLD voxel variance still follows
the pose after correction, and the unmodelled l = 4 share on DWI equals the motion share, so the diffusion design should
be richer at b = 3000. Censoring thresholds are not yet tied to a downstream outcome, which a design-in-metric variant that matches
this solver on phantoms also awaits before it becomes default.

## Conclusion

One objective serves BOLD, ASL, multi-shell diffusion and slow-flow encodings: the acquisition design absorbs what
the acquisition causes, motion keeps what is left, and every term's share is reported with the result.

## Table 1

One solver, four series. Censored: frames above 0.5 mm on raw FD / on the censoring series (band-limited FD for
BOLD, rigid FD otherwise). Pose shares: percent of frame-to-frame step variance on the readout and phase-encode axes
held by head / applied eddy / removed design-locked term. Image: percent of per-voxel temporal variance explained by
the design, by motion after the design, and residual, raw / corrected; r: residual-FD correlation after correction.

| case | T | FD mean / max (mm) | censored | readout: head / eddy / design | PE: head / eddy / design | design | motion | residual | r | time |
|---|---|---|---|---|---|---|---|---|---|---|
| resting BOLD, TR 1.0 s | 300 | 0.62 / 2.78 | 48 % / 16 % | 100 / 0 / 0 | 100 / 0 / 0 | 0 / 0 | 35.0 / 23.6 | 65.0 / 76.4 | 0.62 | 37 s |
| ASL, control/label | 25 | 0.51 / 2.50 | 28 % / 28 % | 100 / 0 / 0 | 101 / 0 / 0 | 97.1 / 98.4 | 0.34 / 0.17 | 2.6 / 1.4 | 0.59 | 25 s |
| DWI, high motion | 100 | 0.57 / 2.53 | 43 % / 43 % | 64 / 0 / 39 | 20 / 75 / 14 | 94.4 / 96.2 | 0.79 / 0.47 | 4.8 / 3.3 | 0.01 | 120 s |
| DWI, low SNR | 99 | 0.52 / 1.71 | 47 % / 47 % | 61 / 0 / 45 | 26 / 71 / 14 | 90.6 / 95.4 | 2.00 / 0.85 | 7.4 / 3.7 | -0.12 | 89 s |

## Figure 1

`docs/reports/ismrm_cases_2026-10-28_figure.png`. One row per series: the head pose along the readout and phase-encode
axes with the applied eddy shift (dashed) and the removed design-locked term (dotted); FD with its band-limited
version; the variance explained per term, raw against corrected (log scale).

## References

[1] Freire L, Mangin JF. NeuroImage 2001. [2] Power JD et al. PNAS 2019. [3] Andersson JLR, Sotiropoulos SN.
NeuroImage 2016. [4] Fair DA et al. NeuroImage 2020. Also: Orchard J et al. IEEE TMI 2003; Huizinga W et al. MedIA 2016.
