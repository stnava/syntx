# DWI motion correction: bias in the generic estimator and the `dwi` preset (syntx 6.9)

Measured 2026-10-13/15 on real data (scratch scripts not committed). Truth data: the multicenter travelling-subject
dataset (Tong, He et al.; CC BY 4.0), DWI already corrected by the authors (1.5 mm, 96 volumes, b=0/1000/2000/3000),
so true residual motion is about zero.

## Problems found
1. Generic `pytorch_batched_adaptive` (8/4 mm caps, motion gate) on a still head reported 1-1.7 mm per DW volume and
   lowered WM FA by 19%. The 4-8 mm caps cannot resolve sub-mm motion on 1.5 mm data.
2. Cross-contrast bias: registering b0 to diffusion-weighted volumes (or one shell to another) is biased and the bias
   grows with the b-value gap (0.6 mm b0-vs-DW in the first design). Adjacent-contrast chaining is accurate.
   MI needs enough bins: 32 is too coarse, 128 is used for the cross step.
3. Power FD is inflated by independent per-volume estimation noise; the series-level Jenkinson FD is reported instead.
4. Linear resampling loses about 5% FA; b-spline about 1%.

## Design (`motion_correct_grouped(image, shell_groups(bvals), preset="dwi")`)
Single pass, k groups (b0, then one group per b-shell, ascending):
1. per-group mean from low-motion frames; each group registered to its own mean (batched Mattes MI + correlation,
   2 mm registration cap, b-spline for b0, linear for the intermediate step);
2. corrected group means chained by rigid registration (b0_mean <- shell1_mean <- shell2_mean ...), 128 MI bins;
3. every frame resampled once from the original through its composed chain [cross_1..cross_g, own];
4. series Jenkinson FD in acquisition order from the composed matrices.
No gradient directions are needed. Parameters: `syntx.motion_presets.DWI_MOTION_PRESET`.
Losses come from `syntx.core.losses` (central Parzen/MI/Pearson); no local copies.

## Results (traveler c01r1, 96 volumes, MPS)
| test | result |
|---|---|
| still scan WM FA ratio vs authors' data | 0.999 (old pipeline 0.81), voxel corr 0.984 |
| still scan per-shell mean offsets | b0 0.028, b1000 0.055, b2000 0.043, b3000 0.021 mm |
| injected motion (sd 1.5 mm/deg), all DW-to-DW error | 0.204 mm (max 0.45); b1000 0.051, b2000 0.171, b3000 0.321 |
| injected b0 vs DW offset | 0.071 mm |
| time | 71-77 s (earlier two-mean single pass 62 s; v6.7 8-25 min) |
| PTBP PEDS022 / PEDS120 | 29 s / 30 s, series FD mean 0.45 / 0.98 mm |
Within-shell error grows with b-value (noisier volumes).

## Negative results (not in code)
Edge/gradient-magnitude images instead of intensities; intensity normalisation; leave-one-out per-direction targets
(second pass; costs half the run time, no gain over the single pass at the k-group design); fast/medium schedules for
DWI; Adam-only schedule; caching small tensors; 2.5/3 mm registration caps (worse than 2 mm).

## Caveats
* Batched L-BFGS shares one line search across frames; frames with slice dropout (PEDS120) can couple. No truth for it yet.
* No ground truth for PTBP; Jenkinson FD thresholds for DWI are not literature-calibrated (antsxdwi flag only).
* Injected-motion and FA checks used c01r1 only.
* Multi-channel (b0_mean + dwi_mean) registration is supported in `syntx.syn` but not in the batched rigid registrar; the
  chain already leaves <=0.14 mm cross bias, so it was not built.
