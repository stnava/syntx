# Metric sample-point selection (`syntx.sampling`)

Every point-sampled metric in syntx draws its points through one module, `syntx.sampling`, designed as a menu of named
strategies (`SAMPLING_STRATEGIES`, `get_sampling_strategy(name)`; an unknown name raises). Call sites never implement
selection: the rigid motion pass and the group-bias pass (`motion_batched`), `robust_affine` (its `fixed_fg` domain,
its weighted draw and its candidate-scoring foreground) and the JAX Mattes MI (`regular_stride`,
`uniform_cube_points`).

## Why

The motion solver drew one fixed 15 % subset of `reference > 0.01` once, and every optimiser stage drew 20 % of it
again: about 3 % of the foreground, ~900 points on a 36 x 40 x 32 phantom, the same subset for every iteration and
frame. Its sampling error became a deterministic local optimum (slow-flow phantom seed 1: 0.53 deg rotation error,
identical on every run). `> 0.01` is intensity-scale dependent and admits background noise and ghosting. The same logic
was copied in several samplers, so a fix in one left the others wrong.

## Design

- **Candidate region (shared by every entry): `background_floor`.** Voxels above the reference's own noise floor and
  below the saturated top of its foreground:
  - `"noise"`: background = positive voxels <= 3 x the mode of the lower half of the positive histogram; floor =
    median + 4 x 1.4826 MAD (1.2533 x mean absolute deviation when the MAD is 0, e.g. integer data). For Rayleigh noise
    this floor sits near 3.8 sigma: about 7.6e-4 of the background voxels stay above it (independent audit; not 4e-6).
  - `"clipped_noise"`: zeros at least half as many as that low population and its positive half tiny next to tissue:
    noise clipped at 0, positive half is half-normal, floor = 4 x median / 0.6745 (real 3D PCASL).
  - `"zero-filled"`: zeros and no tiny positive population: floor 0, `UserWarning` (no noise to measure).
  - `"no_background"`: the "background" is > 3x noisier than the image's own noise from neighbour differences (the object
    fills the field of view): floor 0.
  - Exact zeros in contiguous blocks (padding / zero-filling; a morphological opening keeps them) are set aside before
    the classification; only scattered zeros (clipped noise) count. A zero-filled background (zero blocks over >= 20 %
    of the grid and no noise population among the positives) gives floor 0 with a `UserWarning`.
  - Exact zeros never enter the noise estimate. The 2-98 % percentiles of the saturated top are taken over the voxels above
    the floor (over the whole grid a small head in a large field of view lost 22 % of its tissue to "saturation").
- **One physical pool.** `PointSampler.pool()` draws physical points (mm) uniformly over the region (`sample_fraction /
  0.2` of it), jittered within their voxels. Every pyramid scale evaluates its own images at these same points; a scale
  draws its share with the strategy's weights at its own scale (sigma = the level's voxel size, mm). Coarse levels use
  the whole pool (thinning them to their voxel count left the capture stages a few hundred points: DWI rotation 0.83 vs
  0.36 deg). The caller's `sample_mask` restricts the full-resolution stages only.
- **Weights at a point** are the value in the point's own voxel (nearest): bilinear interpolation mixed in the zeros of
  neighbouring out-of-region voxels and under-weighted every edge point.
- **Speed.** Region and weights are computed once (ANTs, CPU) and uploaded once as float32; draws, jitter and indices
  stay on the compute device with a device generator (no host round trip per draw, no float64).
- **Cost cap.** One evaluation sees at most `MAX_EVAL_POINTS` = 20 000 points (the pool is capped at 20 000 / 0.2).
  Real DWI (140 x 140 x 104, 100 vols, MPS): no cap 170 s / mean FD 0.116 mm; 50 k 115 s / 0.109; 20 k 78 s / 0.125;
  former sampler 65 s / 0.163. The gate phantoms never reach the cap.
- **Menu.** `gradient_floor` (default: probability proportional to `0.1 + |grad n| / max`), `gradient`, `uniform`.

## Evidence (ground-truth phantoms, `syntx.motion_simulation`, 5 seeds; rotation error deg)

| case | former sampler | uniform | gradient | gradient_floor (default) |
|---|---|---|---|---|
| ASL | 0.181 | 0.120 | 0.115 | 0.113 |
| slow flow, no eddy | 0.280 +- 0.138 (max 0.526) | 0.158 | 0.164 | 0.166 |
| slow flow, eddy 0.4 mm | 0.183 | 0.160 | 0.140 | 0.135 |
| DWI | 0.672 | 0.430 | 0.382 | 0.392 (0.430 with the pilot's pe_scale conditioning) |
| BOLD | 0.192 | 0.145 | 0.138 | 0.138 |
| eddy PE scale (truth 0.03) | 0.03 +- 0.012 | 0.0362 | 0.0312 | 0.0322 (0.0300 with conditioning) |

Real references (FPA sub-BLAST079, ds001242 NM slab): DWI b0 / b3000, BOLD and the NM slab take the `"noise"` path
(44-57 % of the grid kept, < 3 % of a rough head mask lost); 3D PCASL takes `"clipped_noise"` (66 %). On EPI the
background is structured (ghosts, non-stationary noise): 40-58 % of the kept region lies outside a rough head mask at
z = 4; accepted, measured limit.

## Known limits

- An object filling the field of view whose dark tissue is MORE than half the voxels is read as "noise" and that dark
  tissue is cut (independent audit: dark class at 55-70 % of the voxels, 99.99 % of it dropped); with fewer dark voxels
  the neighbour-difference check gives `no_background`.
- On EPI the structured background (ghosts) partly survives the floor (see the real-data numbers above).
- The gradient strategies smooth with a Gaussian of the level's spacing; draws are reproducible per device (CPU and MPS
  draws differ).

## Debias of the reported trajectory (merged pilot ideas)

The pilot branch's grouped DWI PE / non-linear-eddy debias (`motion_correct_grouped(debias_pe_pseudomotion=...)`) and
its BOLD slice / pitch "Nyquist" extension (`motion_correction(debias_sagittal_nyquist=...)`) are OPT-IN: both shrink
any frame-to-frame alternation, and independent real jitter alternates too (audit: nodding jitter FD 0.94 -> 0.15 mm).
They change only the reported motion / FD, never the transforms.

## Not done

Restricting the JAX dense-grid MI (SyN / TVF) to the strategy region changes every deformable registration and needs its
own SyN benchmark; it keeps the `regular` stride (now defined in `syntx.sampling`).
