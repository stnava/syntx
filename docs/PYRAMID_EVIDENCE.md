# Pyramid policy evidence (Pass 1)

Harness: `benchmarks/pyramid_compare.py` (sub-commands `synthetic`, `real`, `isotropic`, `motion`, `properties`,
`deformable`); subject builder `benchmarks/pyramid_build_subjects.py`; the solvers expose the private, default-preserving
`_level_builder` seam (`robust_affine._run_pytorch_affine_solver`, `motion_batched._batched_rigid_register_pass_core`), proven by
`tests/test_pyramid_seam.py` and the existing affine/motion tests. Specification: `docs/PYRAMID.md`.

## 1. Policies

| id | pyramid | where it lives today |
|---|---|---|
| DEFAULT | no builder: mm pyramid for an anisotropic fixed image, voxel pooling otherwise; an isotropic moving image is pooled | `robust_affine` as shipped |
| P1 | voxel `avg_pool3d` by L on every axis; the solver's 8-voxel coarsest-level clamp (in voxels) applies | `robust_affine._pool_level`, `motion_batched` |
| P2 | Gaussian (sigma = log2 L voxels) + `F.interpolate(align_corners=True)`, i.e. `build_image_pyramid`; 8-voxel clamp in voxels | `pyramid.build_image_pyramid` (syn, syngs, tvf) |
| P3 | `m = max(int(n/L), 16)` per axis, compact Gaussian `0.5 n/m`, `F.interpolate(align_corners=True)`; the floor is its own clamp | `greedy`, `build_anti_aliased_pyramid` |
| P4 | millimetre isotropic grid of L x finest fixed spacing, area average (coarser axes) / centre-to-centre linear (finer axes), both images; millimetre clamp | `robust_affine._mm_pyramid_level` |
| P5 | P4 at levels > 1, the native fixed grid at level 1 | new |
| `X+m` | any of the above with a valid-region margin: fixed samples whose centre lies within one native slice of either end of the thick axis (c < 1 or c > n-2) do not enter the metric (anisotropic fixed images only) | new |

Notes that bound what the table can say. P2 and P3 are evaluated with the operators of the models they come from, but inside the
affine solver; their warp-field conventions (align_corners) are irrelevant to a rigid pose. The coarse-level phase-correlation
translation candidates of `robust_affine` use `_mm_pyramid_level` in every arm (they are not a policy). `motion_batched`
anisotropic cases use `seed_mode="narrow"` because its phase-correlation seeding pools in voxels in every arm.

## 2. Decision criteria (fixed before the sweep was run)

These were written before the sweep (a single smoke run of the harness at 4:1, pose A, seed 42, one trial per policy, had been
run to check that the harness works; it was not used to choose anything). They are applied mechanically in section 5.

- **C0 zero crashes.** No exception in any solver run (synthetic, real, isotropic, motion). A solver exception counts as a failure
  of that policy, not a retry.
- **C1 no accuracy loss on any isotropic case.** Synthetic ratio 1:1 (12 trials), the isotropic same- and cross-modality pairs and
  the isotropic motion cases: mean TRE <= DEFAULT's mean + 0.05 mm and worst TRE <= DEFAULT's worst + 0.10 mm.
- **C2 strictly better than DEFAULT on thin slabs at every ratio.** Ratios 2, 4, 6, 8, 10:1, 9 trials each (3 poses x 3 seeds; amended before any analysis to 12 trials, 4 poses x 3 seeds, see section 3):
  (a) mean TRE strictly lower than DEFAULT's wherever DEFAULT's mean is >= 0.30 mm; where DEFAULT's mean is already < 0.30 mm
  (below the 0.75 mm in-plane voxel, nothing to improve) the policy must be within 0.05 mm of it; (b) the number of trials with
  TRE < 1.5 mm (the repository's test threshold) is >= DEFAULT's; (c) **4:1 and 10:1 must pass**: all trials < 1.5 mm.
- **C3 time within +10 % on isotropic cases.** Median wall time of back-to-back runs (every policy once per repeat, rotating
  order, same process, same machine load) <= 1.10 x DEFAULT's, for each isotropic pair.
- **C4 real pairs do not regress** (10 ds001242 subjects, T1 slab -> NM, ANTs rigid Mattes as reference): the median pose
  error (max probe displacement, mm) <= DEFAULT's and the number of subjects within 3 mm of the ANTs pose >= DEFAULT's. A solved
  pose whose loss is better than the loss at the ANTs pose is reported as such, not as a failure of the solver.
- **C5 motion correction.** Isotropic known-transform cases: mean TRE <= DEFAULT's + 0.05 mm. Anisotropic cases: not worse than
  DEFAULT by more than 0.05 mm (DEFAULT there is voxel pooling; improvement is reported, not required).
- **C6 builder properties** (any policy that could ship): output axes >= 2 voxels, no NaN and a constant image preserved at sigma 0,
  physical coordinates correct (ramp through `(f, off)` within 0.05 working voxel in the interior), deterministic, CPU vs MPS
  within 1e-4. A builder that raises on a thin axis without the solver clamp (P1, P2 as used by `build_image_pyramid`) is a
  documented requirement for the clamp, not a disqualification, as long as C0 holds through the solver.

Winner rule: among the variants meeting C0-C6, the lowest mean TRE over the thin-slab ratios, ties broken by time. A separate
affine/motion and deformable winner is named only if no single policy satisfies all. The deformable models are not run through
policies (see section 6), so no deformable default is changed on this evidence.

All policies are compared back to back in the same process; the machine was shared with another CPU-heavy project
(about 350 % CPU), so absolute times are inflated and only the within-run ratios mean anything. Device: MPS.

## 3. What was run, and deviations from the plan as first written

- Hardware and load: Apple MPS, shared machine; all policies for a given case run back to back in one process. Wall times of a
  few runs include a laptop sleep (10 synthetic and 2 real runs with times of 900 s or more); only medians are used and the
  dedicated isotropic timing runs (section 4.3) were run alone under `caffeinate`.
- Synthetic sweep: 6 ratios x 4 poses x 3 solver seeds x 11 variants = 732 solver runs, 0 exceptions. Poses B and C
  (rotation 8-10 degrees, translation 6-7 mm) were tried first and were screened out because every policy including DEFAULT
  fails them at 1:1 (TRE 5.7 and 7.4 mm; the partial data are kept in `screen_invalid_poses/`): outside the capture range of the
  solver on this phantom, they cannot discriminate pyramids. Replacements D, E, F were the first three of a fixed list of five
  candidates that DEFAULT solves at 1:1 (TRE < 0.11 mm); no thin-slab result was used to choose them. The solver seed barely
  matters (median spread across seeds of the same case 0.02 mm; up to 2.4 mm in a few cases near a capture boundary), so the
  independent trials are effectively the 4 poses.
- Real pairs: sub-01 is the saved debug pair; sub-02..10 were built with `benchmarks/pyramid_build_subjects.py` (brain extraction,
  `compute_midbrain_priors`, N4 of the NM, T1 cropped to the slab, ANTs rigid Mattes reference, random_seed 1). All ten pairs and
  references are in `/Users/stnava/data/nm_registration_debug/subjects/sub-XX/`. One solver seed (42) per pair and variant: 110 runs, 0 exceptions.
  "Solver loss at the ANTs pose" is the solver's own coarse-level full-grid loss with the optimiser run for zero iterations from
  the ANTs transform (same policy, same level); for sub-07 under DEFAULT, P1, P2 and P3 that pose is rejected by the solver's own
  self-check (not better than identity at the coarse level), which is itself a finding (section 4.2). "Native MI" is ANTs
  histogram MI of the fixed image and the warped moving image on the full native grid, the same for every policy.
- Two harness bugs found and fixed during the run, with the superseded outputs kept in `superseded/`: the ramp test averaged
  over the cross-section (edge voxels of a zero-padded filter polluted it) and the motion truth transform had the wrong direction
  (frames are resampled through T, so fixed-to-moving is T^-1); both were rerun. The C6 tolerance was not changed after seeing
  numbers: it was stated in working voxels and the harness now reports in working voxels.
- Motion: synthetic blob volumes resampled through known rigid transforms (isotropic 2 mm, small and large jumps; anisotropic
  2x2x6 mm and 1.5x1.5x9 mm grids), 2 repeats, MPS. There is no real motion series in this pass.

## 4. Results

All numbers below are generated by `python benchmarks/pyramid_summarize.py <results_dir>` from the JSON-lines in
`/Users/stnava/data/nm_registration_debug/pyramid_compare_results/` (kept outside the repository).

#### Synthetic thin slab, known rigid transform: TRE in mm, mean / worst / trials under 1.5 mm

| ratio | DEFAULT | P1 | P2 | P3 | P4 | P5 | P1+m | P2+m | P3+m | P4+m | P5+m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1:1 | 0.05 / 0.12 / 12 of 12 | 0.05 / 0.12 / 12 of 12 | 0.06 / 0.15 / 12 of 12 | 0.06 / 0.14 / 12 of 12 | 0.05 / 0.10 / 12 of 12 | 0.05 / 0.10 / 12 of 12 | 0.05 / 0.12 / 12 of 12 | 0.06 / 0.15 / 12 of 12 | 0.06 / 0.14 / 12 of 12 | 0.05 / 0.10 / 12 of 12 | 0.05 / 0.10 / 12 of 12 |
| 2:1 | 0.11 / 0.30 / 12 of 12 | 0.11 / 0.30 / 12 of 12 | 0.12 / 0.33 / 12 of 12 | 0.12 / 0.34 / 12 of 12 | 0.11 / 0.29 / 12 of 12 | 0.10 / 0.29 / 12 of 12 | 0.11 / 0.29 / 12 of 12 | 0.12 / 0.33 / 12 of 12 | 0.12 / 0.34 / 12 of 12 | 0.10 / 0.28 / 12 of 12 | 0.10 / 0.28 / 12 of 12 |
| 4:1 | 0.92 / 2.77 / 9 of 12 | 0.91 / 2.82 / 9 of 12 | 0.95 / 2.89 / 9 of 12 | 0.98 / 2.94 / 9 of 12 | 0.91 / 2.80 / 9 of 12 | 0.93 / 2.90 / 9 of 12 | 0.94 / 2.87 / 9 of 12 | 0.94 / 2.84 / 9 of 12 | 0.99 / 2.97 / 9 of 12 | 0.89 / 2.74 / 9 of 12 | 0.92 / 2.85 / 9 of 12 |
| 6:1 | 0.72 / 1.75 / 9 of 12 | 0.69 / 1.76 / 9 of 12 | 0.69 / 1.76 / 9 of 12 | 0.81 / 1.94 / 9 of 12 | 0.72 / 1.79 / 9 of 12 | 0.73 / 1.82 / 9 of 12 | 0.72 / 1.78 / 9 of 12 | 0.72 / 1.78 / 9 of 12 | 0.80 / 1.91 / 9 of 12 | 0.71 / 1.75 / 9 of 12 | 0.74 / 1.86 / 9 of 12 |
| 8:1 | 1.27 / 3.33 / 9 of 12 | 1.18 / 2.99 / 9 of 12 | 1.18 / 2.99 / 9 of 12 | 6.75 / 25.35 / 9 of 12 | 1.26 / 3.28 / 9 of 12 | 1.26 / 3.29 / 9 of 12 | 1.38 / 3.76 / 9 of 12 | 1.38 / 3.76 / 9 of 12 | 6.43 / 25.25 / 9 of 12 | 1.26 / 3.19 / 9 of 12 | 1.25 / 3.14 / 9 of 12 |
| 10:1 | 7.83 / 30.50 / 9 of 12 | 7.69 / 29.65 / 9 of 12 | 7.69 / 29.65 / 9 of 12 | 7.95 / 30.56 / 9 of 12 | 7.76 / 30.75 / 9 of 12 | 7.84 / 30.64 / 9 of 12 | 7.99 / 30.47 / 9 of 12 | 7.99 / 30.47 / 9 of 12 | 8.20 / 31.50 / 9 of 12 | 8.05 / 30.96 / 9 of 12 | 8.06 / 30.72 / 9 of 12 |

#### Real T1-slab to NM pairs (ds001242 sub-01..10), pose error vs the ANTs rigid reference

| policy | pairs | median err mm | mean | worst | within 3 mm | within 2 mm | solver loss better than at ANTs pose | native MI better than ANTs | median s |
|---|---|---|---|---|---|---|---|---|---|
| DEFAULT | 10 | 0.3 | 3.2 | 26.1 | 8 | 8 | 8 | 4 | 11.8 |
| P1 | 10 | 0.1 | 3.1 | 26.8 | 9 | 8 | 6 | 6 | 48.0 |
| P2 | 10 | 0.1 | 3.1 | 26.8 | 9 | 8 | 6 | 6 | 48.0 |
| P3 | 10 | 0.5 | 3.5 | 26.4 | 8 | 7 | 9 | 6 | 8.3 |
| P4 | 10 | 0.2 | 3.1 | 26.7 | 9 | 8 | 10 | 5 | 11.7 |
| P5 | 10 | 0.1 | 3.1 | 26.7 | 9 | 8 | 7 | 7 | 10.8 |
| P1+m | 10 | 0.2 | 3.1 | 27.0 | 9 | 8 | 9 | 4 | 52.3 |
| P2+m | 10 | 0.2 | 3.1 | 27.0 | 9 | 8 | 9 | 4 | 52.2 |
| P3+m | 10 | 0.3 | 1.0 | 5.5 | 9 | 8 | 9 | 4 | 8.0 |
| P4+m | 10 | 0.2 | 0.4 | 1.5 | 10 | 10 | 10 | 5 | 10.6 |
| P5+m | 10 | 0.1 | 0.4 | 1.3 | 10 | 10 | 8 | 6 | 10.0 |

Per subject pose error (mm):

| subject | DEFAULT | P1 | P2 | P3 | P4 | P5 | P1+m | P2+m | P3+m | P4+m | P5+m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 4.4 | 2.7 | 2.7 | 4.8 | 2.8 | 2.7 | 2.6 | 2.6 | 2.6 | 1.5 | 1.3 |
| sub-02 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| sub-03 | 0.2 | 0.1 | 0.1 | 0.3 | 0.2 | 0.1 | 0.1 | 0.1 | 0.3 | 0.1 | 0.1 |
| sub-04 | 0.6 | 0.3 | 0.3 | 2.0 | 0.3 | 0.3 | 0.2 | 0.2 | 0.4 | 0.3 | 0.3 |
| sub-05 | 0.5 | 0.7 | 0.7 | 0.8 | 0.5 | 0.7 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| sub-06 | 0.4 | 0.1 | 0.1 | 0.6 | 0.3 | 0.1 | 0.2 | 0.2 | 0.4 | 0.3 | 0.1 |
| sub-07 | 26.1 | 26.8 | 26.8 | 26.4 | 26.7 | 26.7 | 27.0 | 27.0 | 5.5 | 1.3 | 1.2 |
| sub-08 | 0.1 | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 |
| sub-09 | 0.1 | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.3 | 0.1 |
| sub-10 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.1 | 0.1 | 0.1 |

#### Isotropic pairs: median wall time (s) over back-to-back repeats, and TRE (mm)

| pair | policy | runs | median s | x DEFAULT | mean TRE | worst TRE |
|---|---|---|---|---|---|---|
| cross_modal | DEFAULT | 5 | 2.68 | 1.00 | 0.866 | 0.866 |
| cross_modal | P1 | 5 | 2.70 | 1.01 | 0.866 | 0.866 |
| cross_modal | P2 | 5 | 2.88 | 1.07 | 0.946 | 0.946 |
| cross_modal | P3 | 5 | 2.70 | 1.01 | 1.048 | 1.048 |
| cross_modal | P4 | 5 | 2.71 | 1.01 | 0.866 | 0.866 |
| cross_modal | P5 | 5 | 2.86 | 1.07 | 0.866 | 0.866 |
| same_modality | DEFAULT | 5 | 3.32 | 1.00 | 0.008 | 0.008 |
| same_modality | P1 | 5 | 3.25 | 0.98 | 0.008 | 0.008 |
| same_modality | P2 | 5 | 3.45 | 1.04 | 0.015 | 0.015 |
| same_modality | P3 | 5 | 3.24 | 0.98 | 0.006 | 0.006 |
| same_modality | P4 | 5 | 3.27 | 0.98 | 0.008 | 0.008 |
| same_modality | P5 | 5 | 3.28 | 0.99 | 0.008 | 0.008 |

#### motion_batched, known transforms: TRE mm (mean over frames / worst frame), median seconds

| case | policy | TRE mean / worst | median s | errors |
|---|---|---|---|---|
| aniso_3x | DEFAULT | 0.26 / 0.35 | 1.0 | 0 |
| aniso_3x | P1 | 0.26 / 0.35 | 0.8 | 0 |
| aniso_3x | P2 | 0.28 / 0.38 | 1.0 | 0 |
| aniso_3x | P3 | 0.28 / 0.37 | 1.0 | 0 |
| aniso_3x | P4 | 0.28 / 0.36 | 1.0 | 0 |
| aniso_3x | P5 | 0.28 / 0.36 | 0.9 | 0 |
| aniso_6x | DEFAULT | 0.85 / 0.98 | 1.1 | 0 |
| aniso_6x | P1 | 0.84 / 0.98 | 1.5 | 0 |
| aniso_6x | P2 | 0.83 / 0.98 | 1.2 | 0 |
| aniso_6x | P3 | 0.83 / 0.97 | 1.1 | 0 |
| aniso_6x | P4 | 0.85 / 1.00 | 1.2 | 0 |
| aniso_6x | P5 | 0.85 / 1.00 | 1.1 | 0 |
| iso_large | DEFAULT | 0.04 / 0.05 | 4.1 | 0 |
| iso_large | P1 | 0.04 / 0.05 | 3.9 | 0 |
| iso_large | P2 | 0.04 / 0.05 | 4.3 | 0 |
| iso_large | P3 | 0.04 / 0.06 | 4.2 | 0 |
| iso_large | P4 | 0.04 / 0.05 | 4.0 | 0 |
| iso_large | P5 | 0.04 / 0.05 | 3.8 | 0 |
| iso_small | DEFAULT | 0.03 / 0.03 | 4.1 | 0 |
| iso_small | P1 | 0.03 / 0.03 | 3.9 | 0 |
| iso_small | P2 | 0.03 / 0.03 | 4.0 | 0 |
| iso_small | P3 | 0.03 / 0.03 | 4.0 | 0 |
| iso_small | P4 | 0.03 / 0.03 | 4.0 | 0 |
| iso_small | P5 | 0.03 / 0.03 | 4.0 | 0 |

#### Builder properties (shapes (3,64,64), (11,128,128), (11,256,256), (64,64,64), (7,33,65), (16,96,96), (2,40,40); levels 1,2,4,8)

| policy | built | shapes that raise (no solver clamp) | axis < 2 | NaN | constant not preserved | max ramp error (working voxels, interior) | non-deterministic | max CPU-vs-MPS abs diff |
|---|---|---|---|---|---|---|---|---|
| P1 | 18 of 28 | (2, 40, 40); (3, 64, 64); (7, 33, 65); (11, 128, 128); (11, 256, 256) | 0 | 0 | 0 | 0.000 | 0 | 0.0e+00 |
| P2 | 18 of 28 | (2, 40, 40); (3, 64, 64); (7, 33, 65); (11, 128, 128); (11, 256, 256) | 0 | 0 | 0 | 0.000 | 0 | 1.2e-07 |
| P3 | 28 of 28 | none | 0 | 0 | 21 | 0.000 | 0 | 3.0e-07 |
| P4 | 28 of 28 | none | 2 | 0 | 0 | 0.063 | 0 | 1.2e-07 |
| P5 | 28 of 28 | none | 2 | 0 | 0 | 0.063 | 0 | 1.2e-07 |

Raw shipping pyramids on a thin axis (no solver clamp):

| function | shape | level | result |
|---|---|---|---|
| build_image_pyramid | [3, 64, 64] | 4 | RuntimeError: Input and output sizes should be greater than 0, but got input (D: |
| build_image_pyramid | [3, 64, 64] | 8 | RuntimeError: Input and output sizes should be greater than 0, but got input (D: |
| build_image_pyramid | [11, 128, 128] | 4 | [2, 32, 32] |
| build_image_pyramid | [11, 128, 128] | 8 | [1, 16, 16] |
| build_anti_aliased_pyramid | [3, 64, 64] | 4 | [16, 16, 16] |
| build_anti_aliased_pyramid | [3, 64, 64] | 8 | [16, 16, 16] |
| build_anti_aliased_pyramid | [11, 128, 128] | 4 | [16, 32, 32] |
| build_anti_aliased_pyramid | [11, 128, 128] | 8 | [16, 16, 16] |

Time and allocation, volume (176, 224, 192), MPS time / CPU total allocation:

| policy | level | ms (median of 5) | output voxels | CPU MB allocated |
|---|---|---|---|---|
| P1 | 4 | 0.27 | 118272 | 0 |
| P1 | 2 | 0.24 | 946176 | 4 |
| P2 | 4 | 4.96 | 118272 | 1919 |
| P2 | 2 | 4.48 | 946176 | 1555 |
| P3 | 4 | 3.32 | 118272 | 818 |
| P3 | 2 | 3.50 | 946176 | 821 |
| P4 | 4 | 2.14 | 118272 | 43 |
| P4 | 2 | 3.22 | 946176 | 99 |
| P5 | 4 | 2.23 | 118272 | 43 |
| P5 | 2 | 3.11 | 946176 | 99 |

#### Deformable-model pyramids, characterised (isotropic volumes; RMSE of each level mapped back to the full grid, vs P1)

| volume | level | policy | shape (z,y,x) | mean | std | max | RMSE vs P1 |
|---|---|---|---|---|---|---|---|
| phantom_96x104x88 | 8 | P1 | [11, 13, 12] | 0.433 | 0.280 | 0.798 | - |
| phantom_96x104x88 | 8 | P2 | [11, 13, 12] | 0.366 | 0.294 | 0.746 | 0.0311 |
| phantom_96x104x88 | 8 | P3 | [16, 16, 16] | 0.367 | 0.287 | 0.781 | 0.0572 |
| phantom_96x104x88 | 8 | P4 | [11, 13, 12] | 0.433 | 0.280 | 0.798 | 0.0000 |
| phantom_96x104x88 | 8 | P5 | [11, 13, 12] | 0.433 | 0.280 | 0.798 | 0.0000 |
| phantom_96x104x88 | 4 | P1 | [22, 26, 24] | 0.433 | 0.292 | 0.959 | - |
| phantom_96x104x88 | 4 | P2 | [22, 26, 24] | 0.403 | 0.294 | 0.879 | 0.0186 |
| phantom_96x104x88 | 4 | P3 | [22, 26, 24] | 0.392 | 0.292 | 0.886 | 0.0421 |
| phantom_96x104x88 | 4 | P4 | [22, 26, 24] | 0.433 | 0.292 | 0.959 | 0.0000 |
| phantom_96x104x88 | 4 | P5 | [22, 26, 24] | 0.433 | 0.292 | 0.959 | 0.0000 |
| phantom_96x104x88 | 2 | P1 | [44, 52, 48] | 0.433 | 0.297 | 0.994 | - |
| phantom_96x104x88 | 2 | P2 | [44, 52, 48] | 0.423 | 0.297 | 0.977 | 0.0089 |
| phantom_96x104x88 | 2 | P3 | [44, 52, 48] | 0.418 | 0.296 | 0.979 | 0.0249 |
| phantom_96x104x88 | 2 | P4 | [44, 52, 48] | 0.433 | 0.297 | 0.994 | 0.0000 |
| phantom_96x104x88 | 2 | P5 | [44, 52, 48] | 0.433 | 0.297 | 0.994 | 0.0000 |
| real_T1w_224x256x160 | 8 | P1 | [20, 32, 28] | 0.098 | 0.121 | 0.702 | - |
| real_T1w_224x256x160 | 8 | P2 | [20, 32, 28] | 0.089 | 0.116 | 0.623 | 0.0171 |
| real_T1w_224x256x160 | 8 | P3 | [20, 32, 28] | 0.088 | 0.111 | 0.577 | 0.0194 |
| real_T1w_224x256x160 | 8 | P4 | [20, 32, 28] | 0.098 | 0.121 | 0.702 | 0.0000 |
| real_T1w_224x256x160 | 8 | P5 | [20, 32, 28] | 0.098 | 0.121 | 0.702 | 0.0000 |
| real_T1w_224x256x160 | 4 | P1 | [40, 64, 56] | 0.098 | 0.130 | 0.765 | - |
| real_T1w_224x256x160 | 4 | P2 | [40, 64, 56] | 0.094 | 0.123 | 0.715 | 0.0123 |
| real_T1w_224x256x160 | 4 | P3 | [40, 64, 56] | 0.094 | 0.123 | 0.723 | 0.0128 |
| real_T1w_224x256x160 | 4 | P4 | [40, 64, 56] | 0.098 | 0.130 | 0.765 | 0.0000 |
| real_T1w_224x256x160 | 4 | P5 | [40, 64, 56] | 0.098 | 0.130 | 0.765 | 0.0000 |
| real_T1w_224x256x160 | 2 | P1 | [80, 128, 112] | 0.098 | 0.135 | 0.858 | - |
| real_T1w_224x256x160 | 2 | P2 | [80, 128, 112] | 0.097 | 0.130 | 0.791 | 0.0089 |
| real_T1w_224x256x160 | 2 | P3 | [80, 128, 112] | 0.097 | 0.130 | 0.790 | 0.0097 |
| real_T1w_224x256x160 | 2 | P4 | [80, 128, 112] | 0.098 | 0.135 | 0.858 | 0.0000 |
| real_T1w_224x256x160 | 2 | P5 | [80, 128, 112] | 0.098 | 0.135 | 0.858 | 0.0000 |

#### Criteria applied mechanically

| variant | C0 | C1 synth 1:1 | C1 iso pairs | C2 thin slab | C3 time | C4 real | C5 motion | C6 props | result / C2 detail |
|---|---|---|---|---|---|---|---|---|---|
| P1 | pass | pass | pass | FAIL | pass | pass | pass | pass | 4:1 not all < 1.5 mm; 10:1 not all < 1.5 mm |
| P2 | pass | pass | FAIL | FAIL | pass | pass | pass | pass | 4:1 not better; 4:1 not all < 1.5 mm; 10:1 not all < 1.5 mm |
| P3 | pass | pass | FAIL | FAIL | pass | FAIL | pass | FAIL | 4:1 not better; 4:1 not all < 1.5 mm; 6:1 not better; 8:1 not better; 10:1 not better; 10:1 not all < 1.5 mm |
| P4 | pass | pass | pass | FAIL | pass | pass | pass | FAIL | 4:1 not all < 1.5 mm; 6:1 not better; 10:1 not all < 1.5 mm |
| P5 | pass | pass | pass | FAIL | pass | pass | pass | FAIL | 4:1 not better; 4:1 not all < 1.5 mm; 6:1 not better; 10:1 not better; 10:1 not all < 1.5 mm |
| P1+m | pass | pass | pass | FAIL | pass | pass | pass | pass | 4:1 not better; 4:1 not all < 1.5 mm; 6:1 not better; 8:1 not better; 10:1 not better; 10:1 not all < 1.5 mm |
| P2+m | pass | pass | FAIL | FAIL | pass | pass | pass | pass | 4:1 not better; 4:1 not all < 1.5 mm; 6:1 not better; 8:1 not better; 10:1 not better; 10:1 not all < 1.5 mm |
| P3+m | pass | pass | FAIL | FAIL | pass | pass | pass | FAIL | 4:1 not better; 4:1 not all < 1.5 mm; 6:1 not better; 8:1 not better; 10:1 not better; 10:1 not all < 1.5 mm |
| P4+m | pass | pass | pass | FAIL | pass | pass | pass | FAIL | 4:1 not all < 1.5 mm; 10:1 not better; 10:1 not all < 1.5 mm |
| P5+m | pass | pass | pass | FAIL | pass | pass | pass | FAIL | 4:1 not better; 4:1 not all < 1.5 mm; 6:1 not better; 10:1 not better; 10:1 not all < 1.5 mm |

Variants passing all criteria: none


## 5. Verdict under the pre-stated criteria

No variant meets all criteria. C2 (thin slabs strictly better than DEFAULT at every ratio, and 4:1 and 10:1 passing) fails for
every one of the ten variants, and the reason is the same for all of them (section 6): the failing cases are search failures
that no pyramid fixes. The remaining criteria separate the variants: P2 and P3 lose accuracy on the isotropic cross-modality
pair (C1; P3 also fails C4 and the builder properties), P4/P5 pass everything except C2 and the strict coordinate
tolerance of C6 (see below), P1 passes everything except C2 but is four times slower on the real slabs (section 4.2).

Closest to the criteria: **P4 with the valid-region margin (P4+m)**, equal in accuracy to P5+m. Against DEFAULT on the ten real pairs: 10 of 10 within 3 mm of the ANTs
pose (DEFAULT 8 of 10), mean pose error 0.4 mm (DEFAULT 3.2), worst 1.5 mm (DEFAULT 26.1, sub-07), and the solved pose has a better solver loss than the ANTs pose in 10 of 10 and a
better native-grid MI in 5 of 10 (a better-scoring pose than ANTs' is not evidence that the solver is wrong). What it is missing: (a) the synthetic 4:1 and 10:1 failures
(search, section 6); (b) 6:1 is not strictly better (0.71 vs 0.72 mm, a tie within noise); (c) C6: the area average with a fractional
factor places the working voxel up to 0.063 working voxel (0.08 native voxel) from j f + off because the end voxels are
weighted fractionally (threshold 0.05), and an axis can end with 1 voxel when the whole field of view is thinner than the
target (guarded in the solver by the millimetre clamp).

## 6. Why the synthetic failures are not a pyramid effect

The same four cases fail for every policy (pose A at 4:1, 6:1 and 10:1; pose E at 8:1), with or without the margin (P3 also fails
E at 8:1; its 25 mm there is the same capture failure on a worse path). In each, the solver's coarse-level loss AND the native-grid MI
at the true transform are much better than at the solved transform (`python benchmarks/pyramid_compare.py probe_truth`, `probe_truth.jsonl`):

| ratio, pose | variant | TRE mm | coarse loss solved / truth | native MI solved / truth |
|---|---|---|---|---|
| 4:1, A | DEFAULT | 2.75 | -0.106 / -0.189 | -0.103 / -0.330 |
| 4:1, A | P4 | 2.78 | -0.127 / -0.262 | -0.101 / -0.330 |
| 6:1, A | DEFAULT | 1.74 | -0.146 / -0.190 | -0.181 / -0.334 |
| 8:1, E | DEFAULT | 3.22 | -0.091 / -0.189 | -0.141 / -0.561 |
| 10:1, A | DEFAULT | 29.94 | -0.025 / -0.163 | -0.010 / -0.415 |
| 4:1, F (a pass) | DEFAULT | 0.10 | -0.191 / -0.191 | -0.430 / -0.425 |

The objective prefers the truth at the coarse level and on the native grid; the optimiser (candidate starts plus coarse-to-fine descent) does not
reach it. What is missing for C2 is capture along the thick axis (a start search that scans the slab axis in steps of about half a
slice, or a wider candidate set), not a different downsampling operator.

The real pairs are different: on sub-07 the ANTs pose is rejected by the solver's self-check under DEFAULT, P1, P2 and P3
(its coarse loss is not better than identity), i.e. the objective itself favours a wrong pose by 26 mm. Excluding the outermost
slab-edge slice from the metric repairs that (P4+m: 1.3 mm; the ANTs pose is accepted, loss -0.488 against -0.504 at the solved pose), and P1+m, which keeps the
voxel pyramid, does not (27 mm), so on real data both the millimetre pyramid and the margin are needed. On sub-01 the same pair
gives 4.4 mm (DEFAULT), 2.8 (P4), 1.5 (P4+m), 1.3 (P5+m); the margin has no effect on the synthetic slabs, which have no edge-slice artefact.

## 7. Deformable models (item 2)

A policy cannot be swapped into the deformable models without rewriting them: `syn`, `syngs` and `tvf` call
`build_image_pyramid` and derive the velocity/warp-field grids and `level_spacing` from the resulting shapes under its
align_corners convention, `greedy` builds its pyramid inline with its own floor and field initialisation, and the JAX copies are
separate. Replacing the operator (cell-aligned avg_pool or area average against corner-aligned interpolation) shifts every level
by (f-1)/2 voxels unless every model's coordinate code changes too. No default-preserving seam was added, so no deformable accuracy
or Dice was measured and none is claimed. The pyramids were characterised instead (section 4.6): P1 and P4 coincide on
the isotropic volumes (RMSE 0.0000 after mapping to the full grid), P2 differs from them by 0.009-0.031 and P3 by 0.010-0.057
in intensity, the extra smoothing lowering the level mean and maximum (the phantom's mean 0.433 to 0.37-0.40 at level 8/4), and P3's compact Gaussian is
zero-padded, so a constant image is not preserved (21 of 28 builds darken the borders), which the replicate-padded
`separable_gaussian_filter` avoids. `build_image_pyramid` raises on a 3-voxel axis at level 4 and returns a 1-voxel axis for 11 slices at level 8;
`build_anti_aliased_pyramid` never shrinks below 16 (it up-samples a 3-voxel axis to 16).

## 8. Recommendation

1. Do not change any deformable default on this evidence. Migrate the deformable callers behind behaviour-preserving
   `legacy_interp` and `legacy_floor16` presets (Pass 3 as planned), fix the zero-padded kernel in the greedy/`build_anti_aliased_pyramid` path when migrating.
2. For affine and motion: the central module's default operator should be **P4, the millimetre isotropic area/linear pyramid, with a
   one-slice valid-region margin on the fixed image whenever it is anisotropic**, and `legacy_avgpool` kept as a named preset: P1 and P4 are identical on the isotropic
   volumes tested, accuracy and time on isotropic data are unchanged (time ratios 0.98-1.07 against DEFAULT; TRE equal to within 0.01 mm on the same-modality pair and identical on the cross-modal pair, 0.05 mm at 1:1), and on the real NM pairs P4+m is the only non-hybrid variant with all ten pairs within 3 mm.
   P5 has no advantage (equal accuracy within noise, one more rule), so it is not recommended; P2 and P3 are not recommended
   (isotropic cross-modal TRE 0.95 and 1.05 mm against 0.87; P3 fails the 8:1 pose E case by 25 mm and does not preserve a constant at the borders).
   This is a recommendation from criteria that were not all met; the owner should decide whether the real-pair evidence is enough to adopt P4+m before the full C2 is
   reached, and the xfails in `tests/test_robust_affine_anisotropy.py` should stay until the capture problem of section 6 is dealt with (the 4:1 and 10:1 synthetic cases would still fail).
3. The margin width (one slice) was the only value tried; two slices and a fractional edge weight are untested.
4. A thick-axis capture search is the open item for the synthetic xfails; it is independent of the pyramid and is not addressed here.

## 9. Negative results and what is unfinished

- No variant passes C2; no variant improves the synthetic thin-slab sweep beyond noise (differences of 0.01-0.1 mm; P1/P2 are better by 0.1 mm at 8:1 because the voxel clamp collapses them to the finest level, which costs four times the time on the real slabs).
- P3 is worse than DEFAULT on isotropic cross-modal (1.05 vs 0.87 mm) and fails one 8:1 case by 25 mm; P2 is slightly worse on isotropic cross-modal (0.95).
- The valid-region margin does nothing on the synthetic sweep; its value is entirely on real pairs (sub-01, sub-07) and rests on 2 of 10 subjects.
- P1 and P2 without the solver clamp raise on thin axes; with it they collapse to a single full-resolution level on an 11-slice slab (median 48 s against 12 s on the real pairs).
- Not done: deformable accuracy/Dice through policies (no seam); a real motion series; CPU-run solver comparisons (only builder CPU/MPS parity was checked); more than one solver seed on the real pairs; a second margin width; JAX pyramids.
