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
- **C1 no accuracy loss on any isotropic case.** Synthetic ratio 1:1 (9 trials), the isotropic same- and cross-modality pairs and
  the isotropic motion cases: mean TRE <= DEFAULT's mean + 0.05 mm and worst TRE <= DEFAULT's worst + 0.10 mm.
- **C2 strictly better than DEFAULT on thin slabs at every ratio.** Ratios 2, 4, 6, 8, 10:1, 9 trials each (3 poses x 3 seeds):
  (a) mean TRE strictly lower than DEFAULT's wherever DEFAULT's mean is >= 0.30 mm; where DEFAULT's mean is already < 0.30 mm
  (below the 0.75 mm in-plane voxel, nothing to improve) the policy must be within 0.05 mm of it; (b) the number of trials with
  TRE < 1.5 mm (the repository's test threshold) is >= DEFAULT's; (c) **4:1 and 10:1 must pass**: all 9 trials < 1.5 mm.
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
