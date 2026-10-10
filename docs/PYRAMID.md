# The central pyramid: one contract

Status: **implemented in `src/syntx/pyramid.py` (Pass 2); the deformable callers are migrated (Pass 3b, see Section 9 and Section 10); no default has changed.**
`docs/PYRAMID_EVIDENCE.md` holds the comparison that decided the former open items. Section 9 lists what exists and when to use
each preset. Items are marked **[decided]**, **[implemented]** or **[open]**.

syntx currently has about ten independent pyramid implementations (`pyramid.py` x2, `greedy`, `tvf_adj`, three JAX copies,
`robust_affine` x3, `motion_batched` x3, `scattered/solver`, `imaging_utils.cap_resolution_for_registration`). They disagree about
floors (4, 8, 16), smoothers (three), downsampling operators (avg_pool, `F.interpolate` with align_corners, area/matrix, ANTs
resample) and thin axes. The contract below replaces them.

## 1. Inputs
- A torch tensor `(B, C, *spatial)` in tensor order (z, y, x) plus `spacing`, `origin`, `direction` in ANTs (x, y, z) order.
  Several images may share one call (a `B` batch on one grid). **[decided]**
- An ANTs image through a thin wrapper (`image_to_tensor` in, the same per-level geometry out). The wrapper owns the
  ANTs-resample paths (`cap_resolution_for_registration`, the low-resolution ANTs candidates) so that they use linear
  interpolation and the same planner. **[decided]**

## 2. Levels
- `levels` stay integer shrink factors, e.g. `[8, 4, 2, 1]`, user-facing and unchanged. **[decided]**
- Meaning: the working spacing of level L is `L x (finest spacing of the fixed image)`, isotropic in millimetres.
  For an isotropic image this is the same grid as shrinking every axis by L. **[decided]**
- A moving image is leveled on its own grid with its own spacing and the same target in mm (never below its native spacing);
  it is not forced onto the fixed grid. **[decided]**

## 3. Per-axis rule (thin and thick axes)
Planned per axis from `(n, spacing, target_mm)`:
- coarsened axis (`target > spacing`): area average; working voxel j covers full-resolution indices `[j f, (j+1) f)`,
  `f = n / m`, `m = max(1, round(n spacing / target))`. The two end-voxel weights of a fractional window are solved so that the row
  centroid is exactly the window centre `j f + (f-1)/2` (the plain overlap-weighted box had a bias of up to 0.0625 working voxel
  at f = 4/3; the corrected matrix is the old one exactly for integer f). **[implemented]**
- non-coarsened axis (`target <= spacing`): the axis is not smoothed and is resampled to the target by centre-to-centre linear
  interpolation (policy P4; P5, native at level 1, showed no advantage and is not implemented). **[decided, implemented]**
- An axis never ends with fewer than 2 samples (an input axis of 1 sample passes through) and the central module never raises on a
  thin axis: the planner lifts every axis to `min(floor, n)` samples, so a thin axis keeps its native sampling (`legacy_floor16`
  keeps greedy's behaviour of up-sampling it to the floor). Invalid requests (levels < 1, unknown preset, bad sigma) raise the
  named `PyramidLevelError`. **[implemented; supersedes the earlier "raises" wording]**

## 4. One floor **[implemented]**
- `MIN_VOXELS_PER_AXIS = 16` is the single policy-floor constant (greedy's value, `imaging_utils.DEFAULT_MIN_VOXELS_PER_AXIS`);
  `HARD_MIN_VOXELS = 2` is the structural minimum, applied in every preset. The planner applies the floor per axis as
  `m = max(m_wanted, min(floor, n))`, so no level is ever collapsed and an axis shorter than the floor keeps its native samples.
- Presets that reproduce an incumbent without a floor (`mm_iso`, `legacy_avgpool`, `legacy_interp`) default to the structural
  minimum so their numbers do not change; callers opt in with `floor=MIN_VOXELS_PER_AXIS` (or any value). The affine solver's own
  coarsest-level clamp (8 finest voxels of the shortest extent) stays in `robust_affine` until Pass 4 decides whether to pass
  `floor=8` into the pyramid instead. **[open: Pass 4]** The duplicated floors (4 in `scattered/solver`, 8 in `tvf_adj` and the
  affine solvers) are expressible as `floor=` with the same planner and are removed when those callers migrate (Pass 3).
## 5. Smoothing
- Gaussian, per-axis sigma, given in millimetres or in voxels, applied only through `core/smoothing.separable_gaussian_filter`
  (replicate padding, axes with sigma <= 0 skipped). **[decided]**
- Never on an axis that is not coarsened (`mm_iso*`; the legacy presets smooth every axis when sigma > 0 because that is what the
  incumbents do); never NaN at sigma 0 (a zero-sigma axis is skipped, not a zero-width kernel). **[implemented]**
- Which sigma rule is the default (none beyond the area average, `log2(L)` voxels, `0.5 n/m`) **[open]**.

## 6. Operator presets **[implemented]**
Named, so that migrating a caller cannot change its numbers by accident (golden test per preset in `tests/test_pyramid_central.py`):

| preset | grid and operator | sigma rule | reproduces | use |
|---|---|---|---|---|
| `mm_iso` (default) | millimetre-isotropic: working spacing = level x finest spacing; centroid-exact area average on coarser axes, centre-to-centre linear on finer axes; matrix ops (MPS-safe) | none unless `sigma=` | `robust_affine._mm_pyramid_level` (bias fix, floor, own sigma) | rigid / affine / motion, anisotropic or thin images |
| `mm_iso_margin` | `mm_iso` + one-slice valid-region mask (`Level.valid`) on an anisotropic image | as above | the evidence's `P4+m` | the FIXED image of rigid / affine on a thin slab (Pass 4 turns it on by default in `robust_affine`) |
| `legacy_avgpool` | voxel `avg_pool` by the level; f = L, off = (L-1)/2 | none unless `sigma=` | `robust_affine._pool_level`, `motion_batched` (bit-identical at sigma 0) | only to keep an isotropic incumbent bit-for-bit |
| `legacy_interp` | Gaussian sigma = log2(L) voxels, then `F.interpolate` align_corners to `floor(n / L)` | `log2(L)` | `pyramid.build_image_pyramid` (bit-identical) | syn / syngs / tvf (Pass 3) |
| `legacy_floor16` | shape `max(int(n / L), 16)`, compact Gaussian `0.5 n / m` (zero padding), align_corners | `0.5 n/m` | `build_anti_aliased_pyramid` / greedy (bit-identical); `floor=8, sigma=0` is `tvf_adj`, `floor=4, sigma=0` is `scattered/solver` | greedy, tvf_adj, scattered (Pass 3) |

Exact-reproduction exceptions (all deliberate, all tested): the incumbents' 0- or 1-voxel outputs on a thin axis become 2 (or the
native count); `legacy_avgpool` and `mm_iso` smooth with the replicate-padded ITK Gaussian of `core/smoothing` instead of the
zero-padded truncated Gaussian of `landmarks/blob._separable_gaussian3d` (agreement in the interior within 5e-3 on a smooth image,
border no longer darkened); a moving image gets its own sigma and size (greedy used the fixed image's sigma). JAX uses the same
plan (`build_pyramid_jax`, jax imported lazily): every operator is a separable weight matrix, agreement with torch to 1e-5.
The default preset for deformable models stays their current one (no deformable default changes). **[decided]**

## 7. Per-level outputs
For each level: the tensor `(B, C, *shape)`; per axis (x, y, z) the factor `f` and offset `off` such that working index j sits
at full-resolution continuous index `j f + off` (voxel centres; `f = L`, `off = (L-1)/2` for pooling, `f = (n-1)/(m-1)`,
`off = 0` for corner-aligned operators); the working `spacing`, `origin` and `shape`; a helper mapping working indices to
physical coordinates; and one shared `warp_to_moving_norm(y_phys, level_entry)` returning normalised grid coordinates of the
(possibly differently leveled) moving tensor. **[decided]**

Implemented as `Level` (`tensor`, `f`, `off`, `spacing`, `shape`, `valid`, `mask`, `physical_coordinates()`), module-level
`warp_to_moving_norm(y_phys, level)` and `level_spacing(...)`. The two duplicated `level_spacing` functions (`syngs`, `syn_jax`)
and the solvers' `warp_to_moving_norm` closures are verified equal to the shared ones and are replaced in Pass 3.

## 8. Masks, margin
- A mask is leveled with the image's geometry and a linear (not nearest) operator, then thresholded by the caller. **[decided]**
- A valid-region margin excluding the outermost slab-edge slice(s) of the fixed image from the metric is a per-level boolean
  returned with the level (`Level.valid`; only for an anisotropic image, max/min spacing > 1.25). The evidence recommends it for an
  anisotropic fixed image of rigid/affine registration (`P4+m`: all ten real pairs within 2 mm of the ANTs pose against 8 of 10
  without it). Planner and mask exist in this module (`mm_iso_margin`); making it the default of `robust_affine` is Pass 4. **[decided;
  the switch is Pass 4]**

## 9. API summary (Pass 2) and status of the callers
`plan_levels(shape_zyx, spacing_xyz, levels, policy, floor, sigma, sigma_mode, max_voxels, reference_spacing_xyz, centre_to_centre,
align_corners)` (pure Python) -> `LevelPlan`s; `build_pyramid(image, spacing, levels, policy=..., sigma=..., floor=..., mask=...,
origin=..., direction=..., align_corners=...)` -> `Level`s; `build_pyramid_jax`; `PRESETS`, `PyramidPolicy`;
`MIN_VOXELS_PER_AXIS`, `HARD_MIN_VOXELS`; `area_matrix` / `linear_matrix`; `plan_resample` and `cap_ants_image`, the ANTs wrapper
behind `imaging_utils.cap_resolution_for_registration` (same signature and 16-voxel semantics; the thin-slab early return that
defeated the cap is fixed).
Pass 3b (deformable and remaining callers) migrated: `build_image_pyramid` is now a thin delegate of `build_pyramid(..., 'legacy_interp')`
(public name kept; used by `syn`, `syngs`, `tvf`); `greedy` uses `legacy_floor16` (each image its own sigma and size); `tvf_adj` uses
`legacy_floor16` with floor 8 and sigma 0; `syn_jax` uses `build_pyramid_jax` with `legacy_interp`, `syngs_jax` / `tvf_jax` use
`syn_jax.pyramid_level_jax` (`legacy_floor16` plan, floor 8, Bessel kernel); `syngs._level_spacing` and `syn_jax.level_spacing_itk`
are deleted for `pyramid.level_spacing`. `build_anti_aliased_pyramid` (no production caller) is deleted; both incumbents live on, verbatim,
as the test-only golden references `tests/legacy_pyramid_reference.py`.
Not migrated, on purpose: `scattered/solver.py` (its levels are explicit grid shapes or `max(4, round(n / factor))`, which no preset
reproduces: `round` differs from `int`, and a size-mode level is an arbitrary target shape, so the `F.interpolate(align_corners)` there is a resize to a
prescribed grid, not a pyramid level; the guard of Pass 4 should allow-list it); `benchmark/msd.py` and `dewarp.build_motion_reference`
(ANTs `resample_image` to a fixed spacing or to the finest spacing: not a mm cap, so `cap_ants_image` would return the image unchanged or change the
grid; labels must stay nearest neighbour). `robust_affine` / `motion_batched` are Pass 3a.

## 10. Known inconsistencies in deformable models, not changed
These change numerics and are not crashes, so they are recorded here for the owner (Pass 3b did not touch them).
1. `tvf.py:1085` `curr_spacing = [sp * level for sp in self.spacing]` against the align-corners grid used everywhere else: the level grid
   spans the same box with spacing `sp (N - 1) / (n - 1)` (`pyramid.level_spacing`; `tvf.py` uses it for the velocity grid a few lines above).
   Estimate: the factor is 1 at level 1 and `(N-1)/(n-1) / level` otherwise: 1.024 for 128 -> 32 (level 4), 1.059 for 128 -> 16 (level 8),
   about 1.05 for 64 -> 16; larger where the floor of 8 voxels binds. It scales the CFL / regularisation normalisation `sp_t_*` on coarse levels by that
   amount (a few percent step size), nothing at the finest level.
2. `syn.py:1092` `shrink_ratio = curr_spatial[0] / original_spatial[0]` from the z axis only (the CFL step is `cfl_voxels * sqrt(shrink_ratio)`).
   Isotropic shapes: identical to every axis. Anisotropic or thin z: it follows the z shrink, which the voxel-wise `floor(n / level)` rounds
   unevenly (e.g. 11 slices at level 4: 2 / 11 = 0.18 against 0.25 in x, y: a step 14 percent smaller; with the central thin-axis floor a 3-slice axis gives 2 / 3,
   a step 64 percent larger than the in-plane shrink). Using the minimum over axes, as `scattered/solver.py` does, would be the consistent choice.
3. `tvf_adj.py:329` `curr_spacing = sp * (N / n)` against the align-corners `(N - 1) / (n - 1)` of the images and velocity: 4.0 against 4.2 for 64 -> 16
   (5 percent), 1.0 against 1.0 at the finest level; it scales the flow / total sigmas (in level voxels) on coarse levels. The module is a deprecated prototype.
4. `syn_jax` / `tvf_jax`: a moving image of another shape is resampled to the FIXED level grid (kept as is by the migration, `smooth_resample_to_jax`);
   the PyTorch `syn` / `tvf` give each image its own level grid. `syn_jax.py` then pairs `moving_spacing` with that fixed-sized level.
   Not measured separately; the JAX `syngs` raises for differing shapes.
5. `landmarks/blob._separable_gaussian3d` returns NaN for a sigma of 0 (the zero-sigma bug of the plan). It is not used by any deformable model
   (the central module and `build_image_pyramid` skip sigma <= 0 and never produce NaN: tested), but `robust_affine` (Pass 3a) and the landmark code still call it.
6. `syngs_jax.GeodesicShootingModelJAX.fit` leaves `velocity_0_fwd` at exactly 0 in the configurations tried here (32^3 pair, cfl optimiser): the
   gradient seen by `apply_green_operator` printed a maximum of -inf in a debug probe. Not investigated (outside the pyramid); it means pyramid equality there was measured on the level images fed to
   `forward`, not on the final field.
