# The central pyramid: one contract

Status: **final (6.2.0). `src/syntx/pyramid.py` is the only pyramid implementation and every caller uses it (Sections 11, 12); the defaults are decided on measured evidence (Section 13): `robust_affine` levels BOTH images in millimetres whenever either is anisotropic and drops the slab-edge slice of an anisotropic fixed image from the metric (`valid_margin_slices=1`); `tests/test_pyramid_guard.py` fails if a new pyramid copy appears. One synthetic case (6:1, pose A) remains a documented limitation (Section 13.3).**
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

## 6. Operator presets **[implemented; Section 6 table superseded by Section 12: only `mm_iso`, `mm_iso_margin`, `deformable` exist]**
Named, so that migrating a caller cannot change its numbers by accident (golden test per preset in `tests/test_pyramid_central.py`):

| preset | grid and operator | sigma rule | reproduces | use |
|---|---|---|---|---|
| `mm_iso` (default) | millimetre-isotropic: working spacing = level x finest spacing; centroid-exact area average on coarser axes, centre-to-centre linear on finer axes; matrix ops (MPS-safe) | none unless `sigma=` | replaced `robust_affine._mm_pyramid_level` (deleted in Pass 3c; its overlap-box average was biased at fractional factors) | rigid / affine / motion, anisotropic or thin images |
| `mm_iso_margin` | `mm_iso` + one-slice valid-region mask (`Level.valid`) on an anisotropic image | as above | the evidence's `P4+m` | the FIXED image of rigid / affine on a thin slab (Pass 4 turns it on by default in `robust_affine`) |
| `legacy_avgpool` | voxel `avg_pool` by the level; f = L, off = (L-1)/2 | none unless `sigma=` | the isotropic-image levels of `robust_affine` and `motion_batched` (bit-identical to the former `_pool_level` at sigma 0) | only to keep an isotropic incumbent bit-for-bit |
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
  returned with the level (`Level.valid`; only for an anisotropic image, max/min spacing > 1.25). **[decided and implemented in Pass 4:
  `robust_affine(valid_margin_slices=1)` is the default; 0 disables; the width 1 was chosen from a 0 / 1 / 2 sweep, Section 13.1.]**

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

## 10. Known inconsistencies in deformable models (all resolved in Pass 3d, Section 12.3)
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
5. `landmarks/blob._separable_gaussian3d` returned NaN for a sigma of 0. **Fixed in Pass 3c** (an axis with sigma <= 0 is skipped; test `test_blob_gaussian_zero_sigma_is_identity_not_nan`). Its zero padding (border darkening) is unchanged: DoG / SIFT code still uses it, the pyramids do not.
6. `syngs_jax.GeodesicShootingModelJAX.fit` left `velocity_0_fwd` at exactly 0. **Investigated and fixed in Pass 3c**, see Section 11.4.

## 11. Pass 3c: the affine and motion callers, on the corrected pyramid
Owner's brief for this pass: correctness and consistency over backward compatibility. The old solver pyramid was not preserved.

### 11.1 What changed
- `robust_affine._mm_pyramid_level` and `_axis_resample_matrix` are deleted (about 85 lines), as are `_pool_level` and the 2-D `_gaussian_kernel_2d_sep`. The solver's levels for an
  anisotropic image come from `build_pyramid(..., 'mm_iso', sigma=sig_mm, sigma_mode='physical', reference_spacing=fixed.spacing, max_voxels=...)`; isotropic images from `legacy_avgpool`
  (bit-identical at sigma 0, as in Pass 3a). The anisotropic phase-correlation downsampler uses `mm_iso` with `centre_to_centre=False` and the new `n_out_xyz` (one forced cell-aligned grid
  for two images of different native grids). No `legacy_mm_box` preset exists.
- Centroid-exact area average replaces the overlap-weighted box average (integer factors are identical; a fractional factor was biased by up to 0.0625 working voxel; level f = 1.087 of sub-01 differs by up to 57 intensity units).
- `smooth_sigma_per_level` > 0 now smooths through `core/smoothing` (replicate padding, per-axis sigma in voxels for isotropic images and in millimetres for anisotropic ones). The default (0) path is untouched. The old zero-padded kernel darkened the border.
- `warp_to_moving_norm` is shared: `pyramid.prebuild_warp(origin, direction, native_spacing, f, off, shape_zyx, device, dtype)` builds its constants once (the solver builds one per level; `motion_batched` one per level
  through a small cache); `pyramid.warp_to_moving_norm(y, level)` caches them on the `Level`. The inline copies in `robust_affine` and `motion_batched` are gone. `motion_batched` now uses the level's own
  per-axis `f` and `off` (the old code assumed `f = level`, wrong for an axis shorter than the level).
- `cap_ants_image` accepts 2-D images; the 2-D multi-start branch no longer calls `ants.resample_image(..., interp_type=1)`. `ants.resample_image_to_target(moving, fixed, interp_type=1)` (nearest neighbour) in the
  isotropic phase-correlation branch is now linear (`interp_type=0`).
- Guard: `tests/test_pyramid_migration_affine.py::test_no_private_pyramid_or_downsample_code_in_the_affine_and_motion_callers`. Allow-listed: `ants.resample_image_to_target` (moving onto the fixed or a fine template grid: a registration
  grid, not a pyramid level) and the `_level_builder` seam (benchmark harness only).

### 11.2 Accuracy and time, default path, CPU (before = a40cd5e, after = this pass)
`benchmarks/pyramid_compare.py real|synthetic --policies DEFAULT --device cpu`, synthetic seeds 7 42 123, poses A D E F; real: ds001242 sub-01..10, seed 42.

| synthetic ratio | n | mean TRE before / after (mm) | max TRE before / after | cases > 2 mm before / after |
|---|---|---|---|---|
| 1 | 12 | 0.054 / 0.053 | 0.12 / 0.11 | 0 / 0 |
| 2 | 12 | 0.109 / 0.110 | 0.30 / 0.30 | 0 / 0 |
| 4 | 12 | 0.915 / 0.540 | 2.77 / 1.05 | 3 / 0 |
| 6 | 12 | 0.717 / 1.398 | 1.75 / 4.56 | 0 / 3 |
| 8 | 12 | 1.267 / 0.594 | 3.33 / 1.03 | 3 / 0 |
| 10 | 12 | 7.834 / 0.474 | 30.55 / 0.86 | 3 / 0 |

Real NM pairs, max pose error against the ANTs rigid reference (mm): sub-01 **4.40 -> 0.32** (z error 3.21 -> 0.09; native MI -0.5947 -> -0.5836 against the ANTs pose -0.5799), sub-07 26.10 -> 26.10 (an existing failure, unchanged), the
other eight identical to the printed digits (0.08, 0.19, 0.55, 0.49, 0.35, 0.12, 0.12, 0.13). Run time (sum over the ten pairs, loaded machine): 297 s before, 279 s after. Clean A/B on sub-02, three repeats alternated twice: before 26.0 to 29.0 s
(mean 27.7), after 26.5 to 27.7 s (mean 27.0), i.e. no slowdown (the shared warp measured 44 percent slower before it was prebuilt).

Where results changed and why: ratios 4, 8 and 10 and sub-01 improved because the anisotropic phase-correlation candidates and the solver levels are now centroid-correct (the biased box average shifted the thin-slab image by a fraction of a working voxel
on exactly the fractional levels). **One regression**: ratio 6, pose A, all three seeds (the case is deterministic), 1.75 -> 4.5 mm. Diagnosis: the solver's coarse loss at the TRUE pose is identical under both pyramids (-0.1902), so the objective did not get worse; the
optimiser ends in a worse local minimum (-0.049 against -0.146 for the old operator) because the phase-correlation peak ordering changed (the candidate near the true shift moved from rank 2 to rank 4; `topk=5` recovers only part, 2.95 mm).
This is search fragility, not a pyramid error; it stays open for Pass 4 (candidate set / valid-region margin), and is recorded here rather than tuned away.

Record of what changed (`benchmarks/pyramid_migration_identity.py`, 19 cases a40cd5e against this pass): 6 identical (`iso_rigid`, `slab_2to1_rigid`, `phase_corr_iso`, `real_sub-07_phase_corr`, `motion_batched_group_bias`, `motion_batched_wide`);
13 differ: the anisotropic phase correlations (corrected area average; max 16 to 25 mm in a candidate translation of one peak), `phase_corr_iso_other_grid` (36 mm: moving now resampled linearly onto the fixed grid, not nearest neighbour), `real_sub-01_*` (3.3 mm), `slab_4to1_rigid` (2.2 mm),
and numerically small ones (`slab_1to1`, `slab_8to1`, `iso_affine_smoothed` 0.1 mm from the central smoother, `real_sub-07_rigid` 6e-4, `motion_batched_*` <= 9e-3: float32 warp constants now built in float64).

### 11.3 Other changes in this pass
- `landmarks/blob._separable_gaussian3d`: zero sigma no longer NaN (skip axes with sigma <= 0).
- `benchmarks/pyramid_compare.py` P4 now calls the central `mm_iso` (results recorded before Pass 3c are not bit-comparable); `benchmarks/pyramid_migration_identity.py` is a record of what changed, not a gate.

### 11.4 `GeodesicShootingModelJAX.fit` leaving `velocity_0_fwd` at 0
Present on c68b98e and on a40cd5e (same probe, 32^3 pair, CFL optimiser): every gradient handed to `apply_green_operator` is NaN from the first epoch (the earlier `-inf` print was a different symptom of the same NaN), so the CFL guard
`max_g > 1e-8` is False and the field is never updated. Cause: `shoot` computed `|v|` with `jnp.linalg.norm`, whose derivative is 0/0 at the initial velocity v = 0; `jnp.where` does not stop that NaN from reaching the gradient. Fixed locally (a safe
magnitude with a defined gradient at 0). With the NaN gone a second latent bug appeared: the CFL momentum, carried across levels, was not resampled with the field (shape mismatch at the second level); it is now resampled. After the fix the 32^3 probe ends with
max |v| = 3.0 (finite). Consequence: every earlier JAX `syngs` result was effectively affine-only plus a zero field; they will differ now. Test: `test_syngs_jax_gradient_at_zero_velocity_is_finite_and_fit_moves_the_field`.
Not related to `test_syngs_jax_fails_loudly_for_what_it_does_not_implement`, which fails earlier, at `syn_metric='box_lncc'` not raising `ValueError` (still failing, unchanged).

## 12. Pass 3d: one convention for every deformable model
Owner's brief: correctness and consistency over backward compatibility. The legacy presets reproduced known-wrong behaviour and are deleted; no golden-equality test remains.

### 12.1 Design (one planner, one smoother, one floor, one bookkeeping)
- **One planner**: `plan_levels` / `build_pyramid`. Level L has the working spacing `L x (finest spacing of the fixed image)`, isotropic in millimetres, never below the image's own spacing. A moving image is planned on its own grid with the fixed image's `reference_spacing`.
- **The only per-model difference is the grid convention of the field the model estimates**, a parameter of the planner (`align_corners`, carried by the preset): `False` cell-aligned (`f = N/n`, `off = (f-1)/2`; affine, motion, `mm_iso`) and `True` corner-aligned (`m = round((N-1) s / t) + 1`, `f = (N-1)/(m-1)`, `off = 0`; every deformable model, whose warp / velocity grid spans the same box). The preset `deformable` = corner-aligned + `resample_finer=False` (an axis that is not coarsened is kept: level 1 is exactly the native grid, so a warp lives on it and a thick axis is never up-sampled) + the floor.
- **Operator by use**: deformable image pyramids = anti-aliased Gaussian then linear interpolation (standard; the linear operator alone aliases, measured in `test_alias_suppression_nyquist_pattern_and_constant_preserved`: an alternating pattern survives point sampling at amplitude about 1 and is suppressed below 0.05 by the smoother); millimetre pyramids of the affine solvers = centroid-exact area average (the box average is itself the anti-alias filter, so no extra Gaussian).
- **One smoother**: `core/smoothing.separable_gaussian_filter` (ITK Bessel kernel, replicate padding, per-axis sigma, zero-sigma axes skipped). The zero-padded compact kernel (greedy) and the per-copy JAX kernels are gone (JAX uses the same Bessel kernel and edge padding).
- **One sigma rule** (default): sigma = 0.5 x the axis's actual coarsening factor, in native voxels, only on axes that are coarsened by the linear operator. It replaces `log2(level)` on every axis (syn, tvf, syngs, JAX), `0.5 n/m` with zero padding (greedy) and none (tvf_adj, scattered). The two agree at factors 2 and 4 (1 and 2 voxels); at factor 8 the new rule gives 4 against 3. User `smoothing_sigmas` still overrides (native voxels, coarsened axes only).
- **One floor**: `MIN_VOXELS_PER_AXIS = 16` for the deformable preset (greedy's value; syn / tvf / syngs had none or 8, JAX 8, tvf_adj 8). An axis shorter than the floor keeps its native samples (never up-sampled: greedy used to up-sample it).
- **One spacing bookkeeping**: the spacing that feeds CFL, regularisation, Jacobian and step normalisation of a grid is `level_spacing(native spacing, native shape, level shape)` = `Level.spacing` (`s (N-1)/(n-1)`); the image level and the velocity / warp grid are the same planned grid (`TVFModel` and `GeodesicShootingModel` take the velocity grid from the planner; JAX `syngs`/`tvf` likewise). The CFL shrink ratio is `pyramid.cfl_shrink_ratio` (minimum over axes of finest / level spacing).
- `build_image_pyramid` is **deleted** (its list-of-tensors form cannot carry the per-level spacing the models need); callers use `build_pyramid(..., 'deformable')` and its `Level`s. `build_pyramid` and `plan_levels` are exported from `syntx`. `scattered/solver.py` is left alone: its levels are caller-prescribed grid shapes or `max(4, round(n / factor))`, a resize to a prescribed grid, not a pyramid level of an image.

### 12.2 Deleted
`legacy_interp`, `legacy_floor16`, `legacy_avgpool`, the `grid` / `operator` / `sigma_rule` / `kernel` / `sigma_clip` / `floor_upsamples` policy fields, the `avgpool`, `interp`, `pool` operators and the compact-kernel JAX path (`pyramid.py` 679 -> 590 lines, with the new planner features and docs), `build_image_pyramid`, `syn_jax.smooth_resample_to_jax`, `_JAX_FLOOR8_POLICY`, `tests/test_pyramid_migration_models.py` (golden tests), the golden tests of `test_pyramid_central.py`; `tests/legacy_pyramid_reference.py` moved to `benchmarks/` (only the historical Pass-1 harness `pyramid_compare.py` still reads it). `robust_affine` and `motion_batched` use `mm_iso` for isotropic images (one-token change at 7 sites; an integer factor is the same box average, a non-divisible shape now covers the whole field of view instead of dropping the remainder).

### 12.3 Fixed inconsistencies, one regression test each (`tests/test_pyramid_deformable.py`)
1. `tvf.py` `curr_spacing = sp * level` and the cell-aligned `vel_spacing = sp N/n`: one corner-aligned spacing of the velocity grid.
2. `syn.py` / `syn_jax.py` `shrink_ratio` from the z axis only: `cfl_shrink_ratio` over all axes (on an 11-slice 3 mm slab at level 4 the z-only ratio was 1.0, no shrink, against 0.26 for the in-plane axes).
3. `tvf_adj.py` `sp * N/n`: corner-aligned spacing; images through the central pyramid (now smoothed).
4. JAX `syn` / `tvf` / `syngs`: each image on its own level grid (the moving image was resampled onto the FIXED level grid but paired with its own spacing).
5. `tvf_jax` resized velocity keyframes with `jax.image.resize` (half-pixel) while the level spacing is corner-aligned: now `interpolate_jax` (align corners); a ramp's end values are exact (the old resize was off by about 0.5 voxel).
6. greedy: a level on an anisotropic image shrank every axis by the level; now millimetre-isotropic via the optional `spacing` / `moving_spacing` arguments of `GreedyRegistrationModel.fit` (the wrapper passes the image spacings); the warp grid is the fixed level grid.
7. `syngs_jax` default `smooth_pyramid` was False (aliased levels), now True like every other model.
8. `syngs` / `tvf` kept the velocity grid (floor 8) and the image level (no floor) on different shapes; both come from one plan.

### 12.4 Numeric changes and their reasons (release notes)
- Level shapes are `round((N-1) s/t) + 1`, not `floor(N/L)`: 128 at level 4 gives 33, not 32 (spacing 3.97 mm, not 4.03). Floor 16 binds earlier than before on small test images.
- Anisotropic images: levels are millimetre-isotropic, the thick axis is not shrunk with the thin ones.
- Default smoothing 0.5 x factor (see above), applied to the moving image with its own factor.
- `tvf` normalisation `sp_t` and regularisation spacing at coarse levels change by a few percent (1.024 at 128 -> 32 by the old estimate).
- JAX `syngs` smooths the pyramid by default.

### 12.5 Accuracy and time, before (HEAD 571bf09) against after, CPU, same machine back to back
Known-warp harness: a textured volume with six ellipsoid labels, moving image = the volume warped by a smooth displacement of about 5 voxels (mm: 5 x spacing in plane); identity initial transform; iterations [40, 30, 20]; Dice of the model-warped moving labels against the fixed labels (before registration 0.61 to 0.64), Jacobian of the forward warp. Cases (x, y, z shape / spacing): iso 64^3 / 1 mm; aniso 64x64x32 / (1, 1, 2); thin 64x64x14 / (1, 1, 3); aniso_inplane 48x60x40 / (1.5, 1, 1). tvf rows use the CFL fix of Section 12.6 on both sides (without it TVF does not move, Dice unchanged, on either tree).

| case | model | Dice before / after | fold % before / after | time s before / after |
|---|---|---|---|---|
| iso | syn | 0.962 / 0.963 | 0.052 / 0.045 | 17.0 / 17.6 |
| iso | syngs | 0.964 / 0.964 | 0 / 0 | 34.3 / 35.9 |
| iso | greedy | 0.957 / 0.957 | 0.098 / 0.118 | 22.2 / 21.1 |
| iso | tvf | 0.955 / 0.955 | 0 / 0 | 48.7 / 52.1 |
| aniso | syn | 0.957 / 0.957 | 0.076 / 0.072 | 10.3 / 11.4 |
| aniso | syngs | 0.954 / 0.955 | 0 / 0 | 16.3 / 20.0 |
| aniso | greedy | 0.952 / 0.955 | 0.119 / 0.129 | 11.0 / 16.8 |
| aniso | tvf | 0.949 / 0.951 | 0 / 0 | 25.6 / 28.0 |
| thin | syn | 0.935 / 0.938 | 0.141 / 0.131 | 6.3 / 8.9 |
| thin | syngs | 0.866 / 0.910 | 0 / 0 | 9.9 / 13.5 |
| thin | greedy | 0.925 / 0.923 | 0.075 / 0.096 | 9.3 / 10.8 |
| thin | tvf | 0.908 / 0.922 | 0 / 0 | 19.6 / 14.9 |
| aniso_inplane | syn | 0.957 / 0.951 | 0.030 / 0.042 | 17.4 / 15.4 |
| aniso_inplane | syngs | 0.953 / 0.953 | 0 / 0 | 22.3 / 26.3 |
| aniso_inplane | greedy | 0.944 / 0.947 | 0.0 / 0.01 | 15.8 / 18.9 |

2-D real images (r16 -> r64, `benchmark.tune.twod_evaluator`, 3-class Otsu Dice, the method's defaults): greedy 0.485 / 0.490, syn 0.786 / 0.779, syngs 0.790 / 0.797, tvf 0.794 / 0.796 (folding 0 for all but greedy before, 0.03 percent). `tvf_adj` (deprecated prototype) does not register in either tree (Dice 0.02 to 0.04 on every case, 14 to 40 percent folded); only its spacing bookkeeping was fixed and tested.

Reading: no accuracy regression beyond noise on the isotropic case; thin-slab `syngs` (+0.044) and `tvf` (+0.014) improve because the thick axis is no longer shrunk with the in-plane axes. Two small decreases are recorded, not tuned away: 3-D syn on `aniso_inplane` 0.957 -> 0.951 (an alternative sigma schedule, 2/1/0 or 1.5/0.75/0, gives 0.950 / 0.952, so the sigma rule is not the cause; the grids differ at the 1.5 mm axis) and 2-D syn 0.786 -> 0.779. Time rises 10 to 40 percent on the anisotropic cases because the thick axis stays at its native sampling (more voxels at coarse levels) and the 16-voxel floor binds earlier; the isotropic case is within the noise of a shared machine.


### 12.6 One more fix found by the validation: TVF's CFL scale was clamped at 1
`tvf.py` computed `cfl_scale = (cfl_step / max_voxel_gradient).clamp_max(1.0)`; ITK's `ScaleUpdateField` scales the update so its maximum length equals the step (up or down). With the clamp a small gradient was never scaled up: `max |v|` stayed at 0.0015 and `TVFModel` returned the identity on every case (2-D r16 -> r64 Dice 0.563, the known failure of `test_tvf_defaults_on_2d_pair`; 3-D Dice unchanged from the unregistered 0.64). Without the clamp: 0.796 in 2-D (guard 0.785, folding 0) and 0.92 to 0.955 in 3-D. Introduced in a5362ac; regression test `test_tvf_cfl_step_is_scaled_up_to_the_step_length` and the existing guard.


## 13. Pass 4: defaults switched on evidence, guard, release
Owner's brief: correctness and consistency over backward compatibility; no silent fallback; decide on measurements, report negative results.
All numbers: `benchmarks/pyramid_compare.py synthetic|real` (DEFAULT path through `robust_affine`, `mode='pytorch'`, rigid, CPU unless stated), synthetic = ratios
1, 2, 4, 6, 8, 10 x poses A, D, E, F x seeds 7, 42, 123 (72 cases, TRE in mm over 100 random points), real = ds001242 sub-01..10 (max pose error in mm over 9 probe
points against the ANTs rigid reference, seed 42). "Before" is 70ee29f, "after" is the 6.2.0 code.

### 13.1 What the default is now, and the margin width
1. **Both images are leveled in millimetres whenever either is anisotropic** (`mm_iso`, working spacing `level x` the fixed image's finest spacing, never finer than the image's
   own spacing). Before, only an anisotropic moving image got it: an isotropic 1 mm T1 was pooled by `level` voxels, 2 to 3 times coarser than a 0.43 mm in-plane slab at the same
   level. This was the difference between the harness policy `P4+m` (sub-07 1.2 mm) and the shipped default with the margin only (6.5 mm); with the fix the two are identical (1.2376 mm).
2. **`valid_margin_slices=1`** (new documented parameter; 0 disables; a negative or non-integer value raises `ValueError`): the outermost slice at each end of the thick axis of an
   anisotropic fixed image does not enter the Mattes metric. A no-op for an isotropic fixed image (bit-identical transforms, tested).

Margin sweep (new code, everything else equal; mean / worst TRE and cases under 1.5 mm; the last column of the first table is the 70ee29f default = margin 0 with the old moving level):

| ratio (12 cases each) | before (70ee29f) | margin 0 | **margin 1** | margin 2 |
|---|---|---|---|---|
| 1:1 | 0.05 / 0.11 / 12 | 0.05 / 0.11 / 12 | 0.05 / 0.11 / 12 | 0.05 / 0.11 / 12 |
| 2:1 | 0.11 / 0.30 / 12 | 0.09 / 0.22 / 12 | 0.10 / 0.28 / 12 | 0.11 / 0.29 / 12 |
| 4:1 | 0.54 / 1.05 / 12 | 0.51 / 1.00 / 12 | 0.52 / 1.02 / 12 | 0.51 / 0.96 / 12 |
| 6:1 | 1.40 / 4.56 / 9 | 1.38 / 4.48 / 9 | 1.38 / 4.45 / 9 | 1.39 / 4.61 / 9 |
| 8:1 | 0.59 / 1.03 / 12 | 0.60 / 1.02 / 12 | 0.62 / 1.00 / 12 | 0.66 / 1.06 / 12 |
| 10:1 | 0.47 / 0.86 / 12 | 0.48 / 0.89 / 12 | 0.67 / 1.16 / 12 | 0.89 / 1.75 / 10 |

| real pair | before (70ee29f) | margin 0 | **margin 1** | margin 2 |
|---|---|---|---|---|
| sub-01 | 0.32 | 0.32 | 0.39 | 0.30 |
| sub-02 | 0.08 | 0.08 | 0.06 | 0.11 |
| sub-03 | 0.19 | 0.19 | 0.14 | 0.08 |
| sub-04 | 0.56 | 0.32 | 0.31 | 0.39 |
| sub-05 | 0.49 | 0.50 | 0.19 | 0.24 |
| sub-06 | 0.36 | 0.28 | 0.34 | 0.30 |
| sub-07 | **26.24** | **26.78** | **1.24** | **4.06** |
| sub-08 | 0.13 | 0.10 | 0.11 | 0.14 |
| sub-09 | 0.11 | 0.10 | 0.29 | 0.20 |
| sub-10 | 0.13 | 0.12 | 0.14 | 0.14 |

Reading: the margin is decided by the real data. Without it sub-07 is 26 mm wrong (the partial-volume edge slice has no counterpart in the T1 and drags the pose along z); one slice fixes it
(1.24 mm); two slices over-trim (sub-07 4.06 mm, above the 2 mm bar; synthetic 10:1 worst 1.75 mm). **Negative result, recorded:** on the synthetic slabs the margin is not neutral at 10:1
(mean 0.48 -> 0.67 mm, worst 0.89 -> 1.16, six cases worse by 0.2 to 0.6 mm; all still under 1.5 mm): the synthetic slab has no partial-volume edge slice, so the margin only removes 2 of 7 slices
of valid data. The 6:1 pose A failure is untouched by the margin (it is a search problem, Section 13.3). sub-01 / sub-09 move by 0.07 / 0.19 mm: noise of a near-degenerate landscape.

### 13.2 Before / after, default path (all 72 synthetic cases and ten real pairs)
After = margin 1. Synthetic mean / worst TRE (cases under 1.5 mm): 1:1 0.05 / 0.11 (12); 2:1 0.10 / 0.28 (12); 4:1 0.52 / 1.02 (12); 6:1 1.38 / 4.45 (9); 8:1 0.62 / 1.00 (12); 10:1 0.67 / 1.16 (12).
Real: all ten under 2 mm (worst 1.24 mm, sub-07; sub-07 before 26.2 mm). The final DEFAULT path run on CPU and on MPS gives the same pose errors for all ten pairs (mm): 0.39, 0.06, 0.14, 0.31, 0.19, 0.34, 1.24, 0.11, 0.29, 0.14 (MPS 0.38 for sub-01), 26 to 33 s per pair on CPU and 9 to 11 s on MPS. Isotropic accuracy and time are unchanged (iso pair TRE identical to 7 digits; mean of the median run times 2.37 s before, 2.17 s after, back to back, CPU, shared machine: no slowdown).
Original-input verification (`/Users/stnava/data/nm_registration_debug/repro_syntx.py --device mps`): VERDICT OK, pose error max 0.4 mm (mean 0.2, z +0.1), 10.1 s, no fallback line.

### 13.3 Thick-axis / translation capture search (bounded effort, three attempts) and the decision
The one synthetic failure left after the pyramid fix is 6:1, pose A (1.75 -> 4.4 mm, deterministic, all seeds). Probing the capture range from the TRUE pose (`initial_transform` with a single perturbation, `multi_start=False`): rotation
errors up to 10 degrees about any axis converge (TRE 0.15 mm), but a translation error of 2 mm still converges (0.2), 3.5 mm gives 1.6, 5 mm gives 3.0 and 7 mm gives 5.0, along z AND along x and y (a coarse voxel at
level 4 is 3 mm). The true translation of pose A is 7.5 mm from the CoM, and no phase-correlation or CoM candidate (75 of them) lies within ~2 mm of it, whatever the rotation. So the failure is translation capture, not specific to the thick axis.
1. *Thick-axis scan*: every start also at +-1..4 half-slice steps along the thin axis (675 candidates). 6:1 pose A 4.37 -> 2.86 mm; with `n_starts` 6, 2.81 mm. Not enough (in-plane offsets are as far off as z). Dropped.
2. *Translation lattice scan* (`translation_scan_steps`, implemented, opt-in): score a lattice of +-k coarse-voxel translations (k = 1, 2) along the fixed image's axes around the CoM and phase-correlation starts (identity rotation, coarse level), add the two best
   well-separated points as bases of the rotation-cone sweep. Synthetic (margin 1): 6:1 12 of 12 (mean 1.38 -> 0.20, worst 4.45 -> 0.36 for k = 2), 4:1 0.52 -> 0.24, 8:1 0.62 -> 0.29, 20 of 72 cases better, 10:1 slightly worse (0.67 -> 0.71 mean; 4 cases worse by 0.1 to 0.27 mm). **Real: sub-07 1.24 -> 7.81 mm, sub-01 0.39 -> 1.47 mm, sub-09 0.29 -> 0.66; time +40 percent** (38 to 44 s against 26 to 28 s).
   The sub-07 solution has a deeper coarse-level loss than the ANTs pose and than the default's (-0.520 against -0.505) but a worse policy-independent native MI (-0.320 against -0.388; ANTs pose -0.364): a wider search finds deeper minima of the coarse objective that are not the right pose on real cross-modal data. It violates the acceptance rule (every real pair kept), so it is **not the default**; it stays as the opt-in parameter, with a test that it solves 6:1 pose A.
3. *More refined candidates* (`n_starts` 5, 8 without the scan; 6 with it): 6:1 pose A 4.41 / 4.71 mm; sub-07 with the scan and `n_starts` 6 stays 7.81 mm (the deeper wrong minimum wins the selection). No help.
Decision: no capture-search change is the default. **Documented limitation:** a thin-slab start whose translation is more than ~2 coarse voxels from every CoM / phase-correlation candidate is not recovered (6:1 pose A, 4.4 mm); `translation_scan_steps=2` recovers it on the phantom and is available, but is unsafe on the real NM pairs. The test is a strict xfail with this reason.
Open: the coarse loss of the true pose (-0.19) is much better than the best start (-0.015), so the metric itself could rank a lattice point correctly; what is missing is a lattice objective that does not prefer the spurious deep minimum on real data (for instance a native-MI re-ranking of the lattice winners).

### 13.4 Tests and the guard
- `tests/test_robust_affine_anisotropy.py`: real assertions (TRE < 1.5 mm) at 1, 2, 4, 8, 10:1; a strict xfail for 6:1 pose A only; the opt-in scan test; parameter validation; margin no-op (isotropic) and effect (anisotropic); all ten real pairs within 2 mm of the ANTs reference (skipped when `/Users/stnava/data/nm_registration_debug/subjects` is absent).
- `tests/test_pyramid_guard.py`: scans `src/syntx` (comments and strings blanked) for `avg_pool*d`, `F.interpolate`, `jax.image.resize`, `zoom`, `ants.resample_image`, `resample_image_to_target`, definitions of a level spacing or a pyramid builder, the `(N-1)/(n-1)` spacing formula and `log2(level)` sigmas, outside `pyramid.py`.
  Every allowed occurrence has a count and a reason (box windows of LNCC / MIND / SSIM, prescribed grids of `scattered/solver.py`, field resizes, data loaders, `ants.resample_image_to_target` onto a registration grid, thin wrappers over `build_pyramid`); an allowance larger than the real count fails, so a copy cannot hide in the slack. Checked against a40cd5e it reports 12 violations (the earlier `ants.resample_image` caps, the in-line spacing formulas, `log2` sigmas, JAX resizes, a private pyramid builder) and against an injected `avg_pool3d` copy it reports 1.
- Four tests that were failing: `test_syngs_jax_fails_loudly_for_what_it_does_not_implement` (the JAX SyNGS accepted `box_lncc` since bfa9dea against its own docstring: now `ValueError`), `test_tvf_jax_metric_and_fail_loud` (stale: TVF-JAX implements `box_lncc` through the shared dispatch; the test now uses an unknown metric and checks `box_lncc` runs), `test_scattered_syn_edge_cases_and_robustness` and `test_scattered_syn_point_weights` (the post-solve self-check raised because identity beat the optimised pose: `Identity`, the self-check's own baseline, was not in the start pool; it is now).
- The four remaining in-line copies of `s (N-1)/(n-1)` (`syn.py`, `syn_jax.py`) call `pyramid.level_spacing`.

