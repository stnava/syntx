# Docstring audit (started 2026-10-01)

Goal: every function / class in `src/syntx` documented accurately and readably: what it does,
every parameter with its real default and scope, real return keys / shapes. Every claim is
checked against the code. Behaviour issues found along the way are **listed here, not fixed**
(documentation commits change no code: `python scripts/check_docs_only.py` proves each one).

Inventory at start: 80 modules, 853 definitions, 640 public, 308 public with < 15-word docstrings.

## Completed 2026-10-01 (v5.4.92)

Every module, public class and public function / method in `src/syntx` now has a docstring
checked against the code (nested closures and autograd `forward` / `backward` pairs are covered
by their enclosing docstrings). The issues below are documented, not fixed.

Most consequential (correctness of results or reports) -- **all fixed in v5.4.93**, each with a
fast CPU regression test in `tests/test_audit_fixes.py` that fails on the old code:

| # | Issue | Fix (commit) |
|---|---|---|
| 1 | `core/jacobian.py` 'bspline' derivative stencil signs (slopes x 5/3) | `[1,-8,0,8,-1]/12` (0295eab) |
| 2 | `viz/reports.py` constant 2.8 s affine time, hardcoded "88/90", "/ 90", static parameter table | every number from the records, n/a when missing, recorded `config` table (16cfec7) |
| 3 | `benchmark_data('mbhard')` wrote a synthetic pair under the Mindboggle names | FileNotFoundError; old placeholder rejected (1764197) |
| 4 | `benchmark/tune.py` cache key omitted the dataset | dataset in key, rows, result, default out_dir (62d4f92) |
| 5 | `policy.py` fields ignored by `auto_reg`; TVF policies raised TypeError (`similarity_metric`) | policy holds only applied fields; metric forwarded as `syn_metric` (002f246) |
| 6 | JAX: `tvf(backend='jax')` crashed (alpha=None); TVF sigmas mm vs voxels; extra affine stages; SyN `use_analytical_gradients` overridden and autograd path crashed | fixed (6f133bc). Claim "JAX SyN sigmas in voxels, PyTorch mm" was false: both use voxels (docs corrected) |
| 7 | `scattered/solver.py` inverse errors: reversed 3-D components, mixed units | one helper in the fields' convention (10ea6f8) |
| 8 | `spatial.jacobian_determinant` 2-D spacing swap, 3-D oblique directions | det(`deformation_gradient`) (ed96e57) |
| 9 | `syn` affine-only type + `initial_transform` returns it unchanged | **intended** (affine and deformable interfaces are separate); documented. Also: unknown `type_of_transform` silently ran SyN -> ValueError (3673eed) |
| 10 | `__all__` listed two missing names; import overwrote `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | removed; `setdefault` (a4eb117) |

Per-module lists below are the original findings; entries covered by the table are fixed.

## Order

1. Exported API (`syntx.<name>`): robust_affine, syn, tvf, syngs, greedy, liouville,
   deformation_metrics, motion, template, transform, diagnose / policy, generators,
   image_compare, viz.reports / figures, benchmark (evaluate, tune, high_level, runner, metrics),
   features, landmarks, scattered, surface, classifier, perf_tracking, provenance.
2. Core internals: core/* (grid, losses, smoothing, optimizers, inverse, affine, jacobian,
   pipeline, utils, mps_kernels), spatial, pyramid.
3. Backends / research: syn_jax, syngs_jax, tvf_jax, tvf_adj, motion_batched, scattered/*,
   landmarks/*, viz/*, data/*, benchmark internals, cli, contract.

## Status

| Module | Status | Commit |
|---|---|---|
| robust_affine.py (`robust_affine`, module) | done | v5.4.81 |
| syn.py (`registration`/`syn`, `auto_reg`, `SyNTo` + methods, helpers) | done | v5.4.82 |
| tvf.py (`TVFModel` + methods, `tvf_registration`, helpers) | done | v5.4.83 |
| syngs.py (module, `GeodesicShootingModel` + methods, `syngs_registration`, `integrate_momentum`) | done | v5.4.84 |
| greedy.py (module, `GreedyRegistrationModel`, `fit`, `greedy_registration`) | done | v5.4.85 |
| deformation_metrics.py | done | v5.4.86 |
| motion.py (`motion_correction`, `MotionParameters`, `TransformCollection`, result) | done | v5.4.87 |
| template.py (`build_template`) | done | v5.4.88 |
| transform.py (`SyNToTransform`) | done | v5.4.89 |
| pyramid.py | already accurate (no change) | -- |
| image_utils.py (`reflect_image`) | done | v5.4.90 |
| image_compare.py (`image_compare`; helpers already documented) | done | v5.4.90 |
| core/inverse.py (module, inverse solvers, inverse-error functions, velocity integration) | done | v5.4.91 |
| core/utils.py (module, `normalize_tensor`, `normalize_image`, percentile selection) | done | v5.4.91 |
| core/losses.py, smoothing.py, optimizers.py, mps_kernels.py | done | v5.4.92 |
| syn_jax.py, syngs_jax.py, tvf_jax.py, tvf_adj.py, motion_batched.py (+ tvf.py Catmull-Rom fix) | done | v5.4.92 |
| diagnose, policy, generators, classifier, resnet, perf_tracking, provenance, contract, reporting, tabulate, cli, __init__ | done | v5.4.92 |
| spatial.py, core/grid.py, core/affine.py, core/jacobian.py, core/pipeline.py, core/__init__.py | done | v5.4.92 |
| pyramid.py (module), robust_affine `_AffinePath` methods, `SyNJAX` class | done | v5.4.92 |
| viz/* (figures, reports, core, gallery, stats, modality_report, qc_sections, colormaps, __init__) | done | v5.4.92 |
| features.py, surface.py, landmarks/* | done | v5.4.92 |
| scattered/*, data/* | done | v5.4.92 |
| benchmark/* | done | v5.4.92 |

## Behaviour issues found -- status per item (2026-10-01, v5.4.95)

Every finding below carries a status: **FIXED** (with the commit; each fix has a fast CPU
regression test that fails on the code before it), **NOT-A-BUG** (verified intended / a
documentation error, now documented), or **DEFERRED** (needs a decision outside this audit:
the deprecated `tvf_adj` module, and antsxdwi's separate copy of `contract.py`).

### robust_affine.py
- **[FIXED bd285c6: removed (TypeError with a hint)]** `backend` parameter is unused.
- **[FIXED bd285c6: ValueError]** An unrecognised `mode` silently runs the ANTs path (should raise).
- **[FIXED bd285c6: 'fast' mode alias removed]** `mode='fast'` is the default solver, while `preset='fast'` is the fast schedule -- confusing names.

### syn.py
- **[FIXED d6ee6b3: warned only when flow_sigma is a gate (fast_smooth)]** Warning "flow_sigma ... has no effect on kernel shape with regularizer='sobolev'" is wrong for
  SyN: with the default `fast_smooth=False` flow_sigma is the Gaussian post-filter and its value
  changes the result (measured 2026-09-30).
- **[FIXED: TypeError (they belong to tvf / syngs)]** `cfl_momentum`, `multipoint_loss`, `n_time_steps`, `n_steps` are accepted but unused (inert
  parameters -- by the project rule they should raise).
- **[FIXED v5.4.95: the JAX fit gets the validated regulariser (default 'sobolev', as PyTorch)]** Default regulariser differs by backend: 'sobolev' (PyTorch) vs 'gaussian' (JAX).
- ~~An unrecognised `type_of_transform` ...~~ FIXED v5.4.93: it silently ran SyN; now ValueError.
- **[NOT-A-BUG: documented delegation ('greedy' / formulation='greedy' -> syntx.greedy)]** `type_of_transform='greedy'` silently returns `syntx.greedy`'s result.
- **[FIXED d6ee6b3: liouville folding (fd_* kept), mean_error / max_error read; docs say ANTsImage]** `auto_reg`: folding metric uses a finite-difference Jacobian, not `syntx.liouville_determinant`;
  reads `phi_1['mean'/'max']` but registrations return `mean_error` / `max_error` -> those
  metrics are NaN; claims of NumPy / tensor inputs are not supported by the code paths
  (ANTsImage needed).
- **[NOT-A-BUG (user decision)]** Affine-only `type_of_transform` + `initial_transform` returns the input transform: INTENDED
  (affine and deformable interfaces are separate); documented v5.4.93.

### tvf.py
- **[FIXED d6ee6b3: syntx.tvf exposes inverse_identity_weight]** `TVFModel.forward` adds an inverse-consistency penalty (weight `inverse_identity_weight`,
  fixed at 0.05) whenever `multipoint_loss` contains 0 and 1 -- a hidden setting `syntx.tvf`
  does not expose.
- **[NOT-A-BUG: low-level class option, documented; syntx.tvf does not use it]** `TVFModel(antisymmetric=...)` still exists (rewrites the evaluation times) although
  `syntx.tvf` removed the option.
- **[NOT-A-BUG: documented; only direct fit calls]** `TVFModel.fit` defaults (optimizer 'adam', similarity 'lncc', lncc_radius 4) differ from
  `syntx.tvf`'s ('cfl', 'cc2', 2) -- only matters for direct `fit` calls.

### syngs.py
- **[NOT-A-BUG: docs corrected (behaviour as documented)]** The module / class claimed EPDiff geodesics, minimal energy and det(J) > 0: the default
  'transport' mode integrates a stationary field, there is no energy term, and the true
  minimum determinant reaches ~3e-4 (docs corrected; behaviour unchanged).
- **[FIXED 6ec7188: removed; unknown keywords raise]** 16 accepted-but-unused `syngs_registration` parameters: total_sigma (stored as elastic_sigma,
  never read), type_of_transform, n_time_steps, cfl_momentum, multipoint_loss,
  sampling_percentage, vgg_* (5), project_inverse, projection_frequency, interpolator,
  inverse_method, inverse_steps. `fit`: reg_weight, fluid_sigmas, elastic_sigmas unused.
- **[FIXED 6ec7188: passed to the model]** `symmetric` / `inverse_identity_weight` cannot be set through `syntx.syngs` (passed kwargs
  reach `fit`, which ignores them).
- **[FIXED: unknown regularizer / optimizer_type raise ValueError]** `GeodesicShootingModel`: unknown `regularizer` silently becomes 'sobolev'; 'dsti' silently
  means 'dsti1'. `fit`: unknown `optimizer_type` silently uses LARS.
- **[FIXED 6ec7188: _level_spacing pairs ITK spacing with the reversed shapes]** Axis-order bug: `forward` / `get_forward_warp` / `get_inverse_warp` compute the per-level
  spacing as `zip(self.spacing, self.image_shape, target_shape)` -- ITK (x, y, z) spacing paired
  with tensor (z, y, x) shapes; wrong for anisotropic images at non-native resolution (the
  same bug fixed in tvf.py's vel_spacing on 2026-09-30).
- **[FIXED: one scheme (step t_end / n_steps), backend removed]** `integrate_momentum`: the trajectory / t_end != 1 path re-smooths every step (recursive),
  unlike the t_end = 1 path and the registration -> inconsistent endpoints; `backend` unused.

### greedy.py
- **[FIXED d6ee6b3: unknown metrics raise]** Unknown `similarity_metric` silently uses local correlation (only 'mse' / 'l2' differ).
- **[NOT-A-BUG: tuned default (tuned_greedy_2026_09_29), kept; observation recorded]** Default optimiser 'adam' normalises every voxel separately -- the velocity-roughness source
  found in TVF / SyNGS (2026-09-30); greedy folds 0.05 % (finite-difference) on r16 -> r64.
- **[NOT-A-BUG: docstring corrected]** The old docstring called learning_rate 0.375 an untested stand-in copied from syn; it was
  tuned for greedy (tuned_greedy_2026_09_29).

### deformation_metrics.py
- **[FIXED d6ee6b3: inputs untouched. 'nearestNeighbor' kept: a correct label interpolator; 'genericLabel' would change every recorded Dice]** `compute_bidirectional_dice` overwrites the geometry (origin / spacing / direction) of the
  label images passed in; it warps labels with 'nearestNeighbor' (ANTs recommends
  'genericLabel').

### motion.py
- **[FIXED 440dfc3: raises with the PyTorch backends]** `aff_metric` is used only by backend='ants' (ignored by the default PyTorch backends).
- **[FIXED 440dfc3: other keywords raise]** backend='pytorch_batched' (the 'auto' choice for 3D+t rigid) reads only `num_bins` from
  `**kwargs`; anything else is silently ignored.
- **[FIXED v5.4.95: Translation / QuickRigid / BOLDRigid raise on the PyTorch backends (backend='ants' implements them)]** backend='pytorch': type_of_transform='Translation' becomes a centre-of-mass shift
  (`robust_affine(mode='com_only')`), not an optimised translation; 'QuickRigid' / 'BOLDRigid'
  are plain rigid.

### template.py
- **[FIXED 4a48541: iteration-0 affine passed as initial_transform]** **Likely bug:** with the default `affine_every_iteration=False`, iterations >= 1 register each
  image with 'SyNOnly' but pass no initial transform (not the image's iteration-0 affine), so
  those registrations start from the identity -- wrong unless the inputs are already aligned
  to the template.

### image_utils.py
- **[NOT-A-BUG d6ee6b3: the axis is physical (doc error)]** `reflect_image` axis names 'LR' / 'AP' / 'SI' are fixed aliases for array axes 0 / 1 / 2,
  not resolved through the image's direction matrix -- wrong for images not stored
  axis-aligned in LPS / RAS order (compare `syntx.spatial.restriction_from_orientation`, which
  does resolve anatomical labels).

### core/inverse.py
- **[FIXED 00a3521: one stopping rule]** `update_inverse_field_nd` fixed-point solver: physical mode stops when max AND mean thresholds
  are met, normalised mode when EITHER is met.
- **[FIXED 00a3521: honoured / rejected; thresholds kept]** `relaxation` only affects `hybrid_lm`. Its fallback without `spacing` drops the thresholds.
- **[FIXED 00a3521: dt = 1/T, midpoint, unknown solver raises]** `integrate_time_varying_velocity_field`: `dt` is not derived from T (default 0.25);
  'midpoint' silently runs Euler in normalised mode; unknown `solver` values run Euler.
- **[NOT-A-BUG: both orders are valid and now documented; mean over evaluated voxels FIXED 00a3521]** `calculate_inverse_identity_error` and `compute_inverse_identity_error_nd` compose in opposite
  orders (inverse-then-forward vs forward-then-inverse). The former's `mean_error` counts
  out-of-grid voxels (error zeroed) in its denominator; the latter does no masking.
- **[FIXED 00a3521: checked (ValueError)]** Passing `X_phys` without spacing / origin / direction crashes (reverses None).

### core/utils.py
- **[FIXED 00a3521: zscore not short-circuited]** `normalize_image`'s idempotency shortcut overrides `method`: any input in [0, 1] with
  max >= 0.5 is returned unchanged, even for `method='zscore'`, unless `force=True`.

### core/losses.py
- **[NOT-A-BUG: a documented approximate (ANTs-style own-window) backward, used only with use_ants_pseudo_gradient]** AnalyticalLNCC.backward is not exact: keeps only the centre voxel's own window and uses (F_c - CC*M_c) instead of (F_c - CC*sqrt(var_I/var_J)*M_c); equal only for equal local variances.
- **[FIXED afd33b9: per-window voxel counts at borders; the mask zeroes the backward outside the active set]** ANTsPseudoLNCC / AnalyticalLNCC: N_window = window_size**dim even at border windows (pooling averages fewer voxels); ANTsPseudoLNCC saves `mask` and never uses it.
- **[FIXED afd33b9: even windows raise; unused pool_fn removed]** local_ncc_loss_nd: even window_size not larger than the image crashes (pooling gives n+1); `pool_fn` computed, unused.
- **[FIXED afd33b9: even kernels raise. Border padding / unclamped ratio / no mask / class default squared: NOT-A-BUG, documented -- the tuned 'cc2' metric of syn / greedy / tvf / syngs, so changing it would void the tuning]** BoxLNCCLoss: zero padding counted as data at the border; even kernel_size crashes; ratio not clamped; no mask; class default squared=True vs box_lncc_loss_nd squared=False.
- **[FIXED afd33b9: requested fraction, graph kept, mismatch raises, NaN samples weight 0]** mattes_mi_loss_core: stride int(1/p) (p=0.6 uses every voxel); empty input returns a disconnected leaf (no gradient); mismatched fixed_weights silently ignored; parzen_weights maps NaN to bin 0 silently.
- **[FIXED afd33b9: scale-relative auto_mask, fixed_weights misuse raises]** mattes_mi_loss_nd: auto_mask threshold fixed at 0.01 regardless of intensity scale; fixed_weights silently recomputed when the masked set changes.
- **[NOT-A-BUG: documented soft (non-Euclidean) map and sigma clipping]** compute_soft_distance_transform: -sigma*log(G*m) is not Euclidean (~d^2/(2 sigma), saturates ~18.4 sigma); per-axis sigma clipped to [0.5, 10] voxels in physical mode.
- **[FIXED afd33b9: real soft signed distance, tau > 0 required; (G*m)**10 documented]** distance_transform_loss: 'sdf_mse' == 'edt_mse'; tau=0/None divides by zero / crashes; without fields potential reduces to (G*m)**10.
- **[FIXED afd33b9: float output, signed full-foreground negative; ANTs-order output for ANTsImage input documented]** compute_image_distance_transform: output cast to input dtype (int/bool masks truncate); ANTsImage + return_ants=False returns ANTs (x,y,z) order; all-foreground slice gives 0 even with signed=True.

### core/smoothing.py
- **[FIXED 00a3521: all ones]** get_boundary_mask: rim_size=0 returns an all-zero mask.
- **[FIXED 00a3521: spline_distance follows coord_convention; inert kwargs removed]** smooth_displacement_field_bspline: coord_convention='zyx' reverses spacing/origin/mesh_size but not a sequence spline_distance; **kwargs ignored.
- **[FIXED 00a3521: inert arguments removed; periodic FFT boundary documented]** apply_sobolev_green_operator: border_width, pad_to_fast, **kwargs ignored; FFT boundary is periodic.
- **[FIXED 00a3521: dead code removed; Dirichlet placement documented]** apply_dsti_green_operator: kwargs.get('s'/'spacing') dead code; Dirichlet zero is one voxel outside the grid.
- **[FIXED 00a3521: sigma / spacing / sigma_mode / kernel_type validated]** separable_gaussian_filter: kernel_type='compact' drops `mode`; unknown kernel_type falls back to 'bessel'; short sigma tuple raises (conv1d) or skips axes (conv3d); `spacing` ignored unless sigma_mode='physical' with scalar sigma; "ITK parity" unverified (radius rule differs).
- **[FIXED 00a3521: wrong kernel counts raise; kernel cache bounded (512)]** separable_1d_filter: kernel list not of length 2/3 returns x unchanged; Gaussian kernel cache never evicted.

### core/optimizers.py
- **[FIXED 00a3521: dsti_alpha and spacing used for DST-I]** RegAdam: `dsti_alpha` stored, never read ('dsti' branches use sobolev_alpha; policy.py's dsti_alpha=0.035 works only because it equals sobolev_alpha); 'dsti' branches ignore spacing; spacing has no effect on Gaussian branches.
- **[FIXED 00a3521: unknown regularizers / kwargs raise]** RegAdam: unrecognised regularizer string (typo, 'bspline') is silently Gaussian-smoothed.
- **[FIXED 00a3521: step bound without host sync; the global-max bound is the documented design]** RegAdam: step bound rescales the whole tensor on its global max, host sync every step.
- **[NOT-A-BUG: docstring claim removed]** LARS has no momentum (old "prevents momentum collapse" claim removed).

### core/mps_kernels.py
- **[FIXED 00a3521: padding_mode validated; bilinear / align_corners=True documented as caller requirements]** grid_sample_backward_mps: padding_mode other than 'border' treated as 'zeros'; mode / align_corners unchecked (callers must ensure bilinear, align_corners=True).
- **[NOT-A-BUG: documented bound (65536 contributions per input voxel), far beyond any registration grid ratio]** Fixed-point limbs overflow silently beyond 65536 samples per input voxel.

### syn_jax.py
- **[FIXED 6f133bc]** `SyNJAX.fit`: `kwargs.get('use_analytical_gradients', True)` overrides the explicit argument (False has no effect); only 'dinov2' switches to autograd.
- **[FIXED 6f133bc: interpolator is a static argument]** `warp_images_jax` is jitted with no static args but receives the str `interpolator` -> TypeError; the autograd path (only via 'dinov2') crashes.
- **[NOT-A-BUG 6f133bc: both backends use voxels (the PyTorch claim was wrong; docs corrected)]** All Gaussian sigmas (fluid / elastic / in-loop inverse / pyramid) are in voxels (`sigma_mode='voxel'`, spacing ignored); PyTorch uses mm with the same values from `syntx.syn`.
- **[FIXED 6f133bc: affine_epochs=0 always]** `fit` from `syntx.syn`: `affine_epochs` not passed -> default [100, 50, 20] Mattes-MI affine stage re-optimises on top of `init_M_phys` (PyTorch does not optimise the affine).
- **[FIXED: honours the transform type]** `get_affine_matrix_jax`: non-'Affine' types give rotation x isotropic scale ('Rigid' fits scale, 'Translation' fits rotation and scale).
- **[FIXED: per-channel metric loop with curr_metric_weights on every path]** Per-level `curr_metric_weights` used only on the 'lbfgs' path; nested per-level `syn_metric_weights` crashes the main loop.
- **[FIXED: a real retry loop with CFL halving]** Divergence retry runs at most once (`if`, not loop); ignores optimizer_type / regularizer / per-level weights.
- **[FIXED: fixed geometry; one L-BFGS-B iteration on the preconditioned gradient per epoch is the documented design]** 'lbfgs': r2l inverse uses moving geometry for a fixed-grid field; smoothed (not true) gradient given to L-BFGS-B; maxiter=1 per epoch.
- **[FIXED 233a8d2: pure-JAX soft Dice]** 'soft_dice' / 'dice' wrap torch / np.array / float inside a jitted value_and_grad -> fails under tracing.
- **[FIXED 233a8d2]** `forward` / `forward_inverse` re-permute M_phys already in tensor order (affine applied with reversed axes vs `fit`); expect (x,y,z) input and use constructor geometry, unlike `fit`.
- **[FIXED: fixed geometry on both paths]** With `initial_grid`, analytical path normalises with FIXED geometry, autograd path with MOVING.
- **[FIXED 6e988cd: PyTorch semantics (CC, or CC^2 with squared)]** `local_ncc_loss_nd_jax(use_ants_pseudo_gradient=True, squared=False)` actually computes cc^2: default 'lncc' is cc^2, autograd 'lncc' is signed cc.
- **[NOT-A-BUG (verified): the per-axis normalised-to-voxel factors form a similarity transform and cancel in the determinant]** `compute_physical_jacobian_determinant_jax` omits the (n-1)/2 normalised-to-voxel factors (correct only for equal axis sizes).
- **[FIXED 6e988cd: scaled error on both sides]** Anderson inverse safeguard compares scaled composition error with unscaled step residual.
- **[FIXED: Cramer solve with a fixed-point fallback, method validation; the wrapped differences only reach the masked rim (documented)]** hybrid_lm inverse: periodic jnp.roll Jacobian; try/except around solve never triggers under tracing; no spacing -> silently Anderson.
- **[FIXED: None -> -W; midpoint integration]** `update_inverse_field_nd_jax` fixed point with `W_inv_disp=None` fails; velocity integration 'midpoint' -> Euler in normalised branch.
- **[FIXED: signatures cleaned; Sobolev / DST use the reversed ITK spacing]** `rprop_update_step_jax`: `lr` unused; `boundary_suppression_thresh`, `velocity_clamp`, `cfl_max` never used; `_apply_dsti_green_operator` spacing unused; `_apply_sobolev_green_operator` indexes ITK-order spacing by tensor axis.
- **[NOT-A-BUG: documented approximating sampler]** `jax_grid_sample_bspline` has no prefilter (approximating, blurs).

### syngs_jax.py
- **[NOT-A-BUG: documented separate implementation (module docstring); metric FIXED 4bdfa7b, moving geometry raises]** Not a mirror of PyTorch: `shoot` evolves v by EPDiff (PyTorch: stationary smoothed v0); loss always LNCC; no moving geometry; defaults differ (fluid_sigma 1.0 vs 3.0, n_steps 5 vs 6, inverse_identity_weight 1.0 vs 0.5).
- **[FIXED 4bdfa7b: level_spacing_itk]** Anisotropic spacing order bugs: ITK-order spacing zipped with tensor-order shapes (`shoot`, `forward`); `fit` passes ITK-order spacing as `spacing_zyx` and divides (z,y,x) gradients by it in the CFL step.
- **[FIXED 4bdfa7b: -v0]** `symmetric=False`: `fit` passes +v0 as the inverse velocity (forward / get_inverse_warp use -v0); its gradient is discarded.
- **[FIXED 4bdfa7b]** `init_velocities_from_image_gradients` resizes a channels-last gradient with a channels-first interpolator.
- **[FIXED 4bdfa7b: removed / used; unknown fit keywords raise]** Inert: `image_grad_clip`, `velocity_clamp` (hard-coded 50), `cfl_max`, `elastic_sigma`; forward `multipoint_loss`; fit `similarity_metric`, `moving_*`, `elastic_sigmas`, `fast_smooth`; `integrate_momentum_jax` `t_end`, `return_trajectory`, `alpha`.
- **[FIXED 4bdfa7b: midpoint implemented, unknown solvers raise; n_steps default documented]** 'midpoint' solver silently Euler; `integrate_momentum_jax` n_steps=6 vs PyTorch 8.

### tvf_jax.py
- **[FIXED 6f133bc]** `syntx.tvf(backend='jax', regularizer='gaussian')` passes `alpha=None` (tvf.py:1901) -> `float(None)` TypeError in `fit` (tvf_jax.py:942); spectral regularisers are unaffected.
- **[FIXED 6f133bc: no extra affine stage]** `affine_epochs` default 100 not passed by `syntx.tvf` -> extra Mattes-MI affine stage on top of T_init (PyTorch has none).
- **[FIXED 9ff396a]** `initial_transform` physical matrix stored as T_init without the grid conversion (applied in the wrong space).
- **[FIXED 9ff396a: delegates to syntx.tvf(backend='jax')]** `tvf_registration_jax`: learned affine never exported (TODO); `ants.invert_ants_transform` on path strings; keeps (x,y,z) order while helpers assume (z,y,x); `tempfile.mktemp` files never deleted.
- **[FIXED 4bdfa7b / b2f0f87: level_spacing_itk, tensor-order CFL, reversed spacing for Sobolev / DST]** Spacing: `forward` zips (x,y,z) spacing with tensor shapes and scales by N/n not (N-1)/(n-1); CFL / Sobolev steps pair (x,y,z) spacing with (z,y,x) axes.
- **[FIXED 6f133bc: mm sigmas converted to the JAX per-level voxel variance]** fluid / elastic sigma square-rooted and used in voxels (PyTorch: mm).
- **[FIXED b2f0f87: metric honoured; inert parameters removed]** Inert: `similarity_metric` (always LNCC), `moving_*`, `image_grad_clip`, `velocity_clamp`, `cfl_max`, `use_analytical_gradients`.
- **[NOT-A-BUG: documented (tvf_jax docstring); syntx.tvf passes the setting]** `antisymmetric` (default True) means time-mirror averaging (different meaning, PyTorch default False).
- **[FIXED b2f0f87: training settings]** Early stopping recomputes the loss with `forward` defaults, not the training settings.
- **[NOT-A-BUG: documented separate defaults; syntx.tvf(backend='jax') raises for PyTorch-only options]** Defaults differ from PyTorch: 1 vs 4 steps per interval, epochs [100,100,50] vs [100,100,20], lr 0.15 vs 1.0, multipoint [0,1] vs [0.5], constant_speed False vs True.

### tvf_adj.py
- **[DEFERRED: tvf_adj is deprecated (user decision); deletion is the user's call]** `integrate_svf` does nothing (loop body `pass`).
- **[DEFERRED: tvf_adj deprecated]** `integrate_forward`: `n_steps` unused; normalisation uses size/2; returns T+1 grids.
- **[DEFERRED: tvf_adj deprecated]** Local `get_physical_grid_torch` / `physical_to_normalized_torch` use opposite conventions (not inverses).
- **[DEFERRED: tvf_adj deprecated]** `TVFRegistrationAdjoint`: `initial_transform` unused; coarse voxel-unit velocities not rescaled on upsampling; final upsample raises if the last level is skipped; the adjoint is approximate.
- **[DEFERRED: tvf_adj deprecated]** `tvf_registration_adjoint`: lr=50 vs class 0.5; device='mps' default; direction ignored; temp files never deleted.

### motion_batched.py
- **[FIXED dcf2192: no temp dirs, L-BFGS stage lr entries removed; num_bins documented]** `batched_rigid_register_pass`: `num_bins=18` default overridden by `motion_correction` (32); temp dirs never cleaned; LBFGS schedule `lr_t` / `lr_r` ignored (lr fixed 1.0).
- **[FIXED dcf2192: max_level removed, verbose prints]** `batched_group_bias_register_pass`: `verbose`, `max_level` unused; level>1 / `corr_weight` branches unreachable.
- **[FIXED dcf2192: checked (_check_frames_on_reference)]** Frames assumed to share the reference grid (unchecked); scaling assumes non-negative intensities.

### __init__.py
- **[FIXED a4eb117]** `__all__` lists `plot_comparison`, `plot_structural_comparison`, which do not exist -> `from syntx import *` raises AttributeError (verified).
- **[FIXED a4eb117: setdefault]** Import always sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0, overwriting a user value.

### contract.py
- **[DEFERRED: cross-repo -- making antsxdwi import syntx.contract (and settling the `denoise` field) is the user's call]** antsxdwi does not import `syntx.contract`; it has its own drifting copy (with a `denoise` field).

### diagnose.py
- **[FIXED 616e763: the diagnosis records why the deep tier did not run (errors warn)]** Tier-2 weights `src/syntx/models/diagnostic_resnet10_3d.pth` do not exist -> deep classifier never runs; its exceptions are swallowed.
- **[FIXED 616e763: is_contrast_enhanced removed; the other enum values are reserved and documented]** Labels never produced: `is_contrast_enhanced`, MONO_MODAL_INTER, MRI_FLAIR / ADC / OTHER, modality UNKNOWN; HEART only from the classifier.
- **[FIXED 616e763: SIGNED_FLOAT domain; body-part heuristics documented]** Any non-HU data outside [0, 1.05] (incl. negative) labelled POSITIVE_FLOAT; non-CT body part from shape / extent only (any 2-D or 3-D < 260 mm -> BRAIN).
- **[FIXED 616e763: consistent fast default]** `diagnose_pair` defaults fast=True, `diagnose_image` fast=False.

### policy.py
- **[FIXED 002f246: the policy holds only fields auto_reg applies]** `auto_reg` ignores regularizer, sobolev_alpha, dsti_alpha, grad_step, flow_sigma, total_sigma, guided_weight, parameters -> thorax 'dsti1' and pelvis sobolev_alpha=2.0 rules have no effect; `explanation` describes unapplied settings.
- **[FIXED 0882a94 / 002f246: unreachable ROI rule dropped; to_dict gives auto_reg keywords]** `details['is_roi_crop']` never set -> ROI branch unreachable; `to_dict` omits dsti_alpha, not valid as auto_reg(**d).
- **[FIXED 002f246: forwarded as syn_metric]** Suspected (read, not run): auto_reg puts `similarity_metric` in kwargs when a policy exists and tvf_registration rejects it -> thorax-CT (TVF) policies fail.

### classifier.py / resnet.py
- **[FIXED 616e763: CPU classifier; 2-D input -> ValueError]** `predict_diagnosis_deep` defaults to MPS without CUDA; `preprocess_volume_for_classifier` fails on 2-D input.
- **[FIXED 616e763: weights loaded by name with module. / downsample mapping, error when nothing matches]** `ResNet10(num_classes=...)` has no effect; shortcut layers named `shortcut` vs MedicalNet `downsample`, loaded with strict=False -> pretrained weights may silently not load.

### generators.py
- **[FIXED 1764197]** `benchmark_data('mbhard')`: missing Mindboggle files -> silently writes a synthetic 64^3 sphere/ellipsoid pair under the Mindboggle file names in ~/.syntx/benchmark_data/mbhard/, returned with the Mindboggle description and reused later as if real (verified in code).
- **[FIXED a649e1a]** `CrossProductGenerator`: ANTsImage base used in ANTs order with spacing[0] applied to the W component (swapped for anisotropic images); rotation in normalised coords not rigid for non-square images.
- **[FIXED a649e1a: unknown magnitude_level raises; the integral norm is documented]** `compute_physical_l2_norm` is an integral sqrt(dV sum |u|^2) (grows with image size), not a norm in mm; unknown `magnitude_level` string means 1; `temp_seed` restores only CPU torch / NumPy RNG.

### perf_tracking.py
- **[FIXED e720d92]** Git-commit lookup runs before the directory is created (None on a new path); `detect_regressions` does not cast to float, NaN never flagged, baseline 0 skipped, NaN/inf written as non-standard JSON.

### provenance.py
- **[FIXED e720d92: internal / nested calls captured]** Capture patches module attributes only: pre-imported references / internal imports (robust_affine inside registration, registration inside auto_reg) not recorded; nested capture records nothing; only fit() kwargs recorded.
- **[FIXED e720d92]** `assert_manifest_complete` dirty-diff check never fails (diff_sha256 always present); `environment()` imports jax / antstorch; `code_state` splits untracked paths on whitespace.

### tabulate.py
- **[FIXED e720d92]** `correlation_matrix`: constant column -> 0 (diagonal too) in torch, NaN in numpy; float64 fails on MPS. `roi_mean_timeseries` truncates labels to int, no same-grid check.

### cli.py
- **[FIXED 85ca9ea]** `--report` has no effect (store_true with default=True).
- **[FIXED 85ca9ea: only options given on the command line are passed]** `register` defaults (grad_step 0.25, flow_sigma 5.0, optimizer reg_adam) override library defaults (syn 0.4 / 2.4 / cfl; tvf cfl); SyN-only options ignored for TVF and --optimizer for SyN, yet recorded in metrics.json.
- **[FIXED 85ca9ea: options a model does not use raise; info shows syntx.__version__; the Jacobian of the deformable field is documented]** `--regularizer` choices omit 'bspline' though the TVF branch tests for it; `cmd_info` falls back to stale '4.0.2'; Jacobian from the deformable field only.

### spatial.py
- **[FIXED 00a3521]** `get_spatial_coordinate_grid`: ANTs-order shape indexed as z,y,x -> x / z ranges swapped for non-cubic images (confirmed on CPU; no live callers).
- **[FIXED ed96e57]** `jacobian_determinant` 2-D: axis 0 divided by spacing[1], axis 1 by spacing[0] -> swapped for ANTs-layout anisotropic input (confirmed: 1.2 vs 1.1); ignores direction. 3-D: only the direction diagonal used (wrong for oblique). Tensor converted only with ref_image and batch 1.
- **[FIXED 00a3521]** `restriction_from_orientation`: bare 'I' raises; 'z' on 2-D raises IndexError; "exactly one of" not enforced.
- **[FIXED 3fa190a / ed96e57: spacing / direction it would ignore raise]** `deformation_gradient` / `jacobian_determinant`: tensor-layout ndarray never converted; spacing / direction with tensor + ref_image silently ignored.
- **[NOT-A-BUG: documented conventions (ANTs-order input; to_zyx flag)]** `tensor_to_image` does no transpose (expects ANTs order); `image_to_tensor` defaults to_zyx=False (tvf_jax.py relies on it).
- **[NOT-A-BUG: documented (name kept for API compatibility)]** `physical_to_normalized_torch_cached` caches nothing.
- **[FIXED 3fa190a: exact inverse]** Direction inverted by transpose in some helpers, general inverse in others (disagree for non-orthonormal direction).
- **[FIXED 3fa190a: input dtype kept]** `lps_to_ras` / `ras_to_lps` cast to float32.

### core/jacobian.py
- **[FIXED 0295eab]** `_spatial_jacobian_nd(method='bspline')`: kernel [-1,-8,0,8,1]/12 has wrong outer signs -> 5/3 x the true slope (verified); affects compute_jacobian_determinant_nd / compute_physical_jacobian_determinant / SyNToTransform.get_jacobian_determinant(method='bspline').
- **[FIXED 81cfc9d: one tensor-order convention]** `compute_jacobian_determinant_nd`: spacing order and component order differ across its bspline / physical / normalised paths; passing physical_spacing silently switches to mm; unbatched check `dim() == dim` never fires; channels-first detection misfires when the last spatial size is 2 or 3.
- **[FIXED d6ee6b3: ITK-order spacing of the fixed grid]** syn.py:1304 hinge penalty: spacing reversed twice; in-loop warp probably normalised but treated as mm.
- **[FIXED 81cfc9d: det(I + G D^-1); inert origin / kwargs removed]** `compute_physical_jacobian_determinant`: non-physical path's direction / spacing cancel (no effect); physical path ignores direction; origin / kwargs ignored.

### core/grid.py
- **[FIXED 81cfc9d (confirmed): tensor-order gradients, inv(D)]** Suspected: `prepare_mid_images_and_gradients_torch` (analytic path, not syn's default) applies (x,y,z)-ordered image gradients to (z,y,x) physical warps -> x / z swapped; C > 1 without precomputed gradients gives a wrong layout; warp_*_inv unused.
- **[FIXED 81cfc9d: torch.gradient; AnalyticalGridSample documented as approximate. JAX twin FIXED v5.4.95]** `_image_spatial_gradient` uses torch.roll (border wraps); AnalyticalGridSample is approximate (no input gradient, no outside-region zeroing).
- **[FIXED 81cfc9d: unknown padding raises; no prefilter documented]** `grid_sample_bspline_torch` has no prefilter (smooths); non-'zeros' padding acts as 'border'.
- **[FIXED 81cfc9d]** `_generic_label_sample` 'zeros' padding gives the lowest label present; int input returns float.
- **[FIXED 81cfc9d: unknown interpolators raise]** `grid_sample_nd`: unknown interpolator silently falls back to `mode`; analytical gradients even for nearest. Deterministic backward treats non-'border' padding as 'zeros'.

### core/affine.py
- **[FIXED: transform_type validated; 'Translation' has no rotation]** `HierarchicalAffine`: 'Translation' still optimises rotation (== 'Rigid'); transform_type unvalidated.
- **[FIXED 81cfc9d]** `parse_ants_affine` composes first-item-first (reverse of apply_transforms convention); Euler / Similarity objects skipped silently.

### core/pipeline.py / core/__init__.py
- **[FIXED: quantiles honoured; unequal channel counts / a JAX device raise]** `normalize_and_tensorize`: winsorize_quantiles ignored (2/98 hard-coded); zip drops unmatched channels; device ignored for jax.
- **[FIXED]** core `__all__` omits RegAdam / SobolevAdam / GaussianAdam; compute_jacobian_hinge_penalty not re-exported.

### viz/reports.py
- **[FIXED 16cfec7]** `create_affine_benchmark_report`: syntx affine time is the constant 2.8 s, so the reported speedup is fabricated (verified); ANTs time 28.5 s / Dice 0.3472 defaults hardcoded; reads `results/...` relative to cwd; "90 / 90", comparison table and protocol text are fixed; provenance ignored.
- **[FIXED 16cfec7]** `create_population_benchmark_report`: hyper-parameter table, "/ 90", "TVF beats Sobolev in 88/90" hardcoded; speedup always vs Sobolev time; cohort falls back to idx < 40 = intra; missing ANTs folding counts as 0; provenance and list baseline_source ignored.
- **[FIXED 4fe105a]** `create_registration_report`: "LNCC (w=9)" is window 5 (image_compare ignores window_size); "SSIM" / "NCC" reported as ssim-1 / ncc-1; no inverse map -> 0 mm, no Jacobian -> 0 % folding, Dice errors -> "N/A" silently; Dice not filtered to [0,1]; show_report / kwargs / fixed_name / moving_name no effect; "Verified Provenance" badge always shown; figures never closed.
- **[FIXED b25d021]** `_compute_jacobian_stats` converts to NumPy before `jacobian_determinant` (skips the tensor-order reversal); all-ones map on any error.
- **[FIXED b25d021]** `build_engine_provenance`: "syntx_version" is the cwd's git HEAD; missing flags become False not "N/A".
- **[FIXED 00bb590]** `create_benchmark_report`: missing keys count as 0.0; means not restricted to paired indices; output dir not created.

### viz/figures.py
- **[FIXED: duplicates removed]** Redefines `get_dkt_colormap`, `dkt_colormap`, `get_dkt_label_color_dict` (twice; tab20 version wins), shadowing viz.colormaps.
- **[FIXED e50567e: _display_displacement]** All displacement plots (deformation grid, correspondence vectors, vector field, 4-panel A): mm added to pixel coords without dividing by spacing; vertical sign inverted; field and background sliced separately (different slices when slice_idx=None).
- **[FIXED: shared default slice, per-slice statistics labelled]** `render_standard_4panel`: panels can show different slices; folding / min det / inverse error are per slice; error cites "GEMINI.md Section 3"; failing panels silently zeros.
- **[FIXED 1db63be]** `render_input_pair_figure`: intended docstring sits after the first statement (string left in place, real docstring added); moving sagittal title prints wrong index; 3-D crop_background never crops; dead aspect code; colorbar from axial panel only.
- **[FIXED 1796cab]** `plot_deformation_tensor_rgb`: R/G show y/x not x/y; `alpha` unused; RGB image has default geometry. `compute_deformation_tensor_rgb` NumPy fallback permutes derivative axes (not F).
- **[FIXED 80b1016]** `plot_time_varying_velocity_grid`: channels transposed vs extract_slice; silently substitutes midpoint warps when velocities ~0; dict path unreachable; bending energy spacing swapped; "(px)" unverified.
- **[FIXED cfe758c / 8598cb1 (non-3-D raises, v5.4.95)]** `render_label_alignment_figure`: rot90 sagittal mirrored vs extract_slice; 3-D only; colorbar lists first 16 labels. `extract_2d_slice` ignores ref_image; `plot_edge_overlay` resize import outside the try.

### viz/core.py
- ~~`verify_anatomical_orientation` always returns True.~~ FIXED v5.4.95: removed (unused placeholder).
- ~~`corner_watermark`: `corner` ignored; unseeded noise.~~ FIXED v5.4.95: all four corners, `seed` (default 0), bad corner / patch_size raise.
- ~~`extract_slice` vectors: sagittal not column-flipped; channel 0/1 swapped for every plane; integer plane outside 0-2 raises.~~ FIXED v5.4.95: vector slices get the scalar layout (sagittal flip included) and keep their physical components (display arrows go through `_display_displacement`); unknown planes raise ValueError (unknown strings silently meant axial). 2-D images not row-flipped: NOT-A-BUG (matches `ants.plot`, tested). 3-D arrays with last axis 2 / 3 read as 2-D vector fields without `ref_image`: documented heuristic (with `ref_image` the grid decides).
- ~~`prepare_image`: arrays without ref_image indexed as ANTs order; failed reads swallowed; any leading axis of size 2/3 taken as channels.~~ FIXED v5.4.95: arrays / tensors are syntx tensor layout with or without `ref_image` (also in `_as_displacement_image`, `compute_deformation_tensor_rgb`, `render_standard_4panel`), off-grid arrays raise, read errors propagate, `.mat` raises, a leading channel axis is taken only when the rest matches `ref_image`'s grid.

### viz/colormaps.py
- ~~HSV not HSL (`lightness` is HSV value); `get_dkt_colormap` colours by label ID, `build_dkt_label_palette` by rank -> same label, different colours; "3.0" stays a string, floats truncated.~~ FIXED v5.4.95: one ID-based scheme (palette == colormap entry), `lightness` renamed `value`, numeric strings / integral floats are IDs, non-integral labels raise.

### viz/stats.py
- ~~`plot_label_overlap_stats`: plain {region: list} dict fails; in dict / array mode the three boxes are identical.~~ FIXED v5.4.95: region lists averaged; one "Dice" box unless both directions are given; non-finite values dropped, none left raises.
- ~~`plot_jacobian_distribution`: "Fully Diffeomorphic" whenever no voxel <= 0, unmasked; `.numpy()` before `.detach()`.~~ FIXED v5.4.95: `mask`, non-finite values excluded and counted, "no det(J) <= 0", tensors detached first.
- ~~`plot_loss_convergence`: hardcoded labels, not exported, no output dir creation.~~ FIXED v5.4.95: `xlabel` / `ylabel` / `label`, exported from `syntx.viz`, directories created.

### viz/gallery.py / viz/modality_report.py
- ~~gallery: light-theme figure rendered and unused; `title` only sets <title>; fallback version "1.1.8".~~ FIXED v5.4.95: unused render removed, `title` is the heading (escaped, as are provenance values), version from `syntx.__version__`, skipped figures warn (2-D labels, tensor RGB failure).
- ~~modality_report: `matplotlib.use("Agg")` global side effect; output dir not created; kpis_html / description unescaped; title_override only <title>; footer text with `brand`.~~ FIXED v5.4.95: standalone `Figure` (no backend switch), directories created, `title_override` sets the heading too, plain footer (also in viz.reports). kpis_html / description unescaped: NOT-A-BUG (documented HTML inputs; `kpi_card` escapes its fields).

### features.py
- **[FIXED 616e763]** FeatureSpaceLoss: 'lncc_3d' ignores `lncc_window` (always 5) and uses only the last layer; any other `mode` (typos too) silently runs triplanar.
- **[FIXED 616e763]** _forward_2d_triplanar: 3-channel extractor gives an empty slice at index 0 (D, H or W < 4).
- **[FIXED 616e763]** ResNet10Extractor: 2-D net always random; 3-D MedicalNet loaded strict=False without "module." stripping (may load nothing). SwinUNETRExtractor: `img_size` inert; strict=False; failed download leaves random weights with a warning.
- **[FIXED 616e763]** DINOv2Extractor.extract: MPS input permanently moves model / buffers to CPU; padded patch tokens not cropped; '_reg' variants reshaped wrongly.

### surface.py
- **[FIXED 0882a94: all eight codes kept; contrast dependence documented]** compute_surface_classes: gyral / sulcal = sign of H on intensity iso-surfaces (contrast-dependent); codes 5-8 discarded.

### landmarks/preprocess.py
- **[NOT-A-BUG: docs corrected]** Default use_n4 is False (old docs said True); normalisation is entropy-selected percentiles, not fixed 2-98.
- **[FIXED 7fd5296]** CT without ct_window: percentiles from voxels > 0 only -> all negative-HU tissue clipped to 0.
- **[FIXED 7fd5296]** is_ct_image: any MRI with > 1 % negative voxels (e.g. z-scored) classified as CT.

### landmarks/blob.py
- **[FIXED 4f49cef: renamed _per_scale_maxima; comments corrected]** _scale_space_extrema: no cross-scale test (per-scale spatial maxima only); comment "unused internally" is false (it is the ranking score); threshold is a fraction of the per-scale max, not top 0.5 %.

### landmarks/sift2d.py
- **[FIXED fb45de9]** `_run_slice` u_maps_ix / v_maps_iy unused; NMS ranks by size not response; first / last slices always included.
- **[FIXED fb45de9: return_planes; no monotone-intensity invariance documented]** Axial / coronal / sagittal descriptors pooled with no plane label (cross-plane matches possible); no monotone-intensity invariance.
- **[FIXED fb45de9]** _normalize_slice_uint8 wraps for input outside [0, 1] when there is no usable range.

### landmarks/sift3d.py
- **[NOT-A-BUG: documented detector choice]** Thresholds the sigma^2-normalised DoG (blob detectors threshold the raw response).
- **[FIXED 2d67025]** _build_descriptor: `min_anisotropy=1.0` makes the stability mask always True (inert); sign_mode / frame_scale / return_stability not exposed; frame_rotation ignored with rotation_invariant=True; N=0 returns a bare array regardless of return_* flags.

### landmarks/mind.py
- **[FIXED f8c899c: 'MIND-SSC' label dropped (documented MIND-style variant)]** Not published MIND-SSC (variance denominator, (centre, centre+r) pairs, no max normalisation, asymmetric 12-offset set).
- **[FIXED f8c899c]** Even patch_size gives output one voxel larger per axis; n_offsets > 26 gives 26 silently; extract_mind_at_points reuses mind_vol unchecked.

### landmarks/matcher.py
- **[FIXED ece455d]** match_landmarks: kpts_src / kpts_dst unused; M=1 ratio fixed 0.5 (always accepted at default).
- **[FIXED ece455d]** ransac_filter: fewer than min_inliers -> returns ALL matches unfiltered plus identity; no accepted fit -> empty; unknown model = 'affine'.
- **[NOT-A-BUG: documented conditioning guard / fallback]** _fit_affine rejects condition number > 6; rejected consensus refit keeps the minimal-sample fit.

### landmarks/orient.py
- **[FIXED 6968da8]** match_sift3d_with_rotation_search: ransac_iter not passed to refinement (3000 used); early-exit result lacks 'winner'; other descriptor options go to sift3d_keypoints and raise TypeError.
- **[FIXED 6968da8]** refine_rotation_iteratively returns descs_moving from the previous frame, not the returned R.
- **[NOT-A-BUG: docs corrected]** Old rotation_grid counts were wrong (111 and 172, not ~50 / ~270).

### landmarks/optimal_transport.py
- **[FIXED 53eebaf]** Foreground smaller than min_samples raises in rng.choice; foreground test `> 0.05` on raw intensities.
- **[NOT-A-BUG: documented scope; the returned .mat belongs to the caller (like ants.registration outputs)]** sampled_optimal_transport_affine: only rigid / similarity; spatial cost compares raw physical coords (identity bias); temp .mat not deleted; EMPTY_FG dict keys differ.
- **[FIXED 53eebaf]** weighted_procrustes: Umeyama scale ignores the reflection sign flip.
- **[FIXED 53eebaf]** score_rotation_candidates_sampled: non-'mind' features re-normalised to +-1/0 (score barely rotation-dependent).

### landmarks/spatial.py / landmarks/__init__.py
- **[FIXED 70aa224]** get_image_affine silently returns identity for unsupported inputs; ortho_view_spec(center_mm=None) with tuple geometry crashes.
- **[NOT-A-BUG: docstring corrected]** Old package example passed mm landmarks with domain_bounds (-1, 1) and ANTs-order grid_shape (docstring corrected).

### data/msd.py
- **[FIXED: safe extraction, progress hook, dataset.json check]** download_msd_task: `progress_bar` unused; `tar.extractall` without a filter (path-traversal risk); task_dir existence unchecked.
- **[FIXED: transform, labels, linear interpolation, 4-D slicing]** MSDDataset: `transform` never applied; labels never loaded; resampling is nearest-neighbour (interp_type=1); 4-D tasks (01, 05) with a 3-value target_shape unverified; arrays are ANTs (x,y,z), not [1, D, H, W].

### data/surrogates.py
- **[FIXED: ANTsImage, axial axis from the direction]** extract_ct_body_trunk returns ndarray (siblings return ANTsImage); assumes axis 2 is axial.
- **[NOT-A-BUG: documented surrogate range]** extract_ct_abdominal_viscera HU [25, 160] includes muscle / vessels.
- **[FIXED: warns; channel parameter]** extract_brain_parenchyma: below min_volume silently returns the raw threshold mask; 4-D path hard-codes channel 1.
- **[FIXED: ValueError]** extract_surrogate_target returns None for unrecognised tasks without warning.

### scattered/solver.py
- **[FIXED 10ea6f8 / 2a227c2]** fit: inverse-consistency check calls warp_scattered_coordinates without domain_bounds / vector_convention -> 3-D components reversed and coordinate-unit points treated as [-1, 1]; inverse_identity_error / max_error / mean_error and the u_inv choice wrong in 3-D or non-(-1,1) bounds.
- **[FIXED 2a227c2]** ScatteredRegistrationResult.warp_points / transport_features / pullback_grid / pushforward_features and fit's feature transport omit vector_convention='xyz' (3-D components flipped); SyNScattered methods pass it correctly.
- **[FIXED 10ea6f8]** max_error_l2r / _r2l / compute_inverse_error pass [-1,1] (x,y,z) fields to compute_inverse_identity_error_nd (expects mm, tensor order): not true inverse errors.
- **[FIXED cfe6ba6: unit conversion; documented warm start]** Landmark init overwrites (does not compose with) the affine; copies coordinate-unit fields into the [-1,1] xyz warp (consistent only for bounds (-1,1), 'xyz').
- **[FIXED 7720467]** sigma_eff = max(sigma, 1.5 h) mixes coordinate units with [-1,1] voxel size; step cap / 'cfl' divide (x,y,z) components by (z,y,x) spacing; nothing guarantees det(J) > 0.
- **[FIXED: inverse_method honoured, w_distortion removed, inverse_steps as given, unknown names raise]** Inert: inverse_method (always Anderson), w_distortion; inverse_steps < 25 raised to 25; unknown similarity_metric -> LNCC, unknown regularizer -> Gaussian silently; 'bspline' with a sequence mesh_size raises TypeError.
- **[FIXED 08200ed / cfe6ba6]** `levels` factor-vs-size heuristic fragile ([1, 2] read as sizes); epochs_per_level overrides iterations; folding stats use the last level's mask (crash if not full resolution with domain_mask); deformation_energies['harmonic'] is mean(u^2).
- **[FIXED cfe6ba6]** syn_scattered: dim = ndim-2 (unbatched 3-D grid -> dim 1); kwargs mutate a passed config in place, unknown keys ignored; without a config kwargs never reach fit (`epochs=` raises TypeError).

### scattered/projection.py
- **[FIXED f9664af]** compute_adaptive_sigma: absolute floor / cap 0.01 / 0.2 (0.2 mm cap with mm coords); batch 0 only; random subsample for N > 2000 (varies between calls).
- **[FIXED c43466d: resolved once per registration]** 'auto' bounds margin 3 sigma for scalar sigma, fixed 0.05 otherwise; ProjectionConfig.sigma_scale used only for sigma='auto' with N < 2; `kernel` never read.
- **[FIXED f9664af]** ScatteredProjector static mode: int grid_shape with scalar bounds -> d=2 (3-D points crash); no sigma > 0 check; config.sigma_scale dropped in non-static mode.
- **[FIXED f9664af]** B-spline engine (also fit_bspline_landmark_warp, apply_bspline_fluid_regularizer): 'zyx' reverses origin / points but not size -> transposed or mismatched on non-cubic grids; fill_value only with return_density; mask not applied to density; "density" differs in meaning from the Gaussian engine.
- **[FIXED v5.4.95: ValueError]** compute_distance_transform_to_grid crashes for N=0.

### scattered/mapping.py
- **[FIXED 2a227c2: vector_convention passed everywhere]** warp_scattered_coordinates: without vector_convention, 3-D components reversed for both coord conventions.
- **[FIXED c43466d / d54922a: 'auto' with a field raises; bounds resolved once]** 'auto' domain_bounds derived from the query points (different point sets -> different boxes; also transport).
- **[FIXED 90d7be9]** auto_invert always in normalised units (wrong for physical fields); explicit is_physical overrides scale_displacement; ANTsImage geometry ignored.
- **[FIXED: cached; is_physical / vector_convention settable]** ScatteredWarper.inverse recomputes every call; is_physical / vector_convention not settable.

### scattered/transport.py
- **[FIXED d54922a]** grid_bridge with 'auto' / None bounds: pushforward box (3 sigma margin) and pullback box (5 % margin) differ -> misaligned grid; return_density ignored for grid_bridge.
- **[FIXED d54922a: NumPy inputs; (N, N) = batched is the tested convention (NOT-A-BUG)]** pushforward_scattered_to_grid: (N, N) features read as batched scalar with B=N; NumPy features crash; padding_mode not settable.
- **[FIXED d54922a]** direction='backward' behaves like 'forward' (no auto_invert).

### scattered/bspline.py
- **[NOT-A-BUG: docs corrected]** apply_bspline_fluid_regularizer real defaults: mesh_size 6 (not 2), enforce_stationary_boundary False (not True).
- **[FIXED f9664af]** BSplineScatteredProjector: device / dtype unused, nothing precomputed.
- **[FIXED: annotation / docs]** bspline_syn_scattered returns ScatteredRegistrationResult, not a dict with 'warped_grid' (KeyError).
- **[FIXED f9664af]** spline_distance sequence passed in ITK order unchanged for 'zyx'.

### benchmark/tune.py
- ~~Cache key and default out_dir omit the dataset.~~ FIXED v5.4.93 (62d4f92).
- ~~`pair_violations` does not flag NaN Dice / folding (failed run can look feasible).~~ FIXED v5.4.95: NaN Dice / folding / min Jacobian is a violation.
- `twod_evaluator`: ~~affine not forced to CPU~~ FIXED v5.4.95 (`device='cpu'`; cached affines unchanged); ~~StopIteration without a .nii.gz warp~~ and ~~silent NaN inverse without 'phi_1'~~ FIXED v5.4.95 (ValueError). Raw images and 'time_s' without the affine: NOT-A-BUG (methods normalise internally; the affine is precomputed and constant; both documented).
- DEFAULT_FIXED_PARAMETERS never passed ("fixed" = method default), `--unfix reg_iterations` greedy-only, CLI exposes only `jac_min_rel`, record key `tuned_<method>_<date>`: NOT-A-BUG (documented design).

### benchmark/evaluate.py
- ~~`_evaluate_mindboggle_pair_impl`: `dataset_key` unused (2-D keys from worker / grid register Mindboggle pair 0).~~ FIXED v5.4.95: checked (None / 'mindboggle' / 'mbhard' alias with pair 44); anything else raises before loading.
- ~~Affine cache name ignores use_n4, pairs_csv, data_dir~~ FIXED v5.4.95 ('_noN4', a hash for non-default pairs_csv / data_dir; subject ids stored and checked). ~~Missing ANTs baseline gives folding / min_jac 0.0~~ FIXED (NaN). ~~Regadam arm ignores fast_smooth~~ FIXED (fast_smooth=True raises). 'syntx_time' with a cached t_aff: NOT-A-BUG (documented; the affine is computed once per pair).
- ~~`evaluate_affine_benchmark`: DataFrame passed where a path / dict is required (empty report); unknown `pairs` -> pair 0; failed mode recorded as Dice 0 / time 0.~~ FIXED v5.4.95: own per-mode report, unknown keywords raise, failures NaN with an 'error' column; exported in `__all__`.

### benchmark/grid.py, worker.py, runner.py, config.py
- ~~Grid's top-level regularizer / fast_smooth inert~~ FIXED v5.4.95: `get_model_config` puts them into the block (syn_* names for SyN); a default alias (fluid_sigma) no longer overrides the grid's flow_sigma. Phase-1 2-D tasks now FAIL loudly in the worker (the evaluator is Mindboggle-only; they ran Mindboggle pair 0); 'mbhard' runs pair 44.
- ~~worker exits 0 after writing FAILED; `run_benchmark_suite` output_dir unused~~ FIXED v5.4.95: exit 1 (the runner keeps the worker's error record), `report_path` replaces the inert `output_dir`. Returning `tracker.state`: documented.
- ~~`get_model_config`: models without a block record the whole DEFAULT_BENCHMARK_CONFIG; 'regadam_greedy' maps to no block; gaussian_config has inverse_steps.~~ FIXED v5.4.95 ({}; mapped, also 'syn_mi'; removed).
- ~~`__init__`: evaluate_affine_benchmark not in __all__.~~ FIXED v5.4.95.

### benchmark/metrics.py, msd.py, html_report.py
- ~~compute_pair_metrics: nan_to_num hides NaN inverse errors; energies over the whole grid; kwargs ignored.~~ FIXED v5.4.95: finite values only (warning), energies over the fixed mask (shared `warp_jacobian_and_energies`, also used by high_level), **kwargs removed.
- ~~msd: unused imports; missing auto_reg metrics default to folding 0 / min_jac 1 (look clean).~~ FIXED v5.4.95: imports removed, NaN when not reported. Also found: `affine_iterations` was passed to `auto_reg` (removed from every method), so every MSD pair raised -- parameter removed.
- ~~html_report: header / footer hard-code "cc2", "[100, 100, 20]", "pt7", "Seed 42", "syntx v5.4.10"; NaN ANTs prints "nan"; greedy labelled "LDdMM"; compute_model_stats gives 0.0 means with no valid values.~~ FIXED v5.4.95: header from the records (affine backends, configuration count), installed version, "n/a" for missing / NaN, "Compositive Greedy", None statistics; win rate over pairs with a baseline (was over all, a missing baseline counted as a loss); ids escaped.

### benchmark/high_level.py
- ~~ANTs arm uses type_of_transform='SyN' (re-runs affine)~~ FIXED v5.4.95: 'SyNOnly'.
- ~~2-D scorer: dice_fixed = dice_moving = mean symmetric Dice; thresholds labels 2 and 3 (binary 'c' / 'ellipse' give empty labels).~~ FIXED v5.4.95: real fixed / moving means; binary maps scored on their positive labels. Also found: both scorers warped fixed labels with `invtransforms` but no `whichtoinvert`, so the affine was applied forward, not inverted: every moving-space Dice was wrong -- FIXED (the result's `whichtoinvert_inv`, else invert the `.mat` items).
- 'tvf_jax_cpu' with default overrides raises: NOT-A-BUG (fail-loud, documented). ~~Failing Jacobian step silently drops columns~~ FIXED (NaN + warning). "Pair 00" label: no longer present.

### benchmark/orchestrator.py, cli.py, data.py, codify.py
- ~~orchestrator: 'total_completed' = len(sobolev_results); other models never summarised; seed not passed; kwargs dropped; ANTs scan hard-coded range(90), 'results/'.~~ FIXED v5.4.95: planned pairs with every planned model done (+ 'completed_by_model'), every model summarised, `--seed` / `--denoise` passed, unknown kwargs raise, baselines for every CSV row from `ants_baseline_dir`; unreadable summary / cache / report failures warn.
- ~~cli: default --model 'syn_tvf' (and 'all') raises in --pair-idx mode; cohort mode drops --no-n4 / --denoise; --out-name with --model both overwrites; help texts wrong.~~ FIXED v5.4.95 (`expand_model_set` in both modes, N4 / denoise passed, --out-name needs one model, help corrected).
- ~~data: symlink mode leaves dangling links; unknown mode copies; N4 failure returns the raw volume yet counts as computed; DEFAULT_DATA_DIR is user-specific.~~ FIXED v5.4.95: extracted files are moved (archives extracted safely), unknown mode raises, N4 failure raises (precompute lists failures), `~/data/mindboggle/volumes`. **Found while fixing**: the N4 call used antstorch's old tensor interface, which fails since the 2026-09-24 ANTsTorch N4 reorganisation; the failure was swallowed, so every subject without a cached N4 volume (49 of 71; 67 of 90 pairs have one) was registered uncorrected while its record said `use_n4=True`. Now the ANTsImage interface (CPU by default). The tuning pairs 0 / 44 / 77 are fully cached (unaffected).
- ~~codify: failed tests leave the tune branch behind; margin None crashes the commit message; apply_to_tree can partially write; rewrite_json_block lacks nested values.~~ FIXED v5.4.95 (branch deleted when nothing is committed, "n/a" margin, all texts computed before writing, nested JSON values via the decoder).
- **[FIXED v5.4.95]** Tests: the mock-Mindboggle evaluator tests ran in the repository directory and wrote their affines into `results/canonical_affines` (cwd-relative); they now run in their temp directory.

