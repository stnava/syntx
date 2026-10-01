# Docstring audit (started 2026-10-01)

Goal: every function / class in `src/syntx` documented accurately and readably: what it does,
every parameter with its real default and scope, real return keys / shapes. Every claim is
checked against the code. Behaviour issues found along the way are **listed here, not fixed**
(documentation commits change no code: `python scripts/check_docs_only.py` proves each one).

Inventory at start: 80 modules, 853 definitions, 640 public, 308 public with < 15-word docstrings.

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

## Behaviour issues found (not fixed)

### robust_affine.py
- `backend` parameter is unused.
- An unrecognised `mode` silently runs the ANTs path (should raise).
- `mode='fast'` is the default solver, while `preset='fast'` is the fast schedule -- confusing names.

### syn.py
- Warning "flow_sigma ... has no effect on kernel shape with regularizer='sobolev'" is wrong for
  SyN: with the default `fast_smooth=False` flow_sigma is the Gaussian post-filter and its value
  changes the result (measured 2026-09-30).
- `cfl_momentum`, `multipoint_loss`, `n_time_steps`, `n_steps` are accepted but unused (inert
  parameters -- by the project rule they should raise).
- Default regulariser differs by backend: 'sobolev' (PyTorch) vs 'gaussian' (JAX).
- An unrecognised `type_of_transform` matches no branch (`transform_type` undefined) -> fails
  later with an unclear error instead of a clear ValueError.
- `type_of_transform='greedy'` silently returns `syntx.greedy`'s result.
- `auto_reg`: folding metric uses a finite-difference Jacobian, not `syntx.liouville_determinant`;
  reads `phi_1['mean'/'max']` but registrations return `mean_error` / `max_error` -> those
  metrics are NaN; claims of NumPy / tensor inputs are not supported by the code paths
  (ANTsImage needed).
- The affine is never optimised inside SyN (only `robust_affine` beforehand, skipped when
  `initial_transform` is given). So `type_of_transform='Affine'` / 'Rigid' / 'Translation' with
  an `initial_transform` optimises nothing (reg_iterations forced to 0) and returns the input
  transform, silently.

### tvf.py
- `TVFModel.forward` adds an inverse-consistency penalty (weight `inverse_identity_weight`,
  fixed at 0.05) whenever `multipoint_loss` contains 0 and 1 -- a hidden setting `syntx.tvf`
  does not expose.
- `TVFModel(antisymmetric=...)` still exists (rewrites the evaluation times) although
  `syntx.tvf` removed the option.
- `TVFModel.fit` defaults (optimizer 'adam', similarity 'lncc', lncc_radius 4) differ from
  `syntx.tvf`'s ('cfl', 'cc2', 2) -- only matters for direct `fit` calls.

### syngs.py
- The module / class claimed EPDiff geodesics, minimal energy and det(J) > 0: the default
  'transport' mode integrates a stationary field, there is no energy term, and the true
  minimum determinant reaches ~3e-4 (docs corrected; behaviour unchanged).
- 16 accepted-but-unused `syngs_registration` parameters: total_sigma (stored as elastic_sigma,
  never read), type_of_transform, n_time_steps, cfl_momentum, multipoint_loss,
  sampling_percentage, vgg_* (5), project_inverse, projection_frequency, interpolator,
  inverse_method, inverse_steps. `fit`: reg_weight, fluid_sigmas, elastic_sigmas unused.
- `symmetric` / `inverse_identity_weight` cannot be set through `syntx.syngs` (passed kwargs
  reach `fit`, which ignores them).
- `GeodesicShootingModel`: unknown `regularizer` silently becomes 'sobolev'; 'dsti' silently
  means 'dsti1'. `fit`: unknown `optimizer_type` silently uses LARS.
- Axis-order bug: `forward` / `get_forward_warp` / `get_inverse_warp` compute the per-level
  spacing as `zip(self.spacing, self.image_shape, target_shape)` -- ITK (x, y, z) spacing paired
  with tensor (z, y, x) shapes; wrong for anisotropic images at non-native resolution (the
  same bug fixed in tvf.py's vel_spacing on 2026-09-30).
- `integrate_momentum`: the trajectory / t_end != 1 path re-smooths every step (recursive),
  unlike the t_end = 1 path and the registration -> inconsistent endpoints; `backend` unused.

### greedy.py
- Unknown `similarity_metric` silently uses local correlation (only 'mse' / 'l2' differ).
- Default optimiser 'adam' normalises every voxel separately -- the velocity-roughness source
  found in TVF / SyNGS (2026-09-30); greedy folds 0.05 % (finite-difference) on r16 -> r64.
- The old docstring called learning_rate 0.375 an untested stand-in copied from syn; it was
  tuned for greedy (tuned_greedy_2026_09_29).

### deformation_metrics.py
- `compute_bidirectional_dice` overwrites the geometry (origin / spacing / direction) of the
  label images passed in; it warps labels with 'nearestNeighbor' (ANTs recommends
  'genericLabel').

### motion.py
- `aff_metric` is used only by backend='ants' (ignored by the default PyTorch backends).
- backend='pytorch_batched' (the 'auto' choice for 3D+t rigid) reads only `num_bins` from
  `**kwargs`; anything else is silently ignored.
- backend='pytorch': type_of_transform='Translation' becomes a centre-of-mass shift
  (`robust_affine(mode='com_only')`), not an optimised translation; 'QuickRigid' / 'BOLDRigid'
  are plain rigid.

### template.py
- **Likely bug:** with the default `affine_every_iteration=False`, iterations >= 1 register each
  image with 'SyNOnly' but pass no initial transform (not the image's iteration-0 affine), so
  those registrations start from the identity -- wrong unless the inputs are already aligned
  to the template.

### image_utils.py
- `reflect_image` axis names 'LR' / 'AP' / 'SI' are fixed aliases for array axes 0 / 1 / 2,
  not resolved through the image's direction matrix -- wrong for images not stored
  axis-aligned in LPS / RAS order (compare `syntx.spatial.restriction_from_orientation`, which
  does resolve anatomical labels).

### core/inverse.py
- `update_inverse_field_nd` fixed-point solver: physical mode stops when max AND mean thresholds
  are met, normalised mode when EITHER is met.
- `relaxation` only affects `hybrid_lm`. Its fallback without `spacing` drops the thresholds.
- `integrate_time_varying_velocity_field`: `dt` is not derived from T (default 0.25);
  'midpoint' silently runs Euler in normalised mode; unknown `solver` values run Euler.
- `calculate_inverse_identity_error` and `compute_inverse_identity_error_nd` compose in opposite
  orders (inverse-then-forward vs forward-then-inverse). The former's `mean_error` counts
  out-of-grid voxels (error zeroed) in its denominator; the latter does no masking.
- Passing `X_phys` without spacing / origin / direction crashes (reverses None).

### core/utils.py
- `normalize_image`'s idempotency shortcut overrides `method`: any input in [0, 1] with
  max >= 0.5 is returned unchanged, even for `method='zscore'`, unless `force=True`.

### core/losses.py
- AnalyticalLNCC.backward is not exact: keeps only the centre voxel's own window and uses (F_c - CC*M_c) instead of (F_c - CC*sqrt(var_I/var_J)*M_c); equal only for equal local variances.
- ANTsPseudoLNCC / AnalyticalLNCC: N_window = window_size**dim even at border windows (pooling averages fewer voxels); ANTsPseudoLNCC saves `mask` and never uses it.
- local_ncc_loss_nd: even window_size not larger than the image crashes (pooling gives n+1); `pool_fn` computed, unused.
- BoxLNCCLoss: zero padding counted as data at the border; even kernel_size crashes; ratio not clamped; no mask; class default squared=True vs box_lncc_loss_nd squared=False.
- mattes_mi_loss_core: stride int(1/p) (p=0.6 uses every voxel); empty input returns a disconnected leaf (no gradient); mismatched fixed_weights silently ignored; parzen_weights maps NaN to bin 0 silently.
- mattes_mi_loss_nd: auto_mask threshold fixed at 0.01 regardless of intensity scale; fixed_weights silently recomputed when the masked set changes.
- compute_soft_distance_transform: -sigma*log(G*m) is not Euclidean (~d^2/(2 sigma), saturates ~18.4 sigma); per-axis sigma clipped to [0.5, 10] voxels in physical mode.
- distance_transform_loss: 'sdf_mse' == 'edt_mse'; tau=0/None divides by zero / crashes; without fields potential reduces to (G*m)**10.
- compute_image_distance_transform: output cast to input dtype (int/bool masks truncate); ANTsImage + return_ants=False returns ANTs (x,y,z) order; all-foreground slice gives 0 even with signed=True.

### core/smoothing.py
- get_boundary_mask: rim_size=0 returns an all-zero mask.
- smooth_displacement_field_bspline: coord_convention='zyx' reverses spacing/origin/mesh_size but not a sequence spline_distance; **kwargs ignored.
- apply_sobolev_green_operator: border_width, pad_to_fast, **kwargs ignored; FFT boundary is periodic.
- apply_dsti_green_operator: kwargs.get('s'/'spacing') dead code; Dirichlet zero is one voxel outside the grid.
- separable_gaussian_filter: kernel_type='compact' drops `mode`; unknown kernel_type falls back to 'bessel'; short sigma tuple raises (conv1d) or skips axes (conv3d); `spacing` ignored unless sigma_mode='physical' with scalar sigma; "ITK parity" unverified (radius rule differs).
- separable_1d_filter: kernel list not of length 2/3 returns x unchanged; Gaussian kernel cache never evicted.

### core/optimizers.py
- RegAdam: `dsti_alpha` stored, never read ('dsti' branches use sobolev_alpha; policy.py's dsti_alpha=0.035 works only because it equals sobolev_alpha); 'dsti' branches ignore spacing; spacing has no effect on Gaussian branches.
- RegAdam: unrecognised regularizer string (typo, 'bspline') is silently Gaussian-smoothed.
- RegAdam: step bound rescales the whole tensor on its global max, host sync every step.
- LARS has no momentum (old "prevents momentum collapse" claim removed).

### core/mps_kernels.py
- grid_sample_backward_mps: padding_mode other than 'border' treated as 'zeros'; mode / align_corners unchecked (callers must ensure bilinear, align_corners=True).
- Fixed-point limbs overflow silently beyond 65536 samples per input voxel.
