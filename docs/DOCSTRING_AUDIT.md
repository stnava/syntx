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
| syn_jax.py, syngs_jax.py, tvf_jax.py, tvf_adj.py, motion_batched.py (+ tvf.py Catmull-Rom fix) | done | v5.4.92 |
| diagnose, policy, generators, classifier, resnet, perf_tracking, provenance, contract, reporting, tabulate, cli, __init__ | done | v5.4.92 |
| spatial.py, core/grid.py, core/affine.py, core/jacobian.py, core/pipeline.py, core/__init__.py | done | v5.4.92 |
| viz/* (figures, reports, core, gallery, stats, modality_report, qc_sections, colormaps, __init__) | done | v5.4.92 |

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

### syn_jax.py
- `SyNJAX.fit`: `kwargs.get('use_analytical_gradients', True)` overrides the explicit argument (False has no effect); only 'dinov2' switches to autograd.
- `warp_images_jax` is jitted with no static args but receives the str `interpolator` -> TypeError; the autograd path (only via 'dinov2') crashes.
- All Gaussian sigmas (fluid / elastic / in-loop inverse / pyramid) are in voxels (`sigma_mode='voxel'`, spacing ignored); PyTorch uses mm with the same values from `syntx.syn`.
- `fit` from `syntx.syn`: `affine_epochs` not passed -> default [100, 50, 20] Mattes-MI affine stage re-optimises on top of `init_M_phys` (PyTorch does not optimise the affine).
- `get_affine_matrix_jax`: non-'Affine' types give rotation x isotropic scale ('Rigid' fits scale, 'Translation' fits rotation and scale).
- Per-level `curr_metric_weights` used only on the 'lbfgs' path; nested per-level `syn_metric_weights` crashes the main loop.
- Divergence retry runs at most once (`if`, not loop); ignores optimizer_type / regularizer / per-level weights.
- 'lbfgs': r2l inverse uses moving geometry for a fixed-grid field; smoothed (not true) gradient given to L-BFGS-B; maxiter=1 per epoch.
- 'soft_dice' / 'dice' wrap torch / np.array / float inside a jitted value_and_grad -> fails under tracing.
- `forward` / `forward_inverse` re-permute M_phys already in tensor order (affine applied with reversed axes vs `fit`); expect (x,y,z) input and use constructor geometry, unlike `fit`.
- With `initial_grid`, analytical path normalises with FIXED geometry, autograd path with MOVING.
- `local_ncc_loss_nd_jax(use_ants_pseudo_gradient=True, squared=False)` actually computes cc^2: default 'lncc' is cc^2, autograd 'lncc' is signed cc.
- `compute_physical_jacobian_determinant_jax` omits the (n-1)/2 normalised-to-voxel factors (correct only for equal axis sizes).
- Anderson inverse safeguard compares scaled composition error with unscaled step residual.
- hybrid_lm inverse: periodic jnp.roll Jacobian; try/except around solve never triggers under tracing; no spacing -> silently Anderson.
- `update_inverse_field_nd_jax` fixed point with `W_inv_disp=None` fails; velocity integration 'midpoint' -> Euler in normalised branch.
- `rprop_update_step_jax`: `lr` unused; `boundary_suppression_thresh`, `velocity_clamp`, `cfl_max` never used; `_apply_dsti_green_operator` spacing unused; `_apply_sobolev_green_operator` indexes ITK-order spacing by tensor axis.
- `jax_grid_sample_bspline` has no prefilter (approximating, blurs).

### syngs_jax.py
- Not a mirror of PyTorch: `shoot` evolves v by EPDiff (PyTorch: stationary smoothed v0); loss always LNCC; no moving geometry; defaults differ (fluid_sigma 1.0 vs 3.0, n_steps 5 vs 6, inverse_identity_weight 1.0 vs 0.5).
- Anisotropic spacing order bugs: ITK-order spacing zipped with tensor-order shapes (`shoot`, `forward`); `fit` passes ITK-order spacing as `spacing_zyx` and divides (z,y,x) gradients by it in the CFL step.
- `symmetric=False`: `fit` passes +v0 as the inverse velocity (forward / get_inverse_warp use -v0); its gradient is discarded.
- `init_velocities_from_image_gradients` resizes a channels-last gradient with a channels-first interpolator.
- Inert: `image_grad_clip`, `velocity_clamp` (hard-coded 50), `cfl_max`, `elastic_sigma`; forward `multipoint_loss`; fit `similarity_metric`, `moving_*`, `elastic_sigmas`, `fast_smooth`; `integrate_momentum_jax` `t_end`, `return_trajectory`, `alpha`.
- 'midpoint' solver silently Euler; `integrate_momentum_jax` n_steps=6 vs PyTorch 8.

### tvf_jax.py
- `syntx.tvf(backend='jax', regularizer='gaussian')` passes `alpha=None` (tvf.py:1901) -> `float(None)` TypeError in `fit` (tvf_jax.py:942); spectral regularisers are unaffected.
- `affine_epochs` default 100 not passed by `syntx.tvf` -> extra Mattes-MI affine stage on top of T_init (PyTorch has none).
- `initial_transform` physical matrix stored as T_init without the grid conversion (applied in the wrong space).
- `tvf_registration_jax`: learned affine never exported (TODO); `ants.invert_ants_transform` on path strings; keeps (x,y,z) order while helpers assume (z,y,x); `tempfile.mktemp` files never deleted.
- Spacing: `forward` zips (x,y,z) spacing with tensor shapes and scales by N/n not (N-1)/(n-1); CFL / Sobolev steps pair (x,y,z) spacing with (z,y,x) axes.
- fluid / elastic sigma square-rooted and used in voxels (PyTorch: mm).
- Inert: `similarity_metric` (always LNCC), `moving_*`, `image_grad_clip`, `velocity_clamp`, `cfl_max`, `use_analytical_gradients`.
- `antisymmetric` (default True) means time-mirror averaging (different meaning, PyTorch default False).
- Early stopping recomputes the loss with `forward` defaults, not the training settings.
- Defaults differ from PyTorch: 1 vs 4 steps per interval, epochs [100,100,50] vs [100,100,20], lr 0.15 vs 1.0, multipoint [0,1] vs [0.5], constant_speed False vs True.

### tvf_adj.py
- `integrate_svf` does nothing (loop body `pass`).
- `integrate_forward`: `n_steps` unused; normalisation uses size/2; returns T+1 grids.
- Local `get_physical_grid_torch` / `physical_to_normalized_torch` use opposite conventions (not inverses).
- `TVFRegistrationAdjoint`: `initial_transform` unused; coarse voxel-unit velocities not rescaled on upsampling; final upsample raises if the last level is skipped; the adjoint is approximate.
- `tvf_registration_adjoint`: lr=50 vs class 0.5; device='mps' default; direction ignored; temp files never deleted.

### motion_batched.py
- `batched_rigid_register_pass`: `num_bins=18` default overridden by `motion_correction` (32); temp dirs never cleaned; LBFGS schedule `lr_t` / `lr_r` ignored (lr fixed 1.0).
- `batched_group_bias_register_pass`: `verbose`, `max_level` unused; level>1 / `corr_weight` branches unreachable.
- Frames assumed to share the reference grid (unchecked); scaling assumes non-negative intensities.

### __init__.py
- `__all__` lists `plot_comparison`, `plot_structural_comparison`, which do not exist -> `from syntx import *` raises AttributeError (verified).
- Import always sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0, overwriting a user value.

### contract.py
- antsxdwi does not import `syntx.contract`; it has its own drifting copy (with a `denoise` field).

### diagnose.py
- Tier-2 weights `src/syntx/models/diagnostic_resnet10_3d.pth` do not exist -> deep classifier never runs; its exceptions are swallowed.
- Labels never produced: `is_contrast_enhanced`, MONO_MODAL_INTER, MRI_FLAIR / ADC / OTHER, modality UNKNOWN; HEART only from the classifier.
- Any non-HU data outside [0, 1.05] (incl. negative) labelled POSITIVE_FLOAT; non-CT body part from shape / extent only (any 2-D or 3-D < 260 mm -> BRAIN).
- `diagnose_pair` defaults fast=True, `diagnose_image` fast=False.

### policy.py
- `auto_reg` ignores regularizer, sobolev_alpha, dsti_alpha, grad_step, flow_sigma, total_sigma, guided_weight, parameters -> thorax 'dsti1' and pelvis sobolev_alpha=2.0 rules have no effect; `explanation` describes unapplied settings.
- `details['is_roi_crop']` never set -> ROI branch unreachable; `to_dict` omits dsti_alpha, not valid as auto_reg(**d).
- Suspected (read, not run): auto_reg puts `similarity_metric` in kwargs when a policy exists and tvf_registration rejects it -> thorax-CT (TVF) policies fail.

### classifier.py / resnet.py
- `predict_diagnosis_deep` defaults to MPS without CUDA; `preprocess_volume_for_classifier` fails on 2-D input.
- `ResNet10(num_classes=...)` has no effect; shortcut layers named `shortcut` vs MedicalNet `downsample`, loaded with strict=False -> pretrained weights may silently not load.

### generators.py
- `benchmark_data('mbhard')`: missing Mindboggle files -> silently writes a synthetic 64^3 sphere/ellipsoid pair under the Mindboggle file names in ~/.syntx/benchmark_data/mbhard/, returned with the Mindboggle description and reused later as if real (verified in code).
- `CrossProductGenerator`: ANTsImage base used in ANTs order with spacing[0] applied to the W component (swapped for anisotropic images); rotation in normalised coords not rigid for non-square images.
- `compute_physical_l2_norm` is an integral sqrt(dV sum |u|^2) (grows with image size), not a norm in mm; unknown `magnitude_level` string means 1; `temp_seed` restores only CPU torch / NumPy RNG.

### perf_tracking.py
- Git-commit lookup runs before the directory is created (None on a new path); `detect_regressions` does not cast to float, NaN never flagged, baseline 0 skipped, NaN/inf written as non-standard JSON.

### provenance.py
- Capture patches module attributes only: pre-imported references / internal imports (robust_affine inside registration, registration inside auto_reg) not recorded; nested capture records nothing; only fit() kwargs recorded.
- `assert_manifest_complete` dirty-diff check never fails (diff_sha256 always present); `environment()` imports jax / antstorch; `code_state` splits untracked paths on whitespace.

### tabulate.py
- `correlation_matrix`: constant column -> 0 (diagonal too) in torch, NaN in numpy; float64 fails on MPS. `roi_mean_timeseries` truncates labels to int, no same-grid check.

### cli.py
- `--report` has no effect (store_true with default=True).
- `register` defaults (grad_step 0.25, flow_sigma 5.0, optimizer reg_adam) override library defaults (syn 0.4 / 2.4 / cfl; tvf cfl); SyN-only options ignored for TVF and --optimizer for SyN, yet recorded in metrics.json.
- `--regularizer` choices omit 'bspline' though the TVF branch tests for it; `cmd_info` falls back to stale '4.0.2'; Jacobian from the deformable field only.

### spatial.py
- `get_spatial_coordinate_grid`: ANTs-order shape indexed as z,y,x -> x / z ranges swapped for non-cubic images (confirmed on CPU; no live callers).
- `jacobian_determinant` 2-D: axis 0 divided by spacing[1], axis 1 by spacing[0] -> swapped for ANTs-layout anisotropic input (confirmed: 1.2 vs 1.1); ignores direction. 3-D: only the direction diagonal used (wrong for oblique). Tensor converted only with ref_image and batch 1.
- `restriction_from_orientation`: bare 'I' raises; 'z' on 2-D raises IndexError; "exactly one of" not enforced.
- `deformation_gradient` / `jacobian_determinant`: tensor-layout ndarray never converted; spacing / direction with tensor + ref_image silently ignored.
- `tensor_to_image` does no transpose (expects ANTs order); `image_to_tensor` defaults to_zyx=False (tvf_jax.py relies on it).
- `physical_to_normalized_torch_cached` caches nothing.
- Direction inverted by transpose in some helpers, general inverse in others (disagree for non-orthonormal direction).
- `lps_to_ras` / `ras_to_lps` cast to float32.

### core/jacobian.py
- `_spatial_jacobian_nd(method='bspline')`: kernel [-1,-8,0,8,1]/12 has wrong outer signs -> 5/3 x the true slope (verified); affects compute_jacobian_determinant_nd / compute_physical_jacobian_determinant / SyNToTransform.get_jacobian_determinant(method='bspline').
- `compute_jacobian_determinant_nd`: spacing order and component order differ across its bspline / physical / normalised paths; passing physical_spacing silently switches to mm; unbatched check `dim() == dim` never fires; channels-first detection misfires when the last spatial size is 2 or 3.
- syn.py:1304 hinge penalty: spacing reversed twice; in-loop warp probably normalised but treated as mm.
- `compute_physical_jacobian_determinant`: non-physical path's direction / spacing cancel (no effect); physical path ignores direction; origin / kwargs ignored.

### core/grid.py
- Suspected: `prepare_mid_images_and_gradients_torch` (analytic path, not syn's default) applies (x,y,z)-ordered image gradients to (z,y,x) physical warps -> x / z swapped; C > 1 without precomputed gradients gives a wrong layout; warp_*_inv unused.
- `_image_spatial_gradient` uses torch.roll (border wraps); AnalyticalGridSample is approximate (no input gradient, no outside-region zeroing).
- `grid_sample_bspline_torch` has no prefilter (smooths); non-'zeros' padding acts as 'border'.
- `_generic_label_sample` 'zeros' padding gives the lowest label present; int input returns float.
- `grid_sample_nd`: unknown interpolator silently falls back to `mode`; analytical gradients even for nearest. Deterministic backward treats non-'border' padding as 'zeros'.

### core/affine.py
- `HierarchicalAffine`: 'Translation' still optimises rotation (== 'Rigid'); transform_type unvalidated.
- `parse_ants_affine` composes first-item-first (reverse of apply_transforms convention); Euler / Similarity objects skipped silently.

### core/pipeline.py / core/__init__.py
- `normalize_and_tensorize`: winsorize_quantiles ignored (2/98 hard-coded); zip drops unmatched channels; device ignored for jax.
- core `__all__` omits RegAdam / SobolevAdam / GaussianAdam; compute_jacobian_hinge_penalty not re-exported.

### viz/reports.py
- `create_affine_benchmark_report`: syntx affine time is the constant 2.8 s, so the reported speedup is fabricated (verified); ANTs time 28.5 s / Dice 0.3472 defaults hardcoded; reads `results/...` relative to cwd; "90 / 90", comparison table and protocol text are fixed; provenance ignored.
- `create_population_benchmark_report`: hyper-parameter table, "/ 90", "TVF beats Sobolev in 88/90" hardcoded; speedup always vs Sobolev time; cohort falls back to idx < 40 = intra; missing ANTs folding counts as 0; provenance and list baseline_source ignored.
- `create_registration_report`: "LNCC (w=9)" is window 5 (image_compare ignores window_size); "SSIM" / "NCC" reported as ssim-1 / ncc-1; no inverse map -> 0 mm, no Jacobian -> 0 % folding, Dice errors -> "N/A" silently; Dice not filtered to [0,1]; show_report / kwargs / fixed_name / moving_name no effect; "Verified Provenance" badge always shown; figures never closed.
- `_compute_jacobian_stats` converts to NumPy before `jacobian_determinant` (skips the tensor-order reversal); all-ones map on any error.
- `build_engine_provenance`: "syntx_version" is the cwd's git HEAD; missing flags become False not "N/A".
- `create_benchmark_report`: missing keys count as 0.0; means not restricted to paired indices; output dir not created.

### viz/figures.py
- Redefines `get_dkt_colormap`, `dkt_colormap`, `get_dkt_label_color_dict` (twice; tab20 version wins), shadowing viz.colormaps.
- All displacement plots (deformation grid, correspondence vectors, vector field, 4-panel A): mm added to pixel coords without dividing by spacing; vertical sign inverted; field and background sliced separately (different slices when slice_idx=None).
- `render_standard_4panel`: panels can show different slices; folding / min det / inverse error are per slice; error cites "GEMINI.md Section 3"; failing panels silently zeros.
- `render_input_pair_figure`: intended docstring sits after the first statement (string left in place, real docstring added); moving sagittal title prints wrong index; 3-D crop_background never crops; dead aspect code; colorbar from axial panel only.
- `plot_deformation_tensor_rgb`: R/G show y/x not x/y; `alpha` unused; RGB image has default geometry. `compute_deformation_tensor_rgb` NumPy fallback permutes derivative axes (not F).
- `plot_time_varying_velocity_grid`: channels transposed vs extract_slice; silently substitutes midpoint warps when velocities ~0; dict path unreachable; bending energy spacing swapped; "(px)" unverified.
- `render_label_alignment_figure`: rot90 sagittal mirrored vs extract_slice; 3-D only; colorbar lists first 16 labels. `extract_2d_slice` ignores ref_image; `plot_edge_overlay` resize import outside the try.

### viz/core.py
- `verify_anatomical_orientation` always returns True.
- `corner_watermark`: `corner` ignored; unseeded noise.
- `extract_slice` vectors: sagittal not column-flipped; vertical component not negated; components not reoriented; channel 0/1 fixed as u_y/u_x for every plane (wrong in-plane components for coronal / sagittal). 3-D arrays with last axis 2/3 treated as 2-D vector fields; 2-D arrays not row-flipped; integer plane outside 0-2 raises.
- `prepare_image`: arrays without ref_image indexed as ANTs order; failed reads swallowed; any leading axis of size 2/3 taken as channels.

### viz/colormaps.py
- HSV not HSL (`lightness` is HSV value); `get_dkt_colormap` colours by label ID, `build_dkt_label_palette` by rank -> same label, different colours; "3.0" stays a string, floats truncated.

### viz/stats.py
- `plot_label_overlap_stats`: plain {region: list} dict fails; in dict / array mode the three boxes are identical.
- `plot_jacobian_distribution`: "Fully Diffeomorphic" whenever no voxel <= 0, unmasked; `.numpy()` before `.detach()`.
- `plot_loss_convergence`: hardcoded labels, not exported, no output dir creation.

### viz/gallery.py / viz/modality_report.py
- gallery: light-theme figure rendered and unused; `title` only sets <title>; fallback version "1.1.8".
- modality_report: `matplotlib.use("Agg")` global side effect; output dir not created; kpis_html / description unescaped; title_override only <title>; footer text with `brand`.
