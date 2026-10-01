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
