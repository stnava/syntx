# syntx 6.7.0

Fix and feature: a DWI motion preset. The generic motion estimator was biased on diffusion data (a still, already
corrected scan reported 1-1.7 mm of motion per diffusion volume and lost 19% white-matter FA). Evidence, numbers and
the open items: `docs/MOTION_DWI.md`.

- **`syntx.motion_presets`** (new, one place for modality presets): `DWI_MOTION_PRESET`, `get_motion_preset`.
  `motion_correct_grouped(..., preset="dwi", bvals=..., bvecs=...)` uses the full-precision `pytorch_batched` backend,
  registers every diffusion volume to the leave-one-out Gaussian-process prediction of its own gradient direction
  (`motion_reference.dwi_direction_targets`, per b-shell) after a first pass against the group mean, and resamples
  with a b-spline. Still scan c01r1: translation 1.51 -> 0.27 mm, WM FA ratio 0.81 -> 0.977.
- **Per-frame references**: `motion_correction(reference=[one ANTsImage per frame], backend="pytorch_batched")`.
- **One resampling**: group-B frames are resampled once from the original through the composed chain (cross transform,
  then own transform) instead of resampling a resampled frame.
- **Framewise displacement** (changed result): `jenkinson_framewise_displacement` is factored out of
  `calculate_framewise_displacement` (same numbers) and `motion_correct_grouped` now returns `fd` as the series-level
  Jenkinson FD in acquisition order in group A's space, with the first frame the only zero. The previous per-group
  Power FD is kept as `fd_group_power`. This changes `fd` for every caller of `motion_correct_grouped`.
- Not changed: generic `motion_correction` defaults, the adaptive backend, PET/BOLD paths.
- Not established: Jenkinson-FD thresholds for DWI; slice-dropout robustness (MI + correlation metric unchanged);
  speed (8-25 min per 96-volume session on a loaded machine, versus about 20 s before).

Tests: `tests/test_motion_dwi_preset.py` (11), plus `test_motion_reference.py`, `test_motion.py`,
`test_motion_batched.py`, `test_device_guard.py` (76 passed in total).
