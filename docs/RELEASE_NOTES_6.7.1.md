# syntx 6.7.1

Speed release for the DWI motion preset (6.7.0): a 96-volume 1.5 mm session now takes about 3 minutes on this machine
(186 s measured, machine idle) instead of 8-25 minutes, with accuracy unchanged within noise (still scan: WM FA ratio
0.982, v6.7.0 0.977). Evidence, profile, what was tried and rejected: `docs/MOTION_DWI.md` ("Speed work").

- **Batched per-frame references**: `batched_rigid_register_pass` (and `motion_correction(reference=[...],
  backend="pytorch_batched")`) takes one reference per frame in a single batched call (`seed_mode="narrow"` only). The
  shared-reference path is bit-identical to before (test against the previous implementation). The mutual-information
  intensity range is per reference image.
- **`resolution_cap_mm` for the full-precision batched backend** (registration at a coarser resolution; frames are still
  resampled at native resolution), plus `seed_mode`, `schedule_mode`, `max_batch_frames` passthrough in
  `motion_correction` (`pytorch_batched` only).
- **Vectorised `dwi_direction_targets`**: one matrix product (164 s -> 1 s on 96 volumes; same numbers).
- **Preset** (`syntx.motion_presets.DWI_MOTION_PRESET`): registration at 2 mm, narrow seeds in the per-direction pass,
  linear resampling in pass 1, `loo_batch_frames` (None = one batched call; 1 = one frame per call, exact reference
  behaviour at about twice the time).
- Not changed: generic `motion_correction` defaults, the adaptive backend, PET/BOLD paths.
- Known limits: batched L-BFGS couples frames through one line search; on a scan with slice dropout per-volume
  parameters differ from one-frame-per-call by 0.14 mm on average. 2-minute target not reached (3 minutes).

Tests: `tests/test_motion_batched.py` (per-frame references), `tests/test_motion_dwi_preset.py` (matrix targets equal the
direct formula), plus the existing motion tests.
