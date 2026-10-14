# syntx 6.8.0

BOLD motion preset. Evidence and numbers: `docs/MOTION_BOLD.md`.

- **`motion_correction(image, preset="bold")`** (`syntx.motion_presets.BOLD_MOTION_PRESET`): full-precision batched
  backend, narrow seeds, `medium` schedule. On nine SRPBS travelling-subject runs (QC FD 0.05-0.41 mm) it removes the
  constant 0.15 mm offset of the default adaptive backend (Pearson with the QC table 0.996 vs 0.987), and recovers
  injected known motion with 0.041 mm relative error (adaptive 0.083 mm). 17 s per 240-volume run vs 5 s.
  A conflicting `backend` / option raises; the `dwi` preset is rejected by `motion_correction` (it needs gradients: use
  `motion_correct_grouped`), and `motion_correct_grouped` rejects the `bold` preset.
- **`schedule_mode="medium"`** in `motion_batched`: `full` without the two middle L-BFGS stages and with the polish at 50%
  sampling (40% faster on BOLD at equal accuracy). Validated for BOLD only; the DWI preset keeps `full`.
- Not changed: the default `motion_correction` behaviour (no preset), the adaptive backend, DWI.
- **ASL is not solved on the one dataset checked** (PTBP PCASL, 3.1 x 3.1 x 6 mm): injected-motion error 0.35-0.41 mm for
  both backends (BOLD: 0.04 mm). Other ASL (breacher, SOCOM) was not tested; see the doc. Do not assume the `bold`
  preset is adequate for ASL.

Tests: `tests/test_motion_presets.py` (preset contents, recovery of known shifts, conflicts), plus the existing motion tests.
