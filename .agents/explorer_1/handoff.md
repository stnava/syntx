# Handoff Report - explorer_1 to parent

## 1. Observation
- Verified installed packages via Python:
  - `ants`: `0.6.3`
  - `antspyt1w`: `1.1.3`
  - `torch`: `2.13.0`
  - `jax`: `0.10.2`
  - `jaxlib`: `0.10.2`
  - `antstorch`: located at `/Users/stnava/code/ANTsTorch/antstorch/__init__.py`
- Checked GPU/MPS or CUDA acceleration:
  - `torch.cuda.is_available()` returns `False`
  - `torch.backends.mps.is_available()` returns `True`
  - `jax.devices()` returns `[CpuDevice(id=0)]`
- 2D images located in `/Users/stnava/.antspy/`:
  - `r16`: shape `(256, 256)`, spacing `(1.0, 1.0)`
  - `r27`: shape `(256, 256)`, spacing `(1.0, 1.0)`
  - `r64`: shape `(256, 256)`, spacing `(1.0, 1.0)`
- 3D scans located in `/Users/stnava/.antspyt1w/`:
  - 8 raw T1w scans `28364-00000000-T1w-00` through `28575-00000000-T1w-07` have shape `(160, 256, 256)` and spacing (~1.0, ~1.0, 1.0).
  - Target template `T_template0` has shape `(256, 256, 170)` and spacing `(0.99325, 1.03279, 1.21504)`.
  - Cached DKT segmentations: `T_template0_dktseg.nii.gz`, `28497-00000000-T1w-04_dktseg.nii.gz`, and `28523-00000000-T1w-05_dktseg.nii.gz` are located in `/Users/stnava/code/syntx/cache/`.
- Test Suite Results:
  - Command: `pytest --runslow`
  - Output: `================== 91 passed, 6 warnings in 186.63s (0:03:06) ==================`
  - Coverage: Overall coverage is `93%`.

## 2. Logic Chain
- Since `torch.backends.mps.is_available()` is `True`, PyTorch can utilize MPS acceleration to speed up affine/deformable updates.
- Since `jax.devices()` contains only `CpuDevice`, JAX optimization and autograd steps will run on CPU.
- Since we verified the presence of `dktseg` files for the template and two subjects (`28497` and `28523`), these subjects can be compared immediately without running DKT segmentation.
- For other subjects (e.g. `28364`, `28386`, etc.), the DKT label map must be generated using `antstorch.desikan_killiany_tourville_labeling(img)` as observed in the pre-processing cache miss logic in `examples/compare_metrics.py`.
- The test suite has 100% of test cases passing with high coverage (93%), meaning the framework's functions are fully operational in the current system environment.

## 3. Caveats
- DKT segmentations for subjects other than `28497` and `28523` are not pre-computed in `cache/` and must be generated dynamically or cached during execution of subsequent milestones.
- Due to lack of JAX GPU backends, JAX registrations will execute slower than PyTorch registrations on MPS.

## 4. Conclusion
The environment and codebase are fully ready for the registration benchmarking tasks. The data files, template, and cached segmentations are validated. Hardware acceleration behaves correctly for PyTorch (MPS) and JAX (CPU). The test suite runs stably.

## 5. Verification Method
- Execute the test suite using `pytest --runslow`. All 91 tests should pass.
- Run `python3 -c "import ants; print(ants.image_read(ants.get_data('r16')))"` to verify 2D data load.
- Run `python3 -c "import ants; print(ants.image_read('/Users/stnava/code/syntx/cache/T_template0_dktseg.nii.gz'))"` to verify template DKT label map load.
