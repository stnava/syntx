# Handoff Report — Milestone 1 Exploration & Diagnostics

## 1. Observation
I directly observed the following codebase definitions, test executions, and structural layouts:

### Default Registration Parameters
- **PyTorch backend (`src/syntx/syn.py` line 767)**:
  `class SyNTo(nn.Module):`
  `    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, direction=None, fluid_sigma=1.732, elastic_sigma=1.0, transform_type='Affine', inverse_method='fixed_point', inverse_steps=5):`
  And inside `.fit(...)` (line 828):
  `    def fit(self, fixed_image, moving_image, levels=[8, 4, 2, 1], epochs_per_level=100,`
  `            affine_epochs=[100, 50, 50, 20], affine_lr=1e-2, cfl_voxels=0.75,`
  `            similarity_metric='lncc', use_analytical_gradients=True,`
  `            lncc_radius=4, mattes_bins=32, ...`
- **JAX backend (`src/syntx/syn_jax.py` line 920)**:
  `class SyNTo:`
  `    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, direction=None, fluid_sigma=1.732, elastic_sigma=1.0, transform_type='Affine', inverse_method='fixed_point', inverse_steps=5):`
  And inside `.fit(...)` (line 972):
  `    def fit(self, fixed_image, moving_image, levels=[8, 4, 2, 1], epochs_per_level=100,`
  `            affine_epochs=[100, 50, 50, 20], affine_lr=1e-2, cfl_voxels=0.75,`
  `            similarity_metric='lncc', use_analytical_gradients=True,`
  `            lncc_radius=4, mattes_bins=32, ...`

### Similarity Metrics Setup and Evaluation
- **PyTorch backend (`src/syntx/syn.py` lines 908-912 and lines 923-927)**:
  ```python
  elif metric_name_lower == 'vgg19':
      extractor = VGG19Extractor(feature_layers=vgg_layers).to(device=device)
      self.loss_functions.append(FeatureSpaceLoss(
          extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
      ).to(device=device))
  ```
  Evaluated in loop (lines 1113-1117 & lines 1130-1132):
  ```python
  loss = 0.0
  for fn, weight in zip(self.loss_functions, self.metric_weights):
      loss += weight * fn(J_mid, I_mid)
  ```
- **JAX backend (`src/syntx/syn_jax.py` lines 1051-1054 and lines 1063-1066)**:
  ```python
  elif metric_name_lower == 'vgg19':
      ext = VGG19Extractor(feature_layers=vgg_layers)
      loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
      self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
  ```
  Evaluated in step (lines 1273-1306) via `make_pytorch_loss_jax`, passing JAX arrays to PyTorch via DLPack zero-copy (`to_torch_tensor`) and back (`to_jax_array_dl`).

### Unit Test Execution
- Executed `pytest` command:
  - Command: `pytest`
  - Result: `86 passed, 6 skipped, 6 warnings in 86.37s`
- Executed `pytest --runslow` command:
  - Command: `pytest --runslow`
  - Result: `94 passed, 6 warnings in 183.78s`

---

## 2. Logic Chain
1. By examining `src/syntx/syn.py` and `src/syntx/syn_jax.py`, I verified that optimization parameters (`levels`, `epochs_per_level`, `affine_epochs`, `affine_lr`, `cfl_voxels`) and regularization parameters (`fluid_sigma`, `elastic_sigma`, `inverse_steps`, `inverse_method`) are defined as equivalent defaults.
2. By tracing the execution of `SyNTo.fit` in both backends, I confirmed that `vgg19` and `resnet10` strings instantiate deep feature networks, which are evaluated directly on midpoint images (PyTorch) or bridged through `make_pytorch_loss_jax` using DLPack tensors (JAX).
3. Running `pytest` and `pytest --runslow` verified that all 94 unit tests pass successfully with no failures.
4. Designing a triggering heuristic at coarse scales addresses spatial variance collapse at low resolutions (e.g. dimensions $< 32$), preventing division-by-zero errors or severe folding rates.

---

## 3. Caveats
- No implementation of source code changes was performed (read-only constraint).
- The execution of `examples/evaluate_all_metrics.py` depends on simulated data fallback if real scan files (`T_template0.nii.gz` and `I1499279_Anon_20210819142214_5.nii.gz`) are missing from their expected locations under `/Users/stnava/.antspyt1w/` and `/Users/stnava/.antspymm/`.

---

## 4. Conclusion
The codebase contains fully aligned PyTorch and JAX SyN registration implementations. Both backends successfully interface with deep feature-space loss networks (VGG19, ResNet-10) and lazy-loaded MONAI Swin UNETR networks.
All 94 tests in the test suite execute successfully with 100% pass rates. 
To avoid registration failures at coarse resolution stages, a dynamic triggering mechanism must be integrated in the multi-resolution scale loop of `SyNTo.fit` to check if `min(curr_spatial_shape) < 32` and fall back to raw intensity LNCC.

---

## 5. Verification Method
To verify these findings independently:
1. Run `pytest` or `pytest --runslow` in the project root to ensure all tests pass:
   ```bash
   pytest --runslow
   ```
2. Verify that the detailed exploration report is present at:
   `/Users/stnava/code/syntx/.agents/teamwork_preview_explorer_m1/exploration_report.md`
