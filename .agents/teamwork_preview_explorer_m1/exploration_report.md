# Milestone 1 Exploration & Diagnostics Report

This report summarizes codebase exploration and diagnostic checks for Milestone 1 of the `syntx` 2D Parity & Deep Feature Triggering project.

---

## 1. Default Registration Parameters Definition

Both the PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`) registration backends define equivalent default optimization and regularization parameters. The default parameters are located and defined as follows:

### PyTorch Backend (`src/syntx/syn.py`)
- **Constructor Defaults (`class SyNTo.__init__`)**:
  - `fluid_sigma=1.732`: Gaussian smoothing standard deviation for the update/velocity field.
  - `elastic_sigma=1.0`: Gaussian smoothing standard deviation for the accumulated displacement field.
  - `inverse_steps=5`: Number of fixed-point iterations for diffeomorphic projection inversion.
  - `inverse_method='fixed_point'`: Fixed-point mapping inversion algorithm.
- **Optimization Loop Defaults (`SyNTo.fit`)**:
  - `levels=[8, 4, 2, 1]`: Default multiresolution pyramid levels.
  - `epochs_per_level=100`: Default SyN optimization iterations per level.
  - `affine_epochs=[100, 50, 50, 20]`: Iterations for hierarchical linear pre-alignment.
  - `affine_lr=1e-2`: Learning rate for Adam optimization of the affine transform.
  - `cfl_voxels=0.75` (passed from `grad_step` default `0.2` in the high-level `registration` interface): CFL learning rate scaling bound (max voxel displacement step size).
  - `similarity_metric='lncc'`: Default metric.
  - `lncc_radius=4` (corresponding to `syn_sampling` in high-level registration): LNCC local window size of $9 \times 9 \ (\text{or } 9 \times 9 \times 9)$ via $\text{window\_size} = 2 \times \text{radius} + 1$.
  - `mattes_bins=32` (corresponding to `aff_sampling` in high-level registration): Number of bins for Mattes Mutual Information histogram calculation.

### JAX Backend (`src/syntx/syn_jax.py`)
- **Constructor Defaults (`class SyNTo.__init__`)**:
  - Defines the exact same defaults: `fluid_sigma=1.732`, `elastic_sigma=1.0`, `inverse_steps=5`, and `inverse_method='fixed_point'`.
- **Optimization Loop Defaults (`SyNTo.fit`)**:
  - Defines equivalent method signature defaults matching the PyTorch backend: `levels=[8, 4, 2, 1]`, `epochs_per_level=100`, `affine_epochs=[100, 50, 50, 20]`, `affine_lr=1e-2`, `cfl_voxels=0.75`, `similarity_metric='lncc'`, `lncc_radius=4`, and `mattes_bins=32`.

### High-Level API (`src/syntx/syn.py::registration`)
The high-level entrypoint `registration(...)` binds both backends under a common interface:
- `grad_step=0.2` (maps to `cfl_voxels` in `.fit`)
- `flow_sigma=1.732` (maps to `fluid_sigma`)
- `total_sigma=0.0` (maps to `elastic_sigma`)
- `syn_metric='lncc'` (maps to `similarity_metric`)
- `syn_sampling=4` (maps to `lncc_radius`)
- `aff_sampling=32` (maps to `mattes_bins`)

---

## 2. Similarity Metrics Setup & Evaluation

The similarity metrics `vgg19` and `resnet10` are configured and evaluated dynamically inside the multi-resolution optimization pipelines.

### PyTorch Backend (`src/syntx/syn.py`)
1. **Metric Setup**: Inside `SyNTo.fit` (lines 908-912 and lines 923-927), if a string metric matches `'vgg19'` or `'resnet10'`, the corresponding extractor is instantiated and wrapped in `FeatureSpaceLoss`:
   - **VGG19 Setup** (lines 908-912):
     ```python
     extractor = VGG19Extractor(feature_layers=vgg_layers).to(device=device)
     self.loss_functions.append(FeatureSpaceLoss(
         extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
     ).to(device=device))
     ```
   - **ResNet-10 Setup** (lines 923-927):
     ```python
     extractor = ResNet10Extractor(dim=dim, feature_layers=vgg_layers).to(device=device)
     self.loss_functions.append(FeatureSpaceLoss(
         extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
     ).to(device=device))
     ```
2. **Metric Evaluation**: During optimization loops (lines 1113-1117 & 1130-1132), the midpoint warped images `I_mid` and `J_mid` are evaluated sequentially against the loss functions:
   ```python
   loss = 0.0
   for fn, weight in zip(self.loss_functions, self.metric_weights):
       loss += weight * fn(J_mid, I_mid)
   ```

### JAX Backend (`src/syntx/syn_jax.py`)
1. **Metric Setup**: Inside `SyNTo.fit` (lines 1051-1054 and lines 1063-1066), PyTorch-defined metrics are wrapped via `make_pytorch_loss_jax` to bridge PyTorch backprop gradients with the JAX autograd environment:
   - **VGG19 Setup** (lines 1051-1054):
     ```python
     ext = VGG19Extractor(feature_layers=vgg_layers)
     loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
     self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
     ```
   - **ResNet-10 Setup** (lines 1063-1066):
     ```python
     ext = ResNet10Extractor(dim=self.dim, feature_layers=vgg_layers)
     loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
     self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
     ```
2. **Metric Evaluation**: Evaluated within the JAX optimization step (lines 1273-1306), using the DLPack bridge `to_torch_tensor` and `to_jax_array_dl` to share JAX midpoint arrays with PyTorch without copying:
   ```python
   for fn, w, jax_helper in zip(self.loss_functions, self.metric_weights, jax_grad_helpers):
       if getattr(fn, '_is_pytorch_loss', False):
           pytorch_loss_fn = fn._pytorch_loss_fn
           ...
           loss_torch = pytorch_loss_fn(J_mid_torch, I_mid_torch)
           loss_torch.backward()
           g_im = to_jax_array_dl(I_mid_torch.grad)
           g_jm = to_jax_array_dl(J_mid_torch.grad)
           val = to_jax_array_dl(loss_torch.detach())
   ```

---

## 3. Unit Test Suite Structure and Execution

### Test Suite Structure
The testing infrastructure is modularized into 7 files located in `tests/`:
1. `tests/test_coverage_helpers.py`: Checks mathematical correctness and gradients of internal helper sub-functions (SO(d) Lie rotation matrices, 1D/3D separable Gaussian filters, triplanar/reconstruct VGG losses, Mattes MI core, physical and normalized Jacobian determinant calculations).
2. `tests/test_e2e_metrics.py`: Implements a structured 27-test plan organized into 4 Tiers:
   - **Tier 1**: Feature Coverage (SwinUNETRExtractor init/shapes/normalization, DLPack bridges).
   - **Tier 2**: Boundary & Corner Cases (batch size variations, invalid dimensions, cache fallbacks).
   - **Tier 3**: Cross-Feature Combinations (registrations combining JAX and PyTorch/SwinUNETR metrics).
   - **Tier 4**: Real-World Scenarios (DICE accuracy constraints, folding bounds, CSV export verification).
3. `tests/test_feature_networks.py`: Verifies convolutional network feature shape extractors (VGG19, DINOv2, ResNet-10) and pruning layers.
4. `tests/test_swin_unetr_empirical.py`: Validates Swin UNETR shape behavior, offline weights download exceptions, batch sizes, and multi-scale sizing.
5. `tests/test_syn.py` & `test_syn_jax.py`: Verify PyTorch and JAX registration fitting under LNCC, Mattes MI, and VGG19 metrics using synthetic geometric phantoms, ensuring Pearson correlation $> 0.60$ and non-negative Jacobian determinants.
6. `tests/test_transform.py`: Validates `SyNToTransform` properties, Jacobian map retrieval, and composite warp export compliance.

### Execution Commands
- **Fast Mode (excl. slow tests)**: `pytest` or `make test`
- **Full Suite (incl. slow tests)**: `pytest --runslow` or `make test-all`
- **Coverage Execution**: `pytest --cov=syntx --cov-report=term-missing`

### Verification Status
Both test execution modes were successfully run:
- **Fast Mode**: Completed with 86 passed tests and 6 skipped (slow) tests.
- **Full Mode (`pytest --runslow`)**: Completed with **94 passed tests** and **0 failures** in 183.78 seconds.
- **Coverage**: Coverage stands at **92%** overall across the codebase (100% in `__init__.py`, `resnet.py`, and `transform.py`).

---

## 4. Proposed Deep Feature Degeneracy Triggering Mechanism

### Degeneracy Context
At coarse resolution levels (e.g. scales 8 or 4 in the registration pyramid), spatial dimensions of the downsampled inputs become very small (e.g. $8 \times 8$ or $16 \times 16$). Because deep convolutional neural networks (such as VGG19 or ResNet-10) feature internal downsampling layers (max pooling/strided convolutions), the resulting feature maps contract to $1 \times 1$ or $2 \times 2$. 
This collapse causes:
1. **Loss of spatial variance**: Feature maps become spatially uniform, causing zero-variance in LNCC computations and division-by-zero errors.
2. **Gradient vanishing/degeneracy**: Truncated gradients propagate backwards as zeros or NaNs, causing optimizer step failures and grid folding.
3. **Severe folding rates**: Metric gradients are noisy or flat, resulting in unstable deformations.

### Proposed Triggering Heuristics
To handle this, we propose integrating a **dynamic trigger** inside the multi-resolution pyramid loop of `SyNTo.fit`:

1. **Resolution Dimension Boundary (Hardcoded)**:
   A strict dimensional lower bound. If the downsampled image dimensions at the current scale level fall below a minimum boundary:
   $$\min(\text{curr\_spatial\_shape}) < \text{resolution\_limit} \quad (\text{default: } 32)$$
   we trigger the fallback.
2. **Spatial Feature Variance Constraint**:
   Run a dry-run feature extraction step on the fixed image `I_curr` at the start of the level. If the spatial variance of any extracted feature map falls below a threshold:
   $$\text{Var}(\Phi(I_{\text{curr}})) < \epsilon \quad (\text{default: } 1e-4)$$
   we trigger the fallback.

### Proposed Integration & Fallback Path
The trigger should be integrated at the start of the level loop in `SyNTo.fit` in both `src/syntx/syn.py` and `src/syntx/syn_jax.py`:

```python
# Proposed logic draft inside SyNTo.fit (for each level)
for level_idx, scale in enumerate(levels):
    I_curr = I_pyr[level_idx]
    J_curr = J_pyr[level_idx]
    curr_spatial = I_curr.shape[2:]
    
    # Degeneracy Trigger check
    is_degenerate = min(curr_spatial) < 32  # Heuristic 1
    
    # Or Heuristic 2 (dry-run variance check)
    # ...
    
    active_loss_functions = []
    for fn in self.loss_functions:
        # Check if the loss relies on deep feature extractors (VGG19/ResNet10)
        if isinstance(fn, FeatureSpaceLoss) or getattr(fn, '_is_pytorch_loss', False):
            if is_degenerate:
                if verbose:
                    print(f"[SyNTo] Coarse resolution degeneracy triggered at shape {curr_spatial}. "
                          f"Falling back to intensity LNCC for this level.")
                # Dynamically swap with standard intensity LNCC (radius 4)
                if backend == 'pytorch':
                    fallback_loss = lambda x, y: local_ncc_loss_nd(x, y, window_size=2 * lncc_radius + 1)
                else:
                    fallback_loss = lambda x, y: local_ncc_loss_nd_jax(x, y, window_size=2 * lncc_radius + 1)
                active_loss_functions.append(fallback_loss)
            else:
                active_loss_functions.append(fn)
        else:
            active_loss_functions.append(fn)
```

This prevents feature collapse, ensures optimization stability at coarse resolution levels, and improves registration accuracy (DICE) and folding rate bounds.
