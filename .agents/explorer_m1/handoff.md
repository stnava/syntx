# Handoff Report: Syntx Investigation and Design Proposal

## 1. Observation

A detailed, read-only code and workspace investigation was carried out on `syntx`. The following file structures, implementations, and dependencies were observed:

### A. Existing Registration and Mathematical Framework
The core registration functionality is implemented in both PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`).
- **Optimization Model**:
  The `SyNTo` class (line 767 in `syn.py`, line 800 in `syn_jax.py`) implements Symmetric Normalization with:
  - Linear/Affine stages (Translation, Rigid, Similarity, Affine) parameterized using SO(d) Lie algebra rotations to prevent gimbal lock (e.g. `get_rotation_matrix` at line 9 in `syn.py` and `get_rotation_matrix_jax` at line 175 in `syn_jax.py`).
  - Dense deformable steps mapping symmetric forward/backward velocity fields, composing deformation grids, and calculating physical Jacobian determinant maps.
  - Voxel-level update clipping and Gaussian regularization mapping.
- **Physical Exporter**:
  `src/syntx/transform.py` defines `SyNToTransform` (line 7), which holds the coordinate displacement grids and maps them to ITK/ANTs physical coordinate systems (LPS/RAS) using `_to_physical_displacement` (line 109).
- **Existing Metric Functions**:
  - `local_ncc_loss_nd` (line 527 in `syn.py`) and `local_ncc_loss_nd_jax` (line 543 in `syn_jax.py`): Local Normalized Cross Correlation (LNCC) for N-D images.
  - `mattes_mi_loss_core` (line 586 in `syn.py`) and `mattes_mi_loss_nd_jax` (line 623 in `syn_jax.py`): Parzen-windowing Mattes Mutual Information using third-order B-splines.
  - `FeatureSpaceLoss` (line 295 in `src/syntx/features.py`): Unified wrapper that evaluates LNCC on feature maps from deep vision models.

### B. Existing Deep Feature Networks
`src/syntx/features.py` defines modular feature extractors subclassing `FeatureExtractor` (line 9):
- **`VGG19Extractor`** (line 26): Discards layers beyond max requested layer to optimize memory. Supports axial, coronal, and sagittal slice reconstruction for 3D inputs under `mode='lncc_3d'`.
- **`DINOv2Extractor`** (line 63): Uses ViT backbones, padding input shapes to patch-size (14) divisibility.
- **`ResNet10Extractor`** (line 116): Supports 2D or native 3D convolutional channels.
- **`SwinUNETRExtractor`** (line 178): A 3D transformer encoder (requires `monai` package, lazily imported).

### C. Existing Testing Framework & Test Failure Observation
- Configured in `pyproject.toml` (lines 36-40):
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  addopts = "--cov=syntx --cov-report=term-missing"
  ```
- Command to run: `pytest`
- Total tests: 111 test cases collected under `tests/`.
- **Test Failure Observation**:
  During verification, running the test suite resulted in a failure in JAX-based helper testing:
  ```
  FAILED tests/test_syn_jax.py::test_new_jax_helpers - ValueError: Non-hashable...
  TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'
  tests/test_syn_jax.py:239: ValueError
  ```
  Line 239 in `tests/test_syn_jax.py` calls `prepare_mid_images_and_gradients_jax` with 7 arguments instead of 8:
  ```python
  I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_jax(
      warp_l2r, warp_r2l, I_curr, J_curr,
      True, spacing_arg, identity
  )
  ```
  This causes the JAX array `identity` to be passed as the static argument `spacing` (marked static via `@partial(jax.jit, static_argnums=(5, 6))`), which triggers JAX's unhashable type compiler error.

### D. Available Libraries and Packages
As specified in `pyproject.toml` (lines 15-24) and `requirements.txt`:
- `numpy`, `scipy`, `matplotlib`
- `antspyx` (ANTs Py for image reading, writing, coordinate bounds, and metric baselines)
- `torch`, `torchvision` (for PyTorch models, GPU/CPU extraction, and optimization)
- `jax`, `jaxlib` (for JAX backend optimization, autograd, and GPU/CPU compilation)
- `monai` (lazy loaded for SwinUNETR features)
- `pytest`, `pytest-cov` (test harness)

---

## 2. Logic Chain

1. **Codebase Exploration**: Checking the layout confirmed that PyTorch-based networks exist in `features.py` and optimization routines exist in `syn.py` and `syn_jax.py`. This confirms we can build on top of these models for new metrics.
2. **Package Availability**: Ripgrep searching and viewing `requirements.txt` / `pyproject.toml` confirmed the availability of NumPy, SciPy, Matplotlib, ANTsPy, PyTorch, and JAX.
3. **GEMINI.md Compliance**:
   - **Single Interpolation Policy**: Under NO circumstances should pre-warped inputs be passed to optimization steps. Multi-transform composition must occur in a single execution step.
     - *Observation*: We observed in `syn.py` (lines 1026, 1205, 1393) and `syn_jax.py` that registrations compose initial translation grids and affine transforms in coordinate space, passing them to a single `ants.apply_transforms` call at the end of the pipeline. No intermediate image array warping is performed.
   - **VGG 3D LNCC Layer 4 Requirement**: Default to `lncc_3d` with layer `[4]` to avoid regressions.
     - *Observation*: VGG 3D mode is implemented in `TriPlanarVGG3DLoss` (using reconstructed orthogonal feature volumes) and in `FeatureSpaceLoss` (axial/coronal/sagittal orthogonal volume passing). Layer 4 is the validated target.
   - **HTML Visualization Requirement**: HTML reports must show: (a) edge/region overlap, (b) deformed grids, (c) Jacobian determinant maps, and (d) side-by-side warped vs fixed images.
     - *Observation*: The codebase currently lacks a unified visual HTML reporting generator that fulfills all four visual artifacts.

### 4. Logic for the Test Failure:
- `prepare_mid_images_and_gradients_jax` has signature:
  ```python
  def prepare_mid_images_and_gradients_jax(
      warp_l2r, warp_r2l, I_curr, J_curr, affine_grid_level,
      has_spacing, spacing, identity
  )
  ```
- With static arguments `(5, 6)`.
- The test call fails because it omits `affine_grid_level` (argument 4), shifting `True` to argument 4, `spacing_arg` to static argument 5 (`has_spacing`), and `identity` (a JAX array) to static argument 6 (`spacing`).
- Since JAX arrays are non-hashable, passing `identity` to a static argument triggers the `TypeError: unhashable type`.
- Inserting `identity` as the 5th argument (`affine_grid_level`) and passing `identity` again as the 8th argument restores the correct argument mapping and resolves the compilation crash.

---

## 3. Caveats

- **Computational Resources**: JAX/PyTorch run on CPU by default in this workspace. 3D native registrations are heavy; benchmarking should default to downscaled volumes or synthetic inputs.
- **SwinUNETR Cache**: Pretrained weights download requires internet access. An offline fallback is coded to use random weights or warning fallbacks.
- **Differentiability**: Certain comparison metrics (e.g., PSNR, MSE of gradients, SSIM) are non-differentiable or mathematically complex for AD VJPs. They should be categorized as evaluation metrics in `syntx.image_compare` rather than registration similarity metrics.

---

## 4. Conclusion & Proposed Design

We propose the following architectural and mathematical designs to fulfill the requirements, along with a code fix for the test suite.

### A. Code Fix for `tests/test_syn_jax.py`

#### Before:
```python
    # 1. test prepare_mid_images_and_gradients_jax
    spacing_arg = (1.0, 1.0)
    I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_jax(
        warp_l2r, warp_r2l, I_curr, J_curr,
        True, spacing_arg, identity
    )
```

#### After:
```python
    # 1. test prepare_mid_images_and_gradients_jax
    spacing_arg = (1.0, 1.0)
    I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_jax(
        warp_l2r, warp_r2l, I_curr, J_curr, identity,
        True, spacing_arg, identity
    )
```

### B. Image Comparison Metrics Suite (`syntx.image_compare`)
We design a suite of 64 unique, valid metrics in `syntx.image_compare` supporting both 2D and 3D images. To enforce structure, we define a unified functional interface and class-based metric registry:
```python
def compare_images(img1, img2, metric_name: str, **kwargs) -> float:
    """Computes the similarity/distance between img1 and img2."""
```

#### The 64 Metrics Registry:
1. **Standard Intensity Metrics (18 configurations)**:
   - `mse`: Mean Squared Error
   - `mae`: Mean Absolute Error (L1 distance)
   - `rmse`: Root Mean Squared Error
   - `psnr`: Peak Signal-to-Noise Ratio
   - `ncc`: Global Normalized Cross Correlation
   - `nmi`: Normalized Mutual Information
   - `joint_entropy`: Shannon joint entropy
   - `lncc_w3`, `lncc_w5`, `lncc_w7`, `lncc_w9`, `lncc_w11`: Local NCC with window sizes 3, 5, 7, 9, 11
   - `mmi_b16`, `mmi_b32`, `mmi_b64`, `mmi_b128`, `mmi_b256`: Mattes MI with 16, 32, 64, 128, 256 bins
   - `ssim`: Structural Similarity Index
2. **Gradient and Spatial Structure Metrics (6 configurations)**:
   - `gradient_mse`: MSE computed on Sobel/Central difference gradient magnitudes
   - `gradient_correlation`: Pearson correlation of gradient maps
   - `ngf_e01`, `ngf_e1`, `ngf_e10`: Normalized Gradient Fields with $\epsilon \in \{0.01, 0.1, 1.0\}$
   - `ms_ssim`: Multi-scale SSIM (2D/3D)
3. **VGG19 Deep Feature Space Metrics (12 configurations)**:
   - Features at VGG19 layers $\ell \in \{2, 4, 8, 12\}$ computed under 3 loss formulations:
     - `vgg_l_l1`: L1 loss on layer $\ell$ features
     - `vgg_l_l2`: L2/MSE loss on layer $\ell$ features
     - `vgg_l_lncc`: LNCC loss on reconstructed feature volumes of layer $\ell$ (Note: `vgg_4_lncc` is the high-accuracy VGG 3D LNCC Layer 4)
4. **DINOv2 Deep Feature Space Metrics (12 configurations)**:
   - Features at DINOv2 layers $\ell \in \{1, 2, 6, 11\}$:
     - `dino_l_l1`: L1 loss on features
     - `dino_l_l2`: L2 loss on features
     - `dino_l_lncc`: LNCC on spatial features
5. **ResNet10 Deep Feature Space Metrics (12 configurations)**:
   - Features at ResNet10 layers $\ell \in \{1, 2, 3, 4\}$:
     - `resnet_l_l1`, `resnet_l_l2`, `resnet_l_lncc`
6. **SwinUNETR 3D Deep Feature Space Metrics (4 configurations)**:
   - Features at SwinUNETR layers $\ell \in \{1, 2, 3, 4\}$:
     - `swin_l_lncc`

### C. 2D Generative Cross-Product Space
We define a procedural 2D generator `syntx.generators.CrossProductGenerator` that systematic evaluates metrics under combinations of **6 intensity changes** $\times$ **4 shape changes**.

#### 1. Intensity Transformations ($I(x) \rightarrow I'(x)$):
1. **Noise**: Rician or additive Gaussian noise:
   $$I_{\text{noise}}(x) = I(x) + \eta(x), \quad \eta(x) \sim \mathcal{N}(0, \sigma^2)$$
2. **Bias Field**: Multiplicative low-frequency spatial inhomogeneity:
   $$I_{\text{bias}}(x) = I(x) \cdot (1 + \beta^T x + x^T \mathbf{Q} x)$$
3. **Inhomogeneity**: Hyper/hypo-intense local Gaussian blobs:
   $$I_{\text{inhom}}(x) = I(x) + A \cdot \exp\left(-\frac{\|x - x_0\|^2}{2\gamma^2}\right)$$
4. **Modality Change**: Simulating non-linear contrast changes:
   $$I_{\text{modal}}(x) = \sin(\pi \cdot I(x)) \quad \text{or} \quad I_{\text{modal}}(x) = I(x)^2$$
5. **Step Function**: Quantized intensity steps (e.g. gray/white matter boundaries):
   $$I_{\text{step}}(x) = \lfloor K \cdot I(x) \rfloor / K$$
6. **Missing Data**: Circle or block mask region set to 0:
   $$I_{\text{missing}}(x) = I(x) \cdot (1 - M(x))$$

#### 2. Shape Transformations ($\phi(x)$):
1. **Translation**: Shift vector $t = (t_x, t_y)$.
2. **Rotation**: In-plane angle $\theta$.
3. **Affine**: Combined rotation, shear, scale, translation matrix.
4. **Deformation**: Smooth non-rigid warping field generated via Radial Basis Functions (RBFs) or low-frequency sine/cosine components:
   $$u(x) = \sum_{k} w_k \exp\left(-\frac{\|x - c_k\|^2}{2\sigma_d^2}\right)$$

### D. Grenander's Metric Deformation & Ground-Truth L2 Norm
In Grenander's Pattern Theory, deformations are modeled as the flow of smooth velocity fields or displacement mappings $\phi(x) = x + u(x)$. The deformation cost is represented by the L2 norm of the physical displacement field $u_{\text{phys}}(x)$.

#### Calculation of physical displacement L2 norm:
Given a displacement field $u_{\text{norm}}(x)$ computed on the JAX/PyTorch coordinate grid $[-1, 1]^d$:
1. Convert normalized coordinate offsets to voxel units:
   $$u_{\text{vox}}(x) = u_{\text{norm}}(x) \odot \frac{N - 1}{2}$$
   where $N$ is the grid size vector.
2. Convert voxel displacement to physical space using spacing $S$ (in mm) and direction cosine matrix $D$:
   $$u_{\text{phys}}(x) = D \cdot (u_{\text{vox}}(x) \odot S)$$
3. Compute the L2 norm over the continuous volume $\Omega$:
   $$\|u_{\text{phys}}\|_{L^2} = \sqrt{\int_{\Omega} \|u_{\text{phys}}(x)\|_2^2 \, dx}$$
4. Standard discrete approximation (accounting for voxel volume element $\Delta V = \prod_{i=1}^d S_i$):
   $$\|u_{\text{phys}}\|_{L^2} = \sqrt{\Delta V \sum_{x \in \text{grid}} \|u_{\text{phys}}(x)\|_2^2}$$

### E. Evaluation and HTML Reporting (`syntx.reporting`)
We design a reporting pipeline `generate_registration_report(...)` which outputs a structured HTML page containing all required visual elements from GEMINI.md:
1. **Edge/Region Overlap**: Draw edges of the fixed image using Canny/Sobel filters and overlay them in red onto the green-channel of the registered image.
2. **Deformed Grids**: Generate a grid of coordinate lines, deform the coordinates using the composite displacement field, and plot the grid lines.
3. **Jacobian Determinant Map**: Compute the physical Jacobian determinant map $J(x)$. Render with a diverging colormap (`seismic`), where $J > 1$ is red (expansion), $J < 1$ is blue (compression), and $J \le 0$ represents folding (highlighted in black).
4. **Side-by-Side Images**: Render fixed and warped images side-by-side.

The reporting tool will write to `docs/registration_report.html` embedding all figures as base64 PNGs.

---

## 5. Verification Method

To independently verify this design and code compliance:
1. **Run the existing test command**:
   ```bash
   pytest
   ```
   Assert that all tests pass once the test code fix is applied.
2. **Verification of the single-interpolation policy**:
   Inspect `src/syntx/syn.py` and `src/syntx/syn_jax.py` to ensure that `ants.apply_transforms` is only executed at the very end of the optimization pipeline (or dynamically on coordinate grids) and never on intermediate image arrays.
3. **Verification of the coordinate swap**:
   Run `pytest tests/test_challenger_custom.py`. If it passes, the physical coordinate swap is active and valid.
4. **Verification of HTML report contents**:
   Inspect the generated HTML report file to verify it displays:
   - Overlap maps (edges/regions)
   - Deformed coordinate grid
   - Jacobian determinant map
   - Side-by-side images
