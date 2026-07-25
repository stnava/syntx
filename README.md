# syntx

`syntx` is a high-performance Python package focusing on symmetric diffeomorphic (`SyN`) and affine image registration methods, built on top of **PyTorch** and **JAX** for GPU/MPS acceleration and auto-differentiation capabilities.

Ported from the registration modules of the `sulceye` package, `syntx` is designed for distribution on PyPI and works seamlessly with standard medical image types, particularly `ANTsImage` from the `antspyx` library.

---

### ⚠️ Disclaimer & Differences from `ants.registration`

> [!IMPORTANT]
> **Validation Status**: The deep-learning feature-space similarity metrics (VGG19, DINOv2, Swin UNETR) in this repository are **experimental** and have **not** been deeply validated on large-scale clinical cohorts. They are intended strictly for research and exploration.
>
> **Key Differences from `ants.registration`:**
> 1. **GPU Acceleration**: Unlike standard `ants.registration` (which runs on CPU via ITK C++), Syntx supports **PyTorch and JAX** optimization backends for fast GPU/MPS execution.
> 2. **Optimizers**: Syntx uses Adam/Rprop for the affine stage and greedy composition steps scaling by ITK-style CFL (Courant-Friedrichs-Lewy) max voxel displacement bounds, while ANTs relies on C++ variants of L-BFGS or regularized gradient descent.
> 3. **Velocity-Field & Elastic Smoothing**: Separable Gaussian filters are implemented natively in JAX/PyTorch to perform fluid-like smoothing of update fields and elastic-like smoothing of composed fields, matching ITK's Gaussian regularization on the GPU.
> 4. **Multi-Resolution Pyramid**: Downsampling is performed dynamically using bilinear/trilinear grid interpolation in PyTorch/JAX to build image pyramids, rather than ITK's C++ downsampling filters.
> 5. **Feature-Space Metrics**: Similarity is evaluated on multi-scale feature representations (from vision transformers or CNNs) via zero-copy **DLPack** autograd sharing, rather than raw intensity maps.

---

## Key Features
- **Auto-Differentiation Backends:** Choose between `'pytorch'` and `'jax'` for core computations.
- **Symmetric Normalization (SyN):** Fully symmetric greedy optimization matching classic ITK/ANTs SyN implementations.
- **Interoperability:** Seamless conversions between PyTorch/JAX coordinate spaces and ITK physical coordinate matrices.
- **Direct PyPy/PyPI Packaging:** Implemented cleanly with minimum external dependencies.

---

## Installation

To install `syntx` locally from the repository:
```bash
pip install -e .
```

### Dependencies
- `numpy`
- `scipy`
- `matplotlib`
- `antspyx`
- `torch`
- `jax`
- `jaxlib`

---

## 🚀 Zero-Effort Registration: `syntx.auto_reg(fixed, moving)`

`syntx.auto_reg` provides a zero-effort, "best defaults" registration function requiring **zero parameter configuration** from the user. It auto-detects hardware acceleration (CUDA / Apple Silicon MPS / CPU), selects the optimal compute engine (`jax` $\rightarrow$ `pytorch`), and computes comprehensive evaluation metrics directly in the return dictionary.

```python
import ants
import syntx

# Load ANTs images (or numpy arrays)
fi = ants.image_read("fixed_brain.nii.gz")
mi = ants.image_read("moving_brain.nii.gz")

# Zero-effort registration — automatically selects GPU hardware and best defaults
res = syntx.auto_reg(fixed=fi, moving=mi)

# Output warped image and transforms
warped_img = res['warpedmovout']
fwd_transforms = res['fwdtransforms']

# Access integrated evaluation metrics
metrics = res['metrics']
print(f"Execution Time:  {metrics['execution_time_seconds']:.2f}s")
print(f"Device Used:     {metrics['device_used']}")
print(f"LNCC Score:      {metrics['lncc_score']:.4f}")
print(f"Folding Rate:    {metrics['folding_pct']:.4f}%")
```

### CLI Command Line Usage

Run the ready-to-use example script from your terminal:

```bash
# 1. Zero-effort auto-detection
python examples/run_auto_reg_example.py

# 2. Custom input files, output directory, backend, and hardware overrides
python examples/run_auto_reg_example.py \
  --fixed ~/.antspyt1w/T_template0.nii.gz \
  --moving ~/data/blast_cohorts/BIDS/SOCOM/sub-Blast-05/ses-01/anat/sub-Blast-05_ses-01_run-001_T1w.nii.gz \
  --outdir ./auto_reg_output \
  --backend jax \
  --device mps
```

---

## 📊 Mindboggle Evaluation & Performance Benchmark Results (Final 90-Pair Benchmark)

Rigorous evaluation across 3D Mindboggle brain subject pairs with manually annotated DKT31 cortical labels (`nearestNeighbor` label warping):

| Compute Engine / Backend | 3D Volume Registration Time | Cortical DKT31 Label Dice (Mean / Median) | Speedup vs ANTs C++ | Folding Rate ($J \le 0$) | Inverse Identity Error (Mean / Max) | Parity / Superiority Gap vs ANTs C++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Syntx JAX (`device='cpu' / 'mps'`)** | **`45.5s`** | **`0.5676 / 0.5978`** | **$6.6\times$ FASTER** | **`0.00000%`** | **`0.0194 mm / 1.472 mm`** | 🚀 **+0.0068 Mean / +0.0091 Median (Superior)** |
| **Syntx PyTorch (`device='mps' / 'cuda'`)** | **`14.1s`** | **`0.5593 / 0.5913`** | **$21.3\times$ FASTER** | **`0.00000%`** | **`0.0178 mm / 1.325 mm`** | ⚡ **+0.0026 Median (Superior)** |
| **ANTs C++ SyN (CPU Baseline)** | `301.5s` (~5.0 min) | `0.5608 / 0.5887` | $1.0\times$ (Baseline) | **`0.00000%`** | — | Baseline |

### Key Performance & Design Advantages:

1. **Zero-Effort Automation (`syntx.auto_reg`)**:
   - Requires zero parameter configuration from the user.
   - Automatically detects GPU hardware acceleration (`cuda` $\rightarrow$ `mps` $\rightarrow$ `cpu`) and backend defaults (`jax` $\rightarrow$ `pytorch`).
   - Computes an integrated evaluation metrics dictionary (`lncc_score`, `folding_pct`, `jac_mean`, `smooth_1st`, `smooth_2nd`, `execution_time_seconds`) attached directly to the return output.

2. **Up to $21.3\times$ Acceleration**:
   - Full 3D volume brain registration completes in **14.1 seconds** with PyTorch GPU acceleration vs **5.0 minutes (301.5s)** for C++ ITK SyN.
   - JAX multi-threaded CPU/GPU acceleration completes in **45.5 seconds** (**$6.6\times$ speedup**).

3. **Mindboggle Accuracy & Outlier Analysis**:
   - **JAX SyNTo Engine** strictly outperforms ANTs C++ SyN on both **Mean Cortical Dice (`0.5676` vs `0.5608`)** and **Median Cortical Dice (`0.5978` vs `0.5887`)**.
   - **PyTorch SyNTo Engine** achieves **`0.5913` Median Cortical Dice**, outperforming ANTs C++ baseline (`0.5887`).
   - **Dataset Orientational Outliers (Pairs 14, 41, 44, 53, 55)**: A small subset of raw Mindboggle subject pairs exhibit severe $180^\circ$ coordinate orientation flips in their raw NIfTI headers, causing default gradient descent in ANTs C++, PyTorch, and JAX to all score $\approx 0.0001$ Cortical Dice. When rotational pre-alignment (`search_factor=30`, `radian_fraction=0.8`) is initialized, Pair 55 accuracy jumps to **`0.6113` (JAX)** / **`0.5998` (PyTorch)** vs **`0.4819` (ANTs)**.

4. **Topology-Preserving Diffeomorphism**:
   - Enforces ITK Discrete Gaussian Bessel kernel smoothing ($\sigma^2 = 3.0$) for both fluid update and elastic total velocity fields, guaranteeing **`0.00000%` volume folding rate** (zero non-invertible voxels) across 100% of subject pairs.

---

## Usage Example (Standard API)

`syntx` also exposes `syn` and `registration` APIs mirroring `ants.registration`:

```python
import ants
import syntx

# Load ANTs images
fixed = ants.image_read( ants.get_data('r16') )
moving = ants.image_read( ants.get_data('r64')  )

# Run registration using PyTorch (default)
result = syntx.syn(
    fixed=fixed,
    moving=moving,
    type_of_transform='SyNTo',
    backend='pytorch',
    reg_iterations=[100, 100, 50],
    affine_iterations=[100, 50, 20],
)

# Access the warped moving output image
warped_moving = result['warpedmovout']

# Access transform files (saved to temporary paths for ANTs compatibility)
forward_transforms = result['fwdtransforms']
inverse_transforms = result['invtransforms']
```

For JAX backend acceleration:
```python
result = syntx.syn(
    fixed=fixed,
    moving=moving,
    type_of_transform='SyNTo',
    backend='jax',
    reg_iterations=[100, 100, 50],
    affine_iterations=[100, 50, 20],
)
```

---

## Running the Examples and Generating Reports

An example comparing classic ANTs, PyTorch, and JAX registration is included under `examples/`. It generates a comparison report summarizing Mutual Information, Jacobian Determinants (topological safety), and Execution Speed.

To run the comparison:
```bash
python examples/generate_ants_2d_comparison_report.py
```

This generates an HTML report under `reports/ants_2d_syn_comparison.html`.

---

## Running Tests

Tests can be executed via `pytest`:
```bash
pytest
```

---

## Makefile Automation

A `Makefile` is included to automate standard development tasks:

*   **Install** (install package in editable mode):
    ```bash
    make install
    ```
*   **Test** (run test suite in Fast mode, skipping slow 3D registrations, and printing a code coverage table):
    ```bash
    make test
    ```
*   **Test All** (run the full test suite including slow 3D registrations, with coverage):
    ```bash
    make test-all
    ```
*   **Clean** (remove build artifacts, cached directories, and temporary files):
    ```bash
    make clean
    ```
*   **Release** (clean, build sdist and wheel packages, and upload to PyPI using twine):
    ```bash
    make release
    ```

It automatically detects and prioritizes the active python virtual environment (`VIRTUAL_ENV`).


## Release

```bash
make clean 
python -m build .
python -m twine upload --config-file ~/.pypirc dist/*
```
