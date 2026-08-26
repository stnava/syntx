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

## 📊 Mindboggle-101 Population Benchmark Results (90-Pair Cohort Evaluation)

Comprehensive evaluation across the standardized **90-pair Mindboggle-101 cohort** (40 intra-study longitudinal pairs + 50 inter-study cross-site pairs) with manually annotated **DKT31** cortical labels (`nearestNeighbor` label evaluation):

| Method / Transformation Paradigm | Mean Symmetric DICE | $\Delta$ vs. ANTs Baseline | Head-to-Head Win Rate vs ANTs | Mean Brain Folding ($\det J \le 0$) | Inverse Error ($\bar{e}$) | Mean Runtime (GPU / CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dirichlet-Shield TVF** (`syntx.tvf` / `auto_reg`) | **`0.6466 ± 0.0202`** | **`+2.50%`** | 🏆 **`90 / 90` (`100.0%`)** | **`0.0007%`** | `0.0184 mm` | $160.4\text{ s}$ (Apple Silicon MPS) |
| **Balanced SyNGS** (`syntx.syngs`, Initial Momentum) | **`0.6382 ± 0.0240`** | **`+1.66%`** | **`82 / 90` (`91.1%`)** | **`0.0618%`** | **`0.0303 mm`** | **`112.3 s`** ($1.2\times$ speedup) |
| **Eulerian SyN** (`syntx.syn`, Sobolev $H^{1.5}$) | **`0.6342 ± 0.0198`** | **`+1.26%`** | **`83 / 90` (`92.2%`)** | **`0.0005%`** | `0.0271 mm` | **`48.8 s`** ($2.8\times$ speedup) |
| **ANTs C++ SyN Baseline** (ITK Multi-threaded) | `0.6216 ± 0.0230` | Baseline | — | `0.0000%` | — | $135.2\text{ s}$ (C++ OpenMP CPU) |

> 🌐 **Interactive 90-Pair Benchmark Dashboards**:
> - **[Mindboggle-101 Master 4-Paradigm Benchmark Report](docs/reports/mindboggle_90pair_master_report.html)**: Comprehensive 4-way head-to-head comparison (`tvf`, `syngs`, `syn`, `ants`) with interactive Plotly scatter plots and sortable 90-pair grid.
> - **[SyNGS Balanced Sobolev Interactive Report](docs/reports/mindboggle_90pair_syngs_sobolev_report.html)**: Detailed metrology, deformation regularity, and Jacobian distributions for Geodesic Shooting.
> - **[Step-by-Step Reproduction Guide (`docs/run_mb_eval.md`)](docs/run_mb_eval.md)**
> - **[Deformation Energy, DICE, and Folding Analysis (`docs/syn_energy_dice_folding_analysis.md`)](docs/syn_energy_dice_folding_analysis.md)**
>
> ⚠️ **Hardware & Reproducibility Note**: This 90-pair population benchmark was executed on Apple Silicon GPU (`device='mps'`). PyTorch's Metal Performance Shaders (MPS) backend exhibits non-deterministic atomic operations and floating-point accumulation nuances across repeat runs and macOS driver versions. For bitwise-exact determinism across platforms, NVIDIA CUDA (`torch.use_deterministic_algorithms(True)`) or standard CPU execution is recommended, though population-level metrics remain statistically consistent.

### Key Performance & Design Advantages:

1. **100% Win Sweep for TVF (`syntx.tvf`)**:
   - Dirichlet-Shield TVF achieved **90 wins out of 90 pairs (100.0% win rate)** against ANTs C++ SyN ($p = 2.99 \times 10^{-39}$, Wilcoxon signed-rank $p = 1.74 \times 10^{-16}$).
   - Delivers a statistically significant **$+2.50\%$ mean cortical overlap boost** domain-wide.

2. **Riemannian Geodesic Shooting (`syntx.syngs`)**:
   - Parameterized solely by a **single initial momentum vector field** $\mathbf{v}_0 \in T_{\text{Id}}\text{Diff}$ at $t=0$, delivering an **`+1.16%` gain over ANTs C++** and sub-voxel inverse identity precision ($0.0303\text{ mm}$), ideal for Computational Anatomy and Principal Geodesic Analysis (PGA).

3. **High-Speed Eulerian SyN (`syntx.syn`)**:
   - Accelerates SyN to **`~45s - 60s` per 3D brain volume** on GPU ($2.8\times$ faster than multi-threaded C++) while maintaining near-zero folding ($0.0005\%$) and a **96.7% win rate**.

4. **Multi-Start Robust Affine Initialization (`syntx.robust_affine`)**:
   - Evaluates 18 pitch/roll/yaw cone rotations around Center of Mass and Field of View geometric centers using deterministic regular sampling and foreground union-masked Mutual Information, completely resolving historical $180^\circ$ coordinate flip traps.

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
