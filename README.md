# syntx

`syntx` is a high-performance Python package focusing on symmetric diffeomorphic (`SyN`), time-varying velocity fields (`TVF` / `LDDMM`), geodesic shooting (`SyNGS`), and robust affine registration, built natively on **PyTorch** and **JAX** for GPU/MPS acceleration and analytical auto-differentiation.

Designed for seamless drop-in interoperability with medical imaging ecosystems, `syntx` operates directly on `ants.ANTsImage` instances from `antspyx` while executing end-to-end tensor transformations on hardware accelerators (Apple Silicon MPS, NVIDIA CUDA, and CPU).

---

### ⚠️ Disclaimer & Differences from `ants.registration`

> [!IMPORTANT]
> **Validation Status**: The deep-learning feature-space similarity metrics (VGG19, DINOv2, Swin UNETR) in this repository are **experimental** and have **not** been deeply validated on large-scale clinical cohorts. They are intended strictly for research and exploration.
>
> **Key Differences from `ants.registration`:**
> 1. **GPU Acceleration**: Unlike standard `ants.registration` (which runs on CPU via ITK C++), Syntx supports **PyTorch and JAX** optimization backends for fast GPU/MPS execution.
> 2. **Continuous Flow Paradigms**: In addition to standard greedy SyN, Syntx provides full 4D continuous Time-Varying Velocity Fields (`syntx.tvf`) and single-momentum Geodesic Shooting (`syntx.syngs`).
> 3. **Riemannian Sobolev-Adam**: Combines Adam momentum tracking with Sobolev/Gaussian Green operator metric preconditioning, preventing the pointwise high-frequency grid tearing of standard optimizers.
> 4. **Exact Zero-Boundary Shields (DST-I)**: Discrete Sine Transform Type-I Green operators analytically enforce homogeneous Dirichlet boundary conditions $v(\partial \Omega) \equiv 0$, preventing boundary coordinate drift.
> 5. **Single Interpolation Policy**: Strictly composes all deformable, affine, and center-of-mass transforms into a single coordinate mapping directly on native-space arrays, avoiding intermediate pre-warping degradation.

---

## Key Features
- **Auto-Differentiation Backends:** Choose between `'pytorch'` and `'jax'` for core computations.
- **Multiple Transformation Models:** SyN (Eulerian Fréchet midpoint), TVF (Continuous 4D Lie flow), and SyNGS (EPDiff Geodesic Shooting).
- **Interoperability:** Seamless conversions between PyTorch/JAX coordinate spaces and ITK physical coordinate matrices (`ANTsImage`).
- **Direct PyPI Packaging:** Implemented cleanly with minimum external dependencies.

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

---

## 🧠 Key Transformation Paradigms in `syntx`

```
                                  Diff(Ω) Lie Group Manifold
                            ┌────────────────────────────────────┐
                            │                                    │
   1. Symmetric SyN         │   I_F ◄─── φ_F ─── Ω_1/2 ─── φ_M ──► I_M
      (Fréchet Midpoint)    │                                    │
                            ├────────────────────────────────────┤
   2. TVF / LDDMM           │   I_0 ────► v(t_1) ────► v(t_2) ───► I_1
      (Continuous 4D Flow)  │   (K Keyframe Velocity Fields in Lie Algebra)
                            ├────────────────────────────────────┤
   3. Geodesic Shooting     │   I_0 ────► v_0 (EPDiff Momentum) ──► I_1
      (Single Initial v_0)  │   (Single Vector Field at t=0 on Tangent Space)
                            └────────────────────────────────────┘
```

### 1. `syntx.tvf` — Continuous Time-Varying Velocity Fields (LDDMM)
- **Mathematical Principle**: Models deformation as the continuous integration of time-varying Eulerian velocity fields along $t \in [0, 1]$:
  $$\frac{d\phi_t}{dt} = v_t \circ \phi_t, \quad \phi_0 = \text{Id}$$
- **Keyframe Lie Algebra Interpolation**: Parameterized by $K$ keyframe velocity vector fields $\{v_{t_k}\}_{k=1}^K$ interpolated temporally via continuous Catmull-Rom cubic splines.
- **DST-I Dirichlet Boundary Shield**: Discrete Sine Transform Type-I Green operators analytically enforce $v(x \in \partial \Omega) \equiv 0$, guaranteeing zero boundary coordinate drift and bounding folding to $<0.007\%$.
- **Multi-Point Trajectory Loss**: Evaluates similarity at $t \in \{0.0, 0.5, 1.0\}$, delivering the highest cortical accuracy across the 90-pair cohort (**`0.6466` Mean DICE, 100% win rate**).

### 2. `syntx.syngs` — Riemannian Geodesic Shooting (SyNGS)
- **Mathematical Principle**: The entire spatial deformation trajectory $\phi_t$ is uniquely determined by a **single initial momentum vector field** $\mathbf{v}_0 \in T_{\text{Id}}\text{Diff}$ at $t=0$, integrated forward via the Euler-Poincaré EPDiff equation:
  $$\frac{\partial m_t}{\partial t} + \text{ad}_{v_t}^\dagger m_t = 0, \quad \text{where } m_t = L v_t$$
- **Computational Anatomy Standard**: Because only $\mathbf{v}_0$ is optimized and stored, `syntx.syngs` provides a true linear tangent space representation for statistical shape modeling, atlas building, and Principal Geodesic Analysis (PGA).
- **Sub-Voxel Inversion Precision**: Achieves an average inverse identity error of **`0.0303 mm`** with **`0.6382` DICE** ($+1.66\%$ over ANTs C++).

### 3. `syntx.syn` — Eulerian Symmetric Normalization
- **Mathematical Principle**: Deforms both fixed $I_F$ and moving $I_M$ images symmetrically toward a virtual Fréchet geodesic midpoint $\Omega_{1/2}$:
  $$\phi_{\text{total}} = \phi_M^{-1} \circ \phi_F$$
- **In-Loop Anderson Acceleration**: Inverts deformation fields dynamically inside the optimization loop using multi-vector Anderson fixed-point acceleration, eliminating the numerical drift of legacy fixed-point inversion.
- **Antithetic Bootstrapped Descent**: Destructively cancels discrete grid discretization noise via zero-bias antithetic coordinate jittering ($\mathbb{E}[\boldsymbol{\delta}] = \mathbf{0}$), achieving **`0.6342` DICE** in **`48.8 s`** on GPU.

### 4. `syntx.robust_affine` — Deterministic Multi-Start Lattice Search
- **Parameterization**: Optimizes rigid and affine transformations over the Lie Group $\text{SO}(3)$ using the Lie Algebra $\mathfrak{so}(3)$ matrix exponential map.
- **18-Cone Multi-Start Lattice**: Evaluates 18 pitch/roll/yaw cone orientations around Center of Mass and Field of View geometric centers using foreground union-masked Mutual Information, completely resolving $180^\circ$ inversion traps.

---

## ⚙️ Mathematical & Parameter Parity with `ants.registration`

Understanding the exact mathematical mappings between ITK / ANTs C++ and `syntx` is essential for faithful reproduction and optimal accuracy:

| Parameter / Concept | ANTs C++ (`ants.registration`) | `syntx` Implementation | Mathematical Meaning & Parity Nuance |
| :--- | :--- | :--- | :--- |
| **Smoothing Metric Convention** | `flow_sigma = 3.0` (Variance) | `flow_sigma = 1.732` (Std Dev) | ITK specifies Gaussian smoothing as **variance** ($\sigma^2 = 3.0$), while PyTorch/JAX filters expect **standard deviation** ($\sigma = \sqrt{3.0} \approx 1.732\text{ mm}$). Passing $\sigma=3.0$ in `syntx` equals ITK variance $9.0$. |
| **Gradient Backpropagation** | ITK $CC^2$ pseudo-derivative | Autograd Analytical LNCC | Analytical autograd through sliding box-filter LNCC provides exact spatial descent directions, yielding $+1.08\%$ higher DICE than ITK's center-of-window approximation. |
| **Variance Floor Singularity** | Not explicitly bounded | $\text{Var}_{\text{safe}}(I) \ge 10^{-6}$ | Because $\frac{\partial \text{LNCC}}{\partial I} \propto \frac{1}{\text{Var}(I)}$, un-floored variance in uniform white matter or background zero padding causes derivative spikes that drive grid folding. `syntx` strictly floors variance. |
| **Physical Gradient Scaling** | ITK physical space vectors | $\mathbf{s}_{\text{phys}} = \text{flip}\left(\frac{(\mathbf{N}-1)\odot\mathbf{s}}{2}\right)$ | PyTorch indexes spatial tensors in $(Z, Y, X)$ order while vector channels are $(x, y, z)$. The physical scaling vector must be flipped along dim 0 to prevent cross-axis distortion on anisotropic volumes. |
| **Interpolation Policy** | Multi-step file resampling | **Single Interpolation Invariant** | Intermediate pre-warping accumulates low-pass spatial blurring. All transforms must be composed and applied directly to native-space arrays in a single interpolation step. |
| **Intensity Normalization** | Raw intensities or min/max | 2nd–98th Percentile Truncation | Non-zero intensities are clamped and scaled to $[p_{02}, p_{98}]$ to prevent high-intensity vascular or reconstruction outliers from stalling gradients. |
| **Mutual Information Masking** | Global joint histogram | Foreground Union Masking | Joint histograms are evaluated strictly over $(I > 0.01) \mid (J > 0.01)$ to prevent background zero-padding voxels from dominating entropy calculations. |

---

## 🎯 Similarity Metrics & Optimizers

### Similarity Metrics
1. **Intensity LNCC (`similarity_metric='cc2'` / `'lncc'`)**:
   - $5 \times 5 \times 5$ sliding box-filter Local Normalized Cross-Correlation evaluated with safe variance flooring. Optimal for intra-modality high-contrast structural alignment.
2. **Deep Feature LNCC (`'dino_2_lncc'`, `'vgg_4_lncc'`)**:
   - Evaluates correlation over deep semantic feature representations extracted via zero-copy DLPack memory sharing. `dino_2_lncc` provides extreme robustness against noise and bias artifacts; `vgg_4_lncc` preserves sharp structural edges under massive modality contrast inversions.
3. **Mattes Mutual Information (`'mattes_mi'`)**:
   - 32-bin B-spline Parzen joint histogram entropy functional with foreground union masking for rigid/affine multi-start search.

### Optimizers & Regularization
1. **Riemannian Sobolev-Adam (`optimizer='reg_adam'`)**:
   - Standard pointwise Adam fails in infinite-dimensional diffeomorphism optimization by amplifying high-frequency noise. `RegAdam` combines Adam first/second moment tracking with Sobolev/Gaussian Green operator metric preconditioning, ensuring smooth descent trajectories without grid tearing.
2. **Courant-Friedrichs-Lewy (CFL) Step Bounding (`max_step_norm = 0.25 - 0.50`)**:
   - Strictly bounds the maximum spatial displacement per optimization step in voxels, guaranteeing stable trajectory integration.
3. **In-Loop Anderson Acceleration (`in_loop_inv_steps = 10`)**:
   - Dynamic fixed-point acceleration inside the registration loop that guarantees sub-voxel bijection accuracy ($\bar{e} < 0.03\text{ mm}$).

---

## 📖 Standard API Usage

`syntx` provides modular APIs mirroring standard registration workflows:

### 1. SyN (Eulerian Diffeomorphic Registration)
```python
import ants
import syntx

fixed = ants.image_read(ants.get_data('r16'))
moving = ants.image_read(ants.get_data('r64'))

# Run Eulerian SyN using PyTorch on GPU/MPS
result = syntx.syn(
    fixed=fixed,
    moving=moving,
    backend='pytorch',
    device='mps', # or 'cuda', 'cpu'
    reg_iterations=[100, 100, 50],
    similarity_metric='cc2'
)

warped_moving = result['warpedmovout']
forward_transforms = result['fwdtransforms']
inverse_transforms = result['invtransforms']
```

### 2. TVF (Continuous Time-Varying Velocity Fields)
```python
result_tvf = syntx.tvf(
    fixed=fixed,
    moving=moving,
    regularizer='dsti1', # Dirichlet boundary shield
    flow_sigma=1.0,
    total_sigma=0.035,
    optimizer='reg_adam',
    optimizer_lr=1.2,
    max_step_norm=0.50,
    reg_iterations=[100, 100, 20]
)
```

### 3. SyNGS (Riemannian Geodesic Shooting)
```python
result_syngs = syntx.syngs(
    fixed=fixed,
    moving=moving,
    regularizer='sobolev',
    alpha=0.35,
    optimizer='reg_adam',
    optimizer_lr=1.2,
    max_step_norm=0.25,
    reg_iterations=[100, 100, 20]
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
