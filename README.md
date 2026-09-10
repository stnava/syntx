# syntx

`syntx` is a high-performance Python package focusing on symmetric diffeomorphic (`SyN`), time-varying velocity fields (`TVF` / `LDDMM`), geodesic shooting (`SyNGS`), robust affine registration, and generalized scattered data registration (`SyNScattered`), built natively on **PyTorch** and **JAX** for GPU/MPS acceleration and analytical auto-differentiation.

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
> 6. **Generalized Scattered Data Registration**: Extends SyN to arbitrary Lagrangian scattered coordinate sets (point clouds, surface flatmaps, sparse landmarks) and mixed point-to-grid alignment with autograd differentiability, Nadaraya-Watson kernel regression, and in-loop Anderson inversion.

---

## Key Features
- **Auto-Differentiation Backends:** Choose between `'pytorch'` and `'jax'` for core computations.
- **Multiple Transformation Models:** SyN (Eulerian Fréchet midpoint), TVF (Continuous 4D Lie flow), SyNGS (EPDiff Geodesic Shooting), and Generalized Scattered SyN (differentiable Nadaraya-Watson kernel projection, coordinate mapping, and feature transport).
- **Interoperability & Centralized Spatial Management:** Seamless conversions between PyTorch/JAX coordinate spaces and ITK physical coordinate matrices (`ANTsImage`) managed through `syntx.spatial`.
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
                            ┌─────────────────────────────────────────┐
                            │                                         │
   1. Symmetric SyN         │   I_F ◄─── φ_F ─── Ω_1/2 ─── φ_M ──► I_M │
      (Fréchet Midpoint)    │                                         │
                            ├─────────────────────────────────────────┤
   2. TVF / LDDMM           │   I_0 ────► v(t_1) ────► v(t_2) ───► I_1 │
      (Continuous 4D Flow)  │   (K Keyframe Velocity Fields in Lie)   │
                            ├─────────────────────────────────────────┤
   3. Geodesic Shooting     │   I_0 ────► v_0 (EPDiff Momentum) ──► I_1│
      (Single Initial v_0)  │   (Single Tangent Vector Field at t=0)  │
                            ├─────────────────────────────────────────┤
   4. Scattered SyN         │   X_F, F_F ◄── NW ── φ_F ── Ω_1/2 ── φ_M ── NW ──► X_M, F_M
      (Lagrangian-Eulerian) │   (Point Cloud & Mixed Point-to-Grid Diffeomorphism)
                            └─────────────────────────────────────────┘
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

### 5. `syntx.syn_scattered` — Generalized Scattered Data Diffeomorphic Registration
- **Mathematical Principle**: Bridges discrete Lagrangian point clouds $\{x_i, f_i\}_{i=1}^N \subset \mathbb{R}^d \times \mathbb{R}^C$ and continuous Eulerian diffeomorphism spaces $\text{Diff}(\Omega)$ using differentiable normalized Gaussian kernel regression (Nadaraya-Watson):
  $$G(y) = \frac{\sum_{i=1}^N K_\sigma(y - x_i) w_i f_i}{\sum_{i=1}^N K_\sigma(y - x_i) w_i + \epsilon}, \quad K_\sigma(r) = \exp\left(-\frac{\|r\|^2}{2\sigma^2}\right)$$
- **Vectorized GEMM & Auto-Chunking**: Distance expansion $D^2 = N_Y - 2 Y X^T + N_X$ computed via matrix multiplication with clamping $\max(D^2, 0.0)$ to eliminate floating-point roundoff errors, with dynamic memory auto-chunking capped at $\le 256\text{ MB}$ to ensure safe execution on dense 3D grids.
- **Continuous Fluid Regularization & CFL Step Bounding**: Updates are regularized via Discrete Sine Transform Type-I (DST-I Dirichlet zero-boundary), Sobolev, or Gaussian Green operators, strictly bounded by Courant-Friedrichs-Lewy conditions ($\text{CFL} \le 0.25\text{ voxels}$) to guarantee strictly positive Jacobian determinants ($\det(J) > 0$) and zero grid folding ($< 0.1\%$).
- **In-Loop Anderson Inversion Acceleration**: Type-I multi-secant Anderson fixed-point acceleration guarantees sub-voxel inverse consistency error ($\|\phi \circ \phi^{-1} - \text{Id}\|_\infty < 10^{-3}$) without numerical drift.
- **Full Differentiability & Feature Transport**: Backpropagation propagates gradients seamlessly back to both point coordinates $X$ and scalar/vector features $F$. Supports coordinate warping ($\phi(x) = x + u(x)$), grid pullback ($\Phi^* G(x)$), feature pushforward ($\Phi_* F(g)$), and direct Lagrangian point-to-point feature transport.

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

## 📐 Centralized Spatial Management (`syntx.spatial`)

We created `spatial.py` as the single source of truth to handle all coordinate, spacing, and vector conversions between ITK physical space and PyTorch/JAX tensor grids. We replaced dozens of scattered, ad-hoc `.transpose()` and axis-reversal calls across `syn.py`, `tvf.py`, `syngs.py`, `robust_affine.py`, and `transform.py` with standardized primitives from this new module. Finally, we verified the migration with comprehensive roundtrip and adversarial tests in `test_spatial_roundtrip.py` and `test_spatial_centralization.py`, proving exact numerical roundtripping across anisotropic grids and reflection transforms.

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

## 🌊 Weingarten Curvature & Sulcal Ridge Guidance

Medical image registration along thin, convoluted anatomical sheets (such as the ~2.5 mm human cerebral cortex) suffers from the classic **aperture problem**: within the uniform gray matter ribbon, the intensity gradient $\nabla I(x)$ points almost exclusively normal to the cortex ($\mathbf{n} = \frac{\nabla I}{\|\nabla I\|}$), leaving the tangential directions along sulcal folds in a near-null space ($\nabla_\parallel I \approx \mathbf{0}$). Consequently, opposing sulcal banks across narrow CSF fissures can slide inconsistently, generating extreme cross-bank spatial shear that drives coordinate collisions and grid folding ($\det J \le 0$).

To resolve this, `syntx` and `antstorch` introduce GPU/MPS-accelerated **Weingarten Image Curvature Guidance**, extracting extrinsic differential geometric invariants directly from continuous level sets of 3D scalar volumes.

### 1. Differential Geometric Formulation
The Weingarten shape operator $\mathcal{W}$ measures the directional variation of the unit normal field:
$$\mathcal{W} = -\nabla_{\parallel} \mathbf{n} = -\nabla_{\parallel} \left(\frac{\nabla I_\sigma}{\|\nabla I_\sigma\|_2}\right)$$
From the eigenvalues of $\mathcal{W}$ (principal curvatures $\kappa_1 \ge \kappa_2$), we extract:
- **Mean Curvature ($H$)**: $H = \frac{1}{2}(\kappa_1 + \kappa_2) = -\frac{1}{2} \nabla \cdot \left(\frac{\nabla I_\sigma}{\|\nabla I_\sigma\|_2}\right)$
- **Gaussian Curvature ($K$)**: $K = \kappa_1 \kappa_2 = \det(\mathcal{W})$

Infusing mean curvature into registration ($I_{\text{guided}} = I + \alpha \cdot H$, with $\alpha \approx 0.10, \sigma = 1.5\text{ mm}$) provides dense, monotonic tangential landmarks (sulcal fundi $< 0$, gyral crests $> 0$), pinning opposing banks to their true anatomical geometry and eliminating unconstrained shear.

### 2. ⚡ GPU Acceleration Benchmark: PyTorch MPS vs ITK C++
Curvature extraction was implemented natively in PyTorch (`antstorch.weingarten_image_curvature`) using separable Gaussian derivative filters and vectorized trace operations, achieving over **$20\times$ speedup** over ITK C++:

| Volume Domain | Grid Dimensions | Total Voxels | ITK C++ (`ants`) | PyTorch MPS (`antstorch`) | GPU Speedup | Numerical Parity ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Small Volume | $64 \times 64 \times 64$ | $0.26\text{ M}$ | $0.409\text{ s}$ | **`0.026 s`** | **`15.9x`** | $r = 0.99965$ |
| Medium Volume | $100 \times 100 \times 100$ | $1.00\text{ M}$ | $1.297\text{ s}$ | **`0.063 s`** | **`20.5x`** | $r = 0.99971$ |
| Isotropic Cube | $128 \times 128 \times 128$ | $2.10\text{ M}$ | $2.293\text{ s}$ | **`0.111 s`** | **`20.7x`** | $r = 0.99972$ |
| **Full MNI Brain** | $182 \times 218 \times 182$ | $1.83\text{ M}$ | $9.123\text{ s}$ | **`0.408 s`** | **`22.4x`** | **$r = 0.99973$** |

### 3. Mindboggle-101 Benchmark Evaluation
Evaluation across the 4 canonical Mindboggle benchmark pairs under the full multi-resolution schedule (`[100, 100, 20]`, `[4, 2, 1]`):

| Pair ID | Alignment Type | Cohort / Subject Pair | Baseline SyN DICE | Curvature SyN DICE | $\Delta$ vs Baseline | Gain vs ANTs C++ | Curvature Folding | Min $\det(J)$ |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pair 0** | Intra-subject | OASIS-TRT-20-17 $\rightarrow$ OASIS-TRT-20-16 | `0.65116` | **`0.65273`** | **`+0.16%`** | **`+1.98%`** | `0.00136%` | $>0$ |
| **Pair 39** | Intra-subject | MMRR-21-17 $\rightarrow$ MMRR-21-11 | `0.59854` | **`0.59909`** | **`+0.05%`** | **`+1.67%`** | **`0.00000%`** | **`+0.0223`** |
| **Pair 53** | Inter-subject | NKI-RS-22-2 $\rightarrow$ NKI-TRT-20-1 | `0.61902` | `0.60639` | `-1.26%` | **`+2.34%`** | `0.00481%` | $>0$ |
| **Pair 62** | Inter-subject | OASIS-TRT-20-5 $\rightarrow$ MMRR-21-11 | `0.60293` | `0.60251` | `-0.04%` | **`+0.28%`** | `0.02855%` | $>0$ |
| **Mean** | **4 Pairs** | — | `0.61791` | `0.61518` | `-0.27%` | **`+1.57%`** | **`0.00868%`** | — |

#### Key Insights:
1. **Mean vs. Gaussian Curvature**: Mean Curvature ($H$) strictly outperforms Gaussian Curvature ($K$) ($+0.79\%$ higher DICE). Because cortical sulcal folds are locally developable/cylindrical surfaces ($\kappa_1 \gg 0, \kappa_2 \approx 0$), Gaussian curvature $K = \kappa_1 \kappa_2$ vanishes along sulcal ridges, whereas Mean Curvature $H = \frac{1}{2}(\kappa_1 + \kappa_2)$ captures the complete continuous sulcal trench.
2. **Deformation Regularization**: On intra-subject registrations where sulcal topologies match 1-to-1, curvature guidance locks fundic lines directly, eliminating tangential drift and elevating $\min \det(J)$ from $0.0$ to $+0.0223$ with **$0.00000\%$ folding**.
3. **Inter-Subject Topology**: For inter-subject cohorts with divergent tertiary sulcal branching, fine-scale curvature ($\sigma=1.5\text{ mm}$) can penalize non-homologous folds. A multiscale approach or applying curvature at coarser scales ($\sigma \ge 3.0\text{ mm}$) preserves global sulcal matching without over-constraining individual variations.

### 4. Code Example
```python
import ants
import antstorch
import syntx

# Load native-space T1w images
fixed = ants.image_read("fixed.nii.gz")
moving = ants.image_read("moving.nii.gz")

# 1. Extract Weingarten Mean Curvature on GPU (22x faster than ITK)
curv_fix = antstorch.weingarten_image_curvature(fixed, sigma=1.5, metric="mean", device="mps")
curv_mov = antstorch.weingarten_image_curvature(moving, sigma=1.5, metric="mean", device="mps")

# 2. Blend geometric curvature into intensity target (alpha=0.10)
alpha = 0.10
fixed_guided = fixed + alpha * (curv_fix - curv_fix.mean()) / (curv_fix.std() + 1e-6)
moving_guided = moving + alpha * (curv_mov - curv_mov.mean()) / (curv_mov.std() + 1e-6)

# 3. Register with SyN (Eulerian, RegAdam, or Sobolev)
res = syntx.syn(
    fixed=fixed_guided,
    moving=moving_guided,
    reg_iterations=[100, 100, 20],
    device="mps"
)
```

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

### 4. Scattered Data Diffeomorphic Registration (`syntx.syn_scattered`)

Aligns arbitrary 2D/3D Lagrangian point sets (point-to-point) or point clouds against reference Eulerian grids (point-to-grid) with end-to-end autograd differentiability:

```python
import torch
import syntx

# Point clouds: (N, d) coordinates and (N, C) multi-channel features
fixed_points = torch.randn(500, 2)
fixed_features = torch.randn(500, 1)

moving_points = fixed_points + 0.05 * torch.sin(fixed_points * 3.14159)
moving_features = fixed_features.clone()

# Execute symmetric diffeomorphic scattered data registration
result = syntx.syn_scattered(
    fixed_points=fixed_points,
    fixed_features=fixed_features,
    moving_points=moving_points,
    moving_features=moving_features,
    grid_res=64,
    fluid_sigma=1.5,
    cfl_voxels=0.25,
    regularizer='dsti1', # Dirichlet zero-boundary shield
    iterations=[50, 30],
    levels=[2, 1], # Multi-resolution coarse-to-fine pyramid
)

# Access registered Lagrangian coordinates & Eulerian displacement fields
warped_moving_pts = result.warped_moving_points
fwd_disp = result.disp_fwd # Fixed -> Moving displacement
inv_disp = result.disp_inv # Moving -> Fixed displacement

# Topological regularity and inverse consistency guarantees
print(f"Grid Folding:        {result.folding_percentage:.4f}%") # < 0.1%
print(f"Min det(J):          {result.jacobian_min:.4f}")         # > 0
print(f"Inverse Consistency: {result.inverse_identity_error:.6f}") # < 1e-3

# Coordinate warping & feature transport
warped_pts = result.warp_points(moving_points, direction='forward')
transported_feats = result.transport_features(
    coords_src=moving_points,
    features_src=moving_features,
    coords_tgt=fixed_points,
    direction='forward',
)
```

### 5. Differentiable Scattered-to-Grid Projection (`syntx.project_scattered_to_grid`)

Maps scattered point observations onto a regular Eulerian grid lattice via autograd-differentiable Nadaraya-Watson kernel regression:

```python
import torch
from syntx.scattered import project_scattered_to_grid, ScatteredProjector

points = torch.randn(1000, 3) # 3D point cloud
features = torch.randn(1000, 4) # 4-channel multi-spectral features

# One-shot projection with memory auto-chunking (<= 256 MB)
eulerian_grid = project_scattered_to_grid(
    points=points,
    values=features,
    grid_shape=(64, 64, 64),
    domain_bounds=(-1.0, 1.0),
    sigma=0.03,
)

# Or use pre-cached projector for repeated iterations in optimization loops
projector = ScatteredProjector(grid_shape=(64, 64, 64), sigma=0.03)
grid_tensor = projector(points, features)
```

---

## Running the Examples and Generating Reports

An example comparing classic ANTs, PyTorch, and JAX registration is included under `examples/`. It generates a comparison report summarizing Mutual Information, Jacobian Determinants (topological safety), and Execution Speed.

To run the comparison:
```bash
python examples/generate_ants_2d_comparison_report.py
```

This generates an HTML report under `reports/ants_2d_syn_comparison.html`.

### 🌐 Scattered Data Tutorial: Joint Intensity + Curve / Mesh Alignment

`syntx` provides unified diffeomorphic registration across continuous volumetric images, sparse curve landmarks, and 3D triangular surface meshes through differentiable Nadaraya-Watson kernel projection:

- **Joint Intensity + Curve / Mesh Diffeomorphism**: By projecting Lagrangian point sets, sulcal curves, or surface mesh vertices into continuous Eulerian density channels $\mathcal{P}(X, F)$, `syntx` simultaneously optimizes dense volumetric image intensity and geometric boundary alignment in a single diffeomorphic flow:
  $$\mathcal{L}_{\text{joint}}(\phi) = w_{\text{img}} \mathcal{L}_{\text{sim}}\left(I_{\text{mov}} \circ \phi^{-1}, I_{\text{fix}}\right) + w_{\text{geom}} \mathcal{L}_{\text{geom}}\left(\mathcal{P}(X_{\text{mov}}, F_{\text{mov}}) \circ \phi^{-1}, \mathcal{P}(X_{\text{fix}}, F_{\text{fix}})\right) + \mathcal{R}(v)$$
- **3D Surface Mesh & Curve Parity**: Incorporates Euclidean Distance Transform (EDT) potential regularization and surface feature matching (such as gyral-sulcal depth), achieving sub-0.010 mm vertex accuracy with zero inverted triangles and zero grid folding.
- **Reproducible Quarto Guide**: A complete, step-by-step interactive tutorial is available in [`examples/scattered_registration_guide.qmd`](examples/scattered_registration_guide.qmd) (rendered as [`examples/scattered_registration_guide.html`](examples/scattered_registration_guide.html)). It covers:
  1. **2D Point Cloud Registration**: Non-rigid alignment of complex geometries with multi-vector Anderson acceleration.
  2. **3D Surface Mesh Registration**: Diffeomorphic warping of triangular meshes under dramatic deformations with biological feature transport.
  3. **Joint Image + Point Alignment**: Multi-channel Eulerian fusion combining procedural brain MRI (`siq`) with sparse stereotactic cortical landmarks.

To render and view the guide locally:
```bash
quarto render examples/scattered_registration_guide.qmd
open examples/scattered_registration_guide.html
```

---

## Running Tests

Tests can be executed via `pytest`:
```bash
# Run standard test suite
pytest

# Run scattered data diffeomorphic registration test suite (128 tests)
pytest tests/test_scattered*.py
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
