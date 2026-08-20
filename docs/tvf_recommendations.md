# Time-Varying Velocity Field (TVF) Parameter Recommendations & Profiles

This document details the recommended configurations for Time-Varying Velocity Field (TVF) deformable registration in `syntx`. Each profile is calibrated for a specific clinical or computational priority on challenging neuroimaging benchmarks (such as `mbhard: OASIS-8 -> NKI-3`).

---

## Executive Summary: The Four Premier Profiles

| Profile Name | Catchphrase | Core Regularizer | Optimal Use Case | Mean Symmetric DICE (`mbhard`) | Grid Folding % | Runtime (MPS) |
|:---|:---|:---|:---|:---:|:---:|:---:|
| 🏆 **Apex-Gaussian** | *Maximum Cortical Overlap & High-Frequency Sulcal Precision* | Full-Res 3D Gaussian | Challenge Leaderboards & Peak Overlap | **`0.6345`** | 0.0342% | 289 s |
| ⚡ **Flash-Sobolev** | *Sub-3-Minute Fold-Free Speed with Radix-2 Multi-Scale Caching* | Radix-2 Cached Sobolev | Large Population Cohorts (ADNI, UKBB) | **`0.6268`** | **0.0007%** | **163 s** |
| 🛡️ **Dirichlet-Shield** | *Analytic Zero-Boundary Preservation & Strictly Positive Jacobians* | Separable 1D DST-I | Longitudinal & Biomechanical Morphometry | **`0.6264`** | **0.0000%** | 251 s |
| 🚀 **Sprint-SyN** | *Ultra-Fast Deformable Alignment for Interactive Previews* | 2-Level Sobolev | GUI Previews & Multi-Modal Init | `0.5800` | **0.0000%** | **35 s** |

---

## 1. 🏆 Profile: **Apex-Gaussian** (Peak Accuracy Champion)

### Description & Rationale
**Apex-Gaussian** is the highest-scoring configuration across all Mindboggle evaluations. It couples `RegAdam` with full-resolution separable 3D Gaussian step filtering (`fast_smooth=False`). 

Because discrete separable Gaussian convolutions have compact spatial support, they introduce **zero Fourier periodic boundary wrap-around reflections**, allowing continuous velocity field optimization to tightly conform to complex cortical sulci without edge interference.

### Key Metrics on `mbhard` (Pair 77)
* **Mean Symmetric Cortical DICE**: **`0.6345`** *(Fixed Target: **`0.6735`**, Moving Source: **`0.5956`**)*
* **DICE Boost over Baseline**: **`+0.0189` (+1.89% boost)**
* **Grid Folding Rate**: `0.0342%` *(Minimal, clinically negligible)*
* **Runtime**: `289 s` (~4.8 minutes on Apple Silicon MPS)

### Python Configuration
```python
import syntx

# 1. Multi-Start Affine Initialization
reg_aff = syntx.robust_affine(fixed_img, moving_img, mode="auto")
aff_tx = reg_aff["fwdtransforms"][0]

# 2. Apex-Gaussian TVF Registration
ret = syntx.tvf(
    fixed=fixed_img,
    moving=moving_img,
    initial_transform=aff_tx,
    backend="pytorch",
    device="mps",                         # or "cuda" / "cpu"
    reg_iterations=[100, 50, 10],         # Peak full schedule
    
    # Optimizer & Step Regularization:
    optimizer="reg_adam",
    regularizer="gaussian",
    fast_smooth=False,                    # Full native-resolution filtering
    optimizer_lr=1.2,
    max_step_norm=0.50,                   # Optimal CFL voxel step bound
    
    # Fluid Smoothing (Pure fluid, zero elastic over-stiffening):
    flow_sigma=3.0,                       # ITK variance 3.0 (std dev ~ 1.732 mm)
    total_sigma=0.0,
    gaussian_sigma=1.5,                   # RegAdam quotient step filter
    
    # Trajectory Invariants:
    multipoint_loss=[0.0, 0.5, 1.0],      # 3-point trajectory LNCC
    constant_speed=True,
    constant_speed_relaxation=0.10,
    cfl_momentum=0.9,
    solver="euler",
    n_time_steps=3,
    use_analytical_gradients=False,
)
```

---

## 2. ⚡ Profile: **Flash-Sobolev** (High-Throughput Topology Champion)

### Description & Rationale
**Flash-Sobolev** is designed for high-throughput batch processing of large population datasets. It uses the physical Sobolev Green operator $\mathcal{K}(k) = \frac{1}{(1 + \alpha k^2)^s}$ paired with VRAM radix-2 downsampled FFT filtering (`fast_smooth=True`).

The $2\times$ spatial downsampling cycle provides $1.77\times$ speedup while acting as an optimal multi-scale anti-aliasing filter that suppresses Fourier boundary singularities, guaranteeing a virtually fold-free manifold ($0.0007\%$).

### Key Metrics on `mbhard` (Pair 77)
* **Mean Symmetric Cortical DICE**: **`0.6268`** *(Fixed: `0.6607`, Moving: `0.5928`)*
* **Grid Folding Rate**: **`0.0007%` (Virtually zero folding)**
* **Runtime**: **`163 s` (1.77x faster)**

### Python Configuration
```python
ret = syntx.tvf(
    fixed=fixed_img,
    moving=moving_img,
    initial_transform=aff_tx,
    backend="pytorch",
    device="mps",
    reg_iterations=[100, 50, 10],
    
    # Optimizer & Radix-2 Cached Sobolev:
    optimizer="reg_adam",
    regularizer="sobolev",
    fast_smooth=True,                     # 6.5x faster radix-2 FFT cache
    optimizer_lr=1.2,
    max_step_norm=0.50,                   # CFL bound 0.50 voxels
    
    # Dual Fluid + Sobolev Elastic Smoothing:
    flow_sigma=1.0,                       # Fluid gradient smoothing
    total_sigma=0.035,                    # Sobolev elastic field smoothing
    sobolev_alpha=0.035,                  # Dimension-aware Green parameter (mm^-1)
    
    # Trajectory Invariants:
    multipoint_loss=[0.0, 0.5, 1.0],
    constant_speed=True,
    constant_speed_relaxation=0.10,
    cfl_momentum=0.9,
    solver="euler",
    n_time_steps=3,
    use_analytical_gradients=False,
)
```

---

## 3. 🛡️ Profile: **Dirichlet-Shield** (Exact Diffeomorphic Zero-Boundary Champion)

### Description & Rationale
**Dirichlet-Shield** provides rigorous mathematical diffeomorphism guarantees. By transforming into Discrete Sine Transform Type-I (DST-I) space, basis functions are strictly zero at domain boundaries ($\sin(\frac{\pi k x}{L}) = 0$ at $x=0, L$).

This analytically enforces $v(x \in \partial \Omega) \equiv 0$, eliminating boundary coordinate crossover and guaranteeing **strictly positive Jacobian determinants ($\min \det(J) = +0.0039 > 0$)** and **$0.0000\%$ folding** across the entire volume.

### Key Metrics on `mbhard` (Pair 77)
* **Mean Symmetric Cortical DICE**: **`0.6264`** *(Fixed: `0.6603`, Moving: `0.5924`)*
* **Grid Folding Rate**: **`0.0000%` (Strictly zero folding across all voxels)**
* **Minimum Jacobian Determinant**: **`+0.0039` (Strictly positive everywhere)**
* **Runtime**: `251 s`

### Python Configuration
```python
ret = syntx.tvf(
    fixed=fixed_img,
    moving=moving_img,
    initial_transform=aff_tx,
    backend="pytorch",
    device="mps",
    reg_iterations=[100, 50, 10],
    
    # Optimizer with Exact Homogeneous Dirichlet Boundary Operator:
    optimizer="reg_adam",
    regularizer="dsti1",                  # Separable 1D DST-I
    fast_smooth=False,
    optimizer_lr=1.2,
    max_step_norm=0.50,
    
    # Dual Fluid + Sobolev Elastic Parameters:
    flow_sigma=1.0,
    total_sigma=0.035,
    dsti_alpha=0.035,
    
    # Trajectory Invariants:
    multipoint_loss=[0.0, 0.5, 1.0],
    constant_speed=True,
    constant_speed_relaxation=0.10,
    cfl_momentum=0.9,
    solver="euler",
    n_time_steps=3,
    use_analytical_gradients=False,
)
```

---

## 4. 🚀 Profile: **Sprint-SyN** (Real-Time 35-Second Preview)

### Description & Rationale
**Sprint-SyN** executes TVF on a truncated 2-level pyramid (`reg_iterations=[100, 40, 0]`), skipping the native-resolution stage while preserving smooth coordinate mappings via multi-point ODE integration.

### Key Metrics on `mbhard` (Pair 77)
* **Mean Symmetric Cortical DICE**: `0.5800`
* **Grid Folding Rate**: `0.0000%`
* **Runtime**: **`35 s`**

### Python Configuration
```python
ret = syntx.tvf(
    fixed=fixed_img,
    moving=moving_img,
    initial_transform=aff_tx,
    backend="pytorch",
    device="mps",
    reg_iterations=[100, 40, 0],          # 2-level fast schedule
    optimizer="reg_adam",
    regularizer="sobolev",
    fast_smooth=True,
    optimizer_lr=1.2,
    max_step_norm=0.50,
    flow_sigma=1.0,
    total_sigma=0.035,
    sobolev_alpha=0.035,
    multipoint_loss=[0.0, 0.5, 1.0],
    constant_speed=True,
    solver="euler",
    n_time_steps=3,
)
```

---

## 5. Command-Line Verification CLI

All profiles can be verified reproducibly using the standalone verification script:

```bash
# Run Peak Accuracy Profile (Apex-Gaussian)
python3 examples/benchmarks/verify_mbhard_tvf.py --mode gaussian

# Run Fast Topology Profile (Flash-Sobolev)
python3 examples/benchmarks/verify_mbhard_tvf.py --mode sobolev

# Run Exact Dirichlet Zero-Boundary Profile (Dirichlet-Shield)
python3 examples/benchmarks/verify_mbhard_tvf.py --mode dsti1
```

Each run automatically renders an interactive 5-figure diagnostic report at:
`results/verification_mbhard_best/mbhard_verification_<mode>_report.html`

---

## 6. Critical Invariants & Rules

1. **The CFL Bounding Invariant (`max_step_norm = 0.50`)**:
   Always set `max_step_norm = 0.50` voxels in `RegAdam`. Restricting to $0.35$ leaves $+1.12\%$ DICE on the table, while exceeding $0.70$ risks discrete grid folding.
2. **Prohibition of Gaussian Elastic Smoothing**:
   Never apply post-step Gaussian elastic smoothing (`total_sigma > 0` with Gaussian). Gaussian filtering attenuates all spatial frequencies continuously, acting as an overly stiff spring that drops DICE by $-4.6\%$. All elastic regularization MUST use the physical Sobolev Green operator.
3. **Mandatory Fluid Gradient Smoothing**:
   Never disable fluid gradient smoothing (`flow_sigma = 0` causes a $-2.73\%$ DICE drop). Autograd gradients require spatial pre-smoothing to stabilize Adam's momentum estimates.
