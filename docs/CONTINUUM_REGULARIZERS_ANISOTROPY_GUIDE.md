# Continuum Regularizers, Anisotropic Discretization, and Multi-Resolution Scaling

## 1. Executive Overview

This guide establishes the theoretical foundations, implementation standards, and clinical guidelines for continuum mechanics regularizers (`div_curl`, `navier`, `solenoidal`, `elastic`, `hyperelastic`, `beltrami`, `poroelastic`) within `syntx`. It addresses two fundamental spatial challenges:
1. **Multi-Resolution Scale Invariance**: Preventing fine-level optimizer freezing caused by physical grid scaling across image pyramids.
2. **Anisotropic Voxel Discretization**: Balancing true physical differential kinematics ($\nabla \cdot v$, $\nabla \times v$, Cauchy stress) against discrete through-plane slice noise on anisotropic clinical acquisitions (e.g., $1\times 1\times 3\text{ mm}$ MRI or $0.7\times 0.7\times 2.5\text{ mm}$ CT).
3. **Mitigation Strategies**: Outlining when and how to deploy super-resolution preprocessing, dual-mode isotropic post-filtering, and classical voxel-unit baselines.

---

## 2. The Multi-Resolution Scaling Trap ($L^{2s}$ Freezing)

In coarse-to-fine registration pyramids, the downsampling factor $L$ scales the effective voxel dimension:
$$\Delta x_{\text{level}} = L \cdot \Delta x_{\text{full}}$$
For a standard 3-level schedule ($[4, 2, 1]$), an isotropic $1.0\text{ mm}$ volume is processed at:
- **Level 2 (Coarse)**: $L = 4 \implies \Delta x = 4.0\text{ mm}$
- **Level 1 (Medium)**: $L = 2 \implies \Delta x = 2.0\text{ mm}$
- **Level 0 (Fine)**: $L = 1 \implies \Delta x = 1.0\text{ mm}$

### The Mathematical Origin of Fine-Level Freezing
In continuous mechanics, the Green's operator for an $H^s$ Sobolev or biharmonic/triharmonic continuum filter is defined in Fourier space by:
$$G(k) = \frac{1}{(1 + \alpha \|k\|^2)^s}$$
where $\alpha$ represents the regularizer smoothing scale.

When wave numbers $k$ are formulated with raw physical spacing ($k_d = \frac{2\pi f_d}{N_d \cdot \Delta x_d} \text{ rad/mm}$):
- At Level 2 ($4.0\text{ mm}$): $k_{\max} = \frac{\pi}{4.0} \approx 0.785\text{ rad/mm}$
- At Level 0 ($1.0\text{ mm}$): $k_{\max} = \frac{\pi}{1.0} \approx 3.142\text{ rad/mm}$

Because the Nyquist frequency shifts by a factor of $L = 4$, the squared wavenumber $\|k\|^2$ increases by **$16\times$** at the fine level. When compounded through an $H^3$ envelope ($s=3$):
$$\text{Attenuation Ratio} = \left(\frac{1 + \alpha \cdot 3.142^2}{1 + \alpha \cdot 0.785^2}\right)^3 \approx \left(\frac{1 + 1.5 \times 9.87}{1 + 1.5 \times 0.617}\right)^3 \approx \left(\frac{15.80}{1.93}\right)^3 \approx \mathbf{548\times}$$

**The Failure Mode**: The fine pyramid level experiences over $500\times$ heavier damping than the coarse level. The optimizer is unable to displace velocity vectors along fine sulcal and gyral folds, artificially capping symmetric cortical Dice (e.g. at `0.5872` on `mbhard` Pair 44).

---

## 3. The Unified Solution: Relative Aspect-Ratio Normalization

To eliminate inter-level bandwidth drift while preserving true physical spatial kinematics, `syntx` normalizes the spacing vector by the finest spatial dimension:
$$\mathbf{s}_{\text{rel}} = \frac{\Delta \mathbf{x}}{\min_d(\Delta x_d)}$$

### Frequency Grid Formulation (`_get_k_mesh_rfft`)
Normalized angular frequencies along spatial axis $d$ are constructed via:
$$k_d^{\text{norm}} = \frac{\xi_d}{s_{\text{rel}, d}} \quad \text{where } \xi_d \in [-\pi, \pi]$$
where $\xi_d$ is the standard dimensionless discrete grid frequency.

```
                            ┌───────────────────────────────────────────────┐
                            │    Anisotropic Data: Δx = (1.0, 1.0, 3.0) mm   │
                            └───────────────────────┬───────────────────────┘
                                                    │
                 ┌──────────────────────────────────┼──────────────────────────────────┐
                 ▼                                  ▼                                  ▼
   [Paradigm A: Pure Voxel Units]     [Paradigm B: Raw Physical mm]    [Paradigm C: Relative Aspect Ratio]
   spacing = None                     spacing = (1.0, 1.0, 3.0)        s_rel = Δx / min(Δx) = (1, 1, 3)
   ─────────────────────────────      ────────────────────────────     ───────────────────────────────────
   • Scale invariant across levels    • Physical derivative balance    • Physical derivative balance PRESERVED
   • Directional geometry DISTORTED   • Compound L^(2s) damping        • Scale invariant across all levels
   • z-axis smoothed 3× more in mm      SURGE freezes fine pyramid     • Through-plane Nyquist cutoff natural
```

### Key Mathematical Guarantees:
1. **Pyramid Scale Invariance**:
   For any downsampling factor $L$, $\min_d(L \Delta x_d) = L \min_d(\Delta x_d)$. Thus:
   $$\mathbf{s}_{\text{rel}}^{\text{coarse}} = \frac{L \Delta \mathbf{x}}{L \min_d(\Delta x_d)} \equiv \mathbf{s}_{\text{rel}}^{\text{fine}}$$
   The relative aspect ratio is strictly invariant across all resolution stages. The peak frequency along the primary axis is always $\pi$, ensuring consistent multi-resolution bandwidth.
2. **Physical Kinematic Invariance**:
   Spatial derivatives evaluate as:
   $$\nabla_{\text{rel}} = \left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{1}{s_{\text{rel}, z}}\frac{\partial}{\partial z}\right)$$
   Volumetric dilatation $\nabla \cdot v$, vorticity $\nabla \times v$, and Cauchy-Navier elasticity tensors correctly weight through-plane vs. in-plane physical strains, ensuring orientation invariance.

---

## 4. The Anisotropic Attenuation Duality (The Thick-Slice Trap)

While relative aspect-ratio scaling is mathematically mandatory for physical continuum operators, it introduces a critical discretization trade-off on thick-slice clinical data.

### The Problem: Vanishing Through-Plane Damping
Consider an acquisition with thick 2D slices: **$1.0 \times 1.0 \times 5.0\text{ mm}$** ($\mathbf{s}_{\text{rel}} = (1, 1, 5)$).
Along the through-plane axis $z$, the maximum normalized frequency is:
$$k_z \in \left[-\frac{\pi}{5}, \frac{\pi}{5}\right] \implies k_z^2 \le \frac{\pi^2}{25} \approx 0.395$$

Comparing the attenuation of the **highest discrete grid frequency** (the discrete Nyquist mode $\xi = \pi$, representing alternating voxel noise $+ - + -$):
* **In-plane Nyquist mode ($\xi_x = \pi$)**:
  $$K_x(\pi) = \left(1 + 1.5 \cdot \pi^2\right)^{-3} \approx (1 + 14.80)^{-3} \approx \frac{1}{\mathbf{3944}}$$
* **Through-plane Nyquist mode ($\xi_z = \pi$)**:
  $$K_z(\pi) = \left(1 + 1.5 \cdot \left(\frac{\pi}{5}\right)^2\right)^{-3} \approx (1 + 0.59)^{-3} \approx \frac{1}{\mathbf{4.0}}$$

### The Clinical Phenomenon:
Because the continuum physical model treats $5.0\text{ mm}$ as a "long physical distance", it interprets alternating inter-slice variations as low-frequency signals. Consequently:
- **Discrete through-plane noise is attenuated by only $4\times$ (compared to nearly $4000\times$ in-plane)**.
- Discrete gradient dipoles, slice-timing jitter, or inter-slice interpolation noise pass directly through the filter.
- This can lead to slice-to-slice displacement jaggedness, out-of-plane shear ripples, or localized grid folding.

---

## 5. Mitigation Strategies & Clinical Workflows

To achieve robust registration across all imaging regimes, `syntx` adopts three complementary strategies:

```
                      Clinical Acquisition (e.g. 1.0 × 1.0 × 3.0 mm)
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │   Is Anisotropy Ratio > 2.0:1?  │
                           └────────────────┬────────────────┘
                                            │
                           ┌────────────────┴────────────────┐
                           ▼                                 ▼
                         [YES]                              [NO]
               ┌───────────────────────┐          ┌───────────────────────┐
               │ Super-Resolution /    │          │ Direct Continuum SyN  │
               │ Isotropic Resampling  │          │ (div_curl, navier)    │
               │ (antstorch.resample)  │          │ total_sigma = 0.0     │
               └───────────┬───────────┘          └───────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │ Fallback / Dual Mode: │
               │ • fast_smooth = False │
               │ • total_sigma = 0.05  │
               │ • regularizer='sobolev│
               └───────────────────────┘
```

### Strategy 1: Super-Resolution Preprocessing (Recommended)
When high-accuracy cortical or subcortical registration is required on thick slices ($\Delta z / \Delta x \ge 2.0$), the optimal approach is to upsample the through-plane axis to isotropic or near-isotropic resolution prior to registration:
```python
import antstorch

# Resample anisotropic volume (1.0 x 1.0 x 3.0 mm) to isotropic (1.0 mm)
fixed_iso = antstorch.resample_image(fixed_img, resample_params=(1.0, 1.0, 1.0), use_voxels=False, interp_type=4)
moving_iso = antstorch.resample_image(moving_img, resample_params=(1.0, 1.0, 1.0), use_voxels=False, interp_type=4)

# Register with pure fluid continuum SyN
reg = syntx.syn(fixed_iso, moving_iso, regularizer='div_curl', total_sigma=0.0)
```
* **Why this succeeds**: Super-resolution eliminates spatial grid anisotropy at the input, restoring full tri-harmonic damping along all axes while allowing continuum operators to operate at peak fidelity.

### Strategy 2: Dual-Mode Isotropic Post-Filtering (`fast_smooth=False` or `total_sigma > 0`)
If resampling to isotropic space is impractical (e.g. memory constraints), pair the continuum operator with a light isotropic voxel-space post-filter:
```python
# Enable isotropic Gaussian post-filtering to suppress discrete slice noise
reg = syntx.syn(
    fixed_aniso, moving_aniso,
    regularizer='div_curl',
    total_sigma=0.05,        # Minimal total elastic dissipation
    fast_smooth=False,       # Applies isotropic post-filter
)
```
* **Why this succeeds**: The continuum Green's operator correctly balances physical divergence and curl in relative coordinates, while the isotropic voxel post-filter guarantees that discrete Nyquist grid noise ($\xi = \pi$) is uniformly quenched across all directions.

### Strategy 3: Classical Baseline Fallback (`regularizer='sobolev'` / `'gaussian'`)
For legacy pipelines or raw clinical scans where physical elasticity modeling is secondary to extreme topological stability:
- `regularizer='sobolev'` and `regularizer='gaussian'` in `syntx.syn` operate in **pure voxel units** (`spacing=None`, `sigma_mode='voxel'`).
- They smooth by an equal number of voxels across all dimensions, providing maximum resistance to inter-slice grid tearing at the expense of physical aspect-ratio directional balance.

---

## 6. Implementation Reference Table

| Regularizer | Implementation Function | Spacing Parameter Handling | Recommended Regime |
| :--- | :--- | :--- | :--- |
| **`div_curl`** | `apply_div_curl_green_operator` | `_get_k_mesh_rfft(..., relative_spacing=True)` | Pure fluid SyN (`total_sigma=0.0`), brain MRI, volumetric sliding. |
| **`navier`** | `apply_navier_green_operator` | `_get_k_mesh_rfft(..., relative_spacing=True)` | Diffeomorphic preservation, high tissue incompressibility ($\nu=0.49$). |
| **`solenoidal`** | `project_solenoidal` | `_get_k_mesh_rfft(..., relative_spacing=True)` | Incompressible hydrodynamics, zero volumetric expansion. |
| **`hyperelastic`** | `apply_hyperelastic_regularizer` | `_get_k_mesh_rfft(..., relative_spacing=True)` | Finite strain elasticity, bounded Jacobian determinant ranges. |
| **`sobolev`** | `apply_sobolev_green_operator` | `spacing=None` (Pure Voxel Units) | High-speed general registration, robust baseline on thick slices. |
| **`gaussian`** | `separable_gaussian_filter` | `sigma_mode='voxel'` (Pure Voxel Units) | Standard ANTs/ITK parity, extreme discrete noise suppression. |
| **`bspline`** | `smooth_displacement_field_bspline` | Physical Spacing (mm) | Smooth low-frequency deformation fields, spline parameterization. |

---

## 7. Verification & Historical Evidence

Empirical findings from canonical Mindboggle benchmarks (`mbhard`, Pair 44) under pure fluid SyN (`total_sigma=0.0`):
- **Raw Physical Spacing**: Dice `0.5872` (Level 0 frozen by $548\times$ over-damping).
- **Sobolev Baseline (Voxel Units)**: Dice `0.6127`, Min $\det(J) = +0.0080$, Folding `0.0000%`.
- **`div_curl_iso_tot0` (Relative Aspect Ratio)**: Dice **`0.6170`** (+28.46% over Affine), Min $\det(J) = \mathbf{+0.0162}$, Folding **`0.0000%`**, Mean ICE **`0.0577 mm`**, Runtime **74.77 s** on Apple Silicon MPS.
- Provenance: [`docs/provenance/mbhard_standard_div_curl_iso_tot0_results.json`](file:///Users/stnava/data/repos/syntx/docs/provenance/mbhard_standard_div_curl_iso_tot0_results.json).
- Visual QC Report: [`docs/reports/mbhard_standard_div_curl_iso_tot0_report.html`](file:///Users/stnava/data/repos/syntx/docs/reports/mbhard_standard_div_curl_iso_tot0_report.html).

---

## 8. Diffeomorphic Shape Capture Range and Directional Invariance

### 8.1 The "Single-Dial" Dilemma vs. Continuum Decoupling
Scalar isotropic smoothers (`gaussian`, `sobolev`) have only one dial: **global isotropic stiffness**. When registering brains with substantial morphological disparity (differing sulcal patterns, disparate ventricular sizes):
* Relaxing stiffness to capture large shape displacements causes topological folding ($\det(J) \le 0$) in compressive zones.
* Stiffening the filter prevents folding, but simultaneously freezes tangential shear along the cortex ($J_{\max} \approx 28 - 40$).
* **Continuum mechanics regularizers (`syn_hyperelastic`, `syn_divcurl`)** decouple volumetric strain ($\nabla \cdot \mathbf{v}$) from isochoric shear ($\nabla \times \mathbf{v}$). The high bulk modulus ($K=5.0$, $\beta=4.0$) acts as a hard stop against volume collapse, unlocking the optimizer to safely pursue **large tangential shape morphing** ($J_{\max} \approx 110 - 150$) without folding ($0.0000\%$).

### 8.2 Cohort Population Findings (90-Pair Benchmark)
* **High Morphological Disparity Pairs (Top 40%):** Hyperelastic and DivCurl outperform Sobolev and ANTs C++ by $+1.0\%$ to $+2.9\%$ Dice (e.g. Pair 44 `mbhard`: Hyperelastic `0.6165`, DivCurl `0.6162` vs. Sobolev `0.6011` and ANTs `0.5876`).
* **Directional Invariance under SyN:** Because SyN optimizes symmetric geodesic half-paths to an unobserved virtual midpoint ($t=0.5$), the continuum advantage is direction-invariant. Measured directional disparity $|\text{Dice}_{\text{fixed}} - \text{Dice}_{\text{moving}}|$ in `syntx` is $< 0.0058$ ($0.58\%$) across all pairs, compared to $0.0292$ ($2.92\%$) in ANTs C++.

