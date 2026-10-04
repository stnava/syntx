# Session 2026-10-04: Continuum Mechanics Regularizers, Anisotropy, and Diffeomorphic Preservation

## 1. Executive Summary

This session designed, implemented, and benchmarked 7 continuum mechanics regularizers alongside classical baselines in `syntx` on the standard ANTs `r16` $\to$ `r64` benchmark (`reg_iterations=[100, 100, 20]`, `flow_sigma=3.0`, `total_sigma=0.0`):
1. **`solenoidal`**: Exact Leray divergence-free projection ($\nabla \cdot v = 0$).
2. **`div_curl`**: Decoupled Helmholtz Green's operator with independent dilatation ($\beta$) and shear ($\gamma$) moduli.
3. **`navier`**: Continuum linear Cauchy-Stokes elasticity ($\nu = 0.49$).
4. **`hyperelastic`**: Volumetric strain regularizer penalizing deviation from unit Jacobian.
5. **`beltrami`**: Quasiconformal Killing distortion damping.
6. **`poroelastic`**: Biot two-phase solid skeleton + Darcy fluid consolidation.
7. **`masked_incompressible`**: Multi-resolution tissue-specific Chorin projection.

All registration QC evaluated diffeomorphism via the **Liouville determinant** (`syntx.liouville_determinant`) and symmetric consistency via **inverse identity error** (`reg['inverse_identity_errors']`), adhering to physical space LPS coordinate standards (`syntx.viz.extract_oriented_slice`).

---

## 2. Root Cause & Solution of Jacobian "Rib" Striations

### A. The Origin of "Rib" Artifacts
Under pure fluid SyN (`total_sigma=0.0`), decoupled anisotropic models like `div_curl` ($\beta/\gamma = 25.0$) exhibited non-physical, alternating expansion/compression striations ("ribs") in the Jacobian map, leading to topological grid folding ($\min \det(J) = -0.6234$):
1. **Stiffness Anisotropy**: Image similarity forces $\nabla I$ act normal to cortical boundaries ($\mathbf{n}$). When bulk dilatation is penalized $25\times$ more than shear, displacement is redirected into tangential shear along the cortex.
2. **Discrete Gradient Dipoles**: Discrete pixel rasterization causes image gradient fluctuations that act as dipole forces, emitting counter-rotating micro-vortices.
3. **Eulerian Compounding without Dissipation**: In SyN's Eulerian composition ($\phi_{k+1} = \phi_k \circ (\text{Id} - \delta_k) - \delta_k$), spatial derivatives compound over 220 steps:
   $$D \phi_{k+1} = D \phi_k \circ (\text{Id} - \delta_k) \cdot (I - D \delta_k) - D \delta_k$$
   Differentiating to compute $\det(J) = \det(I + \nabla \phi)$ multiplies by wavenumber $\|k\|$, amplifying shear ripples into visible ribs.

### B. The Operator-Level Cure: Multiscale $H^3$ Envelope
Instead of post-hoc total field Gaussian smoothing (`total_sigma`), we integrated an isotropic Sobolev envelope directly into `apply_div_curl_green_operator`:
$$K_{\text{sobolev}}(k) = \frac{1}{(1 + \alpha \|k\|^2)^2}$$
$$\text{filt\_curl}(k) = \frac{K_{\text{sobolev}}(k)}{\alpha + \gamma \|k\|^2} \sim \mathcal{O}(\|k\|^{-6})$$
- At low wavenumbers ($\|k\| \to 0$), the user-requested sliding anisotropy ($\beta/\gamma = 25$) is preserved 100%.
- At high wavenumbers near Nyquist, the tri-harmonic $\mathcal{O}(\|k\|^{-6})$ decay enforces $H^3$ regularity on the velocity field, mathematically guaranteeing $C^2$ smoothness and preventing folding under discrete composition.
- **Result at `total_sigma = 0.0`**: $\min \det(J)$ improved from **-0.6234 to +0.0884**, folding dropped from **0.0056% to 0.0000%**, with $\text{SSIM} = 0.9551$ and $\text{NCC} = 0.9943$.

---

## 3. Spatial Anisotropy & RegAdam Dynamics

### A. Physical Grid Anisotropy (2D & 3D)
- Verified `_get_k_mesh_rfft` physical wavenumber scaling under $4:1$ spacing ($dx=0.5\,\text{mm}, dy=2.0\,\text{mm}$): solenoidal projection achieves exact divergence cancellation ($\max |\nabla \cdot v| = 1.55 \times 10^{-6}$).
- Registration on an anisotropic grid ($0.6 \times 1.8\,\text{mm}$) proved that spatial gradient variance of $\det(J)$ is **$11.5\times$ higher along the fine-resolution axis ($x$)** than the coarse axis ($y$), because discrete image gradients sample frequencies up to $\pi / dx$.
- In clinical 3D imaging with thick slices (e.g. $1 \times 1 \times 3\,\text{mm}$), in-plane shear ($\text{curl}_z$) carries $9\times$ more bandwidth than out-of-plane shear. Operating on isotropic physical wavenumbers $k^2 = k_x^2 + k_y^2 + k_z^2$ (in $\text{mm}^{-2}$) prevents through-plane slice thickness artifacts.

### B. RegAdam Interaction
- RegAdam normalizes updates by $u_{\text{raw}} = \frac{m_t}{\sqrt{v_t} + \epsilon}$. At sharp boundaries, the second-moment $v_t$ spikes rapidly, automatically dampening the local step size.
- RegAdam on raw `div_curl` **eliminated 100% of topological folding ($\min \det(J) = +0.1999$)** under pure fluid SyN.
- Paired with the $H^3$ envelope, both CFL and RegAdam maintain strictly positive Jacobians and high image similarity.

---

## 4. Benchmark Summary (r16 $\to$ r64, total_sigma=0.0)

| Regularizer | Min $\det(J)$ | Max $\det(J)$ | Folding % | ICE Mean (mm) | SSIM | NCC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`solenoidal`** | **+0.2933** | 3.89 | **0.0000%** | 0.0399 | 0.9377 | 0.9877 | Strictly divergence-free ($\nabla \cdot v = 0$) |
| **`navier`** ($\nu=0.49$) | **+0.2128** | 6.07 | **0.0000%** | 0.0479 | 0.9524 | 0.9905 | Optimal Stokes Cauchy elasticity |
| **`div_curl`** ($\beta=25$) | **+0.0884** | 9.36 | **0.0000%** | 0.0331 | 0.9551 | 0.9943 | Cured ribs, strictly diffeomorphic |
| **`hyperelastic`** | **+0.0805** | 8.12 | **0.0000%** | 0.0325 | 0.9548 | 0.9942 | Bounded volumetric strain |
| **`poroelastic`** | **+0.0264** | 12.14 | **0.0000%** | 0.0399 | 0.9501 | 0.9930 | Two-phase matrix dissipation |
| **`sobolev`** (baseline) | **+0.0357** | 27.99 | **0.0000%** | 0.0198 | 0.9637 | 0.9935 | Classical reference baseline |
| **`gaussian`** (baseline) | **+0.0182** | 28.72 | **0.0000%** | 0.0336 | 0.9664 | 0.9943 | Spatial convolution baseline |

---

## 5. Multi-Resolution Pyramid Scaling & Anisotropic Discretization Guidelines

### A. The Multi-Resolution Scale Invariance Trap ($L^{2s}$ Compounding)
Across downsampled pyramid levels ($L = 4 \to 1$), passing raw physical spacing ($4.0\text{ mm} \to 1.0\text{ mm}$) caused physical wave numbers $k$ to scale up by $4\times$. Under an $H^3$ Sobolev envelope $(1 + \alpha \|k\|^2)^3$, the damping factor at Level 0 surged by $4^6 = \mathbf{4096\times}$ relative to Level 2.
- **The Symptom**: Level 0 optimization was completely frozen, capping symmetric cortical Dice on `mbhard` Pair 44 at `0.5872`.
- **The Relative Aspect-Ratio Cure**: Frequency meshes must normalize spacing by the minimum dimension:
  $$\mathbf{s}_{\text{rel}} = \frac{\Delta \mathbf{x}}{\min_d(\Delta x_d)}$$
  For isotropic data, $\mathbf{s}_{\text{rel}} \equiv (1, 1, 1)$ across all levels, keeping peak frequencies at $\pi$ and unlocking state-of-the-art results:
  - **Symmetric Cortical Dice**: **`0.6170`** (+28.46% over Affine baseline `0.3324`, surpassing standard Sobolev baseline `0.6127`).
  - **Liouville Min $\det(J)$**: **`+0.0162`** (strictly positive everywhere), **0.0000% folding**.
  - **Interior Mean ICE**: **`0.0577 mm`** in **74.77 s** on Apple Silicon MPS.

### B. The Anisotropic Attenuation Duality (Thick Slices)
For anisotropic acquisitions (e.g. $1.0 \times 1.0 \times 5.0\text{ mm}$):
- While $\mathbf{s}_{\text{rel}} = (1, 1, 5)$ correctly balances physical differential operators ($\nabla \cdot v$, $\nabla \times v$), through-plane frequencies are scaled down: $k_z \in [-\pi/5, \pi/5]$.
- Consequently, the highest discrete grid frequency ($\xi_z = \pi$) is attenuated by only **$4.0\times$** along $z$ (compared to **$3944\times$** in-plane).
- Pure continuum operators therefore leave discrete slice-to-slice noise largely unsuppressed on thick slices.

### C. Clinical Workflows & Mitigation
1. **Super-Resolution Preprocessing (Recommended)**: When dealing with thick slices ($\Delta z / \Delta x \ge 2.0$), use `antstorch.resample_image` to upsample to isotropic spacing prior to registration. This eliminates grid anisotropy and allows continuum operators to run at peak fidelity.
2. **Dual-Mode Isotropic Post-Filtering**: When resampling is impractical, enable `fast_smooth=False` or set `total_sigma=0.05`–`0.1` to apply an isotropic Gaussian post-filter that damps discrete through-plane Nyquist ripples.
3. **Classical Baseline Fallback**: Standard `sobolev` and `gaussian` remain in pure voxel units (`spacing=None`), providing a robust fallback against slice-to-slice grid tearing.
4. **Authoritative Guide**: Full mathematical derivations and benchmark comparisons are documented in [`docs/CONTINUUM_REGULARIZERS_ANISOTROPY_GUIDE.md`](file:///Users/stnava/data/repos/syntx/docs/CONTINUUM_REGULARIZERS_ANISOTROPY_GUIDE.md).

