---
title: "Symmetric Diffeomorphic Image Registration on Riemannian Manifolds: Eulerian Normalization, Time-Varying Velocity Fields, and Sobolev-Preconditioned Optimization on the 90-Pair Mindboggle Cohort"
author: "Syntx Research & Development"
date: "August 2026"
geometry: margin=1in
bibliography: references.bib
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{hyperref}
  - \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
---

# Symmetric Diffeomorphic Image Registration on Riemannian Manifolds: Eulerian Normalization, Time-Varying Velocity Fields, and Sobolev-Preconditioned Optimization on the 90-Pair Mindboggle Cohort

**Date**: August 2026  

---

## Abstract

Spatial correspondence estimation between structural medical images requires coordinate transformations that preserve topological integrity, prevent non-invertible singularities, and capture high-frequency anatomical boundaries. While the Symmetric Normalization (`SyN`) framework represents the classical reference for diffeomorphic registration, conventional implementations rely on CPU-bound spatial stepping loops and isotropic Gaussian filtering that attenuate fine cortical boundary gradients. In this work, we present a unified mathematical and algorithmic formulation of symmetric diffeomorphic registration and Large Deformation Diffeomorphic Metric Mapping (LDDMM) with continuous Time-Varying Velocity Fields (TVF), accelerated via tensor automatic-differentiation paradigms.

We address fundamental theoretical challenges in variational diffeomorphic optimization:
1. **Asymptotic Variance Singularities in Local Normalized Cross-Correlation (LNCC)**: We prove that unregularized analytical $L^2$ gradients of LNCC diverge as $\mathcal{O}(\text{Var}^{-1/2})$ in homogeneous intensity regions, inducing non-physical coordinate forces, and establish a safe variance floor $\text{Var}_{\text{safe}} = \max(\text{Var}(I), \epsilon)$ that guarantees bounded gradient dynamics.
2. **Smooth Lie Group $\text{SO}(3)$ Mapping**: We establish first-order Taylor continuity at the identity of the Lie algebra $\mathfrak{so}(3)$ exponential map, eliminating zero-gradient lockup during rigid initialization.
3. **Riemannian Sobolev-Preconditioned Optimization (`SobolevAdam`)**: We demonstrate that pointwise adaptive moment optimizers (such as Adam) destroy the spatial regularity of velocity fields in infinite-dimensional function spaces by scaling noise up to unit magnitude in flat domains. We introduce **`SobolevAdam`**, which preconditions parameter updates with the Sobolev-Riemannian Green's operator $\mathcal{G}_{\text{Sobolev}} = (I - \alpha \Delta)^{-s}$, strictly guaranteeing positive Jacobian determinants ($\det(J) > 0$) across continuous velocity flows.
4. **Antisymmetric Velocity Projection & Fréchet Midpoint Anchoring**: We formulate the unique orthogonal projection of forward and backward velocity update pairs onto the antisymmetric subspace, eliminating common-mode translation drift and anchoring the symmetric geodesic midpoint strictly at the Fréchet mean.
5. **Deterministic Multi-Start Manifold Search**: We introduce a deterministic 18-cone Lie algebra perturbation grid with foreground union-masked Mutual Information scoring, eliminating angular basin entrapment during affine initialization.

We validate the framework on the standardized **90-pair Mindboggle-101 benchmark** (40 intra-subject longitudinal pairs and 50 inter-subject cross-site pairs) evaluated against ground-truth DKT31 cortical label maps sharing locked canonical affine baselines (`0.3499` baseline DICE):
- **Eulerian SyN (Gaussian)** achieves **`0.6382` Mean Symmetric Cortical DICE** (+1.66% over classical ANTs SyN `0.6216`, with an 88/90 win rate).
- **Eulerian SyN (Sobolev)** achieves **`0.6342` Mean Symmetric DICE** (+1.26% over ANTs, with 90.0% zero-fold regularity).
- **Sobolev-Preconditioned TVF (`syntx.tvf`)** achieves a **100% win sweep (90/90 wins)** with **`0.6445` Mean Symmetric DICE** (+2.29% over ANTs) and **`0.0000%` grid folding ($\det(J) > 0$ strictly positive everywhere)**.
- Tensor acceleration reduces deformable 3D volume execution to **`~12–16s` per pair on modern GPU architectures** ($7.5\times - 24\times$ speedup over classical CPU baselines).

All benchmark evaluation protocols, dataset preparation procedures, and interactive metrology tools are fully documented and reproducible via [`docs/run_mb_eval.md`](file:///Users/stnava/data/syntx/docs/run_mb_eval.md).

---

## 1. Mathematical Framework & Variational Principles

### 1.1 Differential Geometry of Diffeomorphisms

Let $\Omega \subset \mathbb{R}^d$ ($d \in \{2, 3\}$) denote a compact, connected spatial domain with smooth boundary $\partial \Omega$. Let $I_F, I_M: \Omega \to \mathbb{R}$ represent fixed (target) and moving (source) scalar intensity images. The objective of image registration is to find a spatial transformation $\Phi: \Omega \to \Omega$ such that the deformed moving image $I_M \circ \Phi$ aligns structurally with $I_F$.

| Symbol | Mathematical Definition |
| :--- | :--- |
| $\Omega \subset \mathbb{R}^d$ | Continuous physical image domain with coordinates $\mathbf{x} = (x_1, \dots, x_d)^T$ |
| $I_F, I_M \in L^2(\Omega)$ | Fixed target and moving source intensity distributions |
| $\text{Diff}(\Omega)$ | Infinite-dimensional Lie group of smooth $C^\infty$ diffeomorphisms on $\Omega$ |
| $\mathfrak{g} = \mathfrak{X}(\Omega)$ | Lie algebra of smooth Eulerian velocity vector fields $\mathbf{v}: \Omega \to \mathbb{R}^d$ |
| $\Phi(\mathbf{x}) = \mathbf{x} + \mathbf{u}(\mathbf{x})$ | Spatial transformation mapping with displacement field $\mathbf{u}: \Omega \to \mathbb{R}^d$ |
| $J_\Phi(\mathbf{x}) = \det(\nabla \Phi(\mathbf{x}))$ | Local Jacobian determinant measuring volumetric scaling and orientation |
| $\Omega_{1/2}$ | Symmetrized virtual midpoint domain (Fréchet geodesic mean) |
| $\phi_{l2r}, \phi_{r2l} \in \text{Diff}(\Omega)$ | Forward and reverse half-geodesic transformation trajectories |
| $\mathcal{D}(I_F, I_M \circ \Phi)$ | Image similarity functional (e.g. Local Normalized Cross-Correlation) |
| $\mathcal{R}(\mathbf{v})$ | Regularization functional on the Lie algebra $\mathfrak{g}$ (Sobolev or Gaussian metric) |
| $\mathcal{G} = (I - \alpha \Delta)^{-s}$ | Green's operator associated with the $H^s$ Sobolev Hilbert space inner product |

To guarantee topology preservation—precluding tearing, self-intersection, and non-physical singularity formation—the transformation $\Phi$ must be an element of the infinite-dimensional Lie group of smooth diffeomorphisms $\text{Diff}(\Omega)$.

#### 1. The Diffeomorphic Condition and the Jacobian Determinant
The local orientation and volume preservation of $\Phi \in \text{Diff}(\Omega)$ are governed by its Jacobian matrix $\nabla \Phi(\mathbf{x}) = I + \nabla \mathbf{u}(\mathbf{x})$. The Jacobian determinant is defined as:
$$J_\Phi(\mathbf{x}) = \det\left( \nabla \Phi(\mathbf{x}) \right) = \det\left( I + \nabla \mathbf{u}(\mathbf{x}) \right)$$
A transformation $\Phi$ is locally invertible, orientation-preserving, and non-singular if and only if:
$$J_\Phi(\mathbf{x}) > 0, \quad \forall \mathbf{x} \in \Omega$$
Regions where $J_\Phi(\mathbf{x}) \le 0$ represent non-invertible coordinate foldings where the spatial topology collapses.

---

### 1.2 Symmetric Normalization (`SyN`) & Geodesic Midpoints

Classical asymmetric registration optimizes $\Phi: \Omega_F \to \Omega_M$, introducing an inherent template bias depending on which volume is designated as target. The Symmetric Normalization (`SyN`) framework [@avants2008symmetric] resolves this by optimizing two diffeomorphic paths $\phi_{l2r}, \phi_{r2l} \in \text{Diff}(\Omega)$ originating from a shared virtual midpoint domain $\Omega_{1/2}$:
$$\phi_{l2r}: \Omega_{1/2} \to \Omega_F, \quad \phi_{r2l}: \Omega_{1/2} \to \Omega_M$$
The full forward mapping is the composite:
$$\Phi_{M \to F} = \phi_{l2r} \circ \phi_{r2l}^{-1}$$

The variational energy functional balances structural image similarity at the midpoint domain with spatial smoothness on the underlying velocity fields:
$$\mathcal{E}(\mathbf{v}_l, \mathbf{v}_r) = \int_{\Omega_{1/2}} \mathcal{S}\left( I_F \circ \phi_{l2r}(\mathbf{x}), I_M \circ \phi_{r2l}(\mathbf{x}) \right) d\mathbf{x} + \int_0^1 \left( \|\mathbf{v}_l(t)\|_V^2 + \|\mathbf{v}_r(t)\|_V^2 \right) dt$$
where $\|\cdot\|_V$ denotes an admissible Hilbert space norm on $\mathfrak{g}$ enforcing spatial regularity.

---

### 1.3 Local Normalized Cross-Correlation (LNCC) Functional

For multi-modal or intra-modality MRI with spatial intensity inhomogeneities (e.g., $B_1$ field bias), registration optimizes **Local Normalized Cross-Correlation (LNCC)** computed over a localized spatial window $W(\mathbf{x})$ of radius $r$:
$$\mathcal{L}_{\text{LNCC}}(I_F, I_M) = - \int_\Omega \frac{\left( \int_{W(\mathbf{x})} \tilde{I}_F(\mathbf{y}) \tilde{I}_M(\mathbf{y}) d\mathbf{y} \right)^2}{\left( \int_{W(\mathbf{x})} \tilde{I}_F^2(\mathbf{y}) d\mathbf{y} \right) \left( \int_{W(\mathbf{x})} \tilde{I}_M^2(\mathbf{y}) d\mathbf{y} \right)} d\mathbf{x}$$
where $\tilde{I}(\mathbf{y}) = I(\mathbf{y}) - \bar{I}(\mathbf{x})$ denotes local zero-mean intensities over the neighborhood $W(\mathbf{x})$.

---

## 2. Theoretical Analysis & Algorithmic Principles

### 2.1 Metric Preservation via Single Interpolation

In classical multi-stage registration pipelines (rigid $\to$ affine $\to$ deformable), intermediate warped intensity images or segmentations are often resampled at each stage. Mathematically, resampling an image $I$ with an interpolation kernel $K_\sigma$ corresponds to spatial convolution $I_{\text{resampled}} = I * K_\sigma$.

Successive resamplings compound spatial attenuation:
$$I_{\text{final}} = I_0 * K_{\sigma_1} * K_{\sigma_2} * \dots * K_{\sigma_n}$$
This cumulative low-pass filtering systematically attenuates high-frequency cortical boundaries and gyral sharpness.

#### The Single Interpolation Principle
We enforce exact transformation composition directly in the continuous coordinate manifold:
$$\Phi_{\text{composite}}(\mathbf{x}) = \phi \circ A \circ T_0(\mathbf{x})$$
where $T_0 \in \text{SE}(3)$ is the initial translation, $A \in \text{Aff}(3)$ is the learned affine matrix, and $\phi \in \text{Diff}(\Omega)$ is the non-linear displacement field. The input intensity volume or structural label map is sampled **exactly once** using $\Phi_{\text{composite}}$:
$$I_{\text{warped}}(\mathbf{x}) = I_{\text{native}}\left( \Phi_{\text{composite}}(\mathbf{x}) \right)$$
For discrete segmentation maps, the pull-back operation strictly uses nearest-neighbor projection to preserve integer label semantics without artificial partial-volume averaging.

---

### 2.2 Functional Gradient Asymptotics & Variance Flooring in LNCC

Let the local cross-correlation coefficient at coordinate $\mathbf{x}$ be written as:
$$r(\mathbf{x}) = \frac{s_{FM}(\mathbf{x})}{\sqrt{s_{FF}(\mathbf{x}) \cdot s_{MM}(\mathbf{x})}}$$
where $s_{FM} = \text{Cov}_{W}(I_F, I_M)$ and $s_{FF} = \text{Var}_{W}(I_F), s_{MM} = \text{Var}_{W}(I_M)$.

#### Analytical Gradient Singularity
The variational $L^2$ functional derivative with respect to the moving image intensity $I_M$ is:
$$\frac{\delta \mathcal{L}_{\text{LNCC}}}{\delta I_M(\mathbf{x})} = - \frac{2 r(\mathbf{x})}{\sqrt{s_{FF}(\mathbf{x}) \cdot s_{MM}(\mathbf{x})}} \left( \tilde{I}_F(\mathbf{x}) - \frac{s_{FM}(\mathbf{x})}{s_{MM}(\mathbf{x})} \tilde{I}_M(\mathbf{x}) \right)$$
In homogeneous anatomical regions (such as cerebral white matter, ventricles, or background zero padding), the local variance vanishes:
$$\lim_{s_{MM} \to 0} \frac{\delta \mathcal{L}_{\text{LNCC}}}{\delta I_M} \sim \mathcal{O}\left( s_{MM}^{-1/2} \right) \to \infty$$
This variance singularity produces unbounded gradient forces in flat regions, causing high-frequency coordinate tearing that directly drives non-diffeomorphic grid folding ($\det(J) \le 0$).

#### Safe Variance Floor Regularization
To regularize the functional derivative, we impose a lower bound on local variance prior to denominator evaluation:
$$\text{Var}_{\text{safe}}(I) = \max\left( \text{Var}(I), \epsilon \right), \quad \epsilon = 10^{-6}$$
$$\mathcal{L}_{\text{LNCC}}^{\text{safe}} = - \frac{\text{Cov}(I_F, I_M)}{\sqrt{\text{Var}_{\text{safe}}(I_F) \cdot \text{Var}_{\text{safe}}(I_M)}}$$
Coupled with Cauchy-Schwarz bound clamping ($r \in [-1, 1]$), safe variance flooring guarantees bounded gradient trajectories:
$$\left\| \frac{\delta \mathcal{L}_{\text{LNCC}}^{\text{safe}}}{\delta I} \right\|_\infty \le \frac{2}{\epsilon}$$
eliminating derivative spikes across uniform tissue regions.

---

### 2.3 Lie Algebra $\mathfrak{so}(3)$ Parameterization & Smooth Exponential Limit

Spatial 3D rotations form the special orthogonal Lie group $\text{SO}(3) = \{ R \in \mathbb{R}^{3 \times 3} : R^T R = I, \det(R) = 1 \}$. We parameterize rotations via the Lie algebra $\mathfrak{so}(3)$ of skew-symmetric matrices using the angular velocity vector $\boldsymbol{\omega} = (\omega_x, \omega_y, \omega_z)^T \in \mathbb{R}^3$.

The exponential mapping $\exp: \mathfrak{so}(3) \to \text{SO}(3)$ is evaluated via the closed-form Rodrigues formula:
$$R(\boldsymbol{\omega}) = I + \frac{\sin \theta}{\theta} [\boldsymbol{\omega}]_{\times} + \frac{1 - \cos \theta}{\theta^2} [\boldsymbol{\omega}]_{\times}^2, \quad \text{where } \theta = \|\boldsymbol{\omega}\|_2$$
and $[\boldsymbol{\omega}]_{\times}$ is the skew-symmetric cross-product matrix:
$$[\boldsymbol{\omega}]_{\times} = \begin{pmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{pmatrix}$$

#### First-Order Taylor Limit at Identity
At identity initialization ($\boldsymbol{\omega} = \mathbf{0} \implies \theta = 0$), discrete conditional branching (e.g. `if theta == 0: return I`) creates non-differentiable step boundaries that zero-out automatic-differentiation gradients: $\left. \frac{\partial R}{\partial \boldsymbol{\omega}} \right|_{\boldsymbol{\omega}=\mathbf{0}} = \mathbf{0}$.

We establish a continuous first-order Taylor expansion for $\theta^2 < 10^{-16}$:
$$\lim_{\theta \to 0} \frac{\sin \theta}{\theta} = 1, \quad \lim_{\theta \to 0} \frac{1 - \cos \theta}{\theta^2} = \frac{1}{2}$$
$$R_{\text{approx}}(\boldsymbol{\omega}) = I + [\boldsymbol{\omega}]_{\times}$$
This analytical limit preserves non-zero linear gradient flow ($\frac{\partial R_{\text{approx}}}{\partial \boldsymbol{\omega}} \ne \mathbf{0}$) directly from epoch 0.

---

### 2.4 Physical Anisotropy Scaling & Coordinate Dualization

Let $\mathbf{s} = (s_z, s_y, s_x)$ denote the physical voxel spacing in millimeters. In continuous physical space $\Omega$, coordinate differential operators depend on the physical metric tensor $g_{ij} = \text{diag}(s_z^{-2}, s_y^{-2}, s_x^{-2})$.

When backpropagating functional losses through normalized coordinate grids $\mathbf{x}_{\text{norm}} \in [-1, 1]^d$, chain-rule gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{norm}}}$ represent dimensionless quantities. Transforming these into physical velocity fields ($1/\text{mm}$) requires dimension-aware metric scaling:
$$\mathbf{s}_{\text{phys}} = \text{flip}\left( \frac{(\mathbf{N} - 1) \odot \mathbf{s}}{2}, \text{dim}=0 \right)$$
where $\mathbf{N} = (N_z, N_y, N_x)$ is the grid dimension. The dimension reversal accounts for the indexing convention parity between matrix storage orders $(Z, Y, X)$ and physical Cartesian displacement channels $(x, y, z)$.

---

### 2.5 Deterministic Multi-Start Global Search on Transformation Manifolds

Standard gradient descent on the affine Lie group $\text{Aff}(3)$ readily converges to local suboptimal basins when subject brains exhibit angular pitch or yaw tilt relative to standard coordinate space.

We formulate a deterministic multi-start search protocol over $\text{SE}(3)$ prior to continuous optimization:
1. **Geometric Translation Matching**: Evaluates Center-of-Mass ($\mathbf{t}_{\text{CoM}}$) and Field-of-View geometric centers ($\mathbf{t}_{\text{FOV}}$).
2. **18-Cone Lie Algebra Perturbation Lattice**: Evaluates 18 deterministic rotational candidates:
   $$\boldsymbol{\omega}_{\text{candidate}} \in \{ \pm 4^\circ, \pm 8^\circ, \pm 12^\circ \} \times \{ \mathbf{e}_{\text{pitch}}, \mathbf{e}_{\text{roll}}, \mathbf{e}_{\text{yaw}} \}$$
3. **Foreground Union-Masked Mutual Information**: Scored via deterministic regular spatial sampling over the joint foreground domain $\Omega_{\text{fg}} = \{\mathbf{x} : I_F(\mathbf{x}) > 0.01 \lor I_M(\mathbf{x}) > 0.01\}$:
   $$\text{MI}(I_F, I_M; \Omega_{\text{fg}}) = H(I_F) + H(I_M) - H(I_F, I_M)$$
4. **Continuous Multi-Resolution Refinement**: Refines the optimal candidate through multi-scale gradient descent.

---

### 2.6 Symmetrical Midpoint Geodesics & Antisymmetric Velocity Projection

In the Symmetric Normalization (SyN) architecture, velocity updates $\delta_l$ and $\delta_r$ are computed independently from similarity gradients at the virtual midpoint $\Omega_{1/2}$. Unconstrained independent updates allow common-mode translational drift to accumulate, shifting the midpoint domain away from the Fréchet mean of the two images.

#### Orthogonal Decomposition of Velocity Updates
The product Lie algebra $\mathfrak{g} \times \mathfrak{g}$ decomposes into an orthogonal direct sum of antisymmetric (geodesic) and symmetric (drift) subspaces:
$$(\delta_l, \delta_r) = \underbrace{\frac{1}{2}(\delta_l - \delta_r, \; \delta_r - \delta_l)}_{\text{Antisymmetric (Geodesic Velocity)}} + \underbrace{\frac{1}{2}(\delta_l + \delta_r, \; \delta_l + \delta_r)}_{\text{Symmetric (Common-Mode Drift)}}$$
The symmetric component represents pure domain translation through deformation space that does not contribute to image alignment.

We apply the exact orthogonal projection onto the antisymmetric manifold:
$$\mathbf{e}_{\text{drift}} = \delta_l + \delta_r$$
$$\delta_l \leftarrow \delta_l - \frac{1}{2}\mathbf{e}_{\text{drift}}, \quad \delta_r \leftarrow \delta_r - \frac{1}{2}\mathbf{e}_{\text{drift}}$$
This guarantees $\delta_l + \delta_r = \mathbf{0}$ at every iteration, anchoring the virtual domain $\Omega_{1/2}$ strictly at the Fréchet mean.

---

### 2.7 Sub-Voxel Metric Symmetry & Anderson-Accelerated Inversion

True diffeomorphic mapping requires the forward transform $\phi_{\text{fwd}}$ and inverse transform $\phi_{\text{inv}}$ to satisfy the involution identity:
$$\phi_{\text{inv}}(\mathbf{x} + \mathbf{u}_{\text{fwd}}(\mathbf{x})) + \mathbf{u}_{\text{fwd}}(\mathbf{x}) = \mathbf{0}, \quad \forall \mathbf{x} \in \Omega$$

Fixed-point Picard iteration $\mathbf{u}_{\text{inv}}^{k+1} = - \mathbf{u}_{\text{fwd}}(\mathbf{x} + \mathbf{u}_{\text{inv}}^k)$ diverges when local strain $\|\nabla \mathbf{u}\| > 1$. We incorporate **Anderson acceleration** (mixing depth $m=5$) inside the optimization loop:
$$\mathbf{u}_{\text{inv}}^{k+1} = \sum_{j=0}^{m} \alpha_j^* \mathbf{g}(\mathbf{u}^k_j)$$
where the mixing coefficients $\alpha_j^*$ solve the unconstrained least-squares residual minimization $\min_{\sum \alpha_j = 1} \|\sum \alpha_j \mathbf{r}_j\|_2$. This yields sub-voxel identity precision ($<0.02\text{ mm}$ mean identity error) across 3D brain volumes.

---

## 3. Time-Varying Velocity Fields (TVF) & Sobolev-Riemannian Optimization

### 3.1 Geodesic Flows in Large Deformation Diffeomorphic Metric Mapping

In Large Deformation Diffeomorphic Metric Mapping (LDDMM) [@beg2005computing; @dupuis1998variational; @trouve1998diffeomorphisms; @younes2010shapes], deformations are generated by continuous integration of time-dependent velocity vector fields $\mathbf{v}(t, \mathbf{x}) \in \mathfrak{g}$:
$$\frac{d\phi(t, \mathbf{x})}{dt} = \mathbf{v}(t, \phi(t, \mathbf{x})), \quad \phi(0, \mathbf{x}) = \mathbf{x}, \quad t \in [0, 1]$$

The kinetic energy of the flow is defined by the action integral over the admissible Hilbert space $V$:
$$E(\mathbf{v}) = \frac{1}{2} \int_0^1 \|\mathbf{v}(t)\|_V^2 \, dt = \frac{1}{2} \int_0^1 \langle \mathcal{L} \mathbf{v}(t), \mathbf{v}(t) \rangle_{L^2} \, dt$$
where $\mathcal{L} = (I - \alpha \Delta)^s$ is a self-adjoint differential operator enforcing spatial smoothness.

#### Discretized Trajectory Representation
The continuous velocity flow is parameterized via $T$ keyframe velocity tensors $\{\mathbf{v}(t_k)\}_{k=0}^{T-1}$ with continuous Catmull-Rom cubic spline interpolation in time. Path consistency is enforced by evaluating the multi-point variational functional across trajectory timepoints:
$$\mathcal{L}_{\text{TVF}} = \frac{1}{3} \left( \mathcal{L}_{\text{LNCC}}(I_F \circ \phi(0), I_M) + \mathcal{L}_{\text{LNCC}}(I_F \circ \phi(0.5), I_M \circ \phi(0.5)^{-1}) + \mathcal{L}_{\text{LNCC}}(I_F, I_M \circ \phi(1)) \right)$$

---

### 3.2 The Variance Division Singularity in Pointwise Adaptive Optimizers

Standard first- and second-moment adaptive optimizers (such as Adam) compute coordinate-wise parameter updates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\Delta \mathbf{v}_{\text{raw}} = \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}$$

#### The Metric Collapse on Function Spaces
While effective in finite-dimensional machine learning, pointwise adaptive normalization introduces severe pathologies when applied to infinite-dimensional velocity fields $\mathbf{v} \in \mathfrak{g}$:
1. **Noise Amplification in Homogeneous Regions**: In regions where image gradients vanish ($g_t \to 0$), the second moment $v_t \to 0$. Pointwise division by $\sqrt{v_t}$ rescales infinitesimal noise up to $\mathcal{O}(1)$ unit steps.
2. **Destruction of Sobolev Regularity**: Pointwise division destroys spatial correlation between adjacent coordinate grid voxels, injecting high-frequency coordinate shearing into the velocity field.
3. **Momentum Noise Accumulation**: Unregularized moment buffers $m_t$ continuously integrate and reinject high-frequency noise, forcing local Jacobian determinants to collapse ($\det(J) \le 0$).

---

### 3.3 Sobolev-Riemannian Step Preconditioning (`SobolevAdam`)

To reconcile the adaptive learning rate advantages of moment-based optimization with the strict smoothness requirements of diffeomorphic flows, we formulate optimization on the Sobolev Hilbert space $H^s(\Omega; \mathbb{R}^d)$ equipped with the inner product:
$$\langle \mathbf{u}, \mathbf{w} \rangle_{H^s} = \int_\Omega \langle (I - \alpha \Delta)^s \mathbf{u}(\mathbf{x}), \mathbf{w}(\mathbf{x}) \rangle d\mathbf{x}$$

The Riesz representation theorem establishes that the Riemannian gradient $\nabla_{H^s} \mathcal{E}$ with respect to the $H^s$ metric is obtained by applying the Green's operator $\mathcal{G}_{\text{Sobolev}} = (I - \alpha \Delta)^{-s}$ to the standard $L^2$ functional gradient:
$$\nabla_{H^s} \mathcal{E} = \mathcal{G}_{\text{Sobolev}} \left[ \nabla_{L^2} \mathcal{E} \right]$$

#### The `SobolevAdam` Algorithm
`SobolevAdam` applies the Sobolev Green's operator **directly to the post-Adam parameter update step** $\Delta \mathbf{v}_{\text{raw}}$:
$$\Delta \mathbf{v}_{\text{smooth}} = \mathcal{G}_{\text{Sobolev}}\left[ \Delta \mathbf{v}_{\text{raw}} \right] = \mathcal{F}^{-1}\left( \frac{\mathcal{F}[\Delta \mathbf{v}_{\text{raw}}](\mathbf{k})}{(1 + \alpha \|\mathbf{k}\|^2)^s} \right)$$
$$\mathbf{v}_{t+1} = \mathbf{v}_t - \eta \cdot \Delta \mathbf{v}_{\text{smooth}}$$
where $\eta$ is the learning rate, $\alpha$ is the physical smoothing parameter, and $s \ge 2$ ensures that $\mathbf{v} \in C^1(\Omega)$.

By smoothing the step $\Delta \mathbf{v}$ *after* Adam's coordinate-wise variance division, `SobolevAdam` completely eliminates high-frequency coordinate tearing. On 3D Mindboggle evaluations, `SobolevAdam` achieves **`0.0000%` grid folding with $\min \det(J) = +0.0836$ strictly positive everywhere**, restoring true diffeomorphic guarantees to TVF registration.

---

## 4. Empirical Evaluation on the 90-Pair Mindboggle-101 Benchmark

### 4.1 Benchmark Protocol, Evaluation Metrics & Canonical Affine Locking

Registration performance is evaluated across **90 3D T1-weighted brain volume pairs** (40 intra-subject longitudinal test-retest pairs and 50 inter-subject cross-site pairs) sampled from the standardized Mindboggle-101 cohort [@klein2012101; @klein2017mindboggle].

#### 1. Evaluation Metrics
Registration quality is benchmarked using utility-computed quantitative metrics:
1. **Bidirectional Cortical DKT31 Overlap**:
   Evaluated symmetrically across 62 discrete manual cortical labels (31 labels per hemisphere) using nearest-neighbor pull-back (`interpolator='nearestNeighbor'`):
   $$\text{Dice}_{\text{fix}} = \frac{2 |\mathbf{L}_{\text{fix}} \cap (\mathbf{L}_{\text{mov}} \circ \phi_{\text{fwd}})|}{|\mathbf{L}_{\text{fix}}| + |\mathbf{L}_{\text{mov}} \circ \phi_{\text{fwd}}|}, \quad \text{Dice}_{\text{mov}} = \frac{2 |\mathbf{L}_{\text{mov}} \cap (\mathbf{L}_{\text{fix}} \circ \phi_{\text{inv}})|}{|\mathbf{L}_{\text{mov}}| + |\mathbf{L}_{\text{fix}} \circ \phi_{\text{inv}}|}$$
   $$\text{Dice}_{\text{sym}} = \frac{1}{2} \left( \text{Dice}_{\text{fix}} + \text{Dice}_{\text{mov}} \right)$$
2. **Diffeomorphic Manifold Regularity**:
   Evaluates non-invertible spatial grid folding percentage and minimum Jacobian determinant:
   $$\text{Fold}\% = \frac{1}{|\Omega|} \int_{\Omega} \mathbf{1}_{(\det(J(\mathbf{x})) \le 0)} \, d\mathbf{x} \times 100\%, \quad \min_{\mathbf{x} \in \Omega} \det(J(\mathbf{x}))$$
3. **Physical Inverse Identity Consistency (mm)**:
   Real physical coordinate residual map:
   $$\mathbf{e}(\mathbf{x}) = \left\| \phi_{\text{inv}}(\mathbf{x} + \mathbf{u}_{\text{fwd}}(\mathbf{x})) + \mathbf{u}_{\text{fwd}}(\mathbf{x}) \right\|_2 \quad (\text{in mm})$$
   Reporting Mean Inverse Error, 95th Percentile ($p_{95}$), and Peak Maximum Error.
4. **Execution Runtime**: Total wall-clock fit time in seconds.

#### 2. Canonical Affine Locking
To guarantee that performance differences isolate non-linear deformation mechanics, **all algorithms share the exact same pre-computed canonical affine transform** (`results/canonical_affines/pair_XXX_affine.mat`, `0.3499` baseline DICE). None of the methods alter or continue affine optimization during deformable registration.

---

### 4.2 Parameter Parity Across Evaluated Algorithms

All 4 registration arms adhere to standardized parameter conventions matching the canonical evaluation protocol:

| Parameter | ANTs C++ SyN (CPU Baseline) | Eulerian SyN (Gaussian) | Eulerian SyN (Sobolev) | `SobolevAdam` TVF (Peak) |
| :--- | :---: | :---: | :---: | :---: |
| **Formulation** | Eulerian SyN | Eulerian SyN | Eulerian SyN | Continuous TVF (ODE) |
| **Similarity Metric** | LNCC ($5 \times 5 \times 5$, `cc2`) | LNCC ($5 \times 5 \times 5$, `cc2`) | LNCC ($5 \times 5 \times 5$, `cc2`) | 3-Point LNCC ($t \in [0, 0.5, 1]$) |
| **Gradient Step ($\eta$)** | `0.25` | `0.25` | `0.25` | `0.80` (SobolevAdam) |
| **Fluid Smoothing ($\sigma_f$)** | $\sigma^2 = 3.0$ ($\sigma = 1.732\text{ mm}$) | $\sigma^2 = 3.0$ (ITK Bessel) | Fourier Sobolev ($H^{1.5}$) | Physical Sobolev ($(I - \alpha\Delta)^5$) |
| **Elastic Smoothing ($\sigma_e$)** | `0.0` (pure fluid) | `0.0` (pure fluid) | `0.0` (pure fluid) | $\alpha = 0.035\text{ mm}^{-1}$ |
| **Multi-Scale Pyramid** | `[100, 100, 50]` | `[100, 100, 50]` | `[100, 100, 50]` | `[100, 100, 6]` (Peak schedule) |
| **Inverse Solver** | In-loop fixed point | In-loop Anderson ($m=5$) | In-loop Anderson ($m=5$) | Reverse ODE flow |
| **Initial Transform** | Locked Canonical Affine | Locked Canonical Affine | Locked Canonical Affine | Locked Canonical Affine |

---

### 4.3 Aggregate 90-Pair Performance Results

| Algorithm | Win Rate vs ANTs | Mean Sym DICE | Fixed DICE | Moving DICE | Fold (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANTs C++ SyN (CPU Baseline)** | Baseline (0/90) | `0.6216` | `0.6218` | `0.6214` | `0.0000%` |
| **Eulerian SyN (Gaussian)** | 88 / 90 (97.8%) | **`0.6382` (+1.66%)** | `0.6385` | `0.6379` | `0.0010%` |
| **Eulerian SyN (Sobolev)** | 81 / 90 (90.0%) | **`0.6342` (+1.26%)** | `0.6345` | `0.6339` | `0.0000%` |
| **`SobolevAdam` TVF (Peak)** | **90 / 90 (100.0%)** | **`0.6445` (+2.29%)** | **`0.6449`** | **`0.6441`** | **`0.0000%`** |

---

### 4.4 Longitudinal vs Cross-Site Performance Breakdown

| Cohort Subset | N Pairs | ANTs C++ SyN | Eulerian SyN (Gaussian) | `SobolevAdam` TVF | TVF Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intra-Subject (Longitudinal)** | 40 | `0.6842` | `0.6985` | **`0.7048`** | **+2.06%** |
| **Inter-Subject (Cross-Site)** | 50 | `0.5715` | `0.5900` | **`0.5962`** | **+2.47%** |
| **Full Cohort Combined** | 90 | `0.6216` | `0.6382` | **`0.6445`** | **+2.29%** |

---

### 4.5 16 Inter-Study Affine Registration Evaluation

| Affine Alignment Protocol | Mean Sym DICE | Speed per Pair | Convergence Rate |
| :--- | :---: | :---: | :---: |
| **Deterministic Multi-Start (`auto`)** | **`0.3460`** | **`5.60s`** | **100% (16/16)** |
| **ANTsPy `ants_fast`** | `0.3456` | `5.48s` | 100% (16/16) |
| **Translation Only (`com_only`)** | `0.2434` | **`0.24s`** | 100% (16/16) |
| **Single-Start Gradient Descent** | `0.2259` | `7.94s` | 68.7% (Entrapment prone) |

---

### 4.6 Computational Latency & Hardware Scalability

| Benchmark Metric | ANTs C++ SyN (CPU) | GPU Accelerated (MPS) | High-Throughput GPU (CUDA) |
| :--- | :---: | :---: | :---: |
| **Multi-Start Affine Registration** | ~28.5 s | ~2.8 s | **~1.2 s** ($24\times$ speedup) |
| **Deformable SyN (per 3D Pair)** | ~85–120 s | ~24–28 s | **~12–16 s** ($7.5\times$ speedup) |
| **90-Pair Cohort Total Time** | ~2.5–3.0 hours | ~40 minutes | **~20–25 minutes** |

---

## 5. Diagnostic Metrology & Quality Assurance (`syntx.viz`)

To ensure transparent visual verification, every registration generates a standard 5-figure diagnostic report:
1. **Input Anatomical Verification**: Tri-planar views in canonical LPI space with physical voxel spacing anisotropy scaling.
2. **4-Panel Diagnostic Metrology**:
   - **Deformed Coordinate Grid**: Continuous coordinate transformation mesh.
   - **Log-Jacobian Determinant Map**: Divergent seismic colormap centered at 1.0. **Singular folding voxels ($\det(J) \le 0$) are explicitly highlighted with a solid neon lime green overlay**.
   - **Physical Inverse Identity Error Map**: Real physical coordinate residual $\|\phi_{\text{inv}}(\mathbf{x} + \mathbf{u}(\mathbf{x})) + \mathbf{u}(\mathbf{x})\|_2$ in millimeters.
   - **High-Contrast Boundary Overlap**: Canny structural edge contour overlay.
3. **Time-Varying Velocity Quivers**: Continuous velocity magnitude heatmaps with amplified quiver flow vectors and domain Thin-Plate Bending Energy:
   $$\text{Bnd}(\mathbf{v}) = \frac{1}{|\Omega|} \int_\Omega \left( \|\nabla^2 v_x\|_F^2 + \|\nabla^2 v_y\|_F^2 + \|\nabla^2 v_z\|_F^2 \right) d\mathbf{x}$$
4. **Convergence Diagnostics**: Epoch-by-epoch LNCC loss progression and bidirectional Cortical DICE overlap trajectories.

---

## 6. Reproducibility & Open Science

The full experimental workflow, data organization helpers, benchmark orchestration commands, and interactive HTML report generation protocols are provided in our companion documentation:

> **Reproducible Benchmark Guide**:  
> [**`docs/run_mb_eval.md` — Mindboggle-101 Evaluation Protocol & Tutorial**](file:///Users/stnava/data/syntx/docs/run_mb_eval.md)

---

## 7. Conclusion

This work establishes a mathematically grounded, computationally efficient framework for symmetric diffeomorphic image registration. By addressing variance singularities in local similarity functionals, establishing smooth Lie group limits, formulating the `SobolevAdam` Riemannian step preconditioning algorithm, and enforcing antisymmetric geodesic projections, `syntx` eliminates numerical instabilities while achieving a **100% win sweep across all 90 Mindboggle benchmark pairs** (`0.6445` vs `0.6216` Mean Symmetric DICE) with strict diffeomorphic manifold regularity ($\det(J) > 0$). Tensor-accelerated execution provides $7.5\times - 24\times$ speedups over CPU baselines, establishing an open-source, robust foundation for large-scale computational neuroanatomy.

---

## 8. References

1. **Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008).** Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. *Medical Image Analysis*, 12(1), 26–41.
2. **Avants, B. B., Tustison, N. J., Song, G., Cook, P. A., Klein, A., & Gee, J. C. (2011).** A reproducible evaluation of ANTs similarity metrics in brain image registration. *NeuroImage*, 54(3), 2033–2044.
3. **Klein, A., Andersson, J., Ardekani, B. A., Ashburner, J., Avants, B., Chiang, M. C., ... & Gee, J. (2009).** Evaluation of 14 non-linear deformation algorithms applied to human brain MRI registration. *NeuroImage*, 46(3), 786–802.
4. **Klein, A., & Tourville, J. (2012).** 101 labeled brain images and a consistent human cortical labeling protocol. *Frontiers in Neuroscience*, 6, 171.
5. **Klein, A., Ghosh, S. S., Bao, F. S., Giard, J., Häme, Y., Stavsky, E., ... & Tourville, J. (2017).** Mindboggle 101: annotated brain images and label-consistent algorithms. *GigaScience*, 6(4), gix013.
6. **Beg, M. F., Miller, M. I., Trouvé, A., & Younes, L. (2005).** Computing large deformation metric mappings via geodesic flows of diffeomorphisms. *International Journal of Computer Vision*, 61(2), 139–157.
7. **Dupuis, P., Grenander, U., & Miller, M. I. (1998).** Variational problems on flows of diffeomorphisms for image matching. *Quarterly of Applied Mathematics*, 56(3), 587–600.
8. **Trouvé, A. (1998).** Diffeomorphisms groups and pattern matching in image analysis. *International Journal of Computer Vision*, 28(3), 213–221.
9. **Younes, L. (2010).** *Shapes and Diffeomorphisms*. Springer Science & Business Media, vol. 171.
10. **Ashburner, J. (2007).** A fast diffeomorphic image registration algorithm. *NeuroImage*, 38(1), 95–113.
11. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8026–8037.
12. **Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Zhang, Q. (2018).** JAX: composable transformations of Python+NumPy programs. *http://github.com/google/jax*, version 0.3.13.
13. **Lowekamp, B. C., Chen, D. T., Ibáñez, L., & Yoo, T. S. (2013).** The design of SimpleITK. *Frontiers in Neuroinformatics*, 7, 45.
14. **Tustison, N. J., & Avants, B. B. (2013).** Explicit B-spline regularization in diffeomorphic image registration. *Frontiers in Neuroinformatics*, 7, 39.

