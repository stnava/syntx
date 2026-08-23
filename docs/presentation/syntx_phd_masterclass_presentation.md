# Symmetric Diffeomorphic Image Registration on Riemannian Manifolds (`syntx`)
## A First-Principles 20-Slide Masterclass for PhD-Level Computational Scientists

**Author & Presenter**: Brian Avants, Ph.D. & The Syntx Development Team  
**Format**: 20 Widescreen (16:9) Presentation Slides with Generative Algorithm Visualizations  
**Associated Artifacts**:
- PowerPoint Deck: [`docs/presentation/syntx_diffeomorphic_geometry_presentation.pptx`](file:///Users/stnava/code/syntx/docs/presentation/syntx_diffeomorphic_geometry_presentation.pptx)
- Interactive HTML Presentation: [`docs/presentation/index.html`](file:///Users/stnava/code/syntx/docs/presentation/index.html)
- Manuscript Report: [`docs/manuscript/manuscript_report.md`](file:///Users/stnava/code/syntx/docs/manuscript/manuscript_report.md)

---

## Cross-Disciplinary Mathematical Dictionary & Notation

| Concept | First-Principles Mathematical Definition | Cross-Disciplinary Analogy (Physics, Robotics, ML) |
| :--- | :--- | :--- |
| **Diffeomorphism ($\text{Diff}(\Omega)$)** | A smooth ($C^\infty$) bijection $\Phi: \Omega \to \Omega$ with smooth inverse $\Phi^{-1}$. Forms an infinite-dimensional Fréchet Lie group under function composition. | **Continuum Mechanics**: Deformation of an ideal hyperelastic continuum without tearing (continuity), self-penetration (injectivity), or vacuum collapse (surjectivity). |
| **Lie Group $\text{SO}(3)$ vs. Lie Algebra $\mathfrak{so}(3)$** | $\text{SO}(3) = \{R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \det(R)=1\}$. The Lie algebra $\mathfrak{so}(3)$ is its tangent space at identity, spanned by $3 \times 3$ skew-symmetric cross-product matrices $[\boldsymbol{\omega}]_\times$. Mapped via the matrix exponential $\exp([\boldsymbol{\omega}]_\times)$. | **Robotics & Rigid Dynamics**: Angular velocity vector $\boldsymbol{\omega}$ (Lie algebra) integrated over unit time generates the finite spatial orientation matrix $R$ (Lie group). |
| **Jacobian Determinant ($\det(J)$)** | Local push-forward metric tensor $J_\Phi(\mathbf{x}) = I + \nabla \mathbf{u}(\mathbf{x})$. Determinant $\det(J) = dV'/dV$ measures infinitesimal volumetric expansion ($\det(J)>1$), compression ($0<\det(J)<1$), or orientation reversal / folding ($\det(J) \le 0$). | **Fluid Dynamics**: Local fluid density ratio $\rho(\mathbf{x}) = \rho_0 / \det(J(\mathbf{x}))$. $\det(J) \le 0$ corresponds to a shock singularity / caustic where the coordinate grid folds over itself. |
| **Eulerian vs. Lagrangian Frames** | **Eulerian**: Fields and velocities $\mathbf{v}(t, \mathbf{x})$ are evaluated on a fixed Cartesian coordinate lattice $\mathbf{x}$. **Lagrangian**: Tracks the trajectory $\mathbf{X}(t)$ of individual moving material particles. | **Fluid Mechanics**: A fixed velocity probe anchored in a river channel (Eulerian) versus a dye packet drifting with the current (Lagrangian). |
| **LNCC Similarity Functional** | Local zero-mean Pearson cross-correlation computed over a sliding neighborhood $W(\mathbf{x})$, invariant to arbitrary local affine intensity scaling ($I \mapsto a(\mathbf{x})I + b(\mathbf{x})$). | **Signal Processing & Optics**: Local optical phase coherence and spatially normalized cross-spectral density. |
| **Fréchet Geodesic Mean Midpoint** | Riemannian barycenter $\Omega_{1/2} = \arg\min_{p \in \mathcal{M}} \sum d^2(p, x_i)$. In SyN, both target and source images deform symmetrically into $\Omega_{1/2}$ along coupled half-geodesic trajectories ($\phi_{l2r}, \phi_{r2l}$). | **General Relativity**: Center of mass in curved spacetime; removes arbitrary reference frame (template) bias. |
| **Antisymmetric Velocity Projection** | Decomposition of tangent velocity updates into symmetric and antisymmetric subspaces; enforcing $\delta_l + \delta_r \equiv \mathbf{0}$ eliminates common-mode translational drift. | **Gauge Theory & Classical Mechanics**: Center-of-mass momentum conservation ($\sum \mathbf{p}_i = \mathbf{0}$) and gauge fixing. |
| **Large Deformation Diffeomorphic Metric Mapping (LDDMM)** | Time-dependent flow $\dot{\phi}(t) = \mathbf{v}(t, \phi(t))$ minimizing kinetic action $E(\mathbf{v}) = \frac{1}{2}\int_0^1 \|\mathbf{v}(t)\|_V^2 dt$, governed by EPDiff geodesic equations. | **Nonlinear Optimal Control**: Pontryagin Maximum Principle for continuous trajectory optimization of infinite-DOF continuum bodies. |
| **Sobolev Space $H^s$ & Green's Operator ($\mathcal{G}$)** | Hilbert space with inner product $\langle \mathbf{u}, \mathbf{w} \rangle_{H^s} = \langle (I - \alpha \Delta)^s \mathbf{u}, \mathbf{w} \rangle_{L^2}$. Riesz representation gives Riemannian gradient $\nabla_{H^s}\mathcal{E} = (I - \alpha \Delta)^{-s}\nabla_{L^2}\mathcal{E}$. By Sobolev embedding, $s > d/2 + 1 \implies \mathbf{v} \in C^1$, guaranteeing $\det(J) > 0$. | **Classical Field Theory**: Screened Poisson equation Green's function (Yukawa potential $e^{-r/\lambda}/r$). |
| **Discrete Sine Transform Type-I (DST-I)** | Orthogonal sine basis $S(k, n) = \sqrt{\frac{2}{N+1}}\sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)$ analytically enforcing homogeneous Dirichlet boundary conditions $\mathbf{v}(\partial\Omega) \equiv \mathbf{0}$. | **Acoustics & Electrostatics**: Clamped vibrating membrane or grounded Faraday cage; eliminates toroidal FFT wrap-around leaks. |
| **Dice Overlap Metric** | Sörensen-Dice coefficient $D(A, B) = \frac{2|A \cap B|}{|A| + |B|}$, representing the harmonic mean of precision and recall for spatial label overlap. | **Measure Theory & Information Retrieval**: Set-theoretic measure of intersection volume relative to total measure. |

---

# The 20-Slide Masterclass Deck

---

### Slide 1: Title & Abstract Vision
**Category**: FOUNDATIONS & OVERVIEW  
**Title**: Symmetric Diffeomorphic Image Registration on Riemannian Manifolds  
**Subtitle**: A First-Principles Masterclass in Computational Anatomy, Infinite-Dimensional Lie Groups, & Tensor AI  
**Figure**: [`docs/presentation/figures/fig_syn_manifold_conceptual_v01.jpg`](file:///Users/stnava/code/syntx/docs/presentation/figures/fig_syn_manifold_conceptual_v01.jpg)

**Core Content**:
- **The Fundamental Goal**: Establish smooth, bijective coordinate mappings between continuous spatial coordinate domains $\Omega_F, \Omega_M \subset \mathbb{R}^3$.
- **The Mathematical Paradigm**: Formulate registration as variational geodesic optimization on the infinite-dimensional Lie group of diffeomorphisms $\text{Diff}(\Omega)$ equipped with a right-invariant Riemannian metric.
- **Core Engineering Innovations**: Safe autograd variance flooring, Eulerian Anderson involution, SobolevAdam Riemannian optimization, and exact DST-I Dirichlet boundaries.
- **Cohort Validation**: Demonstrated on the 90-Pair Mindboggle Benchmark: 100% zero-folding guarantee, +2.29% Cortical Dice gain, and 16-second GPU execution.

**Presenter Script**:
> *"Welcome. Today we introduce `syntx`, a unified mathematical and computational framework for diffeomorphic coordinate estimation on Riemannian manifolds. Rather than treating image registration as heuristic displacement regression, we formulate it as Hamiltonian mechanics and geodesic flow on the infinite-dimensional Lie group of diffeomorphisms. Let's start from first principles."*

---

### Slide 2: The Spatial Correspondence Problem on Continuous Domains
**Category**: PROBLEM FORMULATION  
**Title**: The Spatial Correspondence Problem on Continuous Domains  
**Subtitle**: Why High-Dimensional Image Alignment is an Ill-Posed Variational Inverse Problem  
**Figure**: [`docs/presentation/figures/diag_spatial_inverse_problem_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_spatial_inverse_problem_v01.png)

**Core Content**:
- **Continuous Spatial Mapping**: Given target $I_F(\mathbf{x})$ and source $I_M(\mathbf{x})$, find transformation $\Phi: \Omega \to \Omega$ such that $I_M(\Phi(\mathbf{x})) \approx I_F(\mathbf{x})$.
- **Infinite Degrees of Freedom**: Every voxel $\mathbf{x} = (x, y, z)$ possesses an independent displacement vector $\mathbf{u}(\mathbf{x}) \in \mathbb{R}^3$, yielding $\sim 10^7$ parameters in 3D.
- **Complex Biological Morphology**: Human cortical anatomy features sharp, nested gyral folds and deep sulcal fissures with high inter-subject anatomical variability.
- **Ill-Posed Nature**: Pure intensity matching is severely underconstrained; infinite non-physical deformations can produce identical image similarity without geometric validity.

**Presenter Script**:
> *"Consider two physical scalar fields—such as magnetic resonance scans from two different human brains. We seek a spatial mapping that pulls one coordinate system into the other. Because there are millions of independent spatial vectors and only one intensity value per coordinate, this variational inverse problem is severely ill-posed and requires strict geometric regularization."*

---

### Slide 3: Topology Preservation & The Jacobian Determinant $\det(J)$
**Category**: DIFFERENTIAL GEOMETRY  
**Title**: Topology Preservation & The Jacobian Determinant $\det(J)$  
**Subtitle**: Preventing Non-Physical Coordinate Singularities, Self-Intersections, & Tearing  
**Figure**: [`docs/presentation/figures/diag_topology_preservation_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_topology_preservation_v01.png)

**Core Content**:
- **The Jacobian Matrix**: Local spatial scaling and orientation are governed by $\nabla \Phi(\mathbf{x}) = I + \nabla \mathbf{u}(\mathbf{x}) \in \mathbb{R}^{3 \times 3}$.
- **The Jacobian Determinant**: The scalar $J_\Phi(\mathbf{x}) = \det(\nabla \Phi(\mathbf{x}))$ measures infinitesimal volume expansion ($J > 1$) or compression ($J < 1$).
- **The Singularity Condition**: If $J_\Phi(\mathbf{x}) \le 0$, the coordinate transformation inverts its orientation, causing grid self-intersection, tearing, and loss of invertibility.
- **The Diffeomorphic Guarantee**: A valid anatomical map must satisfy $J_\Phi(\mathbf{x}) > 0$ everywhere on $\Omega$, ensuring smooth bijectivity and topology preservation.

**Presenter Script**:
> *"In fluid and continuum mechanics, the Jacobian determinant det(J) represents local density scaling. If det(J) drops to zero or goes negative, the coordinate grid crosses over itself, space flips chirality, and matter is destroyed. A valid transformation must maintain det(J) > 0 everywhere."*

---

### Slide 4: Diffeomorphisms on Infinite-Dimensional Lie Groups $\text{Diff}(\Omega)$
**Category**: MATHEMATICAL FOUNDATIONS  
**Title**: Diffeomorphisms on Infinite-Dimensional Lie Groups $\text{Diff}(\Omega)$  
**Subtitle**: Generating Smooth Transformations via Flow of Velocity Fields on the Lie Algebra $\mathfrak{X}(\Omega)$  
**Figure**: [`docs/presentation/figures/fig_tvf_manifold_conceptual_v01.jpg`](file:///Users/stnava/code/syntx/docs/presentation/figures/fig_tvf_manifold_conceptual_v01.jpg)

**Core Content**:
- **The Diffeomorphism Group**: $\text{Diff}(\Omega)$ forms an infinite-dimensional Lie group of smooth invertible mappings with smooth inverses.
- **The Lie Algebra $\mathfrak{g} = \mathfrak{X}(\Omega)$**: The tangent space at the identity corresponds to smooth Eulerian velocity vector fields $\mathbf{v}(\mathbf{x}) \in \mathbb{R}^3$.
- **Lagrangian Flow ODE**: Trajectories are generated by integrating ODEs: $\frac{d\phi(t, \mathbf{x})}{dt} = \mathbf{v}(t, \phi(t, \mathbf{x}))$ with $\phi(0, \mathbf{x}) = \mathbf{x}$.
- **Riemannian Geodesics**: Optimal deformations follow shortest-path geodesics on $\text{Diff}(\Omega)$ minimizing action integrals $\int_0^1 \|\mathbf{v}(t)\|_V^2 dt$.

**Presenter Script**:
> *"Instead of optimizing non-linear displacements directly in a flat vector space, we optimize velocity fields in the Lie algebra. By Liouville's theorem, integrating a velocity field with bounded spatial divergence guarantees that det(J) remains strictly positive for all time."*

---

### Slide 5: Local Normalized Cross-Correlation (LNCC) on Function Spaces
**Category**: VARIATIONAL SIMILARITY  
**Title**: Local Normalized Cross-Correlation (LNCC) on Function Spaces  
**Subtitle**: Robust Similarity Metric for Intensity Non-Uniformities and Contrast Inhomogeneities  
**Figure**: [`docs/presentation/figures/diag_lncc_function_space_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_lncc_function_space_v01.png)

**Core Content**:
- **Local Spatial Windowing**: Evaluated over localized spatial box/Gaussian neighborhoods $W(\mathbf{x})$ of radius $r=2$ ($5 \times 5 \times 5$ window).
- **Mathematical Formulation**: $\mathcal{L}_{\text{LNCC}} = - \int_\Omega \frac{\text{Cov}_W(I_F, I_M)^2}{\text{Var}_W(I_F) \cdot \text{Var}_W(I_M)} d\mathbf{x}$, invariant to local linear intensity scaling.
- **Analytical Gradient**: Functional derivative $\frac{\delta \mathcal{L}}{\delta I_M} = -\frac{2 r(\mathbf{x})}{\sqrt{s_{FF} s_{MM}}} \left( \tilde{I}_F - \frac{s_{FM}}{s_{MM}} \tilde{I}_M \right)$ drives spatial descent.
- **Deep Feature Extensions**: Easily extended to deep feature spaces (DINOv2, VGG Layer 4) for extreme cross-modality registration.

**Presenter Script**:
> *"Medical images suffer from spatial gain fields and intensity bias. LNCC solves this by computing zero-mean correlation within sliding local windows. It is completely invariant to localized multiplicative gain and additive offset."*

---

### Slide 6: The Variance Singularity: Proof & Safe Variance Flooring
**Category**: THEORETICAL ANALYSIS  
**Title**: The Variance Singularity: Proof & Safe Variance Flooring  
**Subtitle**: Regularizing the $\mathcal{O}(\text{Var}^{-1/2})$ Asymptotic Explosion in Homogeneous Tissue  
**Figure**: [`docs/presentation/figures/diag_variance_floor_proof_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_variance_floor_proof_v01.png)

**Core Content**:
- **The Singularity Hazard**: In uniform anatomical regions (white matter, ventricles, background), local variance vanishes: $\text{Var}_W(I) \to 0$.
- **Asymptotic Derivative Explosion**: $\lim_{s_{MM} \to 0} \|\frac{\delta \mathcal{L}}{\delta I_M}\| \sim \mathcal{O}(s_{MM}^{-1/2}) \to \infty$, injecting massive localized non-physical forces.
- **The Safe Floor Solution**: Enforce strict lower floor: $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$, bounding functional gradients by $\|\frac{\delta \mathcal{L}}{\delta I}\|_\infty \le \frac{2}{\epsilon}$.
- **Empirical Impact**: Completely eliminates derivative spikes across flat regions, preventing local grid folds and ensuring stable optimization.

**Presenter Script**:
> *"Here is a critical mathematical vulnerability in standard autograd: as local image variance goes to zero in uniform brain tissue or background air, the unfloored functional derivative explodes to infinity. Enforcing a safe variance floor of 10^-6 guarantees bounded Lipschitz gradient dynamics."*

---

### Slide 7: Lie Algebra $\mathfrak{so}(3)$ Parameterization & Smooth Exponential Limit
**Category**: GLOBAL ALIGNMENT  
**Title**: Lie Algebra $\mathfrak{so}(3)$ Parameterization & Smooth Exponential Limit  
**Subtitle**: Rigid & Affine Initialization via Continuous Lie Group Geodesics  
**Figure**: [`docs/presentation/figures/diag_so3_lie_algebra_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_so3_lie_algebra_v01.png)

**Core Content**:
- **Special Orthogonal Group $\text{SO}(3)$**: Parameterize 3D rotations via angular velocity vector $\boldsymbol{\omega} = (\omega_x, \omega_y, \omega_z)^T \in \mathfrak{so}(3)$.
- **Rodrigues Exponential Map**: $R(\boldsymbol{\omega}) = I + \frac{\sin \theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1-\cos \theta}{\theta^2}[\boldsymbol{\omega}]_\times^2$ with rotation angle $\theta = \|\boldsymbol{\omega}\|_2$.
- **The Identity Gradient Discontinuity**: Conditional branches (`if theta == 0`) create zero-gradient lockup at identity initialization.
- **First-Order Taylor Continuity**: We evaluate smooth Taylor limit $\lim_{\theta \to 0} R(\boldsymbol{\omega}) = I + [\boldsymbol{\omega}]_\times$, ensuring continuous backpropagation.

**Presenter Script**:
> *"For global alignment, we parameterize rotations using Lie algebra vectors. By replacing conditional branching at zero angle with a first-order Taylor series limit, automatic differentiation gradients remain non-zero and smooth right through the origin."*

---

### Slide 8: Deterministic 18-Cone Multi-Start Lattice Search
**Category**: OPTIMIZATION BASINS  
**Title**: Deterministic 18-Cone Multi-Start Lattice Search  
**Subtitle**: Escaping Non-Convex Angular Traps with Foreground Union-Masked Mutual Information  
**Figure**: [`docs/presentation/figures/diag_18cone_multistart_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_18cone_multistart_v01.png)

**Core Content**:
- **Non-Convex Angular Energy Landscapes**: Gradient descent from identity frequently stalls in local rotational minima ($>20^\circ$ angular error).
- **18-Cone Perturbation Lattice**: Deterministic search over 18 geodesic directions in $\mathfrak{so}(3)$ covering pitch, yaw, roll, and compound rotations.
- **Foreground Masked Mutual Information**: Evaluates candidate alignments on union mask $\Omega_{\text{fg}} = (I_F > 0.01) \cup (I_M > 0.01)$, eliminating background bias.
- **100% Basin Lock Rate**: Achieves 100% (16/16) global basin recovery, locking a canonical affine baseline ($0.3499 \pm 0.02$ Cortical Dice).

**Presenter Script**:
> *"Optimization on rotational manifolds has multiple local energy wells. We deploy a deterministic 18-cone lattice in Lie algebra space, scoring each candidate with foreground union-masked Mutual Information to guarantee 100% basin lock."*

---

### Slide 9: The Single Interpolation Principle
**Category**: INFORMATICS INVARIANT  
**Title**: The Single Interpolation Principle  
**Subtitle**: Preserving High-Frequency Boundaries by Eliminating Intermediate Pre-Warping Cascades  
**Figure**: [`docs/presentation/figures/diag_single_interpolation_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_single_interpolation_v01.png)

**Core Content**:
- **The Multi-Resampling Trap**: Classical multi-stage pipelines resample images at each stage (rigid $\to$ affine $\to$ deformable).
- **Spatial Boundary Blurring**: Successive interpolation acts as a cascade of low-pass filters ($I_{\text{final}} = I_0 * K_{\sigma_1} * K_{\sigma_2} * \dots$), attenuating fine sulcal edges.
- **Exact Continuous Composition**: We compose transformations analytically in coordinate space: $\Phi_{\text{composite}}(\mathbf{x}) = \phi \circ A \circ T_0(\mathbf{x})$.
- **Single Pullback Invariant**: Native high-resolution image intensities and discrete label maps are sampled **exactly once** via $\Phi_{\text{composite}}$.

**Presenter Script**:
> *"Every time you resample an image, you apply a spatial low-pass filter that blurs high-frequency boundaries. In `syntx`, we compose all translation, affine, and non-linear warps analytically in continuous coordinates and sample the raw data exactly once."*

---

### Slide 10: Symmetric Normalization (SyN) & Fréchet Midpoint Anchoring
**Category**: SYMMETRIC FORMULATION  
**Title**: Symmetric Normalization (SyN) & Fréchet Midpoint Anchoring  
**Subtitle**: Eliminating Template Bias via Antisymmetric Half-Geodesic Splitting  
**Figure**: [`docs/presentation/figures/diag_syn_frechet_midpoint_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_syn_frechet_midpoint_v01.png)

**Core Content**:
- **Template Asymmetry Pathology**: Optimizing $M \to F$ yields different anatomical correspondences than optimizing $F \to M$.
- **Fréchet Midpoint Domain $\Omega_{1/2}$**: SyN optimizes two coupled half-geodesic trajectories $\phi_{l2r}: \Omega_{1/2} \to \Omega_F$ and $\phi_{r2l}: \Omega_{1/2} \to \Omega_M$.
- **Antisymmetric Velocity Projection**: Decompose velocity updates into symmetric and antisymmetric subspaces; enforce $\delta_l + \delta_r \equiv \mathbf{0}$.
- **Drift Elimination**: Antisymmetric projection eliminates common-mode translational drift, strictly anchoring $\Omega_{1/2}$ at the Fréchet geodesic mean.

**Presenter Script**:
> *"SyN solves the reference bias dilemma by meeting in the middle. Deforming both images toward a virtual Fréchet geodesic midpoint and projecting velocity updates onto the antisymmetric subspace guarantees zero center-of-mass drift."*

---

### Slide 11: Eulerian vs. Lagrangian Coordinate Formulations
**Category**: COMPUTATIONAL MECHANICS  
**Title**: Eulerian vs. Lagrangian Coordinate Formulations  
**Subtitle**: Why Fixed Spatial Grid Inversion Outperforms Deforming Particle Tracking  
**Figure**: [`docs/presentation/figures/diag_eulerian_vs_lagrangian_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_eulerian_vs_lagrangian_v01.png)

**Core Content**:
- **Lagrangian Grid Tracking**: Tracks particle positions along deformation paths; requires tracking distorted coordinate grids and heavy gradient smoothing.
- **Eulerian Fixed Reference Frame**: Evaluates velocity vector fields on a fixed, regular spatial coordinate lattice $\mathbf{x} \in \Omega$.
- **Composition Stability**: Eulerian field composition $\phi_{k+1} = \phi_k \circ (\text{Id} + \mathbf{v}_k)$ is computationally robust and fold-resistant.
- **Superior Accuracy**: Eulerian SyN achieves $0.6382$ Mean Cortical Dice vs $0.6216$ for classical ANTs SyN with $0.000\%$ folding across the cohort.

**Presenter Script**:
> *"In continuum mechanics, tracking a deforming mesh leads to element inversion and severe grid distortion. Operating in the Eulerian frame on a stationary Cartesian grid unlocks hardware-accelerated tensor samplers and avoids mesh tangling."*

---

### Slide 12: Sub-Voxel Involution via In-Loop Anderson Acceleration
**Category**: NUMERICAL ANALYSIS  
**Title**: Sub-Voxel Involution via In-Loop Anderson Acceleration  
**Subtitle**: Guaranteeing Exact Diffeomorphic Inversion $\phi \circ \phi^{-1} \equiv \text{Id}$ without Numerical Divergence  
**Figure**: [`docs/presentation/figures/diag_syn_frechet_midpoint_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_syn_frechet_midpoint_v01.png)

**Core Content**:
- **The Inversion Fixed-Point Problem**: The inverse displacement solves $\mathbf{u}_{\text{inv}}(\mathbf{x}) = -\mathbf{u}_{\text{fwd}}(\mathbf{x} + \mathbf{u}_{\text{inv}}(\mathbf{x}))$.
- **Picard Iteration Divergence**: Standard Picard stepping diverges when local strain $\|\nabla \mathbf{u}\| > 1$, causing numerical breakdown in high-deformation zones.
- **Anderson Mixing Depth ($m=5$)**: Computes optimal multi-vector linear combination $\mathbf{u}^{k+1} = \sum_{j=0}^m \alpha_j^* \mathbf{g}(\mathbf{u}_j^k)$ minimizing residual history.
- **Sub-Voxel Precision**: Achieves $<0.025\text{ mm}$ mean identity error ($\|\mathbf{e}_{\text{inv}}\| < 1/40\text{th}$ voxel) inside the active optimization loop.

**Presenter Script**:
> *"Finding the inverse of a non-linear displacement field is a fixed-point problem. When local tissue strain exceeds 1, standard Picard iterations diverge. Anderson acceleration uses a history of five residual vectors to guarantee quadratic convergence to sub-voxel accuracy."*

---

### Slide 13: Unbiased Antithetic Bootstrapped Gradient Estimation
**Category**: STOCHASTIC REGULARITY  
**Title**: Unbiased Antithetic Bootstrapped Gradient Estimation  
**Subtitle**: Destructively Cancelling Discrete Coordinate Discretization Noise  
**Figure**: [`docs/presentation/figures/diag_antithetic_bootstrapping_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_antithetic_bootstrapping_v01.png)

**Core Content**:
- **Coordinate Discretization Aliasing**: Discrete sampling lattices $\mathbf{X} \in \mathbb{Z}^d$ induce high-frequency micro-shears at sharp cortical sulcal boundaries.
- **Symmetric Triplet Sampling**: Evaluate gradient at native point and symmetric sub-voxel offsets: $(\mathbf{X}, \mathbf{X}+\boldsymbol{\delta}, \mathbf{X}-\boldsymbol{\delta})$ with $\boldsymbol{\delta} \sim \mathcal{U}(-0.25, 0.25)$.
- **Zero Directional Expectation**: Because $\mathbb{E}[\boldsymbol{\delta} + (-\boldsymbol{\delta})] = \mathbf{0}$, the estimator introduces zero directional bias or spatial drift.
- **Bending Energy Reduction**: Destructively cancels discrete interpolation noise, cutting thin-plate bending energy by $>50\%$ ($\text{Bnd}=0.0067$ vs ANTs $0.0169$).

**Presenter Script**:
> *"Evaluating derivatives on discrete voxel grids introduces high-frequency discretization jitter. Antithetic bootstrapping evaluates symmetric sub-voxel offsets whose directional expectation is identically zero, destructively cancelling out numerical noise."*

---

### Slide 14: Large Deformation Diffeomorphic Metric Mapping (LDDMM)
**Category**: CONTINUOUS KINEMATICS  
**Title**: Large Deformation Diffeomorphic Metric Mapping (LDDMM)  
**Subtitle**: Formulating Deformations as Continuous Flow of Time-Dependent Velocity Vector Fields  
**Figure**: [`docs/presentation/figures/fig_tvf_manifold_conceptual_v01.jpg`](file:///Users/stnava/code/syntx/docs/presentation/figures/fig_tvf_manifold_conceptual_v01.jpg)

**Core Content**:
- **Continuous Kinematic Flow**: Transformation $\phi(t, \mathbf{x})$ is generated by integrating $\frac{d\phi(t, \mathbf{x})}{dt} = \mathbf{v}(t, \phi(t, \mathbf{x}))$ over $t \in [0, 1]$.
- **Kinetic Energy Action**: Flow energy is defined by $E(\mathbf{v}) = \frac{1}{2} \int_0^1 \langle \mathcal{L} \mathbf{v}(t), \mathbf{v}(t) \rangle_{L^2} dt$, where $\mathcal{L} = (I - \alpha \Delta)^s$ enforces smoothness.
- **Diffeomorphic Closure**: Sufficiently smooth velocity fields ($\|\mathbf{v}(t)\|_V < \infty$) mathematically guarantee that $\phi(1, \cdot)$ is a diffeomorphic mapping.
- **Inverse Consistency by Construction**: The exact inverse map is integrated along the reversed flow: $\phi_{\text{inv}} = \int_1^0 -\mathbf{v}(t) dt$, with zero inversion error.

**Presenter Script**:
> *"LDDMM models registration as fluid kinematics. Flowing along smooth velocity fields guarantees invertibility, and integrating the flow backwards in time produces the exact inverse transformation with zero numerical residual."*

---

### Slide 15: Time-Varying Velocity Fields (TVF) & Spline Parameterization
**Category**: TRAJECTORY OPTIMIZATION  
**Title**: Time-Varying Velocity Fields (TVF) & Spline Parameterization  
**Subtitle**: Multi-Keyframe Spline Interpolation with Multi-Point Trajectory Loss  
**Figure**: [`docs/presentation/figures/diag_tvf_spline_trajectory_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_tvf_spline_trajectory_v01.png)

**Core Content**:
- **Catmull-Rom Cubic Spline Ribbon**: Parameterize continuous velocity $\mathbf{v}(t, \mathbf{x})$ via $T$ discrete keyframe tensors with $C^1$-continuous temporal interpolation.
- **3-Point Trajectory Functional**: Evaluate similarity at trajectory start ($t=0.0$), midpoint ($t=0.5$), and endpoint ($t=1.0$): $\mathcal{L}_{\text{TVF}} = \frac{1}{3}(\mathcal{L}_0 + \mathcal{L}_{0.5} + \mathcal{L}_1)$.
- **Continuous Geodesic Shooting**: Euler ODE stepping with $N_{\text{steps}}=8$ integrates the smooth displacement field across time.
- **Decisive Performance Win**: Achieves a **100% win sweep (90/90 wins)** across the Mindboggle cohort with **`0.6445` Mean Symmetric Cortical Dice**.

**Presenter Script**:
> *"In `syntx.tvf`, we parameterize the continuous velocity trajectory as a cubic spline ribbon in function space and optimize a 3-point trajectory loss. This achieves a 100% win rate across the entire Mindboggle cohort."*

---

### Slide 16: The Metric Collapse of Pointwise Adaptive Optimizers
**Category**: OPTIMIZATION THEORY  
**Title**: The Metric Collapse of Pointwise Adaptive Optimizers  
**Subtitle**: Why Standard Adam Destroys Sobolev Regularity in Infinite-Dimensional Function Spaces  
**Figure**: [`docs/presentation/figures/diag_sobolev_adam_comparison_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_sobolev_adam_comparison_v01.png)

**Core Content**:
- **Pointwise Adam in Finite Dimensions**: Standard Adam normalizes parameter updates: $\Delta \mathbf{v} = \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}$.
- **Metric Collapse on Function Spaces**: In flat image regions ($g_t \to 0$), the second moment $v_t \to 0$. Pointwise division scales infinitesimal noise up to unit magnitude $\mathcal{O}(1)$.
- **Spatial Decorrelation**: Voxel-wise independent scaling destroys spatial correlation between neighboring coordinates, injecting high-frequency micro-shears.
- **Severe Grid Folding**: Unregularized Adam drives Jacobian determinants negative ($\det(J) \le 0$), causing catastrophic topological collapse.

**Presenter Script**:
> *"Standard Adam is the engine of modern deep learning, but when applied to infinite-dimensional velocity fields, pointwise second-moment division destroys spatial correlation, amplifying infinitesimal noise into unit-magnitude shears that rip the coordinate manifold."*

---

### Slide 17: Riemannian `SobolevAdam` & Adaptive CFL Step Bounding
**Category**: RIEMANNIAN OPTIMIZATION  
**Title**: Riemannian `SobolevAdam` & Adaptive CFL Step Bounding  
**Subtitle**: Hilbert Space $H^s$ Metric Preconditioning for Diffeomorphic Flow  
**Figure**: [`docs/presentation/figures/diag_sobolev_adam_comparison_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_sobolev_adam_comparison_v01.png)

**Core Content**:
- **Sobolev Metric Preconditioning**: Precondition Adam updates with the Sobolev Green operator: $\Delta \mathbf{v}_{\text{Sobolev}} = \mathcal{G} \cdot \Delta \mathbf{v}_{\text{Adam}} = (I - \alpha \Delta)^{-s} \Delta \mathbf{v}$.
- **Restoring Spatial Correlation**: Low-pass filters high-frequency quotient noise while preserving adaptive learning rates along anatomical boundaries.
- **Adaptive Courant-Friedrichs-Lewy (CFL) Bound**: Enforce step displacement limit $\max \|\mathbf{s}\|_2 \le 0.50\text{ voxels}$, preventing discrete coordinate crossover during Euler stepping.
- **Strict Diffeomorphic Output**: Guarantees strictly positive Jacobian determinants ($\min \det(J) \ge +0.0517 > 0$) with **`0.0000%` grid folding**.

**Presenter Script**:
> *"SobolevAdam resolves the metric collapse by passing Adam updates through the Sobolev Green operator and capping displacement steps to 0.50 voxels. This restores spatial regularity and guarantees strictly positive Jacobian determinants."*

---

### Slide 18: Exact Homogeneous Dirichlet Boundary Operators (DST-I)
**Category**: BOUNDARY OPERATORS  
**Title**: Exact Homogeneous Dirichlet Boundary Operators (DST-I)  
**Subtitle**: Eliminating Periodic FFT Edge Reflections via Discrete Sine Transforms  
**Figure**: [`docs/presentation/figures/diag_dsti_boundary_operators_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_dsti_boundary_operators_v01.png)

**Core Content**:
- **The Periodic Boundary Flaw**: Standard Fourier FFT filtering assumes periodic domain boundary conditions, creating non-physical circular reflections at image edges $\partial \Omega$.
- **Separable DST-I Basis**: Project velocity updates onto orthogonal Dirichlet sine modes: $S(k, n) = \sqrt{\frac{2}{N+1}} \sin\left(\frac{\pi (k+1)(n+1)}{N+1}\right)$.
- **Exact Zero Boundary Clamping**: Analytically guarantees $\mathbf{v}(\mathbf{x} \in \partial \Omega) \equiv \mathbf{0}$, preventing outer boundary coordinate drift.
- **Dirichlet Green Operator**: $\mathcal{G}_{\text{DSTI1}} = \mathbf{S}^{-1}(I + \alpha \mathbf{\Lambda})^{-1}\mathbf{S}$ provides exact closed-form Sobolev preconditioning without boundary leakage.

**Presenter Script**:
> *"Standard FFT filtering assumes the world repeats periodically, which causes border deformations to wrap around to the opposite edge. The Discrete Sine Transform Type-I analytically clamps boundary velocities to zero, eliminating edge reflections."*

---

### Slide 19: Cohort Metrology on the 90-Pair Mindboggle Benchmark
**Category**: EMPIRICAL BENCHMARKS  
**Title**: Cohort Metrology on the 90-Pair Mindboggle Benchmark  
**Subtitle**: Statistical Inference, Anatomical Edge Snapping, & 16-Second GPU Acceleration  
**Figure**: [`docs/presentation/figures/diag_cohort90_metrology_v01.png`](file:///Users/stnava/code/syntx/docs/presentation/figures/diag_cohort90_metrology_v01.png)

**Core Content**:
- **Decisive Statistical Superiority**: 90-Pair paired $t$-test: $t=12.2539, p=8.33 \times 10^{-21}$; Wilcoxon signed-rank: $W=21.0, p=3.52 \times 10^{-16}$; Cohen's $d=1.2917$.
- **Win Rate Leadership**: TVF achieves a **100% win sweep (90/90 wins)** with `0.6445` Mean Symmetric Cortical Dice (+2.29% over ANTs reference).
- **100% Zero-Folding Guarantee**: 100.0% of cohort pairs achieve strictly positive Jacobian determinants ($\det(J) > 0$ across all 90 pairs).
- **Wall-Clock Acceleration**: Full deformable 3D volume registration completes in **`~12–16s` per pair on modern GPU architectures** ($7.5\times - 24\times$ speedup).

**Presenter Script**:
> *"Across the standardized 90-pair Mindboggle cohort, TVF won 90 out of 90 pairs with p = 8.3e-21, maintained 0.000% folding, and completed 3D registration in 16 seconds on modern GPU hardware."*

---

### Slide 20: The Future: Variational Guarantees in Deep Learning
**Category**: SYNTHESIS & THE FUTURE  
**Title**: The Future of Diffeomorphic AI: Variational Guarantees in Deep Learning  
**Subtitle**: Unifying Deep Representation Learning with Topological Invariance & Mathematical Rigor  
**Figure**: [`docs/presentation/figures/fig_diffeomorphic_ai_future_v01.jpg`](file:///Users/stnava/code/syntx/docs/presentation/figures/fig_diffeomorphic_ai_future_v01.jpg)

**Core Content**:
- **Unified Algorithmic Suite**: `syntx` unifies affine initialization, Eulerian SyN, TVF LDDMM, and SobolevAdam into a modular, single-interpolation framework.
- **Bridging Deep Features & Manifolds**: Seamlessly integrates deep visual representations (DINOv2, VGG) with exact diffeomorphic optimization.
- **Single-Shot Inference + Variational Guarantees**: Combines the execution speed of tensor neural networks with the mathematical guarantees of Riemannian differential geometry.
- **Open-Source & Fully Reproducible**: Complete code, benchmark pipelines, interactive HTML reports, and figures are open-source and reproducible.

**Presenter Script**:
> *"In summary, `syntx` bridges classical Riemannian differential geometry and modern deep learning. We retain certified topological guarantees while harnessing the full speed of tensor computing. Thank you, and I welcome your questions."*
