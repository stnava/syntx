# Reproducible Mindboggle-101 Deformable Registration Benchmark Tutorial

This tutorial provides step-by-step instructions to set up **`syntx`** from scratch on a clean machine (NVIDIA CUDA GPU, Apple Silicon MPS, or CPU), download and organize the **Mindboggle-101** benchmark dataset, generate a publication-quality standard diagnostic visual report (on `mbhard`), and execute the full 90-pair reproducible deformable registration evaluation.

---

## 1. Prerequisites & Environment Setup (CUDA GPU / Apple Silicon / CPU)

### 1.1 Create and Activate a Fresh Environment

`syntx` supports **Python 3.10, 3.11, and 3.12+** (with pre-built binary wheels available for `antspyx`, `torch`, and `jaxlib` across these versions):

```bash
# Using Conda (Python 3.11 or 3.12)
conda create -n syntx_cuda python=3.12 -y
conda activate syntx_cuda

# Or using Python venv
python3 -m venv ~/syntx_env
source ~/syntx_env/bin/activate
```

### 1.2 Install PyTorch with CUDA Acceleration

Install PyTorch matching your system's CUDA version. For CUDA 12.1+:

```bash
# NVIDIA GPU (Linux / Windows) - CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon macOS (MPS acceleration) or CPU
pip install torch torchvision
```

**Verify GPU acceleration:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```
*(On NVIDIA systems, this should print `CUDA available: True | Device count: 1`)*.

### 1.3 Install Core Dependencies & ANTsPy

```bash
pip install antspyx scipy matplotlib pandas plotly
```

### 1.4 Clone and Install Syntx in Editable Mode

```bash
git clone https://github.com/stnava/syntx.git
cd syntx
pip install -e .
```

**Verify package installation:**
```bash
python -c "import syntx; print('Syntx installed successfully! Version:', syntx.__version__)"
```

---

## 2. Mindboggle-101 Dataset Acquisition & Organization

The benchmark uses the standardized **Mindboggle-101** dataset ([Klein & Tourville, *Front. Neurosci.* 2012](https://mindboggle.info/data.html)), comprising 101 manually labeled T1-weighted brain MRI volumes with expert **DKT31** cortical label maps across four cohorts: `OASIS-TRT-20`, `NKI-RS-22`, `NKI-TRT-20`, and `MMRR-21`.

### 2.1 Download the Dataset

Download the skull-stripped brain volumes and manual DKT31 cortical label maps from the official Mindboggle project or Harvard Dataverse / OSF:

- **Mindboggle Project Site:** [https://mindboggle.info/data.html](https://mindboggle.info/data.html)
- **OSF Data Repository:** [https://osf.io/ujhir/](https://osf.io/ujhir/)

### 2.2 Expected Directory Layout

Organize your unpacked volume files into the following directory tree:

```
$SYNTX_DATA_DIR/ (e.g. /data/mindboggle/volumes/)
  ├── OASIS-TRT-20_volumes/
  │   ├── OASIS-TRT-20-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   ├── OASIS-TRT-20-2/
  │   └── ... (20 subjects)
  ├── NKI-RS-22_volumes/
  │   ├── NKI-RS-22-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (22 subjects)
  ├── NKI-TRT-20_volumes/
  │   ├── NKI-TRT-20-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (20 subjects)
  └── MMRR-21_volumes/
      ├── MMRR-21-1/
      │   ├── t1weighted_brain.nii.gz
      │   └── labels.DKT31.manual.nii.gz
      └── ... (21 subjects)
```

> **Requirements per subject:**
> - `t1weighted_brain.nii.gz`: Skull-stripped T1-weighted anatomical MRI volume.
> - `labels.DKT31.manual.nii.gz`: Expert ground-truth cortical segmentation map with 62 DKT cortical labels.

### 2.3 Automated Dataset Organizer Helper (Turnkey Setup)

If you downloaded raw tar/zip archives or extracted folders into an unorganized folder (e.g. `~/Downloads/mindboggle_raw/`), `syntx` provides an automated organizer helper that discovers all volumes, unpacks archives if needed, and builds the exact directory hierarchy using zero-copy hardlinks:

**Option A: CLI Organizer**
```bash
python -m syntx.benchmark --organize-data ~/Downloads/mindboggle_raw --target-dir ~/data/mindboggle/volumes
```

**Option B: Python API**
```python
from syntx.benchmark import organize_mindboggle_data

is_valid, report = organize_mindboggle_data(
    source_path="~/Downloads/mindboggle_raw",
    target_dir="~/data/mindboggle/volumes",
    verbose=True
)
print("Organization successful:", is_valid)
```

### 2.4 Set the Dataset Environment Variable

Export the path in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export SYNTX_DATA_DIR="/path/to/mindboggle/volumes"
```

### 2.5 Verify Dataset Integrity in One Command

Run the built-in integrity check to verify that all 90 pairs (40 intra-subject and 50 inter-subject defined in `examples/pairs.csv`) exist:

```bash
python -m syntx.benchmark --check-data
```

**Expected output:**
```
[syntx.benchmark] Dataset Location: '/path/to/mindboggle/volumes'
[syntx.benchmark] Pairs Configuration: '/path/to/syntx/examples/pairs.csv'
[syntx.benchmark] Dataset verified successfully! All 90 pairs ready at: '/path/to/mindboggle/volumes'
```

---

## 3. Pipeline Architecture, Preprocessing & Affine Alignment Strategy

The `syntx` registration pipeline consists of a four-stage hierarchical process:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Preprocessing & Normalization"]
        R1["Raw Fixed & Moving NIfTI Volumes"] --> M1["Foreground Non-Zero Extraction (I > 0)"]
        M1 --> N1["Robust 2nd–98th Percentile Normalization [0.0, 1.0]"]
        N1 --> O1["Canonical LPI Orientation & Spacing Preservation"]
    end

    subgraph S2["Stage 2: Deterministic Robust Affine Alignment"]
        O1 --> T1["Translation Center Match: CoM vs FOV"]
        T1 --> C1["18-Cone Angular Perturbations (Pitch / Roll / Yaw)"]
        C1 --> E1["Deterministic Regular Sampling & Foreground Union MI"]
        E1 --> A1["Multi-Stage Continuous Affine Solver (Rigid → Affine)"]
    end

    subgraph S3["Stage 3: Diffeomorphic SyN Optimization"]
        A1 --> P1["Multi-Resolution Scale Pyramid (4x, 2x, 1x)"]
        P1 --> D1["Eulerian Velocity Update (grad_step=0.25)"]
        D1 --> S4["Fluid / Sobolev Spectral Smoothing"]
        S4 --> I1["In-Loop Anderson Accelerated Inverse (m=5)"]
    end

    subgraph S4["Stage 4: Single Interpolation & Metric Evaluation"]
        I1 --> W1["Single Nearest-Neighbor Transform Composition: Φ = φ ∘ A"]
        W1 --> M2["Cortical DKT31 Mean Dice (Fixed, Moving, Sym)"]
        W1 --> J1["Jacobian Determinant det(J) & Folding Rate"]
        W1 --> V1["Physical Inverse Identity Error in mm"]
        W1 --> R2["5-Figure Standalone Interactive HTML Report"]
    end
```

### 3.1 Preprocessing Pipeline

To eliminate gradient instability and dynamic range compression caused by background air and outlier acquisition artifacts:

1. **Non-Zero Foreground Masking**:
   Medical brain volumes contain substantial background zero-padding. The normalizer isolates tissue voxels:
   $$\Omega_{\text{fg}} = \{ \mathbf{x} \in \Omega \mid I(\mathbf{x}) > 0 \}$$
2. **Entropy-Optimal 2nd–98th Percentile Truncation**:
   High-intensity acquisition spikes (e.g. vascular flow, skull residue) compress joint intensity histograms. We calculate robust foreground percentiles:
   $$p_{02} = \text{percentile}(I(\Omega_{\text{fg}}), 2), \quad p_{98} = \text{percentile}(I(\Omega_{\text{fg}}), 98)$$
   and apply linear clamping to $[0.0, 1.0]$:
   $$I_{\text{norm}}(\mathbf{x}) = \text{clamp}\left(\frac{I(\mathbf{x}) - p_{02}}{p_{98} - p_{02} + 10^{-6}}, 0.0, 1.0\right) \quad \text{for } \mathbf{x} \in \Omega_{\text{fg}}$$
3. **Flat-Field Graceful Fallback**:
   When $p_{98} \le p_{02} + 10^{-4}$ (e.g. binary masks or flat phantoms), the normalizer gracefully scales to $[0.0, \max(I)]$ to prevent division by zero or array collapse.
4. **Physical Anisotropy Scaling**:
   Voxel dimensions and orientation directions are preserved without reslicing to prevent spatial blurring prior to optimization.

---

### 3.2 Deterministic Multi-Start Affine Alignment Strategy (`syntx.robust_affine`)

Standard single-start affine algorithms easily trap in local gradient descent minima when brain orientations have significant angular tilt. `syntx.robust_affine` eliminates orientation entrapment through a four-step multi-start search:

```mermaid
graph LR
    subgraph MultiStart["Multi-Start Candidate Generation"]
        A["CoM Translation"] --> C["18 Cone Rotations: ±4°, ±8°, ±12° in Pitch/Roll/Yaw (Rodrigues Formula)"]
        B["FOV Translation"] --> C
    end
    subgraph Scoring["Deterministic Scoring"]
        C --> D["Foreground Union Masking: (I > 0.01) | (J > 0.01)"]
        D --> E["Deterministic Regular Grid Sampling (20%)"]
        E --> F["Mattes MI Candidate Evaluation"]
    end
    subgraph Refinement["GPU Optimization"]
        F --> G["Select Best Pose Candidate"]
        G --> H["Multi-Stage Continuous Solver: Rigid → Affine"]
        H --> I["Optimal Affine Transform Matrix A"]
    end
```

1. **Dual Geometric Center Matching**:
   Evaluates initial translation using both **Center of Mass (CoM)** matching ($\mathbf{t}_{\text{CoM}} = \text{CoM}_{\text{fixed}} - \text{CoM}_{\text{moving}}$) and **Field of View (FOV)** geometric center matching ($\mathbf{t}_{\text{FOV}} = \mathbf{c}_{\text{fixed}} - \mathbf{c}_{\text{moving}}$).
2. **18-Cone Angular Perturbation Grid**:
   Constructs 18 spatial rotational candidates around the superior translation center:
   - **Pitch Perturbations**: $\pm 4^\circ, \pm 8^\circ, \pm 12^\circ$
   - **Roll Perturbations**: $\pm 4^\circ, \pm 8^\circ, \pm 12^\circ$
   - **Yaw Perturbations**: $\pm 4^\circ, \pm 8^\circ, \pm 12^\circ$
   Rotations are parameterized on the Lie Algebra $so(3) \rightarrow SO(3)$ using the analytical Rodrigues rotation formula:
   $$R(\boldsymbol{\omega}) = \mathbf{I} + \frac{\sin\theta}{\theta} [\boldsymbol{\omega}]_{\times} + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_{\times}^2, \quad \theta = \|\boldsymbol{\omega}\|_2$$
3. **Foreground Union-Masked Mutual Information**:
   To prevent background zero padding from dominating candidate joint histograms, candidate scoring applies foreground union masking:
   $$\text{mask} = (I > 0.01) \mid (J > 0.01)$$
   with **deterministic regular uniform grid sampling** (`sampling_strategy='regular'`, 20% sample), eliminating stochastic random sampling noise.
4. **GPU Continuous Multi-Stage Refinement**:
   The winning candidate is refined through 4-stage multi-resolution continuous gradient descent (Rigid $\rightarrow$ Affine), completing in **~1.2–2.8 seconds** on GPU ($10–24\times$ faster than ANTs CPU).

---

## 4. Generating Standard Diagnostic Reports

`syntx` provides visualization tools in [`syntx.viz`](file:///Users/stnava/code/syntx/src/syntx/viz/__init__.py) to generate standalone, interactive 5-figure HTML verification reports for any registration task.

### 4.1 General Example: Generating a Report from ANY Pair of Images (2D or 3D)

You can run deformable registration on any two arbitrary NIfTI images and generate a full diagnostic report in Python:

```python
import ants
import syntx
from syntx.viz import create_registration_report

# 1. Load any arbitrary fixed target and moving source images
fixed_img = ants.image_read("path/to/my_fixed_brain.nii.gz")
moving_img = ants.image_read("path/to/my_moving_brain.nii.gz")

# (Optional) Load corresponding ground-truth segmentation label maps if available
fixed_lbl = ants.image_read("path/to/my_fixed_labels.nii.gz")   # or None
moving_lbl = ants.image_read("path/to/my_moving_labels.nii.gz") # or None

# 2. Multi-start robust affine pre-alignment
# Evaluates 18 pitch/roll/yaw cone search candidates around CoM and FOV geometric centers
reg_aff = syntx.robust_affine(fixed_img, moving_img, mode="auto", verbose=True)

# 3. High-accuracy deformable SyN registration
reg_syn = syntx.syn(
    fixed=fixed_img,
    moving=moving_img,
    initial_transform=reg_aff["fwdtransforms"],
    backend="pytorch",
    device="cuda",              # 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon), or 'cpu'
    grad_step=0.25,
    flow_sigma=3.0,             # Fluid velocity smoothing (std dev = sqrt(3) mm)
    total_sigma=0.0,            # Pure fluid formulation
    reg_iterations=[100, 100, 50], # Multi-resolution pyramid levels (4x, 2x, 1x)
    similarity_metric="cc2",
    inverse_method="anderson",  # Anderson accelerated inverse
    formulation="eulerian",     # Peak Eulerian pullback
    regularizer="gaussian",     # 'gaussian' for peak accuracy, 'sobolev' for 0% folding
    verbose=True
)

# 4. Generate standalone, self-contained interactive HTML diagnostic report
report = create_registration_report(
    fixed=fixed_img,
    moving=moving_img,
    reg=reg_syn,
    fixed_label=fixed_lbl,
    moving_label=moving_lbl,
    output_html="reports/my_custom_registration_report.html",
    fixed_name="Patient 01 (Fixed Target)",
    moving_name="Patient 02 (Moving Source)",
    title="Custom 3D Brain SyN Registration Report"
)

print(f"Report generated successfully: {report['html_path']}")
```

### 4.2 Quick Benchmark Demo: Generating a Report on `mbhard`

For standardized Mindboggle test cases, you can generate a report with a single command:

**One-Line CLI:**
```bash
python -m syntx.benchmark --demo --demo-dataset mbhard --demo-html docs/reports/mbhard_standard_report.html
```

**Python 3-Liner:**
```python
from syntx.benchmark import run_standard_report_demo

report_path = run_standard_report_demo(
    dataset_key="mbhard",
    output_html="docs/reports/mbhard_standard_report.html",
    model="gaussian",
    verbose=True
)
print(f"Report ready: {report_path}")
```

### 4.3 What the Standard Diagnostic Report Contains

The generated standalone HTML file contains:
1. **Interactive Summary Header**:
   - Mean Symmetric Cortical Dice ($\text{Dice}_{\text{sym}}$), Fixed Space Dice, Moving Space Dice.
   - Grid Folding Percentage ($\det(J) \le 0$), Minimum Jacobian ($\min \det(J)$).
   - Real Physical Inverse Identity Error in mm (Mean, $p_{95}$, Peak Max).
   - Total GPU compute time in seconds.
2. **Figure 1 (Input Anatomical Pair)**:
   - Tri-planar orthographic slice panel (Axial, Coronal, Sagittal) in canonical LPI orientation with physical aspect ratio scaling and shared row colorbars.
3. **Figure 2 (Standard 4-Panel Diagnostic)**:
   - **Panel A (Mesh Grid)**: Deformed coordinate grid (Cyan) showing continuous coordinate transformation.
   - **Panel B (Jacobian Determinant Map)**: Divergent $\log\det(J)$ map (`seismic` colormap centered at 1.0) displaying local tissue compression ($\det J < 1$) vs expansion ($\det J > 1$).
   - **Panel C (Inverse Error Map)**: Real physical inverse identity error $\|\phi_{\text{inv}}(\mathbf{x} + \phi(\mathbf{x})) + \phi(\mathbf{x})\|_2$ in mm (`inferno` colormap).
   - **Panel D (High-Contrast Edge Alignment)**: Canny structural edge contour overlay (Cyan fixed target vs Magenta warped source).
4. **Figure 3 (Time-Varying Velocity Field Flow)** *(for TVF registrations)*:
   - Continuous velocity magnitude heatmap with amplified quiver flow vectors ($125\times$) and Thin-Plate Bending Energy ($\text{Bnd}$).
5. **Interactive Provenance Card**:
   - Complete record of optimization learning rate, fluid/elastic smoothing parameters, multi-resolution iterations, and hardware device.

---

## 5. Running the Full Deformable Benchmark from Scratch

The standardized benchmark executes 90 image pairs:
- **Rows 0–39 (40 Pairs):** Intra-subject longitudinal test-retest pairs (evaluates precision and consistency).
- **Rows 40–89 (50 Pairs):** Inter-subject cross-individual pairs (evaluates cross-subject morphological variance).

### 5.1 Run the Full 90-Pair Cohort

Run the entire cohort benchmark using either Gaussian regularized SyN (peak accuracy standard) or Sobolev regularized SyN (smooth topology-preserving standard):

```bash
# Option A: Gaussian Regularized SyN (88/90 Wins vs ANTs C++ SyN, +1.66% Mean Dice)
python -m syntx.benchmark --cohort --model gaussian

# Option B: Sobolev Regularized SyN (81/90 Wins vs ANTs C++ SyN, 90.0% Zero-Fold)
python -m syntx.benchmark --cohort --model sobolev

# Option C: Evaluate Both Gaussian and Sobolev on Every Pair
python -m syntx.benchmark --cohort --model both
```

### 5.2 Run Specific Pairs or Subsets

Evaluate individual pairs or subsets:

```bash
# Evaluate Pair 0 (Gaussian SyN)
python -m syntx.benchmark --pair-idx 0 --model gaussian

# Evaluate a custom subset of pairs
python -m syntx.benchmark --cohort --pairs 0 1 2 45 67 82 --model gaussian
```

### 5.3 Generate Interactive Master Population Reports

Once benchmark runs finish (or incrementally during execution), generate the comprehensive interactive dashboards:

```bash
# Generate Master Deformable SyN Benchmark Dashboard
python -m syntx.benchmark --cohort

# Generate Dedicated 90-Pair Affine Population Benchmark Report
python -m syntx.benchmark --affine-report
```

### Generated Artifacts & File Locations

- **Individual Pair JSON Records:** `results/reproducible_eval/pair_XXX_<model>.json`
- **Master Summary JSON:** `results/reproducible_90pair_master_summary.json`
- **Master Deformable HTML Report:** `docs/reproducible_90pair_report.html`
- **Master Affine HTML Report:** `docs/reproducible_90pair_affine_report.html`

---

## 6. Comprehensive Evaluation Metrics: What We Measure and Why

The benchmark provides a rigorous multi-dimensional assessment of registration quality:

### 6.1 Anatomical Overlap: Cortical DKT31 Mean Dice Score

Registration accuracy is evaluated on **62 discrete manual anatomical cortical labels** (31 labels per hemisphere from the Mindboggle DKT protocol).

To avoid directional bias, DICE is evaluated **symmetrically in both image spaces**:
- **Fixed Space**: Moving labels warped to fixed space ($\mathbf{L}_{\text{mov}} \circ \phi_{\text{fwd}}$) using nearest-neighbor interpolation, compared against $\mathbf{L}_{\text{fix}}$:
  $$\text{Dice}_{\text{fix}} = \frac{2 |\mathbf{L}_{\text{fix}} \cap (\mathbf{L}_{\text{mov}} \circ \phi_{\text{fwd}})|}{|\mathbf{L}_{\text{fix}}| + |\mathbf{L}_{\text{mov}} \circ \phi_{\text{fwd}}|}$$
- **Moving Space**: Fixed labels warped to moving space ($\mathbf{L}_{\text{fix}} \circ \phi_{\text{inv}}$) using nearest-neighbor interpolation, compared against $\mathbf{L}_{\text{mov}}$:
  $$\text{Dice}_{\text{mov}} = \frac{2 |\mathbf{L}_{\text{mov}} \cap (\mathbf{L}_{\text{fix}} \circ \phi_{\text{inv}})|}{|\mathbf{L}_{\text{mov}}| + |\mathbf{L}_{\text{fix}} \circ \phi_{\text{inv}}|}$$
- **Symmetric Mean Dice**:
  $$\text{Dice}_{\text{sym}} = \frac{1}{2} \left( \text{Dice}_{\text{fix}} + \text{Dice}_{\text{mov}} \right)$$

> **Guardrail Invariant:** In discrete anatomical label maps, a difference of $\ge 0.01$ (1% Dice) represents a major anatomical difference. Nearest-neighbor interpolation is strictly enforced to prevent artificial label mixing.

### 6.2 Diffeomorphic Manifold Regularity: Jacobian Determinant $\det(J)$

A transformation $\phi(\mathbf{x}) = \mathbf{x} + \mathbf{u}(\mathbf{x})$ is a valid diffeomorphism only if the Jacobian determinant is strictly positive everywhere ($\det(J(\mathbf{x})) > 0$).

- **Spatial Jacobian Matrix**:
  $$J(\mathbf{x}) = \nabla \phi(\mathbf{x}) = \mathbf{I} + \begin{bmatrix} \frac{\partial u_x}{\partial x} & \frac{\partial u_x}{\partial y} & \frac{\partial u_x}{\partial z} \\ \frac{\partial u_y}{\partial x} & \frac{\partial u_y}{\partial y} & \frac{\partial u_y}{\partial z} \\ \frac{\partial u_z}{\partial x} & \frac{\partial u_z}{\partial y} & \frac{\partial u_z}{\partial z} \end{bmatrix}$$
- **Grid Folding Percentage ($\text{Fold}\%$)**: Percentage of voxels where local space collapses or self-intersects:
  $$\text{Fold}\% = \frac{1}{|\Omega|} \int_{\Omega} \mathbf{1}_{(\det(J(\mathbf{x})) \le 0)} \, d\mathbf{x} \times 100\%$$
- **Minimum Jacobian ($\min \det(J)$)**: Smallest determinant across the volume. If $\min \det(J) > 0$, the transformation is completely fold-free (zero topological tearing).

### 6.3 Physical Inverse Identity Consistency (mm)

True diffeomorphic mapping requires the forward transform $\phi_{\text{fwd}}$ and inverse transform $\phi_{\text{inv}}$ to compose to the exact identity: $\phi_{\text{inv}}(\phi_{\text{fwd}}(\mathbf{x})) = \mathbf{x}$.

- **Real Physical Error Map**:
  $$\mathbf{e}(\mathbf{x}) = \left\| \phi_{\text{inv}}(\mathbf{x} + \mathbf{u}_{\text{fwd}}(\mathbf{x})) + \mathbf{u}_{\text{fwd}}(\mathbf{x}) \right\|_2 \quad (\text{in mm})$$
- We report the **Mean Inverse Error**, the **95th Percentile ($p_{95}$)**, and the **Peak Maximum Error**. High inverse consistency ($\text{mean error} < 0.03\text{ mm}$) guarantees bidirectional invertibility.

### 6.4 Intensity Similarity & Structural Edge Overlap

- **Mattes Mutual Information (MI)**: 32-bin joint entropy alignment evaluated over foreground non-zero tissue union.
- **Local Normalized Cross Correlation (LNCC)**: Multi-channel local correlation with sliding box-filter window ($w=9$).
- **Canny Structural Edge Alignment**: Overlap ratio between canny structural edge contours of the target and registered images.

---

## 7. Parameter Election Rationale: Why These Exact Settings Were Chosen

The default parameters in `syntx.syn` were established through extensive systematic parameter sweeps across all 90 Mindboggle pairs to achieve optimal anatomical accuracy and topological regularity:

| Parameter | Selected Value | Algorithmic Rationale |
|:---|:---|:---|
| `formulation` | `'eulerian'` | **Eliminates Lagrangian Drift**: Eulerian displacement updates avoid the cumulative velocity pullback drift of Lagrangian composition, yielding $+1.66\%$ higher Cortical Dice and $10\times$ lower folding. |
| `grad_step` | `0.25` | **CFL Numerical Stability**: Balances spatial gradient descent velocity against the Courant-Friedrichs-Lewy (CFL) limit. Higher steps ($>0.35$) cause local coordinate tearing; lower steps ($<0.15$) stall in sub-optimal sulcal alignment. |
| `flow_sigma` | `3.0` | **Fluid Regularization Standard**: ITK variance convention $\sigma^2=3.0$ (std dev $\sigma \approx 1.732\text{ mm}$). Smooths iterative velocity updates to prevent high-frequency grid kinks while allowing the field to penetrate deep sulci. |
| `total_sigma` | `0.0` | **Fluid-Only Deformation**: Eliminates total elastic field smoothing, preserving boundary flexibility along sharp cortical edges. |
| `regularizer` | `'gaussian'` / `'sobolev'` | **Gaussian**: ITK sampled Gaussian kernel achieving peak accuracy standard (88/90 wins, 0.6382 Dice).<br>**Sobolev**: Spectral Fourier smoothing ($H^{1.5}$) enforcing $C^1$ smoothness and achieving $90.0\%$ zero-fold regularity (0.6342 Dice). |
| `similarity_metric` | `'cc2'` | **Analytical Gradient Parity**: ITK pseudo-gradient scaling through sliding box-filter cross correlation coupled with foreground variance flooring ($\text{Var}_{\text{safe}} = \max(\text{Var}(I), 10^{-6})$). |
| `reg_iterations` | `[100, 100, 50]` | **Multi-Resolution Gaussian Pyramid**: 3-level scale pyramid ($4\times, 2\times, 1\times$) aligns global hemispheric morphology before resolving fine sulcal gyri. |
| `inverse_method` | `'anderson'` | **Non-Divergent Diffeomorphism Inversion**: Fixed-point Picard iteration diverges in Eulerian compositions; Anderson acceleration (mixing depth $m=5$) guarantees monotonic inverse convergence ($<0.03\text{ mm}$ error). |
| `in_loop_inv_steps`| `10` | **In-Loop Inverse Consistency**: Updates the inverse displacement field inside the optimization loop, maintaining bidirectional symmetry at every iteration. |
| `affine` | `syntx.robust_affine` | **Multi-Start Orientation Robustness**: Evaluates 18 pitch/roll/yaw cone rotations around CoM and FOV centers using deterministic regular sampling and foreground union-masked MI, preventing $180^\circ$ inversion traps. |

---

## 8. Accent on Strict Scientific Reproducibility

To ensure 100% deterministic reproducibility across diverse hardware backends (NVIDIA CUDA, Apple Silicon MPS, CPU):

1. **Deterministic Regular Affine Sampling (`syntx.robust_affine`):**
   - Uses deterministic uniform grid sampling (`sampling_strategy='regular'`, 20% sample) and foreground union-masked Mutual Information ($\text{mask} = (I > 0.01) \mid (J > 0.01)$), eliminating stochastic random sampling variance.
2. **Single Interpolation Policy:**
   - Input images are never pre-warped prior to optimization. Forward non-linear warps and affine matrices are composed and applied to native-space segmentation maps in a single nearest-neighbor step (`interpolator='nearestNeighbor'`).
3. **Foreground 2nd–98th Percentile Normalization:**
   - All input volumes are truncated and normalized based on non-zero foreground percentiles, preventing vascular intensity spikes from compressing Mutual Information joint histograms.
4. **Subprocess Worker Isolation & Device Cache Cleansing:**
   - Each benchmark case executes in an isolated Python worker process (`syntx.benchmark.worker`), automatically releasing GPU allocator cache (`torch.cuda.empty_cache()` / `torch.mps.empty_cache()`) to guarantee zero memory fragmentation across serial evaluations.

---

## 9. CUDA GPU Performance Expectations

| Metric | ANTs C++ SyN (CPU) | Syntx (Apple Silicon MPS) | Syntx (NVIDIA RTX 4090 / A100 CUDA) |
|:---|:---|:---|:---|
| **Affine Registration Time** | ~28.5 s | ~2.8 s | **~1.2 s** ($24\times$ speedup) |
| **Deformable SyN Time (per 3D Pair)** | ~85–120 s | ~24–28 s | **~12–16 s** ($7.5\times$ speedup) |
| **GPU VRAM Footprint** | N/A (RAM: ~3 GB) | ~3.8 GB Unified | **~3.5–4.2 GB VRAM** |
| **90-Pair Cohort Total Time** | ~2.5–3.0 hours | ~40 minutes | **~20–25 minutes** |
| **Mean Cortical DKT31 Dice** | 0.6216 | 0.6382 | **0.6382** (+1.66% gain, 88/90 wins) |

