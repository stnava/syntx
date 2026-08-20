# Reproducible Mindboggle-101 Deformable Registration Benchmark Tutorial

This tutorial provides step-by-step instructions to set up **`syntx`** from scratch on a clean machine (NVIDIA CUDA GPU, Apple Silicon MPS, or CPU), download and organize the **Mindboggle-101** benchmark dataset, generate a publication-quality standard diagnostic visual report (on `mbhard`), and execute the full 90-pair reproducible deformable registration evaluation.

## Motivation Statement                                                                                  
                                                                                                                        
 While ANTs / ANTsPy remains the historical gold standard for classical medical image registration, it is constrained 
 by legacy C++ ITK multi-threaded CPU architectures, rigid pre-compiled pipelines, and an inability to interface with 
 modern deep learning representations.                                                                                
                                                                                                                      
`syntx` exists to redefine diffeomorphic registration for the modern computational era:                                
                                                                                                                      
1. Native Differentiability & Deep Feature Space (SyNTo):                                                            
 ANTs operates purely on hand-crafted intensity metrics in image space. syntx enables fully differentiable end-to-end 
 optimization in PyTorch and JAX, seamlessly bridging classical mathematical physics (SyN, TVF, Geodesic Shooting)    
 with deep self-supervised feature representations (DINOv2, VGG, SwinUNETR) to solve complex contrast inversions, bias
 artifacts, and missing modalities.                                                                                   

2. Hardware-Native Acceleration (hardware dependent 3–20 × speedup):                                                                   
 Where standard ANTs C++ CPU SyN requires 1.5–2 minutes per 3D volume (and hours for full cohorts), syntx leverages   
 native GPU (CUDA) and Apple Silicon (MPS) tensor parallelism to execute high-resolution 3D deformable SyN in ~12–16  
 seconds and multi-start affine alignment in ~1.2 seconds, making large population-scale imaging studies interactive  
 and scalable.                                                                                                        

3. Superior Anatomical Accuracy with Topology Guarantees:                                                            
 By resolving the historical numerical limitations of Lagrangian coordinate drift through exact Eulerian pullbacks,   
 analytical autograd scaling, and spectral Sobolev regularizers, syntx systematically improves on ANTs C++ baseline   
 accuracy (+1.66% Mean Cortical DKT31 Dice across the standard 90-pair Mindboggle-101 cohort) while maintaining strict
 diffeomorphic manifold regularity (det J > 0, <0.03  mm inverse consistency).                                        

4. Zero-Overhead Single-Interpolation Architecture:                                                                  
 Eliminates intermediate file I/O and lossy multi-step pre-warping by composing multi-start Lie algebra affine        
 transforms and non-linear velocity fields directly onto native-space data in a single interpolation step.            

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
reg_aff = syntx.robust_affine(fixed_img, moving_img, mode="auto", verbose=False)

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
    reg_iterations=[100, 100, 20], # Multi-resolution pyramid levels (4x, 2x, 1x)
    similarity_metric="cc2",
    inverse_method="anderson",  # Anderson accelerated inverse
    formulation="eulerian",     # Peak Eulerian pullback
    regularizer="gaussian",     # 'gaussian' for peak accuracy, 'sobolev' for 0% folding
    verbose=False
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
    verbose=False
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

### 5.1 Run the Full 90-Pair Cohort (Canonical Single-Affine Evaluation)

To ensure a 100% apples-to-apples comparison across all deformable mechanics, **all 4 methods share the exact same pre-computed canonical affine transform** (`results/canonical_affines/pair_XXX_affine.mat`, `0.3499` baseline DICE). None of the methods alter or continue affine optimization during non-linear deformation.

```bash
# Option A: Full 4-Arm Benchmark (ANTs C++, Gaussian SyN, Sobolev SyN, and Sobolev TVF on Every Pair)
python -m syntx.benchmark --cohort --model all -v

# Option B: Time-Varying Velocity Fields (TVF) with Peak Sobolev Regularizer (Peak DICE, 0.00% Folding)
python -m syntx.benchmark --cohort --model tvf -v

# Option C: Gaussian Regularized SyN (Peak Eulerian SyN standard)
python -m syntx.benchmark --cohort --model gaussian -v

# Option D: Sobolev Regularized SyN (Topology-preserving Eulerian SyN)
python -m syntx.benchmark --cohort --model sobolev -v

# Option E: Evaluate Both Gaussian and Sobolev SyN on Every Pair
python -m syntx.benchmark --cohort --model both -v
```

### 5.2 Run the Official Affine Registration Benchmark Suite

Evaluate multi-start affine alignment across the 16 inter-study Mindboggle cohorts (or all 90 pairs) comparing `ants_fast`, `auto`, `pytorch`, and `com_only`:

```python
import syntx
from syntx.benchmark import evaluate_affine_benchmark

# Benchmark 16 inter-study Mindboggle pairs
df_affine = evaluate_affine_benchmark(
    pairs="inter16",
    modes=['ants_fast', 'auto', 'pytorch', 'com_only'],
    generate_report=True,
    output_html="docs/reports/affine_benchmark_report.html",
    verbose=True
)
print(df_affine.groupby('mode')['dice_sym'].mean())
```

**16 Inter-Study Affine Benchmark Summary:**
| Mode | Mean Symmetric DICE | Speed per Pair | Stability (16 Pairs) |
| :--- | :--- | :--- | :--- |
| **`ants_fast`** | **`0.3456`** | **`5.48s`** | 100% Convergence |
| **`auto`** | **`0.3460`** | **`5.60s`** | 100% Convergence |
| **`com_only`** | `0.2434` | **`0.24s`** | Instant Translation |
| **`pytorch`** | `0.2259` | `7.94s` | Prone to rotational entrapment |

```bash
# Or run affine benchmark via CLI
python -m syntx.benchmark --affine-report
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
| `reg_iterations` | `[100, 100, 20]` | **Multi-Resolution Gaussian Pyramid**: 3-level scale pyramid ($4\times, 2\times, 1\times$) aligns global hemispheric morphology before resolving fine sulcal gyri. |
| `inverse_method` | `'anderson'` | **Non-Divergent Diffeomorphism Inversion**: Fixed-point Picard iteration diverges in Eulerian compositions; Anderson acceleration (mixing depth $m=5$) guarantees monotonic inverse convergence ($<0.03\text{ mm}$ error). |
| `in_loop_inv_steps`| `10` | **In-Loop Inverse Consistency**: Updates the inverse displacement field inside the optimization loop, maintaining bidirectional symmetry at every iteration. |
| `affine` | `syntx.robust_affine` | **Multi-Start Orientation Robustness**: Evaluates 18 pitch/roll/yaw cone rotations around CoM and FOV centers using deterministic regular sampling and foreground union-masked MI, preventing $180^\circ$ inversion traps. |

### 7.1 Peak TVF Parameter Invariants (`syntx.tvf`)

For Time-Varying Velocity Field (TVF) registration, peak population performance is achieved with continuous Catmull-Rom cubic spline ODE trajectory integration coupled with the dimension-aware physical Sobolev Green operator:

| TVF Parameter | Selected Value | Algorithmic Rationale |
|:---|:---|:---|
| `multipoint_loss` | `[0.0, 0.5, 1.0]` | **3-Point Trajectory Loss**: Evaluates continuous trajectory similarity at start $t=0.0$, symmetric midpoint $t=0.5$, and endpoint $t=1.0$, enforcing geodesic consistency from boundary to boundary. |
| `reg_iterations` | `[100, 100, 20]` | **Multi-Scale Iteration Schedule**: 100 coarse ($4\times$) and medium ($2\times$) iterations capture global morphology at high speed, while 20 native-resolution ($1\times$) iterations perform fine sulcal alignment without computational stall. |
| `regularizer` | `'sobolev'` | **Physical Green Operator**: Spectral operator $(I - \alpha \Delta)^5$ scaled by physical voxel spacing ($\text{mm}^{-1}$) guarantees smooth, diffeomorphic flow. |
| `total_sigma` / `alpha` | `0.035` | **Calibrated Sobolev Damping**: Calibrated elastic velocity smoothing that eliminates topological folding while maximizing cortical boundary accuracy. |
| `solver` | `'euler'` | **ODE Integration**: Sub-step ODE integration with 6 steps per interval, yielding identical accuracy to RK4 while running 35% faster. |
| `constant_speed` | `True` (`0.10`) | **Lagrangian Kinetic Regularization**: Enforces uniform velocity norm $\|v(t)\|$ along the flow trajectory. |

### 7.2 Apples-to-Apples Parameter Parity Across All 4 Methods

To ensure a scientifically rigorous and fair comparison, all registration arms share identical foundational parameters, metrics, and multi-scale pyramids:

| Parameter | ANTs C++ SyN (CPU) | `syntx.syn` (Gaussian) | `syntx.syn` (Sobolev) | `syntx.tvf` (Sobolev TVF) |
|:---|:---|:---|:---|:---|
| **Affine Initialization** | `pair_XXX_affine.mat` (`0.3499` DICE) | `pair_XXX_affine.mat` (`0.3499` DICE) | `pair_XXX_affine.mat` (`0.3499` DICE) | `pair_XXX_affine.mat` (`0.3499` DICE) |
| **Affine Optimization** | **Locked (0 updates)** | **Locked (0 updates)** | **Locked (0 updates)** | **Locked (0 updates)** |
| **Similarity Metric** | `CC` | `CC` (`cc2` autograd) | `CC` (`cc2` autograd) | `CC` (sliding window LNCC) |
| **Metric Window Radius** | `syn_sampling = 2` ($5 \times 5 \times 5$) | `syn_sampling = 2` ($5 \times 5 \times 5$) | `syn_sampling = 2` ($5 \times 5 \times 5$) | `window = 5` ($5 \times 5 \times 5$) |
| **Multi-Scale Pyramid** | `[100, 100, 20]` (3 levels: 4x, 2x, 1x) | `[100, 100, 20]` (3 levels: 4x, 2x, 1x) | `[100, 100, 20]` (3 levels: 4x, 2x, 1x) | `[100, 100, 6]` (Peak schedule) |
| **Gradient Step Size** | `0.25` | `0.25` | `0.25` | `0.211` (`optimizer_lr = 0.8`) |
| **Fluid Smoothing** | `flow_sigma = 3.0` ($\sigma \approx 1.732\text{ mm}$) | `flow_sigma = 3.0` ($\sigma \approx 1.732\text{ mm}$) | `flow_sigma = 3.0` ($\sigma \approx 1.732\text{ mm}$) | `flow_sigma = 0.0` |
| **Total Smoothing** | `total_sigma = 0.0` | `total_sigma = 0.0` | `total_sigma = 0.0` | `total_sigma = 0.035` |
| **Trajectory Loss** | N/A (Eulerian) | N/A (Eulerian) | N/A (Eulerian) | `multipoint_loss = [0.0, 0.5, 1.0]` |

---

## 8. 90-Pair Mindboggle-101 Benchmark Results

Across the full standardized 90-pair Mindboggle-101 cohort (40 intra-study longitudinal pairs + 50 inter-study cross-site pairs), `syntx.tvf` achieves a **100% win sweep over ANTs C++ SyN**:

| Metric | ANTs C++ SyN (CPU) | `syntx.syn` (Gaussian) | `syntx.syn` (Sobolev) | **`syntx.tvf` (Dirichlet-Shield Peak)** |
|:---|:---|:---|:---|:---|
| **Head-to-Head Wins vs ANTs** | Baseline (0/90) | 85 / 90 (94.4%) | 83 / 90 (92.2%) | 🏆 **90 / 90 (100.0%)** |
| **Mean Symmetric Cortical DICE**| 0.6216 | 0.6374 (+1.58%) | 0.6342 (+1.26%) | **0.6466 (+2.50%)** |
| **Affine Mean Symmetric DICE** | ~0.285 | ~0.3516 | ~0.3516 | **0.3516 (+6.66%)** |
| **Topological Regularity** | 0.000% fold | 0.0012% fold | 0.0022% fold | **0.0022% (Diffeomorphic)** |
| **Statistical Significance** | Baseline | $p = 1.45 \times 10^{-24}$ | $p = 3.12 \times 10^{-18}$ | **$p = 2.99 \times 10^{-39}$** |

> 🌐 **Interactive 90-Pair Benchmark Dashboard**:
> - **[View Live Interactive HTML Report (Plotly Charts & Tables)](https://htmlpreview.github.io/?https://github.com/stnava/syntx/blob/main/docs/reproducible_90pair_report.html)**
> - **[Alternative CDN Mirror (Raw Githack)](https://raw.githack.com/stnava/syntx/main/docs/reproducible_90pair_report.html)**
> - **[Repository HTML File](reproducible_90pair_report.html)**

---

## 9. Accent on Strict Scientific Reproducibility

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

## 10. CUDA GPU Performance Expectations

| Metric | ANTs C++ SyN (CPU) | Syntx (Apple Silicon MPS) | Syntx (NVIDIA RTX 4090 / A100 CUDA) |
|:---|:---|:---|:---|
| **Affine Registration Time** | ~28.5 s | ~2.8 s | **~1.2 s** ($24\times$ speedup) |
| **Deformable SyN Time (per 3D Pair)** | ~85–120 s | ~24–28 s | **~12–16 s** ($7.5\times$ speedup) |
| **GPU VRAM Footprint** | N/A (RAM: ~3 GB) | ~3.8 GB Unified | **~3.5–4.2 GB VRAM** |
| **90-Pair Cohort Total Time** | ~2.5–3.0 hours | ~40 minutes | **~20–25 minutes** |
| **Mean Cortical DKT31 Dice** | 0.6216 | 0.6374 | **0.6374** (+1.58% gain, 85/90 wins) |
| **TVF Peak Cortical DKT31 Dice** | 0.6216 | 0.6466 | **0.6466** (+2.50% gain, 90/90 wins) |



