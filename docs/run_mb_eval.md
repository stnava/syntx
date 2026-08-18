# Reproducible Mindboggle-101 Deformable Registration Benchmark Tutorial

This tutorial provides step-by-step instructions to set up **`syntx`** from scratch on a clean machine (NVIDIA CUDA GPU, Apple Silicon MPS, or CPU), download and organize the **Mindboggle-101** benchmark dataset, generate a publication-quality standard diagnostic visual report (on `mbhard`), and execute the full 90-pair reproducible deformable registration evaluation.

---

## 1. Prerequisites & Environment Setup (CUDA GPU / Apple Silicon / CPU)

### 1.1 Create and Activate a Fresh Environment

We recommend Python 3.10 or 3.11 with Conda or Python `venv`:

```bash
# Using Conda
conda create -n syntx_cuda python=3.11 -y
conda activate syntx_cuda

# Or using Python venv
python3.11 -m venv ~/syntx_env
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

### 2.3 Set the Dataset Environment Variable

Export the path in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export SYNTX_DATA_DIR="/path/to/mindboggle/volumes"
```

### 2.4 Verify Dataset Integrity in One Command

Run the built-in integrity check to verify that all 90 pairs (40 intra-subject and 50 inter-subject defined in `examples/pairs.csv`) exist:

```bash
python -m syntx.benchmark --check-data
```

**Expected output:**
```
[syntx.benchmark] Dataset verified successfully! All 90 pairs ready.
```

---

## 3. Quick Demo: Generating a Standard Diagnostic Report on `mbhard`

The `mbhard` dataset is the canonical hard 3D inter-subject test pair (`NKI-TRT-20-2` fixed $\rightarrow$ `MMRR-21-2` moving).

### 3.1 Run via CLI (One Line)

```bash
python -m syntx.benchmark --demo --demo-dataset mbhard --demo-html docs/reports/mbhard_standard_report.html
```

### 3.2 Run via Python API (3 Lines)

```python
import syntx
from syntx.benchmark import run_standard_report_demo

# Run deformable SyN and render publication-grade HTML diagnostic report
report_path = run_standard_report_demo(
    dataset_key="mbhard",
    output_html="docs/reports/mbhard_standard_report.html",
    model="gaussian",  # or 'sobolev'
    verbose=True
)
print(f"Report generated: {report_path}")
```

### 3.3 What the Standard Diagnostic Report Contains

Open `docs/reports/mbhard_standard_report.html` in any web browser to view:
- **Header Summary Card:** Bidirectional Cortical DKT31 Mean Dice ($\text{Dice}_{\text{sym}}$), Jacobian singularity metrics, inverse error, and total GPU execution time.
- **Figure 1 (Input Pair):** Orthographic slice panel of Fixed Target and Moving Source volumes.
- **Figure 2 (Standard 4-Panel Diagnostic):**
  - **Panel A (Mesh Grid):** Deformed coordinate grid (Cyan) showing smooth coordinate displacement.
  - **Panel B (Jacobian Determinant):** Divergent $\log\det(J)$ map (`seismic` colormap centered at 1.0) highlighting local volume expansion vs compression.
  - **Panel C (Inverse Error Map):** Real physical inverse identity error $\|\phi_{\text{inv}}(\mathbf{x} + \phi(\mathbf{x})) + \phi(\mathbf{x})\|_2$ in mm.
  - **Panel D (Edge Alignment Overlap):** High-contrast Canny edge alignment (Cyan fixed contours vs Magenta warped source contours).
- **Interactive Provenance Card:** Full record of optimization learning rate, fluid/elastic smoothing parameters ($\sigma_{\text{flow}}=3.0, \sigma_{\text{total}}=0.0$), multi-resolution iterations, and GPU hardware device.

---

## 4. Running the Full Deformable Benchmark from Scratch

The standardized benchmark executes 90 image pairs:
- **Rows 0–39 (40 Pairs):** Intra-subject longitudinal test-retest pairs.
- **Rows 40–89 (50 Pairs):** Inter-subject cross-individual pairs.

### 4.1 Run the Full 90-Pair Cohort

Run the entire cohort benchmark using either Gaussian regularized SyN (peak accuracy standard) or Sobolev regularized SyN (smooth topology-preserving standard):

```bash
# Option A: Gaussian Regularized SyN (88/90 Wins vs ANTs C++ SyN, +1.66% Dice)
python -m syntx.benchmark --cohort --model gaussian

# Option B: Sobolev Regularized SyN (81/90 Wins vs ANTs C++ SyN, 90.0% Zero-Fold)
python -m syntx.benchmark --cohort --model sobolev

# Option C: Evaluate Both Gaussian and Sobolev on Every Pair
python -m syntx.benchmark --cohort --model both
```

### 4.2 Run Specific Pairs or Subsets

Evaluate individual pairs or subsets:

```bash
# Evaluate Pair 0 (Gaussian SyN)
python -m syntx.benchmark --pair-idx 0 --model gaussian

# Evaluate a custom subset of pairs
python -m syntx.benchmark --cohort --pairs 0 1 2 45 67 82 --model gaussian
```

### 4.3 Generate Interactive Master Population Reports

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

## 5. Accent on Strict Scientific Reproducibility

To ensure 100% deterministic reproducibility across diverse hardware backends:

1. **Deterministic Regular Affine Sampling (`syntx.robust_affine`):**
   - Uses deterministic uniform grid sampling (`sampling_strategy='regular'`, 20% sample) and foreground union-masked Mutual Information ($\text{mask} = (I > 0.01) \mid (J > 0.01)$), avoiding stochastic sampling noise.
2. **Single Interpolation Policy:**
   - Input images are never pre-warped prior to optimization. Forward non-linear warps and affine matrices are composed and applied to native-space segmentation maps in a single nearest-neighbor step (`interpolator='nearestNeighbor'`).
3. **Bidirectional Fixed & Moving Space Overlap Evaluation:**
   - DKT31 Cortical Dice is systematically evaluated symmetrically in both coordinate spaces:
     $$\text{Dice}_{\text{sym}} = \frac{1}{2} \left( \text{Dice}_{\text{fixed}} + \text{Dice}_{\text{moving}} \right)$$
4. **Subprocess Worker Isolation & Device Cache Cleansing:**
   - Each benchmark case executes in an isolated Python worker process (`syntx.benchmark.worker`), automatically releasing GPU allocator cache (`torch.cuda.empty_cache()` / `torch.mps.empty_cache()`) to guarantee zero memory fragmentation across serial evaluations.

---

## 6. CUDA GPU Performance Expectations

| Metric | ANTs C++ SyN (CPU) | Syntx (Apple Silicon MPS) | Syntx (NVIDIA RTX 4090 / A100 CUDA) |
|:---|:---|:---|:---|
| **Affine Registration Time** | ~28.5 s | ~2.8 s | **~1.2 s** ($24\times$ speedup) |
| **Deformable SyN Time (per 3D Pair)** | ~85–120 s | ~24–28 s | **~12–16 s** ($7.5\times$ speedup) |
| **GPU VRAM Footprint** | N/A (RAM: ~3 GB) | ~3.8 GB Unified | **~3.5–4.2 GB VRAM** |
| **90-Pair Cohort Total Time** | ~2.5–3.0 hours | ~40 minutes | **~20–25 minutes** |
| **Mean Cortical DKT31 Dice** | 0.6216 | 0.6382 | **0.6382** (+1.66% gain, 88/90 wins) |
