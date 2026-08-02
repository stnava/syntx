# Syntx Visualization Suite

The `syntx` Visualization Suite provides publication-grade, standardized visualization tools for 2D and 3D medical image registration.

---

## 1. Figure 1: Standard Input Pair Visualization (`render_input_pair_figure`)

`syntx.render_input_pair_figure()` is the foundational **Figure 1** tool for visualizing input images before registration.

### Layout Invariants

* **3D Volume Inputs**:
  - **Grid**: $2 \times 3$ panel layout within one single figure.
  - **Top Row**: **Fixed Target Image** (Axial, Coronal, Sagittal orthographic views).
  - **Bottom Row**: **Moving Source Image** (Axial, Coronal, Sagittal orthographic views).
  - **Row Labels**: Explicit `FIXED (Top)` and `MOVING (Bottom)` margin indicators.

* **2D Image Inputs**:
  - **Grid**: $1 \times 2$ panel layout within one single figure.
  - **Left**: **Fixed Target Image**.
  - **Right**: **Moving Source Image**.

### Usage Example

```python
import syntx
import ants

# Load 3D volumes
fixed = ants.image_read("fixed_brain.nii.gz")
moving = ants.image_read("moving_brain.nii.gz")

# Generate Figure 1 (3D: Fixed Top / Moving Bottom)
fig = syntx.render_input_pair_figure(
    fixed=fixed,
    moving=moving,
    output_path="figures/figure1_input_pair.png",
    title="Figure 1: Original Pre-Registration Input Images"
)
```

---

## 2. Figure 2: Standardized 4-Panel Registration Quality (`render_standard_4panel`)

`syntx.render_standard_4panel()` is the standard **Figure 2** tool for evaluating registration quality.

### 4-Panel Layout Invariants
- **Panel A**: Deformed Mesh Grid Overlay (Cyan grid lines).
- **Panel B**: Divergent Jacobian Determinant Map (`seismic` colormap centered at 1.0).
- **Panel C**: Physical Inverse Identity Displacement Error Map ($\text{mm}$, `hot` colormap).
- **Panel D**: High-Contrast Canny Edge Alignment Overlap (Cyan/Magenta edges).

```python
import syntx

# Render Figure 2 at superior cortical slice 80
fig = syntx.render_standard_4panel(
    fixed=fixed,
    moving=moving,
    warped=reg['warpedmovout'],
    warp=reg['fwdtransforms'][0],
    slice_idx=80,
    output_path="figures/figure2_4panel.png"
)
```

---

## 3. Interactive Verification HTML Report (`create_registration_report`)

`syntx.create_registration_report()` generates a standalone HTML verification document combining Figure 1, Figure 2, similarity metrics (LNCC, Dice, MSE), Jacobian statistics, and full engine provenance.

```python
report = syntx.create_registration_report(
    fixed=fixed,
    moving=moving,
    reg=reg,
    output_html="reports/registration_verification.html",
    title="SyN Registration Verification Report"
)
```
