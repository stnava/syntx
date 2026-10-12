# Physical-Space Probe Displacement: `syntx.pose`

## Overview

In multi-modality registration and thin-slab alignment (such as registering 11-slice neuromelanin MRI slabs to whole-brain T1 structural images), intensity-based metrics such as mutual information (MI) can fail to distinguish correct alignments from catastrophic misalignment. For instance, a truncated slab may capture high local contrast in an incorrect anatomical region or at an invalid tilt, yielding plausible MI values despite multi-centimeter localization errors.

`syntx.pose` provides a rigorous, coordinate-aware physical displacement metric:
$$\text{pose\_error}(T_A, T_B) = \max_{i} \| \text{disp}_A(q_i) - \text{disp}_B(q_i) \| \quad (\text{in mm})$$
evaluated across a standardized set of probe points $q_i$ positioned in physical space around the center of mass of the fixed reference image.

---

## Coordinate & Transformation Conventions

### ITK Fixed $\rightarrow$ Moving Mapping
In ITK and ANTs, a forward spatial transformation $T$ maps physical coordinates from the fixed image space to the moving image space:
$$p_{\text{moving}} = T(p_{\text{fixed}})$$
Physical points are expressed in millimeters (mm) following the standard ITK physical coordinate system (LPS).

### Probe Displacement
The displacement vector of a probe point $q$ in fixed physical space under transform $T$ is:
$$\text{disp}(q) = T(q) - q$$
When comparing two candidate transformations $T_A$ and $T_B$ for the same fixed space:
$$\text{disp}_A(q) - \text{disp}_B(q) = (T_A(q) - q) - (T_B(q) - q) = T_A(q) - T_B(q)$$
The difference in displacement equals the Euclidean distance between the locations in moving space to which $T_A$ and $T_B$ map probe point $q$.

The overall pose error between two registrations is defined as the maximum Euclidean displacement difference across all probe points:
$$\text{pose\_error} = \max_{i} \| T_A(q_i) - T_B(q_i) \|$$

---

## Probe Point Geometries

`probe_points(fixed, kind='default')` generates physical probe points centered on $c = \text{ants.get_center_of_mass}(\text{fixed} \times 0 + 1)$:

1. **`kind='default'`**:
   - 8 corner probes placed at $c + (\pm 15, \pm 15, \pm 5)\text{ mm}$.
   - 1 central probe placed at $c$.
   - Total: 9 probe points.
   - Ideal for neuromelanin brainstem slabs and general anatomical structures.

2. **`kind='slab_thick'`**:
   - Expands the $z$ lever arm up to 1.5 slices from the outer slab boundary:
     $$dz = \max\left(5.0, \left(\frac{\text{shape}[2]}{2} - 1.5\right) \times \text{spacing}[2]\right)$$
   - Provides enhanced sensitivity to out-of-plane rotation across thicker slabs.

3. **Custom configurations**:
   - A single float specifies custom $dz$ (with $dx=15, dy=15\text{ mm}$).
   - A 3-tuple `(dx, dy, dz)` specifies custom offsets along each axis in millimeters.

---

## API Reference

### `syntx.probe_points(fixed, kind='default') -> np.ndarray`
Generates an $(N, 3)$ array of physical probe coordinates in millimeters.

### `syntx.pose_error(fixed, tx_a, tx_b, probes=None) -> float`
Calculates $\max_i \| \text{disp}_A(q_i) - \text{disp}_B(q_i) \|$ in millimeters.
Accepts ANTsTransform objects, transform file paths (`.mat`, `.txt`), or $4 \times 4$ / $3 \times 4$ / $3 \times 3$ affine matrices.

### `syntx.pose_error_report(fixed, tx_a, tx_b, probes=None) -> PoseErrorReport`
Computes detailed displacement statistics, returning a `PoseErrorReport` (a dictionary subclass) containing:
- `max_error_mm`: Maximum displacement difference across all probes (mm).
- `mean_error_mm`: Mean displacement difference across all probes (mm).
- `z_mean_error_mm`: Mean absolute difference along the $z$-axis (mm).
- `probe_errors_mm`: Array of per-probe displacement error magnitudes.
- `probes`: Array of probe coordinates.
- `displacements_a`: Displacement vectors under $T_A$.
- `displacements_b`: Displacement vectors under $T_B$.
- `worst_probe_index`: Index of the probe point with maximum error.
- `worst_probe`: Coordinates of the worst probe point.

### `syntx.probe_displacements(fixed, tx, probes=None) -> np.ndarray`
Computes displacement vectors $\text{disp}(q_i) = T(q_i) - q_i$ for each probe point in millimeters.

---

## Example Usage

```python
import ants
import syntx

fixed = ants.image_read("fixed_nm.nii.gz")
moving = ants.image_read("moving_t1.nii.gz")

# Perform registration
reg = syntx.robust_affine(fixed, moving, dof="rigid")
tx_solved = reg["fwdtransforms"][0]
tx_reference = "ants_rigid_reference.mat"

# Scalar pose error (mm)
err_mm = syntx.pose_error(fixed, tx_solved, tx_reference)
print(f"Pose error: {err_mm:.2f} mm")

# Detailed per-probe report
rep = syntx.pose_error_report(fixed, tx_solved, tx_reference)
print(rep.summary())
if rep.max_error_mm > 3.0:
    raise ValueError(f"Pose disagreement: {rep.max_error_mm:.2f} mm exceeds 3.0 mm threshold")
```
