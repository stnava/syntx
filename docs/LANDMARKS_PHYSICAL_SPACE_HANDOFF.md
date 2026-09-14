# Landmark Detection & Physical Space Framework Handoff

**Date**: September 14, 2026  
**Status**: Critical fixes identified & canonical spatial framework implemented in `src/syntx/landmarks/spatial.py`. Ready for final integration and report regeneration.

---

## 1. Executive Summary & Context

The user requested a modality- and anatomy-independent landmark detection, matching, and physical-space visualization suite in `syntx.landmarks`. During evaluation on Mindboggle Hard (`mbhard_pair00`, 3D T1-weighted brain MRI), the user flagged two critical regressions in `mbhard_spatial_report.html`:
1. **"points do not align on images"**: Landmarks appeared scrambled or floating in background padding.
2. **"display does not respect medical standards / physical space"**: Images were displayed in inverted anatomical orientations and match lines criss-crossed across hemispheres.
3. **"you are fucking something up if n4+denoise takes a minute -- are you using antstorch on mps?"**: Preprocessing was taking ~55 seconds because it was invoking CPU single-threaded ITK C++ N4 instead of GPU/MPS ANTsTorch.
4. **"dont use sift2d on 3d data"**: Strictly prohibit slice-wise 2D SIFT on 3D volumetric images; use only true 3D volumetric detectors (`detect_sift3d`, `detect_blobs_dog`, `detect_blobs_log`, and volumetric MIND-SSC).

---

## 2. Work Completed in This Session

### A. GPU/MPS Accelerated Preprocessing (`src/syntx/landmarks/preprocess.py`)
- **Commit**: `6500d39`
- Replaced CPU `ants.n4_bias_field_correction` with `antstorch.n4_bias_field_correction` on GPU/MPS.
- Added device auto-detection (`mps` > `cuda` > `cpu`).
- Explicitly passed `device=dev` to `antstorch.denoise_image`.
- **Benchmark Speedup**: Dropped preprocessing runtime from **55s+** down to **9.4s** on Apple Silicon MPS.

### B. Canonical Spatial Framework (`src/syntx/landmarks/spatial.py`)
- **Commits**: `c9b7b88` & subsequent updates.
- Centralized all physical-to-voxel and voxel-to-physical coordinate transforms to prevent ad-hoc math:
  * `get_image_affine(image)`: Returns `(origin, spacing, direction)`.
  * `vox_to_physical(image, indices_xyz)`: Vectorized mapping $\mathbf{x}_{\text{phys}} = \mathbf{o} + (\mathbf{i}_{\text{XYZ}} \odot \mathbf{s}) \mathbf{D}^T$. Matches `ants.transform_index_to_physical_point` to within $10^{-5}\text{ mm}$.
  * `physical_to_vox(image, points_mm)`: Vectorized inverse $\mathbf{i}_{\text{XYZ}} = (\mathbf{x}_{\text{phys}} - \mathbf{o}) \mathbf{D}^{-T} \oslash \mathbf{s}$. Matches `ants.transform_physical_point_to_index`.
  * `safe_whichtoinvert(transformlist, invert_flags)`: Enforces `len(whichtoinvert) == len(transformlist)`.
  * `extract_ortho_slices(image, center_mm)`: Extracts Axial, Coronal, and Sagittal slices dynamically oriented to Medical Viewing Standards (Axial: Anterior UP, Posterior DOWN, Left on Left; Coronal: Superior UP, Inferior DOWN, Left on Left; Sagittal: Superior UP, Inferior DOWN, Anterior RIGHT) without file-based reorientation.
  * `project_to_slice(image, points_mm, slice_axis, slice_pos_mm, slab_half_mm, flip_u, flip_v)`: Exact projection of 3D physical coordinates onto 2D display slices, handling axis flips.

### C. Invariants Added to `GEMINI.md` and `.agents/skills/spatial/SKILL.md`
1. **Strict Ban on `ants.reorient_image` & `ants.reorient_image2`**: These are Python modules, not callables.
2. **Strict Ban on `sift2d` on 3D Volumetric Data**: Only 3D volumetric scale-space detectors are permitted on 3D data.
3. **`whichtoinvert` Length Invariant**: Must match `len(transformlist)`.

---

## 3. Deep Root Causes Identified & Debugged

### Root Cause 1: X and Z Coordinate Swapping in Detectors
- **Location**: `src/syntx/landmarks/blob.py` line 205 and `src/syntx/landmarks/sift3d.py` line 282.
- **Mechanism**: `_to_tensor` converted `image.numpy()` (shape $(nx, ny, nz)$ in ANTs order $(X, Y, Z)$) to a PyTorch tensor.
  `idxs = np.argwhere(mask_np)` produced indices in $(ix, iy, iz)$ order.
  The code then called `vox_zyx_to_physical`, which assumed indices were $(iz, iy, ix)$ and reversed them with `[::-1]`.
  This swapped $ix$ and $iz$. For the moving image (shape $170 \times 256 \times 256$), $iz$ coordinates (up to 256) were placed into $ix$, causing 14+ points to have $ix > 170$ (out of bounds) and placing points in empty background padding.
- **Fix**: Call `vox_to_physical((origin, spacing, direction), idxs)` directly without reversing, and apply foreground masking `& (vol > 0.05)` to eliminate background artifacts.

### Root Cause 2: Native Coordinate Frame Divergence in `mbhard`
- **Fixed Image**: LPS direction matrix ($\mathbf{D}_{00}=+1, \mathbf{D}_{11}=-1, \mathbf{D}_{22}=+1$), COM $\approx (0, -4, 19)\text{ mm}$.
- **Moving Image**: RAS direction matrix ($\mathbf{D}_{00}=-1, \mathbf{D}_{11}=+1, \mathbf{D}_{22}=+1$), COM $\approx (100, 132, 132)\text{ mm}$.
- **Mechanism**: When raw arrays were displayed side-by-side, Fixed had Left on Left, but Moving had Left on Right! Moving was horizontally mirrored relative to Fixed, and vertically inverted. Match lines connecting homologous points therefore crossed diagonally across the skull.
- **Fix**: `spatial.extract_ortho_slices` and `project_to_slice` dynamically account for direction matrix signs to present both images in canonical medical viewing orientation (Left on Left, Anterior UP).

### Root Cause 3: MIND-SSC 5D `grid_sample` Coordinate Mapping
- **Location**: `src/syntx/landmarks/mind.py` lines 219-232.
- **Mechanism**: PyTorch `F.grid_sample` in 5D expects `grid` coordinates $(gx, gy, gz)$ mapping to dimensions $(W, H, D) = (nz, ny, nx)$. `mind.py` was normalizing $ix$ with $nz$ and $iz$ with $nx$.
- **Fix**: Map $z_{\text{norm}}$ to dim 0, $y_{\text{norm}}$ to dim 1, and $x_{\text{norm}}$ to dim 2 of the grid tensor: `grid = [z_norm, y_norm, x_norm]`.

---

## 4. Immediate Next Steps for the Next LLM

1. **Update `src/syntx/landmarks/blob.py`**:
   - In `_scale_space_extrema`:
     * Add foreground masking: `is_local_max = is_local_max & (curr_raw.abs() > abs_thresh) & (scale_images_raw[0] > 0.05)` (or pass raw normalized volume).
     * Replace `phys = _vox_to_physical(idxs, origin, spacing, direction)` with:
       ```python
       from syntx.landmarks.spatial import vox_to_physical
       phys = vox_to_physical((origin, spacing, direction), idxs)
       ```
2. **Update `src/syntx/landmarks/sift3d.py`**:
   - Line 275: Add `& (vol > 0.05)`.
   - Line 282: Replace `vox_zyx_to_physical` with `vox_to_physical((origin, spacing, direction), idxs)`.
   - Lines 56-62 (`_gradient_3d`): Ensure $gx$ is along dim 2, $gy$ along dim 3, $gz$ along dim 4.
3. **Update `src/syntx/landmarks/mind.py`**:
   - In `extract_mind_at_points`:
     ```python
     x_norm = (idx_xyz[:, 0] / (mind_vol.shape[2] - 1)) * 2 - 1
     y_norm = (idx_xyz[:, 1] / (mind_vol.shape[3] - 1)) * 2 - 1
     z_norm = (idx_xyz[:, 2] / (mind_vol.shape[4] - 1)) * 2 - 1
     grid = torch.from_numpy(np.stack([z_norm, y_norm, x_norm], axis=-1).astype(np.float32)).to(dev).view(1, 1, 1, -1, 3)
     ```
4. **Regenerate the HTML Benchmark Report**:
   - Script target: `mbhard_spatial_report.html`.
   - Use `extract_ortho_slices` and `project_to_slice` from `syntx.landmarks.spatial`.
   - Verify that all match lines in Figure 3 run horizontally across homologous brain structures without flipping or criss-crossing.
5. **Run Verification Suite**:
   ```bash
   pytest -v tests/test_landmarks_spatial.py
   ```
