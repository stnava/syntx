# Landmark Detection & Physical Space Framework Handoff

**Date**: September 14, 2026  
**Status**: RESOLVED (same day). All items in §4 were implemented and verified; see §0.

---

## 0. Resolution (what actually changed)

Review of the "standardised" torch↔spatial transforms found the handoff's own premise wrong:
`torch.from_numpy(image.numpy())` keeps the ANTs **XYZ** layout, so tensors are `[1,1,nx,ny,nz]`
and `np.argwhere` already returns `(ix, iy, iz)`. The `vox_zyx_to_physical` step (and the
"tensors are ZYX" convention in the skill file) was the x/z swap. In addition the package did not
import at all (`vox_zyx_to_physical`/`format_axis_xlabel` were referenced but never defined), the
slice extractor used `D[i,i]` signs with +x/+y interpreted as Right/Anterior (ITK is LPS, so both
were inverted), and MIND's `grid_sample` grid normalised each axis by the wrong size.

Fixes (all verified on siq phantoms stored as LPS / RAS / permuted / anisotropic frames, then on `mbhard`):

* `spatial.py` rewritten: LPS-correct orientation from `dominant_axes(direction)` (any permutation/flip),
  `ortho_view_spec` + `slice_from_spec` + `project_to_slice(view=...)` share one spec so overlays cannot
  disagree with the slice; `convention='radiological'|'neurological'`; `image_to_tensor`,
  `sample_tensor_at_physical` (the only `grid_sample` wrapper), `physical_offset_to_voxel`,
  `voxel_gradient_to_physical`, `axis_orientation_code`, `format_axis_xlabel`.
* `blob.py`: XYZ indices → `vox_to_physical`; foreground mask (`>0.05`); sigmas in **mm** with per-axis
  voxel sigmas and spacing-scaled Laplacian; NMS ranked by response (was array order).
* `sift3d.py`: same detector fixes plus relative response threshold; descriptor rebuilt in physical
  space (gradient rotated by `D`, mm sampling lattice, 8 fixed sphere directions, vectorised on GPU).
  Descriptors of the same anatomy stored in different frames are now bit-identical.
* `mind.py`: offsets defined in mm along LPS axes and converted to voxel shifts through `D`/spacing;
  extraction via `sample_tensor_at_physical`.
* **MPS `F.pad` corruption (found later the same day)**: on Apple MPS (torch 2.13) `F.pad` on a 5-D tensor
  whose trailing `H*W >= 65536` (any 256×256 slice) silently returns garbage along the padded depth dim. It
  zeroed the x-gradient and corrupted the LoG on full-size brains while every small phantom test passed
  (`blob._shift_pad` now does border handling with slicing/cat; regression test added). Symptom that exposed
  it: rotating the descriptor frame by the *known* rotation made same-subject descriptors *less* similar.
* Rotation handling (`syntx.landmarks.orient`): the axis-aligned descriptor tolerates ~30° of relative
  rotation; `match_sift3d_with_rotation_search` seeds a global rotation from the principal axes of the two
  landmark clouds (identity + 4 sign combinations), then iterates descriptors-in-frame → match → rigid RANSAC →
  rotation, coarse-to-fine over keypoint strength, and keeps the converged hypothesis with most inliers.
  Frame voting from per-keypoint structure-tensor frames (also available, `rotation_invariant=True`) is exact
  on phantoms but unreliable on real brains: cortical frames are correlated and wrong matches vote coherently.
  Closed-form batched 3×3 eigensolver `sift3d.sym3x3_eigh` (Cardano) keeps that path on-device.
* Preprocessing: N4 is now OFF by default (`use_n4=False`); ANTsTorch `denoise_image` exists since 2026-09-14
  (it was missing before, so earlier "N4+NLM" runs were N4 only).
* Tests: `tests/test_landmarks_spatial.py`, `tests/test_landmarks_simulated.py` (siq phantom; frame
  invariance, known rigid + nonrigid recovery, 60° rotation search, MPS border-op regression),
  `tests/test_landmarks_mbhard.py` (real data).
* Report: `scripts/landmarks_mbhard_report.py` → `reports/mbhard_spatial_report.html` (Example 1 native pair,
  Example 2 fixed +30° / moving −30° yaw; both via the rotation search, plain variants tabulated).

Sections 1–4 below are kept as the historical record of the investigation.

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
