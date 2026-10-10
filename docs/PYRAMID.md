# The central pyramid: one contract

Status: specification (Pass 0). Nothing here is implemented in `src/syntx/pyramid.py` yet; `docs/PYRAMID_EVIDENCE.md` holds the
comparison that decides the open items. Each item is marked **[decided]** or **[open: Pass 1 evidence]**.

syntx currently has about ten independent pyramid implementations (`pyramid.py` x2, `greedy`, `tvf_adj`, three JAX copies,
`robust_affine` x3, `motion_batched` x3, `scattered/solver`, `imaging_utils.cap_resolution_for_registration`). They disagree about
floors (4, 8, 16), smoothers (three), downsampling operators (avg_pool, `F.interpolate` with align_corners, area/matrix, ANTs
resample) and thin axes. The contract below replaces them.

## 1. Inputs
- A torch tensor `(B, C, *spatial)` in tensor order (z, y, x) plus `spacing`, `origin`, `direction` in ANTs (x, y, z) order.
  Several images may share one call (a `B` batch on one grid). **[decided]**
- An ANTs image through a thin wrapper (`image_to_tensor` in, the same per-level geometry out). The wrapper owns the
  ANTs-resample paths (`cap_resolution_for_registration`, the low-resolution ANTs candidates) so that they use linear
  interpolation and the same planner. **[decided]**

## 2. Levels
- `levels` stay integer shrink factors, e.g. `[8, 4, 2, 1]`, user-facing and unchanged. **[decided]**
- Meaning: the working spacing of level L is `L x (finest spacing of the fixed image)`, isotropic in millimetres.
  For an isotropic image this is the same grid as shrinking every axis by L. **[decided]**
- A moving image is leveled on its own grid with its own spacing and the same target in mm (never below its native spacing);
  it is not forced onto the fixed grid. **[decided]**

## 3. Per-axis rule (thin and thick axes)
Planned per axis from `(n, spacing, target_mm)`:
- coarsened axis (`target > spacing`): area/low-pass average; working voxel j covers full-resolution indices `[j f, (j+1) f)`,
  `f = n / m`, `m = max(1, round(n spacing / target))`. **[decided]**
- non-coarsened axis (`target <= spacing`): the axis is not smoothed. Whether it is kept native or resampled to the target
  (centre-to-centre linear) at coarse levels, and at level 1, is the central open question (policies P4 vs P5). **[open]**
- An axis never ends with fewer than 2 samples; a request that would do so raises `PyramidLevelError`, it does not clamp
  silently or return a zero-size tensor. **[decided]**

## 4. One floor
- One module constant for the minimum voxels per axis, applied by the planner to the *coarsest usable level*, not per axis
  by up-sampling. Its value (8 voxels as in the affine solvers, 16 as in greedy, or a millimetre extent) and whether it is
  counted in finest-spacing voxels of the shortest field of view are **[open]**. Duplicated floors elsewhere are deleted. **[decided]**

## 5. Smoothing
- Gaussian, per-axis sigma, given in millimetres or in voxels, applied only through `core/smoothing.separable_gaussian_filter`
  (replicate padding, axes with sigma <= 0 skipped). **[decided]**
- Never on an axis that is not coarsened; never NaN at sigma 0 (a zero-sigma axis is skipped, not a zero-width kernel). **[decided]**
- Which sigma rule is the default (none beyond the area average, `log2(L)` voxels, `0.5 n/m`) **[open]**.

## 6. Operator presets
Named, so that migrating a caller cannot change its numbers by accident (golden-output test per caller): **[decided]**
`legacy_avgpool` (voxel `avg_pool`, robust_affine/motion), `legacy_interp` (Gaussian `log2(L)` + `F.interpolate`
align_corners, syn/syngs/tvf), `legacy_floor16` (compact Gaussian + floor 16 + align_corners, greedy), `mm_isotropic`
(`_mm_pyramid_level`). The default preset for each use case (affine/motion, deformable) is **[open: Pass 1 evidence]**.
JAX uses the same planner; only the array ops differ.

## 7. Per-level outputs
For each level: the tensor `(B, C, *shape)`; per axis (x, y, z) the factor `f` and offset `off` such that working index j sits
at full-resolution continuous index `j f + off` (voxel centres; `f = L`, `off = (L-1)/2` for pooling, `f = (n-1)/(m-1)`,
`off = 0` for corner-aligned operators); the working `spacing`, `origin` and `shape`; a helper mapping working indices to
physical coordinates; and one shared `warp_to_moving_norm(y_phys, level_entry)` returning normalised grid coordinates of the
(possibly differently leveled) moving tensor. **[decided]**

## 8. Masks, margin
- A mask is leveled with the image's geometry and a linear (not nearest) operator, then thresholded by the caller. **[decided]**
- A valid-region margin excluding the outermost slab-edge slice(s) of the fixed image from the metric is a per-level boolean
  returned with the level. Whether it is on by default **[open: Pass 1 evidence]**.
