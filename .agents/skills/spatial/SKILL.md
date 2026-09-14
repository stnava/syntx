---
name: spatial
description: >-
  Systematic physical-space coordinate utilities for syntx.landmarks.
  All physical↔voxel transforms, slice extraction, and keypoint display
  projection must use syntx.landmarks.spatial rather than ad-hoc code.
---

# Spatial Framework Skill

## Module: `syntx/landmarks/spatial.py`

All code that converts between physical mm and voxel indices, or extracts
display slices, MUST use this module. Do NOT copy affine math inline.

## Core API

```python
from syntx.landmarks.spatial import (
    get_image_affine,           # (origin, spacing, direction)
    vox_to_physical,            # [N,3] XYZ voxel -> [N,3] mm
    physical_to_vox,            # [N,3] mm -> [N,3] XYZ voxel (continuous)
    physical_offset_to_voxel,   # mm displacement (LPS) -> voxel displacement (no origin)
    voxel_gradient_to_physical, # per-axis finite differences -> LPS gradient (per mm)
    image_to_tensor,            # ANTsImage -> [1,1,nx,ny,nz] (XYZ layout kept)
    sample_tensor_at_physical,  # trilinear sample of [1,C,nx,ny,nz] at mm points -> [N,C]
    axis_orientation_code,      # e.g. 'LAS': anatomical direction each ARRAY axis points to
    ortho_view_spec,            # how to cut/arrange the array for axial/coronal/sagittal
    slice_from_spec,            # 2-D display array [n_v, n_u] for imshow(origin='lower')
    extract_ortho_slices,       # ax/cor/sag slices + 'views' specs (+ labels, aspects)
    project_to_slice,           # physical points -> (u, v, mask) on a view spec
    safe_whichtoinvert,         # ensures whichtoinvert matches len(transformlist)
    vox_zyx_to_physical,        # ONLY for tensors explicitly built as arr.transpose(2,1,0)
)
```

## Conventions
- Physical space is ITK **LPS**: +x = Left, +y = Posterior, +z = Superior.
- Image arrays are in ANTs XYZ layout: `arr[ix, iy, iz]`.
- **Tensors keep XYZ layout**: `image_to_tensor` gives `[1,1,nx,ny,nz]`, i.e. torch dims
  (2,3,4) == array axes (0,1,2). `np.argwhere(mask.squeeze())` therefore yields
  `(ix, iy, iz)` and must go through `vox_to_physical` — NEVER `vox_zyx_to_physical`.
  (The earlier "tensors are ZYX" claim swapped x and z and put keypoints in the padding.)
- `torch.nn.functional.grid_sample` grid coordinates are ordered `(W, H, D)` = `(iz, iy, ix)`
  for that layout; only `sample_tensor_at_physical` may build such grids.
- Scale parameters (`sigma_min`, `sigma_max`, MIND `offset_distance`) are **mm**; detectors
  derive per-axis voxel sigmas / integer shifts from spacing and direction.
- Descriptors (SIFT3D gradient bins, MIND offsets) are computed in physical LPS axes so images
  stored in different frames (LAS vs RPS arrays, permuted axes, anisotropic spacing) match.
- Display: axial Anterior UP, coronal/sagittal Superior UP, sagittal Anterior RIGHT;
  L/R follows `convention='radiological'` (default, patient Left on viewer's RIGHT) or
  `'neurological'`. Flips/permutations come from `dominant_axes(direction)`, never from
  hard-coded `D[i,i]` signs, and never by reorienting/resampling the array.
- Always pass `view=slices['views'][name]` from `extract_ortho_slices` into `project_to_slice`
  so the overlay uses exactly the flips the slice was drawn with.

## Forbidden Patterns
- `ants.reorient_image(...)` or `ants.reorient_image2(...)` — these are modules, not callables
- Ad-hoc `x_mm = w * sx` (ignores origin and direction matrix)
- `whichtoinvert=[False]` when `transformlist` has >1 entry (must match length)
- `detect_sift2d(...)` on 3D volumetric images — strictly use true 3D detectors (`detect_sift3d`, `detect_blobs_log`, `detect_blobs_dog`)
