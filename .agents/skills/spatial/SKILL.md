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
    get_image_affine,       # (origin, spacing, direction)
    vox_to_physical,        # [N,3] XYZ voxel -> [N,3] mm
    vox_zyx_to_physical,    # [N,3] ZYX tensor voxel -> [N,3] mm
    physical_to_vox,        # [N,3] mm -> [N,3] XYZ voxel
    extract_ortho_slices,   # -> (ax, cor, sag) slices + centers + labels
    project_to_slice,       # physical points -> 2D display coords (u, v, mask)
    safe_whichtoinvert,     # ensures whichtoinvert matches len(transformlist)
)
```

## Conventions
- Image arrays are in ANTs XYZ layout: `arr[ix, iy, iz]`
- Tensors are in ZYX layout: `[1,1,D,H,W]` = `[1,1,iz,iy,ix]`
- `vox_to_physical` takes XYZ indices (`arr` space)
- `vox_zyx_to_physical` takes ZYX indices (`torch.Tensor` / `argwhere` space)
- Display slices: `arr[:,:,iz].T` -> row=iy, col=ix (NO array reorientation)
- Axis labels from physical direction matrix, not from hardcoded strings

## Forbidden Patterns
- `ants.reorient_image(...)` or `ants.reorient_image2(...)` — these are modules, not callables
- Ad-hoc `x_mm = w * sx` (ignores origin and direction matrix)
- `whichtoinvert=[False]` when `transformlist` has >1 entry (must match length)
- `detect_sift2d(...)` on 3D volumetric images — strictly use true 3D detectors (`detect_sift3d`, `detect_blobs_log`, `detect_blobs_dog`)
