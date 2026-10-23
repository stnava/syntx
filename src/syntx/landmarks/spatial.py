"""Coordinate conversion, tensor layout, and orthographic slice primitives for ``syntx.landmarks``.

Consolidated into :mod:`syntx.spatial` and re-exported here for landmark submodules.
"""

from __future__ import annotations

from ..spatial import (
    _LPS_LABELS,
    _VIEW_ALIASES,
    _VIEW_TABLE,
    _as_points,
    _resolve_convention,
    anatomical_axis_labels,
    axis_orientation_code,
    dominant_axes,
    extract_ortho_slices,
    format_axis_xlabel,
    get_image_affine,
    landmark_image_to_tensor as image_to_tensor,
    ortho_view_spec,
    physical_offset_to_voxel,
    physical_to_vox,
    project_to_slice,
    safe_whichtoinvert,
    sample_tensor_at_physical,
    slice_from_spec,
    vox_to_physical,
    vox_zyx_to_physical,
    voxel_gradient_to_physical,
)

__all__ = [
    "get_image_affine",
    "vox_to_physical",
    "vox_zyx_to_physical",
    "physical_to_vox",
    "physical_offset_to_voxel",
    "voxel_gradient_to_physical",
    "image_to_tensor",
    "sample_tensor_at_physical",
    "dominant_axes",
    "axis_orientation_code",
    "anatomical_axis_labels",
    "format_axis_xlabel",
    "ortho_view_spec",
    "slice_from_spec",
    "extract_ortho_slices",
    "project_to_slice",
    "safe_whichtoinvert",
]
