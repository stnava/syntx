"""
syntx.landmarks.spatial — Canonical Physical-Space Coordinate and Display Framework
==================================================================================

This module provides the single source of truth for landmark coordinate mapping
and standardized medical orientation displays within syntx.landmarks.

It builds directly on the conventions established in `syntx.spatial` and ANTsPy:
    - Physical coordinates: mm in scanner space (ITK / ANTs convention)
    - Mapping from voxel XYZ (ix, iy, iz) to physical mm (x, y, z):
          physical = origin + (index_XYZ * spacing) @ direction.T
    - Vectorized parity with `ants.transform_index_to_physical_point` and
      `ants.transform_physical_point_to_index`.
    - Medical Display Standards:
        * Axial: Anterior (Front of Head) UP, Posterior DOWN; Patient Left on Viewer's Left.
        * Coronal: Superior (Top of Head) UP, Inferior DOWN; Patient Left on Viewer's Left.
        * Sagittal: Superior UP, Inferior DOWN; Anterior on Viewer's RIGHT.
    - Zero intermediate file-based or array-resampling reorientation: native arrays
      are preserved, and directional viewing orientation is handled by dynamic
      orthogonal slice extraction and index mapping.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any
import ants


# ---------------------------------------------------------------------------
# Affine and Coordinate Mapping
# ---------------------------------------------------------------------------

def get_image_affine(image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the full ANTs image affine components.

    Parameters
    ----------
    image : ants.ANTsImage | tuple(origin, spacing, direction)

    Returns
    -------
    origin    : np.ndarray [3] float64 (ox, oy, oz)
    spacing   : np.ndarray [3] float64 (sx, sy, sz)
    direction : np.ndarray [3, 3] float64
    """
    if isinstance(image, tuple) and len(image) == 3:
        org, sp, D = image
        return (
            np.asarray(org, dtype=np.float64),
            np.asarray(sp, dtype=np.float64),
            np.asarray(D, dtype=np.float64).reshape(3, 3),
        )
    if hasattr(image, "spacing"):
        sp = np.array(image.spacing, dtype=np.float64)
        org = np.array(image.origin, dtype=np.float64)
        D = np.array(image.direction, dtype=np.float64).reshape(3, 3)
        if sp.size == 2:
            sp = np.append(sp, 1.0)
            org = np.append(org, 0.0)
            D = np.eye(3, dtype=np.float64)
        return org[:3], sp[:3], D
    return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64), np.eye(3, dtype=np.float64)


def vox_to_physical(image, indices_xyz: np.ndarray) -> np.ndarray:
    """
    Convert ANTs/ITK voxel indices (ix, iy, iz) to physical mm coordinates.

        physical = origin + (index_XYZ * spacing) @ direction.T

    Matches `ants.transform_index_to_physical_point` bit-for-bit.

    Parameters
    ----------
    image       : ANTsImage or (origin, spacing, direction)
    indices_xyz : [N, 3] or [3] float — (ix, iy, iz) ANTs array indices

    Returns
    -------
    [N, 3] float32 — (x_mm, y_mm, z_mm) in physical scanner space
    """
    org, sp, D = get_image_affine(image)
    idx = np.asarray(indices_xyz, dtype=np.float64)
    if idx.ndim == 1:
        idx = idx[np.newaxis]
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    physical = org + (idx * sp) @ D.T
    return physical.astype(np.float32)


def physical_to_vox(image, points_mm: np.ndarray) -> np.ndarray:
    """
    Convert physical mm coordinates (x, y, z) to ANTs/ITK voxel indices (ix, iy, iz).

        index_XYZ = ((physical - origin) @ inv(direction).T) / spacing

    Matches `ants.transform_physical_point_to_index` bit-for-bit.

    Parameters
    ----------
    image     : ANTsImage or (origin, spacing, direction)
    points_mm : [N, 3] or [3] float — (x_mm, y_mm, z_mm)

    Returns
    -------
    [N, 3] float64 — (ix, iy, iz) voxel indices (fractional)
    """
    org, sp, D = get_image_affine(image)
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    D_inv = np.linalg.inv(D)
    idx = ((pts - org) @ D_inv.T) / sp
    return idx


# ---------------------------------------------------------------------------
# Medical Standard Orthographic Slice Extraction
# ---------------------------------------------------------------------------

def extract_ortho_slices(image, center_mm: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Extract axial, coronal, and sagittal 2D slices strictly following
    Medical Viewing Standards:
      - Axial: Anterior UP, Posterior DOWN; Patient Left on Viewer's Left.
      - Coronal: Superior UP, Inferior DOWN; Patient Left on Viewer's Left.
      - Sagittal: Superior UP, Inferior DOWN; Anterior on Viewer's RIGHT.

    Native arrays are preserved without file-based reorientation. Flips are
    applied deterministically from the direction cosine matrix.

    Returns
    -------
    dict with keys:
      'ax', 'cor', 'sag'           : 2D numpy arrays oriented for display with origin='lower'
      'aspect_ax', 'aspect_cor', 'aspect_sag' : physical pixel aspect ratio (height / width)
      'center_vox'                 : [3] int (ix, iy, iz) in native ANTs array
      'center_mm'                  : [3] float32 physical coordinate of the cut
      'flip_ax', 'flip_cor', 'flip_sag' : tuple(flip_u, flip_v) boolean flags for keypoint mapping
    """
    arr = image.numpy()  # [nx, ny, nz]
    org, sp, D = get_image_affine(image)
    nx, ny, nz = arr.shape

    if center_mm is None:
        com = ants.get_center_of_mass(image)
        center_mm = np.array(com, dtype=np.float64)

    ctr_vox = physical_to_vox(image, center_mm)[0]
    ix = int(np.clip(round(ctr_vox[0]), 0, nx - 1))
    iy = int(np.clip(round(ctr_vox[1]), 0, ny - 1))
    iz = int(np.clip(round(ctr_vox[2]), 0, nz - 1))

    # Determine direction signs for each axis:
    # Axis 0 (X): positive means Left -> Right
    dir_x_pos = (D[0, 0] > 0)
    # Axis 1 (Y): positive means Posterior -> Anterior
    dir_y_pos = (D[1, 1] > 0)
    # Axis 2 (Z): positive means Inferior -> Superior
    dir_z_pos = (D[2, 2] > 0)

    # 1. Axial Slice (plane X-Y, fixed iz):
    # Base: arr[:, :, iz] has shape (nx, ny).
    # Horizontal axis is X (ix), vertical axis is Y (iy).
    # Matplotlib origin='lower': row 0 is at bottom, row ny-1 is at top.
    # To have Anterior UP: if dir_y_pos, iy increases towards Anterior (already at top);
    #                      if not dir_y_pos, iy decreases towards Anterior (flip iy).
    flip_ax_u = not dir_x_pos
    flip_ax_v = not dir_y_pos

    ax_raw = arr[:, :, iz]
    if flip_ax_u:
        ax_raw = ax_raw[::-1, :]
    if flip_ax_v:
        ax_raw = ax_raw[:, ::-1]
    # Transpose so col=ix (horizontal, Left->Right), row=iy (vertical, Posterior->Anterior)
    ax_disp = ax_raw.T  # shape (ny, nx)
    aspect_ax = sp[1] / (sp[0] + 1e-8)

    # 2. Coronal Slice (plane X-Z, fixed iy):
    # Base: arr[:, iy, :] has shape (nx, nz).
    # Horizontal axis is X (ix), vertical axis is Z (iz).
    # To have Superior UP: if dir_z_pos, iz increases towards Superior; else flip iz.
    flip_cor_u = not dir_x_pos
    flip_cor_v = not dir_z_pos

    cor_raw = arr[:, iy, :]
    if flip_cor_u:
        cor_raw = cor_raw[::-1, :]
    if flip_cor_v:
        cor_raw = cor_raw[:, ::-1]
    cor_disp = cor_raw.T  # shape (nz, nx)
    aspect_cor = sp[2] / (sp[0] + 1e-8)

    # 3. Sagittal Slice (plane Y-Z, fixed ix):
    # Base: arr[ix, :, :] has shape (ny, nz).
    # Horizontal axis is Y (iy), vertical axis is Z (iz).
    # To have Anterior on RIGHT: if dir_y_pos, iy increases towards Anterior (RIGHT, keep);
    #                           if not dir_y_pos, flip iy.
    # To have Superior UP: if dir_z_pos, keep; else flip iz.
    flip_sag_u = not dir_y_pos
    flip_sag_v = not dir_z_pos

    sag_raw = arr[ix, :, :]
    if flip_sag_u:
        sag_raw = sag_raw[::-1, :]
    if flip_sag_v:
        sag_raw = sag_raw[:, ::-1]
    sag_disp = sag_raw.T  # shape (nz, ny)
    aspect_sag = sp[2] / (sp[1] + 1e-8)

    return dict(
        ax=ax_disp,
        cor=cor_disp,
        sag=sag_disp,
        aspect_ax=aspect_ax,
        aspect_cor=aspect_cor,
        aspect_sag=aspect_sag,
        center_vox=np.array([ix, iy, iz], dtype=int),
        center_mm=np.asarray(center_mm, dtype=np.float32),
        spacing=sp.astype(np.float32),
        origin=org.astype(np.float32),
        direction=D.astype(np.float32),
        flip_ax=(flip_ax_u, flip_ax_v),
        flip_cor=(flip_cor_u, flip_cor_v),
        flip_sag=(flip_sag_u, flip_sag_v),
    )


# ---------------------------------------------------------------------------
# Keypoint Projection onto Display Slices
# ---------------------------------------------------------------------------

def project_to_slice(
    image,
    points_mm: np.ndarray,
    slice_axis: int,  # 0=sagittal, 1=coronal, 2=axial
    slice_pos_mm: float,
    slab_half_mm: Optional[float] = None,
    flip_u: bool = False,
    flip_v: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project physical-space 3D points onto a 2D display slice extracted by `extract_ortho_slices`.

    Parameters
    ----------
    image        : ANTsImage
    points_mm    : [N, 3] float — physical coordinates
    slice_axis   : 0 (Sagittal, X-cut), 1 (Coronal, Y-cut), 2 (Axial, Z-cut)
    slice_pos_mm : physical position along the cut axis
    slab_half_mm : half-width of the slab (default: 4 * max(spacing))
    flip_u       : boolean flag from `extract_ortho_slices`
    flip_v       : boolean flag from `extract_ortho_slices`

    Returns
    -------
    u_disp : [N] float32 — column coordinate on the 2D display slice
    v_disp : [N] float32 — row coordinate on the 2D display slice
    mask   : [N] bool    — whether the point falls within the slab
    """
    org, sp, D = get_image_affine(image)
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis]
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

    if slab_half_mm is None:
        slab_half_mm = float(np.max(sp)) * 4.0

    # Physical distance along the cut axis in scanner coordinates
    mask = np.abs(pts[:, slice_axis] - slice_pos_mm) <= slab_half_mm

    # Convert physical coordinates to native voxel XYZ: (ix, iy, iz)
    vox = physical_to_vox(image, pts)
    nx, ny, nz = image.shape

    if slice_axis == 2:  # Axial: u is X (ix), v is Y (iy)
        u = vox[:, 0]
        v = vox[:, 1]
        Nu, Nv = nx, ny
    elif slice_axis == 1:  # Coronal: u is X (ix), v is Z (iz)
        u = vox[:, 0]
        v = vox[:, 2]
        Nu, Nv = nx, nz
    else:  # Sagittal: u is Y (iy), v is Z (iz)
        u = vox[:, 1]
        v = vox[:, 2]
        Nu, Nv = ny, nz

    # Apply directional flips matching extract_ortho_slices
    if flip_u:
        u = (Nu - 1) - u
    if flip_v:
        v = (Nv - 1) - v

    return u.astype(np.float32), v.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Transform List Guard
# ---------------------------------------------------------------------------

def safe_whichtoinvert(transformlist: list, invert_flags: list) -> list:
    """Ensure whichtoinvert list exactly matches the length of transformlist."""
    n = len(transformlist)
    k = len(invert_flags)
    if k == n:
        return list(invert_flags)
    if k < n:
        return list(invert_flags) + [False] * (n - k)
    return list(invert_flags[:n])
