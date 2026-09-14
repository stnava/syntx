"""
syntx.landmarks.spatial — Canonical physical-space coordinate framework
=======================================================================

All physical ↔ voxel coordinate operations, orthographic slice extraction,
and keypoint display-projection helpers for syntx.landmarks live here.

**No other module should duplicate this math.** Import from here.

ANTsPy physical coordinate convention
--------------------------------------
    physical_XYZ = origin + direction @ (index_XYZ * spacing_XYZ)

where:
  origin      [3]   — physical coordinate of voxel (0, 0, 0) in (x, y, z) mm
  spacing     [3]   — voxel size in mm, (sx, sy, sz) XYZ order
  direction   [3,3] — direction cosine matrix

Tensor (PyTorch/NumPy) vs ANTs array layout
--------------------------------------------
  ANTs array : image.numpy()  -> arr[ix, iy, iz]   shape (nx, ny, nz)
  PyTorch vol: [1,1,D,H,W]   = [1,1,iz,iy,ix]

  Slicing conventions (NO array reorientation — use affine math for labels):
    Axial   (fix iz): arr[:,:,iz].T  -> row=iy, col=ix
    Coronal (fix iy): arr[:,iy,:].T  -> row=iz, col=ix
    Sagittal(fix ix): arr[ix,:,:].T  -> row=iz, col=iy

Forbidden patterns
------------------
* ants.reorient_image / ants.reorient_image2 — these are MODULES, not callables.
  NEVER call them.  Orientation is handled via affine math here.
* x_mm = w * sx — ignores origin and direction; use vox_to_physical().
* whichtoinvert=[False] with a multi-transform list — lengths must match exactly.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Affine extraction
# ---------------------------------------------------------------------------

def get_image_affine(image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the full ANTs image affine components.

    Parameters
    ----------
    image : ants.ANTsImage | tuple(origin, spacing, direction)
        If no ANTs metadata, returns identity affine.

    Returns
    -------
    origin    : np.ndarray [3] float64
    spacing   : np.ndarray [3] float64 — (sx, sy, sz) XYZ
    direction : np.ndarray [3,3] float64
    """
    if isinstance(image, tuple) and len(image) == 3:
        org, sp, D = image
        return (np.asarray(org, np.float64),
                np.asarray(sp,  np.float64),
                np.asarray(D,   np.float64).reshape(3, 3))
    if hasattr(image, "spacing"):
        sp  = np.array(image.spacing,   dtype=np.float64)
        org = np.array(image.origin,    dtype=np.float64)
        D   = np.array(image.direction, dtype=np.float64).reshape(3, 3)
        if sp.size == 2:   # 2-D image — pad to 3-D
            sp  = np.append(sp, 1.0)
            org = np.append(org, 0.0)
            D   = np.eye(3, dtype=np.float64)
        return org[:3], sp[:3], D
    return np.zeros(3), np.ones(3), np.eye(3)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def vox_to_physical(image, indices_xyz: np.ndarray) -> np.ndarray:
    """
    Convert ANTs XYZ voxel indices to physical mm coordinates.

        physical = origin + direction @ (index_XYZ * spacing)

    Parameters
    ----------
    image       : ANTsImage or (origin, spacing, direction)
    indices_xyz : [N, 3] or [3] float — (ix, iy, iz) ANTs array indices

    Returns
    -------
    [N, 3] float32 — (x_mm, y_mm, z_mm)
    """
    org, sp, D = get_image_affine(image)
    idx = np.asarray(indices_xyz, dtype=np.float64)
    if idx.ndim == 1:
        idx = idx[np.newaxis]
    physical = org + (idx * sp) @ D.T
    return physical.astype(np.float32)


def vox_zyx_to_physical(image, indices_zyx: np.ndarray) -> np.ndarray:
    """
    Convert tensor ZYX voxel indices (iz, iy, ix) to physical mm.

    PyTorch tensors are ZYX (argwhere gives (d,h,w)=(iz,iy,ix)).
    This reverses to XYZ before applying the affine.

    Returns
    -------
    [N, 3] float32 — (x_mm, y_mm, z_mm)
    """
    idx_zyx = np.asarray(indices_zyx, dtype=np.float64)
    if idx_zyx.ndim == 1:
        idx_zyx = idx_zyx[np.newaxis]
    if idx_zyx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx_xyz = idx_zyx[:, ::-1]   # (iz,iy,ix) -> (ix,iy,iz)
    return vox_to_physical(image, idx_xyz)


def physical_to_vox(image, points_mm: np.ndarray) -> np.ndarray:
    """
    Convert physical mm coordinates to ANTs XYZ voxel indices.

        index_XYZ = inv(direction) @ (physical - origin) / spacing

    Parameters
    ----------
    image     : ANTsImage or (origin, spacing, direction)
    points_mm : [N, 3] or [3] float — (x_mm, y_mm, z_mm)

    Returns
    -------
    [N, 3] float64 — (ix, iy, iz), may be fractional
    """
    org, sp, D = get_image_affine(image)
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis]
    D_inv = np.linalg.inv(D)
    return ((pts - org) @ D_inv.T) / sp


# ---------------------------------------------------------------------------
# Slice extraction
# ---------------------------------------------------------------------------

def extract_ortho_slices(image, center_mm: Optional[np.ndarray] = None) -> dict:
    """
    Extract axial, coronal, and sagittal 2-D slices from a 3-D ANTsImage.

    Slices are taken from the native ANTs array at the given physical centre
    (defaults to image centre). NO array reorientation is performed.

    Slice conventions
    -----------------
      Axial   (fix iz): arr[:,:,iz].T  -> row=iy, col=ix
      Coronal (fix iy): arr[:,iy,:].T  -> row=iz, col=ix
      Sagittal(fix ix): arr[ix,:,:].T  -> row=iz, col=iy

    Returns
    -------
    dict with keys:
      'ax', 'cor', 'sag'   : np.ndarray 2-D slices
      'center_vox'         : [3] int (ix, iy, iz)
      'center_mm'          : [3] float32
      'spacing'            : [3] float32
      'origin'             : [3] float32
      'direction'          : [3,3] float32
      'ax_col_label'       : (neg_str, pos_str) — column axis labels
      'ax_row_label'       : (neg_str, pos_str) — row axis labels
      'cor_col_label', 'cor_row_label'
      'sag_col_label', 'sag_row_label'
    """
    arr = image.numpy()           # [nx, ny, nz]
    org, sp, D = get_image_affine(image)
    nx, ny, nz = arr.shape

    if center_mm is None:
        mid_xyz = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0])
        center_mm = vox_to_physical(image, mid_xyz)[0].astype(np.float64)

    ctr_vox = physical_to_vox(image, center_mm)[0]
    ix = int(np.clip(round(ctr_vox[0]), 0, nx - 1))
    iy = int(np.clip(round(ctr_vox[1]), 0, ny - 1))
    iz = int(np.clip(round(ctr_vox[2]), 0, nz - 1))

    ax  = arr[:, :, iz].T    # [ny, nx]
    cor = arr[:, iy, :].T    # [nz, nx]
    sag = arr[ix, :, :].T    # [nz, ny]

    xl = [_axis_label(D, i) for i in range(3)]   # labels for x,y,z voxel axes

    return dict(
        ax=ax, cor=cor, sag=sag,
        center_vox=np.array([ix, iy, iz], dtype=int),
        center_mm=np.asarray(center_mm, dtype=np.float32),
        spacing=sp.astype(np.float32),
        origin=org.astype(np.float32),
        direction=D.astype(np.float32),
        # Axial (fix iz): col=ix, row=iy
        ax_col_label=xl[0], ax_row_label=xl[1],
        # Coronal (fix iy): col=ix, row=iz
        cor_col_label=xl[0], cor_row_label=xl[2],
        # Sagittal (fix ix): col=iy, row=iz
        sag_col_label=xl[1], sag_row_label=xl[2],
    )


def _axis_label(direction: np.ndarray, axis: int) -> tuple[str, str]:
    """Return (negative_end, positive_end) anatomical label for voxel axis."""
    col = direction[:, axis]
    dominant = int(np.argmax(np.abs(col)))
    sign = np.sign(col[dominant])
    labels = [('L', 'R'), ('P', 'A'), ('I', 'S')]
    neg, pos = labels[dominant]
    return (pos, neg) if sign < 0 else (neg, pos)


def format_axis_xlabel(label_pair: tuple[str, str]) -> str:
    """'(L, R)' → '← L  |  R →' for matplotlib xlabel."""
    neg, pos = label_pair
    return f"← {neg}  |  {pos} →"


# ---------------------------------------------------------------------------
# Keypoint projection onto display slices
# ---------------------------------------------------------------------------

def project_to_slice(
    image,
    points_mm: np.ndarray,
    slice_axis: int,
    slice_pos_mm: float,
    slab_half_mm: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project physical-space 3-D points onto a 2-D display slice.

    Slice conventions (matching extract_ortho_slices):
      slice_axis=2 (axial,   fix iz): col=ix, row=iy
      slice_axis=1 (coronal, fix iy): col=ix, row=iz
      slice_axis=0 (sagittal,fix ix): col=iy, row=iz

    Parameters
    ----------
    image        : ANTsImage (for affine)
    points_mm    : [N, 3] float — (x_mm, y_mm, z_mm)
    slice_axis   : 0, 1, or 2 (XYZ physical axis that is fixed)
    slice_pos_mm : physical position of the slice on that axis
    slab_half_mm : include only points within ±slab_half_mm (default 4*max_sp)

    Returns
    -------
    u_vox : [N] float32 — column index in the 2-D image (may be out of bounds)
    v_vox : [N] float32 — row    index in the 2-D image
    mask  : [N] bool    — points within the slab
    """
    org, sp, D = get_image_affine(image)
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis]

    if slab_half_mm is None:
        slab_half_mm = float(np.max(sp)) * 4.0

    # Physical coordinate along the fixed axis
    phys_along = pts[:, slice_axis]
    mask = np.abs(phys_along - slice_pos_mm) <= slab_half_mm

    vox = physical_to_vox(image, pts)   # [N, 3] (ix, iy, iz)

    if slice_axis == 2:    # axial (fix iz): col=ix, row=iy
        u, v = vox[:, 0], vox[:, 1]
    elif slice_axis == 1:  # coronal (fix iy): col=ix, row=iz
        u, v = vox[:, 0], vox[:, 2]
    else:                  # sagittal (fix ix): col=iy, row=iz
        u, v = vox[:, 1], vox[:, 2]

    return u.astype(np.float32), v.astype(np.float32), mask


# ---------------------------------------------------------------------------
# whichtoinvert safety guard
# ---------------------------------------------------------------------------

def safe_whichtoinvert(transformlist: list, invert_flags: list) -> list:
    """
    Return a whichtoinvert list that is exactly the same length as transformlist.

    ants.apply_transforms requires len(whichtoinvert) == len(transformlist).
    This function pads with False or truncates with a warning as needed.
    Always call this before ants.apply_transforms.
    """
    import logging
    logger = logging.getLogger(__name__)
    n = len(transformlist)
    k = len(invert_flags)
    if k == n:
        return list(invert_flags)
    if k < n:
        padded = list(invert_flags) + [False] * (n - k)
        logger.warning(
            "safe_whichtoinvert: padded whichtoinvert from %d to %d entries", k, n
        )
        return padded
    truncated = list(invert_flags[:n])
    logger.warning(
        "safe_whichtoinvert: truncated whichtoinvert from %d to %d entries", k, n
    )
    return truncated
