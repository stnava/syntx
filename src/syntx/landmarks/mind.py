"""
syntx.landmarks.mind — MIND-SSC Modality-Independent Neighbourhood Descriptor
==============================================================================

Implements the Modality Independent Neighbourhood Descriptor (MIND) and its
Self-Similarity Context (SSC) variant (Heinrich et al., MedIA 2012 / MICCAI 2013).

MIND replaces raw intensities with local *self-similarity* patterns: for each voxel
the descriptor encodes how similar its local patch is to neighbouring patches within
a search region.  Because structural self-similarity patterns are largely preserved
across modalities (CT↔MRI, T1↔T2, etc.), MIND enables cross-modal landmark matching
without any modality-specific training.

Variance floor: local variance is clamped to ≥ 1e-6 (per GEMINI.md LNCC invariant)
to prevent singularities in flat background regions.

Functions
---------
compute_mind              : Dense MIND-SSC descriptor volume  [1, C, nx, ny, nz]
extract_mind_at_points    : Sparse MIND descriptors at physical coordinates  [N, C]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .blob import _get_device, _to_tensor, _normalize_intensity
from .spatial import get_image_affine, physical_offset_to_voxel, sample_tensor_at_physical

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neighbourhood offset tables
# ---------------------------------------------------------------------------

def _make_offsets(n_offsets: int, distance: float = 1.0) -> list[tuple[float, float, float]]:
    """Return a list of (dx, dy, dz) offsets in **physical LPS mm** units.

    n_offsets=6  → 6-connected axis-aligned neighbourhood (MIND-6).
    n_offsets=12 → 6-connected + 6 face-diagonal neighbours (MIND-SSC / MIND-12).
    n_offsets=26 → full 26-connected neighbourhood.

    Offsets are defined in scanner space so that two images stored in different
    native frames (axis flips / permutations) produce channel-aligned descriptors.
    Convert to integer voxel shifts with ``_offsets_to_voxel_shifts``.
    """
    d = distance
    base6 = [
        (-d, 0, 0), (d, 0, 0),
        (0, -d, 0), (0, d, 0),
        (0, 0, -d), (0, 0, d),
    ]
    if n_offsets <= 6:
        return base6[:n_offsets]

    face12 = [
        (-d, -d, 0), (-d, d, 0), (d, -d, 0), (d, d, 0),
        (-d, 0, -d), (-d, 0, d), (d, 0, -d), (d, 0, d),
        (0, -d, -d), (0, -d, d), (0, d, -d), (0, d, d),
    ]
    offsets = base6 + face12[:n_offsets - 6]
    if n_offsets <= 18:
        return offsets

    corner8 = [
        (-d, -d, -d), (-d, -d, d), (-d, d, -d), (-d, d, d),
        (d, -d, -d), (d, -d, d), (d, d, -d), (d, d, d),
    ]
    return (offsets + corner8)[:n_offsets]


def _offsets_to_voxel_shifts(offsets_mm, image_affine) -> list[tuple[int, int, int]]:
    """Physical mm offsets → integer array-axis shifts (dix, diy, diz), each at
    least one voxel in magnitude along its dominant axis so the shift is non-trivial."""
    vox = physical_offset_to_voxel(image_affine, np.asarray(offsets_mm, dtype=np.float64))
    shifts = []
    for row in vox:
        r = np.rint(row).astype(int)
        if np.all(r == 0):                       # sub-voxel offset: step one voxel along dominant axis
            a = int(np.argmax(np.abs(row)))
            r[a] = 1 if row[a] >= 0 else -1
        shifts.append((int(r[0]), int(r[1]), int(r[2])))
    return shifts


# ---------------------------------------------------------------------------
# Core MIND computation
# ---------------------------------------------------------------------------

def _shift_3d(vol: torch.Tensor, dz: int, dy: int, dx: int) -> torch.Tensor:
    """Shift a [1,1,A,B,C] tensor by (dz, dy, dx) along dims (2, 3, 4) with
    zero-padding at borders.  ``out[i] = vol[i - shift]`` per axis."""
    out = torch.zeros_like(vol)
    # build source and dest slice tuples
    def _slice(size, d):
        if d == 0:
            return slice(None), slice(None)
        elif d > 0:
            return slice(d, size),   slice(0, size - d)
        else:
            return slice(0, size + d), slice(-d, size)

    sz_src, sz_dst = _slice(vol.shape[2], dz)
    sy_src, sy_dst = _slice(vol.shape[3], dy)
    sx_src, sx_dst = _slice(vol.shape[4], dx)
    out[:, :, sz_dst, sy_dst, sx_dst] = vol[:, :, sz_src, sy_src, sx_src]
    return out


def compute_mind(
    image,
    n_offsets: int = 12,
    patch_size: int = 7,
    offset_distance: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute a dense MIND-SSC descriptor volume.

    For each voxel at position **x** and each neighbourhood offset **r**:

        MIND(I, x, r) = exp( -D_patch(x, x+r) / max(V_hat(x), 1e-6) )

    where D_patch is the mean patch SSD and V_hat is the estimated local noise
    variance (approximated via a Gaussian-smoothed variance).

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        3-D volume.  Normalized internally to [0, 1].
    n_offsets : int
        Number of neighbourhood offsets (6, 12, or 26).  Default 12 (MIND-SSC).
    patch_size : int
        Side length of the local comparison patch in voxels (must be odd).
    offset_distance : float
        Neighbour offset distance in **mm** along physical LPS axes; converted
        to integer voxel shifts via the image direction matrix and spacing.
    device : str | None
        Torch device; auto-selected if None.

    Returns
    -------
    torch.Tensor, shape [1, n_offsets, nx, ny, nz]
        Dense MIND descriptor in native XYZ array layout.  Values in [0, 1];
        1.0 = identical patch, 0.0 = maximally dissimilar or border-padded zero region.
    """
    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)          # [1, 1, nx, ny, nz]

    affine = get_image_affine(image)
    offsets_mm = _make_offsets(n_offsets, offset_distance)
    shifts = _offsets_to_voxel_shifts(offsets_mm, affine)   # (dix, diy, diz)
    half = patch_size // 2
    pad = half

    # Estimate local noise variance via smoothed squared intensity
    # V_hat(x) ≈ avg_pool(I^2) - avg_pool(I)^2
    avg_I  = F.avg_pool3d(vol,   patch_size, stride=1, padding=pad)
    avg_I2 = F.avg_pool3d(vol**2, patch_size, stride=1, padding=pad)
    variance = (avg_I2 - avg_I**2).clamp_min(1e-6)  # variance floor (GEMINI.md invariant)

    mind_channels = []
    for (dix, diy, diz) in shifts:
        shifted = _shift_3d(vol, dix, diy, diz)       # dims (2,3,4) = (ix, iy, iz)
        diff_sq = (vol - shifted) ** 2                 # [1, 1, D, H, W]
        patch_ssd = F.avg_pool3d(diff_sq, patch_size, stride=1, padding=pad)
        mind_r = torch.exp(-patch_ssd / variance)     # ∈ (0, 1]
        mind_channels.append(mind_r)

    mind = torch.cat(mind_channels, dim=1)            # [1, n_offsets, nx, ny, nz]
    return mind


# ---------------------------------------------------------------------------
# Sparse extraction at physical coordinates
# ---------------------------------------------------------------------------

def extract_mind_at_points(
    image,
    points_mm: np.ndarray,
    n_offsets: int = 12,
    patch_size: int = 7,
    offset_distance: float = 1.0,
    device: Optional[str] = None,
    mind_vol: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Extract MIND-SSC descriptors at sparse physical-space coordinates.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        3-D volume.
    points_mm : np.ndarray, shape [N, 3]
        Physical coordinates (x_mm, y_mm, z_mm).
    n_offsets : int
        Number of MIND offsets (default 12).
    patch_size : int
        Patch comparison window (default 7).
    offset_distance : float
        Neighbour offset distance in mm (default 1.0).
    device : str | None
        Torch device.
    mind_vol : torch.Tensor | None
        Pre-computed ``compute_mind(image, ...)`` volume to reuse.

    Returns
    -------
    np.ndarray, shape [N, n_offsets]
        MIND descriptors at each point (trilinear interpolation).
    """
    points_mm = np.asarray(points_mm)
    if points_mm.shape[0] == 0:
        return np.zeros((0, n_offsets), dtype=np.float32)

    if mind_vol is None:
        mind_vol = compute_mind(image, n_offsets, patch_size, offset_distance, device)
    # mind_vol: [1, C, nx, ny, nz] in native XYZ layout — sampled via the single
    # canonical physical→tensor sampler in syntx.landmarks.spatial.
    sampled = sample_tensor_at_physical(mind_vol, image, points_mm[:, :3],
                                        mode="bilinear", padding_mode="border")   # [N, C]
    return sampled.cpu().numpy().astype(np.float32)

