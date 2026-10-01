"""
A simplified MIND-style self-similarity descriptor (after Heinrich et al., MedIA 2012).

For each voxel and each neighbour offset r, the descriptor is ``exp(-patch SSD(x, x + r) /
local intensity variance)``: how similar the patch at x is to the patch shifted by r. Self-
similarity patterns are largely shared across modalities, so the descriptors can be compared
between e.g. CT and MRI. Differences from the published MIND / MIND-SSC: the denominator is
the local intensity variance (box filter, floored at 1e-6), not the mean patch distance over
the neighbourhood; patch pairs are always (centre, centre + r), not the SSC six-neighbour pairs;
there is no per-voxel max normalisation.

Functions
---------
compute_mind              : Dense MIND-style descriptor volume  [1, C, nx, ny, nz]
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

    The first ``n_offsets`` of: 6 axis neighbours (±d along x, y, z), 12 in-plane diagonal
    neighbours (two non-zero components), 8 corner neighbours. So 6 = 6-connected, 12 = the
    6 axis + 4 xy and 2 xz diagonals (not a symmetric set), 18 = 18-connected,
    26 = 26-connected. Values outside 1 .. 26 raise ValueError.

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
    if not 1 <= int(n_offsets) <= 26:
        raise ValueError(f"n_offsets must be in 1 .. 26, got {n_offsets}")
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
    """Physical mm offsets → integer array-axis shifts (dix, diy, diz), rounded; an offset
    that rounds to zero becomes a one-voxel step along its dominant axis."""
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
    zero-padding at borders.  ``out[i] = vol[i + shift]`` per axis.  (The argument names
    are generic: callers pass (dix, diy, diz) for an XYZ-layout tensor.)"""
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
    Compute a dense self-similarity (MIND-style) descriptor volume.

    For each voxel at position **x** and each neighbourhood offset **r**:

        MIND(I, x, r) = exp( -D_patch(x, x+r) / max(V(x), 1e-6) )

    where D_patch is the mean over a ``patch_size``^3 box of (I - I shifted by r)^2 (I shifted
    with zeros outside the image) and V is the local intensity variance over the same box
    (``avg_pool(I^2) - avg_pool(I)^2``). Box means use ``avg_pool3d`` with zero padding
    counted in the average, so values near the border are biased. No preprocessing (N4 /
    denoising) is applied; the volume is only rescaled with the blob detectors'
    foreground 2nd-98th percentile normalisation.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        3-D volume. ndarray / tensor input has identity geometry (offsets then in voxels).
    n_offsets : int, default 12
        Number of neighbourhood offsets, 1 .. 26 (see ``_make_offsets``).
    patch_size : int, default 7
        Side length of the local comparison box in voxels; must be odd (ValueError).
    offset_distance : float, default 1.0
        Neighbour offset distance in **mm** along physical LPS axes; converted
        to integer voxel shifts via the image direction matrix and spacing (at least one
        voxel).
    device : str | None
        Torch device; auto-selected (mps > cuda > cpu) if None.

    Returns
    -------
    torch.Tensor, shape [1, n_offsets, nx, ny, nz], float32
        Dense descriptor in native XYZ array layout, on ``device``.  Values in [0, 1];
        1.0 = patch identical to its shifted copy, toward 0 = dissimilar relative to the
        local variance.
    """
    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)          # [1, 1, nx, ny, nz]

    affine = get_image_affine(image)
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")
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
    Sample ``compute_mind`` descriptors at physical-space points.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        3-D volume (also supplies the geometry for the sampling).
    points_mm : np.ndarray, shape [N, >=3]
        Physical coordinates (x_mm, y_mm, z_mm); extra columns (e.g. scale) are ignored.
    n_offsets, patch_size, offset_distance, device
        Passed to ``compute_mind`` (defaults 12, 7, 1.0, None). With ``mind_vol`` only
        ``n_offsets`` is used, to check its channel count.
    mind_vol : torch.Tensor | None
        Pre-computed ``compute_mind(image, ...)`` volume to reuse; ValueError unless it is
        [1, n_offsets, *image grid].

    Returns
    -------
    np.ndarray, shape [N, C], float32
        Descriptors at each point (trilinear interpolation, border values outside the
        image); C = channels of the volume (``n_offsets`` for N = 0).
    """
    points_mm = np.asarray(points_mm)
    if points_mm.shape[0] == 0:
        return np.zeros((0, n_offsets), dtype=np.float32)

    if mind_vol is None:
        mind_vol = compute_mind(image, n_offsets, patch_size, offset_distance, device)
    else:
        grid = tuple(image.shape) if hasattr(image, "shape") else None
        if mind_vol.dim() != 5 or (grid is not None and tuple(mind_vol.shape[2:]) != grid[:3]):
            raise ValueError(f"mind_vol shape {tuple(mind_vol.shape)} does not match the image grid {grid}")
        if mind_vol.shape[1] != n_offsets:
            raise ValueError(f"mind_vol has {mind_vol.shape[1]} channels but n_offsets={n_offsets}")
    # mind_vol: [1, C, nx, ny, nz] in native XYZ layout — sampled via the single
    # canonical physical→tensor sampler in syntx.landmarks.spatial.
    sampled = sample_tensor_at_physical(mind_vol, image, points_mm[:, :3],
                                        mode="bilinear", padding_mode="border")   # [N, C]
    return sampled.cpu().numpy().astype(np.float32)

