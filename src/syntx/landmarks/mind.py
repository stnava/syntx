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
compute_mind              : Dense MIND-SSC descriptor volume  [1, C, D, H, W]
extract_mind_at_points    : Sparse MIND descriptors at physical coordinates  [N, C]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .blob import _get_device, _to_tensor, _normalize_intensity
from .spatial import get_image_affine, physical_to_vox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neighbourhood offset tables
# ---------------------------------------------------------------------------

def _make_offsets(n_offsets: int, distance: int = 1) -> list[tuple[int, int, int]]:
    """Return a list of (dz, dy, dx) integer voxel offsets.

    n_offsets=6  → 6-connected axis-aligned neighbourhood (MIND-6).
    n_offsets=12 → 6-connected + 6 face-diagonal neighbours (MIND-SSC / MIND-12).
    n_offsets=26 → full 26-connected neighbourhood.
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


# ---------------------------------------------------------------------------
# Core MIND computation
# ---------------------------------------------------------------------------

def _shift_3d(vol: torch.Tensor, dz: int, dy: int, dx: int) -> torch.Tensor:
    """Shift a [1,1,D,H,W] tensor by (dz, dy, dx) with zero-padding at borders."""
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
    offset_distance: int = 1,
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
        Side length of the local comparison patch (must be odd).
    offset_distance : int
        Voxel distance of each neighbour offset.
    device : str | None
        Torch device; auto-selected if None.

    Returns
    -------
    torch.Tensor, shape [1, n_offsets, D, H, W]
        Dense MIND descriptor.  Values in [0, 1]; 1.0 = identical patch,
        0.0 = maximally dissimilar or border-padded zero region.
    """
    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)          # [1, 1, D, H, W]

    offsets = _make_offsets(n_offsets, offset_distance)
    half = patch_size // 2
    pad = half

    # Estimate local noise variance via smoothed squared intensity
    # V_hat(x) ≈ avg_pool(I^2) - avg_pool(I)^2
    avg_I  = F.avg_pool3d(vol,   patch_size, stride=1, padding=pad)
    avg_I2 = F.avg_pool3d(vol**2, patch_size, stride=1, padding=pad)
    variance = (avg_I2 - avg_I**2).clamp_min(1e-6)  # variance floor (GEMINI.md invariant)

    mind_channels = []
    for (dz, dy, dx) in offsets:
        shifted = _shift_3d(vol, dz, dy, dx)          # [1, 1, D, H, W]
        diff_sq = (vol - shifted) ** 2                 # [1, 1, D, H, W]
        patch_ssd = F.avg_pool3d(diff_sq, patch_size, stride=1, padding=pad)
        mind_r = torch.exp(-patch_ssd / variance)     # ∈ (0, 1]
        mind_channels.append(mind_r)

    mind = torch.cat(mind_channels, dim=1)            # [1, n_offsets, D, H, W]
    return mind


# ---------------------------------------------------------------------------
# Sparse extraction at physical coordinates
# ---------------------------------------------------------------------------

def extract_mind_at_points(
    image,
    points_mm: np.ndarray,
    n_offsets: int = 12,
    patch_size: int = 7,
    offset_distance: int = 1,
    device: Optional[str] = None,
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
    offset_distance : int
        Voxel offset distance (default 1).
    device : str | None
        Torch device.

    Returns
    -------
    np.ndarray, shape [N, n_offsets]
        MIND descriptors at each point.
    """
    if points_mm.shape[0] == 0:
        return np.zeros((0, n_offsets), dtype=np.float32)

    dev = _get_device(device)

    # Build dense MIND volume
    mind_vol = compute_mind(image, n_offsets, patch_size, offset_distance, device)
    # mind_vol: [1, C, D, H, W]

    # physical → voxel XYZ: [N, 3] via syntx.landmarks.spatial
    idx_xyz = physical_to_vox(image, points_mm)   # (ix, iy, iz)

    # Tensor layout [1,C,D,H,W]: D=iz, H=iy, W=ix
    w_vox = idx_xyz[:, 0]   # ix → W
    h_vox = idx_xyz[:, 1]   # iy → H
    d_vox = idx_xyz[:, 2]   # iz → D

    nD = mind_vol.shape[2]
    nH = mind_vol.shape[3]
    nW = mind_vol.shape[4]

    # Normalise to [-1, 1] for grid_sample
    w_norm = (w_vox / (nW - 1)) * 2 - 1
    h_norm = (h_vox / (nH - 1)) * 2 - 1
    d_norm = (d_vox / (nD - 1)) * 2 - 1

    # grid_sample expects [B, 1, 1, N, 3] with (x, y, z) = (W, H, D) normalised
    grid = torch.from_numpy(
        np.stack([w_norm, h_norm, d_norm], axis=-1).astype(np.float32)
    ).to(dev)
    grid = grid.view(1, 1, 1, -1, 3)    # [1, 1, 1, N, 3]

    # mind_vol: [1, C, D, H, W]  → sample → [1, C, 1, 1, N] → [N, C]
    sampled = F.grid_sample(mind_vol, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    # sampled shape: [1, C, 1, 1, N]
    sampled = sampled.squeeze(0).squeeze(1).squeeze(1)  # → [C, N]
    sampled = sampled.permute(1, 0)                      # → [N, C]
    if sampled.ndim == 1:
        sampled = sampled.unsqueeze(0)

    return sampled.cpu().numpy().astype(np.float32)

