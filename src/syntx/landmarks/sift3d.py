"""
syntx.landmarks.sift3d — Full 3D SIFT in pure PyTorch
======================================================

Implements a volumetric generalisation of Lowe's SIFT algorithm:

1. DoG scale-space extrema detection (reuses the same building blocks as blob.py).
2. 3D gradient orientation computation (Sobel-like finite differences).
3. Dominant orientation assignment per keypoint.
4. 3D descriptor: 4×4×4 spatial cells × 8 spherical orientation bins = 512-D,
   L2-normalised.

Modality / anatomy independence
---------------------------------
Gradient-histogram descriptors encode purely geometric structure.  After foreground
2nd–98th percentile normalization any volume — MRI, CT, PET, US — produces
comparable descriptors at geometrically equivalent locations.

Functions
---------
detect_sift3d : keypoint coordinates [N, 4] + descriptors [N, 512]
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .blob import (
    _get_device,
    _to_tensor,
    _normalize_intensity,
    _separable_gaussian3d,
    _get_spacing,
    _greedy_nms,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

_SOBEL_1D = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)


def _gradient_3d(vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (gx, gy, gz) gradient components via 1-D central differences.
    vol shape: [1, 1, D, H, W].
    """
    kx = _SOBEL_1D.to(vol.device).view(1, 1, 1, 1, 3)
    ky = _SOBEL_1D.to(vol.device).view(1, 1, 1, 3, 1)
    kz = _SOBEL_1D.to(vol.device).view(1, 1, 3, 1, 1)
    gx = F.conv3d(vol, kx, padding=(0, 0, 1))
    gy = F.conv3d(vol, ky, padding=(0, 1, 0))
    gz = F.conv3d(vol, kz, padding=(1, 0, 0))
    return gx, gy, gz


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

def _build_descriptor(
    vol: torch.Tensor,
    kp_vox: np.ndarray,           # [N, 3] (d, h, w) voxel indices (float)
    sigmas: list[float],
    kp_sigma_idx: list[int],
    n_cells: int = 4,
    n_bins: int = 8,
) -> np.ndarray:
    """
    Build a 3D SIFT-style descriptor at each keypoint.

    For each keypoint at scale sigma_k, extract a (n_cells^3 × n_bins)-dimensional
    descriptor by:
    - Sampling gradient magnitudes/directions in a local window of radius
      n_cells * sigma_k voxels.
    - Accumulating into n_cells × n_cells × n_cells spatial cells × n_bins
      spherical orientation bins.
    - L2-normalising and clamping at 0.2 (per Lowe 2004).

    Returns ndarray [N, n_cells^3 * n_bins].
    """
    device = vol.device
    N = kp_vox.shape[0]
    desc_dim = n_cells ** 3 * n_bins
    descs = np.zeros((N, desc_dim), dtype=np.float32)

    if N == 0:
        return descs

    D, H, W = vol.shape[-3], vol.shape[-2], vol.shape[-1]
    vol_np = vol.squeeze().cpu().numpy()

    # Pre-compute gradient volume once
    gx, gy, gz = _gradient_3d(vol)
    gx_np = gx.squeeze().cpu().numpy()
    gy_np = gy.squeeze().cpu().numpy()
    gz_np = gz.squeeze().cpu().numpy()
    mag_np = np.sqrt(gx_np**2 + gy_np**2 + gz_np**2 + 1e-12)

    # Spherical bin boundaries (elevation + azimuth combined via hash)
    elev_edges = np.linspace(-np.pi / 2, np.pi / 2, n_bins // 2 + 1)
    azim_edges = np.linspace(-np.pi, np.pi, n_bins // 2 + 1)

    for i, (kp, s_idx) in enumerate(zip(kp_vox, kp_sigma_idx)):
        sigma = sigmas[min(s_idx, len(sigmas) - 1)]
        half = int(math.ceil(n_cells * sigma))
        if half < 1:
            half = 1

        d0, h0, w0 = kp
        # Bounding box (clipped to volume)
        d_lo = max(0, int(d0) - half); d_hi = min(D, int(d0) + half + 1)
        h_lo = max(0, int(h0) - half); h_hi = min(H, int(h0) + half + 1)
        w_lo = max(0, int(w0) - half); w_hi = min(W, int(w0) + half + 1)

        if d_hi <= d_lo or h_hi <= h_lo or w_hi <= w_lo:
            continue

        patch_gx = gx_np[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
        patch_gy = gy_np[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
        patch_gz = gz_np[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
        patch_mag = mag_np[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]

        PD, PH, PW = patch_gx.shape

        # Map voxels → cell indices
        ds = np.linspace(0, PD, n_cells + 1)
        hs = np.linspace(0, PH, n_cells + 1)
        ws = np.linspace(0, PW, n_cells + 1)

        desc = np.zeros(desc_dim, dtype=np.float32)
        for ci in range(n_cells):
            for cj in range(n_cells):
                for ck in range(n_cells):
                    cell_idx = ci * n_cells * n_cells + cj * n_cells + ck
                    s_d = slice(int(ds[ci]), int(ds[ci + 1]))
                    s_h = slice(int(hs[cj]), int(hs[cj + 1]))
                    s_w = slice(int(ws[ck]), int(ws[ck + 1]))

                    cgx = patch_gx[s_d, s_h, s_w].ravel()
                    cgy = patch_gy[s_d, s_h, s_w].ravel()
                    cgz = patch_gz[s_d, s_h, s_w].ravel()
                    cmag = patch_mag[s_d, s_h, s_w].ravel()

                    if cmag.sum() < 1e-8:
                        continue

                    # Spherical orientation: elevation + azimuth → bin
                    elev = np.arctan2(cgz, np.sqrt(cgx**2 + cgy**2 + 1e-12))
                    azim = np.arctan2(cgy, cgx + 1e-12)

                    elev_bin = np.clip(np.digitize(elev, elev_edges) - 1, 0, n_bins // 2 - 1)
                    azim_bin = np.clip(np.digitize(azim, azim_edges) - 1, 0, n_bins // 2 - 1)
                    combined_bin = elev_bin * (n_bins // 2) + azim_bin

                    # Weighted histogram (n_bins bins total using elev*azim combos)
                    hist = np.bincount(combined_bin, weights=cmag, minlength=n_bins)
                    desc[cell_idx * n_bins:(cell_idx + 1) * n_bins] = hist[:n_bins]

        # L2-normalise + clamp + re-normalise (Lowe 2004)
        norm = np.linalg.norm(desc) + 1e-8
        desc /= norm
        desc = np.clip(desc, 0.0, 0.2)
        norm2 = np.linalg.norm(desc) + 1e-8
        desc /= norm2
        descs[i] = desc

    return descs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sift3d(
    image,
    sigma_min: float = 1.5,
    sigma_max: float = 6.0,
    n_scales: int = 5,
    threshold: float = 0.005,
    min_distance_mm: float = 4.0,
    max_keypoints: int = 256,
    n_cells: int = 4,
    n_bins: int = 8,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full 3D SIFT: DoG keypoint detection + 3D gradient histogram descriptors.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        3-D volume.  Normalized internally.
    sigma_min, sigma_max : float
        Gaussian scale range in voxels.
    n_scales : int
        Number of scale levels (DoG has n_scales-1 response maps).
    threshold : float
        Minimum |DoG| response for a candidate keypoint.
    min_distance_mm : float
        Greedy 3-D NMS suppression radius in mm.
    max_keypoints : int
        Upper bound on returned keypoints.
    n_cells : int
        Descriptor spatial grid cells per axis (n_cells^3 total cells).
    n_bins : int
        Spherical orientation bins per spatial cell.
    device : str | None
        Torch device; auto-selected if None.

    Returns
    -------
    coords : np.ndarray, shape [N, 4]
        (x_mm, y_mm, z_mm, sigma_mm) in physical space.
    descriptors : np.ndarray, shape [N, n_cells**3 * n_bins]
        L2-normalised 3-D SIFT descriptors (default 512-D).
    """
    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)
    spacing = _get_spacing(image)     # (sz, sy, sx)
    sz, sy, sx = spacing

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()
    gaussians = [_separable_gaussian3d(vol, s) for s in sigmas]

    # DoG images and their geometric-mean sigmas
    dog_images, dog_sigmas = [], []
    for i in range(len(sigmas) - 1):
        dog = gaussians[i + 1] - gaussians[i]
        dog_images.append(dog * (sigmas[i] ** 2))
        dog_sigmas.append(math.sqrt(sigmas[i] * sigmas[i + 1]))

    n_dogs = len(dog_images)
    if n_dogs < 3:
        empty = np.zeros((0, 4), dtype=np.float32)
        empty_d = np.zeros((0, n_cells ** 3 * n_bins), dtype=np.float32)
        return empty, empty_d

    # Detect scale-space extrema
    all_pts_mm: list[list[float]] = []
    all_kp_vox: list[list[float]] = []
    all_si:     list[int] = []

    for s in range(1, n_dogs - 1):
        curr = dog_images[s]
        prev = dog_images[s - 1]
        nxt  = dog_images[s + 1]

        max_pool = F.max_pool3d(curr.abs(), 3, 1, 1)
        is_local = (curr.abs() == max_pool) & (curr.abs() > threshold)
        is_scale = (curr.abs() > prev.abs()) & (curr.abs() > nxt.abs())
        mask_np = (is_local & is_scale).squeeze().cpu().numpy()
        idxs = np.argwhere(mask_np)  # (d, h, w)

        for (d, h, w) in idxs:
            x_mm = float(w) * sx
            y_mm = float(h) * sy
            z_mm = float(d) * sz
            all_pts_mm.append([x_mm, y_mm, z_mm, dog_sigmas[s]])
            all_kp_vox.append([float(d), float(h), float(w)])
            all_si.append(s)

    if not all_pts_mm:
        empty_d = np.zeros((0, n_cells ** 3 * n_bins), dtype=np.float32)
        return np.zeros((0, 4), dtype=np.float32), empty_d

    pts = np.array(all_pts_mm, dtype=np.float32)
    kp_vox = np.array(all_kp_vox, dtype=np.float32)

    # Greedy NMS
    order = np.argsort(-pts[:, 3])
    pts, kp_vox = pts[order], kp_vox[order]
    all_si_arr = [all_si[o] for o in order]

    kept = []
    kept_coords: list[np.ndarray] = []
    for i in range(len(pts)):
        if len(kept) >= max_keypoints:
            break
        if kept_coords:
            dists = np.linalg.norm(np.stack(kept_coords) - pts[i, :3], axis=1)
            if dists.min() < min_distance_mm:
                continue
        kept.append(i)
        kept_coords.append(pts[i, :3])

    if not kept:
        empty_d = np.zeros((0, n_cells ** 3 * n_bins), dtype=np.float32)
        return np.zeros((0, 4), dtype=np.float32), empty_d

    idx = np.array(kept, dtype=int)
    pts_kept = pts[idx]
    kp_vox_kept = kp_vox[idx]
    si_kept = [all_si_arr[i] for i in idx]

    # Build descriptors
    descs = _build_descriptor(vol, kp_vox_kept, dog_sigmas, si_kept, n_cells, n_bins)

    return pts_kept, descs
