"""
syntx.landmarks.blob
====================
Multi-scale 3D blob detection using Laplacian-of-Gaussian (LoG) and
Difference-of-Gaussians (DoG) in pure PyTorch.

All computation is modality-independent: input is normalized to [0,1]
foreground 2nd-98th percentile before processing.

Functions
---------
detect_blobs_log : LoG scale-space extrema
detect_blobs_dog : DoG scale-space extrema (faster)
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(image, device: torch.device) -> torch.Tensor:
    """Accept ants.ANTsImage, np.ndarray, or torch.Tensor → [1,1,D,H,W] float32."""
    if hasattr(image, "numpy"):
        # ANTsImage
        arr = image.numpy()
    elif isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    arr = arr.astype(np.float32)
    t = torch.from_numpy(arr).to(device)
    # Ensure 5-D [1, 1, D, H, W]
    while t.ndim < 5:
        t = t.unsqueeze(0)
    return t


def _normalize_intensity(t: torch.Tensor) -> torch.Tensor:
    """Foreground 2nd-98th percentile normalization → [0, 1].

    Per GEMINI.md: when p98 ≤ p02 + 1e-4 (flat/uniform region), falls back to
    global [t.min(), t.max()] range to prevent zero-array collapse on synthetic
    volumes where the foreground is perfectly uniform (e.g. arr=1.0 on zero bg).
    """
    fg = t[t > 0]
    if fg.numel() < 10:
        vmin, vmax = t.min(), t.max()
    else:
        vmin = torch.quantile(fg, 0.02)
        vmax = torch.quantile(fg, 0.98)
    if (vmax - vmin) < 1e-4:
        # Degenerate range: fall back to global extent
        vmin = t.min()
        vmax = t.max()
    if (vmax - vmin) < 1e-6:
        # Completely flat image — return as-is clamped
        return t.clamp(0.0, 1.0)
    out = (t - vmin) / (vmax - vmin)
    return out.clamp(0.0, 1.0)


def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """1-D Gaussian kernel (half-width = truncate * sigma)."""
    radius = max(1, int(math.ceil(truncate * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _separable_gaussian3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a 3D Gaussian blur using three separable 1-D convolutions."""
    k = _gaussian_kernel_1d(sigma).to(vol.device)
    # depth (D)
    kd = k.view(1, 1, -1, 1, 1)
    pad_d = kd.shape[2] // 2
    out = F.conv3d(vol, kd, padding=(pad_d, 0, 0))
    # height (H)
    kh = k.view(1, 1, 1, -1, 1)
    pad_h = kh.shape[3] // 2
    out = F.conv3d(out, kh, padding=(0, pad_h, 0))
    # width (W)
    kw = k.view(1, 1, 1, 1, -1)
    pad_w = kw.shape[4] // 2
    out = F.conv3d(out, kw, padding=(0, 0, pad_w))
    return out


def _laplacian3d(vol: torch.Tensor) -> torch.Tensor:
    """Discrete 3D Laplacian via finite differences."""
    # [1,1,D,H,W]
    lap = (
        F.pad(vol, (0, 0, 0, 0, 1, 1))[:, :, 2:, :, :]
        + F.pad(vol, (0, 0, 0, 0, 1, 1))[:, :, :-2, :, :]
        - 2 * vol
        + F.pad(vol, (0, 0, 1, 1, 0, 0))[:, :, :, 2:, :]
        + F.pad(vol, (0, 0, 1, 1, 0, 0))[:, :, :, :-2, :]
        - 2 * vol
        + F.pad(vol, (1, 1, 0, 0, 0, 0))[:, :, :, :, 2:]
        + F.pad(vol, (1, 1, 0, 0, 0, 0))[:, :, :, :, :-2]
        - 2 * vol
    )
    return lap


def _scale_space_extrema(
    scale_images: list[torch.Tensor],
    sigmas: list[float],
    threshold: float,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """
    Find local extrema across scale and 3D space.
    Returns array of shape [N, 4]: (x_mm, y_mm, z_mm, sigma_mm).
    spacing is (sz, sy, sx) — voxel size in mm.
    """
    n_scales = len(scale_images)
    if n_scales < 3:
        return np.zeros((0, 4), dtype=np.float32)

    candidates = []
    for s in range(1, n_scales - 1):
        curr = scale_images[s]        # [1,1,D,H,W]
        prev = scale_images[s - 1]
        nxt  = scale_images[s + 1]

        # 3D local max/min in the current scale image (26-neighbourhood in D×H×W)
        curr_sq = curr.squeeze()      # [D, H, W]
        D, H, W = curr_sq.shape

        # Max-pool with 3x3x3 neighbourhood
        max_pool = F.max_pool3d(curr.abs(), kernel_size=3, stride=1, padding=1)
        # Candidate: absolute value equals max_pool value at that location
        is_local_max = (curr.abs() == max_pool) & (curr.abs() > threshold)

        # Also check scale-space: must be extremum relative to prev and next scales
        is_scale_ext = (
            (curr.abs() > prev.abs()) & (curr.abs() > nxt.abs())
        )
        mask = is_local_max & is_scale_ext
        mask_np = mask.squeeze().cpu().numpy()
        idxs = np.argwhere(mask_np)  # [N, 3] in (d, h, w)

        sz, sy, sx = spacing
        for d, h, w in idxs:
            x_mm = float(w) * sx
            y_mm = float(h) * sy
            z_mm = float(d) * sz
            candidates.append([x_mm, y_mm, z_mm, sigmas[s]])

    if not candidates:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(candidates, dtype=np.float32)


def _greedy_nms(pts: np.ndarray, min_dist_mm: float, max_kpts: int) -> np.ndarray:
    """Greedy spatial NMS: sort by scale (descending), suppress nearby points."""
    if pts.shape[0] == 0:
        return pts
    # sort by scale descending (stronger/larger blobs first)
    order = np.argsort(-pts[:, 3])
    pts = pts[order]
    kept = []
    for pt in pts:
        if len(kept) >= max_kpts:
            break
        if not kept:
            kept.append(pt)
            continue
        dists = np.linalg.norm(np.array(kept)[:, :3] - pt[:3], axis=1)
        if dists.min() >= min_dist_mm:
            kept.append(pt)
    return np.array(kept, dtype=np.float32) if kept else np.zeros((0, 4), dtype=np.float32)


def _get_spacing(image) -> tuple[float, float, float]:
    """Extract spacing (sz, sy, sx) from image; default (1,1,1)."""
    if hasattr(image, "spacing"):
        sp = image.spacing
        # ANTsImage spacing is (sx, sy, sz) for 3D
        if len(sp) >= 3:
            return (float(sp[2]), float(sp[1]), float(sp[0]))
        elif len(sp) == 2:
            return (1.0, float(sp[1]), float(sp[0]))
    return (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_blobs_log(
    image,
    sigma_min: float = 2.0,
    sigma_max: float = 8.0,
    n_scales: int = 6,
    threshold: float = 0.005,
    min_distance_mm: float = 4.0,
    max_keypoints: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Detect 3D blobs using Laplacian-of-Gaussian scale-space extrema.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        Input volume. Normalized internally to [0,1] foreground 2nd-98th percentile.
    sigma_min, sigma_max : float
        Range of Gaussian sigmas in voxels.
    n_scales : int
        Number of scale levels.
    threshold : float
        Minimum |LoG| response to accept a candidate.
    min_distance_mm : float
        NMS suppression radius in mm.
    max_keypoints : int
        Maximum number of returned keypoints.
    device : str | None
        Torch device string; auto-selected (mps > cuda > cpu) if None.

    Returns
    -------
    np.ndarray, shape [N, 4]
        Columns: (x_mm, y_mm, z_mm, sigma_mm).
    """
    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)
    spacing = _get_spacing(image)

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()

    scale_images = []
    for sigma in sigmas:
        smoothed = _separable_gaussian3d(vol, sigma)
        lap = _laplacian3d(smoothed)
        # Scale-normalize: multiply by sigma^2
        scale_images.append(lap * (sigma ** 2))

    pts = _scale_space_extrema(scale_images, sigmas, threshold, spacing)
    pts = _greedy_nms(pts, min_distance_mm, max_keypoints)
    return pts


def detect_blobs_dog(
    image,
    sigma_min: float = 2.0,
    sigma_max: float = 8.0,
    n_scales: int = 6,
    threshold: float = 0.005,
    min_distance_mm: float = 4.0,
    max_keypoints: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Detect 3D blobs using Difference-of-Gaussians (DoG) scale-space extrema.

    DoG is a fast approximation to LoG: DoG(x, sigma) ≈ (k-1)*sigma^2 * LoG(x, sigma)
    where k = sigma_{i+1} / sigma_i.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        Input volume.
    sigma_min, sigma_max : float
        Range of Gaussian sigmas in voxels.
    n_scales : int
        Number of Gaussian scale levels (DoG has n_scales-1 levels).
    threshold : float
        Minimum |DoG| response.
    min_distance_mm : float
        NMS suppression radius in mm.
    max_keypoints : int
        Maximum number of returned keypoints.
    device : str | None
        Torch device string; auto-selected if None.

    Returns
    -------
    np.ndarray, shape [N, 4]
        Columns: (x_mm, y_mm, z_mm, sigma_mm).
    """
    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)
    spacing = _get_spacing(image)

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()

    # Compute Gaussian blurs
    gaussians = [_separable_gaussian3d(vol, s) for s in sigmas]

    # DoG = G(sigma_{i+1}) - G(sigma_i)
    dog_images = []
    dog_sigmas = []
    for i in range(len(sigmas) - 1):
        dog = gaussians[i + 1] - gaussians[i]
        # Scale normalize
        dog_images.append(dog * (sigmas[i] ** 2))
        dog_sigmas.append(math.sqrt(sigmas[i] * sigmas[i + 1]))

    pts = _scale_space_extrema(dog_images, dog_sigmas, threshold, spacing)
    pts = _greedy_nms(pts, min_distance_mm, max_keypoints)
    return pts
