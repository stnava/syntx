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
    scale_images_raw: list[torch.Tensor],
    sigmas: list[float],
    threshold: float,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Find local extrema across scale and 3D space.

    Parameters
    ----------
    scale_images     : σ²-normalized LoG/DoG stack (for signature compat; not used internally).
    scale_images_raw : raw LoG/DoG stack — spatial local-max detection performed here.
    sigmas           : scale values (mm) for each level.
    threshold        : relative fraction of per-scale max (e.g. 0.005 = top 0.5%).
    origin, spacing, direction : full ANTs image affine (from _get_image_affine).

    Returns
    -------
    np.ndarray [N, 4] — physical (x_mm, y_mm, z_mm, sigma_mm) in ANTs world space.
    """
    n_scales = len(scale_images)
    if n_scales < 3:
        return np.zeros((0, 4), dtype=np.float32)

    candidates = []
    for s in range(1, n_scales - 1):
        curr_raw = scale_images_raw[s]        # [1,1,D,H,W] — raw LoG/DoG

        curr_abs  = curr_raw.abs()
        scale_max = float(curr_abs.max())
        if scale_max < 1e-9:
            continue
        abs_thresh = threshold * scale_max

        # Spatial 3×3×3 local maximum
        max_pool     = F.max_pool3d(curr_abs, kernel_size=3, stride=1, padding=1)
        is_local_max = (curr_abs == max_pool) & (curr_abs > abs_thresh)

        mask_np = is_local_max.squeeze().cpu().numpy()
        idxs    = np.argwhere(mask_np)   # [N, 3] in tensor (iz, iy, ix) = (d, h, w)

        if idxs.shape[0] == 0:
            continue

        # Apply the full ANTs affine: physical = origin + direction @ (idx_XYZ * spacing)
        phys = _vox_to_physical(idxs, origin, spacing, direction)  # [N, 3]
        sigma_col = np.full((phys.shape[0], 1), sigmas[s], dtype=np.float32)
        candidates.append(np.hstack([phys, sigma_col]))

    if not candidates:
        return np.zeros((0, 4), dtype=np.float32)
    return np.vstack(candidates).astype(np.float32)



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


def _get_image_affine(image):
    """
    Extract the full ANTs image affine components.

    ANTs physical coordinate convention:
        physical = origin + direction_matrix @ (index_XYZ * spacing_XYZ)

    Returns
    -------
    origin    : np.ndarray [3]  — physical coordinate of voxel (0,0,0) in (x,y,z)
    spacing   : np.ndarray [3]  — voxel size in mm (sx, sy, sz) in XYZ order
    direction : np.ndarray [3,3] — direction cosines (column i = physical direction of axis i)
    """
    if hasattr(image, "spacing"):
        sp  = np.array(image.spacing,   dtype=np.float64)
        org = np.array(image.origin,    dtype=np.float64)
        D   = np.array(image.direction, dtype=np.float64).reshape(3, 3)
        # Pad to 3 components for 2D images
        if sp.size == 2:
            sp  = np.append(sp,  1.0)
            org = np.append(org, 0.0)
            D   = np.eye(3, dtype=np.float64)
        return org[:3], sp[:3], D
    # No metadata — identity affine
    return np.zeros(3), np.ones(3), np.eye(3)


def _vox_to_physical(
    indices_zyx: np.ndarray,          # [N, 3] in tensor (iz, iy, ix) order
    origin: np.ndarray,               # [3]
    spacing: np.ndarray,              # [3] in XYZ = (sx, sy, sz)
    direction: np.ndarray,            # [3, 3]
) -> np.ndarray:
    """
    Convert tensor voxel indices (ZYX = iz, iy, ix) to physical mm coordinates
    (x_mm, y_mm, z_mm) via the ANTs image affine:

        physical_XYZ = origin + direction @ (index_XYZ * spacing_XYZ)

    Tensor layout [1,1,D,H,W]: D=iz, H=iy, W=ix → reorder to XYZ for affine.
    """
    if indices_zyx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # (iz, iy, ix) → (ix, iy, iz)
    idx_xyz = indices_zyx[:, ::-1].astype(np.float64)   # [N, 3]
    # physical [N,3] = origin + (idx * spacing) @ direction.T
    physical = origin + (idx_xyz * spacing) @ direction.T
    return physical.astype(np.float32)


# Kept for backward compatibility — returns (sz, sy, sx) from image spacing.
# New code should use _get_image_affine instead.
def _get_spacing(image) -> tuple[float, float, float]:
    """Extract spacing (sz, sy, sx) from image; default (1,1,1)."""
    if hasattr(image, "spacing"):
        sp = image.spacing
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
    preprocess: bool = True,
    use_n4: bool = True,
    use_denoise: bool = True,
) -> np.ndarray:
    """
    Detect 3D blobs using Laplacian-of-Gaussian scale-space extrema.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        Input volume.
    sigma_min, sigma_max : float
        Range of Gaussian sigmas in voxels.
    n_scales : int
        Number of scale levels.
    threshold : float
        Relative |LoG| response threshold per scale (fraction of per-scale max).
    min_distance_mm : float
        NMS suppression radius in mm.
    max_keypoints : int
        Maximum number of returned keypoints.
    device : str | None
        Torch device string; auto-selected (mps > cuda > cpu) if None.
    preprocess : bool
        Apply standard benchmark preprocessing (N4+NLM+normalization for MRI;
        foreground normalization only for CT).  Default True.
        Set False if you have already preprocessed the image.
    use_n4 : bool
        Apply N4 bias field correction (MRI only, requires preprocess=True).
    use_denoise : bool
        Apply NLM denoising (MRI only, requires preprocess=True).

    Returns
    -------
    np.ndarray, shape [N, 4]
        Columns: (x_mm, y_mm, z_mm, sigma_mm).
    """
    if preprocess and hasattr(image, "numpy") and hasattr(image, "new_image_like"):
        from .preprocess import preprocess_for_landmarks
        image = preprocess_for_landmarks(image, use_n4=use_n4, use_denoise=use_denoise)

    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)   # safety clamp; already [0,1] after preprocess
    origin, spacing, direction = _get_image_affine(image)

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()

    # Build TWO parallel lists:
    #   log_raw    — σ²-normalized LoG  (passed for API compat, unused internally)
    #   log_unnorm — raw LoG (no σ² weight): used for spatial local-max
    log_raw   = []
    log_unnorm = []
    for sigma in sigmas:
        smoothed = _separable_gaussian3d(vol, sigma)
        lap = _laplacian3d(smoothed)
        log_raw.append(lap * (sigma ** 2))   # σ²-normalized (API compat)
        log_unnorm.append(lap)               # raw for detection

    pts = _scale_space_extrema(log_raw, log_unnorm, sigmas, threshold, origin, spacing, direction)
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
    preprocess: bool = True,
    use_n4: bool = True,
    use_denoise: bool = True,
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
        Relative |DoG| response threshold per scale (fraction of per-scale max).
    min_distance_mm : float
        NMS suppression radius in mm.
    max_keypoints : int
        Maximum number of returned keypoints.
    device : str | None
        Torch device string; auto-selected if None.
    preprocess : bool
        Apply standard benchmark preprocessing (N4+NLM+norm for MRI;
        norm only for CT).  Default True.
    use_n4 : bool
        N4 bias correction (MRI only, requires preprocess=True).
    use_denoise : bool
        NLM denoising (MRI only, requires preprocess=True).

    Returns
    -------
    np.ndarray, shape [N, 4]
        Columns: (x_mm, y_mm, z_mm, sigma_mm).
    """
    if preprocess and hasattr(image, "numpy") and hasattr(image, "new_image_like"):
        from .preprocess import preprocess_for_landmarks
        image = preprocess_for_landmarks(image, use_n4=use_n4, use_denoise=use_denoise)

    dev = _get_device(device)
    vol = _to_tensor(image, dev)
    vol = _normalize_intensity(vol)
    origin, spacing, direction = _get_image_affine(image)

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()

    # Compute Gaussian blurs
    gaussians = [_separable_gaussian3d(vol, s) for s in sigmas]

    # DoG = G(sigma_{i+1}) - G(sigma_i)
    dog_images = []     # σ²-normalized (API compat)
    dog_raw    = []     # raw DoG — used for detection
    dog_sigmas = []
    for i in range(len(sigmas) - 1):
        dog = gaussians[i + 1] - gaussians[i]
        dog_images.append(dog * (sigmas[i] ** 2))   # normalized
        dog_raw.append(dog)                          # raw
        dog_sigmas.append(math.sqrt(sigmas[i] * sigmas[i + 1]))

    pts = _scale_space_extrema(dog_images, dog_raw, dog_sigmas, threshold, origin, spacing, direction)
    pts = _greedy_nms(pts, min_distance_mm, max_keypoints)
    return pts

