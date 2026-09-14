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

from .spatial import get_image_affine, image_to_tensor, vox_to_physical

logger = logging.getLogger(__name__)

FOREGROUND_LEVEL = 0.05   # on the [0,1]-normalized volume


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
    """ants.ANTsImage / ndarray / tensor → [1, 1, nx, ny, nz] float32.

    Layout is the native ANTs XYZ array layout (see ``spatial.image_to_tensor``):
    tensor dims (2, 3, 4) are array axes (ix, iy, iz).  ``np.argwhere`` on the
    squeezed tensor therefore yields XYZ indices for ``spatial.vox_to_physical``.
    """
    return image_to_tensor(image, device)


def _voxel_sigmas(sigma_mm: float, spacing) -> tuple[float, float, float]:
    """Isotropic physical sigma (mm) → per-array-axis sigma in voxels."""
    sp = np.asarray(spacing, dtype=np.float64)
    return tuple(float(max(sigma_mm / s, 0.3)) for s in sp)


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


def _separable_gaussian3d(vol: torch.Tensor, sigma) -> torch.Tensor:
    """3D Gaussian blur via three separable 1-D convolutions.

    ``sigma`` is either a scalar (voxels, same for all axes) or a 3-tuple of
    per-axis voxel sigmas ordered like the tensor dims (2, 3, 4) = (ix, iy, iz).
    """
    if np.isscalar(sigma):
        sig = (float(sigma),) * 3
    else:
        sig = tuple(float(x) for x in sigma)
    out = vol
    for dim, sg in zip((2, 3, 4), sig):
        k = _gaussian_kernel_1d(sg).to(vol.device)
        shape = [1, 1, 1, 1, 1]
        shape[dim] = -1
        kk = k.view(*shape)
        pad = [0, 0, 0]
        pad[dim - 2] = k.numel() // 2
        out = F.conv3d(out, kk, padding=tuple(pad))
    return out


def _shift_pad(vol: torch.Tensor, dim: int, offset: int, mode: str = "zeros") -> torch.Tensor:
    """``out[i] = vol[i + offset]`` along ``dim`` with zero or replicate border handling,
    implemented with slicing + ``torch.cat`` only.

    ``torch.nn.functional.pad`` is NOT used on purpose: on Apple MPS (torch 2.13)
    padding the depth dim of a 5-D tensor whose trailing H*W >= 65536 (e.g. any
    256x256 slice) silently returns garbage, which zeroed the x-gradient and
    corrupted the LoG on full-size brain volumes.  Slicing/cat are exact on all devices.
    """
    n = vol.shape[dim]
    o = abs(int(offset))
    if o == 0:
        return vol
    if offset > 0:
        core = vol.narrow(dim, o, n - o)                       # vol[o:]
        edge = vol.narrow(dim, n - 1, 1)                       # last slice
        pad = torch.zeros_like(edge) if mode == "zeros" else edge
        return torch.cat([core, pad.expand(*[-1 if d != dim else o for d in range(vol.ndim)])], dim=dim)
    core = vol.narrow(dim, 0, n - o)                           # vol[:-o]
    edge = vol.narrow(dim, 0, 1)
    pad = torch.zeros_like(edge) if mode == "zeros" else edge
    return torch.cat([pad.expand(*[-1 if d != dim else o for d in range(vol.ndim)]), core], dim=dim)


def _laplacian3d(vol: torch.Tensor, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """Discrete 3D Laplacian (per mm²) via second central differences with zero
    borders.  Each axis term is divided by spacing² so the operator is the
    physical Laplacian on anisotropic grids.  Tensor dims (2, 3, 4) = (ix, iy, iz).
    Uses slicing (see ``_shift_pad``) rather than ``F.pad`` — MPS-safe.
    """
    sp = [float(x) for x in spacing]
    lap = torch.zeros_like(vol)
    for dim, s_ in zip((2, 3, 4), sp):
        lap = lap + (_shift_pad(vol, dim, +1) + _shift_pad(vol, dim, -1) - 2 * vol) / (s_ ** 2)
    return lap


def _scale_space_extrema(
    scale_images: list[torch.Tensor],
    scale_images_raw: list[torch.Tensor],
    sigmas: list[float],
    threshold: float,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray,
    fg_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Find local extrema across scale and 3D space.

    Parameters
    ----------
    scale_images     : σ²-normalized LoG/DoG stack — used to rank candidates in NMS.
    scale_images_raw : raw LoG/DoG stack — spatial local-max detection performed here.
    sigmas           : scale values (mm) for each level.
    threshold        : relative fraction of per-scale max (e.g. 0.005 = top 0.5%).
    origin, spacing, direction : full ANTs image affine (from _get_image_affine).
    fg_mask          : optional [1,1,nx,ny,nz] bool tensor; candidates outside it are dropped
                       (suppresses responses in zero-padded background).

    Returns
    -------
    np.ndarray [N, 5] — physical (x_mm, y_mm, z_mm, sigma_mm, |response|) in ANTs world space.
    """
    n_scales = len(scale_images)
    if n_scales < 3:
        return np.zeros((0, 5), dtype=np.float32)

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
        if fg_mask is not None:
            is_local_max = is_local_max & fg_mask

        mask_np = is_local_max.squeeze().cpu().numpy()
        idxs    = np.argwhere(mask_np)   # [N, 3] in native XYZ order (ix, iy, iz)

        if idxs.shape[0] == 0:
            continue

        # Full ANTs affine: physical = origin + direction @ (idx_XYZ * spacing)
        phys = vox_to_physical((origin, spacing, direction), idxs)  # [N, 3]
        sigma_col = np.full((phys.shape[0], 1), sigmas[s], dtype=np.float32)
        # response magnitude (σ²-normalised so scales are comparable) for ranking in NMS
        resp = scale_images[s].abs().squeeze().cpu().numpy()[tuple(idxs.T)].reshape(-1, 1)
        candidates.append(np.hstack([phys, sigma_col, resp]))

    if not candidates:
        return np.zeros((0, 5), dtype=np.float32)
    return np.vstack(candidates).astype(np.float32)



def _greedy_nms(pts: np.ndarray, min_dist_mm: float, max_kpts: int) -> np.ndarray:
    """Greedy spatial NMS.  Candidates are visited in decreasing order of
    response magnitude (column 4 if present, otherwise scale) so the selection
    is independent of array storage order.  Returns [K, 4] (x, y, z, sigma)."""
    if pts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    key = pts[:, 4] if pts.shape[1] >= 5 else pts[:, 3]
    order = np.lexsort((-pts[:, 3], -key))      # primary: response desc, secondary: scale desc
    pts = pts[order]
    kept: list[np.ndarray] = []
    for pt in pts:
        if len(kept) >= max_kpts:
            break
        if kept:
            dists = np.linalg.norm(np.array(kept)[:, :3] - pt[:3], axis=1)
            if dists.min() < min_dist_mm:
                continue
        kept.append(pt)
    if not kept:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(kept, dtype=np.float32)[:, :4]


def _get_image_affine(image):
    """Thin wrapper — delegates to syntx.landmarks.spatial.get_image_affine."""
    return get_image_affine(image)


def _foreground_mask(vol: torch.Tensor, level: float = 0.05) -> torch.Tensor:
    """Boolean [1,1,nx,ny,nz] foreground mask on the [0,1]-normalized volume."""
    return vol > level


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
    use_n4: bool = False,
    use_denoise: bool = True,
) -> np.ndarray:
    """
    Detect 3D blobs using Laplacian-of-Gaussian scale-space extrema.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        Input volume.
    sigma_min, sigma_max : float
        Range of Gaussian sigmas in **mm** (converted to per-axis voxel sigmas
        from the image spacing, so anisotropic images are treated isotropically
        in physical space).
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

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()   # mm
    fg = _foreground_mask(vol, FOREGROUND_LEVEL)

    # Build TWO parallel lists:
    #   log_raw    — σ²-normalized LoG  (passed for API compat, unused internally)
    #   log_unnorm — raw LoG (no σ² weight): used for spatial local-max
    log_raw   = []
    log_unnorm = []
    for sigma in sigmas:
        smoothed = _separable_gaussian3d(vol, _voxel_sigmas(sigma, spacing))
        lap = _laplacian3d(smoothed, spacing)
        log_raw.append(lap * (sigma ** 2))   # σ²-normalized (API compat)
        log_unnorm.append(lap)               # raw for detection

    pts = _scale_space_extrema(log_raw, log_unnorm, sigmas, threshold, origin, spacing, direction, fg)
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
    use_n4: bool = False,
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
        Range of Gaussian sigmas in **mm** (per-axis voxel sigmas derived from spacing).
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

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()   # mm
    fg = _foreground_mask(vol, FOREGROUND_LEVEL)

    # Gaussian blurs at isotropic *physical* scale (per-axis voxel sigmas)
    gaussians = [_separable_gaussian3d(vol, _voxel_sigmas(s, spacing)) for s in sigmas]

    # DoG = G(sigma_{i+1}) - G(sigma_i)
    dog_images = []     # σ²-normalized (API compat)
    dog_raw    = []     # raw DoG — used for detection
    dog_sigmas = []
    for i in range(len(sigmas) - 1):
        dog = gaussians[i + 1] - gaussians[i]
        dog_images.append(dog * (sigmas[i] ** 2))   # normalized
        dog_raw.append(dog)                          # raw
        dog_sigmas.append(math.sqrt(sigmas[i] * sigmas[i + 1]))

    pts = _scale_space_extrema(dog_images, dog_raw, dog_sigmas, threshold, origin, spacing, direction, fg)
    pts = _greedy_nms(pts, min_distance_mm, max_keypoints)
    return pts

