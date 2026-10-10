"""TEST-ONLY golden references: the incumbent ``build_image_pyramid`` and ``build_anti_aliased_pyramid`` exactly as they were in
``syntx/pyramid.py`` before Pass 3b migrated their callers (code copied verbatim; only the imports are added). The central module
must keep reproducing them under the ``legacy_interp`` / ``legacy_floor16`` presets (tests/test_pyramid_central.py).
Known defects, kept on purpose: a thin axis can give a 0-voxel output or a crash; the anti-aliased one uses a zero-padded kernel."""
import math
from typing import Sequence

import torch
import torch.nn.functional as F


def build_image_pyramid(image, spacing, levels, smoothing_sigmas=None, sigma_mode='voxel'):
    """
    Constructs a multi-resolution image pyramid, applying Gaussian smoothing before downsampling.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape `(B, C, *spatial)`.
    spacing : tuple or list
        Physical spacing of the image voxels.
    levels : list of int
        Downsampling factors for each level (e.g. `[8, 4, 2, 1]`).
    smoothing_sigmas : list of float, optional
        Gaussian smoothing sigmas for each level. If None, derived as `log2(scale)`.
    sigma_mode : str
        'voxel' or 'physical'.
        
    Returns
    -------
    list of torch.Tensor
        List of image tensors for each pyramid level.
    """
    from syntx.syn import separable_gaussian_filter
    
    dim = image.dim() - 2
    interp_mode = 'bilinear' if dim == 2 else 'trilinear'
    
    if smoothing_sigmas is None:
        smoothing_sigmas = [float(math.log2(s)) if s > 1 else 0.0 for s in levels]
    elif isinstance(smoothing_sigmas, (int, float)):
        smoothing_sigmas = [float(smoothing_sigmas)] * len(levels)
    elif len(smoothing_sigmas) != len(levels):
        raise ValueError(f"Length of smoothing_sigmas ({len(smoothing_sigmas)}) must match levels ({len(levels)})")
        
    pyramid = []
    for level_idx, s in enumerate(levels):
        sig = float(smoothing_sigmas[level_idx])
        if sig > 0.0:
            smoothed = separable_gaussian_filter(
                image.movedim(1, -1), sig, spacing=spacing if sigma_mode == 'physical' else None,
                sigma_mode=sigma_mode).movedim(-1, 1)
        else:
            smoothed = image
            
        if s > 1:
            level_img = F.interpolate(smoothed, scale_factor=1.0/s, mode=interp_mode, align_corners=True)
        else:
            level_img = smoothed
            
        pyramid.append(level_img)
        
    return pyramid


def build_anti_aliased_pyramid(
    image: torch.Tensor,
    levels: Sequence[int],
    truncated: float = 2.0,
    min_size: int = 16,
) -> list:
    """
    Constructs an anti-aliased image pyramid using proportional compact Gaussian filtering.
    Operates directly on channel-first tensors (B, C, *spatial) with zero transposition overhead.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape `(B, C, *spatial)`.
    levels : list or tuple of int
        Downsampling scale factors (e.g. `[4, 2, 1]`).
    truncated : float
        Gaussian kernel truncation multiplier.
    min_size : int
        Minimum spatial dimension allowed at coarse scales.
        
    Returns
    -------
    list of torch.Tensor
        Multi-resolution tensors ordered according to `levels`.
    """
    from syntx.core.smoothing import gaussian_1d_compact, separable_1d_filter
    dim = image.dim() - 2
    interp_mode = 'bilinear' if dim == 2 else 'trilinear'
    full_shape = list(image.shape[2:])

    pyramid = []
    for s in levels:
        if s <= 1:
            pyramid.append(image)
            continue
        target_shape = [max(int(sz / s), min_size) for sz in full_shape]
        sigmas = [0.5 * (sz / szdown) for sz, szdown in zip(full_shape, target_shape)]
        gaussians = [gaussian_1d_compact(sig, truncated=truncated, device=image.device, dtype=image.dtype) for sig in sigmas]
        filtered = separable_1d_filter(image, gaussians)
        down = F.interpolate(filtered, size=target_shape, mode=interp_mode, align_corners=True)
        pyramid.append(down)
    return pyramid
