import torch
import torch.nn.functional as F
import math

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
    from .syn import separable_gaussian_filter
    
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
            smoothed = separable_gaussian_filter(image.movedim(1, -1), sig, spacing=spacing, sigma_mode=sigma_mode).movedim(-1, 1)
        else:
            smoothed = image
            
        if s > 1:
            level_img = F.interpolate(smoothed, scale_factor=1.0/s, mode=interp_mode, align_corners=True)
        else:
            level_img = smoothed
            
        pyramid.append(level_img)
        
    return pyramid
