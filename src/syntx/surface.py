"""
surface.py — Topographic Surface Classification & Label Generation
===================================================================

This module provides tools to extract discrete differential geometric surface
classifications from 3D volumetric images using the Weingarten shape operator,
and to convert those discrete classifications into smooth, continuous,
differentiable multi-channel representations (soft probabilities and distance
potentials) for label-wise registration in `syntx.syn`.

Topographic Classification Categories (Weingarten Shape Operator)
-----------------------------------------------------------------
1: Peak          (H > 0, K > 0) — Convex spherical cap / gyral crown
2: Pit           (H < 0, K > 0) — Concave spherical cup / deep sulcal pit
3: Saddle Ridge  (H > 0, K < 0) — Upward-curving saddle / gyral flank
4: Saddle Valley (H < 0, K < 0) — Downward-curving saddle / sulcal trench
5: Ridge         (H > 0, K = 0) — Developable cylindrical ridge
6: Valley        (H < 0, K = 0) — Developable cylindrical valley
7: Flat          (H = 0, K = 0) — Planar surface
8: Minimal       (H = 0, K < 0) — Minimal surface
"""

import numpy as np
from typing import Optional, Union, List
from scipy import ndimage as ndi
import ants
import antstorch


def compute_surface_classes(
    image: ants.ANTsImage,
    sigma: float = 1.5,
    mask: Optional[ants.ANTsImage] = None,
    grouping: str = 'gyral_sulcal',
    device: Optional[str] = None
) -> ants.ANTsImage:
    """
    Computes discrete differential geometric surface classifications using the
    Weingarten shape operator on GPU/MPS via `antstorch`.

    Parameters
    ----------
    image : ants.ANTsImage
        Input 3D scalar anatomical image.
    sigma : float, default 1.5
        Gaussian smoothing scale (mm) for image gradient and Hessian estimation.
    mask : ants.ANTsImage, optional
        Foreground brain mask. If None, computed automatically via `ants.get_mask(image)`.
    grouping : {'gyral_sulcal', 'full', 'sulcal_only'}, default 'gyral_sulcal'
        Classification grouping strategy:
        - 'gyral_sulcal': 2 classes:
            * Class 1: Gyral Crests (Peaks [1] + Saddle Ridges [3], convex H > 0)
            * Class 2: Sulcal Fundi (Pits [2] + Saddle Valleys [4], concave H < 0)
        - 'full': 4 distinct classes (1: Peak, 2: Pit, 3: Saddle Ridge, 4: Saddle Valley)
        - 'sulcal_only': 1 class:
            * Class 1: Sulcal Fundi (Pits [2] + Saddle Valleys [4])
    device : str, optional
        Compute device ('cuda', 'mps', or 'cpu'). Defaults to auto-detection.

    Returns
    -------
    ants.ANTsImage
        Discrete integer label image matching spatial geometry of `image`.
    """
    if mask is None:
        mask = ants.get_mask(image)

    # Extract 8-class Weingarten characterization
    raw_curv = antstorch.weingarten_image_curvature(
        image,
        sigma=sigma,
        opt='characterize',
        mask=mask,
        device=device
    )
    raw_np = raw_curv.numpy()

    out_np = np.zeros_like(raw_np, dtype=np.int32)

    if grouping == 'gyral_sulcal':
        # Class 1: Gyral Crests (Peak=1, Saddle Ridge=3)
        gyral_mask = (raw_np == 1) | (raw_np == 3)
        # Class 2: Sulcal Fundi (Pit=2, Saddle Valley=4)
        sulcal_mask = (raw_np == 2) | (raw_np == 4)

        out_np[gyral_mask] = 1
        out_np[sulcal_mask] = 2

    elif grouping == 'sulcal_only':
        sulcal_mask = (raw_np == 2) | (raw_np == 4)
        out_np[sulcal_mask] = 1

    elif grouping == 'full':
        for c in [1, 2, 3, 4]:
            out_np[raw_np == c] = c
    else:
        raise ValueError(f"Unknown grouping: '{grouping}'. Choose from 'gyral_sulcal', 'full', 'sulcal_only'.")

    return ants.copy_image_info(image, ants.from_numpy(out_np))


def generate_surface_channels(
    class_image: ants.ANTsImage,
    num_classes: Optional[int] = None,
    mode: str = 'distance_potential',
    smoothing_sigma: float = 1.0,
    tau: float = 2.0,
    sigma: Optional[float] = None
) -> List[ants.ANTsImage]:
    """
    Converts a discrete surface classification label image into smooth, continuous,
    differentiable float channels for multi-channel registration in `syntx.syn`.

    Parameters
    ----------
    class_image : ants.ANTsImage
        Discrete integer surface label image (classes 1..K).
    num_classes : int, optional
        Number of classes. If None, derived from max label value in `class_image`.
    mode : {'distance_potential', 'soft_prob'}, default 'distance_potential'
        Channel representation mode:
        - 'distance_potential': Exponential distance potential P = exp(-EDT / tau).
          Provides long-range, continuous spatial gradient forces pulling opposing
          banks toward the skeleton.
        - 'soft_prob': Gaussian-smoothed binary membership indicator.
    smoothing_sigma : float, default 1.0
        Smoothing sigma in mm when `mode='soft_prob'`.
    tau : float, default 2.0
        Distance decay bandwidth in mm when `mode='distance_potential'`.
    sigma : float, optional
        Convenience alias for `smoothing_sigma`.

    Returns
    -------
    list of ants.ANTsImage
        List of continuous float channels, one per non-zero surface class.
    """
    if sigma is not None:
        smoothing_sigma = sigma

    c_arr = class_image.numpy()
    if num_classes is None:
        num_classes = int(np.max(c_arr))

    sp = class_image.spacing
    channels = []

    for k in range(1, num_classes + 1):
        mask_k = (c_arr == k)

        if not np.any(mask_k):
            # Empty class: zero image
            zero_arr = np.zeros_like(c_arr, dtype=np.float32)
            channels.append(ants.copy_image_info(class_image, ants.from_numpy(zero_arr)))
            continue

        if mode == 'distance_potential':
            # Euclidean distance from class k
            d = ndi.distance_transform_edt(~mask_k, sampling=sp)
            pot = np.exp(-d / float(tau)).astype(np.float32)
            channels.append(ants.copy_image_info(class_image, ants.from_numpy(pot)))

        elif mode == 'soft_prob':
            # Gaussian smoothed binary membership
            mask_img = ants.copy_image_info(class_image, ants.from_numpy(mask_k.astype(np.float32)))
            smooth_img = ants.smooth_image(mask_img, sigma=smoothing_sigma)
            channels.append(smooth_img)

        else:
            raise ValueError(f"Unknown mode: '{mode}'. Choose from 'distance_potential', 'soft_prob'.")

    return channels
