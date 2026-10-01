"""
Curvature-based surface labels of an intensity image, and smooth channels made from them.

``compute_surface_classes`` labels voxels with the shape-operator classes of the intensity
iso-surfaces (``antstorch.weingarten_image_curvature(..., opt='characterize')``), grouped e.g.
into "gyral" (H > 0) and "sulcal" (H < 0) classes. ``generate_surface_channels`` turns a label
image into one float image per class (distance potential or smoothed indicator), usable as extra
channels for multi-channel registration. ``extract_sulcal_probability_map`` chains the two.

antstorch characterisation codes (H mean, K Gaussian curvature; thresholds |H| <= 1e-6,
|K| <= 1e-12 count as zero; voxels outside the mask, with intensity <= 0 or within 3 voxels of
the border are 0):

1 Peak (H > 0, K > 0); 2 Pit (H < 0, K > 0); 3 Saddle ridge (H > 0, K < 0);
4 Saddle valley (H < 0, K < 0); 5 Ridge (H > 0, K = 0); 6 Valley (H < 0, K = 0);
7 Flat (H = 0, K = 0); 8 Minimal (H = 0, K < 0).

The sign of H depends on the intensity gradient direction, so whether H > 0 means a gyral crest
depends on the image contrast; the gyral / sulcal names below follow the code's convention.
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
    Label voxels by the curvature class of the intensity iso-surface through them.

    Calls ``antstorch.weingarten_image_curvature(image, sigma, opt='characterize', mask,
    device)`` and regroups its codes (see the module docstring). 'gyral_sulcal' / 'sulcal_only'
    use codes 1-4 only (codes 5-8 -- ridge, valley, flat, minimal -- become 0); 'full' keeps all
    eight. The classes are curvature signs of *intensity* iso-surfaces, so which one is "gyral"
    depends on the contrast (bright tissue inside the fold, as for T1 white matter, is assumed).

    Parameters
    ----------
    image : ants.ANTsImage
        2-D or 3-D scalar image (antstorch handles 2-D by stacking it into a thin volume).
    sigma : float, default 1.5
        Gaussian scale (physical units, mm) of the gradient used for the curvature. antstorch
        replaces values <= 0.5 with 1.66.
    mask : ants.ANTsImage, optional
        Voxels to label (> 0). None: ``ants.get_mask(image)``. Voxels with intensity <= 0 are
        never labelled.
    grouping : {'gyral_sulcal', 'full', 'sulcal_only'}, default 'gyral_sulcal'
        - 'gyral_sulcal': 1 = codes 1 or 3 (H > 0), 2 = codes 2 or 4 (H < 0).
        - 'full': codes 1-8 kept as they are.
        - 'sulcal_only': 1 = codes 2 or 4.
        Any other value raises ValueError.
    device : str, optional
        Torch device for antstorch ('cuda', 'mps', 'cpu'); None uses
        ``antstorch.get_default_device()``.

    Returns
    -------
    ants.ANTsImage
        Integer label image (built from int32, so ANTs pixel type 'unsigned int') with the
        header of ``image``.
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
        for c in range(1, 9):
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
    Turn a label image into one smooth float image per label 1..num_classes.

    Parameters
    ----------
    class_image : ants.ANTsImage
        Integer label image (0 = background, classes 1..K).
    num_classes : int, optional
        Number of channels K. None: the maximum label in ``class_image``.
    mode : {'distance_potential', 'soft_prob'}, default 'distance_potential'
        - 'distance_potential': ``exp(-d / tau)``, with d the Euclidean distance (physical
          units, ``class_image.spacing``) to the nearest voxel of the class; 1 on the class,
          decaying away from it.
        - 'soft_prob': the 0/1 class indicator smoothed with ``ants.smooth_image(sigma=
          smoothing_sigma)``.
        Any other value raises ValueError (only once a non-empty class is reached).
    smoothing_sigma : float, default 1.0
        Smoothing sigma for 'soft_prob' (``ants.smooth_image`` default units: physical).
        Ignored in 'distance_potential'.
    tau : float, default 2.0
        Decay length (physical units) for 'distance_potential'. Ignored in 'soft_prob'.
    sigma : float, optional
        If given, overrides ``smoothing_sigma``.

    Returns
    -------
    list of ants.ANTsImage
        ``num_classes`` float images with the header of ``class_image``, element k-1 for label
        k. A label with no voxels gives an all-zero image.
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


def extract_sulcal_probability_map(
    image: ants.ANTsImage,
    curv_sigma: float = 1.5,
    prob_sigma: float = 1.0,
    device: Optional[str] = None
) -> ants.ANTsImage:
    """
    Smoothed indicator of the "sulcal" (H < 0) voxels of an image.

    ``compute_surface_classes(image, sigma=curv_sigma, grouping='gyral_sulcal')`` (mask from
    ``ants.get_mask``), then the class-2 channel of ``generate_surface_channels(mode=
    'soft_prob', smoothing_sigma=prob_sigma, num_classes=2)``.

    Parameters
    ----------
    image : ants.ANTsImage
        2-D or 3-D scalar image.
    curv_sigma : float, default 1.5
        Curvature scale (mm), passed as ``sigma`` to ``compute_surface_classes``.
    prob_sigma : float, default 1.0
        Smoothing sigma (physical units) of the class indicator.
    device : str, optional
        Torch device for the curvature computation; None uses the antstorch default.

    Returns
    -------
    ants.ANTsImage
        Float image in [0, 1] with the header of ``image`` (a smoothed indicator, not a
        calibrated probability).
    """
    classes = compute_surface_classes(
        image,
        sigma=curv_sigma,
        grouping='gyral_sulcal',
        device=device
    )
    channels = generate_surface_channels(
        classes,
        mode='soft_prob',
        smoothing_sigma=prob_sigma,
        num_classes=2
    )
    # Channel index 1 is sulcal fundus (class 2 in gyral_sulcal grouping)
    return channels[1]

