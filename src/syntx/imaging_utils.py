"""syntx.imaging_utils — small, dependency-free image-preprocessing helpers shared across
this ecosystem's registration call sites.

``cap_resolution_for_registration`` was found duplicated during a 2026-10-02 ecosystem-wide
synergy review: a canonical copy lived in ``antsxfunctional.imaging_utils`` (correct,
``interp_type=0``/linear), and a second, independently-written copy existed in
``syntx.motion_batched`` (``_cap_resolution``, with a real bug -- ``interp_type=1``/nearest-
neighbor, the wrong choice for a continuous-intensity image feeding an intensity-based
registration metric). Since syntx is the lower-level repo (antsxfunctional already depends
on it, never the reverse), the canonical implementation belongs here; antsxfunctional's
copy now re-exports this one instead of maintaining its own.
"""

from __future__ import annotations

from typing import Any

__all__ = ["cap_resolution_for_registration", "clip_cervical_spine_fov"]


def cap_resolution_for_registration(image: Any, max_resolution_mm: float) -> Any:
    """Downsample ``image`` so its finest voxel dimension is no smaller than
    ``max_resolution_mm``, never upsampling if it is already coarser.

    Purpose: many deformable registrations (atlas-to-T1 label transfer, population-template
    construction, etc.) do not need sub-2mm information to place labels or align shape
    correctly -- only rough-to-moderate anatomical correspondence. Registering at native
    high resolution (e.g. 0.5mm) is real, avoidable compute cost for no accuracy benefit at
    that stage. The estimated transform remains a continuous physical-space warp, valid to
    apply at any resolution afterward (e.g. warping a label image onto the caller's
    original, uncapped grid) -- only the registration optimisation itself runs at the
    capped resolution, not the final output.

    Parameters
    ----------
    image : ants.ANTsImage
    max_resolution_mm : float
        The coarsest allowed voxel spacing to register at (e.g. 2.0).

    Returns
    -------
    ants.ANTsImage
        ``image`` resampled to isotropic ``max_resolution_mm`` spacing (linear
        interpolation -- the correct choice for a continuous-intensity image feeding an
        intensity-based registration metric, not nearest-neighbor) if its native spacing
        is finer than that; otherwise ``image`` unchanged.
    """
    import ants

    spc = tuple(float(s) for s in image.spacing[:3])
    finest = min(spc)
    if finest >= max_resolution_mm:
        return image
    return ants.resample_image(image, [max_resolution_mm] * 3, interp_type=0)


def clip_cervical_spine_fov(
    image: Any,
    cutoff_mm_below_com: float = 75.0,
    background_value: float = 0.0,
) -> Any:
    """Zero out lower cervical spine and shoulder tissue below the cranial vault.

    In whole-head or neck clinical MR acquisitions, field-of-view extends significantly
    inferiorly (into C1-C7 cervical spine and clavicles), whereas cranial atlases (e.g.
    MIDA, SIAM) terminate near the jaw/mandible. Unconstrained 3D deformable registration
    stretches cranial structures downward to fill cervical tissue.

    This function isolates the cranial domain by zeroing out voxels situated more than
    ``cutoff_mm_below_com`` mm inferior (in physical LPS coordinate space, along the
    Superior-Inferior axis) relative to the cranial tissue Center of Mass.

    Parameters
    ----------
    image : ants.ANTsImage
        3D input image.
    cutoff_mm_below_com : float, default 75.0
        Distance in physical mm inferior to cranial Center of Mass below which voxels are zeroed.
    background_value : float, default 0.0
        Value assigned to clipped voxels.

    Returns
    -------
    ants.ANTsImage
        Image with inferior cervical spine zeroed out, retaining exact physical geometry.
    """
    import ants
    import numpy as np
    from .core.utils import normalize_image
    from .robust_affine import robust_center_of_mass

    if image.dimension != 3:
        return image

    norm = normalize_image(image, method='auto')
    com_pos, _ = robust_center_of_mass(norm, weighted=True)
    if com_pos is None:
        return image

    z_cutoff = float(com_pos[2] - cutoff_mm_below_com)

    shape = image.shape
    sp = image.spacing
    orig = image.origin
    d = np.array(image.direction).reshape(3, 3)

    ix, iy, iz = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    z_phys = orig[2] + ix * (d[0, 2] * sp[0]) + iy * (d[1, 2] * sp[1]) + iz * (d[2, 2] * sp[2])

    arr = image.numpy().copy()
    arr[z_phys < z_cutoff] = background_value

    return ants.from_numpy(arr, origin=image.origin, spacing=image.spacing, direction=image.direction)

