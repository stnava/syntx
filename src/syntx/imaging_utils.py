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

__all__ = ["cap_resolution_for_registration"]


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
