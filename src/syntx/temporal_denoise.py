"""syntx.temporal_denoise -- raising per-frame SNR in a 4D time series via a Gaussian-
weighted temporal moving average, centered at each frame.

Motivated by a real finding on dynamic FDG-PET data (2026-10-04): a single dynamic PET
frame is a short, low-count acquisition -- genuinely noisy, on top of its real tracer-
uptake contrast drift over time (see ``syntx.motion_reference`` and
``docs/ADAPTIVE_MOTION_CORRECTION_AND_RECOVERY.html`` for the full context). That per-
frame noise, not the uptake drift itself, was found to be the dominant cause of
erratic intensity-based (Mattes-MI) rigid registration on this data: replacing each
frame with a Gaussian-weighted average of its own temporal neighbors (narrow enough
that real head position and the uptake curve itself barely change across the window,
wide enough that uncorrelated voxel noise averages down) stabilized BOTH an intensity-
based registration (0-9.4mm erratic -> 0-1.3mm stable) and an independent feature-based
one (``syntx.landmarks.detect_sift3d``, 0-9.4mm-equivalent erratic/unfittable -> 0-0.85mm
stable) on the same real frames -- two unrelated mechanisms converging after the same
denoising step is itself evidence the result reflects something real, not an artifact of
either method alone.

This is a general per-frame SNR primitive, not a PET-specific one: any 4D series whose
frame-to-frame real motion is small relative to the temporal window used (true of nearly
any clinical acquisition at a window of a few frames) benefits from it before expensive
per-frame processing (registration, segmentation, keypoint detection) that is sensitive
to voxel-level noise.
"""

from __future__ import annotations

from typing import Optional

import ants
import numpy as np


def gaussian_temporal_average(
    image4d: ants.ANTsImage,
    center: Optional[int] = None,
    sigma: float = 1.5,
    half_window: int = 3,
) -> ants.ANTsImage:
    """Gaussian-weighted average of ``image4d``'s frames in
    ``[center - half_window, center + half_window]``, clipped to the series' valid
    frame range, weighted by ``exp(-k^2 / (2 sigma^2))`` for offset ``k`` from
    ``center``.

    Parameters
    ----------
    image4d : ants.ANTsImage
        4D time series (time is the last axis).
    center : int, optional
        Frame index to center the Gaussian window on. If None, every frame is
        smoothed with its own window (see ``smooth_all_frames``, which this delegates
        to internally for that case) and the full smoothed 4D series is returned.
    sigma : float, default 1.5
        Gaussian standard deviation, in units of FRAMES (not physical time) -- scale
        this to the acquisition's actual frame interval if a specific physical-time
        window is wanted (e.g. a 1-second-per-frame series and a desired ~3s window
        implies roughly this same default).
    half_window : int, default 3
        Window half-width in frames; the window is clipped at the series' boundaries
        (asymmetric weighting there, not wrapping or reflecting).

    Returns
    -------
    ants.ANTsImage
        3D (single averaged frame, if ``center`` given) or 4D (if ``center`` is None),
        same spatial grid as ``image4d``.
    """
    if center is None:
        return smooth_all_frames(image4d, sigma=sigma, half_window=half_window)

    arr = image4d.numpy()
    T = arr.shape[-1]
    if not (0 <= center < T):
        raise ValueError(f"center={center} out of range for a {T}-frame series.")

    lo, hi = max(0, center - half_window), min(T - 1, center + half_window)
    idxs = np.arange(lo, hi + 1)
    w = np.exp(-0.5 * ((idxs - center) / sigma) ** 2)
    w /= w.sum()
    smoothed = np.tensordot(arr[..., idxs], w, axes=([-1], [0])).astype(np.float32)

    frame0 = ants.slice_image(image4d, axis=image4d.dimension - 1, idx=0)
    return frame0.new_image_like(smoothed)


def smooth_all_frames(
    image4d: ants.ANTsImage,
    sigma: float = 1.5,
    half_window: int = 3,
) -> ants.ANTsImage:
    """Apply ``gaussian_temporal_average`` independently to every frame of
    ``image4d`` (each frame gets its OWN centered window), returning a full 4D series
    of the same shape -- the "denoise every frame, then run the exact same per-frame
    pipeline" use case (e.g. before motion correction or keypoint detection).
    """
    arr = image4d.numpy()
    T = arr.shape[-1]
    out = np.empty_like(arr, dtype=np.float32)
    for t in range(T):
        lo, hi = max(0, t - half_window), min(T - 1, t + half_window)
        idxs = np.arange(lo, hi + 1)
        w = np.exp(-0.5 * ((idxs - t) / sigma) ** 2)
        w /= w.sum()
        out[..., t] = np.tensordot(arr[..., idxs], w, axes=([-1], [0]))
    return image4d.new_image_like(out)
