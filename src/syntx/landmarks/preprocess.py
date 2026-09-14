"""
syntx.landmarks.preprocess — Standard registration preprocessing for landmark detection
========================================================================================

Mirrors the preprocessing pipeline used in the syntx benchmark:

    MRI:  (optional N4 bias correction) → NLM denoising → foreground 2nd–98th pct normalization
    CT:   foreground 2nd–98th pct normalization ONLY  (no N4, no denoising)

CT is detected automatically from the presence of negative voxel values (Hounsfield units).

Public API
----------
preprocess_for_landmarks(image, use_n4, use_denoise, is_ct) -> ants.ANTsImage
is_ct_image(image) -> bool
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CT detection
# ---------------------------------------------------------------------------

def is_ct_image(image) -> bool:
    """
    Heuristic: CT images have Hounsfield-unit negative values (air ≈ −1000 HU).
    MRI images are non-negative by construction (magnitude images).

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray

    Returns
    -------
    bool  True if the image appears to be CT.
    """
    arr = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    # CT: at least 1% of non-zero voxels are negative (air/lung/background in HU)
    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        return False
    frac_neg = float((nonzero < 0).sum()) / nonzero.size
    return frac_neg > 0.01


# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess_for_landmarks(
    image,
    use_n4: bool = True,
    use_denoise: bool = True,
    is_ct: Optional[bool] = None,
    verbose: bool = False,
) -> "ants.ANTsImage":
    """
    Apply standard syntx benchmark preprocessing before landmark detection.

    Pipeline
    --------
    MRI (is_ct=False):
        1. N4 bias field correction  (if use_n4=True)
        2. NLM denoising             (if use_denoise=True, via antstorch)
        3. Foreground 2nd–98th percentile normalization → [0, 1]

    CT (is_ct=True):
        1. Foreground 2nd–98th percentile normalization → [0, 1] ONLY
           (no N4 — CT is already quantitative; no denoising — NLM is tuned
            for Rician MRI noise, not Poisson/CT noise)

    Parameters
    ----------
    image : ants.ANTsImage
        Input volume.
    use_n4 : bool
        Apply N4 bias field correction (MRI only).  Default True.
    use_denoise : bool
        Apply adaptive non-local means denoising (MRI only).  Default True.
    is_ct : bool | None
        Override CT/MRI detection.  If None, auto-detected via ``is_ct_image()``.
    verbose : bool
        Log preprocessing steps.

    Returns
    -------
    ants.ANTsImage
        Preprocessed image with intensities in [0, 1].
    """
    import ants
    from syntx.core.utils import normalize_image

    if not hasattr(image, "numpy"):
        raise TypeError(
            f"preprocess_for_landmarks expects an ants.ANTsImage, got {type(image)}"
        )

    # ── Auto-detect modality ─────────────────────────────────────────────────
    if is_ct is None:
        is_ct = is_ct_image(image)
    modality = "CT" if is_ct else "MRI"
    if verbose:
        logger.info("preprocess_for_landmarks: modality=%s", modality)

    img = image

    # ── N4 bias field correction (MRI only) ──────────────────────────────────
    if use_n4 and not is_ct:
        try:
            img = ants.n4_bias_field_correction(img)
            if verbose:
                logger.info("  [N4] applied")
        except Exception as e:
            logger.warning("  [N4] failed — skipping: %s", e)

    # ── NLM denoising (MRI only) ─────────────────────────────────────────────
    if use_denoise and not is_ct:
        try:
            import antstorch
            img = antstorch.denoise_image(
                img, shrink_factor=2, p=1, r=1, noise_model="Rician"
            )
            if verbose:
                logger.info("  [NLM] denoise applied (Rician, shrink=2, r=1)")
        except Exception as e:
            logger.warning("  [NLM] antstorch.denoise_image failed — skipping: %s", e)

    # ── Foreground 2nd–98th percentile normalization (all modalities) ────────
    img = normalize_image(img, method="auto")
    if verbose:
        logger.info("  [norm] foreground 2nd–98th pct → [0,1]")

    return img
