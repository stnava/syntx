"""
syntx.landmarks.preprocess — Standard registration preprocessing for landmark detection
========================================================================================

Mirrors the preprocessing pipeline used in the syntx benchmark:

    MRI:  (optional N4 bias correction, OFF by default) → NLM denoising → foreground 2nd–98th pct normalization
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
    use_n4: bool = False,
    use_denoise: bool = True,
    is_ct: Optional[bool] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> "ants.ANTsImage":
    """
    Apply standard syntx benchmark preprocessing before landmark detection.

    Pipeline
    --------
    MRI (is_ct=False):
        1. N4 bias field correction  (if use_n4=True, via antstorch on GPU/MPS)
        2. NLM denoising             (if use_denoise=True, via antstorch on GPU/MPS)
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
    device : str | None
        Torch compute device ('mps', 'cuda', 'cpu'). Auto-detected if None.
    verbose : bool
        Log preprocessing steps.

    Returns
    -------
    ants.ANTsImage
        Preprocessed image with intensities in [0, 1].
    """
    import ants
    import torch
    from syntx.core.utils import normalize_image

    if not hasattr(image, "numpy"):
        raise TypeError(
            f"preprocess_for_landmarks expects an ants.ANTsImage, got {type(image)}"
        )

    # Resolve device (MPS > CUDA > CPU)
    if device is None:
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    else:
        dev = device

    # ── Auto-detect modality ─────────────────────────────────────────────────
    if is_ct is None:
        is_ct = is_ct_image(image)
    modality = "CT" if is_ct else "MRI"
    if verbose:
        logger.info("preprocess_for_landmarks: modality=%s, device=%s", modality, dev)

    img = image

    # ── N4 bias field correction (MRI only, via ANTsTorch on GPU/MPS) ────────
    if use_n4 and not is_ct:
        n4_applied = False
        try:
            import antstorch
            arr = img.numpy().astype(np.float32)
            tensor = torch.from_numpy(arr.transpose(2, 1, 0)).unsqueeze(0).unsqueeze(0).to(dev)
            mask = (tensor > 0.01).to(tensor.dtype)
            corrected_tensor = antstorch.n4_bias_field_correction(
                tensor,
                mask=mask,
                shrink_factor=4,
                convergence={"iters": [50, 50, 50], "tol": 1e-6},
            )
            corrected_arr = corrected_tensor.squeeze().detach().cpu().numpy().transpose(2, 1, 0)
            img = ants.from_numpy(
                corrected_arr,
                origin=img.origin,
                spacing=img.spacing,
                direction=img.direction,
            )
            n4_applied = True
            if verbose:
                logger.info("  [N4] antstorch.n4_bias_field_correction applied on %s", dev)
        except Exception as e:
            logger.warning("  [N4] antstorch failed (%s) — trying fallback", e)

        if not n4_applied:
            try:
                img = ants.n4_bias_field_correction(img, shrink_factor=4)
                if verbose:
                    logger.info("  [N4] ants fallback applied (shrink=4)")
            except Exception as e:
                logger.warning("  [N4] fallback failed — skipping: %s", e)

    # ── NLM denoising (MRI only, via ANTsTorch on GPU/MPS) ───────────────────
    if use_denoise and not is_ct:
        try:
            import antstorch
            img = antstorch.denoise_image(
                img, shrink_factor=2, p=1, r=1, noise_model="Rician", device=dev
            )
            if verbose:
                logger.info("  [NLM] antstorch.denoise_image applied on %s (Rician, shrink=2, r=1)", dev)
        except Exception as e:
            logger.warning("  [NLM] antstorch.denoise_image failed — skipping: %s", e)

    # ── Foreground 2nd–98th percentile normalization (all modalities) ────────
    img = normalize_image(img, method="auto")
    if verbose:
        logger.info("  [norm] foreground 2nd–98th pct → [0,1]")

    return img
