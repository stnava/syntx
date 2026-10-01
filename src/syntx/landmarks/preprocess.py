"""
Intensity preprocessing applied by the landmark detectors before feature extraction.

MRI: optional N4 (off by default), NLM denoising (``antstorch.denoise_image``), then
``syntx.core.utils.normalize_image(method='auto')`` to [0, 1]. CT: no N4 / denoising; either a
fixed HU window or the same 'auto' normalisation. CT is detected by ``is_ct_image`` (negative
voxel values).
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
    Guess whether an image is CT: True if more than 1% of its non-zero voxels are negative.

    Based on CT air / lung being negative in Hounsfield units while magnitude MRI is
    non-negative. A processed MRI with negative values (e.g. z-scored) is classified as CT.

    Parameters
    ----------
    image : ants.ANTsImage or np.ndarray

    Returns
    -------
    bool
        False for an all-zero image.
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
    ct_window: Optional[str] = None,
    channel_idx: int = 0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> "ants.ANTsImage":
    """
    Preprocess an image for landmark detection: (N4), (NLM denoising), intensity to [0, 1].

    Steps, in order:

    1. A 4-D input is reduced to the volume ``channel_idx`` along axis 3.
    2. MRI and ``use_n4``: ``antstorch.n4_bias_field_correction`` on the tensor (mask =
       intensity > 0.01, shrink 4, 3 x 50 iterations); on any exception, ``ants.
       n4_bias_field_correction(shrink_factor=4)``; if that fails too, a warning is logged and
       N4 is skipped. The antstorch path assumes a 3-D image (2-D falls through to ANTs).
    3. MRI and ``use_denoise``: ``antstorch.denoise_image(shrink_factor=2, p=1, r=1,
       noise_model='Rician')``; on failure a warning is logged and the step is skipped.
    4. CT with ``ct_window`` 'soft_tissue' [-120, 250] HU, 'lung' [-1000, -200] HU or 'bone'
       [100, 1500] HU: clip to the window and rescale to [0, 1]. Otherwise (MRI, or CT with
       any other ``ct_window``): ``normalize_image(img, method='auto')``, which clips to
       entropy-selected percentiles of the voxels > 0 and rescales to [0, 1], and returns an
       image already in [0, 1] (max >= 0.5) unchanged apart from clipping.

    Parameters
    ----------
    image : ants.ANTsImage
        Input image (2-D, 3-D or 4-D). Anything without ``.numpy`` raises TypeError.
    use_n4 : bool, default False
        N4 bias correction (MRI only).
    use_denoise : bool, default True
        NLM denoising (MRI only).
    is_ct : bool, optional
        CT / MRI choice; None uses ``is_ct_image(image)``.
    ct_window : {None, 'soft_tissue', 'lung', 'bone'}, default None
        Fixed HU window for CT; ignored for MRI. Unknown values fall back to 'auto'.
    channel_idx : int, default 0
        Volume index for 4-D input.
    device : str, optional
        Torch device for the antstorch steps; None picks 'mps', then 'cuda', then 'cpu'.
    verbose : bool, default False
        Log each step at INFO level (``logging``, not print).

    Returns
    -------
    ants.ANTsImage
        Preprocessed image, intensities in [0, 1], same geometry as the (sliced) input.
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
    if hasattr(img, "dimension") and img.dimension == 4:
        img = ants.slice_image(img, axis=3, idx=channel_idx)

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

    # ── Intensity normalization ──────────────────────────────────────────────
    if is_ct and ct_window == "soft_tissue":
        # Abdominal / pelvic soft-tissue CT window: [-120, 250] HU
        arr = img.numpy()
        arr_clipped = np.clip(arr, -120.0, 250.0)
        arr_norm = (arr_clipped - (-120.0)) / (250.0 - (-120.0))
        img = img.new_image_like(arr_norm)
        if verbose:
            logger.info("  [norm] CT soft-tissue window [-120, 250] HU → [0,1]")
    elif is_ct and ct_window == "lung":
        # Thoracic lung CT window: [-1000, -200] HU
        arr = img.numpy()
        arr_clipped = np.clip(arr, -1000.0, -200.0)
        arr_norm = (arr_clipped - (-1000.0)) / (-200.0 - (-1000.0))
        img = img.new_image_like(arr_norm)
        if verbose:
            logger.info("  [norm] CT lung window [-1000, -200] HU → [0,1]")
    elif is_ct and ct_window == "bone":
        # Bone CT window: [100, 1500] HU
        arr = img.numpy()
        arr_clipped = np.clip(arr, 100.0, 1500.0)
        arr_norm = (arr_clipped - 100.0) / (1500.0 - 100.0)
        img = img.new_image_like(arr_norm)
        if verbose:
            logger.info("  [norm] CT bone window [100, 1500] HU → [0,1]")
    else:
        img = normalize_image(img, method="auto")
        if verbose:
            logger.info("  [norm] foreground 2nd–98th pct → [0,1]")

    return img
