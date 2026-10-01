"""
syntx.diagnose — guess modality, body part and intensity domain of an image (or a pair).

Used by ``syntx.auto_reg`` (through ``syntx.policy.synthesize_policy``) to pick registration
settings. Two tiers:

- Tier 2 (tried first when ``fast=False``): the 3-D ResNet-10 classifier of
  ``syntx.classifier``, loaded from ``src/syntx/models/diagnostic_resnet10_3d.pth``. That file
  is not shipped in the repository (``scripts/train_diagnostic_classifier.py`` writes it); when
  it is missing, or the model fails, Tier 1 is used and ``details["tier2"]`` says why
  ("weights_missing", "error: ..." with a warning, "skipped (...)", or "used").
- Tier 1: fixed-threshold heuristics on the intensity range / histogram and the physical
  extent.

Values actually produced:

- modality: "CT", "MRI_T1", "MRI_T2" (the classifier can also return "MRI_FLAIR", "MRI_ADC");
- body part: "BRAIN", "THORAX", "ABDOMEN", "PELVIS", "UNKNOWN" (the classifier can also
  return "HEART");
- intensity domain: "HOUNSFIELD", "NORMALIZED_01", "POSITIVE_FLOAT" (non-negative, beyond
  1.05) or "SIGNED_FLOAT" (non-HU data with negative values, e.g. z-scored);
- pair relationship: "MONO_MODAL_INTRA" (same modality), "MULTI_CONTRAST" (two different MRI
  labels), "CROSS_MODAL".
"""

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
import ants


@dataclass
class ImageDiagnosis:
    """Diagnosis of one image (returned by ``diagnose_image``).

    Attributes
    ----------
    modality : str
        "CT", "MRI_T1", "MRI_T2" (classifier only: "MRI_FLAIR", "MRI_ADC").
    body_part : str
        "BRAIN", "THORAX", "ABDOMEN", "PELVIS", "HEART" (classifier only) or "UNKNOWN".
    intensity_domain : str
        "HOUNSFIELD", "NORMALIZED_01", "POSITIVE_FLOAT" or "SIGNED_FLOAT".
    hu_min, hu_max, hu_mean : float
        Minimum / maximum / mean voxel intensity, for every modality (HU only for CT).
    air_ratio, bone_ratio, soft_tissue_ratio : float
        Tier-1 CT only (else 0.0): fraction of voxels in [-1050, -400], >= 300 and [20, 80].
    spatial_extent_mm : tuple of float
        Array shape times spacing per axis (ANTs axis order for an ANTsImage; spacing 1 for
        tensors / arrays).
    dimension : int
        ``image.dimension`` for ANTsImage, ``ndim`` for tensors / arrays.
    confidence : float
        Fixed per-rule value for Tier 1 (0.70-0.90); mean of the two top softmax
        probabilities for the classifier.
    source : str
        "deep_resnet10_tier2", "statistical_hu_tier1" or "statistical_tier1_fallback".
    details : dict
        Rule inputs: always ``header_hint`` (extent, spacing, shape, anisotropy for >= 3-D)
        and ``tier2`` (why the classifier was or was not used);
        CT tier 1: ``fat_ratio``; non-CT tier 1: foreground percentiles ``p25``, ``p50``,
        ``p75``, ``p98``; classifier: per-class probabilities and the two confidences.
    """
    modality: str
    body_part: str
    intensity_domain: str
    hu_min: float = 0.0
    hu_max: float = 0.0
    hu_mean: float = 0.0
    air_ratio: float = 0.0
    bone_ratio: float = 0.0
    soft_tissue_ratio: float = 0.0
    spatial_extent_mm: Tuple[float, ...] = field(default_factory=tuple)
    dimension: int = 3
    confidence: float = 1.0
    source: str = "statistical_heuristic"
    details: Dict[str, Any] = field(default_factory=dict)

    def is_ct(self) -> bool:
        """True if modality == "CT"."""
        return self.modality == "CT"

    def is_mri(self) -> bool:
        """True if modality starts with "MRI"."""
        return self.modality.startswith("MRI")

    def is_brain(self) -> bool:
        """True if body_part == "BRAIN"."""
        return self.body_part == "BRAIN"

    def is_thorax(self) -> bool:
        """True if body_part == "THORAX"."""
        return self.body_part == "THORAX"

    def is_abdomen(self) -> bool:
        """True if body_part == "ABDOMEN"."""
        return self.body_part == "ABDOMEN"

    def is_pelvis(self) -> bool:
        """True if body_part == "PELVIS"."""
        return self.body_part == "PELVIS"

    def is_heart(self) -> bool:
        """True if body_part == "HEART"."""
        return self.body_part == "HEART"


@dataclass
class PairDiagnosis:
    """Diagnosis of a fixed / moving pair (returned by ``diagnose_pair``).

    Attributes
    ----------
    fixed, moving : ImageDiagnosis
    relationship : str
        "MONO_MODAL_INTRA" (same modality label), "MULTI_CONTRAST" (both MRI, different labels)
        or "CROSS_MODAL".
    is_same_anatomy : bool
        Same body part, and not "UNKNOWN".
    is_same_modality : bool
        Same modality label.
    recommended_policy : any, default None
        Not filled by ``diagnose_pair``; use ``syntx.policy.synthesize_policy``.
    confidence : float
        Minimum of the two image confidences.
    """
    fixed: ImageDiagnosis
    moving: ImageDiagnosis
    relationship: str
    is_same_anatomy: bool
    is_same_modality: bool
    recommended_policy: Optional[Any] = None
    confidence: float = 1.0


def _extract_numpy(image: Union[ants.ANTsImage, torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, Tuple[float, ...], int]:
    """Return ``(float32 array, spacing, dimension)``; spacing is all 1.0 for tensors / arrays.

    Raises TypeError for other input types."""
    if isinstance(image, ants.ANTsImage):
        arr = image.numpy().astype(np.float32)
        spacing = tuple(float(s) for s in image.spacing)
        dim = image.dimension
    elif isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy().astype(np.float32)
        spacing = (1.0,) * arr.ndim
        dim = arr.ndim
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        spacing = (1.0,) * arr.ndim
        dim = arr.ndim
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return arr, spacing, dim


def _non_hu_domain(vmin: float, vmax: float) -> str:
    """Intensity domain of non-HU data."""
    if vmin < -1e-4:
        return "SIGNED_FLOAT"
    return "NORMALIZED_01" if vmax <= 1.05 else "POSITIVE_FLOAT"


def diagnose_image(image: Union[ants.ANTsImage, torch.Tensor, np.ndarray], fast: bool = False) -> ImageDiagnosis:
    """
    Guess modality, body part and intensity domain of one image.

    The data are "HU-like" when min <= -750 and max >= 200.

    Classifier (Tier 2) is tried only when ``fast=False``, the image has >= 3 dimensions, the
    first three axes have >= 40 voxels, the largest physical extent is >= 80 mm, and the weights
    file ``syntx/models/diagnostic_resnet10_3d.pth`` exists (it is not in the repository); it
    runs on the CPU. Its modality is then overridden: HU-like data -> "CT"; non-HU data with
    min >= 0 predicted "CT" -> "MRI_T1". If the classifier fails, a warning is issued and Tier 1
    is used; ``details["tier2"]`` records the outcome in every case.

    Tier 1, HU-like data: modality "CT"; body part from voxel fractions -- centre-box (middle
    half of each axis) air fraction >= 0.20 -> THORAX; else centre soft tissue [15, 85] HU >=
    0.35 -> ABDOMEN; else bone >= 0.04, soft tissue >= 0.15 and centre air < 0.10 -> BRAIN; else
    soft tissue >= 0.15 and fat [-120, -40] >= 0.02 -> ABDOMEN; else ABDOMEN (3-D) / UNKNOWN.

    Tier 1, otherwise: modality "MRI_T2" if, over foreground voxels (> 2 % of max) and when
    there are more than 100 of them, (p98 - p50) / (p50 - p25) > 3, else "MRI_T1". Body part
    from the shape only: 4-D with >= 4 last-axis entries -> BRAIN; 4-D with exactly 2 -> PELVIS;
    2-D -> BRAIN; 3-D with largest extent < 260 mm -> BRAIN; else UNKNOWN.

    Parameters
    ----------
    image : ANTsImage, torch.Tensor or numpy.ndarray
        Scalar image (2-D, 3-D or 4-D). Tensors / arrays are treated as spacing 1 (so
        "extent in mm" is the voxel count).
    fast : bool, default False
        True skips the classifier (Tier 1 only).

    Returns
    -------
    ImageDiagnosis
        See the class for the meaning of each field; ``source`` says which tier decided.

    """
    arr, spacing, dim = _extract_numpy(image)

    # Basic dynamic range statistics
    arr_flat = arr.ravel()
    vmin = float(np.min(arr_flat))
    vmax = float(np.max(arr_flat))
    vmean = float(np.mean(arr_flat))
    total_vox = max(len(arr_flat), 1)

    # Physical spatial extent (in mm if spacing available)
    spatial_extent = tuple(float(s * sz) for s, sz in zip(spacing, arr.shape))

    # Basic physical Hounsfield check (calibrated air -1000 HU, water 0 HU, bone > 500 HU)
    is_hu = (vmin <= -750.0) and (vmax >= 200.0)

    # Secondary header metadata hints (NON-DECISIVE)
    header_hint = {
        "spatial_extent_mm": spatial_extent,
        "spacing": tuple(float(s) for s in spacing),
        "shape": tuple(int(s) for s in arr.shape),
    }
    if len(spacing) >= 3 and len(arr.shape) >= 3:
        min_sp = min(spacing[:3])
        max_sp = max(spacing[:3])
        header_hint["anisotropy"] = float(max_sp / max(min_sp, 1e-5))

    # Tier 2: Deep 3D ResNet-10 Multi-Task Classification (PRIMARY AUTHORITATIVE DECISION)
    # The 3D ResNet evaluates normalized, resampled (64, 64, 64) voxel arrays without any
    # header dimensions, spacing, or orientation, guaranteeing decisions are ML-driven from voxels.
    weights_path = os.path.join(os.path.dirname(__file__), "models", "diagnostic_resnet10_3d.pth")
    if fast:
        tier2_status = "skipped (fast=True)"
    elif not (dim >= 3 and all(s >= 40 for s in arr.shape[:3]) and max(spatial_extent) >= 80.0):
        tier2_status = "skipped (needs >= 3-D, >= 40 voxels per axis, >= 80 mm extent)"
    elif not os.path.exists(weights_path):
        tier2_status = "weights_missing"
    else:
        tier2_status = None
        try:
            from .classifier import DiagnosticClassifier3D, predict_diagnosis_deep
            model = DiagnosticClassifier3D()
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            deep_res = predict_diagnosis_deep(model, image, device="cpu")
        except Exception as e:
            tier2_status = f"error: {type(e).__name__}: {e}"
            warnings.warn(f"syntx.diagnose: deep classifier failed ({tier2_status}); "
                          "using the Tier-1 heuristics")
        else:
            pred_mod = str(deep_res["modality"])
            pred_anat = str(deep_res["body_part"])

            # Physical intensity calibration sanity
            if is_hu and pred_mod != "CT":
                # Absolute physical HU units confirm CT modality
                pred_mod = "CT"
            elif (not is_hu) and (vmin >= 0.0) and (pred_mod == "CT"):
                # Strictly positive non-Hounsfield domain indicates MRI
                pred_mod = "MRI_T1"

            return ImageDiagnosis(
                modality=pred_mod,
                body_part=pred_anat,
                intensity_domain="HOUNSFIELD" if pred_mod == "CT" else _non_hu_domain(vmin, vmax),
                hu_min=vmin,
                hu_max=vmax,
                hu_mean=vmean,
                spatial_extent_mm=spatial_extent,
                dimension=dim,
                confidence=float(deep_res["confidence"]),
                source=str(deep_res["source"]),
                details={
                    "anatomy_probabilities": deep_res["anatomy_probabilities"],
                    "modality_probabilities": deep_res["modality_probabilities"],
                    "deep_anatomy_confidence": deep_res["anatomy_confidence"],
                    "deep_modality_confidence": deep_res["modality_confidence"],
                    "header_hint": header_hint,
                    "tier2": "used",
                }
            )

    # 1. Physical Hounsfield Unit Analysis (CT Detection)
    if is_hu:
        intensity_domain = "HOUNSFIELD"
        modality = "CT"

        # Count voxel fractions in standardized HU windows
        air_vox = float(np.sum((arr_flat >= -1050.0) & (arr_flat <= -400.0))) / total_vox
        bone_vox = float(np.sum(arr_flat >= 300.0)) / total_vox
        soft_tissue_vox = float(np.sum((arr_flat >= 20.0) & (arr_flat <= 80.0))) / total_vox
        fat_vox = float(np.sum((arr_flat >= -120.0) & (arr_flat <= -40.0))) / total_vox

        # Anatomy diagnosis based on internal physical tissue composition (NOT header shape)
        slices = tuple(slice(s // 4, 3 * s // 4) for s in arr.shape)
        center_arr = arr[slices].ravel()
        center_air = float(np.sum((center_arr >= -1050.0) & (center_arr <= -400.0))) / max(len(center_arr), 1)
        center_soft = float(np.sum((center_arr >= 15.0) & (center_arr <= 85.0))) / max(len(center_arr), 1)

        if center_air >= 0.20:
            body_part = "THORAX"
            confidence = 0.90
        elif center_soft >= 0.35:
            body_part = "ABDOMEN"
            confidence = 0.90
        elif (bone_vox >= 0.04) and (soft_tissue_vox >= 0.15) and (center_air < 0.10):
            body_part = "BRAIN"
            confidence = 0.85
        elif (soft_tissue_vox >= 0.15) and (fat_vox >= 0.02):
            body_part = "ABDOMEN"
            confidence = 0.85
        else:
            body_part = "ABDOMEN" if dim == 3 else "UNKNOWN"
            confidence = 0.70

        return ImageDiagnosis(
            modality=modality,
            body_part=body_part,
            intensity_domain=intensity_domain,
            hu_min=vmin,
            hu_max=vmax,
            hu_mean=vmean,
            air_ratio=air_vox,
            bone_ratio=bone_vox,
            soft_tissue_ratio=soft_tissue_vox,
            spatial_extent_mm=spatial_extent,
            dimension=dim,
            confidence=confidence,
            source="statistical_hu_tier1",
            details={"fat_ratio": fat_vox, "header_hint": header_hint, "tier2": tier2_status}
        )

    # 2. Non-Negative Intensity Domain (MRI or Normalized Domain)
    intensity_domain = _non_hu_domain(vmin, vmax)

    modality = "MRI_T1"

    # Non-zero foreground analysis
    fg_mask = arr > 0.02 * vmax if vmax > 0 else np.zeros_like(arr, dtype=bool)
    fg_vox = arr[fg_mask] if np.any(fg_mask) else arr_flat

    if len(fg_vox) > 0:
        p25 = float(np.percentile(fg_vox, 25.0))
        p50 = float(np.percentile(fg_vox, 50.0))
        p75 = float(np.percentile(fg_vox, 75.0))
        p98 = float(np.percentile(fg_vox, 98.0))
    else:
        p25, p50, p75, p98 = 0.0, 0.0, 0.0, 1.0

    details = {"p25": p25, "p50": p50, "p75": p75, "p98": p98, "header_hint": header_hint,
               "tier2": tier2_status}

    # MRI Contrast Profile Diagnosis (T1 vs T2 from voxel histogram tail)
    if len(fg_vox) > 100:
        ratio_tail = (p98 - p50) / max(p50 - p25, 1e-5)
        if ratio_tail > 3.0:
            modality = "MRI_T2"
        else:
            modality = "MRI_T1"

    # Multi-contrast check or standard 2D/3D MRI
    if dim == 4 and arr.shape[-1] >= 4:
        body_part = "BRAIN"
        confidence = 0.85
    elif dim == 4 and arr.shape[-1] == 2:
        body_part = "PELVIS"
        confidence = 0.80
    elif dim == 2:
        body_part = "BRAIN"
        confidence = 0.85
    else:
        body_part = "BRAIN" if dim == 3 and max(spatial_extent) < 260.0 else "UNKNOWN"
        confidence = 0.70

    return ImageDiagnosis(
        modality=modality,
        body_part=body_part,
        intensity_domain=intensity_domain,
        hu_min=vmin,
        hu_max=vmax,
        hu_mean=vmean,
        spatial_extent_mm=spatial_extent,
        dimension=dim,
        confidence=confidence,
        source="statistical_tier1_fallback",
        details=details
    )


def diagnose_pair(
    fixed: Union[ants.ANTsImage, torch.Tensor, np.ndarray],
    moving: Union[ants.ANTsImage, torch.Tensor, np.ndarray],
    fast: bool = False
) -> PairDiagnosis:
    """
    Diagnose both images with ``diagnose_image`` and classify their relationship.

    Parameters
    ----------
    fixed, moving : ANTsImage, torch.Tensor or numpy.ndarray
        The two images.
    fast : bool, default False
        Passed to ``diagnose_image``; True means Tier-1 heuristics only.

    Returns
    -------
    PairDiagnosis
        ``relationship`` is "MONO_MODAL_INTRA" when the modality labels match,
        "MULTI_CONTRAST" when both are MRI labels, else "CROSS_MODAL"; ``confidence`` is the
        smaller of the two image confidences; ``recommended_policy`` is left None.
    """
    diag_fix = diagnose_image(fixed, fast=fast)
    diag_mov = diagnose_image(moving, fast=fast)

    is_same_anatomy = (diag_fix.body_part == diag_mov.body_part) and (diag_fix.body_part != "UNKNOWN")
    is_same_modality = (diag_fix.modality == diag_mov.modality) and (diag_fix.modality != "UNKNOWN")

    if is_same_modality:
        relationship = "MONO_MODAL_INTRA"
    elif diag_fix.is_mri() and diag_mov.is_mri():
        relationship = "MULTI_CONTRAST"
    else:
        relationship = "CROSS_MODAL"

    pair_conf = min(diag_fix.confidence, diag_mov.confidence)

    return PairDiagnosis(
        fixed=diag_fix,
        moving=diag_mov,
        relationship=relationship,
        is_same_anatomy=is_same_anatomy,
        is_same_modality=is_same_modality,
        confidence=pair_conf
    )
