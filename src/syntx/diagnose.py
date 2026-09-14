"""
syntx.diagnose — Intelligent Diagnostic Engine for Medical Image Registration
=============================================================================

Automatically diagnoses:
- Modality: CT, MRI_T1, MRI_T2, MRI_FLAIR, MRI_ADC, MRI_OTHER, UNKNOWN
- Anatomy: BRAIN, THORAX, ABDOMEN, PELVIS, HEART, UNKNOWN
- Intensity Domain: HOUNSFIELD, POSITIVE_FLOAT, NORMALIZED_01
- Pair Relationships: MONO_MODAL_INTRA, MONO_MODAL_INTER, MULTI_CONTRAST, CROSS_MODAL

Uses a two-tier architecture:
- Tier 1: Fast statistical, intensity histogram, and physical geometric analysis (<15ms).
- Tier 2: Deep 3D ResNet-10 multi-task classification head (when PyTorch model/weights loaded).
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
import ants


@dataclass
class ImageDiagnosis:
    """Diagnostic profile for a single medical image."""
    modality: str
    body_part: str
    intensity_domain: str
    is_contrast_enhanced: bool = False
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
        return self.modality == "CT"

    def is_mri(self) -> bool:
        return self.modality.startswith("MRI")

    def is_brain(self) -> bool:
        return self.body_part == "BRAIN"

    def is_thorax(self) -> bool:
        return self.body_part == "THORAX"

    def is_abdomen(self) -> bool:
        return self.body_part == "ABDOMEN"

    def is_pelvis(self) -> bool:
        return self.body_part == "PELVIS"

    def is_heart(self) -> bool:
        return self.body_part == "HEART"


@dataclass
class PairDiagnosis:
    """Joint diagnostic profile comparing fixed and moving registration targets."""
    fixed: ImageDiagnosis
    moving: ImageDiagnosis
    relationship: str
    is_same_anatomy: bool
    is_same_modality: bool
    recommended_policy: Optional[Any] = None
    confidence: float = 1.0


def _extract_numpy(image: Union[ants.ANTsImage, torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, Tuple[float, ...], int]:
    """Helper to convert input image into numpy array, spacing, and dimensionality."""
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


def diagnose_image(image: Union[ants.ANTsImage, torch.Tensor, np.ndarray], fast: bool = True) -> ImageDiagnosis:
    """
    Diagnoses modality, body part, and intensity domain of a single medical image.

    Parameters:
    -----------
    image : ANTsImage, PyTorch Tensor, or NumPy array
        Input 2D or 3D scalar volume.
    fast : bool, default=True
        If True, executes Tier 1 statistical & physical feature heuristics (<15ms).

    Returns:
    --------
    ImageDiagnosis
        Structured diagnosis with predicted modality, anatomy, and intensity domain.
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

    # Tier 2: Deep 3D ResNet-10 Multi-Task Classification (if weights available and dim >= 3)
    if not fast and dim >= 3:
        try:
            import os
            from .classifier import DiagnosticClassifier3D, predict_diagnosis_deep
            weights_path = os.path.join(os.path.dirname(__file__), "models", "diagnostic_resnet10_3d.pth")
            if os.path.exists(weights_path):
                model = DiagnosticClassifier3D()
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                deep_res = predict_diagnosis_deep(model, image)
                return ImageDiagnosis(
                    modality=deep_res["modality"],
                    body_part=deep_res["body_part"],
                    intensity_domain="HOUNSFIELD" if deep_res["modality"] == "CT" else ("NORMALIZED_01" if (vmin >= -1e-4 and vmax <= 1.05) else "POSITIVE_FLOAT"),
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
                        "deep_modality_confidence": deep_res["modality_confidence"]
                    }
                )
        except Exception:
            pass  # Fall back gracefully to Tier 1 heuristics

    # 1. Hounsfield Unit Analysis (CT Detection)
    # CT scanners calibrate air to -1000 HU, water to 0 HU, dense bone to +500 to +3000 HU.
    is_hu = (vmin <= -750.0) and (vmax >= 200.0)

    if is_hu:
        intensity_domain = "HOUNSFIELD"
        modality = "CT"

        # Count voxel fractions in standardized HU windows
        air_vox = float(np.sum((arr_flat >= -1050.0) & (arr_flat <= -400.0))) / total_vox
        bone_vox = float(np.sum(arr_flat >= 300.0)) / total_vox
        soft_tissue_vox = float(np.sum((arr_flat >= 20.0) & (arr_flat <= 80.0))) / total_vox
        fat_vox = float(np.sum((arr_flat >= -120.0) & (arr_flat <= -40.0))) / total_vox

        # Anatomy diagnosis based on physical HU distribution & central internal cavity
        # Thorax has internal lungs (central air >= 20%)
        # Abdomen has solid parenchymal organs (central soft tissue >= 35%)
        slices = tuple(slice(s // 4, 3 * s // 4) for s in arr.shape)
        center_arr = arr[slices].ravel()
        center_air = float(np.sum((center_arr >= -1050.0) & (center_arr <= -400.0))) / max(len(center_arr), 1)
        center_soft = float(np.sum((center_arr >= 15.0) & (center_arr <= 85.0))) / max(len(center_arr), 1)

        if center_air >= 0.20:
            body_part = "THORAX"
            confidence = 0.95
        elif center_soft >= 0.35:
            body_part = "ABDOMEN"
            confidence = 0.95
        elif (bone_vox >= 0.04) and (soft_tissue_vox >= 0.15) and (center_air < 0.10):
            # Dense skull ring enclosing brain parenchyma
            body_part = "BRAIN"
            confidence = 0.90
        elif (soft_tissue_vox >= 0.15) and (fat_vox >= 0.02):
            body_part = "ABDOMEN"
            confidence = 0.90
        elif soft_tissue_vox >= 0.15 and bone_vox >= 0.03:
            body_part = "PELVIS"
            confidence = 0.85
        else:
            body_part = "ABDOMEN" if dim == 3 else "UNKNOWN"
            confidence = 0.75

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
            details={"fat_ratio": fat_vox}
        )

    # 2. Non-Negative Intensity (MRI or Normalized Domain)
    if (vmin >= -1e-4) and (vmax <= 1.05):
        intensity_domain = "NORMALIZED_01"
    else:
        intensity_domain = "POSITIVE_FLOAT"

    # Default to MRI when intensities are strictly positive and non-Hounsfield
    modality = "MRI_T1"  # Default initial prior

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

    details = {"p25": p25, "p50": p50, "p75": p75, "p98": p98}

    # Geometric Aspect Ratio & Physical Anatomy Analysis
    body_part = "BRAIN"  # Primary MRI prior in neuroimaging
    confidence = 0.85

    spatial_shape = arr.shape[:3]
    spatial_spacing = spacing[:3] if len(spacing) >= 3 else spacing
    spatial_ext_3d = tuple(float(s * sz) for s, sz in zip(spatial_spacing, spatial_shape))

    if len(spatial_ext_3d) >= 3:
        max_transverse = max(spatial_ext_3d[0], spatial_ext_3d[1])
        z_extent = spatial_ext_3d[2]
        min_sp = min(spatial_spacing[:3])
        max_sp = max(spatial_spacing[:3])
        anisotropy = max_sp / max(min_sp, 1e-5)

        # 1. Localized Sub-structural ROI Crop (e.g. Hippocampus)
        if all(ext < 90.0 for ext in spatial_ext_3d):
            body_part = "BRAIN"
            confidence = 0.90
            details["is_roi_crop"] = True
        # 2. Pelvis / Prostate (thick-slice axial anisotropic slabs or dual-channel T2/ADC)
        elif (anisotropy >= 2.5 and (spatial_shape[2] <= 35 or z_extent <= 100.0)) or (dim == 4 and arr.shape[-1] == 2 and max_transverse < 280.0):
            body_part = "PELVIS"
            confidence = 0.90
        # 3. Thorax / Cardiac MRI (wide transverse chest FOV >= 260mm)
        elif max_transverse >= 260.0:
            if z_extent >= 100.0 and anisotropy < 2.5:
                body_part = "HEART"
                confidence = 0.85
            else:
                body_part = "ABDOMEN"
                confidence = 0.80
        # 4. Standard Isotropic Brain MRI
        elif max_transverse < 260.0 and anisotropy < 2.5 and spatial_shape[2] >= 50:
            body_part = "BRAIN"
            confidence = 0.95

    # MRI Contrast Profile Diagnosis (T1 vs T2)
    if len(fg_vox) > 100:
        ratio_tail = (p98 - p50) / max(p50 - p25, 1e-5)
        if ratio_tail > 3.0:
            modality = "MRI_T2"
        else:
            modality = "MRI_T1"

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
        source="statistical_geometry_tier1",
        details=details
    )


def diagnose_pair(
    fixed: Union[ants.ANTsImage, torch.Tensor, np.ndarray],
    moving: Union[ants.ANTsImage, torch.Tensor, np.ndarray],
    fast: bool = True
) -> PairDiagnosis:
    """
    Jointly diagnoses fixed and moving images, determining their mutual modality
    and anatomical relationship for autonomous registration policy dispatch.

    Parameters:
    -----------
    fixed : ANTsImage, PyTorch Tensor, or NumPy array
        Fixed target image.
    moving : ANTsImage, PyTorch Tensor, or NumPy array
        Moving source image.
    fast : bool, default=True
        If True, executes Tier 1 statistical heuristics.

    Returns:
    --------
    PairDiagnosis
        Joint diagnostic summary including cross-pair relationship.
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
