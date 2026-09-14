"""
syntx.classifier — Deep Multi-Task 3D Diagnostic Classifier for Medical Images
=============================================================================

Leverages a 3D ResNet-10 convolutional backbone to jointly classify:
- Anatomical Region: BRAIN, THORAX, ABDOMEN, PELVIS, HEART
- Imaging Modality: CT, MRI_T1, MRI_T2, MRI_FLAIR, MRI_ADC

Provides sub-10ms inference on standard 3D volumes.
"""

import os
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ants

from .resnet import resnet10_3d


ANATOMY_CLASSES = ["BRAIN", "THORAX", "ABDOMEN", "PELVIS", "HEART"]
MODALITY_CLASSES = ["CT", "MRI_T1", "MRI_T2", "MRI_FLAIR", "MRI_ADC"]

ANATOMY_TO_IDX = {name: i for i, name in enumerate(ANATOMY_CLASSES)}
MODALITY_TO_IDX = {name: i for i, name in enumerate(MODALITY_CLASSES)}


class DiagnosticClassifier3D(nn.Module):
    """
    Lightweight 3D ResNet-10 Multi-Task Classification Network.
    """

    def __init__(self, num_anatomy: int = len(ANATOMY_CLASSES), num_modality: int = len(MODALITY_CLASSES)):
        super().__init__()
        self.backbone = resnet10_3d()
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        self.anatomy_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(128, num_anatomy)
        )
        self.modality_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(128, num_modality)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor of shape (B, 1, D, H, W)
        
        Returns
        -------
        logits_anatomy : torch.Tensor of shape (B, num_anatomy)
        logits_modality : torch.Tensor of shape (B, num_modality)
        """
        features = self.backbone(x)  # (B, 512, D', H', W')
        pooled = self.pool(features).flatten(1)  # (B, 512)
        logits_anatomy = self.anatomy_head(pooled)
        logits_modality = self.modality_head(pooled)
        return logits_anatomy, logits_modality


def preprocess_volume_for_classifier(
    image: Union[ants.ANTsImage, torch.Tensor, np.ndarray],
    target_shape: Tuple[int, int, int] = (64, 64, 64)
) -> torch.Tensor:
    """
    Resamples, standardizes dynamic range, and batches an input 3D volume
    into a PyTorch tensor (1, 1, D, H, W) for classifier inference.
    """
    if isinstance(image, ants.ANTsImage):
        if image.dimension == 4:
            arr = image.numpy()[..., 0].astype(np.float32)
        else:
            arr = image.numpy().astype(np.float32)
    elif isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Foreground intensity normalization
    v_max = float(np.max(arr))
    v_min = float(np.min(arr))
    if v_max > v_min:
        fg = arr[arr > (v_min + 0.05 * (v_max - v_min))]
        if len(fg) > 0:
            p02, p98 = np.percentile(fg, (2.0, 98.0))
            if p98 > p02 + 1e-4:
                arr = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0)
            else:
                arr = np.clip((arr - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)
        else:
            arr = np.clip((arr - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)

    # Convert to PyTorch (1, 1, D, H, W)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    
    # Fast trilinear resampling to uniform target shape
    if t.shape[2:] != target_shape:
        t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)

    return t


def predict_diagnosis_deep(
    model: DiagnosticClassifier3D,
    image: Union[ants.ANTsImage, torch.Tensor, np.ndarray],
    device: Optional[str] = None
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Runs inference using deep ResNet-10 diagnostic model.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    tensor = preprocess_volume_for_classifier(image).to(device)

    with torch.no_grad():
        logits_anat, logits_mod = model(tensor)
        prob_anat = F.softmax(logits_anat, dim=1)[0].cpu().numpy()
        prob_mod = F.softmax(logits_mod, dim=1)[0].cpu().numpy()

    pred_anat_idx = int(np.argmax(prob_anat))
    pred_mod_idx = int(np.argmax(prob_mod))

    pred_anat = ANATOMY_CLASSES[pred_anat_idx]
    pred_mod = MODALITY_CLASSES[pred_mod_idx]

    conf_anat = float(prob_anat[pred_anat_idx])
    conf_mod = float(prob_mod[pred_mod_idx])
    overall_conf = float((conf_anat + conf_mod) / 2.0)

    anat_probs = {cls: float(prob_anat[i]) for i, cls in enumerate(ANATOMY_CLASSES)}
    mod_probs = {cls: float(prob_mod[i]) for i, cls in enumerate(MODALITY_CLASSES)}

    return {
        "body_part": pred_anat,
        "modality": pred_mod,
        "confidence": overall_conf,
        "anatomy_confidence": conf_anat,
        "modality_confidence": conf_mod,
        "anatomy_probabilities": anat_probs,
        "modality_probabilities": mod_probs,
        "source": "deep_resnet10_tier2"
    }
