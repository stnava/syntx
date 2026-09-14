"""
Unit tests for syntx.classifier (Deep 3D Diagnostic Classifier).
"""

import pytest
import numpy as np
import torch
import ants

from syntx.classifier import (
    DiagnosticClassifier3D,
    preprocess_volume_for_classifier,
    predict_diagnosis_deep,
    ANATOMY_CLASSES,
    MODALITY_CLASSES,
)


def test_diagnostic_classifier_forward_pass():
    model = DiagnosticClassifier3D()
    model.eval()

    # Create dummy 3D batch: (B=2, C=1, D=16, H=16, W=16)
    x = torch.randn(2, 1, 16, 16, 16)

    with torch.no_grad():
        logits_anat, logits_mod = model(x)

    assert logits_anat.shape == (2, len(ANATOMY_CLASSES))
    assert logits_mod.shape == (2, len(MODALITY_CLASSES))


def test_preprocess_volume_for_classifier():
    arr = np.random.uniform(0, 1000, size=(30, 40, 50)).astype(np.float32)
    img = ants.from_numpy(arr)

    tensor = preprocess_volume_for_classifier(img, target_shape=(32, 32, 32))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 32, 32, 32)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_predict_diagnosis_deep():
    model = DiagnosticClassifier3D()
    arr = np.random.uniform(0, 255, size=(24, 24, 24)).astype(np.float32)
    img = ants.from_numpy(arr)

    res = predict_diagnosis_deep(model, img, device="cpu")

    assert "body_part" in res
    assert res["body_part"] in ANATOMY_CLASSES
    assert "modality" in res
    assert res["modality"] in MODALITY_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert "anatomy_probabilities" in res
    assert len(res["anatomy_probabilities"]) == len(ANATOMY_CLASSES)
