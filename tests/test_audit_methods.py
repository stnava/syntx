"""Regression tests for the docstring-audit behaviour issues in the registration-method
modules (syn, syngs, tvf, greedy, template, motion, deformation_metrics, image_utils).
CPU only, tiny inputs, spies instead of full registrations where possible."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _no_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


class _Stop(Exception):
    pass


# ---------------------------------------------------------------------------------------
# motion.py
# ---------------------------------------------------------------------------------------

def test_motion_pytorch_backend_calls_robust_affine_without_backend_kw(monkeypatch):
    import ants
    from syntx.motion import motion_correction
    ra = sys.modules["syntx.robust_affine"]            # syntx.robust_affine is the function
    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise _Stop
    monkeypatch.setattr(ra, "robust_affine", spy)
    arr = np.random.default_rng(0).random((12, 12, 12, 3)).astype(np.float32)
    with pytest.raises(_Stop):
        motion_correction(ants.from_numpy(arr), backend="pytorch", type_of_transform="Rigid")
    assert "backend" not in seen and seen["mode"] == "auto"


def test_motion_rejects_ignored_options():
    import ants
    from syntx.motion import motion_correction
    img = ants.from_numpy(np.random.default_rng(0).random((12, 12, 12, 3)).astype(np.float32))
    with pytest.raises(ValueError, match="aff_metric"):
        motion_correction(img, backend="pytorch", aff_metric="mattes")
    with pytest.raises(TypeError, match="num_bins"):
        motion_correction(img, backend="pytorch_batched", sampling_percentage=0.3)
