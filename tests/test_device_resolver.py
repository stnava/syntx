"""Unit tests for the central device resolver ``syntx.device`` and ``ComputeContext`` resolution."""
from __future__ import annotations

import os

import pytest
import torch

from syntx import device as D
from syntx.contract import ComputeContext


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ANTSXMM_DEVICE", "ANTSXMM_REG_DEVICE"):
        monkeypatch.delenv(name, raising=False)


def _fake(monkeypatch, cuda=False, mps=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if cuda else 0)
    monkeypatch.setattr(D, "_mps_available", lambda: mps)


def test_auto_precedence_cuda_over_mps_over_cpu(monkeypatch):
    _fake(monkeypatch, cuda=True, mps=True)
    assert D.resolve_device_with_source() == ("cuda", "auto:cuda")
    _fake(monkeypatch, cuda=False, mps=True)
    assert D.resolve_device_with_source() == ("mps", "auto:mps")
    _fake(monkeypatch, cuda=False, mps=False)
    assert D.resolve_device_with_source() == ("cpu", "auto:cpu")


def test_explicit_wins_over_env(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    monkeypatch.setenv("ANTSXMM_DEVICE", "mps")
    assert D.resolve_device_with_source("cpu") == ("cpu", "explicit")


def test_explicit_unavailable_raises_no_silent_downgrade(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=False)
    with pytest.raises(D.DeviceUnavailableError):
        D.resolve_device("mps")
    with pytest.raises(D.DeviceUnavailableError):
        D.resolve_device("cuda")
    with pytest.raises(D.DeviceUnavailableError):
        D.resolve_device("cuda:1")
    with pytest.raises(ValueError):
        D.resolve_device("tpu")


def test_env_unavailable_raises_and_auto_env_falls_through(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    monkeypatch.setenv("ANTSXMM_DEVICE", "cuda")
    with pytest.raises(D.DeviceUnavailableError):
        D.resolve_device()
    monkeypatch.setenv("ANTSXMM_DEVICE", "auto")
    assert D.resolve_device() == "mps"
    monkeypatch.setenv("ANTSXMM_DEVICE", "  ")
    assert D.resolve_device() == "mps"


def test_env_names_by_task(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    monkeypatch.setenv("ANTSXMM_REG_DEVICE", "cpu")
    assert D.resolve_device(None, task="syn") == "cpu"
    assert D.resolve_device_with_source(None, task="reg") == ("cpu", "env:ANTSXMM_REG_DEVICE")
    assert D.resolve_device(None, task="model") == "mps"  # REG env does not apply to model tasks
    assert D.resolve_device(None) == "mps"
    monkeypatch.setenv("ANTSXMM_DEVICE", "cpu")
    assert D.resolve_device_with_source(None, task="model") == ("cpu", "env:ANTSXMM_DEVICE")
    monkeypatch.setenv("ANTSXMM_REG_DEVICE", "mps")
    assert D.resolve_device(None, task="affine") == "mps"  # REG env wins over DEVICE for registration


def test_torch_device_object_accepted(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=False)
    assert D.resolve_device(torch.device("cpu")) == "cpu"


def test_mps_choice_sets_fallback_env(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    D.resolve_device()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")  # a deliberate user value is respected
    D.resolve_device()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_cpu_choice_does_not_set_fallback_env(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    D.resolve_device()
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_compute_context_defaults_resolve_never_cpu_by_default(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    c = ComputeContext()
    assert (c.reg_device, c.model_device) == ("mps", "mps")
    assert c.reg_device_source == "auto:mps" and c.model_device_source == "auto:mps"
    assert "auto" not in (c.reg_device, c.model_device)


def test_compute_context_explicit_and_source_preserved(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    c = ComputeContext(reg_device="cpu", model_device="mps", reg_device_source="policy")
    assert (c.reg_device, c.model_device, c.reg_device_source) == ("cpu", "mps", "policy")
    assert c.model_device_source == "explicit"


def test_compute_context_explicit_unavailable_raises(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=False)
    with pytest.raises(D.DeviceUnavailableError):
        ComputeContext(reg_device="mps")


def test_compute_context_reg_env(monkeypatch):
    _fake(monkeypatch, cuda=False, mps=True)
    monkeypatch.setenv("ANTSXMM_REG_DEVICE", "cpu")
    c = ComputeContext()
    assert (c.reg_device, c.reg_device_source, c.model_device) == ("cpu", "env:ANTSXMM_REG_DEVICE", "mps")


def test_motion_correction_forwards_device(monkeypatch):
    """motion_correction resolves the device, forwards it to the batched pass and records it."""
    import ants
    import numpy as np

    import syntx
    import syntx.motion_batched as MB

    _fake(monkeypatch, cuda=False, mps=False)
    seen = {}

    real = MB.batched_rigid_register_pass

    def spy(*a, **kw):
        seen["device"] = kw.get("device")
        return real(*a, **kw)

    monkeypatch.setattr(MB, "batched_rigid_register_pass", spy)
    rng = np.random.default_rng(0)
    vol = ants.from_numpy(rng.random((16, 16, 12, 3)).astype("float32"))
    res = syntx.motion_correction(vol, backend="pytorch_batched", device="cpu")
    assert seen["device"] == "cpu"
    assert res["summary"]["device"] == "cpu"
    with pytest.raises(D.DeviceUnavailableError):
        syntx.motion_correction(vol, backend="pytorch_batched", device="mps")
