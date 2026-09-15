"""
Computational reproducibility of syntx.robust_affine (mode='pytorch') on real data.

Same device, same seed, same inputs must give bitwise-identical transform parameters;
CPU and GPU must agree to a tolerance; Dice must stay within 0.005 of the best ANTs C++
affine recorded by scripts/affine_repro_harness.py.  Skipped without Mindboggle data.
"""
from __future__ import annotations

import json, os
import numpy as np
import pytest
import ants
import torch

import syntx

ANTS_JSON = "results/affine_baseline/mbhard_ants_affine_baseline.json"
CACHE = "results/affine_baseline/cache"


def _pair():
    try:
        from syntx.benchmark.data import resolve_data_dir
        base = resolve_data_dir()
    except Exception:
        return None
    if not os.path.exists(os.path.join(base, "NKI-TRT-20_volumes", "NKI-TRT-20-2", "t1weighted_brain.nii.gz")):
        return None
    ds = syntx.benchmark_data("mbhard")
    if os.path.exists(os.path.join(CACHE, "fi_p.nii.gz")):
        fi_p, mi_p = ants.image_read(os.path.join(CACHE, "fi_p.nii.gz")), ants.image_read(os.path.join(CACHE, "mi_p.nii.gz"))
    else:
        from syntx.landmarks import preprocess_for_landmarks
        fi_p, mi_p = preprocess_for_landmarks(ds["fixed"]), preprocess_for_landmarks(ds["moving"])
    return ds, fi_p, mi_p


@pytest.fixture(scope="module")
def pair():
    p = _pair()
    if p is None:
        pytest.skip("real Mindboggle mbhard volumes not available")
    return p


def _run(fi_p, mi_p, device):
    from syntx.robust_affine import robust_affine
    r = robust_affine(fi_p, mi_p, mode="pytorch", device=device, seed=42)
    return np.array(ants.read_transform(r["fwdtransforms"][0]).parameters, float), r


def _dice(ds, fwd):
    from syntx.deformation_metrics import compute_bidirectional_dice
    return compute_bidirectional_dice(ds["fixed_label"], ds["moving_label"], ds["fixed"], ds["moving"], fwd, fwd, [True])[2]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a GPU to tell the devices apart")
def test_cpu_device_is_honoured(pair):
    """device='cpu' used to be silently upgraded to the GPU.  Each device is bitwise reproducible
    but CPU and MPS differ in the last bits, so equal CPU runs that differ from MPS prove the flag works."""
    ds, fi_p, mi_p = pair
    pc1, _ = _run(fi_p, mi_p, "cpu")
    pc2, _ = _run(fi_p, mi_p, "cpu")
    pg, _ = _run(fi_p, mi_p, "mps")
    assert np.array_equal(pc1, pc2)
    assert not np.array_equal(pc1, pg), "cpu result identical to mps: device flag ignored?"


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_same_device_bitwise_reproducible(pair, device):
    ds, fi_p, mi_p = pair
    p1, _ = _run(fi_p, mi_p, device)
    p2, _ = _run(fi_p, mi_p, device)
    assert np.array_equal(p1, p2), f"{device}: max |Δ| = {np.abs(p1 - p2).max():.3e}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS for cross-device check")
def test_cpu_vs_gpu_agree(pair):
    ds, fi_p, mi_p = pair
    pc, rc = _run(fi_p, mi_p, "cpu")
    pg, rg = _run(fi_p, mi_p, "mps")
    # each device is bitwise reproducible, but float-level differences accumulate over ~130 Adam
    # steps and the multi-start selection, so devices may settle in neighbouring optima:
    # require agreement in outcome (Dice) and a loosely similar transform, not identical parameters
    assert np.abs(pc[:9] - pg[:9]).max() < 0.05 and np.abs(pc[9:] - pg[9:]).max() < 3.0, "cpu/gpu transform mismatch"
    assert abs(_dice(ds, rc["fwdtransforms"]) - _dice(ds, rg["fwdtransforms"])) < 0.006


@pytest.mark.skipif(not os.path.exists(ANTS_JSON), reason="run scripts/affine_repro_harness.py to create the ANTs baseline")
def test_dice_within_ants_baseline(pair):
    ds, fi_p, mi_p = pair
    ab = json.load(open(ANTS_JSON))
    best = max(np.mean([r["dice_sym"] for r in v]) for v in ab.values())
    _, r = _run(fi_p, mi_p, "mps" if torch.backends.mps.is_available() else "cpu")
    assert _dice(ds, r["fwdtransforms"]) >= best - 0.005


def test_auto_mode_uses_the_pytorch_solver_and_falls_back(monkeypatch):
    """Since 2026-09-15 'auto' runs the PyTorch solver (validated faster/better/reproducible on a
    10-pair cohort) and only falls back to the ANTs C++ path if that solver raises."""
    import ants as _ants
    import importlib
    ra = importlib.import_module("syntx.robust_affine")   # the package shadows this name with the function
    fixed = _ants.image_read(_ants.get_ants_data("r16"))
    moving = _ants.image_read(_ants.get_ants_data("r64"))

    calls = {"pt": 0, "ants": 0}
    real_pt = ra._run_pytorch_affine_solver

    def spy_pt(*a, **k):
        calls["pt"] += 1
        return real_pt(*a, **k)

    monkeypatch.setattr(ra, "_run_pytorch_affine_solver", spy_pt)
    r = ra.robust_affine(fixed, moving, mode="auto", seed=42)
    assert calls["pt"] == 1 and r["fwdtransforms"]

    # fail-safe: when the solver raises, 'auto' still returns a transform via the ANTs path
    def boom(*a, **k):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(ra, "_run_pytorch_affine_solver", boom)
    r2 = ra.robust_affine(fixed, moving, mode="auto", seed=42, multi_start=False)
    assert r2["fwdtransforms"], "auto mode must fall back to ANTs C++ when the solver fails"

    # explicit 'pytorch' must NOT silently fall back
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        ra.robust_affine(fixed, moving, mode="pytorch", seed=42)
