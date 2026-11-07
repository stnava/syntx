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


@pytest.fixture(scope="module")
def mbhard_repro_runs(pair):
    """Run CPU twice and MPS twice once per module instead of 10 redundant 256^3 solves."""
    _, fi_p, mi_p = pair
    out = {
        "cpu": [_run(fi_p, mi_p, "cpu"), _run(fi_p, mi_p, "cpu")],
    }
    if torch.backends.mps.is_available():
        out["mps"] = [_run(fi_p, mi_p, "mps"), _run(fi_p, mi_p, "mps")]
    return out


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a GPU to tell the devices apart")
def test_cpu_device_is_honoured(mbhard_repro_runs):
    """device='cpu' used to be silently upgraded to the GPU.  Each device is bitwise reproducible
    but CPU and MPS differ in the last bits, so equal CPU runs that differ from MPS prove the flag works."""
    (pc1, _), (pc2, _) = mbhard_repro_runs["cpu"]
    (pg, _), _ = mbhard_repro_runs["mps"]
    assert np.array_equal(pc1, pc2)
    assert not np.array_equal(pc1, pg), "cpu result identical to mps: device flag ignored?"


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_same_device_bitwise_reproducible(mbhard_repro_runs, device):
    (p1, _), (p2, _) = mbhard_repro_runs[device]
    assert np.array_equal(p1, p2), f"{device}: max |Δ| = {np.abs(p1 - p2).max():.3e}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS for cross-device check")
def test_cpu_vs_gpu_agree(pair, mbhard_repro_runs):
    ds, _, _ = pair
    (pc, rc), _ = mbhard_repro_runs["cpu"]
    (pg, rg), _ = mbhard_repro_runs["mps"]
    # each device is bitwise reproducible, but float-level differences accumulate over ~130 Adam
    # steps and the multi-start selection, so devices may settle in neighbouring optima:
    # require agreement in outcome (Dice) and a loosely similar transform, not identical parameters
    assert np.abs(pc[:9] - pg[:9]).max() < 0.05 and np.abs(pc[9:] - pg[9:]).max() < 3.0, "cpu/gpu transform mismatch"
    assert abs(_dice(ds, rc["fwdtransforms"]) - _dice(ds, rg["fwdtransforms"])) < 0.006


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(ANTS_JSON), reason="run scripts/affine_repro_harness.py to create the ANTs baseline")
def test_dice_within_ants_baseline(pair, mbhard_repro_runs):
    ds, _, _ = pair
    ab = json.load(open(ANTS_JSON))
    best = max(np.mean([r["dice_sym"] for r in v]) for v in ab.values())
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    _, r = mbhard_repro_runs[dev][0]
    assert _dice(ds, r["fwdtransforms"]) >= best - 0.005


@pytest.mark.slow
def test_synthetic_3d_reproducibility_and_device_honoured_fast():
    """Fast default-suite guard for bitwise same-device reproducibility and CPU vs MPS device honoring."""
    from tests.conftest import get_cached_procedural_brain_array
    from syntx.robust_affine import robust_affine

    arr = get_cached_procedural_brain_array(shape=(28, 30, 26), seed=2026)
    fixed = ants.from_numpy(arr, origin=(-28.0, -30.0, -26.0), spacing=(2.0, 2.0, 2.0))
    com = np.asarray(fixed.get_center_of_mass(), dtype=float).tolist()
    tx_gt = ants.create_ants_transform(
        transform_type="Euler3DTransform",
        dimension=3,
        center=com,
        translation=(2.2, -1.6, 1.2),
    )
    tx_gt.set_parameters([0.06, -0.05, 0.04, 2.2, -1.6, 1.2])
    moving = tx_gt.apply_to_image(fixed, reference=fixed, interpolation="linear")

    def _solve_fast(dev):
        r = robust_affine(fixed, moving, mode="pytorch", preset="fast", multi_start=False, device=dev, seed=42)
        return np.array(ants.read_transform(r["fwdtransforms"][0]).parameters, float)

    pc1 = _solve_fast("cpu")
    pc2 = _solve_fast("cpu")
    assert np.array_equal(pc1, pc2), f"cpu: max |Δ| = {np.abs(pc1 - pc2).max():.3e}"

    if torch.backends.mps.is_available():
        pg1 = _solve_fast("mps")
        pg2 = _solve_fast("mps")
        assert np.array_equal(pg1, pg2), f"mps: max |Δ| = {np.abs(pg1 - pg2).max():.3e}"
        assert not np.array_equal(pc1, pg1), "cpu result identical to mps: device flag ignored?"
        assert np.abs(pc1[:9] - pg1[:9]).max() < 0.05 and np.abs(pc1[9:] - pg1[9:]).max() < 3.0


def test_auto_mode_uses_the_pytorch_solver_and_never_falls_back(monkeypatch):
    """'auto' runs the PyTorch solver; if it raises, RobustAffineSolverError propagates (the
    owner's rule: no silent fallbacks). The ANTs path is reachable only by mode='ants'."""
    import ants as _ants
    import importlib
    import pytest as _pytest
    from syntx import RobustAffineSolverError
    ra = importlib.import_module("syntx.robust_affine")   # the package shadows this name with the function
    fixed = _ants.image_read(_ants.get_ants_data("r16"))
    moving = _ants.image_read(_ants.get_ants_data("r64"))

    calls = {"pt": 0}
    real_pt = ra._run_pytorch_affine_solver

    def spy_pt(*a, **k):
        calls["pt"] += 1
        return real_pt(*a, **k)

    monkeypatch.setattr(ra, "_run_pytorch_affine_solver", spy_pt)
    r = ra.robust_affine(fixed, moving, mode="auto", seed=42)
    assert calls["pt"] == 1 and r["fwdtransforms"] and r["solver"] == "pytorch"

    def boom(*a, **k):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(ra, "_run_pytorch_affine_solver", boom)
    for mode in ("auto", "pytorch"):
        with _pytest.raises(RobustAffineSolverError) as ei:
            ra.robust_affine(fixed, moving, mode=mode, seed=42, multi_start=False)
        assert isinstance(ei.value.__cause__, RuntimeError) and "synthetic" in str(ei.value)
