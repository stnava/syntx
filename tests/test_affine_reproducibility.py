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


def test_cpu_device_is_honoured(pair):
    ds, fi_p, mi_p = pair
    from syntx.robust_affine import _run_pytorch_affine_solver
    import syntx.robust_affine as ra
    seen = {}
    orig = torch.device
    # cheap check: solver must construct torch.device('cpu') when asked for cpu
    def spy(*a, **k):
        d = orig(*a, **k); seen[str(d)] = True; return d
    ra.torch.device = spy
    try:
        _run(fi_p, mi_p, "cpu")
    finally:
        ra.torch.device = orig
    assert "cpu" in seen and not any(k.startswith(("mps", "cuda")) for k in seen)


@pytest.mark.parametrize("device", ["cpu"] + ([pytest.param("mps", marks=pytest.mark.xfail(
    strict=True, reason="MPS matmul over ~1e6-sample Parzen weights is nondeterministic (fixed by the chunked histogram in the next commit)"))]
    if torch.backends.mps.is_available() else []))
def test_same_device_bitwise_reproducible(pair, device):
    ds, fi_p, mi_p = pair
    p1, _ = _run(fi_p, mi_p, device)
    p2, _ = _run(fi_p, mi_p, device)
    assert np.array_equal(p1, p2), f"{device}: max |Δ| = {np.abs(p1 - p2).max():.3e}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS for cross-device check")
@pytest.mark.xfail(strict=True, reason="MPS Parzen-histogram matmul bug degrades the GPU solve (Dice 0.22 vs 0.30 on CPU); fixed next commit")
def test_cpu_vs_gpu_agree(pair):
    ds, fi_p, mi_p = pair
    pc, rc = _run(fi_p, mi_p, "cpu")
    pg, rg = _run(fi_p, mi_p, "mps")
    assert np.abs(pc[:9] - pg[:9]).max() < 2e-3 and np.abs(pc[9:] - pg[9:]).max() < 0.5, "cpu/gpu parameter mismatch"
    assert abs(_dice(ds, rc["fwdtransforms"]) - _dice(ds, rg["fwdtransforms"])) < 0.003


@pytest.mark.skipif(not os.path.exists(ANTS_JSON), reason="run scripts/affine_repro_harness.py to create the ANTs baseline")
@pytest.mark.xfail(strict=True, reason="PyTorch affine currently 0.30 (cpu) / 0.22 (mps) vs best ANTs 0.326 on mbhard; target of the affine commit series")
def test_dice_within_ants_baseline(pair):
    ds, fi_p, mi_p = pair
    ab = json.load(open(ANTS_JSON))
    best = max(np.mean([r["dice_sym"] for r in v]) for v in ab.values())
    _, r = _run(fi_p, mi_p, "mps" if torch.backends.mps.is_available() else "cpu")
    assert _dice(ds, r["fwdtransforms"]) >= best - 0.005
