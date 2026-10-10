"""robust_affine on very anisotropic (thin-slab) fixed images, plus the no-silent-fallback contract.

Evidence and measurements: /Users/stnava/data/nm_registration_debug/RESULTS.md.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import ants
import torch

import syntx
from syntx import RobustAffineSolverError

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
RA = importlib.import_module("syntx.robust_affine")


def _phantom():
    import siq
    from scipy.ndimage import gaussian_filter
    np.random.seed(1234)
    arr = gaussian_filter(siq.simulate_brain_procedural((96, 104, 88)).numpy().astype(np.float32), 1.0)
    return ants.from_numpy(arr, origin=(-48.0, -52.0, -44.0), spacing=(1.0, 1.0, 1.0))


def _slab_pair(r, inplane=0.75, fov=54.0):
    """Cross-modal thin slab: fixed = intensity-inverted phantom sampled on a (inplane, inplane,
    inplane*r) slab through a KNOWN rigid transform (box-averaged through each thick slice); moving =
    the 1 mm phantom. Returns (fixed, moving, true fixed->moving transform)."""
    mov = _phantom()
    a = mov.numpy(); a = (a - a.min()) / (a.max() - a.min())
    cmap = mov.new_image_like((1.0 - a ** 0.8).astype(np.float32))
    a1, a2 = np.deg2rad(6.0), np.deg2rad(-4.0)
    Rz = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([(Rz @ Rx).ravel(), [4.0, -3.0, 5.0]])); tx.set_fixed_parameters(np.zeros(3))
    n = int(round(fov / inplane)); nz = int(round(fov / (inplane * r)))
    grid = ants.make_image((n, n, nz * r), spacing=(inplane,) * 3, origin=(-fov / 2,) * 3)
    fine = ants.apply_ants_transform_to_image(tx, cmap, grid, interpolation="linear")
    fa = fine.numpy().reshape(n, n, nz, r).mean(axis=3)
    fixed = ants.from_numpy(fa.astype(np.float32), origin=(-fov / 2, -fov / 2, -fov / 2 + (r - 1) * inplane / 2),
                            spacing=(inplane, inplane, inplane * r))
    return fixed, mov, tx


def _tre(reg, tx):
    rec = ants.read_transform(reg["fwdtransforms"][0])
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-20, 20, 100), rng.uniform(-20, 20, 100), rng.uniform(-15, 15, 100)])
    return float(np.mean([np.linalg.norm(np.array(rec.apply_to_point(tuple(p))) - np.array(tx.apply_to_point(tuple(p))))
                          for p in pts]))


_OPEN = "open: capture range along a thick slice axis is ~half a slice; see the commit message / report"


@pytest.mark.parametrize("r", [
    1, 2, 8,
    pytest.param(4, marks=pytest.mark.xfail(reason=_OPEN, strict=False)),
    pytest.param(6, marks=pytest.mark.xfail(reason=_OPEN, strict=False)),
    pytest.param(10, marks=pytest.mark.xfail(reason=_OPEN, strict=False)),
])
def test_thin_slab_cross_modal_known_rigid(r):
    fixed, mov, tx = _slab_pair(r)
    reg = syntx.robust_affine(fixed, mov, mode="auto", dof="rigid", device=DEV, seed=42)
    assert reg["solver"] == "pytorch"
    assert _tre(reg, tx) < 1.5


@pytest.mark.parametrize("shape", [(33, 35, 37), (25, 27, 29)])
def test_odd_voxel_counts_with_cached_fixed_weights(shape):
    """Regression for 'fixed_weights has N+1 rows for N samples' (mask_mode='none', sampling < 1, odd voxel counts)."""
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(rng.random(shape).astype(np.float32), 2.0); a = (a - a.min()) / (a.max() - a.min())
    img = ants.from_numpy(a)
    reg = syntx.robust_affine(img, img, mode="pytorch", dof="rigid", device="cpu", preset="accurate", seed=1)
    assert reg["status"] == "SUCCESS"


@pytest.mark.parametrize("mode", ["auto", "pytorch"])
def test_solver_exception_propagates_no_fallback(monkeypatch, mode):
    img = ants.from_numpy(np.random.default_rng(0).random((24, 24, 24)).astype(np.float32))

    def boom(*a, **k):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(RA, "_run_pytorch_affine_solver", boom)
    with pytest.raises(RobustAffineSolverError) as ei:
        syntx.robust_affine(img, img, mode=mode, device="cpu")
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_solver_key_present():
    a = ants.from_numpy(np.random.default_rng(0).random((20, 20, 20)).astype(np.float32))
    assert syntx.robust_affine(a, a, mode="com_only")["solver"] == "com_only"
    assert syntx.robust_affine(a, a, mode="auto", device="cpu")["solver"] == "pytorch"


def test_self_check_raises_when_pose_is_worse_than_doing_nothing(monkeypatch):
    """Sabotage the optimiser's result: a final pose worse than both trivial baselines must raise."""
    a = _phantom()
    orig_matrix_t = RA._AffinePath.clamp_

    def clamp_bad(self):
        orig_matrix_t(self)
        with torch.no_grad():
            self.t.add_(60.0)

    monkeypatch.setattr(RA._AffinePath, "clamp_", clamp_bad)
    with pytest.raises(RobustAffineSolverError, match="not better than doing nothing"):
        syntx.robust_affine(a, a, mode="auto", dof="rigid", device="cpu", preset="fast")


_D = "/Users/stnava/data/nm_registration_debug"


@pytest.mark.skipif(not all(os.path.exists(f"{_D}/{f}") for f in
                            ("fixed_nm_n4.nii.gz", "moving_t1_slab_crop.nii.gz", "ants_rigid_reference.mat")),
                    reason="local NM debug data absent")
@pytest.mark.xfail(strict=False, reason="measured 4.4 mm vs the ANTs reference (was 12.6 mm before the millimetre pyramid): "
                   "the MI landscape on this pair is near-degenerate along z; see report")
def test_real_nm_t1_pose_within_2mm_of_ants():
    fixed = ants.image_read(f"{_D}/fixed_nm_n4.nii.gz")
    moving = ants.image_read(f"{_D}/moving_t1_slab_crop.nii.gz")
    c = np.array(ants.get_center_of_mass(fixed * 0 + 1))
    probes = [c + np.array([x, y, z]) for x in (-15, 15) for y in (-15, 15) for z in (-5, 5)] + [c]

    def disp(p):
        tx = ants.read_transform(p)
        return np.array([np.array(tx.apply_to_point(tuple(q))) - q for q in probes])

    reg = syntx.robust_affine(fixed, moving, mode="auto", dof="rigid", seed=42, device=DEV)
    err = np.linalg.norm(disp(reg["fwdtransforms"][0]) - disp(f"{_D}/ants_rigid_reference.mat"), axis=1).max()
    assert err < 2.0, f"pose error {err:.1f} mm"
