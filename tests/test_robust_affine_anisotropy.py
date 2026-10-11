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


_OPEN_6TO1 = ("known limitation, deterministic: ratio 6, pose A ends at ~4.4 mm. The coarse objective at the TRUE pose is as good as for "
              "the other ratios (-0.19) but no start candidate lies within the ~1 coarse-voxel translation capture of the metric, so the "
              "optimiser ends in a worse minimum (docs/PYRAMID.md Section 13). It passes with the opt-in translation_scan_steps=2 "
              "(test below), which is NOT the default because it regresses real neuromelanin pairs (sub-07 1.2 -> 7.8 mm).")


@pytest.mark.parametrize("r", [
    1, 2, 4, 8, 10,
    pytest.param(6, marks=pytest.mark.xfail(reason=_OPEN_6TO1, strict=True)),
])
def test_thin_slab_cross_modal_known_rigid(r):
    """Default path (mm pyramid for both images + one-slice valid margin): known-rigid TRE < 1.5 mm at every thin-slab ratio but 6 (pose A)."""
    fixed, mov, tx = _slab_pair(r)
    reg = syntx.robust_affine(fixed, mov, mode="auto", dof="rigid", device=DEV, seed=42)
    assert reg["solver"] == "pytorch"
    assert _tre(reg, tx) < 1.5


def test_thin_slab_6to1_pose_a_is_solved_by_the_opt_in_translation_scan():
    fixed, mov, tx = _slab_pair(6)
    reg = syntx.robust_affine(fixed, mov, mode="auto", dof="rigid", device=DEV, seed=42, translation_scan_steps=2)
    assert _tre(reg, tx) < 1.5


@pytest.mark.parametrize("name,value", [("valid_margin_slices", -1), ("valid_margin_slices", 1.5), ("valid_margin_slices", True),
                                        ("translation_scan_steps", -2), ("translation_scan_steps", "2")])
def test_margin_and_scan_parameters_are_validated(name, value):
    img = ants.from_numpy(np.random.default_rng(0).random((20, 20, 20)).astype(np.float32))
    with pytest.raises(ValueError, match=name):
        syntx.robust_affine(img, img, mode="auto", device="cpu", **{name: value})


def test_valid_margin_is_a_noop_for_an_isotropic_fixed_image_and_changes_an_anisotropic_one():
    iso = _phantom()
    mov = ants.apply_transforms(fixed=iso, moving=iso, transformlist=[], interpolator="linear")
    a = syntx.robust_affine(iso, mov, mode="pytorch", dof="rigid", device="cpu", preset="fast", seed=3, valid_margin_slices=0)
    b = syntx.robust_affine(iso, mov, mode="pytorch", dof="rigid", device="cpu", preset="fast", seed=3, valid_margin_slices=1)
    np.testing.assert_array_equal(ants.read_transform(a["fwdtransforms"][0]).parameters, ants.read_transform(b["fwdtransforms"][0]).parameters)
    fixed, mov_t1, _ = _slab_pair(4)
    r0 = syntx.robust_affine(fixed, mov_t1, mode="pytorch", dof="rigid", device="cpu", preset="fast", seed=3, valid_margin_slices=0)
    r1 = syntx.robust_affine(fixed, mov_t1, mode="pytorch", dof="rigid", device="cpu", preset="fast", seed=3, valid_margin_slices=1)
    assert r0["self_check"]["final_loss_coarse"] != r1["self_check"]["final_loss_coarse"]      # the margin changes the sample set


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


_D = "/Users/stnava/data/nm_registration_debug/subjects"
_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


@pytest.mark.parametrize("sub", _SUBJECTS)
def test_real_nm_t1_pose_within_2mm_of_ants(sub):
    """Ten real ds001242 neuromelanin-slab / T1 pairs (local data): the DEFAULT path lands within 2 mm of the ANTs rigid reference."""
    d = f"{_D}/{sub}"
    if not all(os.path.exists(f"{d}/{f}") for f in ("fixed_nm_n4.nii.gz", "moving_t1_slab_crop.nii.gz", "ants_rigid_reference.mat")):
        pytest.skip("local NM debug data absent")
    fixed = ants.image_read(f"{d}/fixed_nm_n4.nii.gz")
    moving = ants.image_read(f"{d}/moving_t1_slab_crop.nii.gz")
    c = np.array(ants.get_center_of_mass(fixed * 0 + 1))
    probes = [c + np.array([x, y, z]) for x in (-15, 15) for y in (-15, 15) for z in (-5, 5)] + [c]

    def disp(p):
        tx = ants.read_transform(p)
        return np.array([np.array(tx.apply_to_point(tuple(q))) - q for q in probes])

    reg = syntx.robust_affine(fixed, moving, mode="auto", dof="rigid", seed=42, device=DEV)
    err = np.linalg.norm(disp(reg["fwdtransforms"][0]) - disp(f"{d}/ants_rigid_reference.mat"), axis=1).max()
    assert err < 2.0, f"{sub}: pose error {err:.2f} mm"
