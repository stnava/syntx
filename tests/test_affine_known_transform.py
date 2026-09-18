"""
robust_affine(mode='pytorch') must recover a known affine on synthetic data (2-D and 3-D),
in physical space, with sub-voxel target registration error, and must honour an initial transform.
"""
from __future__ import annotations

import numpy as np
import pytest
import ants
import torch

from syntx.robust_affine import robust_affine

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def _phantom3d():
    import siq
    from scipy.ndimage import gaussian_filter
    np.random.seed(1234)
    arr = gaussian_filter(siq.simulate_brain_procedural((96, 104, 88)).numpy().astype(np.float32), 1.0)
    return ants.from_numpy(arr, origin=(-40.0, -55.0, -30.0), spacing=(1.0, 1.0, 1.0), direction=np.diag([1.0, -1.0, 1.0]))


def _apply_known(img, A, t):
    """moving(x) = fixed(A (x - c) + c + t): a point p in moving corresponds to A(p-c)+c+t in fixed."""
    ctr = np.array(ants.get_center_of_mass(img), dtype=np.float64)
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=img.dimension)
    tx.set_parameters(np.concatenate([A.ravel(), t])); tx.set_fixed_parameters(ctr)
    return tx.apply_to_image(img, img, interpolation="linear"), ctr


def _tre(fwd_mat, fixed, A, t, ctr):
    """Recovered fixed->moving map vs truth: moving point q of fixed point p is A^-1 (p - c - t) + c."""
    tx = ants.read_transform(fwd_mat)
    P = np.array(tx.parameters, float); d = fixed.dimension
    M, tt, c = P[:d * d].reshape(d, d), P[d * d:d * d + d], np.array(tx.fixed_parameters, float)[:d]
    rng = np.random.default_rng(0)
    idx = np.column_stack([rng.uniform(0.25 * n, 0.75 * n, 200) for n in fixed.shape])
    from syntx.landmarks.spatial import vox_to_physical
    p = vox_to_physical(fixed, idx if d == 3 else np.column_stack([idx, np.zeros(len(idx))])).astype(np.float64)[:, :d]
    pred = (M @ (p - c).T).T + c + tt
    truth = (np.linalg.inv(A) @ (p - ctr - t).T).T + ctr
    return float(np.linalg.norm(pred - truth, axis=1).mean())


def test_recover_known_affine_3d():
    fixed = _phantom3d()
    a1, a2 = np.deg2rad(7.0), np.deg2rad(-4.0)
    Rz = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    A = Rz @ Rx @ np.diag([1.06, 0.97, 1.02]); t = np.array([3.0, -2.5, 1.5])
    moving, ctr = _apply_known(fixed, A, t)
    for preset in ("default", "fast"):
        r = robust_affine(fixed, moving, mode="pytorch", device=DEV, seed=42, preset=preset)
        tre = _tre(r["fwdtransforms"][0], fixed, A, t, ctr)
        assert tre < 0.75, f"preset {preset}: mean TRE {tre:.2f} mm"


def test_initial_transform_is_used_as_candidate_3d():
    """A 60-degree yaw is outside the +-24 degree cone search; a correct initial transform must rescue it."""
    fixed = _phantom3d()
    a = np.deg2rad(60.0)
    A = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]); t = np.array([2.0, 1.0, -1.0])
    moving, ctr = _apply_known(fixed, A, t)
    r0 = robust_affine(fixed, moving, mode="pytorch", device=DEV, seed=42, preset="fast")
    tre0 = _tre(r0["fwdtransforms"][0], fixed, A, t, ctr)
    # true fixed->moving map as an ITK .mat: q = A^-1 (p - c - t) + c
    import tempfile, os
    Ai = np.linalg.inv(A); tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([Ai.ravel(), Ai @ (-t)])); tx.set_fixed_parameters(ctr)
    path = os.path.join(tempfile.mkdtemp(), "init.mat"); ants.write_transform(tx, path)
    r1 = robust_affine(fixed, moving, mode="pytorch", device=DEV, seed=42, preset="fast", initial_transform=path)
    tre1 = _tre(r1["fwdtransforms"][0], fixed, A, t, ctr)
    assert r1["init_candidate"] == "Provided_Initial_Transform"
    assert tre1 < 0.75 and tre1 < tre0, f"with init {tre1:.2f} mm, without {tre0:.2f} mm"


def test_recover_known_affine_2d():
    fixed = ants.image_read(ants.get_ants_data("r16"))
    a = np.deg2rad(9.0)
    A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([1.05, 0.96]); t = np.array([4.0, -3.0])
    moving, ctr = _apply_known(fixed, A, t)
    r = robust_affine(fixed, moving, mode="pytorch", device="cpu", seed=42)
    tre = _tre(r["fwdtransforms"][0], fixed, A, t, ctr)
    assert tre < 1.0, f"2-D mean TRE {tre:.2f} mm"
