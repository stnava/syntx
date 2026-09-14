"""
Landmark detector / descriptor / matcher tests on siq simulated data.

Checks that keypoints and descriptors are functions of the *physical* image,
not of how the array happens to be stored (flips, permutations, anisotropic
spacing), and that known rigid and nonrigid transforms are recovered.
"""
from __future__ import annotations

import numpy as np
import pytest
import ants

from syntx.landmarks import (
    detect_blobs_dog, detect_blobs_log, detect_sift3d,
    extract_mind_at_points, match_landmarks, ransac_filter,
)
from syntx.landmarks import spatial as S
from tests.test_landmarks_spatial import reframe  # noqa: F401  (fixture helper)


@pytest.fixture(scope="module")
def phantom():
    import siq
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    np.random.seed(1234)                      # siq draws from the global numpy RNG
    ph = siq.simulate_brain_procedural((96, 104, 88))
    arr = gaussian_filter(ph.numpy().astype(np.float32), 1.0)
    arr = arr + rng.normal(0, 0.02, arr.shape).astype(np.float32) * (arr > 0)
    return ants.from_numpy(arr, origin=(-40.0, -55.0, -30.0), spacing=(1.0, 1.0, 1.0), direction=np.eye(3))


@pytest.fixture(scope="module")
def frames(phantom):
    return {
        "identity": phantom,
        "flip_xy": reframe(phantom, (0, 1, 2), (True, True, False)),
        "perm_zxy_flipz": reframe(phantom, (2, 0, 1), (True, False, False)),
        "aniso": ants.resample_image(phantom, (1.2, 1.0, 0.9), use_voxels=False, interp_type=0),
    }


@pytest.fixture(scope="module")
def sift_ref(phantom):
    return detect_sift3d(phantom, preprocess=False, max_keypoints=200)


def _nn(P, Q):
    return np.linalg.norm(P[:, None, :3] - Q[None, :, :3], axis=2).min(axis=1)


def test_blob_keypoints_are_frame_invariant(frames):
    ref = detect_blobs_dog(frames["identity"], preprocess=False, max_keypoints=200)
    assert ref.shape[1] == 4 and len(ref) > 50
    for name in ("flip_xy", "perm_zxy_flipz"):
        kp = detect_blobs_dog(frames[name], preprocess=False, max_keypoints=200)
        assert len(kp) == len(ref), name
        assert np.mean(_nn(kp, ref) < 1e-3) > 0.98, name
    kp = detect_blobs_log(frames["perm_zxy_flipz"], preprocess=False, max_keypoints=200)
    ref = detect_blobs_log(frames["identity"], preprocess=False, max_keypoints=200)
    assert np.mean(_nn(kp, ref) < 1e-3) > 0.9


def test_blob_keypoints_survive_anisotropic_resampling(frames):
    ref = detect_blobs_dog(frames["identity"], preprocess=False, max_keypoints=200)
    kp = detect_blobs_dog(frames["aniso"], preprocess=False, max_keypoints=200)
    d = _nn(kp, ref)
    assert np.median(d) < 1.5 and np.mean(d < 2.0) > 0.6


def test_blob_keypoints_inside_foreground(frames):
    im = frames["identity"]
    kp = detect_blobs_dog(im, preprocess=False, max_keypoints=300)
    idx = np.rint(S.physical_to_vox(im, kp[:, :3])).astype(int)
    assert (idx >= 0).all() and (idx < np.array(im.shape)).all()
    vals = im.numpy()[tuple(idx.T)]
    assert (vals > 0.0).all()


def test_sift3d_descriptors_frame_invariant(frames, sift_ref):
    cA, dA = sift_ref
    assert len(cA) > 20 and dA.shape == (len(cA), 512)
    assert np.allclose(np.linalg.norm(dA, axis=1), 1.0, atol=1e-3)
    for name in ("flip_xy", "perm_zxy_flipz"):
        cB, dB = detect_sift3d(frames[name], preprocess=False, max_keypoints=200)
        assert len(cB) == len(cA), name
        m = match_landmarks(cA, cB, dA, dB, ratio_thresh=0.8)
        assert len(m) >= 0.95 * len(cA), name
        res = np.linalg.norm(cA[m[:, 0], :3] - cB[m[:, 1], :3], axis=1)
        assert np.max(res) < 1e-3, name


def test_sift3d_descriptors_robust_to_anisotropic_resampling(frames, sift_ref):
    from syntx.landmarks.sift3d import _build_descriptor, _physical_gradient
    from syntx.landmarks.blob import _to_tensor, _normalize_intensity
    cA, dA = sift_ref
    im = frames["aniso"]
    vol = _normalize_intensity(_to_tensor(im, "cpu"))
    g = _physical_gradient(vol, np.array(im.spacing), np.array(im.direction))
    dB = _build_descriptor(g, S.get_image_affine(im), cA[:, :3], cA[:, 3])
    cos = np.sum(dA * dB, axis=1)
    assert np.median(cos) > 0.95 and cos.min() > 0.85
    cB, dB2 = detect_sift3d(im, preprocess=False, max_keypoints=200)
    m = match_landmarks(cA, cB, dA, dB2, ratio_thresh=0.8)
    mf, M = ransac_filter(cA, cB, m, inlier_thresh_mm=3.0)
    assert len(mf) >= 0.5 * len(cA)
    assert np.abs(M[:3, 3]).max() < 2.0


def test_mind_frame_invariant(frames, sift_ref):
    pts = sift_ref[0][:, :3]
    mA = extract_mind_at_points(frames["identity"], pts, device="cpu")
    assert mA.shape == (len(pts), 12)
    for name in ("flip_xy", "perm_zxy_flipz"):
        mB = extract_mind_at_points(frames[name], pts, device="cpu")
        assert np.abs(mA - mB).max() < 1e-3, name     # float32 trilinear sampling noise


def test_mind_channel_semantics_via_direct_indexing(frames):
    """Sampling the dense MIND volume at voxel centres equals direct indexing."""
    from syntx.landmarks.mind import compute_mind
    im = frames["perm_zxy_flipz"]
    vol = compute_mind(im, device="cpu")
    idx = np.array([[10, 20, 30], [40, 50, 20], [5, 60, 33]], dtype=float)
    pts = S.vox_to_physical(im, idx)
    got = extract_mind_at_points(im, pts, device="cpu", mind_vol=vol)
    direct = vol[0][:, 10, 20, 30].numpy(), vol[0][:, 40, 50, 20].numpy(), vol[0][:, 5, 60, 33].numpy()
    assert np.allclose(got, np.stack(direct), atol=1e-5)


def test_known_rigid_transform_recovered(frames, sift_ref):
    A = frames["identity"]
    cA, dA = sift_ref
    ctr = np.array(ants.get_center_of_mass(A))
    a1, a2 = np.deg2rad(8.0), np.deg2rad(5.0)
    R = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    t = np.array([4.0, -3.0, 2.5])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), t]))
    tx.set_fixed_parameters(ctr)
    B = reframe(tx.apply_to_image(A, A), (0, 1, 2), (True, True, False))   # B(x) = A(T(x)), stored flipped
    cB, dB = detect_sift3d(B, preprocess=False, max_keypoints=250)
    m = match_landmarks(cA, cB, dA, dB, ratio_thresh=0.8)
    mf, M = ransac_filter(cA, cB, m, model="rigid", inlier_thresh_mm=3.0)
    assert len(mf) >= 10
    # point q in A corresponds to T^-1(q) in B
    truth = (np.linalg.inv(R) @ (cA[:, :3] - ctr - t).T).T + ctr
    pred = (M[:3, :3] @ cA[:, :3].T).T + M[:3, 3]
    tre = np.linalg.norm(pred - truth, axis=1)
    assert tre.mean() < 1.5


def test_nonrigid_correspondences(frames, sift_ref):
    A = frames["identity"]
    cA, dA = sift_ref
    np.random.seed(3)
    disp = ants.simulate_displacement_field(A, field_type="bspline", number_of_random_points=200, sd_noise=6.0,
                                            enforce_stationary_boundary=True, number_of_fitting_levels=3, mesh_size=2)
    dtx = ants.transform_from_displacement_field(disp)
    C = dtx.apply_to_image(A, A)                      # C(p) = A(p + u(p))
    cC, dC = detect_sift3d(C, preprocess=False, max_keypoints=250)
    m = match_landmarks(cA, cC, dA, dC, ratio_thresh=0.8)
    mf, _ = ransac_filter(cA, cC, m, model="affine", inlier_thresh_mm=6.0)
    assert len(mf) >= 8
    truth = np.array([dtx.apply_to_point(list(map(float, p))) for p in cC[mf[:, 1], :3]])
    err = np.linalg.norm(truth - cA[mf[:, 0], :3], axis=1)
    assert np.median(err) < 3.0


def test_rotation_invariant_descriptor_survives_60_degrees(frames):
    A = frames["identity"]
    ctr = np.array(ants.get_center_of_mass(A))
    axis = np.array([0.2, 0.3, 1.0]); axis /= np.linalg.norm(axis); a = np.deg2rad(60.0)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    t = np.array([3.0, -2.0, 1.0])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), t])); tx.set_fixed_parameters(ctr)
    B = tx.apply_to_image(A, A)
    kw = dict(preprocess=False, max_keypoints=400, n_scales=8, sigma_min=1.0)
    cA, dA = detect_sift3d(A, rotation_invariant=True, **kw)
    cB, dB = detect_sift3d(B, rotation_invariant=True, **kw)
    m = match_landmarks(cA, cB, dA, dB, ratio_thresh=0.9, mutual=True)
    mf, M = ransac_filter(cA, cB, m, model="rigid", inlier_thresh_mm=3.0)
    assert len(mf) >= 20
    pred = (M[:3, :3] @ cA[:, :3].T).T + M[:3, 3]
    true_pts = (np.linalg.inv(R) @ (cA[:, :3] - ctr - t).T).T + ctr
    assert np.linalg.norm(pred - true_pts, axis=1).mean() < 1.0
    # the axis-aligned descriptor is expected to fail here; make sure the flag actually changes the output
    _, dA_aligned = detect_sift3d(A, rotation_invariant=False, **kw)
    assert not np.allclose(dA, dA_aligned)


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="MPS not available")
def test_border_ops_match_cpu_on_mps_large_slices():
    """Regression: torch F.pad on MPS silently corrupts 5-D tensors whose trailing H*W >= 65536
    (torch 2.13); the Laplacian and gradient must therefore use slicing and agree with CPU."""
    import torch
    from syntx.landmarks.blob import _laplacian3d
    from syntx.landmarks.sift3d import _gradient_axes
    torch.manual_seed(0)
    v = torch.randn(1, 1, 8, 256, 256)
    assert float((_laplacian3d(v) - _laplacian3d(v.to("mps")).cpu()).abs().max()) < 1e-5
    assert float((_gradient_axes(v) - _gradient_axes(v.to("mps")).cpu()).abs().max()) < 1e-5


def test_rotation_search_recovers_60_degrees(frames):
    from syntx.landmarks import match_sift3d_with_rotation_search
    from syntx.landmarks.orient import rotation_geodesic_deg
    A = frames["identity"]
    ctr = np.array(ants.get_center_of_mass(A))
    axis = np.array([0.2, 0.3, 1.0]); axis /= np.linalg.norm(axis); a = np.deg2rad(60.0)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    t = np.array([3.0, -2.0, 1.0])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), t])); tx.set_fixed_parameters(ctr)
    B = tx.apply_to_image(A, A)                      # B(x) = A(T x): fixed->moving directions = R^-1
    res = match_sift3d_with_rotation_search(A, B, ransac_model="rigid", inlier_thresh_mm=3.0, ratio_thresh=0.9,
                                            max_keypoints=400, n_scales=8, sigma_min=1.0)
    assert res["used_rotation"]
    assert rotation_geodesic_deg(res["rotation"][None], np.linalg.inv(R)[None])[0, 0] < 5.0
    assert len(res["inliers"]) >= 50
    M = res["affine"]; cA = res["coords_fixed"]
    pred = (M[:3, :3] @ cA[:, :3].T).T + M[:3, 3]
    truth = (np.linalg.inv(R) @ (cA[:, :3] - ctr - t).T).T + ctr
    assert np.linalg.norm(pred - truth, axis=1).mean() < 1.0
