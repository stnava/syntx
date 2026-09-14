"""
Real-data smoke tests for syntx.landmarks on the Mindboggle `mbhard` pair.

The fixed (NKI-TRT-20-2) and moving (MMRR-21-2) T1 volumes are stored in
different native frames (LAS vs RPS arrays, 1.0 vs 1.2 mm x-spacing), which is
exactly the situation the physical-space framework must handle.  Skipped when
the real Mindboggle volumes are not available (the loader would otherwise fall
back to synthetic spheres, which is not what these tests are about).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import ants

import syntx
from syntx.landmarks import spatial as S
from syntx.landmarks import detect_sift3d, detect_blobs_dog, match_landmarks, ransac_filter


def _real_mbhard():
    try:
        from syntx.benchmark.data import resolve_data_dir
        base = resolve_data_dir()
    except Exception:
        return None
    p = os.path.join(base, "NKI-TRT-20_volumes", "NKI-TRT-20-2", "t1weighted_brain.nii.gz")
    if not os.path.exists(p):
        return None
    return syntx.benchmark_data("mbhard")


@pytest.fixture(scope="module")
def ds():
    d = _real_mbhard()
    if d is None:
        pytest.skip("real Mindboggle mbhard volumes not available")
    return d


def test_mbhard_frames_and_parity(ds):
    fi, mi = ds["fixed"], ds["moving"]
    assert S.axis_orientation_code(fi) != S.axis_orientation_code(mi)
    for im in (fi, mi):
        idx = np.array([[0, 0, 0], [12, 24, 36], [60, 110, 75]], dtype=float)
        phys = S.vox_to_physical(im, idx)
        for v, p in zip(idx, phys):
            ref = ants.transform_index_to_physical_point(im, [int(x) for x in v])
            assert np.allclose(ref, p, atol=1e-4)
        assert np.allclose(S.physical_to_vox(im, phys), idx, atol=1e-4)


def test_mbhard_display_consistency(ds):
    """Both images render with the same anatomical labels and their centres of mass project to the slice centre."""
    for im in (ds["fixed"], ds["moving"]):
        sl = S.extract_ortho_slices(im)
        assert sl["labels_ax"] == dict(left="R", right="L", bottom="P", top="A")
        assert sl["labels_cor"] == dict(left="R", right="L", bottom="I", top="S")
        assert sl["labels_sag"] == dict(left="P", right="A", bottom="I", top="S")
        for view in ("ax", "cor", "sag"):
            spec = sl["views"][view]
            u, v, m = S.project_to_slice(im, sl["center_mm"][None], view=spec)
            assert m[0] and 0 <= u[0] < spec["n_u"] and 0 <= v[0] < spec["n_v"]
            assert sl[view].shape == (spec["n_v"], spec["n_u"])


def test_mbhard_keypoints_inside_brain(ds):
    for im in (ds["fixed"], ds["moving"]):
        kp = detect_blobs_dog(im, max_keypoints=64, preprocess=False)
        assert kp.shape[1] == 4 and len(kp) > 0
        idx = np.rint(S.physical_to_vox(im, kp[:, :3])).astype(int)
        assert (idx >= 0).all() and (idx < np.array(im.shape)).all()
        assert (im.numpy()[tuple(idx.T)] > 0).all()


def test_mbhard_sift3d_matches_are_anatomically_plausible(ds):
    fi, mi = ds["fixed"], ds["moving"]
    cf, df = detect_sift3d(fi, max_keypoints=200, preprocess=False)
    cm, dm = detect_sift3d(mi, max_keypoints=200, preprocess=False)
    assert len(cf) > 50 and len(cm) > 50
    m = match_landmarks(cf, cm, df, dm, ratio_thresh=0.85)
    mf, M = ransac_filter(cf, cm, m, model="affine", inlier_thresh_mm=8.0, min_inliers=6)
    assert len(mf) >= 6
    # The two subjects' heads sit ~ (100, 136, 112) mm apart in scanner space; RANSAC must
    # recover a translation of that order with a near-orthonormal linear part.
    com_f, com_m = np.array(ants.get_center_of_mass(fi)), np.array(ants.get_center_of_mass(mi))
    pred_com = M[:3, :3] @ com_f + M[:3, 3]
    assert np.linalg.norm(pred_com - com_m) < 15.0
    sv = np.linalg.svd(M[:3, :3], compute_uv=False)
    assert sv.min() > 0.7 and sv.max() < 1.4
