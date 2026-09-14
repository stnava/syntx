"""
Unit and integration tests for syntx.landmarks on Medical Segmentation Decathlon (MSD) data.

Tests cover:
1. Modality detection (CT vs MRI) across Decathlon tasks.
2. 4D multi-channel volume handling (Task01 BrainTumour, Task05 Prostate).
3. Foreground landmark validity on both CT and MRI (100% inside anatomy).
4. Physical frame-invariance on real clinical Decathlon volumes.
5. Known rigid transformation recovery and TRE (< 1.0 mm).
6. Large rotation recovery via match_sift3d_with_rotation_search (< 0.5 deg).
7. Inter-subject pairwise landmark matching and segmentation alignment.
8. Orthographic slice extraction and projection to medical viewing standard.
"""

from __future__ import annotations

import os
import glob
import json
import numpy as np
import pytest
import torch
import ants

from syntx.landmarks import (
    preprocess_for_landmarks,
    is_ct_image,
    detect_sift3d,
    detect_blobs_dog,
    detect_blobs_log,
    compute_mind,
    extract_mind_at_points,
    match_landmarks,
    ransac_filter,
    match_sift3d_with_rotation_search,
)
from syntx.landmarks import spatial as S

DECATHLON_DIR = "/Users/stnava/data/decathlon"

def _task_path(task_name: str) -> str:
    return os.path.join(DECATHLON_DIR, task_name)

def _has_task(task_name: str) -> bool:
    tp = _task_path(task_name)
    return os.path.isdir(tp) and os.path.isfile(os.path.join(tp, "dataset.json"))


# ---------------------------------------------------------------------------
# Test 1: Modality Classification across Decathlon Tasks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(DECATHLON_DIR), reason="Decathlon data not found")
def test_decathlon_modality_detection():
    """Verify that CT vs MRI is correctly classified across all tasks."""
    expected = {
        "Task01_BrainTumour": False,   # MRI
        "Task02_Heart": False,         # MRI
        "Task03_Liver": True,          # CT
        "Task04_Hippocampus": False,   # MRI
        "Task05_Prostate": False,      # MRI
        "Task06_Lung": True,           # CT
        "Task07_Pancreas": True,       # CT
        "Task08_HepaticVessel": True,  # CT
        "Task09_Spleen": True,         # CT
        "Task10_Colon": True,          # CT
    }
    for task_name, expected_is_ct in expected.items():
        if not _has_task(task_name):
            continue
        with open(os.path.join(_task_path(task_name), "dataset.json")) as f:
            meta = json.load(f)
        first = meta["training"][0]
        img_rel = first["image"] if isinstance(first, dict) else first
        img = ants.image_read(os.path.join(_task_path(task_name), img_rel))
        detected_ct = is_ct_image(img)
        assert detected_ct == expected_is_ct, (
            f"{task_name}: expected is_ct={expected_is_ct}, got {detected_ct}"
        )


# ---------------------------------------------------------------------------
# Test 2: 4D Volume Ingestion & Slicing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task01_BrainTumour"), reason="Task01 not found")
def test_decathlon_4d_volume_handling():
    """Verify 4D images (BrainTumour 4-channel, Prostate 2-channel) degrade to 3D cleanly."""
    for task_name in ["Task01_BrainTumour", "Task05_Prostate"]:
        if not _has_task(task_name):
            continue
        with open(os.path.join(_task_path(task_name), "dataset.json")) as f:
            meta = json.load(f)
        img_path = os.path.join(_task_path(task_name), meta["training"][0]["image"])
        img4d = ants.image_read(img_path)
        assert img4d.dimension == 4

        # Preprocess should automatically extract primary spatial channel
        pre = preprocess_for_landmarks(img4d)
        assert pre.dimension == 3
        assert pre.min() >= 0.0 and pre.max() <= 1.0

        # SIFT3D should succeed without error
        c, d = detect_sift3d(pre, preprocess=False, max_keypoints=64)
        assert len(c) > 0
        assert c.shape[1] == 4
        assert d.shape == (len(c), 512)


# ---------------------------------------------------------------------------
# Test 3: Foreground Landmark Validity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task02_Heart"), reason="Task02 not found")
def test_decathlon_keypoints_inside_foreground():
    """Verify that 100% of detected keypoints are inside the anatomical foreground."""
    for task_name in ["Task02_Heart", "Task09_Spleen"]:
        if not _has_task(task_name):
            continue
        with open(os.path.join(_task_path(task_name), "dataset.json")) as f:
            meta = json.load(f)
        img_path = os.path.join(_task_path(task_name), meta["training"][0]["image"])
        img = ants.image_read(img_path)
        pre = preprocess_for_landmarks(img)

        c, _ = detect_sift3d(pre, preprocess=False, max_keypoints=100)
        assert len(c) > 0

        # Sample intensities at physical coordinates
        vals = S.sample_tensor_at_physical(S.image_to_tensor(pre), pre, c[:, :3]).squeeze()
        if isinstance(vals, torch.Tensor):
            vals = vals.cpu().numpy()
        # All keypoints must reside in non-zero anatomy
        assert (vals > 0.01).all(), (
            f"{task_name}: {(vals <= 0.01).sum()} of {len(vals)} keypoints fell in zero padding!"
        )


# ---------------------------------------------------------------------------
# Test 4: LoG, DoG, and MIND-SSC on Decathlon
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task02_Heart"), reason="Task02 not found")
def test_decathlon_blob_and_mind():
    """Verify DoG, LoG, and 12-channel MIND-SSC descriptors on Decathlon data."""
    img_path = os.path.join(_task_path("Task02_Heart"), "imagesTr", "la_007.nii.gz")
    img = ants.image_read(img_path)
    pre = preprocess_for_landmarks(img)

    dog_kpts = detect_blobs_dog(pre, preprocess=False, max_keypoints=50)
    log_kpts = detect_blobs_log(pre, preprocess=False, max_keypoints=50)
    assert len(dog_kpts) > 0 and dog_kpts.shape[1] == 4
    assert len(log_kpts) > 0 and log_kpts.shape[1] == 4

    mind_vol = compute_mind(pre, offset_distance=2.0)
    assert mind_vol.shape[1] == 12
    descs = extract_mind_at_points(pre, dog_kpts[:, :3], mind_vol=mind_vol)
    assert descs.shape == (len(dog_kpts), 12)
    assert not np.isnan(descs).any()
    assert (descs >= 0.0).all() and (descs <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 5: Known Rigid Transform Recovery and TRE (< 1.0 mm)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task02_Heart"), reason="Task02 not found")
def test_decathlon_known_rigid_recovery():
    """Apply a known 3D rigid transform to real heart MRI and verify sub-millimeter TRE."""
    img_path = os.path.join(_task_path("Task02_Heart"), "imagesTr", "la_007.nii.gz")
    A = preprocess_for_landmarks(ants.image_read(img_path))
    cA, dA = detect_sift3d(A, preprocess=False, max_keypoints=300)

    ctr = np.array(ants.get_center_of_mass(A))
    a1, a2 = np.deg2rad(10.0), np.deg2rad(6.0)
    R = (
        np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    )
    t = np.array([4.0, -3.0, 2.0])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), t]))
    tx.set_fixed_parameters(ctr)
    B = tx.apply_to_image(A, A)

    cB, dB = detect_sift3d(B, preprocess=False, max_keypoints=300)
    m = match_landmarks(cA, cB, dA, dB, ratio_thresh=0.85, mutual=True)
    mf, M = ransac_filter(cA, cB, m, model="rigid", inlier_thresh_mm=3.0)
    assert len(mf) >= 20

    # Ground truth: T(x) = R (x - ctr) + ctr + t => x_B = R^-1 (x_A - ctr - t) + ctr
    R_inv = np.linalg.inv(R)
    truth = (R_inv @ (cA[mf[:, 0], :3] - ctr - t).T).T + ctr
    pred = (M[:3, :3] @ cA[mf[:, 0], :3].T).T + M[:3, 3]
    tre = np.linalg.norm(pred - truth, axis=1)
    assert tre.mean() < 1.0, f"Mean TRE {tre.mean():.3f} mm exceeded 1.0 mm!"


# ---------------------------------------------------------------------------
# Test 6: Rotation Search on Decathlon Volume
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task02_Heart"), reason="Task02 not found")
def test_decathlon_large_rotation_search():
    """Verify that match_sift3d_with_rotation_search recovers a 45 deg relative rotation."""
    img_path = os.path.join(_task_path("Task02_Heart"), "imagesTr", "la_007.nii.gz")
    A = preprocess_for_landmarks(ants.image_read(img_path))
    ctr = np.array(ants.get_center_of_mass(A))
    axis = np.array([0.0, 0.0, 1.0])
    a = np.deg2rad(45.0)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    t = np.array([2.0, -1.0, 0.5])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), t]))
    tx.set_fixed_parameters(ctr)
    B = tx.apply_to_image(A, A)

    res = match_sift3d_with_rotation_search(A, B, max_keypoints=300, n_scales=8)
    assert len(res["inliers"]) >= 50
    rot_error = abs(res["rotation_deg"] - 45.0)
    assert rot_error < 0.5, f"Rotation error {rot_error:.3f} deg exceeded 0.5 deg!"


# ---------------------------------------------------------------------------
# Test 7: Inter-Subject Landmark Matching & Alignment
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task04_Hippocampus"), reason="Task04 not found")
def test_decathlon_intersubject_hippocampus_alignment(tmp_path):
    """Verify inter-subject landmark matching improves Dice overlap on Hippocampus."""
    meta_path = os.path.join(_task_path("Task04_Hippocampus"), "dataset.json")
    with open(meta_path) as f:
        meta = json.load(f)

    c0, c1 = meta["training"][0], meta["training"][1]
    img0 = ants.image_read(os.path.join(_task_path("Task04_Hippocampus"), c0["image"]))
    lbl0 = ants.image_read(os.path.join(_task_path("Task04_Hippocampus"), c0["label"]))
    img1 = ants.image_read(os.path.join(_task_path("Task04_Hippocampus"), c1["image"]))
    lbl1 = ants.image_read(os.path.join(_task_path("Task04_Hippocampus"), c1["label"]))

    p0 = preprocess_for_landmarks(img0)
    p1 = preprocess_for_landmarks(img1)

    cf, df = detect_sift3d(p0, preprocess=False, max_keypoints=300, sigma_min=0.8, sigma_max=4.0)
    cm, dm = detect_sift3d(p1, preprocess=False, max_keypoints=300, sigma_min=0.8, sigma_max=4.0)

    matches = match_landmarks(cf, cm, df, dm, ratio_thresh=0.9, mutual=True)
    inliers, M = ransac_filter(cf, cm, matches, model="affine", inlier_thresh_mm=5.0)
    assert len(inliers) >= 6

    # Resample moving label to fixed space initially
    lbl1_init = ants.apply_transforms(fixed=lbl0, moving=lbl1, transformlist=[], interpolator="nearestNeighbor")
    df_init = ants.label_overlap_measures(lbl0, lbl1_init)
    dice_init = float(df_init.loc[df_init["Label"] == "All", "MeanOverlap"].iloc[0])

    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([M[:3, :3].ravel(), M[:3, 3]]))
    tx_file = str(tmp_path / "landmark_affine.mat")
    ants.write_transform(tx, tx_file)

    warped_lbl = ants.apply_transforms(
        fixed=lbl0, moving=lbl1, transformlist=[tx_file], whichtoinvert=[False], interpolator="nearestNeighbor"
    )
    df_aligned = ants.label_overlap_measures(lbl0, warped_lbl)
    dice_aligned = float(df_aligned.loc[df_aligned["Label"] == "All", "MeanOverlap"].iloc[0])

    assert dice_aligned > dice_init, f"Landmark alignment did not improve Dice: {dice_init:.3f} -> {dice_aligned:.3f}"
    assert dice_aligned > 0.50, f"Aligned Dice {dice_aligned:.3f} was below 0.50"


# ---------------------------------------------------------------------------
# Test 8: Ortho Slices and Display Projection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_task("Task02_Heart"), reason="Task02 not found")
def test_decathlon_ortho_slices_and_projection():
    """Verify orthographic slice extraction and projection on Decathlon data."""
    img_path = os.path.join(_task_path("Task02_Heart"), "imagesTr", "la_007.nii.gz")
    img = preprocess_for_landmarks(ants.image_read(img_path))
    kpts, _ = detect_sift3d(img, preprocess=False, max_keypoints=30)

    sl = S.extract_ortho_slices(img)
    for view_key in ("ax", "cor", "sag"):
        v_spec = sl["views"][view_key]
        u, v, in_slab = S.project_to_slice(img, kpts[:, :3], view=v_spec, slab_half_mm=25.0)
        assert len(u) == len(kpts)
        assert len(v) == len(kpts)
        assert in_slab.sum() > 0
