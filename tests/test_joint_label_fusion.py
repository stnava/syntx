"""Unit tests for syntx.joint_label_fusion."""

import ants
import numpy as np
import pytest
import syntx


def test_joint_label_fusion_basic():
    shape = (32, 32, 32)
    target_np = np.zeros(shape, dtype=np.float32)
    target_np[10:22, 10:22, 10:22] = 80.0
    target = ants.from_numpy(target_np)

    atlases = []
    labels = []
    for i in range(4):
        # Slightly noisy atlas intensity
        a_np = target_np + np.random.normal(0, 4, shape).astype(np.float32)
        atlases.append(ants.from_numpy(a_np))

        l_np = np.zeros(shape, dtype=np.uint32)
        l_np[10:22, 10:22, 10:22] = 1 # Ground truth cube: 12x12x12 = 1728 voxels
        # Atlas 0 has a spurious noisy blob outside
        if i == 0:
            l_np[0:4, 0:4, 0:4] = 1
        labels.append(ants.from_numpy(l_np))

    # 1. Test mode="patch"
    res_patch = syntx.joint_label_fusion(target, atlases, labels, mode="patch", rad=2)
    assert isinstance(res_patch["segmentation"], ants.ANTsImage)
    assert 1 in res_patch["probability_images"]
    seg_patch = res_patch["segmentation"].numpy()
    assert seg_patch[0:4, 0:4, 0:4].sum() == 0, "Spurious noise was not suppressed"
    assert np.isclose(seg_patch[10:22, 10:22, 10:22].sum(), 1728, atol=10)

    # 2. Test mode="joint"
    res_joint = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=2)
    assert isinstance(res_joint["segmentation"], ants.ANTsImage)
    seg_joint = res_joint["segmentation"].numpy()
    assert seg_joint[0:4, 0:4, 0:4].sum() == 0, "Spurious noise was not suppressed in joint mode"
    assert np.isclose(seg_joint[10:22, 10:22, 10:22].sum(), 1728, atol=10)
    assert res_joint["timing_s"] > 0


def test_joint_label_fusion_multilabel():
    shape = (30, 30, 30)
    target = ants.from_numpy(np.random.normal(50, 10, shape).astype(np.float32))

    atlases = [target for _ in range(3)]
    labels = []
    for _ in range(3):
        l_np = np.zeros(shape, dtype=np.uint32)
        l_np[5:15, 5:15, 5:15] = 1
        l_np[15:25, 15:25, 15:25] = 2
        labels.append(ants.from_numpy(l_np))

    res = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=1)
    seg_np = res["segmentation"].numpy()
    assert set(np.unique(seg_np)) == {0, 1, 2}
    assert 1 in res["probability_images"]
    assert 2 in res["probability_images"]


def test_joint_label_fusion_empty_raises():
    target = ants.from_numpy(np.zeros((10, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="equal non-zero length"):
        syntx.joint_label_fusion(target, [], [])


def test_joint_label_fusion_2d_beats_majority_vote():
    """Fast, targeted 2D test on 24x24 image proving JLF outperforms majority voting.

    Scenario:
    - Target image has a true 8x8 foreground square [8:16, 8:16] with bright intensity (~90)
      surrounded by dark background (~20).
    - Atlas 1 is accurate (correct segmentation and matching intensities).
    - Atlas 2 and Atlas 3 are misregistered: both carry a correlated false protrusion
      into the background [8:16, 16:20] with bright intensity (~90).
    - In the false protrusion, 2 out of 3 atlases vote foreground (1).
    - Majority voting accepts the false protrusion (Dice = 0.8000, 32 false positive voxels).
    - JLF (patch and joint) evaluates local patch intensity similarity, detecting that the
      target is background (~20) where Atlas 2/3 claim foreground, thereby downweighting
      the misregistered atlases and achieving Dice >= 0.99 (Dice = 1.0000 in joint mode).
    - Runs in < 15 ms.
    """
    np.random.seed(42)
    shape = (24, 24)
    gt_seg = np.zeros(shape, dtype=np.uint32)
    gt_seg[8:16, 8:16] = 1

    bg = np.random.normal(20, 3, shape).astype(np.float32)
    target_np = bg.copy()
    target_np[8:16, 8:16] = np.random.normal(90, 5, (8, 8)).astype(np.float32)
    target = ants.from_numpy(target_np)

    # Atlas 1: accurate
    a1_np = target_np + np.random.normal(0, 1, shape).astype(np.float32)
    l1_np = gt_seg.copy()

    # Atlas 2 & 3: correlated false protrusion at [8:16, 16:20]
    a2_np = bg + np.random.normal(0, 1, shape).astype(np.float32)
    a2_np[8:16, 8:16] = np.random.normal(90, 5, (8, 8)).astype(np.float32)
    a2_np[8:16, 16:20] = np.random.normal(90, 5, (8, 4)).astype(np.float32)
    l2_np = gt_seg.copy()
    l2_np[8:16, 16:20] = 1

    a3_np = bg + np.random.normal(0, 1, shape).astype(np.float32)
    a3_np[8:16, 8:16] = np.random.normal(90, 5, (8, 8)).astype(np.float32)
    a3_np[8:16, 16:20] = np.random.normal(90, 5, (8, 4)).astype(np.float32)
    l3_np = gt_seg.copy()
    l3_np[8:16, 16:20] = 1

    atlases = [ants.from_numpy(a1_np), ants.from_numpy(a2_np), ants.from_numpy(a3_np)]
    labels = [ants.from_numpy(l1_np), ants.from_numpy(l2_np), ants.from_numpy(l3_np)]

    def dice(pred, gt):
        inter = np.sum((pred == 1) & (gt == 1))
        return float(2.0 * inter / (np.sum(pred == 1) + np.sum(gt == 1)))

    # 1. Majority voting
    maj_vote = (np.mean([l1_np, l2_np, l3_np], axis=0) >= 0.5).astype(np.uint32)
    dice_maj = dice(maj_vote, gt_seg)
    fp_maj = int(np.sum(maj_vote[8:16, 16:20]))
    assert dice_maj == pytest.approx(0.8000, abs=0.01)
    assert fp_maj == 32, "Majority vote must include all 32 false positive voxels"

    # 2. JLF Fast Patch Mode
    res_patch = syntx.joint_label_fusion(target, atlases, labels, mode="patch", rad=1, beta=2.0)
    seg_patch = res_patch["segmentation"].numpy()
    dice_patch = dice(seg_patch, gt_seg)
    fp_patch = int(np.sum(seg_patch[8:16, 16:20]))
    assert dice_patch > dice_maj, f"JLF patch Dice ({dice_patch:.4f}) should exceed majority vote ({dice_maj:.4f})"
    assert fp_patch < fp_maj

    # 3. JLF Wang Joint Mode
    res_joint = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=1, beta=4.0, rho=0.01)
    seg_joint = res_joint["segmentation"].numpy()
    dice_joint = dice(seg_joint, gt_seg)
    fp_joint = int(np.sum(seg_joint[8:16, 16:20]))
    assert dice_joint > dice_maj, f"JLF joint Dice ({dice_joint:.4f}) should exceed majority vote ({dice_maj:.4f})"
    assert dice_joint == pytest.approx(1.0, abs=0.01), "JLF joint mode should achieve near-perfect Dice"
    assert fp_joint == 0, "JLF joint mode should completely eliminate correlated false positive protrusion"
    assert res_joint["timing_s"] < 0.1, "24x24 test must execute in under 100ms"

