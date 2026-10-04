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
