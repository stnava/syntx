"""Unit tests for multi-resolution pyramid generation in syntx.pyramid (the ``deformable`` preset: corner-aligned grid, 16-voxel floor)."""

import torch

from syntx.pyramid import build_pyramid


def test_build_pyramid_2d():
    img = torch.randn(1, 1, 64, 64)
    pyramid = build_pyramid(img, (1.0, 1.0), [4, 2, 1], 'deformable')
    assert [lv.tensor.shape[2:] for lv in pyramid] == [(17, 17), (33, 33), (64, 64)]     # round((n - 1) / L) + 1: corners stay on corners
    assert pyramid[-1].tensor is img                                                      # level 1 is the native grid


def test_build_pyramid_3d_floor_and_finite():
    img = torch.randn(1, 1, 32, 32, 32)
    pyramid = build_pyramid(img, (1.0,) * 3, [4, 2, 1], 'deformable')
    assert [lv.tensor.shape[2:] for lv in pyramid] == [(16, 16, 16), (17, 17, 17), (32, 32, 32)]   # level 4 hits the one 16-voxel floor
    for lv in pyramid:
        assert not torch.isnan(lv.tensor).any()
