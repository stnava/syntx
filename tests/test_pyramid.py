"""
Unit tests for multi-resolution pyramid generation in syntx.pyramid.
"""

import pytest
import torch
import numpy as np
from syntx.pyramid import build_image_pyramid, build_anti_aliased_pyramid


def test_build_image_pyramid_2d():
    img = torch.randn(1, 1, 32, 32)
    pyramid = build_image_pyramid(img, spacing=(1.0, 1.0), levels=[4, 2, 1])
    assert len(pyramid) == 3
    assert pyramid[0].shape == (1, 1, 8, 8)
    assert pyramid[1].shape == (1, 1, 16, 16)
    assert pyramid[2].shape == (1, 1, 32, 32)


def test_build_anti_aliased_pyramid_3d():
    img = torch.randn(1, 1, 32, 32, 32)
    pyramid = build_anti_aliased_pyramid(img, levels=[4, 2, 1], min_size=8)
    assert len(pyramid) == 3
    assert pyramid[0].shape == (1, 1, 8, 8, 8)
    assert pyramid[1].shape == (1, 1, 16, 16, 16)
    assert pyramid[2].shape == (1, 1, 32, 32, 32)
    for p in pyramid:
        assert not torch.isnan(p).any()
