import pytest
import torch
import numpy as np

from syntx.core.smoothing import (
    separable_gaussian_filter,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    get_boundary_mask,
)


def test_separable_gaussian_filter_constant():
    # Gaussian filter of constant tensor should be identical constant
    grid = torch.ones(1, 16, 16, 2)
    filtered = separable_gaussian_filter(grid, sigma=2.0)
    assert torch.allclose(filtered, grid, atol=1e-5)


def test_get_boundary_mask():
    mask = get_boundary_mask((8, 8), device='cpu', dtype=torch.float32, rim_size=1)
    assert mask.shape == (1, 8, 8, 1)
    # Boundaries should be 0, interior should be 1
    assert (mask[0, 0, :, 0] == 0).all()
    assert (mask[0, -1, :, 0] == 0).all()
    assert (mask[0, :, 0, 0] == 0).all()
    assert (mask[0, :, -1, 0] == 0).all()
    assert mask[0, 2:6, 2:6, 0].mean() == 1.0


def test_sobolev_green_operator_2d():
    m = torch.randn(1, 16, 16, 2)
    v = apply_sobolev_green_operator(m, fluid_sigma=3.0)
    assert v.shape == m.shape
    # Smoothing reduces energy/variance
    assert torch.std(v) < torch.std(m)


def test_dsti_green_operator_2d_and_3d():
    m_2d = torch.randn(1, 16, 16, 2)
    v_dst_2d = apply_dsti_green_operator(m_2d, fluid_sigma=3.0)
    assert v_dst_2d.shape == m_2d.shape

    v_dst1_2d = apply_dsti1_green_operator(m_2d, fluid_sigma=3.0)
    assert v_dst1_2d.shape == m_2d.shape

    m_3d = torch.randn(1, 8, 8, 8, 3)
    v_dst_3d = apply_dsti_green_operator(m_3d, fluid_sigma=2.0)
    assert v_dst_3d.shape == m_3d.shape
