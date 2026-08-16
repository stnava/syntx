import pytest
import torch
import numpy as np

from syntx.core.losses import (
    local_ncc_loss_nd,
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
)


def test_lncc_identical_images():
    # Identical images should yield CC = 1.0, hence loss = -1.0
    img = torch.randn(1, 1, 16, 16)
    loss = local_ncc_loss_nd(img, img, window_size=5)
    assert np.isclose(loss.item(), -1.0, atol=1e-4)


def test_lncc_analytical_vs_autograd():
    img1 = torch.randn(1, 1, 16, 16, requires_grad=True)
    img2 = torch.randn(1, 1, 16, 16, requires_grad=True)

    loss_autograd = local_ncc_loss_nd(img1, img2, window_size=5, use_ants_pseudo_gradient=False)
    loss_analytical = local_ncc_loss_nd(img1, img2, window_size=5, use_ants_pseudo_gradient=True)

    assert torch.allclose(loss_autograd, loss_analytical, atol=1e-4)


def test_lncc_cauchy_schwarz_and_variance_floor():
    # Flat image (zero variance) should not produce NaNs
    flat = torch.zeros(1, 1, 16, 16)
    loss = local_ncc_loss_nd(flat, flat, window_size=5)
    assert not torch.isnan(loss)


def test_b_spline_3():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    b = b_spline_3(x)
    assert np.isclose(b[0].item(), 2.0 / 3.0)
    assert np.isclose(b[1].item(), 1.0 / 6.0)
    assert np.isclose(b[2].item(), 0.0)
    assert np.isclose(b[3].item(), 0.0)


def test_mattes_mi_identical_images():
    img = torch.linspace(0, 1, 64).repeat(64, 1).unsqueeze(0).unsqueeze(0)
    loss = mattes_mi_loss_nd(img, img, num_bins=16)
    assert not torch.isnan(loss)
    assert loss < 0.0
