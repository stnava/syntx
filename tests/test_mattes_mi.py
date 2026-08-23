"""
Unit tests for Mattes Mutual Information (Mattes MI) loss functional in syntx.
"""

import os
import pytest
import numpy as np
import torch
import ants
import syntx
from syntx.core.losses import mattes_mi_loss_nd, mattes_mi_loss_core, b_spline_3
from syntx.image_compare import image_compare


def test_b_spline_3():
    """Verify properties of 3rd order B-spline kernel."""
    x_zero = torch.tensor([0.0])
    # At x=0, B3(0) = 2/3
    assert torch.isclose(b_spline_3(x_zero), torch.tensor([2.0 / 3.0]))

    # At |x| >= 2, B3(x) = 0
    x_out = torch.tensor([-2.5, -2.0, 2.0, 3.0])
    assert torch.all(b_spline_3(x_out) == 0.0)

    # Symmetry B3(-x) == B3(x)
    x_rand = torch.rand(20) * 2.0
    assert torch.allclose(b_spline_3(x_rand), b_spline_3(-x_rand))


def test_mattes_mi_gradient_flow():
    """Verify that Mattes MI propagates valid non-zero gradients."""
    I = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    J = torch.randn(1, 1, 16, 16, 16)
    
    loss = mattes_mi_loss_nd(I, J, num_bins=16)
    assert loss.requires_grad
    loss.backward()
    
    assert I.grad is not None
    assert not torch.isnan(I.grad).any()
    assert I.grad.norm().item() > 0.0


def test_mattes_mi_identical_images():
    """Identical images should have higher Mutual Information (more negative loss) than random noise."""
    I = torch.randn(1, 1, 24, 24, 24)
    loss_identical = mattes_mi_loss_nd(I, I, num_bins=32).item()
    
    J_rand = torch.randn(1, 1, 24, 24, 24)
    loss_random = mattes_mi_loss_nd(I, J_rand, num_bins=32).item()
    
    # Negative MI minimization convention: lower loss means higher mutual information
    assert loss_identical < loss_random


def test_mattes_mi_inverted_contrast():
    """Mattes MI can capture inverted contrast (e.g. T1 vs T2 or inverted intensities)."""
    I = torch.randn(1, 1, 24, 24, 24)
    I_inverted = -I
    loss_inverted = mattes_mi_loss_nd(I, I_inverted, num_bins=32).item()
    
    J_rand = torch.randn(1, 1, 24, 24, 24)
    loss_random = mattes_mi_loss_nd(I, J_rand, num_bins=32).item()
    
    # Inverted images share identical information, so MI is high (loss is very negative)
    assert loss_inverted < loss_random


def test_image_compare_mattes_aliases():
    """Verify that image_compare accepts mattes, mattes_mi, mi, and bin variations."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))
    
    mi_val1 = image_compare(r16, r64, 'mattes_mi')
    mi_val2 = image_compare(r16, r64, 'mattes')
    mi_val3 = image_compare(r16, r64, 'mi')
    mi_val4 = image_compare(r16, r64, 'mattes_16')
    
    assert isinstance(mi_val1, float)
    assert np.isclose(mi_val1, mi_val2)
    assert np.isclose(mi_val1, mi_val3)
    assert isinstance(mi_val4, float)


def test_syntx_syn_with_mattes_mi():
    """Test 2D SyN registration using similarity_metric='mattes_mi'."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))
    
    reg = syntx.syn(
        fixed=r16,
        moving=r64,
        similarity_metric='mattes_mi',
        reg_iterations=[20, 10],
        verbose=False
    )
    
    assert 'warpedmovout' in reg
    assert 'fwdtransforms' in reg
    assert reg['warpedmovout'] is not None


def test_syntx_tvf_with_mattes_mi():
    """Test 2D TVF registration using similarity_metric='mattes_mi'."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))
    
    reg = syntx.tvf(
        fixed=r16,
        moving=r64,
        similarity_metric='mattes_mi',
        reg_iterations=[15, 10],
        verbose=False
    )
    
    assert 'warpedmovout' in reg
    assert 'fwdtransforms' in reg
    assert reg['warpedmovout'] is not None
