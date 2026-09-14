"""
Tests for Sulcal Soft Dice Guided Registration:
- Differentiable ND Soft Dice Loss
- Sulcal probability map extraction
- Turnkey syntx.syn(..., guided='sulcal') workflow
"""

import pytest
import numpy as np
import torch
import ants
import syntx
from syntx.core.losses import soft_dice_loss_nd
from syntx.surface import (
    compute_surface_classes,
    generate_surface_channels,
    extract_sulcal_probability_map,
)

def test_soft_dice_loss_nd_exact_and_gradients():
    """Verify Soft Dice loss identity, disjointness, and autograd gradient flow."""
    # 1. Exact overlap -> Loss should be 0.0 (Dice = 1.0)
    A = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    loss_identical = soft_dice_loss_nd(A, A)
    assert pytest.approx(loss_identical.item(), abs=1e-5) == 0.0

    # 2. Completely disjoint -> Loss should be ~1.0 (Dice ~ 0.0)
    B = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    B[:, :, :8, :] = 1.0
    C = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    C[:, :, 8:, :] = 1.0
    loss_disjoint = soft_dice_loss_nd(B, C)
    assert pytest.approx(loss_disjoint.item(), abs=1e-4) == 1.0

    # 3. Autograd gradient flow
    X = torch.rand((1, 2, 8, 8, 8), dtype=torch.float32, requires_grad=True)
    Y = torch.rand((1, 2, 8, 8, 8), dtype=torch.float32)
    loss = soft_dice_loss_nd(X, Y)
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()
    assert not torch.isinf(X.grad).any()

def test_extract_sulcal_probability_map_properties():
    """Verify extract_sulcal_probability_map preserves geometry and normalizes to [0, 1]."""
    img = ants.image_read(ants.get_ants_data('r16'))

    sulc_map = extract_sulcal_probability_map(img, curv_sigma=1.0, prob_sigma=1.0)
    assert isinstance(sulc_map, ants.ANTsImage)
    assert sulc_map.shape == img.shape
    assert sulc_map.spacing == img.spacing

    s_arr = sulc_map.numpy()
    assert s_arr.min() >= -1e-6
    assert s_arr.max() <= 1.0 + 1e-4
    assert np.any(s_arr > 0.0)

def test_guided_syn_turnkey_2d():
    """Verify end-to-end turnkey syntx.syn with guided='sulcal' on 2D images."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))

    res = syntx.syn(
        fixed=r16,
        moving=r64,
        guided='sulcal',
        cohort_type='inter',
        reg_iterations=[20, 10],
        affine_iterations=[10, 10],
        levels=[2, 1],
        verbose=False
    )

    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res
    assert len(res['fwdtransforms']) >= 1
    assert isinstance(res['warpedmovout'], ants.ANTsImage)

def test_guided_syn_weight_presets():
    """Verify that cohort_type and guided_weight select appropriate weight presets."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))

    # 1. Inter-study preset -> w=[0.30, 0.70]
    res_inter = syntx.syn(
        fixed=r16, moving=r64,
        guided='sulcal', cohort_type='inter',
        reg_iterations=[5], affine_iterations=[0], levels=[1],
        verbose=False
    )
    assert res_inter['model'].syn_metric_weights == [0.30, 0.70]

    # 2. Intra-study preset -> w=[0.80, 0.20]
    res_intra = syntx.syn(
        fixed=r16, moving=r64,
        guided='sulcal', cohort_type='intra',
        reg_iterations=[5], affine_iterations=[0], levels=[1],
        verbose=False
    )
    assert res_intra['model'].syn_metric_weights == [0.80, 0.20]

    # 3. Explicit guided_weight=0.65 -> w=[0.35, 0.65]
    res_custom = syntx.syn(
        fixed=r16, moving=r64,
        guided='sulcal', guided_weight=0.65,
        reg_iterations=[5], affine_iterations=[0], levels=[1],
        verbose=False
    )
    assert pytest.approx(res_custom['model'].syn_metric_weights[0], abs=1e-5) == 0.35
    assert pytest.approx(res_custom['model'].syn_metric_weights[1], abs=1e-5) == 0.65
