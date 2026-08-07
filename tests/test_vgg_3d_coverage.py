"""
Targeted tests for VGG 3D modes, patch-walk, patch-grid sampling in syntx.syn.
"""

import torch
import pytest
from syntx.syn import TriPlanarVGG3DLoss


def test_vgg_feature_loss_nd_3d_modes():
    # Test vgg_mode='lncc_3d' in 3D
    vgg_loss_3d = TriPlanarVGG3DLoss(dim=3, mode='lncc_3d', feature_layers=[4])
    input_3d = torch.randn(1, 1, 10, 10, 10, requires_grad=True)
    target_3d = torch.randn(1, 1, 10, 10, 10, requires_grad=True)

    loss_lncc_3d = vgg_loss_3d(input_3d, target_3d)
    assert isinstance(loss_lncc_3d, torch.Tensor)
    loss_lncc_3d.backward()

    # Test vgg_mode='lncc' 2D orthogonal slice sampling in 3D
    vgg_loss_lncc = TriPlanarVGG3DLoss(dim=3, mode='lncc', feature_layers=[4], num_slices=2)
    loss_lncc = vgg_loss_lncc(input_3d, target_3d)
    assert isinstance(loss_lncc, torch.Tensor)

    # Test vgg_mode='patch_grid'
    vgg_loss_grid = TriPlanarVGG3DLoss(dim=3, mode='patch_grid', feature_layers=[4], patch_size=8, num_patches=2, num_slices=2)
    loss_grid = vgg_loss_grid(input_3d, target_3d)
    assert isinstance(loss_grid, torch.Tensor)

    # Test vgg_mode='patch_walk'
    vgg_loss_walk = TriPlanarVGG3DLoss(dim=3, mode='patch_walk', feature_layers=[4], patch_size=8, num_patches=2, num_slices=2)
    loss_walk = vgg_loss_walk(input_3d, target_3d)
    assert isinstance(loss_walk, torch.Tensor)
