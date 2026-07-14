import pytest
import torch
import numpy as np

from syntx.features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, FeatureExtractor
from syntx.resnet import resnet10_2d, resnet10_3d
from syntx import registration, SyNTo

def test_resnet10_architectures():
    # Test 2D ResNet-10
    model_2d = resnet10_2d()
    model_2d.eval()
    x_2d = torch.randn(2, 1, 64, 64)
    out_2d = model_2d(x_2d)
    assert out_2d.shape == (2, 512, 2, 2)

    # Test 3D ResNet-10
    model_3d = resnet10_3d()
    model_3d.eval()
    x_3d = torch.randn(2, 1, 32, 32, 32)
    out_3d = model_3d(x_3d)
    assert out_3d.shape == (2, 512, 1, 1, 1)


def test_extractors_and_loss_shapes():
    device = 'cpu'
    
    # 1. VGG19 Extractor shape test (2D features)
    vgg_ext = VGG19Extractor(feature_layers=[3, 8])
    x_rgb = torch.randn(2, 3, 64, 64)
    feats = vgg_ext.extract(vgg_ext.normalize(x_rgb))
    assert len(feats) == 2
    # Layer 3 is conv1_2 (out channels 64, size 64x64)
    assert feats[0].shape == (2, 64, 64, 64)
    # Layer 8 is conv2_2 (out channels 128, size 32x32)
    assert feats[1].shape == (2, 128, 32, 32)

    # 2. DINOv2 Extractor shape test & Pruning validation
    dino_ext = DINOv2Extractor(version='vits14', feature_layers=[2])
    # DINOv2 ViT-S block list should be pruned to 3 blocks (indices 0, 1, 2)
    assert len(dino_ext.model.blocks) == 3
    x_rgb_dino = torch.randn(2, 3, 56, 56) # divisible by 14
    feats_dino = dino_ext.extract(dino_ext.normalize(x_rgb_dino))
    assert len(feats_dino) == 1
    # ViT-S hidden size is 384, patches count = (56/14) * (56/14) = 4 * 4
    assert feats_dino[0].shape == (2, 384, 4, 4)

    # 3. ResNet-10 Extractor shape test
    resnet_ext_2d = ResNet10Extractor(dim=2, feature_layers=[1, 4])
    assert not resnet_ext_2d.is_3d
    x_gray_2d = torch.randn(2, 1, 32, 32)
    feats_resnet_2d = resnet_ext_2d.extract(resnet_ext_2d.normalize(x_gray_2d))
    assert len(feats_resnet_2d) == 2
    # Layer 1 shape: (B, 64, H/4, W/4) = (2, 64, 8, 8)
    assert feats_resnet_2d[0].shape == (2, 64, 8, 8)
    # Layer 4 shape: (B, 512, H/32, W/32) = (2, 512, 1, 1)
    assert feats_resnet_2d[1].shape == (2, 512, 1, 1)

    resnet_ext_3d = ResNet10Extractor(dim=3, feature_layers=[2])
    assert resnet_ext_3d.is_3d
    x_gray_3d = torch.randn(2, 1, 32, 32, 32)
    feats_resnet_3d = resnet_ext_3d.extract(resnet_ext_3d.normalize(x_gray_3d))
    assert len(feats_resnet_3d) == 1
    # Layer 2 shape: (B, 128, D/8, H/8, W/8) = (2, 128, 4, 4, 4)
    assert feats_resnet_3d[0].shape == (2, 128, 4, 4, 4)


def test_feature_space_loss():
    # 1. 2D direct mode
    dino_ext = DINOv2Extractor(version='vits14', feature_layers=[2])
    loss_fn_2d = FeatureSpaceLoss(extractor=dino_ext, mode='lncc')
    I_2d = torch.rand(1, 1, 56, 56)
    J_2d = torch.rand(1, 1, 56, 56)
    val_2d = loss_fn_2d(I_2d, J_2d)
    assert isinstance(val_2d, torch.Tensor)
    assert val_2d.ndim == 0 # scalar

    # 2. 3D native mode
    resnet_ext_3d = ResNet10Extractor(dim=3, feature_layers=[2])
    loss_fn_3d = FeatureSpaceLoss(extractor=resnet_ext_3d, mode='lncc_3d')
    I_3d = torch.rand(1, 1, 16, 16, 16)
    J_3d = torch.rand(1, 1, 16, 16, 16)
    val_3d = loss_fn_3d(I_3d, J_3d)
    assert isinstance(val_3d, torch.Tensor)
    assert val_3d.ndim == 0


def test_multimetric_fitting():
    # Test multi-metric initialization and forward loss sum in PyTorch SyNTo
    device = 'cpu'
    model = SyNTo(dim=2, grid_shape=(16, 16))
    
    I_2d = torch.rand(1, 1, 32, 32)
    J_2d = torch.rand(1, 1, 32, 32)
    
    # Standardize list of metrics: combined intensity + perceptual ResNet10
    model.fit(
        I_2d, J_2d,
        levels=[2, 1],
        epochs_per_level=[2, 1],
        affine_epochs=[2, 1],
        similarity_metric=['lncc', 'resnet10'],
        syn_metric_weights=[0.6, 0.4],
        vgg_layers=[2]
    )
    
    assert len(model.metrics) == 2
    assert model.metrics[0] == 'lncc'
    assert model.metrics[1] == 'resnet10'
    assert len(model.loss_functions) == 2
    assert len(model.affine_losses) > 0
    assert len(model.syn_losses) > 0


def test_feature_space_triplanar_and_reconstruct():
    # Test VGG19 triplanar and reconstruct-3D forward passes
    vgg_ext = VGG19Extractor(feature_layers=[3])
    loss_triplanar = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc')
    loss_reconstruct = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d')

    I_3d = torch.rand(1, 1, 16, 16, 16)
    J_3d = torch.rand(1, 1, 16, 16, 16)

    val_triplanar = loss_triplanar(I_3d, J_3d)
    assert isinstance(val_triplanar, torch.Tensor)
    assert val_triplanar.ndim == 0

    val_reconstruct = loss_reconstruct(I_3d, J_3d)
    assert isinstance(val_reconstruct, torch.Tensor)
    assert val_reconstruct.ndim == 0


def test_extractor_base_class_errors():
    class DummyExtractor(FeatureExtractor):
        pass

    dummy = DummyExtractor()
    with pytest.raises(NotImplementedError):
        _ = dummy.is_3d
    with pytest.raises(NotImplementedError):
        _ = dummy.in_channels
    with pytest.raises(NotImplementedError):
        dummy.normalize(torch.zeros(1))
    with pytest.raises(NotImplementedError):
        dummy.extract(torch.zeros(1))


def test_extractor_dimension_error():
    resnet_ext_3d = ResNet10Extractor(dim=3, feature_layers=[2])
    loss_fn = FeatureSpaceLoss(extractor=resnet_ext_3d, mode='lncc_3d')
    I_2d = torch.rand(1, 1, 16, 16)
    J_2d = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError):
        loss_fn(I_2d, J_2d)

