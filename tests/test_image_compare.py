import os
import sys
import types
import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp
import ants

# Mock MONAI modules at runtime to avoid import issues
class MockSwinViT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B = x.shape[0]
        return [
            torch.zeros(B, 48, 48, 48, 48), # layer 1
            torch.zeros(B, 96, 24, 24, 24), # layer 2
            torch.zeros(B, 192, 12, 12, 12), # layer 3
            torch.zeros(B, 384, 6, 6, 6),    # layer 4
            torch.zeros(B, 384, 3, 3, 3)     # layer 5
        ]

class MockSwinUNETR(torch.nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, **kwargs):
        super().__init__()
        self.swinViT = MockSwinViT()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.zeros(x.shape[0], 14, x.shape[2]//2, x.shape[3]//2, x.shape[4]//2)

monai_module = types.ModuleType('monai')
monai_networks = types.ModuleType('monai.networks')
monai_networks_nets = types.ModuleType('monai.networks.nets')

monai_networks_nets.SwinUNETR = MockSwinUNETR
monai_networks_nets.SwinViT = MockSwinViT
monai_networks.nets = monai_networks_nets
monai_module.networks = monai_networks

sys.modules['monai'] = monai_module
sys.modules['monai.networks'] = monai_networks
sys.modules['monai.networks.nets'] = monai_networks_nets

# Now we can import image_compare
from syntx import image_compare

# Generate the 88 supported metric names
ALL_METRICS = [
    'mse', 'mae', 'rmse', 'psnr', 'ncc', 'nmi', 'joint_entropy',
    'lncc_w3', 'lncc_w5', 'lncc_w7', 'lncc_w9', 'lncc_w11',
    'mmi_b16', 'mmi_b32', 'mmi_b64', 'mmi_b128', 'mmi_b256',
    'ssim', 'gradient_mse', 'gradient_correlation', 'ngf_e01', 'ngf_e1', 'ngf_e10', 'ms_ssim'
]
# VGG
for l in [2, 4, 8, 12]:
    for m in ['l1', 'l2', 'lncc', 'cos']:
        ALL_METRICS.append(f'vgg_{l}_{m}')
# DINOv2
for l in [1, 2, 6, 11]:
    for m in ['l1', 'l2', 'lncc', 'cos']:
        ALL_METRICS.append(f'dino_{l}_{m}')
# ResNet10
for l in [1, 2, 3, 4]:
    for m in ['l1', 'l2', 'lncc', 'cos']:
        ALL_METRICS.append(f'resnet_{l}_{m}')
# SwinUNETR
for l in [1, 2, 3, 4]:
    for m in ['l1', 'l2', 'lncc', 'cos']:
        ALL_METRICS.append(f'swin_{l}_{m}')


def test_input_conversions_and_dimensions():
    # Create different formats
    np_2d_a = np.random.rand(16, 16).astype(np.float32)
    np_2d_b = np.random.rand(16, 16).astype(np.float32)
    
    # 1. NumPy arrays
    score_np = image_compare(np_2d_a, np_2d_b, 'mse')
    assert isinstance(score_np, float)
    
    # 2. PyTorch tensors
    torch_2d_a = torch.from_numpy(np_2d_a)
    torch_2d_b = torch.from_numpy(np_2d_b)
    score_torch = image_compare(torch_2d_a, torch_2d_b, 'mse')
    assert isinstance(score_torch, float)
    assert np.allclose(score_np, score_torch)
    
    # 3. JAX arrays
    jax_2d_a = jnp.array(np_2d_a)
    jax_2d_b = jnp.array(np_2d_b)
    score_jax = image_compare(jax_2d_a, jax_2d_b, 'mse')
    assert isinstance(score_jax, float)
    assert np.allclose(score_np, score_jax)
    
    # 4. ANTsImage
    ants_2d_a = ants.from_numpy(np_2d_a)
    ants_2d_b = ants.from_numpy(np_2d_b)
    score_ants = image_compare(ants_2d_a, ants_2d_b, 'mse')
    assert isinstance(score_ants, float)
    assert np.allclose(score_np, score_ants)


def test_invalid_arguments_and_shapes():
    np_2d = np.random.rand(16, 16).astype(np.float32)
    np_3d = np.random.rand(16, 16, 16).astype(np.float32)
    np_mismatch = np.random.rand(16, 15).astype(np.float32)
    
    # Mismatched shapes
    with pytest.raises(ValueError):
        image_compare(np_2d, np_mismatch, 'mse')
    with pytest.raises(ValueError):
        image_compare(np_2d, np_3d, 'mse')
        
    # Invalid metric name
    with pytest.raises(ValueError):
        image_compare(np_2d, np_2d, 'invalid_metric_name_123')


def test_classical_and_spatial_metrics_2d():
    # Generate structured images so that identical vs mismatched is clear
    np.random.seed(42)
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    img_a = (100.0 * (xx**2 + yy**2)).astype(np.float32)
    img_b = img_a.copy()
    # Add noise to make a mismatched image
    img_c = img_a + 5.0 * np.random.rand(16, 16).astype(np.float32)
    
    classical_spatial = [
        'mse', 'mae', 'rmse', 'psnr', 'ncc', 'nmi', 'joint_entropy',
        'lncc_w3', 'lncc_w5', 'lncc_w7', 'lncc_w9', 'lncc_w11',
        'mmi_b16', 'mmi_b32', 'mmi_b64', 'mmi_b128', 'mmi_b256',
        'ssim', 'gradient_mse', 'gradient_correlation', 'ngf_e01', 'ngf_e1', 'ngf_e10', 'ms_ssim'
    ]
    
    for metric in classical_spatial:
        score_identical = image_compare(img_a, img_b, metric)
        score_diff = image_compare(img_a, img_c, metric)
        assert isinstance(score_identical, float)
        assert isinstance(score_diff, float)
        
        # Verify smaller is better: score_identical should be <= score_diff
        # Use a tolerance for float differences if any
        assert score_identical <= score_diff + 1e-5


def test_classical_and_spatial_metrics_3d():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    z = np.linspace(-1, 1, 16)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    img_a = (100.0 * (xx**2 + yy**2 + zz**2)).astype(np.float32)
    img_b = img_a.copy()
    img_c = img_a + 5.0 * np.random.rand(16, 16, 16).astype(np.float32)
    
    # Run a representative subset of classical/spatial metrics in 3D to keep test suite fast
    test_subset = [
        'mse', 'mae', 'rmse', 'psnr', 'ncc', 'nmi', 'joint_entropy',
        'lncc_w5', 'mmi_b32', 'ssim', 'gradient_mse', 'gradient_correlation',
        'ngf_e1', 'ms_ssim'
    ]
    
    for metric in test_subset:
        score_identical = image_compare(img_a, img_b, metric)
        score_diff = image_compare(img_a, img_c, metric)
        assert isinstance(score_identical, float)
        assert isinstance(score_diff, float)
        assert score_identical <= score_diff + 1e-5


def test_deep_feature_metrics_2d():
    # VGG19, DINOv2, ResNet10, SwinUNETR
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    img_a = (100.0 * (xx**2 + yy**2)).astype(np.float32)
    img_b = img_a.copy()
    img_c = img_a + 5.0 * np.random.rand(16, 16).astype(np.float32)
    
    # Run a representative subset of deep feature configurations in 2D
    test_subset = [
        'vgg_2_l1', 'vgg_4_l2', 'vgg_8_lncc', 'vgg_12_cos',
        'dino_1_l1', 'dino_2_l2', 'dino_6_lncc', 'dino_11_cos',
        'resnet_1_l1', 'resnet_2_l2', 'resnet_3_lncc', 'resnet_4_cos'
    ]
    
    for metric in test_subset:
        score_identical = image_compare(img_a, img_b, metric)
        score_diff = image_compare(img_a, img_c, metric)
        assert isinstance(score_identical, float)
        assert isinstance(score_diff, float)
        assert score_identical <= score_diff + 1e-5


def test_deep_feature_metrics_3d():
    # VGG19, DINOv2, ResNet10, SwinUNETR in 3D
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16)
    z = np.linspace(-1, 1, 16)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    img_a = (100.0 * (xx**2 + yy**2 + zz**2)).astype(np.float32)
    img_b = img_a.copy()
    img_c = img_a + 5.0 * np.random.rand(16, 16, 16).astype(np.float32)
    
    test_subset = [
        'vgg_4_lncc',  # VGG 3D LNCC Layer 4 Requirement
        'vgg_2_l1',
        'dino_2_l2',
        'resnet_2_lncc',
        'swin_2_lncc',
        'swin_4_cos'
    ]
    
    for metric in test_subset:
        score_identical = image_compare(img_a, img_b, metric)
        score_diff = image_compare(img_a, img_c, metric)
        assert isinstance(score_identical, float)
        assert isinstance(score_diff, float)
        assert score_identical <= score_diff + 1e-5


def test_all_88_configurations_runnable_2d():
    # Verify that all 88 metrics run without throwing errors
    img_a = np.random.rand(16, 16).astype(np.float32)
    img_b = np.random.rand(16, 16).astype(np.float32)
    
    for metric in ALL_METRICS:
        # SwinUNETR is a 3D-only model natively, so running SwinUNETR on 2D images
        # might raise a ValueError "Cannot run 3D feature extractor on 2D input."
        # We catch that or skip SwinUNETR on 2D
        if metric.startswith('swin_'):
            with pytest.raises(ValueError):
                _ = image_compare(img_a, img_b, metric)
        else:
            val = image_compare(img_a, img_b, metric)
            assert isinstance(val, float)


def test_all_88_configurations_runnable_3d():
    img_a = np.random.rand(16, 16, 16).astype(np.float32)
    img_b = np.random.rand(16, 16, 16).astype(np.float32)
    
    for metric in ALL_METRICS:
        val = image_compare(img_a, img_b, metric)
        assert isinstance(val, float)
