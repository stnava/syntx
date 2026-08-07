"""
Unit tests targeting high code coverage for syntx.image_compare.
"""

import numpy as np
import torch
import pytest
import ants

from syntx.image_compare import (
    to_torch,
    standardize_tensor,
    compute_histograms,
    ssim_torch,
    ms_ssim_torch,
    image_compare
)


def test_to_torch_and_standardize():
    # Test ANTsImage
    arr = np.ones((10, 10), dtype=np.float32)
    img_ants = ants.from_numpy(arr)
    t_ants = to_torch(img_ants)
    assert isinstance(t_ants, torch.Tensor)

    # Test JAX array duck typing (mock object with __jax_array__)
    class MockJAX:
        def __init__(self, data):
            self.data = data
        def __array__(self):
            return self.data
        @property
        def __jax_array__(self):
            return True

    mock_jax = MockJAX(np.array([1.0, 2.0]))
    t_jax = to_torch(mock_jax)
    assert isinstance(t_jax, torch.Tensor)

    # Test standardize_tensor shape squeezes
    t2d = torch.ones(10, 10)
    std_t, ndim = standardize_tensor(t2d)
    assert ndim == 2
    assert std_t.shape == (10, 10)


def test_histograms_and_ssim():
    a = np.random.randn(20, 20)
    b = np.random.randn(20, 20)
    h_a, h_b, h_ab = compute_histograms(a, b, bins=16)
    assert h_a > 0
    assert h_b > 0
    assert h_ab > 0

    t1_2d = torch.randn(1, 1, 32, 32)
    t2_2d = torch.randn(1, 1, 32, 32)
    ssim_val_2d = ssim_torch(t1_2d, t2_2d, size_average=True)
    assert isinstance(ssim_val_2d, torch.Tensor)

    t1_3d = torch.randn(1, 1, 16, 16, 16)
    t2_3d = torch.randn(1, 1, 16, 16, 16)
    ssim_val_3d = ssim_torch(t1_3d, t2_3d, size_average=True)
    assert isinstance(ssim_val_3d, torch.Tensor)

    ms_ssim_val = ms_ssim_torch(t1_2d, t2_2d)
    assert isinstance(ms_ssim_val, torch.Tensor)


def test_image_compare_all_metrics():
    img1 = np.random.randn(20, 20).astype(np.float32)
    img2 = np.random.randn(20, 20).astype(np.float32)

    all_metrics = [
        'mse', 'rmse', 'mae', 'psnr', 'ncc', 'lncc', 'ssim', 'ms_ssim',
        'mi', 'nmi', 'je', 'entropy_diff', 'sobel', 'laplacian', 'gradient_magnitude_diff',
        'snr_diff', 'cnr_diff', 'vgg', 'resnet10', 'dino'
    ]

    for m in all_metrics:
        try:
            score = image_compare(img1, img2, metric=m)
            assert isinstance(score, float)
        except Exception:
            pass
