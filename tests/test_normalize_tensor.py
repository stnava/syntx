import torch
import pytest
import syntx

def test_normalize_tensor_minmax():
    x = torch.tensor([-5.0, 0.0, 5.0, 10.0])
    norm = syntx.normalize_tensor(x, method='minmax')
    assert torch.allclose(norm, torch.tensor([0.0, 1/3, 2/3, 1.0]), atol=1e-4)

def test_normalize_tensor_zscore():
    x = torch.tensor([-5.0, 0.0, 5.0, 10.0])
    norm = syntx.normalize_tensor(x, method='zscore')
    assert abs(norm.mean().item()) < 1e-5
    assert abs(norm.std(unbiased=False).item() - 1.0) < 1e-4

def test_normalize_tensor_robust():
    x = torch.linspace(-10.0, 10.0, 100)
    norm = syntx.normalize_tensor(x, method='robust', p_min=5, p_max=95)
    assert norm.min().item() >= 0.0
    assert norm.max().item() <= 1.0

def test_normalize_tensor_l2():
    x = torch.tensor([3.0, 4.0])
    norm = syntx.normalize_tensor(x, method='l2')
    assert torch.allclose(norm, torch.tensor([0.6, 0.8]), atol=1e-4)

def test_normalize_tensor_l1():
    x = torch.tensor([3.0, 4.0])
    norm = syntx.normalize_tensor(x, method='l1')
    assert torch.allclose(norm, torch.tensor([3/7, 4/7]), atol=1e-4)

def test_normalize_tensor_sigmoid():
    x = torch.tensor([0.0])
    norm = syntx.normalize_tensor(x, method='sigmoid')
    assert torch.allclose(norm, torch.tensor([0.5]))
