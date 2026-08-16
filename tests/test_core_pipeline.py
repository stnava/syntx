import pytest
import torch
import numpy as np
import ants

from syntx.core.pipeline import (
    auto_detect_device,
    normalize_and_tensorize,
    cleanup_gpu,
)


def test_auto_detect_device():
    dev = auto_detect_device(backend='pytorch')
    assert dev in ('cuda', 'mps', 'cpu')

    dev_explicit = auto_detect_device(requested_device='cpu')
    assert dev_explicit == 'cpu'


def test_normalize_and_tensorize():
    fi = ants.from_numpy(np.random.rand(16, 16).astype(np.float32))
    mi = ants.from_numpy(np.random.rand(16, 16).astype(np.float32))

    I_t, J_t = normalize_and_tensorize(fi, mi, backend='pytorch', device='cpu')
    assert isinstance(I_t, torch.Tensor)
    assert isinstance(J_t, torch.Tensor)
    assert I_t.shape == (1, 1, 16, 16)
    assert J_t.shape == (1, 1, 16, 16)


def test_cleanup_gpu():
    # Should run without error on CPU/MPS/CUDA
    cleanup_gpu(device='cpu', backend='pytorch')
