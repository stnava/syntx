"""Batched similarity losses in ``syntx.core.losses`` (moved there from ``motion_batched``, where they were local copies)."""

import numpy as np
import pytest
import torch

from syntx.core.losses import (
    batched_mattes_mi_from_weights,
    mattes_mi_from_weights,
    parzen_weights,
    pearson_correlation_loss,
)


def _samples(b=3, s=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    ref = torch.rand(s, generator=g) * 2 - 1
    frames = torch.stack([0.7 * ref + 0.3 * torch.rand(s, generator=g) * 2 - 0.3 + 0.05 * k for k in range(b)])
    return frames.clamp(-1, 1), ref


def test_batched_mi_matches_the_single_pair_loss_shared_and_per_frame_reference():
    frames, ref = _samples()
    wx = parzen_weights(frames.reshape(-1), num_bins=32).reshape(3, -1, 32)
    wy = parzen_weights(ref, num_bins=32)
    shared = batched_mattes_mi_from_weights(wx, wy)
    per = batched_mattes_mi_from_weights(wx, wy.unsqueeze(0).expand(3, -1, -1).contiguous())
    loop = torch.stack([mattes_mi_from_weights(wx[i], wy) for i in range(3)])
    assert torch.allclose(shared, loop, atol=1e-5) and torch.allclose(per, loop, atol=1e-5)
    assert (loop < 0).all()  # negative MI


def test_pearson_loss_matches_numpy_and_is_minus_one_for_identical_rows():
    frames, ref = _samples()
    shared = pearson_correlation_loss(frames, ref)
    expected = torch.tensor([-np.corrcoef(frames[i].numpy(), ref.numpy())[0, 1] for i in range(3)], dtype=torch.float32)
    assert torch.allclose(shared, expected, atol=1e-4)
    assert torch.allclose(pearson_correlation_loss(frames, frames), torch.full((3,), -1.0), atol=1e-4)
    assert torch.allclose(pearson_correlation_loss(frames, ref.unsqueeze(0).expand(3, -1)), shared, atol=1e-6)


def test_central_parzen_weights_on_a_batch_equal_the_dense_cubic_bspline():
    """Oracle: the dense cubic B-spline over all bins (what motion_batched used before it called the central function)."""
    v = (torch.rand(4, 300) * 2.4 - 1.2)
    nb, pad = 32, 2.0
    got = parzen_weights(v.reshape(-1), num_bins=nb).reshape(4, 300, nb)
    vc = torch.clamp(v, -1.0, 1.0)
    u = pad + (vc.unsqueeze(-1) + 1.0) * ((nb - 1 - 2 * pad) / 2.0) - torch.arange(nb, dtype=torch.float32)
    ax = u.abs()
    dense = torch.where(ax < 1.0, 2.0 / 3.0 - ax**2 + 0.5 * ax**3, torch.where(ax < 2.0, (2.0 - ax) ** 3 / 6.0, torch.zeros_like(ax)))
    assert torch.allclose(got, dense, atol=1e-6)
    assert torch.allclose(got.sum(-1), torch.ones(4, 300), atol=1e-5)
