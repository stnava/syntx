"""Regression: a sigma of 0 (or an axis with sigma <= 0) in ``landmarks/blob._separable_gaussian3d`` was 0 / 0 = NaN."""
import torch


def test_blob_gaussian_zero_sigma_is_identity_not_nan():
    from syntx.landmarks.blob import _separable_gaussian3d
    x = torch.rand(1, 1, 7, 8, 9)
    for sig in (0.0, (0.0, 0.0, 0.0)):
        out = _separable_gaussian3d(x, sig)
        assert torch.isfinite(out).all() and torch.equal(out, x)
    mixed = _separable_gaussian3d(x, (0.0, 1.0, 0.0))                              # only the y axis is smoothed
    assert torch.isfinite(mixed).all() and not torch.equal(mixed, x)
    assert torch.allclose(mixed, _separable_gaussian3d(x, (1e-9, 1.0, 1e-9)), atol=1e-6)
