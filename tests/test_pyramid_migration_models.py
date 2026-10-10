"""Pass 3b: the deformable models and their one-off resamples use the central pyramid (``syntx.pyramid``).

Golden tests: the migrated call sites reproduce the incumbent pyramid code (``tests/legacy_pyramid_reference.py``, the code that
used to live in ``syntx/pyramid.py`` and in ``greedy`` / ``tvf_adj`` / ``scattered``) bit for bit. Regression tests: the latent
crash / NaN / wrong-axis bugs that were fixed on the way.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _smooth(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.rand(1, 1, 5, 5, 5, generator=g), size=shape, mode='trilinear', align_corners=True)


# ---------------------------------------------------------------------------------------------------------- greedy
def test_greedy_moving_image_gets_its_own_antialias_sigma(monkeypatch):
    """greedy applied the FIXED image's anti-alias sigma to the moving image when the sizes differ (identical for equal shapes)."""
    import importlib
    G = importlib.import_module("syntx.greedy")
    from syntx.pyramid import plan_levels

    fixed, moving = _smooth((40, 48, 36)), _smooth((30, 44, 52), seed=1)
    used = []
    real = G._separable_1d_filter

    def spy(x, gaussians):
        used.append((tuple(x.shape[2:]), tuple(int(g.numel()) for g in gaussians)))
        return real(x, gaussians)

    monkeypatch.setattr(G, '_separable_1d_filter', spy)
    model = G.GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[4], iterations=[1])
    by_shape = dict(used)
    pf = plan_levels(fixed.shape[2:], (1, 1, 1), [4], 'legacy_floor16')[0]
    pm_ = plan_levels(moving.shape[2:], (1, 1, 1), [4], 'legacy_floor16')[0]
    from syntx.core.smoothing import gaussian_1d_compact
    want_f = tuple(int(gaussian_1d_compact(s, truncated=2.0).numel()) for s in pf.sigma_vox_zyx)
    want_m = tuple(int(gaussian_1d_compact(s, truncated=2.0).numel()) for s in pm_.sigma_vox_zyx)
    assert want_f != want_m
    assert by_shape[tuple(fixed.shape[2:])] == want_f
    assert by_shape[tuple(moving.shape[2:])] == want_m
