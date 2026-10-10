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
    from syntx.pyramid import build_pyramid as real_build, plan_levels
    G = importlib.import_module("syntx.greedy")

    fixed, moving = _smooth((40, 48, 36)), _smooth((30, 44, 52), seed=1)
    seen = {}

    def spy(image, *a, **k):
        out = real_build(image, *a, **k)
        seen[tuple(image.shape[2:])] = (out[0].plan.sigma_vox_zyx, tuple(out[0].tensor.shape[2:]))
        return out

    monkeypatch.setattr(G, 'build_pyramid', spy)
    model = G.GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[4], iterations=[1])
    for img in (fixed, moving):
        want = plan_levels(img.shape[2:], (1, 1, 1), [4], 'legacy_floor16')[0]
        assert seen[tuple(img.shape[2:])] == (want.sigma_vox_zyx, want.shape_zyx)
    assert seen[(40, 48, 36)][0] != seen[(30, 44, 52)][0]


def test_greedy_full_resolution_level_on_a_thin_axis_does_not_crash():
    """An axis shorter than 16 voxels at scale 1: the warp grid was floored to 16 while the images stayed native -> RuntimeError."""
    import syntx
    fixed, moving = _smooth((10, 40, 40)), torch.roll(_smooth((10, 40, 40)), (1, 2, -1), (2, 3, 4))
    from syntx.greedy import GreedyRegistrationModel
    model = GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    warp = model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[2, 1], iterations=[2, 2])
    assert tuple(warp.shape) == (1, 10, 40, 40, 3) and torch.isfinite(warp).all()
