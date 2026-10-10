"""Pass 3a: robust_affine and motion_batched build their pyramid levels through the central ``syntx.pyramid`` module.

Byte-level evidence against the pre-migration commit (transforms of the real NM pairs, the thin-slab sweep, motion_batched) is
produced by ``benchmarks/pyramid_migration_identity.py``; these are the fast structural and numerical checks.
"""
from __future__ import annotations

import importlib
import inspect
import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from syntx.pyramid import area_matrix, build_pyramid, linear_matrix

RA = importlib.import_module("syntx.robust_affine")
MB = importlib.import_module("syntx.motion_batched")


@pytest.mark.parametrize("shape", [(16, 20, 24), (17, 19, 23), (33, 35, 37)])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_legacy_avgpool_is_bit_identical_to_voxel_avg_pool(shape, level):
    x = torch.rand(1, 1, *shape)
    lv = build_pyramid(x, (1.0, 1.0, 1.0), [level], policy="legacy_avgpool")[0]
    ref = F.avg_pool3d(x, kernel_size=level, stride=level) if level > 1 else x
    assert torch.equal(lv.tensor, ref)
    assert list(lv.f) == [float(level)] * 3 and list(lv.off) == [(level - 1) / 2.0] * 3


def test_legacy_avgpool_2d_and_batched_moving_stack():
    x = torch.rand(1, 1, 30, 26)
    assert torch.equal(build_pyramid(x, (1.0, 1.0), [3], policy="legacy_avgpool")[0].tensor, F.avg_pool2d(x, 3, 3))
    s = torch.rand(5, 1, 20, 22, 24)
    assert torch.equal(build_pyramid(s, (1.0,) * 3, [4], policy="legacy_avgpool")[0].tensor, F.avg_pool3d(s, 4, 4))


def test_mm_legacy_axis_matrix_shares_central_linear_and_matches_area_on_integer_factors():
    assert np.array_equal(RA._axis_resample_matrix(11, 40, 10 / 39, 0.0), linear_matrix(11, 40, 10 / 39, 0.0))
    for f in (2.0, 4.0):
        assert np.allclose(RA._axis_resample_matrix(64, int(64 / f), f, (f - 1) / 2), area_matrix(64, int(64 / f), f), atol=1e-12)


def test_no_private_voxel_pooling_left_in_the_affine_and_motion_callers():
    for mod in (RA, MB):
        src = inspect.getsource(mod)
        assert not re.search(r"F\.avg_pool[23]d\(|avg_pool[23]d\(", src.replace("``avg_pool3d``", "")), mod.__name__


def test_motion_cap_resolution_delegates_to_the_canonical_cap(monkeypatch):
    import syntx.imaging_utils as iu
    seen = {}
    monkeypatch.setattr(iu, "cap_resolution_for_registration", lambda img, mm: seen.setdefault("args", (img, mm)) and img)
    sentinel = object()
    MB._cap_resolution(sentinel, 4.0)
    assert seen["args"] == (sentinel, 4.0)


def test_landmark_low_res_inputs_are_linear_not_nearest_neighbour(monkeypatch):
    """ants.resample_image defaults to nearest neighbour; the landmark candidate's 2 mm inputs must be linear resamples."""
    import ants
    import syntx.landmarks.orient as orient
    from syntx.pyramid import cap_ants_image
    rng = np.random.default_rng(0)
    a = ants.from_numpy(rng.random((60, 64, 56)).astype(np.float32), spacing=(1.3, 1.3, 1.3))
    seen = {}

    def fake(fi, mi, **kw):
        seen["fi"] = fi
        return {"affine": None, "inliers": []}

    monkeypatch.setattr(orient, "match_sift3d_with_rotation_search", fake)
    RA._extract_landmark_candidate(a, a, np.zeros(3))
    nn = ants.resample_image(a, (2.0, 2.0, 2.0), use_voxels=False)
    assert seen["fi"].shape == nn.shape and tuple(np.round(seen["fi"].spacing, 6)) == (2.0, 2.0, 2.0)
    assert np.array_equal(seen["fi"].numpy(), cap_ants_image(a, 2.0).numpy())
    assert not np.array_equal(seen["fi"].numpy(), nn.numpy())
