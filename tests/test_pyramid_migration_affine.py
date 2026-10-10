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

from syntx.pyramid import build_pyramid

RA = importlib.import_module("syntx.robust_affine")
MB = importlib.import_module("syntx.motion_batched")


@pytest.mark.parametrize("shape", [(16, 20, 24), (32, 36, 40)])
@pytest.mark.parametrize("level", [1, 2, 4])
def test_mm_iso_on_an_isotropic_image_with_a_divisible_shape_is_the_voxel_average(shape, level):
    """The isotropic callers use ``mm_iso`` (the ``legacy_avgpool`` preset was removed): an integer factor is exactly the box average."""
    x = torch.rand(1, 1, *shape)
    lv = build_pyramid(x, (1.0, 1.0, 1.0), [level], policy="mm_iso")[0]
    ref = F.avg_pool3d(x, kernel_size=level, stride=level) if level > 1 else x
    assert torch.allclose(lv.tensor, ref, atol=1e-6)
    assert list(lv.f) == [float(level)] * 3 and list(lv.off) == [(level - 1) / 2.0] * 3


def test_mm_iso_covers_the_whole_field_of_view_when_the_shape_is_not_divisible():
    """voxel pooling dropped the trailing remainder (17 -> 8 voxels covering 16); the area average covers all 17 (f = 17 / 8)."""
    x = torch.rand(1, 1, 17, 19, 23)
    lv = build_pyramid(x, (1.0,) * 3, [2], policy="mm_iso")[0]
    assert lv.shape == (8, 10, 12) and abs(lv.f[2] - 17 / 8) < 1e-12 and abs(lv.off[2] - (17 / 8 - 1) / 2) < 1e-12
    assert float(lv.tensor.mean()) == pytest.approx(float(x.mean()), abs=2e-2)                      # nothing is dropped: the mean is unchanged


def test_mm_iso_2d_and_batched_moving_stack():
    x = torch.rand(1, 1, 30, 27)
    assert torch.allclose(build_pyramid(x, (1.0, 1.0), [3], policy="mm_iso")[0].tensor, F.avg_pool2d(x, 3, 3), atol=1e-6)
    s = torch.rand(5, 1, 20, 24, 28)
    assert torch.allclose(build_pyramid(s, (1.0,) * 3, [4], policy="mm_iso")[0].tensor, F.avg_pool3d(s, 4, 4), atol=1e-6)


def test_deleted_private_pyramid_helpers_stay_deleted():
    for name in ("_mm_pyramid_level", "_axis_resample_matrix", "_pool_level", "_gaussian_kernel_2d_sep"):
        assert not hasattr(RA, name), name


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


def test_no_private_pyramid_or_downsample_code_in_the_affine_and_motion_callers():
    """Pass 3c: the only resampling left in the two modules is ``ants.resample_image_to_target`` (moving onto the fixed / a fine
    template grid, a registration-grid operation allow-listed here, not a pyramid level); no ``F.interpolate``, no raw
    ``ants.resample_image(`` cap, no private Gaussian, no private 2-D conv kernel."""
    for mod in (RA, MB):
        src = inspect.getsource(mod)
        code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
        for banned in ("F.interpolate(", "ants.resample_image(", "_separable_gaussian3d", "F.conv2d(", "F.conv3d(", "area_matrix(", "_mm_pyramid_level"):
            assert banned not in code, (mod.__name__, banned)


def _blob_pair(dim, shift_mm=4.0, spacing=1.5):
    import ants
    n = 40
    g = np.mgrid[tuple(slice(0, n) for _ in range(dim))].astype(np.float32)
    ax = [12.0, 9.0, 7.0][:dim]
    def blob(c):
        return (0.8 * (sum(((g[i] - c[i]) / ax[i]) ** 2 for i in range(dim)) <= 1.0)).astype(np.float32)
    c0 = [n / 2.0] * dim; c1 = list(c0); c1[0] -= shift_mm / spacing
    return (ants.from_numpy(blob(c0), spacing=(spacing,) * dim), ants.from_numpy(blob(c1), spacing=(spacing,) * dim))


@pytest.mark.parametrize("dim", [2, 3])
def test_smooth_sigma_per_level_goes_through_the_central_smoother_and_still_registers(dim):
    """``smooth_sigma_per_level`` > 0 (replicate-padded central Gaussian, per-axis sigma; the zero-padded blob kernel and the 2-D conv
    kernel are gone) runs in 2-D and 3-D and recovers a known shift about as well as sigma 0."""
    import ants
    from syntx.robust_affine import robust_affine
    fi, mi = _blob_pair(dim)
    errs = {}
    for sig in (0.0, 1.0):
        res = robust_affine(fi, mi, preset="fast", smooth_sigma_per_level=sig, multi_start=False, verbose=False)
        assert res["status"] == "SUCCESS"
        tx = ants.read_transform(res["fwdtransforms"][0])
        c = (20.0 * 1.5,) * dim                                                       # the fixed blob centre maps to the moving blob centre
        want = np.array(c) - np.array([4.0] + [0.0] * (dim - 1))
        errs[sig] = float(np.linalg.norm(np.array(tx.apply_to_point(c)) - want))
    assert errs[0.0] < 1.5 and errs[1.0] < 1.5, errs
