"""
tests/test_imaging_utils.py -- syntx.imaging_utils.cap_resolution_for_registration

The canonical home for a resolution-capping trick found duplicated across the ecosystem
(2026-10-02 synergy review) -- including a real bug in syntx's own prior private copy
(nearest-neighbor instead of linear interpolation). These tests guard both the no-op
(already-coarse) case and the actual resampling + interpolation-type correctness.
"""

import numpy as np
import pytest
import ants

from syntx.imaging_utils import cap_resolution_for_registration


def _smooth_phantom(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0)):
    arr = np.zeros(shape, dtype=np.float32)
    c = [s // 2 for s in shape]
    gx, gy, gz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    arr = np.exp(-(((gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - c[2]) ** 2)) / (2.0 * 4.0 ** 2)).astype(np.float32)
    return ants.from_numpy(arr, spacing=spacing)


def test_does_not_upsample_already_coarse_image():
    img = _smooth_phantom(spacing=(3.0, 3.0, 3.0))
    capped = cap_resolution_for_registration(img, max_resolution_mm=1.0)
    assert capped is img


def test_downsamples_fine_image_to_target_spacing():
    img = _smooth_phantom(spacing=(0.5, 0.5, 0.5))
    capped = cap_resolution_for_registration(img, max_resolution_mm=2.0)
    assert capped is not img
    assert all(abs(s - 2.0) < 1e-6 for s in capped.spacing)


def test_uses_linear_not_nearest_neighbor_interpolation():
    """Regression guard for the real bug found this session: syntx's own prior private
    copy of this function (in motion_batched.py) passed interp_type=1 (nearest-neighbor)
    instead of the correct 0 (linear) for a continuous-intensity image feeding an
    intensity-based registration metric. Directly compares cap_resolution_for_registration's
    output against both interpolation modes computed independently -- it must match linear
    exactly and must NOT match nearest-neighbor (confirming the two modes actually differ
    for this phantom, so the match-to-linear check is meaningful, not vacuous)."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, size=(31, 31, 31)).astype(np.float32)
    img = ants.from_numpy(arr, spacing=(0.5, 0.5, 0.5))
    capped = cap_resolution_for_registration(img, max_resolution_mm=1.7)

    linear = ants.resample_image(img, [1.7, 1.7, 1.7], interp_type=0)
    nearest = ants.resample_image(img, [1.7, 1.7, 1.7], interp_type=1)
    assert not np.allclose(linear.numpy(), nearest.numpy()), (
        "linear and nearest-neighbor resampling gave identical output for this phantom -- "
        "test phantom isn't discriminating, strengthen it"
    )
    assert np.allclose(capped.numpy(), linear.numpy())
    assert not np.allclose(capped.numpy(), nearest.numpy())


def test_exact_target_spacing_is_noop():
    img = _smooth_phantom(spacing=(2.0, 2.0, 2.0))
    capped = cap_resolution_for_registration(img, max_resolution_mm=2.0)
    assert capped is img
