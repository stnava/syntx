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
    # Large enough extent (64 * 0.5mm = 32mm) that the voxel floor doesn't interfere --
    # this test is purely about the requested target spacing being honored.
    img = _smooth_phantom(shape=(64, 64, 64), spacing=(0.5, 0.5, 0.5))
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
    # Large enough extent (64 * 0.5mm = 32mm) that the voxel floor doesn't interfere --
    # this test is purely about interpolation-mode correctness.
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, size=(64, 64, 64)).astype(np.float32)
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


def test_tiny_image_falls_back_to_native_resolution_instead_of_collapsing():
    """Regression guard: an 8-voxel, 1mm-spacing image capped at 8mm would otherwise
    resample to ~1 voxel per axis -- degenerate for any gradient-based registration
    metric (confirmed to break torch.gradient in syntx's adaptive motion backend on a
    small PET test fixture). Too small to reach the voxel floor even at native
    resolution, so it is registered unchanged rather than resampled at all."""
    img = _smooth_phantom(shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    capped = cap_resolution_for_registration(img, max_resolution_mm=8.0)
    assert capped is img


def test_voxel_floor_is_a_noop_for_real_clinical_scale_extent():
    """The default floor (16 voxels/axis) must not perturb real data: a typical head's
    extent at this session's validated motion-correction coarse-ranking cap (8mm) is
    ~20-32 voxels/axis -- comfortably above the floor, so the originally requested 8mm
    spacing must come through exactly, not get silently tightened."""
    img = _smooth_phantom(shape=(224, 224, 224), spacing=(1.0, 1.0, 1.0))  # ~224mm extent
    capped = cap_resolution_for_registration(img, max_resolution_mm=8.0)
    assert all(abs(s - 8.0) < 1e-6 for s in capped.spacing)


def test_voxel_floor_clamps_an_intermediate_size_image():
    """An image too small to hit the floor at the requested cap, but large enough that
    native resolution isn't the only option, should land exactly on the floor rather
    than collapsing further or being left at full native resolution."""
    img = _smooth_phantom(shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0))  # 64mm extent
    capped = cap_resolution_for_registration(img, max_resolution_mm=8.0, min_voxels_per_axis=16)
    assert all(sh == 16 for sh in capped.shape[:3])
    assert all(abs(s - 4.0) < 1e-6 for s in capped.spacing)  # 64mm / 16 voxels


def test_min_voxels_per_axis_is_overridable():
    img = _smooth_phantom(shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0))
    capped = cap_resolution_for_registration(img, max_resolution_mm=8.0, min_voxels_per_axis=32)
    assert all(sh == 32 for sh in capped.shape[:3])


def test_exact_target_spacing_is_noop():
    img = _smooth_phantom(spacing=(2.0, 2.0, 2.0))
    capped = cap_resolution_for_registration(img, max_resolution_mm=2.0)
    assert capped is img


def test_clip_cervical_spine_fov():
    from syntx.imaging_utils import clip_cervical_spine_fov
    # Create elongated volume: Z length 40mm
    arr = np.zeros((16, 16, 40), dtype=np.float32)
    # Cranial mass at Z = 25..35 (superior)
    arr[6:10, 6:10, 25:35] = 100.0
    # Cervical mass at Z = 2..8 (inferior)
    arr[6:10, 6:10, 2:8] = 80.0
    img = ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

    clipped = clip_cervical_spine_fov(img, cutoff_mm_below_com=15.0)
    c_arr = clipped.numpy()

    # Superior cranial mass must be retained
    assert c_arr[6:10, 6:10, 25:35].max() == 100.0
    # Inferior cervical mass must be zeroed out
    assert c_arr[6:10, 6:10, 2:8].max() == 0.0

