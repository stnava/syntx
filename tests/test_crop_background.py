"""Tests for lossless physical-space background cropping (`syntx.crop_background`)."""

from __future__ import annotations

import math
import ants
import numpy as np
import pytest

import syntx


def _oblique_3d_direction() -> np.ndarray:
    """Construct an exact orthonormal 3-D rotation matrix (oblique acquisition frame)."""
    th_x, th_y, th_z = 0.23, -0.17, 0.31
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(th_x), -math.sin(th_x)], [0.0, math.sin(th_x), math.cos(th_x)]]
    )
    ry = np.array(
        [[math.cos(th_y), 0.0, math.sin(th_y)], [0.0, 1.0, 0.0], [-math.sin(th_y), 0.0, math.cos(th_y)]]
    )
    rz = np.array(
        [[math.cos(th_z), -math.sin(th_z), 0.0], [math.sin(th_z), math.cos(th_z), 0.0], [0.0, 0.0, 1.0]]
    )
    return (rz @ ry @ rx).astype(np.float64)


def test_crop_background_per_axis_tightness_and_physical_lps_identity():
    rng = np.random.default_rng(20261125)
    arr = np.zeros((96, 110, 128), dtype=np.float32)
    # Place foreground box at [30:55, 10:60, 42:82] (notice Y-low only has 10 empty voxels!)
    fg_block = rng.uniform(0.2, 5.0, size=(25, 50, 40)).astype(np.float32)
    arr[30:55, 10:60, 42:82] = fg_block

    D = _oblique_3d_direction()
    origin = (84.5, -112.25, 45.75)
    spacing = (0.9, 1.25, 1.6)
    img = ants.from_numpy(arr, origin=origin, spacing=spacing, direction=D)

    # Axis-specific tightness: 8 along X, 24 along Y (requires lossless zero-pad on Y-low!), 16 along Z
    cropped, info = syntx.crop_background(
        img,
        margin=(8, 24, 16),
        divisor=8,
        return_info=True,
    )

    assert tuple(cropped.shape) == info["new_shape"]
    assert all(s % 8 == 0 for s in cropped.shape)
    assert cropped.spacing == spacing
    np.testing.assert_allclose(cropped.direction, D, atol=1e-12)

    # Verify empty margins on all 6 faces meet or exceed (8, 24, 16)
    c_coords = np.argwhere(cropped.numpy() > 0)
    c_lo = c_coords.min(axis=0)
    c_hi = np.asarray(cropped.shape) - 1 - c_coords.max(axis=0)
    assert c_lo[0] >= 8 and c_hi[0] >= 8
    assert c_lo[1] >= 24 and c_hi[1] >= 24
    assert c_lo[2] >= 16 and c_hi[2] >= 16

    # Bitwise value & exact physical LPS coordinate identity across foreground voxels
    idx_start = np.asarray(info["idx_start"], dtype=np.int64)
    orig_pts = np.argwhere(arr > 0)
    sample = orig_pts[rng.choice(len(orig_pts), size=60, replace=False)]
    for pt_old in sample:
        pt_new = pt_old - idx_start
        assert float(arr[tuple(pt_old)]) == float(cropped.numpy()[tuple(pt_new)])
        mm_old = np.asarray(ants.transform_index_to_physical_point(img, [int(v) for v in pt_old]))
        mm_new = np.asarray(ants.transform_index_to_physical_point(cropped, [int(v) for v in pt_new]))
        np.testing.assert_allclose(mm_new, mm_old, atol=1e-11)

    # Verify ANTs physical-space nearest-neighbor resampling agrees bitwise
    resamp = ants.resample_image_to_target(img, cropped, interp_type="nearestNeighbor")
    np.testing.assert_array_equal(resamp.numpy(), cropped.numpy())


def test_crop_background_per_face_pairs_and_mm_units():
    arr = np.zeros((64, 64, 64), dtype=np.float32)
    arr[20:30, 20:30, 20:30] = 1.0
    img = ants.from_numpy(arr, origin=(-32.0, -32.0, -32.0), spacing=(2.0, 0.5, 1.0))

    # Asymmetric face margins in mm: X=(4mm, 8mm)->(2,4)vox, Y=(3mm, 5mm)->(6,10)vox, Z=(5mm, 5mm)->(5,5)vox
    cropped, info = syntx.crop_background(
        img,
        margin=((4.0, 8.0), (3.0, 5.0), (5.0, 5.0)),
        margin_unit="mm",
        divisor=1,
        return_info=True,
    )
    assert info["pad_lo"] == (2, 6, 5)
    assert info["pad_hi"] == (4, 10, 5)
    assert cropped.shape == (10 + 6, 10 + 16, 10 + 10)


def test_crop_background_allow_pad_false_clamps_to_bounds():
    arr = np.zeros((40, 40), dtype=np.float32)
    arr[3:15, 10:20] = 2.0
    img = ants.from_numpy(arr, origin=(10.0, 20.0), spacing=(1.5, 1.5))

    cropped, info = syntx.crop_background(img, margin=12, allow_pad=False, return_info=True)
    # X-low only had 3 empty pixels, so pad_lo[0] clamps to 3 instead of padding outside [0, 40)
    assert info["idx_start"][0] == 0
    assert info["pad_lo"][0] == 3
    assert cropped.shape[0] <= 40 and cropped.shape[1] <= 40


def test_crop_background_companions_multicomponent_and_4d():
    arr3 = np.zeros((48, 52, 56), dtype=np.float32)
    arr3[16:28, 18:30, 20:36] = 3.0
    img = ants.from_numpy(arr3, origin=(10.0, 20.0, 30.0), spacing=(1.0, 1.0, 1.0))

    lbl = img.new_image_like((arr3 > 0).astype(np.float32) * 7.0)
    vec_arr = np.zeros((48, 52, 56, 3), dtype=np.float32)
    vec_arr[16:28, 18:30, 20:36, :] = [1.5, -2.0, 0.5]
    vec_img = ants.from_numpy(
        vec_arr, origin=img.origin, spacing=img.spacing, direction=img.direction, has_components=True
    )

    c_img, c_comps, info = syntx.crop_background(
        img, margin=6, divisor=4, companions=[lbl, vec_img], return_info=True
    )
    c_lbl, c_vec = c_comps
    assert c_lbl.shape == c_img.shape
    assert c_vec.has_components is True
    assert c_vec.shape == c_img.shape
    assert int((c_lbl.numpy() == 7.0).sum()) == int((lbl.numpy() == 7.0).sum())
    np.testing.assert_allclose(c_vec.origin, c_img.origin, atol=1e-12)

    # 4-D time series
    arr4 = np.zeros((48, 52, 56, 5), dtype=np.float32)
    for t in range(5):
        arr4[16:28, 18:30, 20:36, t] = float(t + 1)
    img4 = ants.from_numpy(
        arr4,
        origin=(10.0, 20.0, 30.0, 0.0),
        spacing=(1.0, 1.0, 1.0, 2.5),
        direction=np.eye(4),
    )
    c4 = syntx.crop_background(img4, margin=6, divisor=4)
    assert c4.shape[:3] == info["new_shape"]
    assert c4.shape[3] == 5
    assert c4.origin[3] == 0.0 and c4.spacing[3] == 2.5


def test_crop_background_validation_errors():
    empty = ants.from_numpy(np.zeros((20, 20, 20), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        syntx.crop_background(empty, margin=4)

    non_empty = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match=">= 0"):
        syntx.crop_background(non_empty, margin=-1)
    with pytest.raises(ValueError, match="spatial dimension"):
        syntx.crop_background(non_empty, margin=(4, 4))
    with pytest.raises(ValueError, match="margin_unit"):
        syntx.crop_background(non_empty, margin=4, margin_unit="inches")
