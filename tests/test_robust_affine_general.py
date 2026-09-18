"""
tests/test_robust_affine_general.py
====================================

Unit tests for general-purpose robust affine registration and tournament pipeline.
Verifies:
1. compute_center_of_mass handles negative HU CT scans without moment corruption.
2. robust_affine tournament mode generates valid ANTs transforms and runs end-to-end.
3. 2D fallback gracefully handles 2D inputs.
4. Output transform contracts obey Syntx invariants (single interpolation, valid files).
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import ants

from syntx.robust_affine import (
    compute_center_of_mass,
    robust_center_of_mass,
    robust_affine,
    _compute_salient_gradient_correlation,
)


def test_robust_center_of_mass_bipolar():
    """Verify robust_center_of_mass returns accurate distinct coordinates for positive and negative halves."""
    shape = (50, 50, 50)
    arr = np.zeros(shape, dtype=np.float32)
    # Positive region at (10, 10, 10)
    arr[8:13, 8:13, 8:13] = 50.0
    # Negative region at (38, 38, 38)
    arr[36:41, 36:41, 36:41] = -100.0

    img = ants.from_numpy(arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    com_pos, com_neg = robust_center_of_mass(img, weighted=True)

    assert com_pos is not None, "Positive CoM should not be None"
    assert com_neg is not None, "Negative CoM should not be None"

    expected_pos = np.array([10.0, 10.0, 10.0])
    expected_neg = np.array([38.0, 38.0, 38.0])

    assert np.allclose(com_pos, expected_pos, atol=0.5), f"Expected pos {expected_pos}, got {com_pos}"
    assert np.allclose(com_neg, expected_neg, atol=0.5), f"Expected neg {expected_neg}, got {com_neg}"


def test_robust_center_of_mass_positive_only():
    """Verify robust_center_of_mass returns None for negative half when no negative data exists."""
    arr = np.zeros((30, 30, 30), dtype=np.float32)
    arr[12:18, 12:18, 12:18] = 25.0
    img = ants.from_numpy(arr)

    com_pos, com_neg = robust_center_of_mass(img)
    assert com_pos is not None
    assert com_neg is None, f"Expected com_neg to be None, got {com_neg}"


def test_robust_center_of_mass_negative_only():
    """Verify robust_center_of_mass returns None for positive half when data is strictly negative."""
    arr = np.zeros((30, 30, 30), dtype=np.float32)
    arr[12:18, 12:18, 12:18] = -50.0
    img = ants.from_numpy(arr)

    com_pos, com_neg = robust_center_of_mass(img)
    assert com_pos is None, f"Expected com_pos to be None, got {com_pos}"
    assert com_neg is not None


def test_robust_center_of_mass_dict_and_return_both():
    """Verify return_dict=True and compute_center_of_mass(return_both=True) API contracts."""
    arr = np.zeros((30, 30), dtype=np.float32)
    arr[5:10, 5:10] = 10.0
    arr[20:25, 20:25] = -10.0
    img = ants.from_numpy(arr)

    d = robust_center_of_mass(img, return_dict=True)
    assert isinstance(d, dict)
    assert "positive" in d and "negative" in d
    assert d["positive"] is not None and d["negative"] is not None

    p, n = compute_center_of_mass(img, return_both=True)
    assert np.allclose(p, d["positive"])
    assert np.allclose(n, d["negative"])



def test_compute_center_of_mass_ct_negative_hu():
    """Verify compute_center_of_mass handles CT air background without moment corruption."""
    shape = (64, 64, 64)
    # Simulate CT with air background (-1000 HU)
    arr = np.full(shape, -1000.0, dtype=np.float32)
    # Insert soft-tissue organ (+40 HU) centered at voxel (45, 45, 45)
    arr[40:50, 40:50, 40:50] = 40.0
    img = ants.from_numpy(arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

    # Center of mass should be near (44.5, 44.5, 44.5)
    com = compute_center_of_mass(img, weighted=True)
    expected = np.array([44.5, 44.5, 44.5])
    err = np.linalg.norm(com - expected)
    assert err < 1.5, f"CT CoM error {err}mm exceeds tolerance; got {com}"


def test_compute_salient_gradient_correlation():
    """Verify salient gradient correlation is bounded in [-1, 1] and detects identical images."""
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    arr[10:22, 10:22, 10:22] = 1.0
    img1 = ants.from_numpy(arr)
    img2 = ants.from_numpy(arr)

    c = _compute_salient_gradient_correlation(img1, img2)
    assert abs(c - 1.0) < 1e-4, f"Salient correlation for identical images should be 1.0; got {c}"


def test_robust_affine_tournament_synthetic_3d():
    """Verify robust_affine with tournament=True executes end-to-end on synthetic 3D pair."""
    # Create fixed image with a distinct off-center ellipsoid
    grid = np.zeros((48, 48, 48), dtype=np.float32)
    for z in range(48):
        for y in range(48):
            for x in range(48):
                if ((x - 24)**2 / 12**2 + (y - 24)**2 / 10**2 + (z - 24)**2 / 8**2) <= 1.0:
                    grid[x, y, z] = 0.8
    fi = ants.from_numpy(grid, spacing=(1.5, 1.5, 1.5), origin=(0.0, 0.0, 0.0))

    # Moving image shifted by 6 mm in physical space
    grid_m = np.zeros((48, 48, 48), dtype=np.float32)
    for z in range(48):
        for y in range(48):
            for x in range(48):
                if ((x - 20)**2 / 12**2 + (y - 24)**2 / 10**2 + (z - 24)**2 / 8**2) <= 1.0:
                    grid_m[x, y, z] = 0.8
    mi = ants.from_numpy(grid_m, spacing=(1.5, 1.5, 1.5), origin=(0.0, 0.0, 0.0))

    res = robust_affine(fi, mi, tournament=True, preset='fast', verbose=False)

    assert "fwdtransforms" in res
    assert len(res["fwdtransforms"]) == 1
    assert os.path.exists(res["fwdtransforms"][0])
    assert "winner" in res
    assert "candidates" in res
    assert res["time"] > 0.0
    assert res["status"] == "SUCCESS"

    # Warped image should have higher correlation with fixed than moving
    f_arr = fi.numpy()
    m_arr = mi.numpy()
    w_arr = res["warpedmovout"].numpy()
    cc_init = np.corrcoef(f_arr.ravel(), m_arr.ravel())[0, 1]
    cc_reg = np.corrcoef(f_arr.ravel(), w_arr.ravel())[0, 1]
    assert cc_reg > cc_init, f"Registration failed to improve correlation: {cc_init} -> {cc_reg}"


def test_robust_affine_tournament_2d_graceful():
    """Verify tournament mode gracefully handles 2D inputs by delegating to 2D solver."""
    grid = np.zeros((32, 32), dtype=np.float32)
    grid[10:22, 10:22] = 1.0
    fi = ants.from_numpy(grid)
    mi = ants.from_numpy(grid)

    res = robust_affine(fi, mi, tournament=True, preset='fast')
    assert "fwdtransforms" in res
    assert len(res["fwdtransforms"]) == 1
    assert os.path.exists(res["fwdtransforms"][0])
