"""Comprehensive unit tests for syntx.curves.geometry across dimensions N=2, 3, 4, 5."""

import numpy as np
import pytest

from syntx.curves.geometry import (
    as_curve_array,
    count_curvature_extrema,
    curve_arc_fraction,
    curve_curvature,
    curve_length,
    curve_segment_lengths,
    curve_tangents,
    curvature_scale_space,
    resample_curve,
    select_matching_scale,
    smooth_curve,
    tangent_scale_space,
)


def sample_circle_2d(n=120, radius=15.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def sample_helix_3d(n=100, turns=2.0, radius=10.0, height=30.0):
    t = np.linspace(0, 2 * np.pi * turns, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, height, n)
    return np.column_stack([x, y, z])


def sample_hypercurve_nd(n=100, d=4, length=50.0):
    t = np.linspace(0, 1, n)[:, None]
    coords = []
    for k in range(d):
        freq = float(k + 1)
        coords.append(np.sin(2 * np.pi * freq * t) * (10.0 / freq) + (length / d) * t)
    return np.hstack(coords)


def sample_curve_nd(n=100, d=3):
    t = np.linspace(0, 1, n)[:, None]
    coords = []
    for k in range(d):
        freq = float(k + 1)
        coords.append(np.sin(np.pi * freq * t) * (15.0 / freq) + 50.0 * t * (1.0 if k == 0 else 0.2))
    return np.hstack(coords)


# --------------------------------------------------------------------------- #
# Array validation & length tests
# --------------------------------------------------------------------------- #

def test_as_curve_array_dimensions():
    # 2D, 3D, 4D, 5D arrays
    for d in (2, 3, 4, 5):
        pts = np.zeros((10, d))
        arr = as_curve_array(pts, mode="open")
        assert arr.shape == (10, d)

    # Closed curve closure stripping
    c_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
    c_arr = as_curve_array(c_pts, mode="closed")
    assert c_arr.shape == (3, 2)

    # Invalid input
    with pytest.raises(ValueError):
        as_curve_array(np.array([1, 2, 3]), mode="open")
    with pytest.raises(ValueError):
        as_curve_array(np.zeros((1, 3)), mode="open")


def test_curve_lengths_nd():
    # 2D Circle of radius R
    r = 12.0
    circ = sample_circle_2d(200, radius=r)
    circ_len = curve_length(circ, mode="closed")
    assert np.isclose(circ_len, 2 * np.pi * r, rtol=1e-2)

    # 3D Helix of known length
    turns, radius, height = 2.0, 10.0, 30.0
    helix = sample_helix_3d(200, turns=turns, radius=radius, height=height)
    expected_len = np.sqrt((2 * np.pi * turns * radius) ** 2 + height ** 2)
    assert np.isclose(curve_length(helix, mode="open"), expected_len, rtol=1e-2)

    # 4D Curve length
    curve4d = sample_hypercurve_nd(100, d=4)
    segs4d = curve_segment_lengths(curve4d, mode="open")
    assert len(segs4d) == 99
    assert np.all(segs4d > 0)
    assert np.isclose(curve_length(curve4d, mode="open"), np.sum(segs4d))


def test_curve_arc_fraction_nd():
    for d in (2, 3, 4, 5):
        curve = sample_hypercurve_nd(80, d=d)
        af = curve_arc_fraction(curve, mode="open")
        assert len(af) == 80
        assert np.isclose(af[0], 0.0)
        assert np.isclose(af[-1], 1.0)
        assert np.all(np.diff(af) > 0)


# --------------------------------------------------------------------------- #
# Resampling & smoothing in N-D
# --------------------------------------------------------------------------- #

def test_resample_curve_preserves_endpoints_nd():
    for d in (2, 3, 4, 5):
        curve = sample_curve_nd(100, d=d)
        res = resample_curve(curve, 50, mode="open")
        assert res.shape == (50, d)
        assert np.allclose(res[0], curve[0])
        assert np.allclose(res[-1], curve[-1])
        segs = curve_segment_lengths(res, mode="open")
        assert np.ptp(segs) / np.mean(segs) < 0.08
        assert np.isclose(curve_length(res, mode="open"), curve_length(curve, mode="open"), rtol=1e-2)


def test_smooth_curve_preserves_endpoints_nd():
    rng = np.random.default_rng(42)
    for d in (2, 3, 4, 5):
        base = sample_hypercurve_nd(60, d=d)
        noisy = base + rng.normal(scale=0.2, size=base.shape)
        noisy[0] = base[0]
        noisy[-1] = base[-1]

        sm = smooth_curve(noisy, sigma_pts=2.0, mode="open")
        assert sm.shape == (60, d)
        assert np.allclose(sm[0], noisy[0])
        assert np.allclose(sm[-1], noisy[-1])
        # High-frequency variance reduction
        assert np.var(np.diff(sm, axis=0)) < np.var(np.diff(noisy, axis=0))


# --------------------------------------------------------------------------- #
# Tangents & curvature in N-D
# --------------------------------------------------------------------------- #

def test_tangents_unit_norm_nd():
    for d in (2, 3, 4, 5):
        curve = sample_hypercurve_nd(70, d=d)
        T = curve_tangents(curve, mode="open")
        assert T.shape == (70, d)
        norms = np.linalg.norm(T, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_curvature_analytical_2d_and_3d():
    # 2D Circle: kappa = 1/R
    r = 10.0
    circ = sample_circle_2d(250, radius=r)
    kappa_2d = curve_curvature(circ, mode="closed")
    assert np.allclose(kappa_2d, 1.0 / r, rtol=2e-2)

    # 3D Helix: kappa = r / (r^2 + c^2), where c = height / (2 * pi * turns)
    turns, radius, height = 3.0, 8.0, 40.0
    c = height / (2.0 * np.pi * turns)
    expected_kappa = radius / (radius ** 2 + c ** 2)
    helix = sample_helix_3d(300, turns=turns, radius=radius, height=height)
    kappa_3d = curve_curvature(helix, mode="open")
    # Verify interior curvature
    assert np.allclose(kappa_3d[10:-10], expected_kappa, rtol=3e-2)

    # 4D Curve: intrinsic curvature is positive
    curve4d = sample_hypercurve_nd(100, d=4)
    kappa_4d = curve_curvature(curve4d, mode="open")
    assert kappa_4d.shape == (100,)
    assert np.all(kappa_4d >= 0)


def test_scale_space_conservation_nd():
    for d in (2, 3, 4):
        curve = sample_hypercurve_nd(80, d=d)
        css = curvature_scale_space(curve, sigmas=[1.0, 2.0, 4.0], mode="open")
        assert css.shape == (3, 80)
        k0_sum = np.sum(curve_curvature(curve, mode="open"))
        for row in css:
            assert np.isclose(np.sum(row), k0_sum, rtol=1e-5)


def test_extrema_and_scale_selection_nd():
    t = np.linspace(0, 1, 100)
    # 2D wavy curve with 3 crests
    wavy = np.column_stack([t * 50.0, 4.0 * np.sin(2 * np.pi * 3 * t)])
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
    best_sig, F1, F2 = select_matching_scale(wavy, wavy, sigmas, mode="open")
    assert len(F1) == len(sigmas)
    assert np.array_equal(F1, F2)
    assert best_sig == sigmas[0]
