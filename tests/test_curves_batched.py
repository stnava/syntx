"""Tests for syntx.curves.batched and device/float32 upstream accelerations."""

from __future__ import annotations

from unittest.mock import patch

import ants
import numpy as np
import pytest

from syntx.curves import (
    can_batch_open_match,
    curve_curvature,
    curve_tangents,
    match_curves,
    match_open_curves_batch,
    open_tangents_and_curvature,
    resample_curve,
    resample_open_curves_batch,
    smooth_curve,
    smooth_open_curves_batch,
)
from syntx.transforms import signed_jacobian_determinant


def _make_open_bundle(n_curves: int = 16, dim: int = 3, flip_fraction: float = 0.5, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_curves):
        m = int(rng.integers(24, 52))
        s = np.linspace(0.0, 1.0, m)
        w = np.clip(s + 0.08 * np.sin(2.0 * np.pi * s + rng.uniform(-0.4, 0.4)), 0.0, 1.0)
        coords = [40.0 * np.cos(np.pi * w) + rng.normal(0.0, 1.5), 40.0 * np.sin(np.pi * w) + rng.normal(0.0, 1.5)]
        for k in range(2, dim):
            coords.append(12.0 * np.sin((k + 1) * np.pi * w) + rng.normal(0.0, 1.0))
        pts = np.column_stack(coords) + rng.normal(0.0, 0.15, size=(m, dim))
        if i > 0 and rng.random() < flip_fraction:
            pts = pts[::-1].copy()
        out.append(pts.astype(np.float32 if i % 2 == 0 else np.float64))
    return out


@pytest.mark.parametrize("dim", [2, 3])
def test_match_open_curves_batch_parity_shared_ref(dim: int):
    """Shared reference curve P matches per-curve match_curves to machine precision."""
    bundle = _make_open_bundle(n_curves=14, dim=dim, flip_fraction=0.5, seed=10 + dim)
    ref = bundle[0]
    batch = match_open_curves_batch(ref, bundle, n_points=36, both_orientations=True)

    assert len(batch["matches"]) == len(bundle)
    assert batch["warped_Q"].shape == (len(bundle), 36, dim)
    assert batch["warped_Q"].dtype == np.float32

    for i, q in enumerate(bundle):
        m_ref = match_curves(ref, q, mode="open", both_orientations=True, n_points=36)
        m_got = batch["matches"][i]
        assert m_got.reversed == m_ref.reversed
        assert m_got.dimension == dim
        np.testing.assert_allclose(m_got.t2_unwrapped, m_ref.t2_unwrapped, atol=1e-11)
        np.testing.assert_allclose(m_got.t2, m_ref.t2, atol=1e-11)
        assert abs(m_got.elastic_rms_mm - m_ref.elastic_rms_mm) < 1e-11
        assert abs(m_got.feature_gain - m_ref.feature_gain) < 1e-12
        rel_tol = 1e-8 if i == 0 else 1e-11  # i == 0 is self-match (0 / ~1e-8 baseline energy)
        assert abs(m_got.feature_gain_rel - m_ref.feature_gain_rel) < rel_tol
        np.testing.assert_allclose(batch["warped_Q"][i], m_ref.warp_Q().astype(np.float32), atol=1e-5)


def test_match_open_curves_batch_per_curve_targets_and_options():
    """Per-curve targets P_i, fixed distance_scale_mm, sigma_mm, and sigmas_rel match match_curves."""
    targets = _make_open_bundle(n_curves=10, dim=3, flip_fraction=0.0, seed=71)
    queries = _make_open_bundle(n_curves=10, dim=3, flip_fraction=0.0, seed=72)
    P_arr = resample_open_curves_batch(targets, 32)
    Q_arr = resample_open_curves_batch(queries, 32)

    batch = match_open_curves_batch(
        P_arr,
        Q_arr,
        n_points=32,
        both_orientations=False,
        distance_scale_mm=6.5,
        sigma_mm=2.5,
        build_match_objects=True,
    )
    for i in range(10):
        m_ref = match_curves(
            P_arr[i],
            Q_arr[i],
            mode="open",
            both_orientations=False,
            n_points=32,
            distance_scale_mm=6.5,
            sigma_mm=2.5,
        )
        m_got = batch["matches"][i]
        np.testing.assert_allclose(m_got.t2_unwrapped, m_ref.t2_unwrapped, atol=1e-11)
        assert abs(m_got.elastic_rms_mm - m_ref.elastic_rms_mm) < 1e-11

    batch_rel = match_open_curves_batch(
        P_arr[0],
        Q_arr,
        n_points=28,
        sigmas_rel=(0.02, 0.05, 0.1),
        smooth_sigma_grid=(None, 0.0, 2.0),
    )
    for i in range(10):
        m_ref = match_curves(
            P_arr[0],
            Q_arr[i],
            mode="open",
            n_points=28,
            sigmas_rel=(0.02, 0.05, 0.1),
            smooth_sigma_grid=(None, 0.0, 2.0),
        )
        np.testing.assert_allclose(batch_rel["matches"][i].t2_unwrapped, m_ref.t2_unwrapped, atol=1e-11)


def test_batched_geometry_primitives_and_edge_cases():
    """Verify resample_open_curves_batch, smooth_open_curves_batch, open_tangents_and_curvature, and empty/degenerate cases."""
    curves = _make_open_bundle(n_curves=6, dim=3, flip_fraction=0.0, seed=91)
    res_b = resample_open_curves_batch(curves, 40)
    for i, c in enumerate(curves):
        np.testing.assert_allclose(res_b[i], resample_curve(c, 40, mode="open"), atol=1e-12)

    sigmas = np.array([0.0, 1.25, 3.5], dtype=np.float64)
    sm_b = smooth_open_curves_batch(res_b[:3, None, :, :], np.broadcast_to(sigmas, (3, 3)))
    T_b, k_b = open_tangents_and_curvature(sm_b)
    for i in range(3):
        for j, sig in enumerate(sigmas):
            sm_ref = smooth_curve(res_b[i], float(sig), mode="open")
            np.testing.assert_allclose(sm_b[i, j], sm_ref, atol=1e-12)
            np.testing.assert_allclose(T_b[i, j], curve_tangents(sm_ref, mode="open"), atol=1e-12)
            np.testing.assert_allclose(k_b[i, j], curve_curvature(sm_ref, mode="open"), atol=1e-12)

    # Empty batch and degenerate zero-length curve
    empty = match_open_curves_batch(res_b[0], [], n_points=20)
    assert empty["warped_Q"].shape == (0, 20, 3)
    assert empty["matches"] == []

    deg = np.repeat(np.array([[1.0, 2.0, 3.0]]), 8, axis=0)
    deg_res = match_open_curves_batch(res_b[0], [deg], n_points=16)
    assert np.all(np.isfinite(deg_res["warped_Q"]))

    assert can_batch_open_match({"mode": "open", "distance_scale_mm": 5.0})
    assert not can_batch_open_match({"mode": "closed"})
    assert not can_batch_open_match({"rigid": True})
    assert not can_batch_open_match({"bidirectional": True})
    assert not can_batch_open_match({"unknown_kwarg": 1})


def test_motion_correct_grouped_forwards_device_to_low_motion_reference():
    """Verify motion_correct_grouped forwards explicit device= to build_low_motion_reference."""
    import syntx.motion_reference as mr

    rng = np.random.default_rng(123)
    arr4d = rng.uniform(10.0, 100.0, size=(10, 10, 10, 6)).astype(np.float32)
    img4d = ants.from_numpy(arr4d, spacing=(2.5, 2.5, 2.5, 1.0))
    groups = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    seen_devices: list[str] = []
    orig_fn = mr.build_low_motion_reference

    def _spy(images, *args, **kwargs):
        seen_devices.append(str(kwargs.get("device")))
        return orig_fn(images, *args, **kwargs)

    with patch.object(mr, "build_low_motion_reference", side_effect=_spy):
        res = mr.motion_correct_grouped(img4d, groups, preset="dwi", device="cpu", n_low_motion=2)
    assert res["motion_corrected"].shape == (10, 10, 10, 6)
    assert seen_devices == ["cpu", "cpu"]


def test_signed_jacobian_determinant_float32_sarrus_matches_float64_reference():
    """Verify float32 Sarrus signed_jacobian_determinant matches exact float64 np.linalg.det."""
    rng = np.random.default_rng(55)
    shape = (10, 11, 12)
    theta = np.deg2rad(14.0)
    direction = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    spacing = (1.5, 1.8, 2.2)
    ref = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, direction=direction)
    u = rng.normal(0.0, 0.15, size=shape + (3,)).astype(np.float32)
    field = ants.from_numpy(u, spacing=spacing, direction=direction, has_components=True)

    got = signed_jacobian_determinant(field, ref).numpy()
    u64 = u.astype(np.float64)
    ds_inv = np.linalg.inv(direction @ np.diag(spacing))
    du64 = np.stack([np.stack(np.gradient(u64[..., i]), axis=-1) for i in range(3)], axis=-2)
    expected = np.linalg.det(np.eye(3) + du64 @ ds_inv).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
