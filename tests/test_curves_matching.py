"""Deep tests and counterfactual evaluations for syntx.curves.matching across N dimensions."""

import numpy as np
import pytest

from syntx.curves.geometry import curve_curvature, curve_length, resample_curve
from syntx.curves.matching import (
    CurveMatch,
    _clamp_sod_rotation,
    dp_match_grid,
    enforce_bounded_increments,
    enforce_monotone,
    kabsch_rigid,
    match_curves,
)


def sample_curve_nd(n=60, d=3, seed=42):
    """Synthetic S-shaped curve in R^d."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)[:, None]
    coords = []
    for k in range(d):
        freq = float(k + 1)
        coords.append(np.sin(np.pi * freq * t) * (15.0 / freq) + 50.0 * t * (1.0 if k == 0 else 0.2))
    return np.hstack(coords)


def sample_closed_loop_nd(n=80, d=3):
    """Synthetic closed loop in R^d."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
    coords = []
    for k in range(d):
        freq = float(k + 1)
        r = 15.0 / freq
        coords.append(r * np.cos(freq * t) if k % 2 == 0 else r * np.sin(freq * t))
    return np.hstack(coords)


# --------------------------------------------------------------------------- #
# 1. Procrustes in SO(d) & Geodesic Clamping Counterfactual
# --------------------------------------------------------------------------- #

def test_procrustes_sod_recovery_and_clamp_counterfactual():
    rng = np.random.default_rng(77)

    # Test exact recovery in 2D, 3D, 4D
    for d in (2, 3, 4):
        P = rng.normal(size=(50, d))
        # Generate random proper rotation in SO(d) via QR
        H = rng.normal(size=(d, d))
        Q_rot, _ = np.linalg.qr(H)
        if np.linalg.det(Q_rot) < 0:
            Q_rot[:, 0] *= -1.0
        t_true = rng.normal(size=d) * 5.0

        Q = P @ Q_rot.T + t_true
        R_hat, t_hat, rms = kabsch_rigid(P, Q)
        assert np.allclose(R_hat, Q_rot, atol=1e-5)
        assert np.allclose(t_hat, t_true, atol=1e-5)
        assert rms < 1e-6

    # Counterfactual: SO(3) rotation clamping along geodesic
    th_true = np.deg2rad(40.0)
    R40 = np.array([
        [np.cos(th_true), 0.0, np.sin(th_true)],
        [0.0, 1.0, 0.0],
        [-np.sin(th_true), 0.0, np.cos(th_true)],
    ])
    P3 = rng.normal(size=(50, 3))
    Q3 = P3 @ R40.T

    # Clamp to 15 degrees: recovered rotation angle must be strictly 15.0 degrees
    R_clamped, _, _ = kabsch_rigid(P3, Q3, max_rotation_deg=15.0)
    recovered_angle = np.rad2deg(np.arccos(np.clip(0.5 * (np.trace(R_clamped) - 1.0), -1.0, 1.0)))
    assert np.isclose(recovered_angle, 15.0, atol=1e-3)

    # Counterfactual: Clamp to 0 degrees returns identity matrix
    R_zero, t_zero, _ = kabsch_rigid(P3, Q3, max_rotation_deg=0.0)
    assert np.allclose(R_zero, np.eye(3), atol=1e-6)
    assert np.allclose(t_zero, Q3.mean(axis=0) - P3.mean(axis=0), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Identical Curve Matching across Dimensions
# --------------------------------------------------------------------------- #

def test_open_curve_identical_nd():
    for d in (2, 3, 4):
        curve = sample_curve_nd(60, d=d)
        m = match_curves(curve, curve, mode="open", n_points=50, rigid=False)
        assert isinstance(m, CurveMatch)
        assert m.reversed is False
        assert m.dimension == d
        assert m.elastic_rms_mm < 0.05
        assert np.allclose(m.t2, m.t1, atol=0.03)
        assert m.elastic_gain_mm >= 0.0


# --------------------------------------------------------------------------- #
# 3. Counterfactual: Reversed Curve Orientation
# --------------------------------------------------------------------------- #

def test_counterfactual_reversed_orientation_nd():
    """Counterfactual: If a curve is reversed, naive direct matching fails;

    automated orientation search detects reversed traversal and recovers near-zero error.
    """
    for d in (2, 3, 4):
        P = sample_curve_nd(60, d=d)
        Q = P[::-1].copy()

        # Counterfactual: without orientation search (both_orientations=False), error is large
        m_direct = match_curves(P, Q, mode="open", n_points=50, rigid=False, both_orientations=False)
        assert m_direct.reversed is False
        assert m_direct.elastic_rms_mm > 5.0  # Mismatched ends

        # With automated orientation search: reversed is detected, near-zero error
        m_auto = match_curves(P, Q, mode="open", n_points=50, rigid=False, both_orientations=True)
        assert m_auto.reversed is True
        assert m_auto.elastic_rms_mm < 0.05
        assert np.allclose(m_auto.t2, 1.0 - m_auto.t1, atol=0.03)


# --------------------------------------------------------------------------- #
# 4. Counterfactual: Rigid vs Fixed Coordinate Frame
# --------------------------------------------------------------------------- #

def test_counterfactual_rigid_vs_fixed_coordinate_frame():
    """Counterfactual: Rigid=False preserves physical placement displacement;

    Rigid=True factors out extrinsic pose to isolate intrinsic shape differences.
    """
    P = sample_curve_nd(60, d=3)
    # Apply 4 mm translation and 8-degree rotation
    th = np.deg2rad(8.0)
    R_known = np.array([
        [np.cos(th), -np.sin(th), 0.0],
        [np.sin(th), np.cos(th), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_known = np.array([3.0, -2.0, 1.5])
    Q = P @ R_known.T + t_known

    # Fixed coordinate frame (rigid=False): physical displacement is reported
    m_fixed = match_curves(P, Q, mode="open", n_points=50, rigid=False)
    assert m_fixed.elastic_rms_mm > 3.0

    # Rigid alignment (rigid=True): extrinsic pose factored out, shape difference near zero
    m_rigid = match_curves(P, Q, mode="open", n_points=50, rigid=True)
    assert m_rigid.elastic_rms_mm < 0.05
    assert np.allclose(m_rigid.R, R_known, atol=1e-3)
    assert np.allclose(m_rigid.t, t_known, atol=1e-3)


# --------------------------------------------------------------------------- #
# 5. Counterfactual: Non-Linear Speed Warp vs Uniform MDF (Elastic Gain)
# --------------------------------------------------------------------------- #

def test_counterfactual_speed_warp_vs_uniform_mdf():
    """Counterfactual with ground truth: features displaced along the curve are recovered by the
    elastic match (error << ramp error), and feature-space gain is positive.

    The earlier version asserted a spatial-RMS gain on a pair that differed in *shape* rather than
    speed, which is ill-posed (a correct correspondence can increase spatial RMS).
    """
    from syntx.curves import correspondence_error, curve_arc_fraction

    tt = np.linspace(0, 1, 100)

    def curve(c1, c2):
        y = 10.0 * np.exp(-((tt - c1) / 0.05) ** 2) + 10.0 * np.exp(-((tt - c2) / 0.05) ** 2)
        return np.column_stack([100.0 * tt, y, 3.0 * np.sin(np.pi * tt)])

    # Q's bumps are at 0.4 / 0.6: the required slope between them is 0.2/0.4 = 0.5, inside the
    # default [0.45, 2.2] bounds (a displacement needing slope < 0.45 is infeasible by design).
    sl1, sl2 = curve(0.3, 0.7), curve(0.4, 0.6)
    af1, af2 = curve_arc_fraction(sl1), curve_arc_fraction(sl2)
    gt1 = np.array([af1[np.argmin(abs(tt - 0.3))], af1[np.argmin(abs(tt - 0.7))]])
    gt2 = np.array([af2[np.argmin(abs(tt - 0.4))], af2[np.argmin(abs(tt - 0.6))]])

    m = match_curves(sl1, sl2, mode="open", n_points=80, rigid=False)
    assert m.reversed is False
    elastic = correspondence_error(m, gt1, gt2)["max_error_fraction"]
    ramp_err = float(np.max(np.abs(gt1 - gt2)))  # error of the identity (uniform) map
    assert ramp_err > 0.05
    assert elastic < 0.02 < ramp_err
    assert m.feature_gain > 0.0 and m.feature_gain_rel > 0.25
    # max_speed_stretch = max(slope, 1/slope), bounded by max(2.2, 1/0.45)
    assert m.energies["max_speed_stretch"] <= max(2.2, 1.0 / 0.45) + 1e-6


# --------------------------------------------------------------------------- #
# 6. Counterfactual: Tangent Direction Discrimination (Opposite Bending)
# --------------------------------------------------------------------------- #

def test_counterfactual_tangent_direction_discrimination():
    """Counterfactual: Curves with identical unsigned curvature that bend in opposite

    directions in space are differentiated by the directional tangent channel.
    """
    t = np.linspace(0, 1, 80)
    # Curve 1 bends in +Y direction
    c1 = np.column_stack([t * 60.0, 12.0 * np.sin(np.pi * t), np.zeros_like(t)])
    # Curve 2 bends in -Y direction (identical unsigned curvature profile)
    c2 = np.column_stack([t * 60.0, -12.0 * np.sin(np.pi * t), np.zeros_like(t)])

    k1 = curve_curvature(c1, mode="open")
    k2 = curve_curvature(c2, mode="open")
    # Both curves have identical scalar curvature
    assert np.allclose(k1, k2, atol=1e-5)

    # Counterfactual: with tangent_weight=1.0, tangent energy penalizes opposite turn
    m_with_tangents = match_curves(c1, c2, mode="open", n_points=50, rigid=False, tangent_weight=2.0)
    assert m_with_tangents.energies["tangent"] > 0.5


# --------------------------------------------------------------------------- #
# 7. Counterfactual: Regularization Sensitivity Ablation
# --------------------------------------------------------------------------- #

def test_counterfactual_regularization_sensitivity_ablation():
    """Counterfactual: Increasing membrane and bending penalties strictly constrains

    reparameterization strain and brings the path closer to the linear diagonal.
    """
    sl1 = sample_curve_nd(60, d=3, seed=1)
    sl2 = sl1.copy()
    t = np.linspace(0, 1, 60)[:, None]
    sl2[:, 1] += 6.0 * np.sin(np.pi * t[:, 0])

    m_flexible = match_curves(sl1, sl2, mode="open", n_points=50, membrane=0.1, bending=0.1, rigid=False)
    m_stiff = match_curves(sl1, sl2, mode="open", n_points=50, membrane=80.0, bending=80.0, rigid=False)

    dev_flexible = np.max(np.abs(m_flexible.t2_unwrapped - m_flexible.t1))
    dev_stiff = np.max(np.abs(m_stiff.t2_unwrapped - m_stiff.t1))

    # Stiff regularization forces deviation toward zero (linear ramp)
    assert dev_stiff <= dev_flexible + 1e-6
    assert m_stiff.energies["max_speed_stretch"] <= m_flexible.energies["max_speed_stretch"] + 1e-6


# --------------------------------------------------------------------------- #
# 8. Closed Curve Cyclic Shift Recovery in N-D
# --------------------------------------------------------------------------- #

def test_closed_curve_cyclic_offset_recovery_nd():
    for d in (2, 3):
        base = resample_curve(sample_closed_loop_nd(120, d=d), 60, mode="closed")
        k = 15
        loop_shifted = np.roll(base, -k, axis=0)
        m = match_curves(base, loop_shifted, mode="closed", n_points=60, rigid=False)
        assert m.reversed is False
        assert m.mode == "closed"
        assert m.dimension == d
        # Exact cyclic fraction recovery (60 - 15) / 60 = 0.75
        assert np.isclose(m.offset_fraction, (60 - k) / 60)
        assert m.elastic_rms_mm < 0.05


# --------------------------------------------------------------------------- #
# 9. Diffeomorphism & Inverse Consistency
# --------------------------------------------------------------------------- #

def test_diffeomorphism_and_inverse_consistency_nd():
    for d in (2, 3):
        P = sample_curve_nd(60, d=d, seed=12)
        Q = P + np.array([0.4, 0.8] if d == 2 else [0.4, 0.8, -0.3])

        m = match_curves(P, Q, mode="open", n_points=50, rigid=True, bidirectional=True)
        assert m.inverse_consistency_rms is not None
        # Strict round-trip inverse consistency error < 0.05
        assert m.inverse_consistency_rms < 0.05
        # Strict monotonicity: dt2/dt1 > 0 everywhere
        dt2 = np.diff(m.t2_unwrapped)
        assert np.all(dt2 > 0)
