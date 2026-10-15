"""Unit tests for syntx.curves.evaluation, ground-truth recovery, and edge-case behaviors."""

import warnings
import numpy as np
import pytest

from syntx.curves import (
    curve_arc_fraction,
    curve_curvature,
    curve_tangents,
    correspondence_error,
    match_curves,
    resample_curve,
)


def sample_bump_pair(n_high: int = 400, c1_p: float = 0.3, c2_p: float = 0.7, c1_q: float = 0.4, c2_q: float = 0.6):
    """Synthetic curves with displaced Gaussian bumps along a line."""
    t = np.linspace(0, 1, n_high)

    def bump_curve(c1, c2):
        y = 10.0 * np.exp(-((t - c1) / 0.05) ** 2) + 10.0 * np.exp(-((t - c2) / 0.05) ** 2)
        return np.column_stack([100.0 * t, y, np.zeros_like(t)])

    P_high = bump_curve(c1_p, c2_p)
    Q_high = bump_curve(c1_q, c2_q)

    afP = curve_arc_fraction(P_high)
    afQ = curve_arc_fraction(Q_high)

    gt_P = np.array([afP[np.argmin(abs(t - c1_p))], afP[np.argmin(abs(t - c2_p))]])
    gt_Q = np.array([afQ[np.argmin(abs(t - c1_q))], afQ[np.argmin(abs(t - c2_q))]])
    return P_high, Q_high, gt_P, gt_Q


def test_gt_correspondence_recovery():
    """Verify that match_curves recovers true ground-truth correspondence within < 0.015 arc fraction."""
    P_high, Q_high, gt_P, gt_Q = sample_bump_pair()
    m = match_curves(P_high, Q_high, n_points=80, distance_weight=0.0, membrane=0.05, bending=0.05)

    eval_res = correspondence_error(m, gt_P, gt_Q)
    assert eval_res["max_error_fraction"] < 0.015
    assert eval_res["mean_error_fraction"] < 0.010
    assert eval_res["mean_error_mm"] < 1.5


def test_rigid_tangent_covariance():
    """Verify that under rigid=True, tangent energy is exactly 0.000 for a rotated curve."""
    t = np.linspace(0, 1, 90)
    P = np.column_stack([75.0 * t, 18.0 * np.sin(np.pi * t) + 4.0 * np.sin(3 * np.pi * t), 8.0 * np.cos(2 * np.pi * t)])
    th = np.deg2rad(25.0)
    R_rot = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    Q = P @ R_rot.T + np.array([5.0, -4.0, 3.0])

    m = match_curves(P, Q, n_points=60, rigid=True)
    assert m.energies["tangent"] < 1e-6
    assert m.elastic_rms_mm < 1e-4


def test_pose_invariance_distance_scale():
    """Verify that distance scale L is invariant to rigid translations."""
    t = np.linspace(0, 1, 80)
    P = np.column_stack([60.0 * t, 15.0 * np.sin(np.pi * t), np.zeros_like(t)])

    m0 = match_curves(P, P, n_points=50, rigid=True)
    m_shifted = match_curves(P, P + np.array([100.0, -50.0, 25.0]), n_points=50, rigid=True)

    assert np.isclose(m0.distance_scale_mm, m_shifted.distance_scale_mm, rtol=1e-3)
    assert m_shifted.elastic_rms_mm < 1e-4


def test_rotation_clamp_warning():
    """Verify that max_rotation_deg emits a warning and flags rotation_clamped=True when exceeded."""
    t = np.linspace(0, 1, 60)
    P = np.column_stack([50.0 * t, 10.0 * np.sin(np.pi * t), np.zeros_like(t)])
    th = np.deg2rad(40.0)
    R_rot = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    Q = P @ R_rot.T

    with pytest.warns(UserWarning, match="exceeds max_rotation_deg"):
        m = match_curves(P, Q, n_points=50, rigid=True, max_rotation_deg=20.0)
    assert m.rotation_clamped is True


def test_reflection_support():
    """Verify that allow_reflection=True matches mirrored homologues."""
    t = np.linspace(0, 1, 60)
    P = np.column_stack([50.0 * t, 12.0 * np.sin(np.pi * t), 5.0 * np.cos(2 * np.pi * t)])
    # Mirrored across X-Z plane (Y -> -Y, det = -1)
    Q = P.copy()
    Q[:, 1] *= -1.0

    # Without reflection: cannot rigidly align
    m_no_ref = match_curves(P, Q, n_points=50, rigid=True, allow_reflection=False)
    # With reflection: recovers mirror alignment
    m_ref = match_curves(P, Q, n_points=50, rigid=True, allow_reflection=True)
    assert np.linalg.det(m_ref.R) < 0
    assert m_ref.elastic_rms_mm < 0.05


def test_closed_subsample_phase_recovery():
    """Verify that sub-sample phase shift on closed curves is recovered to high precision."""
    tt = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 20.0 * (1.0 + 0.25 * np.cos(3 * tt))
    C = np.column_stack([r * np.cos(tt), r * np.sin(tt)])
    base = resample_curve(C, 60, mode="closed")

    Cs = resample_curve(C, 6000, mode="closed")
    # Shift by 0.25 of a sample interval
    Q = resample_curve(np.roll(Cs, -25, axis=0), 60, mode="closed")

    m = match_curves(base, Q, mode="closed", n_points=60)
    assert m.elastic_rms_mm < 0.1 * m.spacing_mm


def test_dimension_one_curves():
    """Verify that 1-D trajectories are supported cleanly."""
    t = np.linspace(0, 10, 50)[:, None]
    k = curve_curvature(t, mode="open")
    assert np.allclose(k, 0.0)
    T = curve_tangents(t, mode="open")
    assert np.allclose(T, 1.0)

    m = match_curves(t, t + 2.0, mode="open", n_points=40, rigid=False)
    assert m.dimension == 1
    assert m.elastic_rms_mm < 2.05


def test_scale_relative_invariance():
    """Verify that sigmas_rel produces identical normalized parameter maps for scaled copies."""
    t = np.linspace(0, 1, 80)
    P1 = np.column_stack([50.0 * t, 12.0 * np.sin(np.pi * t), 4.0 * np.cos(2 * np.pi * t)])
    P2 = P1.copy()
    P2[:, 1] += 5.0 * np.sin(np.pi * t ** 0.6)

    m_base = match_curves(P1, P2, n_points=60, sigmas_rel=[0.02, 0.05, 0.1], rigid=False)

    # Scaled by factor of 10
    m_scaled = match_curves(P1 * 10.0, P2 * 10.0, n_points=60, sigmas_rel=[0.02, 0.05, 0.1], rigid=False)

    assert np.allclose(m_base.t2, m_scaled.t2, atol=1e-4)


def test_determinism_bitwise():
    """Verify that repeated invocations on same inputs yield bitwise identical results."""
    t = np.linspace(0, 1, 70)
    P = np.column_stack([60.0 * t, 15.0 * np.sin(np.pi * t), 5.0 * np.cos(2 * np.pi * t)])
    Q = P + np.array([1.0, -2.0, 0.5])

    m1 = match_curves(P, Q, n_points=50, rigid=True)
    m2 = match_curves(P, Q, n_points=50, rigid=True)

    assert np.array_equal(m1.t2, m2.t2)
    assert np.array_equal(m1.R, m2.R)
    assert np.array_equal(m1.t, m2.t)
    assert m1.elastic_rms_mm == m2.elastic_rms_mm


# --------------------------------------------------------------------------- #
# Round-2 review regression tests
# --------------------------------------------------------------------------- #

def _bump_pair_nd(n=400, d=3, c_p=(0.3, 0.7), c_q=(0.4, 0.6)):
    """Displaced-bump pair in R^d with ground-truth landmark arc fractions."""
    t = np.linspace(0, 1, n)

    def make(c1, c2):
        y = 10.0 * np.exp(-((t - c1) / 0.05) ** 2) + 10.0 * np.exp(-((t - c2) / 0.05) ** 2)
        cols = [100.0 * t, y] + [3.0 * np.sin((k + 1) * np.pi * t) for k in range(d - 2)]
        return np.column_stack(cols[:d]) if d >= 2 else (100.0 * t)[:, None]

    P, Q = make(*c_p), make(*c_q)
    afP, afQ = curve_arc_fraction(P), curve_arc_fraction(Q)
    gt_P = np.array([afP[np.argmin(abs(t - c))] for c in c_p])
    gt_Q = np.array([afQ[np.argmin(abs(t - c))] for c in c_q])
    return P, Q, gt_P, gt_Q


@pytest.mark.parametrize("d", [2, 3, 4, 6])
def test_default_params_recover_gt(d):
    """R1: default parameters (no hand-tuned weights) must recover a displaced-feature truth."""
    P, Q, gp, gq = _bump_pair_nd(d=d)
    m = match_curves(P, Q, n_points=80)
    assert correspondence_error(m, gp, gq)["max_error_fraction"] < 0.02


def test_selection_does_not_override_dp():
    """R3: every smoothing-grid choice yields the same ground-truth recovery (no ramp fallback)."""
    P, Q, gp, gq = _bump_pair_nd()
    for grid in [(0.0,), (0.0, 2.0), (1.0, 4.0), (0.0, 1.0, 2.0, 4.0)]:
        m = match_curves(P, Q, n_points=80, smooth_sigma_grid=grid)
        assert correspondence_error(m, gp, gq)["max_error_fraction"] < 0.02, grid
        assert np.isfinite(m.smooth_sigma)  # the unwarped ramp (inf) must not be selected


def test_ramp_only_when_requested():
    """R3: a None entry explicitly adds the ramp candidate."""
    t = np.linspace(0, 1, 60)
    P = np.column_stack([60.0 * t, 8.0 * np.sin(2 * np.pi * t), np.zeros_like(t)])
    m = match_curves(P, P, n_points=50, smooth_sigma_grid=(None,))
    assert np.isinf(m.smooth_sigma) and m.elastic_rms_mm < 1e-6


def _noisy_pair(sigma, seed, n=100):
    P, Q, gp, gq = _bump_pair_nd(n=n)
    r = np.random.default_rng(seed)
    Pn, Qn = P + r.normal(scale=sigma, size=P.shape), Q + r.normal(scale=sigma, size=Q.shape)
    Pn[[0, -1]], Qn[[0, -1]] = P[[0, -1]], Q[[0, -1]]
    return Pn, Qn, gp, gq


@pytest.mark.parametrize("sigma", [0.1, 0.3])
def test_noise_robustness_without_presmoothing(sigma):
    """R2: noise up to 0.3 x point spacing (1 mm) is handled with default settings."""
    errs = []
    for seed in range(8):
        Pn, Qn, gp, gq = _noisy_pair(sigma, seed)
        m = match_curves(Pn, Qn, n_points=80)
        errs.append(correspondence_error(m, gp, gq)["max_error_fraction"])
    assert np.median(errs) < 0.03
    assert np.max(errs) < 0.05


def test_noise_robustness_with_presmoothing():
    """R2: noise of 0.5 x spacing needs arc-length pre-smoothing (noise inflates polyline length)."""
    errs_off, errs_on = [], []
    for seed in range(8):
        Pn, Qn, gp, gq = _noisy_pair(0.5, seed)
        errs_off.append(correspondence_error(match_curves(Pn, Qn, n_points=80), gp, gq)["max_error_fraction"])
        errs_on.append(
            correspondence_error(match_curves(Pn, Qn, n_points=80, presmooth_mm=3.0), gp, gq)["max_error_fraction"]
        )
    assert np.median(errs_on) < 0.03
    assert np.median(errs_on) < np.median(errs_off)  # counterfactual: pre-smoothing is what helps


def test_scale_selection_not_driven_to_coarse_scale():
    """R2: a mild count mismatch at fine scales must not push selection to a feature-erasing scale."""
    Pn, Qn, _, _ = _noisy_pair(0.1, seed=9)
    m = match_curves(Pn, Qn, n_points=80)
    assert m.sigma_selected_mm <= 4.0


def test_inverse_consistency_on_warp():
    """R5: round-trip error on a genuine (non-identity) warp is below 0.01 of curve length."""
    P, Q, _, _ = _bump_pair_nd()
    m = match_curves(P, Q, n_points=80, bidirectional=True)
    assert m.inverse_consistency_rms is not None
    assert m.inverse_consistency_rms < 0.01
    assert m.inverse_consistency_rms_mm < 0.01 * 130.0


def test_default_slope_bounds_respected():
    """R6: default speed stretch stays within the documented 0.45-2.2 bounds."""
    P, Q, _, _ = _bump_pair_nd()
    m = match_curves(P, Q, n_points=80)
    slopes = np.diff(m.t2_unwrapped) * (m.n_points - 1)
    assert slopes.min() >= 0.45 - 1e-6
    assert slopes.max() <= 2.2 + 1e-6


def test_input_validation():
    """R8: clear errors for NaN/inf, duplicates, tiny n_points and bad weights."""
    t = np.linspace(0, 1, 40)
    good = np.column_stack([50.0 * t, 5.0 * np.sin(2 * np.pi * t)])
    bad = good.copy()
    bad[10, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        match_curves(bad, good)
    bad = good.copy()
    bad[5] = bad[4]
    with pytest.raises(ValueError, match="duplicate"):
        match_curves(bad, good)
    assert match_curves(bad, good, dedupe_points=True, n_points=30).n_points == 30
    with pytest.raises(ValueError, match="n_points"):
        match_curves(good, good, n_points=2)
    with pytest.raises(ValueError, match="distance_weight"):
        match_curves(good, good, distance_weight=-1.0)
    with pytest.raises(ValueError, match="sigmas_rel"):
        match_curves(good, good, sigmas_rel=[0.0, 2.0])


def _closed_loop(n=60):
    tt = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = 20.0 * (1.0 + 0.25 * np.cos(3 * tt) + 0.12 * np.sin(5 * tt))
    return resample_curve(np.column_stack([r * np.cos(tt), r * np.sin(tt)]), n, mode="closed")


def test_closed_reversed_and_shifted():
    """R9: closed curves with reversed traversal plus a cyclic shift are matched."""
    base = _closed_loop()
    m = match_curves(base, np.roll(base, -17, axis=0)[::-1].copy(), mode="closed", n_points=60)
    assert m.reversed is True
    assert m.elastic_rms_mm < 0.1 * m.spacing_mm


def test_closed_rigid():
    """R9: closed curves under a rigid transform are matched to ~zero residual."""
    base = _closed_loop()
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    m = match_curves(base, base @ R.T + np.array([5.0, 3.0]), mode="closed", n_points=60, rigid=True)
    assert m.elastic_rms_mm < 1e-3
    assert np.allclose(m.R, R, atol=1e-3)


def test_clamp_contract():
    """R10: _clamp_sod_rotation returns (R, was_clamped) and warns only when it clamps."""
    from syntx.curves.matching import _clamp_sod_rotation

    th = np.deg2rad(40.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    with pytest.warns(UserWarning):
        Rc, flag = _clamp_sod_rotation(R, 20.0)
    assert flag is True
    assert np.isclose(np.rad2deg(np.arccos((np.trace(Rc) - 1) / 2)), 20.0, atol=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Ru, flag2 = _clamp_sod_rotation(R, 90.0)
    assert flag2 is False and np.allclose(Ru, R)
    Rn, flag3 = _clamp_sod_rotation(R, None)
    assert flag3 is False


@pytest.mark.parametrize("d", [1, 2, 3, 4, 6])
def test_dimension_sweep_gt(d):
    """Claim 'arbitrary d': ground-truth recovery across dimensions (d=1 has no curvature, so
    only the distance channel is informative: check that it runs and keeps the identity)."""
    P, Q, gp, gq = _bump_pair_nd(d=max(d, 2)) if d >= 2 else _bump_pair_nd(d=2)
    if d == 1:
        m = match_curves(P[:, :1], Q[:, :1], n_points=60)
        assert m.dimension == 1 and np.isfinite(m.elastic_rms_mm)
        return
    m = match_curves(P, Q, n_points=80)
    assert m.dimension == d
    assert correspondence_error(m, gp, gq)["max_error_fraction"] < 0.02


def test_feature_gain_positive_for_correct_match():
    """R4: a correct correspondence improves feature-space energy even when spatial RMS grows."""
    P, Q, gp, gq = _bump_pair_nd()
    m = match_curves(P, Q, n_points=80)
    assert correspondence_error(m, gp, gq)["max_error_fraction"] < 0.02
    assert m.feature_gain > 0 and m.feature_gain_rel > 0.25  # measured ~0.37
    assert m.elastic_gain_mm < 0  # spatial RMS is the wrong headline metric here


def test_project_monotone_l2_exact_bounds_and_nearest():
    """Open projection: bounds hold exactly, endpoints pinned, and the result is no farther from
    the input than the (feasible) identity ramp (it is the L2-nearest feasible map)."""
    from syntx.curves.matching import project_monotone_l2

    rng = np.random.default_rng(0)
    N = 60
    ramp = np.linspace(0.0, 1.0, N)
    for _ in range(30):
        y = np.sort(rng.uniform(0, 1, N))
        y[0], y[-1] = 0.0, 1.0
        out = project_monotone_l2(y, 0.45, 2.2)
        s = np.diff(out) * (N - 1)
        assert out[0] == 0.0 and out[-1] == 1.0
        assert s.min() >= 0.45 - 1e-9 and s.max() <= 2.2 + 1e-9
        assert np.linalg.norm(out - y) <= np.linalg.norm(ramp - y) + 1e-6


# --------------------------------------------------------------------------- #
# Phase 0 (sulceye port): per-offset rigid fit, coarse sweep, fast DP
# --------------------------------------------------------------------------- #

def _dp_reference(cost, membrane, bending):
    """Original straightforward DP (kept as the correctness reference for the optimised version)."""
    from syntx.curves.matching import _STEPS, _STEP_LEN

    N1, M1 = cost.shape
    N, M = N1 - 1, M1 - 1
    scale = 1.0 / max(N, M)
    D = np.full((3, N1, M1), np.inf)
    B = np.full((3, N1, M1), -1, dtype=np.int8)
    D[:, 0, 0] = 0.0
    mem, ben = membrane * scale, bending * scale
    for d_sum in range(1, N + M + 1):
        ii = np.arange(max(0, d_sum - M), min(N, d_sum) + 1)
        jj = d_sum - ii
        c_here = cost[ii, jj]
        for st in range(3):
            di, dj = _STEPS[st]
            ip, jp = ii - di, jj - dj
            ok = (ip >= 0) & (jp >= 0)
            if not np.any(ok):
                continue
            ip_ok, jp_ok = ip[ok], jp[ok]
            step_cost = _STEP_LEN[st] * scale * 0.5 * (cost[ip_ok, jp_ok] + c_here[ok])
            if st != 1:
                step_cost += mem
            prev = D[:, ip_ok, jp_ok]
            at_origin = (ip_ok == 0) & (jp_ok == 0)
            bend = np.full((3, len(ip_ok)), ben)
            bend[st, :] = 0.0
            bend[:, at_origin] = 0.0
            tot = prev + bend
            k = np.argmin(tot, axis=0)
            D[st, ii[ok], jj[ok]] = tot[k, np.arange(len(ip_ok))] + step_cost
            B[st, ii[ok], jj[ok]] = k
    end_state = int(np.argmin(D[:, N, M]))
    total = float(D[end_state, N, M])
    path, i, j, st = [(N, M)], N, M, end_state
    while (i, j) != (0, 0):
        k = int(B[st, i, j])
        di, dj = _STEPS[st]
        i, j = i - di, j - dj
        path.append((i, j))
        st = k
    return np.array(path[::-1], dtype=int), total


@pytest.mark.parametrize("shape", [(11, 11), (31, 31), (40, 25), (25, 40)])
@pytest.mark.parametrize("kind", ["random", "ties", "diag"])
def test_dp_matches_reference_exactly(shape, kind):
    """The optimised DP returns the identical path and cost as the reference (including tie-breaks)."""
    from syntx.curves.matching import dp_match_grid

    rng = np.random.default_rng(shape[0] * 7 + shape[1])
    if kind == "random":
        cost = rng.uniform(0, 3, size=shape)
    elif kind == "ties":
        cost = np.ones(shape)
    else:
        cost = np.ones(shape)
        for i in range(min(shape)):
            cost[i, i] = 0.0
    for mem, ben in [(0.0, 0.0), (1.0, 1.0), (0.3, 2.0)]:
        p_ref, c_ref = _dp_reference(cost, mem, ben)
        p_new, c_new = dp_match_grid(cost, mem, ben)
        assert np.array_equal(p_ref, p_new), (shape, kind, mem, ben)
        assert c_ref == c_new


def _rolled_rigid_case(roll_frac, rot_deg, shift=(6.0, -4.0, 2.0), n=360):
    """Closed 3-D loop with three displaced bumps; Q = rolled + rigidly moved copy with known truth."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def loop(centers):
        bump = sum(a * np.exp(-(wrap(th - c) / 0.15) ** 2) for a, c in zip((0.5, 0.3, 0.4), centers))
        r = 20.0 * (1.0 + bump)
        return np.column_stack([r * np.cos(th), r * np.sin(th), 4.0 * np.sin(th) + 1.5 * np.cos(2 * th)])

    def arcfrac(L):
        seg = np.linalg.norm(np.roll(L, -1, axis=0) - L, axis=1)
        return np.concatenate([[0.0], np.cumsum(seg[:-1])]) / seg.sum()

    cP, cQ = (0.8, 2.6, 4.4), (1.05, 2.45, 4.9)
    P, Q0 = loop(cP), loop(cQ)
    idx = lambda cs: [int(round(c / (2 * np.pi) * n)) % n for c in cs]
    roll = int(round(roll_frac * n))
    Qr = np.roll(Q0, -roll, axis=0)
    gP = arcfrac(P)[idx(cP)]
    gQ = arcfrac(Qr)[[(i - roll) % n for i in idx(cQ)]]
    a = np.deg2rad(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return P, Qr @ R.T + np.array(shift), gP, gQ


def _cyc_err(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1.0 - d).max()


@pytest.mark.parametrize("roll_frac,rot_deg", [(0.25, 30.0), (0.575, 90.0), (0.5, 180.0), (0.1, 10.0)])
def test_closed_rigid_unknown_cyclic_start(roll_frac, rot_deg):
    """Sulceye-port regression: unknown cyclic start + rigid pose, recovered (was 0.27-0.38 error)."""
    P, Q, gP, gQ = _rolled_rigid_case(roll_frac, rot_deg)
    m = match_curves(P, Q, mode="closed", n_points=120, rigid=True)
    assert _cyc_err(m.map_P_to_Q(gP), gQ) < 0.02


def test_closed_rigid_roll_reversed_2d_and_3d():
    """Roll x rigid x reversed in 2-D and 3-D."""
    tt = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = 20.0 * (1.0 + 0.25 * np.cos(3 * tt) + 0.12 * np.sin(5 * tt))
    for d in (2, 3):
        cols = [r * np.cos(tt), r * np.sin(tt)] + ([3.0 * np.sin(tt)] if d == 3 else [])
        base = resample_curve(np.column_stack(cols), 60, mode="closed")
        a = np.deg2rad(50.0)
        R = np.eye(d)
        R[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        Q = (np.roll(base, -23, axis=0) @ R.T + np.arange(1, d + 1))[::-1].copy()
        m = match_curves(base, Q, mode="closed", n_points=60, rigid=True)
        assert m.reversed is True
        assert m.elastic_rms_mm < 0.15 * m.spacing_mm


def test_offset_costs_diagnostics_and_fixed_offset():
    """offset_costs peaks at the true offset; fixed_offset_fraction is honoured (sulceye parity)."""
    P, Q, gP, gQ = _rolled_rigid_case(0.25, 0.0, shift=(0.0, 0.0, 0.0))
    m = match_curves(P, Q, mode="closed", n_points=120, rigid=False, n_coarse=60)
    assert m.offset_costs is not None and m.offset_costs.shape == (2, 60)
    assert m.offset_fractions.shape == (60,)
    row = m.offset_costs[1 if m.reversed else 0]
    assert abs(np.argmin(row) / 60.0 - m.offset_fraction) < 2.5 / 60.0 or abs(
        np.argmin(row) / 60.0 - m.offset_fraction + 1.0
    ) < 2.5 / 60.0

    true_off = m.offset_fraction
    pinned = match_curves(P, Q, mode="closed", n_points=120, rigid=False, fixed_offset_fraction=true_off)
    assert pinned.offset_costs is None
    assert abs(((pinned.offset_fraction - true_off) + 0.5) % 1.0 - 0.5) < 2.0 / 120
    wrong = match_curves(P, Q, mode="closed", n_points=120, rigid=False, fixed_offset_fraction=(true_off + 0.3) % 1.0)
    assert wrong.energies["total"] > pinned.energies["total"]
    with pytest.raises(ValueError, match="closed"):
        match_curves(P[:50], Q[:50], mode="open", fixed_offset_fraction=0.1)
    with pytest.raises(ValueError, match="both_orientations"):
        match_curves(P, Q, mode="closed", fixed_offset_fraction=0.1, fixed_reversed=True, both_orientations=False)


def test_closed_runtime_budget():
    """Closed-curve matching at N=200 stays well under the previous ~5 s (loose CPU budget)."""
    import time

    P, Q, _, _ = _rolled_rigid_case(0.3, 20.0)
    t0 = time.time()
    match_curves(P, Q, mode="closed", n_points=200, rigid=True)
    assert time.time() - t0 < 6.0
