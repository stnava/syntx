"""N-dimensional continuous elastic curve matching via dynamic programming.

Generalizes the symmetric scale-space curve matching framework of Avants & Gee
(Scale-Space 2003, LNCS 2695) to curves in R^d for arbitrary d >= 1:
- Open curves: Pinned endpoints, direct vs. flipped orientation search.
- Closed curves: Cyclic boundary conditions, coarse-to-fine cyclic offset sweep
  with continuous sub-sample phase refinement.
- Directional tangent vector field alignment in S^(d-1) with SO(d) rotation covariance.
- Procrustes / Kabsch rigid alignment in SO(d) with optional Lie-group geodesic clamping.
- Monotone L2 parameter projection via ADMM with Thomas algorithm O(N) solve.
- Multi-scale curvature and tangent matching with signed elastic gain.
- Bidirectional round-trip inverse consistency evaluation in arc-fraction and physical mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import warnings
import numpy as np
from scipy.linalg import expm, logm
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from syntx.curves.geometry import (
    as_curve_array,
    curve_length,
    curvature_scale_space,
    resample_curve,
    select_matching_scale,
    smooth_curve,
    tangent_scale_space,
)


# --------------------------------------------------------------------------- #
# Rigid alignment in SO(d) with Lie-group geodesic clamping
# --------------------------------------------------------------------------- #

def _clamp_sod_rotation(R: np.ndarray, max_rotation_deg: Optional[float] = None) -> tuple[np.ndarray, bool]:
    """Clamp an orthogonal proper rotation R in SO(d) along its geodesic to at most max_rotation_deg."""
    if max_rotation_deg is None:
        return R, False

    max_rad = float(np.deg2rad(max(float(max_rotation_deg), 0.0)))
    d = R.shape[0]

    if d == 1:
        return np.ones((1, 1), dtype=float), False

    if d == 2:
        theta = float(np.arctan2(R[1, 0], R[0, 0]))
        if abs(theta) <= max_rad or abs(theta) < 1e-12:
            return R, False
        warnings.warn(
            f"Rotation angle ({np.rad2deg(abs(theta)):.1f} deg) exceeds max_rotation_deg ({max_rotation_deg:.1f} deg); clamping along geodesic.",
            UserWarning,
            stacklevel=2,
        )
        th_c = float(np.sign(theta) * max_rad)
        c, s = float(np.cos(th_c)), float(np.sin(th_c))
        return np.array([[c, -s], [s, c]], dtype=float), True

    if d == 3:
        tr = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
        theta = float(np.arccos(tr))
        if theta <= max_rad or theta < 1e-8:
            return R, False
        warnings.warn(
            f"Rotation angle ({np.rad2deg(theta):.1f} deg) exceeds max_rotation_deg ({max_rotation_deg:.1f} deg); clamping along geodesic.",
            UserWarning,
            stacklevel=2,
        )
        if max_rad <= 1e-12:
            return np.eye(3, dtype=float), True
        sin_th = float(np.sin(theta))
        if abs(sin_th) < 1e-6:
            vals, vecs = np.linalg.eigh(0.5 * (R + R.T))
            axis = vecs[:, int(np.argmax(vals))]
            K = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ], dtype=float)
        else:
            K = (R - R.T) / (2.0 * sin_th)
        R_clamped = np.eye(3, dtype=float) + np.sin(max_rad) * K + (1.0 - np.cos(max_rad)) * (K @ K)
        return R_clamped, True

    # General d > 3: matrix logarithm on Lie algebra so(d)
    try:
        Omega = np.real(logm(R))
        Omega = 0.5 * (Omega - Omega.T)
        theta = float(np.linalg.norm(Omega, "fro") / np.sqrt(2.0))
        if theta <= max_rad or theta < 1e-8:
            return R, False
        warnings.warn(
            f"Rotation angle ({np.rad2deg(theta):.1f} deg) exceeds max_rotation_deg ({max_rotation_deg:.1f} deg); clamping along geodesic.",
            UserWarning,
            stacklevel=2,
        )
        scale = max_rad / theta
        R_clamped = np.real(expm(Omega * scale))
        return R_clamped, True
    except Exception as exc:
        warnings.warn(f"Lie algebra logm/expm clamping in SO({d}) encountered numerical instability: {exc}", UserWarning, stacklevel=2)
        return R, False


def kabsch_rigid(
    P: np.ndarray,
    Q: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    max_rotation_deg: Optional[float] = None,
    allow_reflection: bool = False,
    return_clamped_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, bool]:
    """Optimal rigid transformation (R, t) minimizing sum w_i ||R p_i + t - q_i||^2 in R^d.

    Returns ``(R (d, d), t (d,), rms)`` or ``(R, t, rms, was_clamped)`` if ``return_clamped_flag=True``.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have matching shape; got {P.shape} and {Q.shape}")
    N, d = P.shape

    w = np.ones(N, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        raise ValueError("Weights must sum to positive value")
    w = w / w_sum

    cp = (w[:, None] * P).sum(axis=0)
    cq = (w[:, None] * Q).sum(axis=0)
    H = (P - cp).T @ ((Q - cq) * w[:, None])

    U, _, Vt = np.linalg.svd(H)
    det_val = float(np.linalg.det(Vt.T @ U.T))
    sign_det = 1.0 if (det_val >= 0 or allow_reflection) else -1.0

    diag_vals = np.ones(d, dtype=float)
    diag_vals[-1] = sign_det
    D = np.diag(diag_vals)

    R = Vt.T @ D @ U.T
    was_clamped = False
    if max_rotation_deg is not None:
        R, was_clamped = _clamp_sod_rotation(R, float(max_rotation_deg))

    t = cq - R @ cp
    res = (P @ R.T + t) - Q
    rms = float(np.sqrt((w * (res ** 2).sum(axis=1)).sum()))
    if return_clamped_flag:
        return R, t, rms, was_clamped
    return R, t, rms


# --------------------------------------------------------------------------- #
# Dynamic programming on the parameter grid
# --------------------------------------------------------------------------- #

_STEPS = np.array([[1, 0], [1, 1], [0, 1]])  # step types: 0 (horizontal), 1 (diagonal), 2 (vertical)
_STEP_LEN = np.array([1.0, np.sqrt(2.0), 1.0])


def dp_match_grid(
    cost: np.ndarray,
    membrane: float,
    bending: float,
    mode: str = "open",
) -> tuple[np.ndarray, float]:
    """Find minimum-cost monotone path from (0, 0) to (N, M) on node cost field.

    ``cost`` has shape ``(N1, M1)`` (where N = N1 - 1, M = M1 - 1).
    Trapezoid rule integration of node costs along path, plus:
    - ``membrane / max(N, M)`` penalty for non-diagonal step.
    - ``bending / max(N, M)`` penalty when the step direction changes.

    Returns ``(path (L, 2), total_cost)``.
    """
    N1, M1 = cost.shape
    N, M = N1 - 1, M1 - 1
    scale = 1.0 / max(N, M)
    inf = np.inf
    D = np.full((3, N1, M1), inf, dtype=np.float64)
    B = np.full((3, N1, M1), -1, dtype=np.int8)
    D[:, 0, 0] = 0.0
    mem = membrane * scale
    ben = bending * scale
    # bend_rows[st][k] = ben if the previous step type k differs from st, else 0
    bend_rows = (ben * (1.0 - np.eye(3)))[:, :, None]
    coef = _STEP_LEN * scale * 0.5  # same association order as the reference: (len*scale)*0.5
    ar = np.arange(max(N, M) + 1)

    for d_sum in range(1, N + M + 1):
        i_lo = max(0, d_sum - M)
        i_hi = min(N, d_sum)
        n_d = i_hi - i_lo + 1
        ii = ar[i_lo:i_hi + 1]
        jj = d_sum - ii
        c_here = cost[ii, jj]

        for st in range(3):
            di, dj = _STEPS[st]
            # Valid predecessors form a contiguous run: ip >= 0 trims the first cell when i_lo == 0
            # (di == 1); jp >= 0 trims the last cell when it sits on j == 0 (dj == 1).
            lo = 1 if (di == 1 and i_lo == 0) else 0
            hi = n_d - 1 if (dj == 1 and i_hi == d_sum) else n_d
            if lo >= hi:
                continue
            si, sj = ii[lo:hi], jj[lo:hi]
            ip, jp = si - di, sj - dj
            step_cost = coef[st] * (cost[ip, jp] + c_here[lo:hi])
            if st != 1:
                step_cost += mem

            prev = D[:, ip, jp]
            tot = prev + bend_rows[st]
            if d_sum <= 2:  # bending is not charged on the first step out of the origin
                at_origin = (ip == 0) & (jp == 0)
                if at_origin.any():
                    tot[:, at_origin] = prev[:, at_origin]
            k = tot.argmin(axis=0)
            D[st, si, sj] = tot[k, ar[:hi - lo]] + step_cost
            B[st, si, sj] = k

    end_state = int(np.argmin(D[:, N, M]))
    total = float(D[end_state, N, M])
    if not np.isfinite(total):
        raise RuntimeError("Dynamic programming found no finite path")

    # Backtrack path
    path = [(N, M)]
    i, j, st = N, M, end_state
    while (i, j) != (0, 0):
        k = int(B[st, i, j])
        di, dj = _STEPS[st]
        i, j = i - di, j - dj
        path.append((i, j))
        st = k
    return np.array(path[::-1], dtype=int), total


def path_to_map_subsample(path: np.ndarray, N: int, M: int, mode: str = "open") -> np.ndarray:
    """Project discrete DP path onto continuous sub-sample parameter map t2(i) for i = 0..N-1.

    Resolves vertical and horizontal multi-point steps by taking segment midpoints and
    interpolating across plateau regions.
    """
    t2 = np.empty(N, dtype=np.float64)
    denom = float(M) if mode == "closed" else float(max(M - 1, 1))

    present_i = []
    present_j = []
    for i in range(N):
        matching_j = path[path[:, 0] == i, 1]
        if len(matching_j) > 0:
            present_i.append(i)
            # Midpoint across multiple j entries
            mid_j = 0.5 * (float(np.min(matching_j)) + float(np.max(matching_j)))
            present_j.append(mid_j)

    if len(present_i) == 0:
        return np.linspace(0.0, 1.0, N) if mode == "open" else np.arange(N) / N

    # Linearly interpolate missing indices across horizontal steps
    all_i = np.arange(N, dtype=float)
    t2_interp = np.interp(all_i, present_i, present_j) / denom

    if mode == "open":
        t2_interp[0] = 0.0
        t2_interp[-1] = 1.0
    return t2_interp


def path_to_map(path: np.ndarray, N: int, M: int, mode: str = "open") -> np.ndarray:
    """Project DP path to parameter map t2(i) for i = 0..N-1 (backwards compatibility)."""
    return path_to_map_subsample(path, N, M, mode=mode)


# --------------------------------------------------------------------------- #
# Monotone L2 Projection via ADMM with Thomas Algorithm
# --------------------------------------------------------------------------- #

def _solve_tridiagonal(diag: np.ndarray, off_diag: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve symmetric tridiagonal linear system A x = b in O(N) using Thomas algorithm."""
    n = len(diag)
    c_prime = np.zeros(n - 1, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)

    denom0 = float(diag[0])
    c_prime[0] = off_diag[0] / denom0
    d_prime[0] = rhs[0] / denom0

    for i in range(1, n - 1):
        denom = diag[i] - off_diag[i - 1] * c_prime[i - 1]
        denom = max(denom, 1e-14)
        c_prime[i] = off_diag[i] / denom
        d_prime[i] = (rhs[i] - off_diag[i - 1] * d_prime[i - 1]) / denom

    denom_end = diag[-1] - off_diag[-1] * c_prime[-1]
    denom_end = max(denom_end, 1e-14)
    d_prime[-1] = (rhs[-1] - off_diag[-1] * d_prime[-2]) / denom_end

    x = np.zeros(n, dtype=np.float64)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def project_monotone_l2(
    y: np.ndarray,
    min_slope: float = 0.45,
    max_slope: Optional[float] = 2.2,
    mode: str = "open",
    max_iters: int = 80,
    rho: float = 2.0,
) -> np.ndarray:
    """Project parameter map y onto strictly increasing domain with slope bounds.

    ``mode='open'`` solves the exact L2 projection (ADMM with a Thomas tridiagonal solve):
        min_t 0.5 * ||t - y||^2
        subject to:
            lo <= t_{i+1} - t_i <= hi
            t_0 = 0, t_{N-1} = 1
    ``mode='closed'`` returns a *feasible* map (slope bounds and unit total span hold) obtained by
    clip-and-redistribute of the increments; it is NOT the exact L2-nearest projection.
    """
    N = len(y)
    if N < 3:
        return y.copy()

    if mode == "open":
        dt1 = 1.0 / (N - 1)
        lo = max(float(min_slope), 1e-4) * dt1
        hi = float(max_slope) * dt1 if max_slope is not None else 1.0
        if lo > dt1:
            lo = dt1
        if hi < dt1:
            hi = dt1

        M = N - 2
        diag = (1.0 + 2.0 * rho) * np.ones(M, dtype=np.float64)
        off = -rho * np.ones(M - 1, dtype=np.float64)

        z = np.diff(y)
        z = np.clip(z, lo, hi)
        z_sum = float(np.sum(z))
        if z_sum > 0:
            z *= 1.0 / z_sum
        u = np.zeros(N - 1, dtype=np.float64)

        t = y.copy()
        t[0] = 0.0
        t[-1] = 1.0

        for _ in range(max_iters):
            w = z - u
            rhs = y[1:-1] + rho * (w[:-1] - w[1:])
            rhs[-1] += rho * 1.0

            t[1:-1] = _solve_tridiagonal(diag, off, rhs)

            # Update z
            dt = np.diff(t)
            v = dt + u
            v_clip = np.clip(v, lo, hi)
            for _ in range(12):
                diff = 1.0 - float(np.sum(v_clip))
                if abs(diff) < 1e-10:
                    break
                free = (v_clip < hi - 1e-9) if diff > 0 else (v_clip > lo + 1e-9)
                if not np.any(free):
                    break
                v_clip[free] = np.clip(v_clip[free] + diff / np.sum(free), lo, hi)
            z = v_clip

            res = dt - z
            u += res
            if np.max(np.abs(res)) < 1e-5:
                break
        # ADMM converges only to a tolerance: enforce the hard bounds exactly on the increments.
        inc = np.clip(np.diff(t), lo, hi)
        for _ in range(50):
            diff = 1.0 - float(np.sum(inc))
            if abs(diff) < 1e-12:
                break
            free = (inc < hi - 1e-12) if diff > 0 else (inc > lo + 1e-12)
            if not np.any(free):
                break
            inc[free] = np.clip(inc[free] + diff / np.sum(free), lo, hi)
        t = np.concatenate([[0.0], np.cumsum(inc)])
        t[-1] = 1.0
        return t

    # Closed curve
    dt1 = 1.0 / N
    lo = max(float(min_slope), 1e-4) * dt1
    hi = float(max_slope) * dt1 if max_slope is not None else 1.0
    inc = np.diff(np.concatenate([y, [y[0] + 1.0]]))
    inc_clip = np.clip(inc, lo, hi)
    for _ in range(12):
        diff = 1.0 - float(np.sum(inc_clip))
        if abs(diff) < 1e-10:
            break
        free = (inc_clip < hi - 1e-9) if diff > 0 else (inc_clip > lo + 1e-9)
        if not np.any(free):
            break
        inc_clip[free] = np.clip(inc_clip[free] + diff / np.sum(free), lo, hi)
    return y[0] + np.concatenate([[0.0], np.cumsum(inc_clip[:-1])])


def enforce_bounded_increments(
    inc: np.ndarray,
    total_span: float,
    min_slope: float = 0.45,
    max_slope: Optional[float] = 2.2,
) -> np.ndarray:
    """Project step increments onto bounded interval (backwards compatibility)."""
    inc = np.asarray(inc, dtype=np.float64).copy()
    n = len(inc)
    if n == 0 or total_span <= 0:
        return np.full(n, total_span / max(n, 1), dtype=np.float64)
    mean_step = float(total_span) / n
    lo = max(0.0, float(min_slope)) * mean_step
    hi = float(max_slope) * mean_step if (max_slope is not None and np.isfinite(max_slope)) else float(total_span)
    inc = np.clip(inc, lo, hi)

    for _ in range(16):
        diff = float(total_span - inc.sum())
        if abs(diff) <= 1e-12 * total_span:
            break
        if diff > 0:
            free = inc < (hi - 1e-12 * mean_step)
        else:
            free = inc > (lo + 1e-12 * mean_step)
        if not np.any(free):
            inc += diff / n
            break
        inc[free] = np.clip(inc[free] + diff / int(np.sum(free)), lo, hi)
    inc *= float(total_span) / max(float(inc.sum()), 1e-15)
    return inc


def enforce_monotone(
    t: np.ndarray,
    eps: float,
    max_step: Optional[float] = None,
    mode: str = "open",
) -> np.ndarray:
    """Enforce strictly increasing map with increments in [eps, max_step] (backwards compatibility)."""
    n = len(t)
    min_slope = float(eps) * (n - 1 if mode == "open" else n)
    max_slope = (float(max_step) * (n - 1 if mode == "open" else n)) if max_step is not None else None
    return project_monotone_l2(t, min_slope=min_slope, max_slope=max_slope, mode=mode)


# --------------------------------------------------------------------------- #
# Curve matching dataclass
# --------------------------------------------------------------------------- #

@dataclass
class CurveMatch:
    """Outcome of N-dimensional elastic curve matching."""

    t1: np.ndarray
    t2: np.ndarray
    t2_unwrapped: np.ndarray
    offset_fraction: float
    reversed: bool
    R: np.ndarray
    t: np.ndarray
    baseline_rms_mm: float
    elastic_rms_mm: float
    elastic_gain_mm: float
    sigma_selected_mm: float
    distance_scale_mm: float
    spacing_mm: float
    energies: dict
    energies_baseline: dict
    smooth_sigma: float
    n_points: int
    mode: str
    dimension: int
    path: np.ndarray = field(repr=False)
    cost_field: Optional[np.ndarray] = field(default=None, repr=False)
    features_P: Optional[np.ndarray] = field(default=None, repr=False)
    features_Q_oriented: Optional[np.ndarray] = field(default=None, repr=False)
    scales_mm: Optional[np.ndarray] = field(default=None, repr=False)
    P_resampled: Optional[np.ndarray] = field(default=None, repr=False)
    Q_resampled: Optional[np.ndarray] = field(default=None, repr=False)
    extrema_counts: Optional[dict] = field(default=None, repr=False)
    inverse_consistency_rms: Optional[float] = None
    inverse_consistency_rms_mm: Optional[float] = None
    rotation_clamped: bool = False
    orientation_cost_gap: float = 0.0
    orientation_cost_gap_rel: float = 0.0
    feature_gain: float = 0.0
    feature_gain_rel: float = 0.0
    offset_costs: Optional[np.ndarray] = field(default=None, repr=False)
    offset_fractions: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def rigid_rms_mm(self) -> float:
        """Backwards compatibility alias for baseline_rms_mm."""
        return self.baseline_rms_mm

    def map_P_to_Q(self, s: np.ndarray | float) -> np.ndarray:
        """Map arc-fraction(s) on P to corresponding arc-fraction(s) on Q."""
        s_arr = np.asarray(s, dtype=np.float64)
        if self.mode == "closed":
            s_mod = np.mod(s_arr, 1.0)
            x = np.concatenate([self.t1, [1.0]])
            y = np.concatenate([self.t2_unwrapped, [self.t2_unwrapped[0] + 1.0]])
            s_o = np.interp(s_mod, x, y)
            N = self.n_points
            t_q = np.mod(((N - 1) / N - s_o) if self.reversed else s_o, 1.0)
            return t_q

        # Open curve
        s_clip = np.clip(s_arr, 0.0, 1.0)
        t_mapped = np.interp(s_clip, self.t1, self.t2_unwrapped)
        if self.reversed:
            t_mapped = 1.0 - t_mapped
        return t_mapped

    def map_Q_to_P(self, s: np.ndarray | float) -> np.ndarray:
        """Map arc-fraction(s) on Q to corresponding arc-fraction(s) on P."""
        s_arr = np.asarray(s, dtype=np.float64)
        if self.mode == "closed":
            N = self.n_points
            s_o = ((N - 1) / N - s_arr) if self.reversed else s_arr
            y = np.concatenate([self.t2_unwrapped, [self.t2_unwrapped[0] + 1.0]])
            x = np.concatenate([self.t1, [1.0]])
            s_un = self.t2_unwrapped[0] + np.mod(s_o - self.t2_unwrapped[0], 1.0)
            return np.mod(np.interp(s_un, y, x), 1.0)

        # Open curve
        s_clip = np.clip(s_arr, 0.0, 1.0)
        s_o = (1.0 - s_clip) if self.reversed else s_clip
        return np.interp(s_o, self.t2_unwrapped, self.t1)

    def warp_P(self) -> np.ndarray:
        """Rigidly move and resample P into correspondence frame of Q."""
        if self.P_resampled is None:
            raise ValueError("P_resampled is required to evaluate warped coordinates")
        return self.P_resampled @ self.R.T + self.t

    def warp_Q(self) -> np.ndarray:
        """Interpolate Q at the matched correspondence points of P."""
        if self.Q_resampled is None:
            raise ValueError("Q_resampled is required to evaluate warped coordinates")
        t_on_Q = self.map_P_to_Q(self.t1)
        N = len(self.Q_resampled)
        if self.mode == "closed":
            idx = np.mod(t_on_Q, 1.0) * N
            i0 = np.floor(idx).astype(int) % N
            i1 = (i0 + 1) % N
            w = (idx - np.floor(idx))[:, None]
            return self.Q_resampled[i0] * (1.0 - w) + self.Q_resampled[i1] * w

        # Open curve
        idx = np.clip(t_on_Q, 0.0, 1.0) * (N - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.clip(i0 + 1, 0, N - 1)
        w = (idx - i0)[:, None]
        return self.Q_resampled[i0] * (1.0 - w) + self.Q_resampled[i1] * w


# --------------------------------------------------------------------------- #
# Energy computation
# --------------------------------------------------------------------------- #

def _interp_points_nd(Q: np.ndarray, t: np.ndarray, mode: str = "open") -> np.ndarray:
    """Interpolate curve points in R^d at fractions t."""
    N = len(Q)
    if mode == "closed":
        idx = np.mod(t, 1.0) * N
        i0 = np.floor(idx).astype(int) % N
        i1 = (i0 + 1) % N
        w = (idx - np.floor(idx))[:, None]
        return Q[i0] * (1.0 - w) + Q[i1] * w

    idx = np.clip(t, 0.0, 1.0) * (N - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, N - 1)
    w = (idx - i0)[:, None]
    return Q[i0] * (1.0 - w) + Q[i1] * w


def _curve_energy_nd(
    t2_un: np.ndarray,
    FP: np.ndarray,
    FQ: np.ndarray,
    TP: np.ndarray,
    TQ: np.ndarray,
    Pa: np.ndarray,
    Q: np.ndarray,
    L: float,
    feature_weight: float,
    tangent_weight: float,
    distance_weight: float,
    membrane: float,
    bending: float,
    power: float,
    mode: str = "open",
) -> dict:
    """Calculate continuous energy terms for a reparameterization map t2(t1) in R^d."""
    N = len(t2_un)
    if mode == "closed":
        idx = np.mod(t2_un, 1.0) * N
        i0 = np.floor(idx).astype(int) % N
        i1 = (i0 + 1) % N
        w = (idx - np.floor(idx))
        FQm = FQ[:, i0] * (1.0 - w) + FQ[:, i1] * w
        if TQ.ndim == 3:
            w_3d = w[None, :, None]
            TQm = TQ[:, i0, :] * (1.0 - w_3d) + TQ[:, i1, :] * w_3d
        else:
            TQm = TQ[i0] * (1.0 - w[:, None]) + TQ[i1] * w[:, None]
    else:
        idx = np.clip(t2_un, 0.0, 1.0) * (N - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.clip(i0 + 1, 0, N - 1)
        w = (idx - i0)
        FQm = FQ[:, i0] * (1.0 - w) + FQ[:, i1] * w
        if TQ.ndim == 3:
            w_3d = w[None, :, None]
            TQm = TQ[:, i0, :] * (1.0 - w_3d) + TQ[:, i1, :] * w_3d
        else:
            TQm = TQ[i0] * (1.0 - w[:, None]) + TQ[i1] * w[:, None]

    Qm = _interp_points_nd(Q, t2_un, mode=mode)

    # Curvature feature mismatch
    feat_diff = np.linalg.norm(FP - FQm, axis=0) ** power

    # Multi-scale tangent mismatch: 1 - sum_k T_P,k * T_Q,k averaged across scales
    if TP.ndim == 3 and TQm.ndim == 3:
        cos_sim = np.clip(np.sum(TP * TQm, axis=2).mean(axis=0), -1.0, 1.0)
    else:
        cos_sim = np.clip(np.sum(TP * TQm, axis=-1), -1.0, 1.0)
    tangent_diff = 1.0 - cos_sim

    # Euclidean distance in R^d
    dist = ((Pa - Qm) ** 2).sum(axis=1) / max(L ** 2, 1e-12)

    # Step length and speeds
    if mode == "closed":
        dt2 = np.diff(np.concatenate([t2_un, [t2_un[0] + 1.0]]))
        dt1 = 1.0 / N
    else:
        dt2 = np.diff(t2_un)
        dt1 = 1.0 / max(N - 1, 1)

    gdot = np.sqrt(dt1 ** 2 + dt2 ** 2)
    w_step = np.empty(N, dtype=np.float64)
    if mode == "closed":
        w_step = 0.5 * (gdot + np.roll(gdot, 1))
    else:
        w_step[0] = 0.5 * gdot[0]
        w_step[-1] = 0.5 * gdot[-1]
        w_step[1:-1] = 0.5 * (gdot[:-1] + gdot[1:])

    e_feat = float(feature_weight * np.sum(w_step * feat_diff))
    e_tangent = float(tangent_weight * np.sum(w_step * tangent_diff))
    e_dist = float(distance_weight * np.sum(w_step * dist))

    slope = dt2 / max(dt1, 1e-12)
    e_mem = float(membrane * np.mean((slope - 1.0) ** 2))
    if mode == "closed":
        curv = np.diff(np.concatenate([slope, [slope[0]]]))
    else:
        curv = np.diff(slope) if len(slope) > 1 else np.array([0.0])
    e_bend = float(bending * np.mean(curv ** 2))

    slope_safe = np.maximum(slope, 1e-6)
    max_speed_stretch = float(np.max(np.maximum(slope_safe, 1.0 / slope_safe)))
    rms_mm = float(np.sqrt(((Pa - Qm) ** 2).sum(axis=1).mean()))

    return {
        "feature": e_feat,
        "tangent": e_tangent,
        "distance": e_dist,
        "membrane": e_mem,
        "bending": e_bend,
        "total": e_feat + e_tangent + e_dist + e_mem + e_bend,
        "distance_rms_mm": rms_mm,
        "max_speed_stretch": max_speed_stretch,
    }


# --------------------------------------------------------------------------- #
# Unified N-D curve matching function
# --------------------------------------------------------------------------- #

def _validate_curve(curve, mode: str, name: str, dedupe: bool) -> np.ndarray:
    """Validate an input curve: finite values, no consecutive duplicate points."""
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim == 2 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite (NaN/inf) coordinates")
    P = as_curve_array(arr, mode=mode)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if mode == "closed" else P, axis=0), axis=1)
    dup = seg <= 1e-12
    if np.any(dup):
        if not dedupe:
            raise ValueError(
                f"{name} has {int(dup.sum())} consecutive duplicate point(s); "
                "remove them or pass dedupe_points=True"
            )
        keep = np.concatenate([[True], ~dup[: len(P) - 1]])
        P = P[keep]
        as_curve_array(P, mode=mode)
    return P


def match_curves(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    mode: str = "open",
    n_points: int = 60,
    sigmas_mm: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    sigmas_rel: Optional[Sequence[float]] = None,
    sigma_mm: Optional[float] = None,
    distance_scale_mm: Optional[float] = None,
    distance_scale_rel: float = 0.3,
    feature_weight: float = 1.0,
    tangent_weight: float = 1.0,
    distance_weight: float = 1.0,
    membrane: float = 1.0,
    bending: float = 1.0,
    power: float = 0.5,
    rigid: bool = False,
    rigid_iters: int = 2,
    max_rotation_deg: Optional[float] = None,
    allow_reflection: bool = False,
    both_orientations: bool = True,
    smooth_sigma_grid: Sequence[Optional[float]] = (0.0, 1.0, 2.0, 4.0),
    min_slope: float = 0.45,
    max_slope: Optional[float] = 2.2,
    bidirectional: bool = False,
    dedupe_points: bool = False,
    presmooth_mm: Optional[float] = None,
    n_coarse: int = 60,
    n_refine: int = 3,
    fixed_offset_fraction: Optional[float] = None,
    fixed_reversed: bool = False,
) -> CurveMatch:
    """Continuous elastic matching of curve P onto curve Q in R^d.

    Parameters
    ----------
    P, Q : (N, d) arrays
        Curve coordinates in physical space (mm) for arbitrary dimension d >= 1.
    mode : 'open' or 'closed'
        'open' for polylines, centerlines, or sulcal fundi (endpoints pinned).
        'closed' for boundary loops or contours (cyclic boundary conditions).
    n_points : int
        Number of arc-length equispaced sample points.
    sigmas_mm : sequence of float
        Gaussian smoothing scales in millimeters for scale-space features.
    sigmas_rel : sequence of float, optional
        Gaussian smoothing scales expressed as fractions of total curve length.
        If provided, overrides ``sigmas_mm`` to guarantee scale invariance.
    sigma_mm : float, optional
        Fixed scale in mm. If None, selected via curvature extrema topology.
    distance_scale_mm : float, optional
        Normalization distance d0. Defaults to the nearest-neighbour RMS distance, floored at
        ``distance_scale_rel`` x mean curve length (see below) and at the sample spacing.
    distance_scale_rel : float
        Floor on d0 as a fraction of mean curve length (default 0.3). The nearest-neighbour RMS is
        tiny whenever the curves overlap spatially, even if the true correspondence moves features
        far along the curve; without a floor the distance term dominates and pins the match to
        the identity ramp. Empirically 0.1 was still too small on a displaced-feature test; >= 0.3
        recovers the ground truth.
    feature_weight : float
        Weight on scale-space curvature differences.
    tangent_weight : float
        Weight on tangent direction differences (1 - cos(theta)).
    distance_weight : float
        Weight on normalized Euclidean distance.
    membrane, bending : float
        Penalties on local stretch/compression and path curvature.
    power : float
        Exponent on curvature difference (default 0.5).
    rigid : bool
        If True, estimates d-dimensional rigid alignment via Kabsch/Procrustes iteration.
        If False (default), curves are matched in their native coordinate frame.
    rigid_iters : int
        Number of alternating rigid-alignment and DP iterations.
    max_rotation_deg : float, optional
        Bound on SO(d) rotation angle (in degrees) during rigid alignment (default None).
    allow_reflection : bool
        If True, allows improper orthogonal transformations with det(R) = -1.
    both_orientations : bool
        If True, tests direct (Q) and flipped (Q_reversed) orientations.
    smooth_sigma_grid : sequence of float or None
        Smoothing standard deviations (in samples) applied to the DP residual. The best warp among
        these candidates is chosen. ``None`` entries add the unwarped linear ramp as an explicit
        candidate; the ramp is otherwise never selected, so an energy mismatch cannot silently
        discard the DP correspondence.
    min_slope, max_slope : float
        Bounds on the local slope dt2/dt1 (speed stretch).
    bidirectional : bool
        If True, also matches Q to P to evaluate inverse consistency error.
    dedupe_points : bool
        If True, drop consecutive duplicate points before matching; otherwise they raise ValueError.
    presmooth_mm : float, optional
        Gaussian pre-smoothing (mm) of both inputs *before* arc-length resampling, endpoints
        preserved. Measurement noise inflates polyline arc length and distorts the arc-length
        parametrisation (measured: sigma = 0.5 mm noise at 1 mm spacing inflates length by 1.33x and
        breaks correspondence recovery); set this to a few times the noise level to suppress it.
    n_coarse : int
        Closed curves: number of points of the coarse cyclic-offset sweep (default 60). For every
        candidate offset a rigid pose is fitted under that hypothesised correspondence (when
        ``rigid=True``) before scoring it, so an unknown cyclic start no longer corrupts the pose.
    n_refine : int
        Closed curves: number of best coarse offsets per orientation refined at full resolution.
    fixed_offset_fraction : float, optional
        Closed curves: pin the cyclic offset (fraction of Q's perimeter matched to P's point 0)
        instead of sweeping; the coarse sweep is skipped, so ``offset_costs`` is None.
    fixed_reversed : bool
        With ``fixed_offset_fraction``, pin the orientation of Q (requires ``both_orientations=True``).

    Returns
    -------
    CurveMatch
        Rich match outcome with correspondences, energies, and metrics.
    """
    P_raw = _validate_curve(P, mode, "P", dedupe_points)
    Q_raw = _validate_curve(Q, mode, "Q", dedupe_points)
    if presmooth_mm is not None and presmooth_mm > 0:
        for _nm in ("P_raw", "Q_raw"):
            _c = P_raw if _nm == "P_raw" else Q_raw
            _sp = curve_length(_c, mode=mode) / max(len(_c) - (0 if mode == "closed" else 1), 1)
            _sm = smooth_curve(_c, float(presmooth_mm) / max(_sp, 1e-9), mode=mode)
            if _nm == "P_raw":
                P_raw = _sm
            else:
                Q_raw = _sm
    if P_raw.shape[1] != Q_raw.shape[1]:
        raise ValueError(f"P and Q must have matching dimension; got {P_raw.shape[1]} and {Q_raw.shape[1]}")
    if mode not in ("open", "closed"):
        raise ValueError(f"mode must be 'open' or 'closed'; got {mode!r}")
    if int(n_points) < 3:
        raise ValueError(f"n_points must be >= 3; got {n_points}")
    if sigmas_rel is not None and not all(0.0 < float(s) <= 1.0 for s in sigmas_rel):
        raise ValueError("sigmas_rel entries must lie in (0, 1]")
    for _name, _w in (("feature_weight", feature_weight), ("tangent_weight", tangent_weight),
                      ("distance_weight", distance_weight), ("membrane", membrane), ("bending", bending)):
        if not np.isfinite(_w) or _w < 0:
            raise ValueError(f"{_name} must be finite and >= 0; got {_w}")
    d = P_raw.shape[1]
    N = int(n_points)

    P_res = resample_curve(P_raw, N, mode=mode)
    Q_res = resample_curve(Q_raw, N, mode=mode)

    LP, LQ = curve_length(P_res, mode=mode), curve_length(Q_res, mode=mode)
    spacing_mm = 0.5 * (LP + LQ) / max(N - (0 if mode == "closed" else 1), 1)

    if sigmas_rel is not None:
        sigmas_mm = [float(s) * float(LP) for s in sigmas_rel]

    sigmas = [float(s) / max(spacing_mm, 1e-6) for s in sigmas_mm]

    sig_sel, F1, F2 = select_matching_scale(P_res, Q_res, sigmas, mode=mode)
    if sigma_mm is not None:
        sig_sel = float(sigma_mm) / max(spacing_mm, 1e-6)
    use_sig = [s for s in sigmas if s >= sig_sel] or [sig_sel]

    # Scale-space curvature features
    FP = curvature_scale_space(P_res, use_sig, mode=mode) * LP
    FQ = curvature_scale_space(Q_res, use_sig, mode=mode) * LQ
    feat_scale = float(np.sqrt(0.5 * (np.mean(FP ** 2) + np.mean(FQ ** 2))))
    if feat_scale > 1e-12:
        FP /= feat_scale
        FQ /= feat_scale

    # Multi-scale tangents in S^(d-1)
    TP_ms = tangent_scale_space(P_res, use_sig, mode=mode)  # (S, N, d)
    TQ_ms = tangent_scale_space(Q_res, use_sig, mode=mode)

    d0_floor = max(float(distance_scale_rel) * 0.5 * (LP + LQ), spacing_mm)

    def dist_scale(Pa_x, Q_x):
        """Distance normaliser d0: NN-RMS between the (aligned) curves, floored (see distance_scale_rel)."""
        if distance_scale_mm is not None:
            return max(float(distance_scale_mm), spacing_mm)
        d_pq = cKDTree(Q_x).query(Pa_x)[0]
        d_qp = cKDTree(Pa_x).query(Q_x)[0]
        return max(float(np.sqrt(0.5 * (np.mean(d_pq ** 2) + np.mean(d_qp ** 2)))), d0_floor)

    def node_cost(Pa_x, TP_x, FP_x, Q_x, FQ_x, TQ_x, L_x):
        """Node cost field M[i, j] = feature + tangent + normalised-distance mismatch of P_i vs Q_j."""
        M_feat = np.linalg.norm(FP_x.T[:, None, :] - FQ_x.T[None, :, :], axis=2) ** power
        cos_grid = np.clip(np.sum(TP_x[:, :, None, :] * TQ_x[:, None, :, :], axis=3).mean(axis=0), -1.0, 1.0)
        d2 = ((Pa_x[:, None, :] - Q_x[None, :, :]) ** 2).sum(axis=2)
        return feature_weight * M_feat + tangent_weight * (1.0 - cos_grid) + distance_weight * d2 / (L_x ** 2)

    def fit_pose(A, B, quiet=False):
        """Rigid pose mapping A onto B (identity if rigid=False). ``quiet`` suppresses clamp warnings."""
        if not rigid:
            return np.eye(d), np.zeros(d), False
        with warnings.catch_warnings():
            if quiet:
                warnings.simplefilter("ignore")
            R_, t_, _, flag = kabsch_rigid(
                A, B, max_rotation_deg=max_rotation_deg, allow_reflection=allow_reflection,
                return_clamped_flag=True,
            )
        return R_, t_, flag

    def closed_grid(M, k):
        """(N+1, N+1) node field for Q rolled by k, closed with the wrapped first row/column."""
        Mk = np.roll(M, -k, axis=1)
        Mk = np.vstack([Mk, Mk[:1]])
        return np.hstack([Mk, Mk[:, :1]])

    def solve_from(Q_o, FQ_o, TQ_o, R_cur, t_cur, k_center, win, was_clamped):
        """Alternate DP (over cyclic offsets k_center +- win for closed curves) and rigid refit."""
        L = dist_scale(P_res @ R_cur.T + t_cur, Q_o)
        n_it = max(1, rigid_iters if rigid else 1)
        for it in range(n_it):
            Pa_cur = P_res @ R_cur.T + t_cur
            M_node = node_cost(Pa_cur, TP_ms @ R_cur.T, FP, Q_o, FQ_o, TQ_o, L)
            if mode == "closed":
                best = None
                w_it = win if it == 0 else min(win, 1)  # offset is already localised after iteration 0
                for k in range(k_center - w_it, k_center + w_it + 1):
                    kk = k % N
                    M_grid_k = closed_grid(M_node, kk)
                    path_k, c_k = dp_match_grid(M_grid_k, membrane, bending, mode="closed")
                    if best is None or c_k < best[0]:
                        best = (c_k, kk, path_k, M_grid_k)
                cost_val, chosen_offset, path, M_grid = best
                k_center = chosen_offset
            else:
                path, cost_val = dp_match_grid(M_node, membrane, bending, mode="open")
                M_grid, chosen_offset = M_node, 0
            if rigid and it < n_it - 1:
                t2_map = path_to_map_subsample(path, N, N, mode=mode)
                if mode == "closed":
                    t2_map = (chosen_offset / N + t2_map) % 1.0
                R_cur, t_cur, c_flag = fit_pose(P_res, _interp_points_nd(Q_o, t2_map, mode=mode), quiet=True)
                was_clamped = was_clamped or c_flag
        return (cost_val, path, R_cur, t_cur, M_grid, chosen_offset, L, was_clamped)

    # Coarse cyclic-offset sweep (closed curves): at n_coarse points, with a rigid fit PER candidate
    # offset (the pose must be estimated under each hypothesised correspondence, not once beforehand).
    n_c = min(int(n_coarse), N)
    if mode == "closed":
        if n_c < N:
            sig_c = [s * n_c / N for s in use_sig]
            Pc = resample_curve(P_res, n_c, mode="closed")
            FPc = curvature_scale_space(Pc, sig_c, mode="closed") * LP / (feat_scale if feat_scale > 1e-12 else 1.0)
            TPc = tangent_scale_space(Pc, sig_c, mode="closed")
        else:
            Pc, FPc, TPc = P_res, FP, TP_ms

    def coarse_offset_costs(Q_o, FQ_o, TQ_o):
        if n_c < N:
            Qc = resample_curve(Q_o, n_c, mode="closed")
            FQc = curvature_scale_space(Qc, sig_c, mode="closed") * LQ / (feat_scale if feat_scale > 1e-12 else 1.0)
            TQc = tangent_scale_space(Qc, sig_c, mode="closed")
        else:
            Qc, FQc, TQc = Q_o, FQ_o, TQ_o
        costs = np.empty(n_c)
        for kc in range(n_c):
            R_k, t_k, _ = fit_pose(Pc, np.roll(Qc, -kc, axis=0), quiet=True)
            Pa_c = Pc @ R_k.T + t_k
            Mc = node_cost(Pa_c, TPc @ R_k.T, FPc, Qc, FQc, TQc, dist_scale(Pa_c, Qc))
            _, costs[kc] = dp_match_grid(closed_grid(Mc, kc), membrane, bending, mode="closed")
        return costs

    if fixed_offset_fraction is not None:
        if mode != "closed":
            raise ValueError("fixed_offset_fraction applies to closed curves only")
        if fixed_reversed and not both_orientations:
            raise ValueError("fixed_reversed=True requires both_orientations=True")
        orientations = [bool(fixed_reversed)]
    else:
        orientations = [False, True] if both_orientations else [False]
    orientation_cands = []
    offset_costs_list = []

    for rev in orientations:
        Q_o = Q_res[::-1].copy() if rev else Q_res.copy()
        FQ_o = FQ[:, ::-1].copy() if rev else FQ.copy()
        TQ_o = (-TQ_ms[:, ::-1, :]).copy() if rev else TQ_ms.copy()

        if mode == "open":
            R0, t0, flag0 = fit_pose(P_res, Q_o)
            cands = [solve_from(Q_o, FQ_o, TQ_o, R0, t0, 0, 0, flag0)]
        else:
            if fixed_offset_fraction is not None:
                centers, win = [int(round((fixed_offset_fraction % 1.0) * N)) % N], 0
            else:
                costs = coarse_offset_costs(Q_o, FQ_o, TQ_o)
                offset_costs_list.append(costs)
                order = np.argsort(costs)[: max(1, int(n_refine))]
                centers = [int(round(int(kc) * N / n_c)) % N for kc in order]
                win = int(np.ceil(N / n_c)) if n_c < N else 0
            cands = []
            for k_center in centers:
                R0, t0, flag0 = fit_pose(P_res, np.roll(Q_o, -k_center, axis=0), quiet=True)
                cands.append(solve_from(Q_o, FQ_o, TQ_o, R0, t0, k_center, win, flag0))

        cost_val, path, R_cur, t_cur, M_grid, chosen_offset, L, was_clamped = min(cands, key=lambda c: c[0])
        orientation_cands.append(
            (cost_val, rev, path, R_cur, t_cur, M_grid, Q_o, FQ_o, TQ_o, chosen_offset, L, was_clamped)
        )

    # Sort orientations by cost
    orientation_cands.sort(key=lambda item: item[0])
    best_cand = orientation_cands[0]
    cost_gap = float(orientation_cands[1][0] - orientation_cands[0][0]) if len(orientation_cands) > 1 else 0.0

    cost_val, rev, path, R_b, t_b, M_grid, Q_o, FQ_o, TQ_o, chosen_offset, L, was_clamped = best_cand
    Pa = P_res @ R_b.T + t_b
    TP_b = TP_ms @ R_b.T

    def _energy(map_):
        return _curve_energy_nd(
            map_, FP, FQ_o, TP_b, TQ_o, Pa, Q_o, L,
            feature_weight, tangent_weight, distance_weight, membrane, bending, power, mode=mode
        )

    # Sub-sample continuous phase refinement for closed loops: minimise the actual matching
    # energy (not a rigid-sample RMS) over a bounded shift of +-2 samples around the DP offset.
    subsample_theta = 0.0
    if mode == "closed":
        base_map = chosen_offset / N + path_to_map_subsample(path, N, N, mode=mode)
        res = minimize_scalar(
            lambda th: _energy(base_map + th)["total"],
            bounds=(-2.0 / N, 2.0 / N), method="bounded", options={"xatol": 1e-6 / N},
        )
        subsample_theta = float(res.x)

    if mode == "closed":
        t2_stair = chosen_offset / N + subsample_theta + path_to_map_subsample(path, N, N, mode=mode)
        ramp = chosen_offset / N + subsample_theta + np.arange(N) / N
    else:
        t2_stair = path_to_map_subsample(path, N, N, mode=mode)
        ramp = np.linspace(0.0, 1.0, N)

    resid = t2_stair - ramp
    E_base = _energy(ramp)

    # Candidate warps. The ramp is a candidate only when the grid contains an explicit None, so a
    # mismatch between this energy and the DP objective can never silently discard the DP map.
    best_sm = None
    for sm in smooth_sigma_grid:
        if sm is None:
            cand_map, E_cand = ramp.copy(), E_base
        else:
            if sm <= 0:
                r_s = resid.copy()
            elif mode == "closed":
                r_s = gaussian_filter1d(resid, sm, mode="wrap")
            else:
                r_pad = np.pad(resid, (1, 1), mode="constant", constant_values=0.0)
                r_s = gaussian_filter1d(r_pad, sm, mode="nearest")[1:-1]
                r_s[0] = 0.0
                r_s[-1] = 0.0
            cand_map = project_monotone_l2(ramp + r_s, min_slope=min_slope, max_slope=max_slope, mode=mode)
            E_cand = _energy(cand_map)
        if best_sm is None or E_cand["total"] < best_sm[0]["total"]:
            best_sm = (E_cand, cand_map, sm)

    if best_sm is None:  # empty grid: fall back to the projected DP map
        cand_map = project_monotone_l2(t2_stair, min_slope=min_slope, max_slope=max_slope, mode=mode)
        best_sm = (_energy(cand_map), cand_map, 0.0)

    E_final, t2_final, sm_final = best_sm

    t1 = np.linspace(0.0, 1.0, N) if mode == "open" else np.arange(N) / N
    if mode == "closed":
        t2_on_Q = np.mod((N - 1) / N - t2_final if rev else t2_final, 1.0)
    else:
        t2_on_Q = 1.0 - t2_final if rev else t2_final

    # Baseline and elastic RMS distance in R^d (signed gain: negative when the correct
    # correspondence moves features apart in space) and feature-space gain (curvature + tangent).
    baseline_Qm = _interp_points_nd(Q_o, ramp, mode=mode)
    baseline_rms = float(np.sqrt(((Pa - baseline_Qm) ** 2).sum(axis=1).mean()))
    elastic_rms = float(E_final["distance_rms_mm"])
    elastic_gain = baseline_rms - elastic_rms
    feat_base = E_base["feature"] + E_base["tangent"]
    feat_final = E_final["feature"] + E_final["tangent"]
    feature_gain = float(feat_base - feat_final)
    feature_gain_rel = float(feature_gain / feat_base) if feat_base > 1e-12 else 0.0

    inv_rms = None
    inv_rms_mm = None
    if bidirectional:
        # Match Q to P to check inverse consistency
        rev_match = match_curves(
            Q, P, mode=mode, n_points=n_points, sigmas_mm=sigmas_mm,
            sigmas_rel=sigmas_rel, sigma_mm=sigma_mm, distance_scale_mm=distance_scale_mm,
            distance_scale_rel=distance_scale_rel, dedupe_points=dedupe_points, presmooth_mm=presmooth_mm,
            n_coarse=n_coarse, n_refine=n_refine,
            feature_weight=feature_weight, tangent_weight=tangent_weight,
            distance_weight=distance_weight, membrane=membrane, bending=bending,
            power=power, rigid=rigid, rigid_iters=rigid_iters,
            max_rotation_deg=max_rotation_deg, allow_reflection=allow_reflection,
            both_orientations=both_orientations,
            smooth_sigma_grid=smooth_sigma_grid, min_slope=min_slope, max_slope=max_slope,
            bidirectional=False,
        )
        t1_test = t1
        t2_mapped = np.interp(t1_test, t1, t2_on_Q)
        t1_roundtrip = rev_match.map_P_to_Q(t2_mapped)
        if mode == "closed":
            diff = (t1_roundtrip - t1_test + 0.5) % 1.0 - 0.5
        else:
            diff = t1_roundtrip - t1_test
        inv_rms = float(np.sqrt(np.mean(diff ** 2)))
        inv_rms_mm = float(inv_rms * 0.5 * (LP + LQ))

    return CurveMatch(
        t1=t1,
        t2=t2_on_Q,
        t2_unwrapped=t2_final,
        offset_fraction=float((chosen_offset / N + subsample_theta) % 1.0),
        reversed=bool(rev),
        R=R_b,
        t=t_b,
        baseline_rms_mm=baseline_rms,
        elastic_rms_mm=elastic_rms,
        elastic_gain_mm=elastic_gain,
        sigma_selected_mm=sig_sel * spacing_mm,
        distance_scale_mm=float(L),
        spacing_mm=float(spacing_mm),
        energies=E_final,
        energies_baseline=E_base,
        smooth_sigma=float(sm_final if sm_final is not None else float("inf")),
        n_points=N,
        mode=mode,
        dimension=d,
        path=path,
        cost_field=M_grid,
        features_P=FP,
        features_Q_oriented=FQ_o,
        scales_mm=np.asarray([s * spacing_mm for s in use_sig], dtype=float),
        P_resampled=P_res,
        Q_resampled=Q_res,
        extrema_counts={"sigmas_mm": np.asarray(sigmas_mm, float), "F_P": F1, "F_Q": F2},
        inverse_consistency_rms=inv_rms,
        inverse_consistency_rms_mm=inv_rms_mm,
        rotation_clamped=bool(was_clamped),
        orientation_cost_gap=cost_gap,
        orientation_cost_gap_rel=float(cost_gap / max(abs(cost_val), 1e-9)),
        feature_gain=feature_gain,
        feature_gain_rel=feature_gain_rel,
        offset_costs=(np.stack(offset_costs_list) if offset_costs_list else None),
        offset_fractions=(np.arange(n_c) / n_c if offset_costs_list else None),
    )
