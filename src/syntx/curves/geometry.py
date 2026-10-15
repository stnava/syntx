"""Geometric operations and scale-space features for N-dimensional curves.

Supports curves in R^d for arbitrary d >= 1 (e.g. 2-D planar contours, 3-D space
curves, and higher-dimensional trajectories). Supports both open curves (with
strictly preserved physical endpoints) and closed curves (with cyclic boundary
conditions).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.ndimage import gaussian_filter1d


def as_curve_array(curve: np.ndarray | Sequence, mode: str = "open") -> np.ndarray:
    """Validate and return curve as an ``(N, d)`` float array.

    For ``mode='closed'``, an identical closure point (``curve[0] == curve[-1]``)
    is stripped on input. For ``mode='open'``, points are kept verbatim.
    """
    P = np.asarray(curve, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"Curve must be a 2-D array (N, d); got shape {P.shape}")
    min_pts = 3 if mode == "closed" else 2
    if P.shape[0] < min_pts:
        raise ValueError(f"Curve in mode '{mode}' requires at least {min_pts} points; got {P.shape[0]}")
    if mode == "closed" and np.allclose(P[0], P[-1]):
        P = P[:-1]
    return P


def curve_segment_lengths(curve: np.ndarray, mode: str = "open") -> np.ndarray:
    """Euclidean chord lengths between successive points in R^d.

    If ``mode='open'``, returns ``N - 1`` segment lengths.
    If ``mode='closed'``, returns ``N`` segment lengths (including closing segment).
    """
    P = as_curve_array(curve, mode=mode)
    if mode == "closed":
        return np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    return np.linalg.norm(np.diff(P, axis=0), axis=1)


def curve_length(curve: np.ndarray, mode: str = "open") -> float:
    """Total physical arc length along the polyline in R^d."""
    return float(np.sum(curve_segment_lengths(curve, mode=mode)))


def curve_arc_fraction(curve: np.ndarray, mode: str = "open") -> np.ndarray:
    """Normalized arc-length fraction along the curve.

    For ``mode='open'``, returns array of length ``N`` with values in ``[0.0, 1.0]``,
    where point 0 is 0.0 and point N-1 is 1.0.
    For ``mode='closed'``, returns array of length ``N`` with values in ``[0.0, 1.0)``,
    measured cyclically from point 0.
    """
    seg = curve_segment_lengths(curve, mode=mode)
    total = float(np.sum(seg))
    if total <= 0:
        raise ValueError("Curve has zero length")
    if mode == "closed":
        return np.concatenate([[0.0], np.cumsum(seg[:-1])]) / total
    return np.concatenate([[0.0], np.cumsum(seg)]) / total


def resample_curve(curve: np.ndarray, n_points: int, mode: str = "open") -> np.ndarray:
    """Resample curve in R^d to ``n_points`` equispaced along physical arc length.

    For ``mode='open'``:
        Endpoints ``curve[0]`` and ``curve[-1]`` in R^d are preserved exactly.
    For ``mode='closed'``:
        Closed loop is resampled into ``n_points`` cyclically equispaced points,
        with ``curve[0]`` fixed.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2; got {n_points}")
    P = as_curve_array(curve, mode=mode)
    d = P.shape[1]

    if mode == "closed":
        Pc = np.vstack([P, P[:1]])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))])
        total = s[-1]
        if total <= 0:
            raise ValueError("Curve has zero length")
        q = np.linspace(0.0, total, n_points, endpoint=False)
        return np.column_stack([np.interp(q, s, Pc[:, k]) for k in range(d)])

    # Open curve in R^d
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(P[:1], n_points, axis=0)
    q = np.linspace(0.0, total, n_points)
    out = np.column_stack([np.interp(q, s, P[:, k]) for k in range(d)])
    # Enforce exact endpoint preservation in R^d
    out[0] = P[0]
    out[-1] = P[-1]
    return out


def smooth_curve(curve: np.ndarray, sigma_pts: float, mode: str = "open") -> np.ndarray:
    """Gaussian-smooth curve coordinates index-wise in R^d.

    For ``mode='closed'``:
        Uses cyclic wrapping (``mode='wrap'``).
    For ``mode='open'``:
        Detrends the linear chord connecting endpoints in R^d, applies Gaussian
        smoothing to the residual with zero boundary padding, and restores the chord.
        This guarantees strict endpoint preservation with zero boundary shrinkage.
    """
    P = as_curve_array(curve, mode=mode)
    if sigma_pts <= 0:
        return P.copy()
    if mode == "closed":
        return gaussian_filter1d(P, sigma=float(sigma_pts), axis=0, mode="wrap")

    # Open curve in R^d
    n = len(P)
    t = np.linspace(0.0, 1.0, n)[:, None]
    linear_chord = (1.0 - t) * P[0] + t * P[-1]
    residual = P - linear_chord

    # Zero-padded residual smoothing
    res_padded = np.pad(residual, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
    res_smooth = gaussian_filter1d(res_padded, sigma=float(sigma_pts), axis=0, mode="nearest")[1:-1]
    res_smooth[0] = 0.0
    res_smooth[-1] = 0.0
    return linear_chord + res_smooth


def curve_tangents(curve: np.ndarray, mode: str = "open") -> np.ndarray:
    """Unit tangent vector field T(s) in S^(d-1) for an N-D curve, shape (N, d).

    Computed via central differences in the interior, and forward/backward
    differences at boundaries for open curves.
    """
    P = as_curve_array(curve, mode=mode)
    N = len(P)
    if mode == "closed":
        d_f = np.roll(P, -1, axis=0) - P
        d_b = P - np.roll(P, 1, axis=0)
        n_f = np.linalg.norm(d_f, axis=1, keepdims=True)
        n_b = np.linalg.norm(d_b, axis=1, keepdims=True)
        T_f = np.divide(d_f, np.maximum(n_f, 1e-12))
        T_b = np.divide(d_b, np.maximum(n_b, 1e-12))
        T = 0.5 * (T_f + T_b)
        n_T = np.linalg.norm(T, axis=1, keepdims=True)
        return np.divide(T, np.maximum(n_T, 1e-12))

    # Open curve in R^d
    T = np.empty_like(P)
    d0 = P[1] - P[0]
    T[0] = d0 / max(np.linalg.norm(d0), 1e-12)
    d_end = P[-1] - P[-2]
    T[-1] = d_end / max(np.linalg.norm(d_end), 1e-12)
    if N > 2:
        d_mid = P[2:] - P[:-2]
        n_mid = np.linalg.norm(d_mid, axis=1, keepdims=True)
        T[1:-1] = np.divide(d_mid, np.maximum(n_mid, 1e-12))
    return T


def curve_curvature(curve: np.ndarray, mode: str = "open") -> np.ndarray:
    """Discrete intrinsic curvature kappa(s) = ||dT/ds|| in R for an N-D curve.

    Measures the magnitude of the rate of change of the unit tangent vector with
    respect to physical arc length: kappa = ||dT / ds|| >= 0.
    """
    P = as_curve_array(curve, mode=mode)
    N, d = P.shape
    if d == 1 or N < 3:
        return np.zeros(N, dtype=np.float64)

    if mode == "closed":
        d_f = np.roll(P, -1, axis=0) - P
        d_b = P - np.roll(P, 1, axis=0)
        n_f = np.linalg.norm(d_f, axis=1)
        n_b = np.linalg.norm(d_b, axis=1)
        if np.any(n_f <= 0) or np.any(n_b <= 0):
            raise ValueError("Curve contains coincident consecutive points")
        T_f = d_f / n_f[:, None]
        T_b = d_b / n_b[:, None]
        ds = 0.5 * (n_f + n_b)
        return np.linalg.norm(T_f - T_b, axis=1) / np.maximum(ds, 1e-12)

    # Open curve in R^d (d >= 2)
    T = curve_tangents(P, mode="open")
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    kappa = np.empty(N, dtype=np.float64)

    # Interior central differences: T[2:] - T[:-2] spans interval s[i+1] - s[i-1] = seg[i-1] + seg[i]
    ds_interior = seg[:-1] + seg[1:]
    dT_interior = np.linalg.norm(T[2:] - T[:-2], axis=1)
    kappa[1:-1] = dT_interior / np.maximum(ds_interior, 1e-12)

    # Boundary curvature extrapolated from adjacent interior to avoid endpoint truncation artifact
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def curvature_scale_space(
    curve: np.ndarray,
    sigmas: Sequence[float],
    mode: str = "open",
) -> np.ndarray:
    """Curvature-conservative scale-space image in R^d (Avants & Gee 2003 eq. 6).

    Returns array of shape ``(len(sigmas), N)``:
    ``Pi[s, i] = kappa_sigma(i) * sum(kappa_0) / sum(kappa_sigma)``
    such that every scale conserves the same total curvature as the base curve.
    """
    P = as_curve_array(curve, mode=mode)
    k0 = curve_curvature(P, mode=mode)
    kbar = float(np.sum(k0))
    if kbar <= 1e-12:
        return np.zeros((len(sigmas), len(P)), dtype=np.float64)

    out = np.empty((len(sigmas), len(P)), dtype=np.float64)
    for s_idx, sig in enumerate(sigmas):
        k_s = curve_curvature(smooth_curve(P, sig, mode=mode), mode=mode)
        tot = float(np.sum(k_s))
        if tot <= 1e-12:
            out[s_idx] = 0.0
        else:
            out[s_idx] = k_s * (kbar / tot)
    return out


def tangent_scale_space(
    curve: np.ndarray,
    sigmas: Sequence[float],
    mode: str = "open",
) -> np.ndarray:
    """Multi-scale unit tangent vector field of shape ``(len(sigmas), N, d)``."""
    P = as_curve_array(curve, mode=mode)
    d = P.shape[1]
    out = np.empty((len(sigmas), len(P), d), dtype=np.float64)
    for s_idx, sig in enumerate(sigmas):
        out[s_idx] = curve_tangents(smooth_curve(P, sig, mode=mode), mode=mode)
    return out


def count_curvature_extrema(
    kappa: np.ndarray,
    mode: str = "open",
    rel_tol: float = 1e-9,
) -> int:
    """Count local maxima of 1-D curvature profile."""
    k = np.asarray(kappa, dtype=np.float64)
    if len(k) < 3:
        return 0
    tol = rel_tol * max(1.0, float(np.abs(k).max()))
    if mode == "closed":
        left = k > np.roll(k, 1) + tol
        right = k >= np.roll(k, -1) - tol
        return int(np.count_nonzero(left & right))

    # Open curve: interior extrema only
    left = k[1:-1] > k[:-2] + tol
    right = k[1:-1] >= k[2:] - tol
    return int(np.count_nonzero(left & right))


def select_matching_scale(
    curve1: np.ndarray,
    curve2: np.ndarray,
    sigmas: Sequence[float],
    mode: str = "open",
    rel_tol: float = 0.25,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Select the scale matching curvature extrema topology (Avants & Gee 2003 sec 3.4).

    Returns ``(sigma, F1, F2)`` where ``F_k[s]`` is the extrema count of curve k at ``sigmas[s]``.
    ``sigma`` is the *finest* scale whose count mismatch ``|F1 - F2|`` is within
    ``max(min_mismatch, 1, rel_tol * max(F1[s], F2[s]))``. The previous rule (finest scale attaining
    the global minimum mismatch) is recovered with ``rel_tol=0``; it was brittle because counts agree
    trivially at coarse scales where all features are blurred away, so a single noisy count at a fine
    scale pushed the selection to a scale that erased the features being matched.
    """
    sig = np.asarray(sigmas, dtype=np.float64)
    F1 = np.array([
        count_curvature_extrema(curve_curvature(smooth_curve(curve1, s, mode=mode), mode=mode), mode=mode)
        for s in sig
    ])
    F2 = np.array([
        count_curvature_extrema(curve_curvature(smooth_curve(curve2, s, mode=mode), mode=mode), mode=mode)
        for s in sig
    ])
    diff = np.abs(F1 - F2)
    if rel_tol <= 0:
        best = int(np.flatnonzero(diff == diff.min())[0])
    else:
        tol = np.maximum(float(diff.min()), np.maximum(1.0, rel_tol * np.maximum(F1, F2)))
        best = int(np.flatnonzero(diff <= tol)[0])
    return float(sig[best]), F1, F2
