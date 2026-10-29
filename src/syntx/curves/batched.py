"""syntx.curves.batched — Vectorised open-curve elastic matching in R^d.

Vectorises the open-curve, non-rigid continuous scale-space matching equations of
``syntx.curves.match_curves`` (Avants & Gee 2003) across ``B`` curves simultaneously
(batch dimension 0), eliminating per-curve Python dispatch, ``scipy.ndimage.gaussian_filter1d``
loop overhead, ``cKDTree`` rebuilds, and scalar Thomas ADMM loops during along-tract
profiling, bundle medoid/centroid estimation, and medial-sheet outlier trimming.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from syntx.curves.matching import CurveMatch


def resample_open_curves_batch(
    curves: Sequence[np.ndarray] | np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Vectorised uniform-arclength resampling in ``float64`` matching ``resample_curve``."""
    N = int(n_points)
    if N < 2:
        raise ValueError(f"n_points must be >= 2; got {n_points}")

    if isinstance(curves, np.ndarray) and curves.ndim == 3:
        B, M, dim = curves.shape
        if B == 0:
            return np.zeros((0, N, dim), dtype=np.float64)
        c64 = np.asarray(curves, dtype=np.float64)
        if M == 1:
            return np.repeat(c64, N, axis=1)
        seg = np.linalg.norm(np.diff(c64, axis=1), axis=2)  # (B, M-1)
        cum = np.concatenate([np.zeros((B, 1), dtype=np.float64), np.cumsum(seg, axis=1)], axis=1)  # (B, M)
        tot = cum[:, -1:]
        deg = tot[:, 0] < 1e-12
        u_grid = np.linspace(0.0, 1.0, N, dtype=np.float64)[None, :] * np.maximum(tot, 1e-12)  # (B, N)
        stride = np.maximum(tot.max(), 1.0) * 4.0 + 1.0
        row_off = np.arange(B, dtype=np.float64)[:, None] * stride
        hi_flat = np.searchsorted((cum + row_off).ravel(), (u_grid + row_off).ravel(), side="right")
        b_ix = np.arange(B, dtype=np.int64)[:, None]
        hi = np.clip(hi_flat.reshape(B, N) - b_ix * M, 1, M - 1)
        lo = hi - 1
        c_lo = cum[b_ix, lo]
        c_hi = cum[b_ix, hi]
        alpha = np.clip((u_grid - c_lo) / np.maximum(c_hi - c_lo, 1e-30), 0.0, 1.0)[..., None]
        out = (1.0 - alpha) * c64[b_ix, lo] + alpha * c64[b_ix, hi]
        out[:, 0, :] = c64[:, 0, :]
        out[:, -1, :] = c64[:, -1, :]
        if np.any(deg):
            out[deg] = c64[deg, :1, :]
        return out

    c_list = [np.asarray(c, dtype=np.float64) for c in curves]
    B = len(c_list)
    if B == 0:
        return np.zeros((0, N, 3), dtype=np.float64)
    dim = int(c_list[0].shape[-1])
    lengths = np.fromiter((len(c) for c in c_list), dtype=np.int64, count=B)
    if np.all(lengths == lengths[0]):
        return resample_open_curves_batch(np.stack(c_list, axis=0), N)

    pts_cat = np.concatenate(c_list, axis=0)
    total_v = len(pts_cat)
    starts = np.zeros(B, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    ends = starts + lengths - 1
    diffs = np.diff(pts_cat, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    if B > 1:
        seg[ends[:-1]] = 0.0
    cum_g = np.empty(total_v, dtype=np.float64)
    cum_g[0] = 0.0
    np.cumsum(seg, out=cum_g[1:])
    sl_id = np.repeat(np.arange(B, dtype=np.int64), lengths)
    local_s = cum_g - cum_g[starts[sl_id]]
    tot = local_s[ends]
    deg = tot < 1e-12
    stride = float(np.max(tot) * 4.0 + 1.0)
    keys_v = sl_id.astype(np.float64) * stride + local_s
    u_frac = np.linspace(0.0, 1.0, N, dtype=np.float64)[None, :]
    q_s = u_frac * np.maximum(tot[:, None], 1e-12)
    q_keys = (np.arange(B, dtype=np.float64)[:, None] * stride + q_s).ravel()
    hi = np.searchsorted(keys_v, q_keys, side="right").reshape(B, N)
    hi = np.clip(hi, (starts + 1)[:, None], ends[:, None])
    lo = np.maximum(hi - 1, starts[:, None])
    s_lo = local_s[lo]
    s_hi = local_s[hi]
    alpha = np.clip((q_s - s_lo) / np.maximum(s_hi - s_lo, 1e-30), 0.0, 1.0)[..., None]
    out = (1.0 - alpha) * pts_cat[lo] + alpha * pts_cat[hi]
    out[:, 0, :] = pts_cat[starts]
    out[:, -1, :] = pts_cat[ends]
    if np.any(deg):
        out[deg] = pts_cat[starts[deg], None, :]
    return out


def _gaussian_toeplitz_kernels(sigmas: np.ndarray, n_pts: int) -> np.ndarray:
    """Build exact chord-detrended Gaussian smoothing matrices ``(*sigmas.shape, n_pts, n_pts)``.

    Matches ``syntx.curves.geometry.smooth_curve(..., mode='open')``: zero-padded boundary
    outside ``[0, n_pts - 1]`` with ``mode='nearest'`` on the 1-sample zero pad reduces to
    truncating the ``truncate=4.0`` Gaussian kernel to ``[0, n_pts - 1]`` while normalising by
    the full ``[-r, +r]`` discrete Gaussian sum.
    """
    sig_flat = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    k_num = len(sig_flat)
    out = np.zeros((k_num, n_pts, n_pts), dtype=np.float64)

    pos = sig_flat > 0.0
    if not np.all(pos):
        eye = np.eye(n_pts, dtype=np.float64)
        out[~pos] = eye

    if np.any(pos):
        sig_pos = sig_flat[pos]
        radii = np.floor(4.0 * sig_pos + 0.5).astype(np.int64)
        zero_r = radii <= 0
        if np.any(zero_r):
            pos_idx = np.flatnonzero(pos)
            out[pos_idx[zero_r]] = np.eye(n_pts, dtype=np.float64)

        valid_mask = radii > 0
        if np.any(valid_mask):
            pos_idx = np.flatnonzero(pos)[valid_mask]
            s_v = sig_pos[valid_mask]
            r_v = radii[valid_mask]
            max_r = int(r_v.max())

            lags = np.arange(-max_r, max_r + 1, dtype=np.float64)[None, :]
            in_win = np.abs(lags) <= r_v[:, None]
            w_lags = np.where(in_win, np.exp(-0.5 * (lags / s_v[:, None]) ** 2), 0.0)
            z_norm = np.maximum(w_lags.sum(axis=1), 1e-30)

            ii = np.arange(n_pts, dtype=np.int64)[:, None]
            jj = np.arange(n_pts, dtype=np.int64)[None, :]
            diff = (jj - ii)[None, :, :]
            in_band = np.abs(diff) <= r_v[:, None, None]
            W = np.where(in_band, np.exp(-0.5 * (diff / s_v[:, None, None]) ** 2), 0.0)
            W /= z_norm[:, None, None]
            out[pos_idx] = W

    return out.reshape((*sigmas.shape, n_pts, n_pts))


def smooth_open_curves_batch(curves: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Chord-detrended Gaussian smoothing of open curves ``(..., N, d)`` with per-item ``sigmas`` ``(...)``."""
    orig_shape = sigmas.shape
    n_pts = curves.shape[-2]
    dim = curves.shape[-1]
    c_b = np.broadcast_to(curves, (*orig_shape, n_pts, dim))
    c_flat = c_b.reshape(-1, n_pts, dim).astype(np.float64, copy=False)
    s_flat = np.asarray(sigmas, dtype=np.float64).reshape(-1)

    t = np.linspace(0.0, 1.0, n_pts, dtype=np.float64)[None, :, None]
    chord = (1.0 - t) * c_flat[:, :1, :] + t * c_flat[:, -1:, :]
    resid = c_flat - chord

    W = _gaussian_toeplitz_kernels(s_flat, n_pts)  # (M, N, N)
    res_sm = W @ resid  # (M, N, d)
    res_sm[:, 0, :] = 0.0
    res_sm[:, -1, :] = 0.0
    return (chord + res_sm).reshape((*orig_shape, n_pts, dim))


def open_tangents_and_curvature(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched unit tangents ``(..., N, d)`` and discrete curvature ``(..., N)`` for open curves."""
    n_pts = curves.shape[-2]
    T = np.empty_like(curves, dtype=np.float64)

    d0 = curves[..., 1, :] - curves[..., 0, :]
    n0 = np.maximum(np.linalg.norm(d0, axis=-1, keepdims=True), 1e-12)
    T[..., 0, :] = d0 / n0

    de = curves[..., -1, :] - curves[..., -2, :]
    ne = np.maximum(np.linalg.norm(de, axis=-1, keepdims=True), 1e-12)
    T[..., -1, :] = de / ne

    if n_pts > 2:
        dm = curves[..., 2:, :] - curves[..., :-2, :]
        nm = np.maximum(np.linalg.norm(dm, axis=-1, keepdims=True), 1e-12)
        T[..., 1:-1, :] = dm / nm

    if n_pts < 3:
        return T, np.zeros(curves.shape[:-1], dtype=np.float64)

    seg = np.linalg.norm(np.diff(curves, axis=-2), axis=-1)
    kappa = np.empty(curves.shape[:-1], dtype=np.float64)
    ds_int = np.maximum(seg[..., :-1] + seg[..., 1:], 1e-12)
    dT_int = np.linalg.norm(T[..., 2:, :] - T[..., :-2, :], axis=-1)
    kappa[..., 1:-1] = dT_int / ds_int
    kappa[..., 0] = kappa[..., 1]
    kappa[..., -1] = kappa[..., -2]
    return T, kappa


def _count_extrema_batch(kappa: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    """Count interior curvature maxima along the last axis ``(..., N)`` -> ``(...)`` int64."""
    n_pts = kappa.shape[-1]
    if n_pts < 3:
        return np.zeros(kappa.shape[:-1], dtype=np.int64)
    k_max = np.maximum(1.0, np.max(np.abs(kappa), axis=-1, keepdims=True))
    tol = rel_tol * k_max
    left = kappa[..., 1:-1] > (kappa[..., :-2] + tol)
    right = kappa[..., 1:-1] >= (kappa[..., 2:] - tol)
    return np.count_nonzero(left & right, axis=-1)


def _dp_match_open_batch(
    costs: np.ndarray,
    membrane: float,
    bending: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched anti-diagonal dynamic programming on ``(B, N, N)`` open-curve node costs."""
    bsz, n1, m1 = costs.shape
    n_max, m_max = n1 - 1, m1 - 1
    scale = 1.0 / max(n_max, m_max)
    D = np.full((bsz, 3, n1, m1), np.inf, dtype=np.float64)
    Bp = np.full((bsz, 3, n1, m1), -1, dtype=np.int8)
    D[:, :, 0, 0] = 0.0

    mem = float(membrane) * scale
    ben = float(bending) * scale
    bend_rows = (ben * (1.0 - np.eye(3, dtype=np.float64)))[None, :, :, None]
    coef = np.array([1.0, np.sqrt(2.0), 1.0], dtype=np.float64) * scale * 0.5
    steps = ((1, 0), (1, 1), (0, 1))
    ar = np.arange(max(n_max, m_max) + 1, dtype=np.int64)
    b_idx = np.arange(bsz, dtype=np.int64)[:, None]

    for d_sum in range(1, n_max + m_max + 1):
        i_lo = max(0, d_sum - m_max)
        i_hi = min(n_max, d_sum)
        n_d = i_hi - i_lo + 1
        ii = ar[i_lo : i_hi + 1]
        jj = d_sum - ii
        c_here = costs[:, ii, jj]

        for st in range(3):
            di, dj = steps[st]
            lo = 1 if (di == 1 and i_lo == 0) else 0
            hi = n_d - 1 if (dj == 1 and i_hi == d_sum) else n_d
            if lo >= hi:
                continue
            si, sj = ii[lo:hi], jj[lo:hi]
            ip, jp = si - di, sj - dj
            step_cost = coef[st] * (costs[:, ip, jp] + c_here[:, lo:hi])
            if st != 1:
                step_cost = step_cost + mem

            prev = D[:, :, ip, jp]
            tot = prev + bend_rows[:, st, :, :]
            if d_sum <= 2:
                at_origin = (ip == 0) & (jp == 0)
                if np.any(at_origin):
                    tot[:, :, at_origin] = prev[:, :, at_origin]
            k_best = np.argmin(tot, axis=1)
            col_idx = ar[: hi - lo][None, :]
            D[:, st, si, sj] = tot[b_idx, k_best, col_idx] + step_cost
            Bp[:, st, si, sj] = k_best.astype(np.int8)

    end_states = np.argmin(D[:, :, n_max, m_max], axis=1)
    b_flat = np.arange(bsz, dtype=np.int64)
    totals = D[b_flat, end_states, n_max, m_max]

    min_j = np.full((bsz, n1), m1 + 1, dtype=np.int64)
    max_j = np.full((bsz, n1), -1, dtype=np.int64)

    i_cur = np.full(bsz, n_max, dtype=np.int64)
    j_cur = np.full(bsz, m_max, dtype=np.int64)
    st_cur = end_states.copy()
    step_di = np.array([1, 1, 0], dtype=np.int64)
    step_dj = np.array([0, 1, 1], dtype=np.int64)

    min_j[b_flat, i_cur] = j_cur
    max_j[b_flat, i_cur] = j_cur

    for _ in range(n_max + m_max):
        active = (i_cur > 0) | (j_cur > 0)
        if not np.any(active):
            break
        act_b = b_flat[active]
        ic = i_cur[active]
        jc = j_cur[active]
        sc = st_cur[active]

        k_prev = Bp[act_b, sc, ic, jc].astype(np.int64)
        in_next = ic - step_di[sc]
        jn_next = jc - step_dj[sc]

        i_cur[active] = in_next
        j_cur[active] = jn_next
        st_cur[active] = k_prev

        min_j[act_b, in_next] = np.minimum(min_j[act_b, in_next], jn_next)
        max_j[act_b, in_next] = np.maximum(max_j[act_b, in_next], jn_next)

    denom = float(max(m1 - 1, 1))
    t2_stair = 0.5 * (min_j.astype(np.float64) + max_j.astype(np.float64)) / denom
    t2_stair[:, 0] = 0.0
    t2_stair[:, -1] = 1.0
    return t2_stair, totals


def _project_monotone_l2_open_batch(
    y: np.ndarray,
    min_slope: float = 0.45,
    max_slope: float | None = 2.2,
    max_iters: int = 80,
    rho: float = 2.0,
) -> np.ndarray:
    """Batched open-curve monotone L2 ADMM projection with pre-factored Thomas solve on ``(K, N)``."""
    k_num, n_pts = y.shape
    if n_pts < 3 or k_num == 0:
        return y.copy()

    dt1 = 1.0 / (n_pts - 1)
    lo = min(max(float(min_slope), 1e-4) * dt1, dt1)
    hi = max(float(max_slope) * dt1 if max_slope is not None else 1.0, dt1)

    m_int = n_pts - 2
    diag_val = 1.0 + 2.0 * rho
    off_val = -rho

    c_prime = np.zeros(max(m_int - 1, 0), dtype=np.float64)
    inv_denom = np.zeros(m_int, dtype=np.float64)
    inv_denom[0] = 1.0 / diag_val
    if m_int > 1:
        c_prime[0] = off_val * inv_denom[0]
        for i in range(1, m_int - 1):
            denom = max(diag_val - off_val * c_prime[i - 1], 1e-14)
            inv_denom[i] = 1.0 / denom
            c_prime[i] = off_val * inv_denom[i]
        inv_denom[-1] = 1.0 / max(diag_val - off_val * c_prime[-1], 1e-14)

    z = np.clip(np.diff(y, axis=1), lo, hi)
    z_sum = z.sum(axis=1, keepdims=True)
    z = np.where(z_sum > 0, z / np.where(z_sum > 0, z_sum, 1.0), z)
    u = np.zeros((k_num, n_pts - 1), dtype=np.float64)

    t = y.copy()
    t[:, 0] = 0.0
    t[:, -1] = 1.0
    alive = np.ones(k_num, dtype=bool)

    for _ in range(max_iters):
        idx_a = np.flatnonzero(alive)
        if len(idx_a) == 0:
            break
        n_a = len(idx_a)
        ya = y[idx_a]
        za = z[idx_a]
        ua = u[idx_a]

        w = za - ua
        rhs = ya[:, 1:-1] + rho * (w[:, :-1] - w[:, 1:])
        rhs[:, -1] += rho

        d_p = np.empty((n_a, m_int), dtype=np.float64)
        x_i = np.empty((n_a, m_int), dtype=np.float64)
        d_p[:, 0] = rhs[:, 0] * inv_denom[0]
        for i in range(1, m_int):
            d_p[:, i] = (rhs[:, i] - off_val * d_p[:, i - 1]) * inv_denom[i]
        x_i[:, -1] = d_p[:, -1]
        for i in range(m_int - 2, -1, -1):
            x_i[:, i] = d_p[:, i] - c_prime[i] * x_i[:, i + 1]

        ta = np.empty((n_a, n_pts), dtype=np.float64)
        ta[:, 0] = 0.0
        ta[:, -1] = 1.0
        ta[:, 1:-1] = x_i

        dt = np.diff(ta, axis=1)
        v_clip = np.clip(dt + ua, lo, hi)
        for _ in range(12):
            diff = 1.0 - v_clip.sum(axis=1, keepdims=True)
            active_rows = np.abs(diff[:, 0]) >= 1e-10
            if not np.any(active_rows):
                break
            free = np.where(diff > 0, v_clip < hi - 1e-9, v_clip > lo + 1e-9)
            n_free = free.sum(axis=1, keepdims=True)
            can_step = active_rows[:, None] & (n_free > 0) & free
            if not np.any(can_step):
                break
            step = diff / np.maximum(n_free, 1)
            v_clip = np.where(can_step, np.clip(v_clip + step, lo, hi), v_clip)

        res = dt - v_clip
        ua_new = ua + res
        t[idx_a] = ta
        z[idx_a] = v_clip
        u[idx_a] = ua_new
        done_now = np.max(np.abs(res), axis=1) < 1e-5
        alive[idx_a[done_now]] = False

    inc = np.clip(np.diff(t, axis=1), lo, hi)
    for _ in range(50):
        diff = 1.0 - inc.sum(axis=1, keepdims=True)
        active_rows = np.abs(diff[:, 0]) >= 1e-12
        if not np.any(active_rows):
            break
        free = np.where(diff > 0, inc < hi - 1e-12, inc > lo + 1e-12)
        n_free = free.sum(axis=1, keepdims=True)
        can_step = active_rows[:, None] & (n_free > 0) & free
        if not np.any(can_step):
            break
        step = diff / np.maximum(n_free, 1)
        inc = np.where(can_step, np.clip(inc + step, lo, hi), inc)

    out = np.empty_like(y)
    out[:, 0] = 0.0
    out[:, 1:] = np.cumsum(inc, axis=1)
    out[:, -1] = 1.0
    return out


def _open_curve_energy_batch(
    t2_maps: np.ndarray,
    FP: np.ndarray,
    FQ: np.ndarray,
    TP: np.ndarray,
    TQ: np.ndarray,
    n_scales: np.ndarray,
    Pa: np.ndarray,
    Qo: np.ndarray,
    L_scale: np.ndarray,
    feature_weight: float,
    tangent_weight: float,
    distance_weight: float,
    membrane: float,
    bending: float,
    power: float,
) -> dict[str, np.ndarray]:
    """Batched open-curve continuous energy evaluation matching ``_curve_energy_nd``."""
    bsz, n_cand, n_pts = t2_maps.shape
    idx = np.clip(t2_maps, 0.0, 1.0) * (n_pts - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, n_pts - 1)
    w = idx - i0
    b_ix = np.arange(bsz, dtype=np.int64)[:, None, None]

    Qm = Qo[b_ix, i0] * (1.0 - w[..., None]) + Qo[b_ix, i1] * w[..., None]

    FQ_ns = np.swapaxes(FQ, 1, 2)
    FQm = FQ_ns[b_ix, i0] * (1.0 - w[..., None]) + FQ_ns[b_ix, i1] * w[..., None]
    FP_ns = np.swapaxes(FP, 1, 2)[:, None, :, :]
    feat_diff = np.linalg.norm(FP_ns - FQm, axis=-1) ** power

    TQ_ns = np.swapaxes(TQ, 1, 2)
    w5 = w[..., None, None]
    TQm = TQ_ns[b_ix, i0] * (1.0 - w5) + TQ_ns[b_ix, i1] * w5
    TP_ns = np.swapaxes(TP, 1, 2)[:, None, :, :, :]
    dot_sum = np.sum(TP_ns * TQm, axis=-1).sum(axis=-1)
    cos_sim = np.clip(dot_sum / np.maximum(n_scales[:, None, None], 1.0), -1.0, 1.0)
    tangent_diff = 1.0 - cos_sim

    diff_sq = ((Pa[:, None, :, :] - Qm) ** 2).sum(axis=-1)
    dist = diff_sq / np.maximum(L_scale[:, None, None] ** 2, 1e-12)

    dt2 = np.diff(t2_maps, axis=-1)
    dt1 = 1.0 / max(n_pts - 1, 1)
    gdot = np.sqrt(dt1 * dt1 + dt2 * dt2)
    w_step = np.empty((bsz, n_cand, n_pts), dtype=np.float64)
    w_step[:, :, 0] = 0.5 * gdot[:, :, 0]
    w_step[:, :, -1] = 0.5 * gdot[:, :, -1]
    if n_pts > 2:
        w_step[:, :, 1:-1] = 0.5 * (gdot[:, :, :-1] + gdot[:, :, 1:])

    e_feat = feature_weight * np.sum(w_step * feat_diff, axis=-1)
    e_tangent = tangent_weight * np.sum(w_step * tangent_diff, axis=-1)
    e_dist = distance_weight * np.sum(w_step * dist, axis=-1)

    slope = dt2 / max(dt1, 1e-12)
    e_mem = membrane * np.mean((slope - 1.0) ** 2, axis=-1)
    if slope.shape[-1] > 1:
        curv = np.diff(slope, axis=-1)
        e_bend = bending * np.mean(curv**2, axis=-1)
    else:
        e_bend = np.zeros((bsz, n_cand), dtype=np.float64)

    slope_safe = np.maximum(slope, 1e-6)
    max_stretch = np.max(np.maximum(slope_safe, 1.0 / slope_safe), axis=-1)
    rms_mm = np.sqrt(diff_sq.mean(axis=-1))

    return {
        "feature": e_feat,
        "tangent": e_tangent,
        "distance": e_dist,
        "membrane": e_mem,
        "bending": e_bend,
        "total": e_feat + e_tangent + e_dist + e_mem + e_bend,
        "distance_rms_mm": rms_mm,
        "max_speed_stretch": max_stretch,
    }


_SUPPORTED_BATCH_KWARGS = frozenset(
    {
        "sigmas_mm",
        "sigmas_rel",
        "sigma_mm",
        "distance_scale_mm",
        "distance_scale_rel",
        "feature_weight",
        "tangent_weight",
        "distance_weight",
        "membrane",
        "bending",
        "power",
        "smooth_sigma_grid",
        "min_slope",
        "max_slope",
        "mode",
        "rigid",
        "bidirectional",
        "presmooth_mm",
        "dedupe_points",
    }
)


def can_batch_open_match(matching_kwargs: dict[str, Any]) -> bool:
    """Return True when ``matching_kwargs`` uses open, non-rigid, forward-only matching."""
    if not set(matching_kwargs).issubset(_SUPPORTED_BATCH_KWARGS):
        return False
    if matching_kwargs.get("mode", "open") != "open":
        return False
    if matching_kwargs.get("rigid", False):
        return False
    if matching_kwargs.get("bidirectional", False):
        return False
    if matching_kwargs.get("presmooth_mm", None) not in (None, 0, 0.0):
        return False
    if matching_kwargs.get("dedupe_points", False):
        return False
    return True


def match_open_curves_batch(
    P_curves: np.ndarray | Sequence[np.ndarray],
    Q_curves: Sequence[np.ndarray] | np.ndarray,
    *,
    n_points: int = 50,
    both_orientations: bool = True,
    sigmas_mm: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    sigmas_rel: Sequence[float] | None = None,
    sigma_mm: float | None = None,
    distance_scale_mm: float | None = None,
    distance_scale_rel: float = 0.3,
    feature_weight: float = 1.0,
    tangent_weight: float = 1.0,
    distance_weight: float = 1.0,
    membrane: float = 1.0,
    bending: float = 1.0,
    power: float = 0.5,
    smooth_sigma_grid: Sequence[float | None] = (0.0, 1.0, 2.0, 4.0),
    min_slope: float = 0.45,
    max_slope: float | None = 2.2,
    build_match_objects: bool = True,
) -> dict[str, Any]:
    """Match ``B`` open curves ``P_curves`` against ``Q_curves`` in R^d without Python curve loops.

    Parameters
    ----------
    P_curves : ndarray or sequence of ndarray
        Single shared reference curve ``(M, d)`` (broadcast across all ``B`` target curves)
        or a batch ``(B, M_i, d)`` of per-curve reference curves.
    Q_curves : sequence of ndarray or ndarray
        Batch of ``B`` open curves ``(B, K_i, d)`` to match against ``P_curves``.
    """
    N = int(n_points)
    Q_res = resample_open_curves_batch(Q_curves, N)
    B = Q_res.shape[0]
    dim = int(P_curves.shape[-1]) if isinstance(P_curves, np.ndarray) and P_curves.ndim >= 2 else int(Q_res.shape[-1])

    if B == 0:
        return {
            "matches": [],
            "warped_Q": np.zeros((0, N, dim), dtype=np.float32),
            "t2_unwrapped": np.zeros((0, N), dtype=np.float64),
            "reversed": np.zeros(0, dtype=bool),
            "elastic_rms_mm": np.zeros(0, dtype=np.float64),
            "feature_gain_rel": np.zeros(0, dtype=np.float64),
            "max_speed_stretch": np.zeros(0, dtype=np.float64),
            "bending_energy": np.zeros(0, dtype=np.float64),
            "tangent_energy": np.zeros(0, dtype=np.float64),
            "P_res": np.zeros((0, N, dim), dtype=np.float64),
            "Q_res": np.zeros((0, N, dim), dtype=np.float64),
        }

    shared_P = isinstance(P_curves, np.ndarray) and P_curves.ndim == 2
    if shared_P:
        P_single = resample_open_curves_batch(np.asarray(P_curves, dtype=np.float64)[None, :, :], N)[0]
        P_res = np.broadcast_to(P_single[None, :, :], (B, N, dim)).copy()
    else:
        P_res = resample_open_curves_batch(P_curves, N)
        dim = int(P_res.shape[-1])

    if Q_res.shape[-1] != dim:
        raise ValueError(f"Dimension mismatch between P ({dim}) and Q ({Q_res.shape[-1]})")

    LP = np.linalg.norm(np.diff(P_res, axis=1), axis=2).sum(axis=1)
    LQ = np.linalg.norm(np.diff(Q_res, axis=1), axis=2).sum(axis=1)
    spacing_mm = 0.5 * (LP + LQ) / max(N - 1, 1)

    if sigmas_rel is not None:
        sig_mm_arr = LP[:, None] * np.asarray(sigmas_rel, dtype=np.float64)[None, :]
    else:
        sig_mm_arr = np.broadcast_to(np.asarray(sigmas_mm, dtype=np.float64)[None, :], (B, len(sigmas_mm)))

    sigmas_pts = sig_mm_arr / np.maximum(spacing_mm[:, None], 1e-6)

    _, k0_P = open_tangents_and_curvature(P_res)
    _, k0_Q = open_tangents_and_curvature(Q_res)
    kbar_P = k0_P.sum(axis=1)
    kbar_Q = k0_Q.sum(axis=1)

    P_sm = smooth_open_curves_batch(P_res[:, None, :, :], sigmas_pts)
    Q_sm = smooth_open_curves_batch(Q_res[:, None, :, :], sigmas_pts)
    TP_all, k_P_all = open_tangents_and_curvature(P_sm)
    TQ_all, k_Q_all = open_tangents_and_curvature(Q_sm)

    F1 = _count_extrema_batch(k_P_all)
    F2 = _count_extrema_batch(k_Q_all)
    diff_ext = np.abs(F1 - F2)
    tol_ext = np.maximum(diff_ext.min(axis=1, keepdims=True), np.maximum(1.0, 0.25 * np.maximum(F1, F2)))
    best_s_idx = np.argmax(diff_ext <= tol_ext, axis=1)
    sig_sel = sigmas_pts[np.arange(B), best_s_idx]
    if sigma_mm is not None:
        sig_sel = float(sigma_mm) / np.maximum(spacing_mm, 1e-6)

    scale_mask = sigmas_pts >= (sig_sel[:, None] - 1e-12)
    none_selected = ~np.any(scale_mask, axis=1)
    if np.any(none_selected):
        idx_ns = np.flatnonzero(none_selected)
        P_extra = smooth_open_curves_batch(P_res[idx_ns, None, :, :], sig_sel[idx_ns, None])
        Q_extra = smooth_open_curves_batch(Q_res[idx_ns, None, :, :], sig_sel[idx_ns, None])
        TP_ex, kP_ex = open_tangents_and_curvature(P_extra)
        TQ_ex, kQ_ex = open_tangents_and_curvature(Q_extra)
        TP_all[idx_ns, -1:] = TP_ex
        TQ_all[idx_ns, -1:] = TQ_ex
        k_P_all[idx_ns, -1:] = kP_ex
        k_Q_all[idx_ns, -1:] = kQ_ex
        scale_mask[idx_ns, -1] = True

    n_scales_used = scale_mask.sum(axis=1).astype(np.float64)

    tot_kP = k_P_all.sum(axis=2)
    tot_kQ = k_Q_all.sum(axis=2)
    FP = np.where(
        (kbar_P[:, None, None] > 1e-12) & (tot_kP[:, :, None] > 1e-12) & scale_mask[:, :, None],
        k_P_all * (kbar_P[:, None, None] / np.maximum(tot_kP[:, :, None], 1e-12)) * LP[:, None, None],
        0.0,
    )
    FQ = np.where(
        (kbar_Q[:, None, None] > 1e-12) & (tot_kQ[:, :, None] > 1e-12) & scale_mask[:, :, None],
        k_Q_all * (kbar_Q[:, None, None] / np.maximum(tot_kQ[:, :, None], 1e-12)) * LQ[:, None, None],
        0.0,
    )
    denom_entries = np.maximum(n_scales_used * N, 1.0)
    feat_scale = np.sqrt(0.5 * ((FP**2).sum(axis=(1, 2)) / denom_entries + (FQ**2).sum(axis=(1, 2)) / denom_entries))
    nz_fs = feat_scale > 1e-12
    FP = np.where(nz_fs[:, None, None], FP / np.maximum(feat_scale[:, None, None], 1e-12), FP)
    FQ = np.where(nz_fs[:, None, None], FQ / np.maximum(feat_scale[:, None, None], 1e-12), FQ)

    TP_ms = np.where(scale_mask[:, :, None, None], TP_all, 0.0)
    TQ_ms = np.where(scale_mask[:, :, None, None], TQ_all, 0.0)

    d0_floor = np.maximum(float(distance_scale_rel) * 0.5 * (LP + LQ), spacing_mm)

    def _build_node_costs(Q_o: np.ndarray, FQ_o: np.ndarray, TQ_o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d2 = ((P_res[:, :, None, :] - Q_o[:, None, :, :]) ** 2).sum(axis=3)
        if distance_scale_mm is not None:
            L_x = np.maximum(float(distance_scale_mm), spacing_mm)
        else:
            nn_pq = d2.min(axis=2).mean(axis=1)
            nn_qp = d2.min(axis=1).mean(axis=1)
            L_x = np.maximum(np.sqrt(0.5 * (nn_pq + nn_qp)), d0_floor)

        FP_ns = np.swapaxes(FP, 1, 2)
        FQ_ns = np.swapaxes(FQ_o, 1, 2)
        m_feat = np.linalg.norm(FP_ns[:, :, None, :] - FQ_ns[:, None, :, :], axis=3) ** power

        dot_all = (TP_ms @ np.swapaxes(TQ_o, 2, 3)).sum(axis=1)
        cos_grid = np.clip(dot_all / np.maximum(n_scales_used[:, None, None], 1.0), -1.0, 1.0)
        costs = (
            float(feature_weight) * m_feat
            + float(tangent_weight) * (1.0 - cos_grid)
            + float(distance_weight) * d2 / np.maximum(L_x[:, None, None] ** 2, 1e-12)
        )
        return costs, L_x

    costs_fwd, L_fwd = _build_node_costs(Q_res, FQ, TQ_ms)
    if both_orientations:
        Q_rev = Q_res[:, ::-1, :].copy()
        FQ_rev = FQ[:, :, ::-1].copy()
        TQ_rev = (-TQ_ms[:, :, ::-1, :]).copy()
        costs_rev, L_rev = _build_node_costs(Q_rev, FQ_rev, TQ_rev)

        t2_both, dp_tot_both = _dp_match_open_batch(
            np.concatenate([costs_fwd, costs_rev], axis=0),
            membrane=membrane,
            bending=bending,
        )
        t2_fwd, t2_rev_s = t2_both[:B], t2_both[B:]
        tot_fwd, tot_rev = dp_tot_both[:B], dp_tot_both[B:]
        rev_flag = tot_rev < tot_fwd
        t2_stair = np.where(rev_flag[:, None], t2_rev_s, t2_fwd)
        dp_cost = np.where(rev_flag, tot_rev, tot_fwd)
        cost_gap = np.abs(tot_rev - tot_fwd)
        Q_o = np.where(rev_flag[:, None, None], Q_rev, Q_res)
        FQ_o = np.where(rev_flag[:, None, None], FQ_rev, FQ)
        TQ_o = np.where(rev_flag[:, None, None, None], TQ_rev, TQ_ms)
        L_best = np.where(rev_flag, L_rev, L_fwd)
    else:
        t2_stair, dp_cost = _dp_match_open_batch(costs_fwd, membrane=membrane, bending=bending)
        rev_flag = np.zeros(B, dtype=bool)
        cost_gap = np.zeros(B, dtype=np.float64)
        Q_o, FQ_o, TQ_o, L_best = Q_res, FQ, TQ_ms, L_fwd

    ramp = np.linspace(0.0, 1.0, N, dtype=np.float64)[None, :]
    resid = t2_stair - ramp

    E_base = _open_curve_energy_batch(
        np.broadcast_to(ramp[:, None, :], (B, 1, N)),
        FP,
        FQ_o,
        TP_ms,
        TQ_o,
        n_scales_used,
        P_res,
        Q_o,
        L_best,
        feature_weight,
        tangent_weight,
        distance_weight,
        membrane,
        bending,
        power,
    )

    grid_list = list(smooth_sigma_grid) if len(smooth_sigma_grid) > 0 else [0.0]
    n_grid = len(grid_list)
    cand_maps = np.empty((B, n_grid, N), dtype=np.float64)

    non_none_slots = [idx_g for idx_g, sm in enumerate(grid_list) if sm is not None]
    none_slots = [idx_g for idx_g, sm in enumerate(grid_list) if sm is None]
    for idx_g in none_slots:
        cand_maps[:, idx_g, :] = ramp

    if non_none_slots:
        sm_vals = np.array([float(grid_list[idx_g]) for idx_g in non_none_slots], dtype=np.float64)
        W_sm = _gaussian_toeplitz_kernels(sm_vals, N)
        r_smoothed = np.einsum("gij,bj->bgi", W_sm, resid)
        r_smoothed[:, :, 0] = 0.0
        r_smoothed[:, :, -1] = 0.0
        y_proj = (ramp[:, None, :] + r_smoothed).reshape(-1, N)
        t_proj = _project_monotone_l2_open_batch(
            y_proj,
            min_slope=min_slope,
            max_slope=max_slope,
        ).reshape(B, len(non_none_slots), N)
        for k_loc, idx_g in enumerate(non_none_slots):
            cand_maps[:, idx_g, :] = t_proj[:, k_loc, :]

    E_cands = _open_curve_energy_batch(
        cand_maps,
        FP,
        FQ_o,
        TP_ms,
        TQ_o,
        n_scales_used,
        P_res,
        Q_o,
        L_best,
        feature_weight,
        tangent_weight,
        distance_weight,
        membrane,
        bending,
        power,
    )
    best_g = np.argmin(E_cands["total"], axis=1)
    b_ar = np.arange(B, dtype=np.int64)
    t2_final = cand_maps[b_ar, best_g, :]
    t2_on_Q = np.where(rev_flag[:, None], 1.0 - t2_final, t2_final)

    idx_q = np.clip(t2_on_Q, 0.0, 1.0) * (N - 1)
    iq0 = np.floor(idx_q).astype(np.int64)
    iq1 = np.clip(iq0 + 1, 0, N - 1)
    wq = (idx_q - iq0)[..., None]
    b_ix2 = b_ar[:, None]
    warped_Q = (Q_res[b_ix2, iq0] * (1.0 - wq) + Q_res[b_ix2, iq1] * wq).astype(np.float32)

    elastic_rms = E_cands["distance_rms_mm"][b_ar, best_g]
    baseline_rms = E_base["distance_rms_mm"][:, 0]
    elastic_gain = baseline_rms - elastic_rms
    feat_base = E_base["feature"][:, 0] + E_base["tangent"][:, 0]
    feat_final = E_cands["feature"][b_ar, best_g] + E_cands["tangent"][b_ar, best_g]
    feat_gain = feat_base - feat_final
    feat_gain_rel = np.where(feat_base > 1e-12, feat_gain / np.maximum(feat_base, 1e-12), 0.0)
    max_stretch = E_cands["max_speed_stretch"][b_ar, best_g]
    bend_energy = E_cands["bending"][b_ar, best_g]
    tang_energy = E_cands["tangent"][b_ar, best_g]

    matches: list[CurveMatch] = []
    if build_match_objects:
        t1_shared = np.linspace(0.0, 1.0, N, dtype=np.float64)
        eye_d = np.eye(dim, dtype=np.float64)
        zero_d = np.zeros(dim, dtype=np.float64)
        empty_path = np.zeros((0, 2), dtype=int)
        for b in range(B):
            g = int(best_g[b])
            sm_chosen = grid_list[g]
            e_fin_b = {k: float(v[b, g]) for k, v in E_cands.items()}
            e_bas_b = {k: float(v[b, 0]) for k, v in E_base.items()}
            m_obj = CurveMatch(
                t1=t1_shared,
                t2=t2_on_Q[b],
                t2_unwrapped=t2_final[b],
                offset_fraction=0.0,
                reversed=bool(rev_flag[b]),
                R=eye_d,
                t=zero_d,
                baseline_rms_mm=float(baseline_rms[b]),
                elastic_rms_mm=float(elastic_rms[b]),
                elastic_gain_mm=float(elastic_gain[b]),
                sigma_selected_mm=float(sig_sel[b] * spacing_mm[b]),
                distance_scale_mm=float(L_best[b]),
                spacing_mm=float(spacing_mm[b]),
                energies=e_fin_b,
                energies_baseline=e_bas_b,
                smooth_sigma=float(sm_chosen if sm_chosen is not None else float("inf")),
                n_points=N,
                mode="open",
                dimension=dim,
                path=empty_path,
                P_resampled=P_res[b],
                Q_resampled=Q_res[b],
                orientation_cost_gap=float(cost_gap[b]),
                orientation_cost_gap_rel=float(cost_gap[b] / max(abs(float(dp_cost[b])), 1e-9)),
                feature_gain=float(feat_gain[b]),
                feature_gain_rel=float(feat_gain_rel[b]),
            )
            m_obj.max_speed_stretch = float(max_stretch[b])
            matches.append(m_obj)

    return {
        "matches": matches,
        "warped_Q": warped_Q,
        "t2_unwrapped": t2_final,
        "reversed": rev_flag,
        "elastic_rms_mm": elastic_rms,
        "feature_gain_rel": feat_gain_rel,
        "max_speed_stretch": max_stretch,
        "bending_energy": bend_energy,
        "tangent_energy": tang_energy,
        "P_res": P_res,
        "Q_res": Q_res,
    }
