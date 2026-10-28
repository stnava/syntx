"""Statistical account of a motion-corrected time series: what the acquisition model explains, what motion explains, what
is left (docs/MOTION_NEXT_STEPS_PLAN_2026-10-28.md, step 1). Measurements only; no verdicts.

Two decompositions, both modality-agnostic:

- ``pose_step_decomposition``: the applied displacement of every volume along each acquisition axis is the sum of the head
  pose (what FD reports), an applied acquisition term (the eddy phase-encode shift of DWI) and a removed design-locked
  term (the level offset / l = 2 gauge the solver takes out of the pose). Share of the frame-to-frame step variance of each.
- ``image_variance_decomposition``: at foreground voxels, sequential sums of squares of the per-voxel time series against
  the acquisition design blocks (in order), then the motion regressors, then the residual; the residual's per-frame RMS
  against FD says whether what is left still follows the pose.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def real_even_sh(g: np.ndarray, lmax: int) -> np.ndarray:
    """Real symmetric spherical harmonics of even order 2..lmax of unit vectors ``g`` (N, 3), (N, K); K = 5 for lmax 2,
    14 for lmax 4. Rows with a zero vector (b0) are zero."""
    from scipy.special import sph_harm_y   # scipy >= 1.15 (n, m, polar, azimuth)

    g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(g, axis=1)
    gu = np.where(n[:, None] > 1e-9, g / np.maximum(n[:, None], 1e-9), 0.0)
    theta = np.arccos(np.clip(gu[:, 2], -1, 1))
    phi = np.arctan2(gu[:, 1], gu[:, 0])
    cols = []
    for l in range(2, int(lmax) + 1, 2):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            c = (np.sqrt(2.0) * (Y.imag if m < 0 else Y.real)) if m != 0 else Y.real
            cols.append(np.where(n > 1e-9, c, 0.0))
    return np.stack(cols, axis=1) if cols else np.zeros((len(g), 0))


def design_blocks(levels: np.ndarray, name: str, bvecs: Optional[np.ndarray] = None, bvals: Optional[np.ndarray] = None,
                  lmax: int = 4, min_frames_per_column: int = 6) -> List[Tuple[str, np.ndarray]]:
    """Ordered acquisition design blocks (name, (T, K)) for ``image_variance_decomposition``: one intercept per level;
    for ``shells`` the l = 2 and then the l = 4 direction terms per shell (each block only where the shell has
    ``min_frames_per_column`` frames per column)."""
    levels = np.asarray(levels, dtype=int)
    ks = sorted(set(levels.tolist()))
    blocks = [("level intercepts", np.stack([(levels == k).astype(float) for k in ks], axis=1))]
    if name == "shells" and bvecs is not None:
        prev = 0
        for lm in (2, 4, 6):
            if lm > lmax:
                break
            Y = real_even_sh(bvecs, lm)[:, prev:]
            prev += Y.shape[1]
            cols = []
            for k in ks:
                m = levels == k
                if bvals is not None and np.all(np.asarray(bvals)[m] < 50):
                    continue
                if m.sum() < min_frames_per_column * Y.shape[1]:
                    continue
                B = np.zeros_like(Y)
                B[m] = Y[m]
                if np.any(np.ptp(B, axis=0) > 0):
                    cols.append(B)
            if cols:
                blocks.append((f"l={lm} direction terms per shell", np.hstack(cols)))
    return blocks


def sequential_sums_of_squares(Y: np.ndarray, blocks: Sequence[Tuple[str, np.ndarray]]) -> Tuple[float, List[Tuple[str, float, int]]]:
    """``Y`` (T, V) with the per-voxel mean removed; blocks enter in order, each credited with what it adds.
    Returns the total SS and [(name, SS explained, columns), ..., ('residual', SS, dof)]."""
    T = Y.shape[0]
    tot = float((Y ** 2).sum())
    X = np.zeros((T, 0))
    out: List[Tuple[str, float, int]] = []
    prev = 0.0
    for name, B in blocks:
        B = np.asarray(B, dtype=np.float64).reshape(T, -1)
        X = np.c_[X, B]
        Q, R = np.linalg.qr(X)
        keep = np.abs(np.diag(R)) > 1e-10 * max(np.abs(np.diag(R)).max(), 1e-30)
        fit = Q[:, keep] @ (Q[:, keep].T @ Y)
        ss = float((fit ** 2).sum())
        out.append((name, ss - prev, int(B.shape[1])))
        prev = ss
    out.append(("residual", tot - prev, int(T - X.shape[1])))
    return tot, out


def image_variance_decomposition(series: np.ndarray, blocks: Sequence[Tuple[str, np.ndarray]], motion: Optional[np.ndarray],
                                 mask: np.ndarray, fd: Optional[np.ndarray] = None, n_voxels: int = 60000, seed: int = 0,
                                 frame_gain: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """``series`` (X, Y, Z, T) array, ``blocks`` from ``design_blocks``, ``motion`` (T, k) regressors (poses and FD) entered after
    the design, ``mask`` (X, Y, Z) bool. Returns total SS, the sequential fractions, the residual's per-frame RMS and its
    correlation with ``fd``."""
    rng = np.random.default_rng(seed)
    T = series.shape[-1]
    idx = np.flatnonzero(np.asarray(mask).reshape(-1))
    if idx.size == 0:
        raise ValueError("image_variance_decomposition: empty mask")
    idx = rng.choice(idx, size=min(int(n_voxels), idx.size), replace=False)
    Y = series.reshape(-1, T)[idx].T.astype(np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)
    gain_term = None
    if frame_gain is not None:
        # the solver's per-frame global gain: a KNOWN spatial pattern (the voxel's mean) times a known time course, removed
        # first and credited as its own term (it is not a free-pattern block: a free pattern on a slow time course would
        # take slow motion with it)
        gfac = 1.0 / np.asarray(frame_gain, dtype=np.float64).reshape(T)            # the drift the gain corrected
        mu = series.reshape(-1, T)[idx].T.astype(np.float64).mean(axis=0, keepdims=True)
        gain_term = (gfac[:, None] - gfac.mean()) * mu
        Y = Y - gain_term
    blocks = list(blocks)
    if motion is not None:
        M = np.asarray(motion, dtype=np.float64).reshape(T, -1)
        blocks.append(("motion regressors after the design", M - M.mean(axis=0, keepdims=True)))
    tot, ss = sequential_sums_of_squares(Y, blocks)
    if gain_term is not None:
        g_ss = float((gain_term ** 2).sum())
        tot = tot + g_ss
        ss = [("global gain drift (solver term)", g_ss, 1)] + ss
    design_only = [b for b in blocks if not b[0].startswith("motion")]
    X = np.hstack([np.asarray(B, dtype=np.float64).reshape(T, -1) for _, B in design_only])
    Q, _ = np.linalg.qr(X)
    R = Y - Q @ (Q.T @ Y)
    res_rms = np.sqrt((R ** 2).mean(axis=1))
    r = float("nan")
    if fd is not None and T > 3:
        f = np.asarray(fd, dtype=float)[1:]
        if np.std(f) > 0 and np.std(res_rms[1:]) > 0:
            r = float(np.corrcoef(res_rms[1:], f)[0, 1])
    return {"n_voxels": int(idx.size), "total_ss": tot,
            "terms": [{"term": n, "fraction": v / tot if tot > 0 else float("nan"), "columns": k} for n, v, k in ss],
            "residual_rms_per_frame": res_rms, "residual_rms_fd_corr": r}


def pose_step_decomposition(translations_mm: np.ndarray, rotations_deg: np.ndarray, direction: np.ndarray,
                            applied_pe_shift_mm: Optional[np.ndarray] = None, pe_axis: Optional[np.ndarray] = None,
                            removed_bias_mm_rad: Optional[np.ndarray] = None, rot_radius_mm: float = 50.0) -> Dict[str, Any]:
    """Per acquisition axis (columns of ``direction``: readout, phase encode, slice) and per rotation: the step SD (mm) of the
    total applied displacement and the share of its step variance held by the head pose, the applied acquisition term and
    the removed design-locked term. Shares are variance ratios of the components to the total and need not sum to one."""
    t = np.asarray(translations_mm, dtype=np.float64).reshape(-1, 3)
    T = t.shape[0]
    rot = np.radians(np.asarray(rotations_deg, dtype=np.float64).reshape(-1, 3)) * float(rot_radius_mm)
    D = np.asarray(direction, dtype=np.float64)[:3, :3]
    ed = np.zeros(T) if applied_pe_shift_mm is None else np.asarray(applied_pe_shift_mm, dtype=np.float64).reshape(-1)
    p = None if pe_axis is None else np.asarray(pe_axis, dtype=np.float64) / max(np.linalg.norm(pe_axis), 1e-12)
    bias = np.zeros((T, 6)) if removed_bias_mm_rad is None else np.asarray(removed_bias_mm_rad, dtype=np.float64).reshape(T, 6)
    out: Dict[str, Any] = {"axes": {}, "rotations": {}}

    def share(c, total):
        vt = float(np.var(np.diff(total)))
        return {"step_sd_mm": float(np.std(np.diff(c))), "share": float(np.var(np.diff(c))) / vt if vt > 0 else float("nan")}

    for j, name in enumerate(("readout", "phase_encode", "slice")):
        v = D[:, j]
        head = t @ v
        e = ed * float(p @ v) if p is not None else np.zeros(T)
        b = bias[:, :3] @ v
        total = head + e + b
        out["axes"][name] = {"total_step_sd_mm": float(np.std(np.diff(total))), "head": share(head, total),
                             "applied_acquisition_term": share(e, total), "removed_design_locked_term": share(b, total)}
    for j, name in enumerate(("x", "y", "z")):
        head = rot[:, j]
        b = bias[:, 3 + j] * float(rot_radius_mm)
        total = head + b
        out["rotations"][name] = {"total_step_sd_mm": float(np.std(np.diff(total))), "head": share(head, total),
                                  "removed_design_locked_term": share(b, total)}
    return out


def misattribution_angles(series: np.ndarray, blocks: Sequence[Tuple[str, np.ndarray]], spacing: Sequence[float],
                          direction: np.ndarray, center_of_mass: np.ndarray, mask: np.ndarray, n_voxels: int = 60000,
                          rot_radius_mm: float = 50.0, seed: int = 0) -> Dict[str, Any]:
    """Principal angles between the six rigid-derivative images of the series mean (what a pose change does to the
    images) and the design-locked spatial patterns (the per-point coefficients of every non-intercept design block fitted
    to the series). An angle of zero means a design-locked pattern is indistinguishable from a rigid motion: no metric can
    separate them and the pose absorbs it; ninety degrees means the design cannot be mistaken for motion. Returns the
    angles in degrees (ascending), the smallest, and its cos^2 (the misattribution fraction), per cumulative block set."""
    rng = np.random.default_rng(seed)
    T = series.shape[-1]
    mean = series.mean(axis=-1)
    sp = np.asarray(spacing, dtype=np.float64)[:3]
    D = np.asarray(direction, dtype=np.float64)[:3, :3]
    grads = np.gradient(mean, *sp)                                  # d/d(voxel axis i) in intensity per mm, voxel axes
    # physical gradient at every voxel: chain rule through the direction matrix (voxel-axis derivatives -> physical)
    gv = np.stack([g.reshape(-1) for g in grads], axis=1)         # (N, 3) along voxel axes
    gp = gv @ D.T                                                   # (N, 3) along physical axes
    idx = np.flatnonzero(np.asarray(mask).reshape(-1))
    idx = rng.choice(idx, size=min(int(n_voxels), idx.size), replace=False)
    shape = mean.shape
    ii = np.stack(np.unravel_index(idx, shape), axis=1).astype(np.float64)
    phys = (ii * sp) @ D.T                                          # origin irrelevant for the lever about the centre of mass
    com = np.asarray(center_of_mass, dtype=np.float64)
    rel = phys - (com - 0.0)
    J = np.zeros((idx.size, 6))
    J[:, :3] = gp[idx]
    for j in range(3):
        e = np.zeros(3); e[j] = 1.0
        J[:, 3 + j] = (gp[idx] * np.cross(e, rel)).sum(axis=1) / rot_radius_mm          # per (radian x lever / 50 mm): mm units
    Y = series.reshape(-1, T)[idx].T.astype(np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qj, _ = np.linalg.qr(J - J.mean(axis=0, keepdims=True))
    out: Dict[str, Any] = {"n_voxels": int(idx.size), "sets": []}
    X = np.zeros((T, 0)); names = []
    for name, B in blocks:
        B = np.asarray(B, dtype=np.float64).reshape(T, -1)
        X = np.c_[X, B]; names.append(name)
        if name.startswith("level intercepts"):
            continue
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]                 # (K, V)
        n_int = sum(np.asarray(b, dtype=np.float64).reshape(T, -1).shape[1] for n, b in blocks if n.startswith("level intercepts"))
        C = coef[n_int:].T                                          # (V, K_design) design-locked spatial patterns
        C = C - C.mean(axis=0, keepdims=True)
        Qc, Rc = np.linalg.qr(C)
        keep = np.abs(np.diag(Rc)) > 1e-10 * max(np.abs(np.diag(Rc)).max(), 1e-30)
        sv = np.linalg.svd(Qj.T @ Qc[:, keep], compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(sv, -1, 1)))
        out["sets"].append({"blocks": list(names), "angles_deg": sorted(float(a) for a in ang),
                            "min_angle_deg": float(ang.min()), "misattribution_fraction": float(np.max(sv) ** 2)})
    return out
