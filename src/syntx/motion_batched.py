"""
Batched rigid registration of many 3-D frames to one shared reference, for motion correction.

All frames are optimised together: one forward / backward pass per optimisation step covers
the whole series (per-frame rotation and translation parameters, per-frame losses summed),
instead of one registration call per frame. Used by ``syntx.motion.motion_correction`` with
``backend='pytorch_batched'``.

- ``batched_rigid_register_pass``: independent rigid transform per frame.
- ``batched_group_bias_register_pass``: per-frame rigid transforms plus one shared rigid
  "group bias" transform applied to a subset of frames (e.g. DWI frames vs b0 frames).

Similarity: Mattes mutual information (cubic B-spline Parzen windows) on a fixed random
sample of foreground reference voxels, optionally plus negative Pearson correlation at the
coarse stages. Rotations are axis-angle (Rodrigues) about the reference's intensity-weighted
centre of mass. Results are written as ANTs ``AffineTransform`` ``.mat`` files.

Limitations: 3-D only (``NotImplementedError`` otherwise); all frames must share the
reference's grid (ValueError otherwise); no mask support. The L-BFGS stages use strong-Wolfe
line search with lr 1.0 (no per-parameter rates). Background and benchmark results (speed and
accuracy versus ``ants.registration``) are in
``docs/SESSION_2026-09-25_PHASE_CORRELATION_AND_BATCHED_MOTION_CORRECTION.md`` and the
prototypes ``scripts/prototype_batched_motion_correction.py`` /
``scripts/prototype_batched_joint_group_bias.py``.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from typing import Any, List, NamedTuple, Optional, Tuple

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .robust_affine import compute_center_of_mass
from .sampling import DEFAULT_SAMPLING
from .core.losses import batched_mattes_mi_from_weights, huber_loss_nd, parzen_weights, pearson_correlation_loss
from .core.optimizers import PointStepTracker
from .core.parameter_scales import probe_manifold, rigid_axes, rotation_axis_scale
from .pyramid import build_pyramid, prebuild_warp
from .core.grid import grid_sample_nd, _image_spatial_gradient


def _batched_rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: axis-angle vectors omega (B, 3), radians, -> rotation matrices (B, 3, 3).
    The angle is clamped to >= 1e-8, so omega = 0 gives the identity."""
    theta = torch.norm(omega, dim=1, keepdim=True).clamp_min(1e-8)
    u = omega / theta
    zeros = torch.zeros_like(u[:, 0])
    K = torch.stack([
        torch.stack([zeros, -u[:, 2], u[:, 1]], dim=1),
        torch.stack([u[:, 2], zeros, -u[:, 0]], dim=1),
        torch.stack([-u[:, 1], u[:, 0], zeros], dim=1),
    ], dim=1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1.0 - cos_t) * torch.bmm(K, K)


def _auto_device() -> str:
    """Central resolver (explicit > env > cuda > mps > cpu); see ``syntx.device``."""
    from .device import resolve_device

    return resolve_device(None, task="rigid_motion")


def _check_frames_on_reference(reference_img, moving_imgs):
    """The batched passes sample every frame on the reference grid: ValueError otherwise."""
    for i, img in enumerate(moving_imgs):
        if (tuple(img.shape) != tuple(reference_img.shape)
                or not np.allclose(img.spacing, reference_img.spacing)
                or not np.allclose(img.origin, reference_img.origin)
                or not np.allclose(img.direction, reference_img.direction)):
            raise ValueError(f"frame {i} is not on the reference grid (shape / spacing / origin / "
                             "direction); the batched motion passes need a shared grid")


# Empirically confirmed on real 356-frame FDG-PET data: a single batched call over ALL
# frames at once (moving_stack shape (B, Z, Y, X)) crashed inside _image_spatial_gradient's
# torch.gradient call with "RuntimeError: value cannot be converted to type dest_t without
# overflow" at B=355 frames of a head-sized PET volume -- consistent with MPS's gradient
# kernel overflowing a 32-bit element-index internally once B * volume_numel gets large
# enough (355 frames x a few hundred-thousand-voxel volume lands well past 1e9 elements).
# This cap keeps any single internal batch safely below that regardless of volume size, by
# splitting long series into chunks and registering each chunk to the same reference
# independently -- the per-frame optimisation has no cross-frame coupling (each frame's
# rigid parameters are solved independently against the shared `reference_img`), so
# chunking changes nothing about the result, only how much GPU memory/tensor size a single
# call touches at once.
_MAX_BATCH_TENSOR_ELEMENTS = 1_500_000_000


# Optimiser step sizes are stated as the DISPLACEMENT OF THE SAMPLED POINTS PER STEP, in voxels of the finest working grid
# (``step_t_vox`` for translations, ``step_r_vox`` for rotations), and converted to parameter units with the physical-shift
# scales of ``core.parameter_scales`` (a rotation of 1 rad moves the points by their RMS distance from the axis). One schedule
# therefore behaves the same at any image spacing and for any object size in the field of view. The numbers are the former
# per-parameter constants (mm, rad) re-expressed at the geometry they were tuned on (2 mm voxels, 60 mm RMS perpendicular
# radius): 0.08 mm -> 0.04 voxel, 0.04 rad -> 1.2 voxel.
# The remaining Newton distance is confirmed against this multiple of the step tolerance: the sampled objective (15 % of the
# points) cannot resolve its own optimum better than ~0.1 voxel (per-volume error vs truth ~0.2 voxel on the noisy phantoms).
_CONFIRM_FACTOR = 3.0
_CHECK_EVERY_ADAM = 10
# Fraction of the full-resolution sample pool each default-schedule stage draws per metric evaluation. The pool is sized
# ``sample_fraction / _STAGE_SAMPLING`` of the head mask so that a stage evaluates ``sample_fraction`` of the head
# (pool 50 % and stage 20 % -> 10 %); with ``resample_points`` the stage subset is re-drawn at every convergence checkpoint.
_STAGE_SAMPLING = 0.2
_CHECK_EVERY_LBFGS = 5
_N_CONVERGENCE_POINTS = 2000


def build_encoding_covariates(
    bvals: Optional[np.ndarray] = None,
    bvecs: Optional[np.ndarray] = None,
    state_labels: Optional[np.ndarray] = None,
    encoding_covariates: Optional[np.ndarray] = None,
    num_frames: Optional[int] = None,
    window_size: int = 16,
    window_stride: Optional[int] = None,
    b0_threshold: float = 50.0,
) -> dict:
    """Build an orthonormalized encoding basis E (T, K_eff), its second-order surround-difference
    basis E_surround (T-2, K_eff), and partition-of-unity overlapping Hann window weights psi (T, M).

    Handles:
    - DWI continuous encoding from ``bvals`` and ``bvecs`` (normalized ``b_tilde = b / max(b)``,
      ``sqrt(b_tilde) * g``, and ``b_tilde * vec6(g g^T)``),
    - Discrete alternating or multi-state labels via ``state_labels`` (e.g. ASL control/label or
      SlowFlow velocity/diffusion states),
    - Arbitrary continuous/discrete covariates via ``encoding_covariates`` (or passed positionally
      as the first argument when ``bvecs`` is None).
    """
    cols = []
    T = int(num_frames) if num_frames is not None else None

    if bvals is not None and bvecs is None and state_labels is None and encoding_covariates is None:
        arr0 = np.asarray(bvals, dtype=np.float64)
        if arr0.ndim == 2 or np.any(arr0 < 0.0):
            encoding_covariates = arr0
            bvals = None

    if encoding_covariates is not None:
        try:
            ec = np.asarray(encoding_covariates, dtype=np.float64)
        except (ValueError, TypeError):  # non-numeric (string) state labels
            ec = None
            if state_labels is None:
                state_labels = np.asarray(encoding_covariates).reshape(-1)
        if ec is not None:
            if ec.ndim == 1:
                ec = ec[:, None]
            if T is None:
                T = int(ec.shape[0])
            elif ec.shape[0] != T:
                raise ValueError(f"encoding_covariates has {ec.shape[0]} rows for {T} frames")
            # A single integer-valued column with more than two distinct values is a categorical STATE label
            # (slow flow [0, 1, 2, 3]): one indicator column per state, never one numeric ramp 0..3 (which would claim the
            # bias of state 3 is three times that of state 1). Two-valued columns span the same space either way.
            categorical = (ec.shape[1] == 1 and bool(np.all(np.isfinite(ec))) and np.allclose(ec, np.round(ec))
                           and len(np.unique(ec)) > 2)
            if categorical and state_labels is None:
                state_labels = ec[:, 0].astype(np.int64)
            else:
                cols.append(ec)

    if state_labels is not None:
        lbl = np.asarray(state_labels)
        if T is None:
            T = int(lbl.shape[0])
        elif lbl.shape[0] != T:
            raise ValueError(f"state_labels has {lbl.shape[0]} entries for {T} frames")
        uniq = np.unique(lbl)
        if len(uniq) == 2:
            cols.append(np.where(lbl == uniq[0], 1.0, -1.0).astype(np.float64)[:, None])
        elif len(uniq) > 2:
            cols.append(np.column_stack([(lbl == u).astype(np.float64) for u in uniq[1:]]))

    if bvals is not None:
        b = np.asarray(bvals, dtype=np.float64).reshape(-1)
        if T is None:
            T = int(b.shape[0])
        elif b.shape[0] != T:
            raise ValueError(f"bvals has {b.shape[0]} entries for {T} frames")
        b_max = max(float(np.max(np.abs(b))) if b.size > 0 else 1.0, 1.0)
        b_tilde = np.clip(b, 0.0, None) / b_max
        is_dwi = (b > float(b0_threshold)).astype(np.float64)
        cols.append(np.c_[is_dwi, b_tilde])
        if bvecs is not None:
            g = np.asarray(bvecs, dtype=np.float64)
            if g.ndim == 2 and g.shape[0] == 3 and g.shape[1] == T and T != 3:
                g = g.T
            g = g.reshape(T, 3).copy()
            nrm = np.linalg.norm(g, axis=1, keepdims=True)
            valid = (nrm[:, 0] > 1e-6) & (b > float(b0_threshold))
            g_unit = np.zeros_like(g)
            g_unit[valid] = g[valid] / nrm[valid]
            sb = np.sqrt(b_tilde)[:, None]
            gx, gy, gz = g_unit[:, 0], g_unit[:, 1], g_unit[:, 2]
            s2 = np.sqrt(2.0)
            q6 = b_tilde[:, None] * np.c_[
                gx * gx, gy * gy, gz * gz, s2 * gx * gy, s2 * gx * gz, s2 * gy * gz
            ]
            cols.append(np.c_[sb * g_unit, q6])
    elif bvecs is not None:
        g = np.asarray(bvecs, dtype=np.float64)
        if T is not None and g.ndim == 2 and g.shape[0] == 3 and g.shape[1] == T and T != 3:
            g = g.T
        if T is None:
            T = int(g.shape[0])
        g = g.reshape(T, 3)
        gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
        s2 = np.sqrt(2.0)
        cols.append(np.c_[g, gx * gx, gy * gy, gz * gz, s2 * gx * gy, s2 * gx * gz, s2 * gy * gz])

    if T is None:
        raise ValueError("build_encoding_covariates requires at least one covariate input or num_frames")

    if not cols or T < 2:
        E = np.zeros((T, 0), dtype=np.float64)
        Q = np.zeros((T, 0), dtype=np.float64)
        E_surr = np.zeros((max(0, T - 2), 0), dtype=np.float64)
    else:
        F_raw = np.hstack(cols).astype(np.float64)
        F_c = F_raw - np.mean(F_raw, axis=0, keepdims=True)
        col_std = np.std(F_c, axis=0)
        if T >= 3:
            F_surr = F_c[1:-1] - 0.5 * (F_c[:-2] + F_c[2:])
            surr_std = np.std(F_surr, axis=0)
            keep = (col_std > 1e-7) & (surr_std > 1e-7)
        else:
            keep = col_std > 1e-7
        F_act = F_c[:, keep]
        if F_act.shape[1] == 0:
            E = np.zeros((T, 0), dtype=np.float64)
            Q = np.zeros((T, 0), dtype=np.float64)
            E_surr = np.zeros((max(0, T - 2), 0), dtype=np.float64)
        else:
            F_act = F_act / np.maximum(np.std(F_act, axis=0, keepdims=True), 1e-8)
            U, s_vals, _ = np.linalg.svd(F_act, full_matrices=False)
            rank = int(np.sum(s_vals > 1e-4 * s_vals[0]))
            Q = U[:, :rank]
            Q = Q - np.mean(Q, axis=0, keepdims=True)
            q_nrm = np.linalg.norm(Q, axis=0, keepdims=True)
            valid_q = q_nrm[0] > 1e-7
            Q = Q[:, valid_q] / np.maximum(q_nrm[:, valid_q], 1e-12)
            E = Q * np.sqrt(float(T))
            E_surr = E[1:-1] - 0.5 * (E[:-2] + E[2:]) if T >= 3 else np.zeros((0, E.shape[1]), dtype=np.float64)

    K_eff = int(E.shape[1])
    w_size = int(window_size) if (window_size is not None and int(window_size) > 0) else T
    w_stride = int(window_stride) if (window_stride is not None and int(window_stride) > 0) else max(1, w_size // 2)

    if w_size >= T or T < max(2 * w_size, 16):
        psi = np.ones((T, 1), dtype=np.float64)
    else:
        centers = list(range(0, T, w_stride))
        if centers[-1] < T - 1:
            centers.append(T - 1)
        t_idx = np.arange(T, dtype=np.float64)[:, None]
        c_arr = np.asarray(centers, dtype=np.float64)[None, :]
        u_dist = np.abs(t_idx - c_arr) / float(w_size)
        raw_w = np.where(u_dist < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.minimum(u_dist, 1.0))), 0.0) + 1e-6
        psi = raw_w / np.sum(raw_w, axis=1, keepdims=True)

    M_win = int(psi.shape[1])
    E_psi = psi[:, :, None] * E[:, None, :]
    return {
        "E": E,
        "Q": Q,
        "E_surround": E_surr,
        "psi": psi,
        "E_psi": E_psi,
        "K_eff": K_eff,
        "num_windows": M_win,
        "window_size": w_size,
        "window_stride": w_stride,
    }


def _project_encoding_surround_tukey_np(
    Y_head: np.ndarray,
    E: np.ndarray,
    psi: np.ndarray,
    fd_radius: float = 50.0,
    n_irls: int = 4,
    ridge: float = 1e-3,
    surround_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form Tukey biweight surround projection within overlapping windows psi (T, M).

    Given pose trajectory ``Y_head`` (T, 6) = [tx, ty, tz, rx, ry, rz] (mm and radians) and
    orthonormalized encoding basis ``E`` (T, K), estimates window bias matrices ``delta_B`` (M, K, 6)
    from the second-order surround differences ``u_t = Y_t - 0.5 * (Y_{t-1} + Y_{t+1})`` against
    ``e_tilde_t = e_t - 0.5 * (e_{t-1} + e_{t+1})`` so smooth physical head drift (constant, linear,
    and quadratic within each window) is in the nullspace of the surround operator and sudden real
    head motion jumps are downweighted by Tukey's biweight (c = 4.685).

    Returns ``(delta_B, delta_Y, w_tukey)`` where ``delta_Y[t] = sum_m psi[t, m] * (E[t] @ delta_B[m])``.
    """
    T, K = E.shape
    M = psi.shape[1]
    if T < 4 or K == 0:
        return (
            np.zeros((M, K, 6), dtype=np.float64),
            np.zeros((T, 6), dtype=np.float64),
            np.ones(max(0, T - 2), dtype=np.float64),
        )

    u_head = Y_head[1:-1] - 0.5 * (Y_head[:-2] + Y_head[2:])  # (T-2, 6)
    E_surr = E[1:-1] - 0.5 * (E[:-2] + E[2:])  # (T-2, K)
    psi_int = psi[1:-1]  # (T-2, M)
    T_int = T - 2

    # Subtract constant curvature offset -c_2 from u_head so quadratic head drift is in the nullspace
    if K == 1:
        g1 = E_surr[:, 0]
        pos_m = g1 > 0.1 * float(np.max(np.abs(g1)))
        neg_m = g1 < -0.1 * float(np.max(np.abs(g1)))
        if np.any(pos_m) and np.any(neg_m):
            u_curv = 0.5 * (
                np.median(u_head[pos_m], axis=0, keepdims=True)
                + np.median(u_head[neg_m], axis=0, keepdims=True)
            )
        else:
            u_curv = np.median(u_head, axis=0, keepdims=True)
    else:
        u_curv = np.median(u_head, axis=0, keepdims=True)
    u_center = u_head - u_curv

    # 1) Robust initialization & global Tukey biweight IRLS (c = 4.685) of global encoding bias B_bar (K, 6):
    #    - For K == 1: exact unimodal parity-ratio median + component-wise Tukey biweight IRLS (50% breakdown).
    #    - For K > 1: L1/Cauchy-weighted coupled initialization + joint 6-DOF Tukey biweight IRLS.
    B_bar = np.zeros((K, 6), dtype=np.float64)
    w_tukey = np.ones(T_int, dtype=np.float64)
    if K == 1:
        ek = E_surr[:, 0]
        act = np.abs(ek) > max(0.25 * float(np.max(np.abs(ek))), 1e-6)
        if int(np.sum(act)) >= 2:
            z = u_center[act] / ek[act, None]  # (N_act, 6)
            b0 = np.median(z, axis=0)
            mad = np.median(np.abs(z - b0[None, :]), axis=0)
            scale = np.maximum(1.4826 * mad, 1e-6)
            w_act = np.ones_like(z)
            for _ in range(n_irls):
                r = (z - b0[None, :]) / (4.685 * scale[None, :])
                w_act = np.where(np.abs(r) < 1.0, (1.0 - r ** 2) ** 2, 0.0)
                w_sum = np.sum(w_act, axis=0)
                b0 = np.where(w_sum > 1e-6, np.sum(w_act * z, axis=0) / np.maximum(w_sum, 1e-6), b0)
            B_bar[0] = b0
            w_tukey[act] = np.mean(w_act, axis=1)
    else:
        d0 = np.sqrt(
            np.sum(u_center[:, :3] ** 2, axis=1)
            + (float(fd_radius) ** 2) * np.sum(u_center[:, 3:] ** 2, axis=1)
        )
        s0 = max(float(np.median(d0)), 0.02)
        w0 = 1.0 / (1.0 + (d0 / (1.5 * s0)) ** 4)
        for _ in range(3):
            sw0 = np.sqrt(w0)[:, None]
            Xw0 = E_surr * sw0
            Yw0 = u_center * sw0
            B_bar = np.linalg.solve(Xw0.T @ Xw0 + float(ridge) * np.eye(K), Xw0.T @ Yw0)
            res0 = u_center - E_surr @ B_bar
            d0 = np.sqrt(
                np.sum(res0[:, :3] ** 2, axis=1)
                + (float(fd_radius) ** 2) * np.sum(res0[:, 3:] ** 2, axis=1)
            )
            s0 = max(1.4826 * float(np.median(np.abs(d0 - np.median(d0)))), 0.015)
            w0 = 1.0 / (1.0 + (d0 / (2.0 * s0)) ** 2)

        for _ in range(n_irls):
            res_g = u_center - E_surr @ B_bar
            d_mm = np.sqrt(np.sum(res_g[:, :3] ** 2, axis=1) + (float(fd_radius) ** 2) * np.sum(res_g[:, 3:] ** 2, axis=1))
            med_d = float(np.median(d_mm))
            mad_d = max(1.4826 * float(np.median(np.abs(d_mm - med_d))), 0.025)
            u_scaled = np.maximum(0.0, d_mm - med_d) / (4.685 * mad_d)
            w_tukey = np.where(u_scaled < 1.0, (1.0 - u_scaled ** 2) ** 2, 1e-4)
            if float(np.sum(w_tukey)) > 1e-6:
                sw = np.sqrt(w_tukey)[:, None]
                Xw = E_surr * sw
                Yw = u_center * sw
                gram_g = Xw.T @ Xw + float(ridge) * np.eye(K, dtype=np.float64)
                B_bar = np.linalg.solve(gram_g, Xw.T @ Yw)

    # 3) Windowed estimation around B_bar using outlier-free Tukey weights and sample-adaptive shrinkage
    delta_B = np.broadcast_to(B_bar[None, :, :], (M, K, 6)).copy()
    if M > 1:
        res_global = u_center - E_surr @ B_bar
        for m in range(M):
            wm = psi_int[:, m] * w_tukey
            eff_n = float(np.sum(wm > 0.05))
            if eff_n < max(float(3 * K), 6.0):
                continue
            sw = np.sqrt(wm)[:, None]
            Xw = E_surr * sw
            Rw = res_global * sw
            gram_raw = Xw.T @ Xw
            diag_scale = max(float(np.trace(gram_raw)) / max(float(K), 1.0), 1e-3)
            ridge_win = diag_scale * max(float(ridge), float(2.0 * K) / max(eff_n / 6.0, 1.0))
            gram_m = gram_raw + ridge_win * np.eye(K, dtype=np.float64)
            delta_B[m] = B_bar + np.linalg.solve(gram_m, Xw.T @ Rw)
        delta_Y = np.einsum("tm,tk,mkd->td", psi, E, delta_B)
    else:
        delta_Y = E @ delta_B[0]

    # 4) Optional gentle non-spike surround damping on the residual head trajectory when surround_weight > 0:
    #    Damps residual high-frequency state-transition jitter on steady-state frames (w_tukey > 0.5) while
    #    leaving real motion spikes (w_tukey <= 0.5) completely untouched.
    if float(surround_weight) > 0.0 and T >= 6:
        Y_rem = Y_head - delta_Y
        u_rem = Y_rem[1:-1] - 0.5 * (Y_rem[:-2] + Y_rem[2:])
        alpha = min(0.5, float(surround_weight) * 5.0)
        damp = np.zeros_like(Y_rem)
        inlier_w = np.where(w_tukey > 0.5, w_tukey, 0.0)[:, None]
        damp[1:-1] = alpha * inlier_w * u_rem
        delta_Y = delta_Y + damp

    return delta_B, delta_Y, w_tukey


def _fit_tukey_trend(Y: np.ndarray, window_size: int, n_iter: int = 3) -> np.ndarray:
    """Robust windowed trend filtering using raised-cosine (Hann) basis and Tukey biweight IRLS.

    Fits Y ~ Psi @ theta where Psi is the partition-of-unity basis with kernel radius window_size.
    Returns high-pass residual Y - Psi @ theta.
    Accepts 1D (T,) or 2D (T, K) arrays.
    """
    Y_arr = np.asarray(Y, dtype=np.float64)
    is_1d = (Y_arr.ndim == 1)
    Y_cols = Y_arr[:, None] if is_1d else Y_arr
    T = Y_cols.shape[0]
    H_loc = build_encoding_covariates(num_frames=T, window_size=int(window_size))["psi"]
    gram_loc = H_loc.T @ H_loc + 1e-4 * np.eye(H_loc.shape[1], dtype=np.float64)
    theta_loc = np.linalg.solve(gram_loc, H_loc.T @ Y_cols)
    for _ in range(n_iter):
        res_loc = Y_cols - H_loc @ theta_loc
        med_loc = np.median(res_loc, axis=0)
        mad_loc = np.maximum(1.4826 * np.median(np.abs(res_loc - med_loc), axis=0), 1e-6)
        u_loc = (res_loc - med_loc) / (4.685 * mad_loc)
        w_loc = np.where(np.abs(u_loc) < 1.0, (1.0 - u_loc ** 2) ** 2, 1e-4)
        for c_idx in range(Y_cols.shape[1]):
            Hw_loc = H_loc * w_loc[:, c_idx:c_idx + 1]
            theta_loc[:, c_idx] = np.linalg.solve(
                Hw_loc.T @ H_loc + 1e-4 * np.eye(H_loc.shape[1], dtype=np.float64),
                Hw_loc.T @ Y_cols[:, c_idx],
            )
    res = Y_cols - H_loc @ theta_loc
    return res[:, 0] if is_1d else res


def _winsorized_step_ac1(series: np.ndarray) -> float:
    """Winsorized lag-1 autocorrelation of step differences Delta x_t."""
    ds = np.diff(series)
    if ds.size < 4:
        return 0.0
    mu = float(np.mean(ds))
    q75 = max(float(np.percentile(np.abs(ds - mu), 75)), 1e-12)
    ds_c = np.clip(ds - mu, -3.0 * q75, 3.0 * q75)
    ds0 = ds_c[:-1] - float(np.mean(ds_c[:-1]))
    ds1 = ds_c[1:] - float(np.mean(ds_c[1:]))
    denom = float(np.sqrt(np.sum(ds0 ** 2) * np.sum(ds1 ** 2)))
    return float(np.sum(ds0 * ds1) / denom) if denom > 1e-12 else 0.0


def _shrink_pe_pseudomotion_core(
    R_all: np.ndarray,
    t_all: np.ndarray,
    com: np.ndarray,
    p_pe: np.ndarray,
    p_sl: np.ndarray,
    p_ro: np.ndarray,
    fd_radius: float = 50.0,
    window_size: Optional[int] = None,
    mode: str = "bold",
    sagittal_nyquist: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
    """Centralized core for post-hoc respiratory phase-encode pseudomotion shrinkage.

    Supports mode='bold' (3 DOFs: t_pe, t_sl, r_ro with gate < -0.25) and
    mode='grouped' (6 DOFs with gate < -0.38).
    Returns (R_clean, t_clean, deltas, meta).
    """
    from scipy.spatial.transform import Rotation as Rot

    T = len(t_all)
    rv = Rot.from_matrix(R_all).as_rotvec()
    rv_mm = float(fd_radius) * rv

    t_ro = t_all @ p_ro
    t_pe = t_all @ p_pe
    t_sl = t_all @ p_sl
    r_ro = rv_mm @ p_ro
    r_pe = rv_mm @ p_pe
    r_sl = rv_mm @ p_sl

    if mode == "bold":
        w_size = int(window_size) if (window_size is not None and int(window_size) > 0) else max(8, min(24, T // 5))
        r_pe_res = _fit_tukey_trend(t_pe, w_size)
        w_nod = min(6, max(4, T // 8))
        r_sag_res = _fit_tukey_trend(np.column_stack([t_sl, r_ro]), w_nod)
        r_epi = np.column_stack([r_pe_res, r_sag_res])

        raw_epi = np.column_stack([t_pe, t_sl, r_ro])
        lag1_autocorrs = [_winsorized_step_ac1(raw_epi[:, k]) for k in range(3)]
        sigs = [float(np.median(np.abs(np.diff(r_epi[:, k])))) for k in range(3)]

        imm_floors = []
        for imm_s in (t_ro, r_pe, r_sl):
            sig_i = float(np.median(np.abs(np.diff(imm_s))))
            ac1_i = _winsorized_step_ac1(imm_s)
            eta_nyq_i = float(np.clip((-ac1_i - 0.20) / 0.45, 0.0, 0.75)) if (sagittal_nyquist and ac1_i < -0.25) else 0.0
            imm_floors.append(sig_i * (1.0 - eta_nyq_i))
        sig_ref = float(np.median(imm_floors))

        sag_min_sig = max(1.35 * sig_ref, 0.025)
        eta_pe = max(0.0, 1.0 - (sig_ref / max(sigs[0], 1e-12))) if sigs[0] > 1.10 * sig_ref else 0.0
        eta_sl = (
            max(0.0, 1.0 - (sig_ref / max(sigs[1], 1e-12)))
            if (sigs[1] > sag_min_sig and lag1_autocorrs[1] < -0.25)
            else 0.0
        )
        eta_r_ro = (
            max(0.0, 1.0 - (sig_ref / max(sigs[2], 1e-12)))
            if (sigs[2] > sag_min_sig and lag1_autocorrs[2] < -0.25)
            else 0.0
        )
        if not sagittal_nyquist:
            eta_sl = eta_r_ro = 0.0
        etas = [eta_pe, eta_sl, eta_r_ro]
        if max(etas) <= 0.05:
            return R_all, t_all, np.zeros_like(r_epi), None

        d_ref_step = np.zeros(T, dtype=np.float64)
        d_ref_step[1:] = np.abs(np.diff(t_ro)) + np.abs(np.diff(r_pe)) + np.abs(np.diff(r_sl))
        d_ref_local = np.maximum(d_ref_step, np.r_[d_ref_step[1:], 0.0])
        tau_ref = max(0.25, 3.5 * float(np.median(d_ref_step[1:])))
        w_b0 = np.exp(-np.maximum(0.0, (d_ref_local - tau_ref) / max(0.15, tau_ref)) ** 2)

        deltas = np.zeros_like(r_epi)
        for k in range(3):
            if etas[k] <= 0.0:
                continue
            dr_k = np.zeros(T, dtype=np.float64)
            dr_k[1:] = np.abs(np.diff(r_epi[:, k]))
            dr_local = np.maximum(dr_k, np.r_[dr_k[1:], 0.0])
            scale_k = max(1.4826 * sigs[k], 0.02)
            u_step = np.maximum(0.0, dr_local - 3.5 * scale_k) / (2.5 * scale_k)
            w_step_k = np.exp(-(u_step ** 2))
            w_gate = w_b0 * w_step_k if k == 0 else w_step_k
            deltas[:, k] = etas[k] * w_gate * r_epi[:, k]

        t_clean = t_all - np.outer(deltas[:, 0], p_pe) - np.outer(deltas[:, 1], p_sl)
        if etas[2] > 0.0:
            rv_clean = rv - np.outer(deltas[:, 2] / float(fd_radius), p_ro)
            R_clean = Rot.from_rotvec(rv_clean).as_matrix()
        else:
            R_clean = R_all

        meta = {
            "method": "overlapping_window_wiener_tukey_epi_plane",
            "pe_vector_lps": [float(x) for x in p_pe],
            "slice_vector_lps": [float(x) for x in p_sl],
            "readout_vector_lps": [float(x) for x in p_ro],
            "step_noise_floor_ref_mm": sig_ref,
            "step_noise_floor_pe_mm": sigs[0],
            "wiener_shrinkage_eta": float(eta_pe),
            "pe_b0_shift_rms_mm": float(np.sqrt(np.mean(deltas[:, 0] ** 2))),
            "pe_b0_shift_max_mm": float(np.max(np.abs(deltas[:, 0]))),
            "pe_b0_shift_mm": [float(x) for x in deltas[:, 0]],
            "window_size": int(w_size),
            "step_lag1_autocorr_pe": float(lag1_autocorrs[0]),
            "sagittal_nyquist": bool(sagittal_nyquist),
        }
        if sagittal_nyquist:
            meta.update(
                step_noise_floor_sl_mm=sigs[1],
                step_noise_floor_r_ro_mm=sigs[2],
                step_lag1_autocorr_sl=float(lag1_autocorrs[1]),
                step_lag1_autocorr_r_ro=float(lag1_autocorrs[2]),
                wiener_shrinkage_eta_sl=float(eta_sl),
                wiener_shrinkage_eta_r_ro=float(eta_r_ro),
                sl_shift_rms_mm=float(np.sqrt(np.mean(deltas[:, 1] ** 2))),
                r_ro_shift_rms_mm=float(np.sqrt(np.mean(deltas[:, 2] ** 2))),
            )
        return R_clean, t_clean, deltas, meta

    elif mode == "grouped":
        w_pe = max(8, min(16, T // 5))
        w_nod = min(6, max(4, T // 8))
        r_pe_res = _fit_tukey_trend(t_pe, w_pe)
        raw_non_pe = np.column_stack([t_sl, r_ro, t_ro, r_pe, r_sl])
        r_non_pe = _fit_tukey_trend(raw_non_pe, w_nod)
        r_epi = np.column_stack([r_pe_res, r_non_pe])

        raw_all6 = np.column_stack([t_pe, raw_non_pe])
        lag1_ac = [_winsorized_step_ac1(raw_all6[:, k]) for k in range(6)]
        sigs = [float(np.median(np.abs(np.diff(r_epi[:, k])))) for k in range(6)]

        etas_non_pe = [0.0] * 5
        for idx_np in range(5):
            k = idx_np + 1
            ac1_k = lag1_ac[k]
            sig_k_mm = 1.4826 * sigs[k]
            if ac1_k < -0.38 and sig_k_mm > 0.052:
                etas_non_pe[idx_np] = float(np.clip((-ac1_k - 0.20) / 0.45, 0.0, 0.85))

        imm_clean = []
        for k in (3, 4, 5):
            eta_k = etas_non_pe[k - 1]
            if eta_k > 0.0:
                dr_k = np.zeros(T, dtype=np.float64)
                dr_k[1:] = np.abs(np.diff(r_epi[:, k]))
                dr_local = np.maximum(dr_k, np.r_[dr_k[1:], 0.0])
                scale_k = max(1.4826 * sigs[k], 0.02)
                w_step_k = np.exp(-(np.maximum(0.0, dr_local - 3.5 * scale_k) / (2.5 * scale_k)) ** 2)
                imm_clean.append(raw_all6[:, k] - eta_k * w_step_k * r_epi[:, k])
            else:
                imm_clean.append(raw_all6[:, k])

        sig_ref = float(np.median([np.median(np.abs(np.diff(c))) for c in imm_clean]))
        sag_min_sig = max(1.50 * sig_ref, 0.052 / 1.4826)
        for idx_sag in (0, 1):
            k = idx_sag + 1
            if lag1_ac[k] < -0.38 and sigs[k] > sag_min_sig:
                eta_wiener = max(0.0, 1.0 - (sig_ref / max(sigs[k], 1e-12)))
                etas_non_pe[idx_sag] = max(etas_non_pe[idx_sag], eta_wiener)

        sig_pe = sigs[0]
        eta_pe = max(0.0, 1.0 - (sig_ref / max(sig_pe, 1e-12))) if sig_pe > 1.20 * sig_ref else 0.0
        etas = [eta_pe] + etas_non_pe
        if max(etas) <= 0.05:
            meta = {
                "method": "grouped_overlapping_window_wiener_tukey_epi",
                "applied": False,
                "wiener_shrinkage_eta_all6": [float(e) for e in etas],
                "lag1_step_autocorr_all6": [float(a) for a in lag1_ac],
                "removed_pe_rms_mm": 0.0,
            }
            return R_all, t_all, np.zeros_like(r_epi), meta

        d_ref_step = np.zeros(T, dtype=np.float64)
        d_ref_step[1:] = sum(np.abs(np.diff(c)) for c in imm_clean)
        d_ref_local = np.maximum(d_ref_step, np.r_[d_ref_step[1:], 0.0])
        tau_ref = max(0.25, 3.5 * float(np.median(d_ref_step[1:])))
        w_b0 = np.exp(-np.maximum(0.0, (d_ref_local - tau_ref) / max(0.15, tau_ref)) ** 2)

        deltas = np.zeros_like(r_epi)
        for k in range(6):
            if etas[k] <= 0.0:
                continue
            dr_k = np.zeros(T, dtype=np.float64)
            dr_k[1:] = np.abs(np.diff(r_epi[:, k]))
            dr_local = np.maximum(dr_k, np.r_[dr_k[1:], 0.0])
            scale_k = max(1.4826 * sigs[k], 0.02)
            u_step = np.maximum(0.0, dr_local - 3.5 * scale_k) / (2.5 * scale_k)
            w_step_k = np.exp(-(u_step ** 2))
            w_gate = w_b0 * w_step_k if k == 0 else w_step_k
            deltas[:, k] = etas[k] * w_gate * r_epi[:, k]

        t_clean = (
            t_all
            - np.outer(deltas[:, 0], p_pe)
            - np.outer(deltas[:, 1], p_sl)
            - np.outer(deltas[:, 3], p_ro)
        )
        if max(etas[2], etas[4], etas[5]) > 0.0:
            rv_clean = (
                rv
                - np.outer(deltas[:, 2] / float(fd_radius), p_ro)
                - np.outer(deltas[:, 4] / float(fd_radius), p_pe)
                - np.outer(deltas[:, 5] / float(fd_radius), p_sl)
            )
            R_clean = Rot.from_rotvec(rv_clean).as_matrix()
        else:
            R_clean = R_all

        meta = {
            "method": "grouped_overlapping_window_wiener_tukey_epi",
            "pe_vector_lps": [float(x) for x in p_pe],
            "sigma_ref_mm": sig_ref,
            "sigma_pe_mm": sig_pe,
            "wiener_shrinkage_eta": eta_pe,
            "wiener_shrinkage_eta_all6": [float(e) for e in etas],
            "lag1_step_autocorr_all6": [float(a) for a in lag1_ac],
            "removed_pe_rms_mm": float(np.sqrt(np.mean(deltas[:, 0] ** 2))),
            "removed_non_pe_rms_mm": float(np.sqrt(np.mean(np.sum(deltas[:, 1:] ** 2, axis=1)))),
            "window_size": int(w_pe),
            "applied": True,
        }
        return R_clean, t_clean, deltas, meta

    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'bold' or 'grouped'")


def separate_bold_pe_pseudomotion(
    bold_image: ants.ANTsImage,
    moco: dict,
    pe_axis: Optional[np.ndarray] = None,
    fd_radius: float = 50.0,
    window_size: Optional[int] = None,
    sagittal_nyquist: bool = False,
    slice_axis: Any = None,
    sidecar: Any = None,
) -> Tuple[dict, Optional[dict]]:
    """Separate respiratory B0 phase-encode pseudomotion from the REPORTED rigid head motion of an EPI series.

    Opt-in (``motion_correction(..., debias_pe_pseudomotion=True)``). In EPI a respiratory field offset
    ``Delta B0(t)`` displaces the image along the phase-encode axis ``p_PE`` only (bandwidth along PE is low; readout
    and slice selection are not affected to first order). By default only the translation along ``p_PE`` is modelled:
    slice translation and pitch are dominated by real head nodding and are not modified.

    ``sagittal_nyquist=True`` (opt-in, from pilot/motion-sampling-and-epi-debias) also shrinks a TR-to-TR alternation of
    slice translation and pitch (thick-slice through-plane B0 gradients), gated on a negative lag-1 step autocorrelation
    (< -0.25). Caution: independent real jitter between frames also has a step autocorrelation near -0.5, so this can
    remove real motion from the reported trajectory (independent audit: real nodding jitter FD 0.94 -> 0.15 mm in the
    grouped variant); use it only where the alternation is known to be pseudomotion. The slice axis is ``slice_axis``
    (BIDS i/j/k or voxel index), else the sidecar's ``SliceEncodingDirection`` / ``SliceTiming``, else measured from the
    series (``syntx.acquisition_axes.infer_slice_axis``: voxel thickness, slice interleave); undecidable raises.

    The resampled series and the transforms keep the full registration pose (the B0 shift is a real displacement of
    the image), so after this step ``motion_parameters`` / ``fd`` differ from ``fwdtransforms`` by ``pe_b0_shift_mm``
    along ``p_PE``; the unmodified trajectory stays in ``motion_parameters_raw`` / ``fd_raw``. The PE translation is
    split into a smooth postural part (overlapping Hann windows, Tukey IRLS) and a residual; the residual is shrunk by
    ``eta = max(0, 1 - sigma_ref / sigma_PE)`` where ``sigma_ref`` is the step noise floor of the B0-immune DOFs
    (readout translation, rotations about PE and slice), and gated to 0 on frames where the immune DOFs or the PE
    residual itself show a real head jump. Returns ``(moco, meta)``; ``meta`` is ``None`` when no excess PE variance
    is found (nothing is changed).
    """
    from scipy.spatial.transform import Rotation as Rot
    from .motion import MotionParameters, _extract_rigid_parameters, calculate_framewise_displacement

    fwd_paths = moco.get("fwdtransforms")
    ref_img = moco.get("reference")
    if not fwd_paths or ref_img is None:
        raise ValueError("separate_bold_pe_pseudomotion needs a motion_correction result with fwdtransforms and reference")
    T = len(fwd_paths)
    if T < 12:
        warnings.warn(f"separate_bold_pe_pseudomotion: {T} frames < 12, too few to separate PE pseudomotion; unchanged",
                      RuntimeWarning, stacklevel=2)
        return moco, None
    tx_objs = [ants.read_transform(p[-1]) for p in fwd_paths]

    com = np.asarray(tx_objs[0].fixed_parameters[:3], dtype=np.float64)
    R_all = np.zeros((T, 3, 3), dtype=np.float64)
    t_all = np.zeros((T, 3), dtype=np.float64)
    for b, tx in enumerate(tx_objs):
        p = np.asarray(tx.parameters, dtype=np.float64)
        c_b = np.asarray(tx.fixed_parameters[:3], dtype=np.float64)
        R_b = p[:9].reshape(3, 3)
        t_b = p[9:12]
        R_all[b] = R_b
        t_all[b] = t_b + (R_b - np.eye(3)) @ (com - c_b)

    rv = Rot.from_matrix(R_all).as_rotvec()
    rv_mm = float(fd_radius) * rv

    if pe_axis is None:
        raise ValueError("separate_bold_pe_pseudomotion needs pe_axis (physical unit vector or voxel axis index 0/1/2)")
    from .acquisition_axes import encoding_frame, infer_slice_axis

    # slice axis: stated (slice_axis / sidecar) or measured from the series (thickness, interleave); never assumed
    sl_inf = infer_slice_axis(bold_image, slice_axis=slice_axis, sidecar=sidecar)
    p_pe, p_sl, p_ro = encoding_frame(getattr(bold_image, "direction", np.eye(3)), pe_axis, sl_inf.voxel_axis)

    R_clean, t_clean, deltas, meta = _shrink_pe_pseudomotion_core(
        R_all,
        t_all,
        com,
        p_pe,
        p_sl,
        p_ro,
        fd_radius=fd_radius,
        window_size=window_size,
        mode="bold",
        sagittal_nyquist=bool(sagittal_nyquist),
    )
    if meta is None:
        return moco, None

    raw_params = np.zeros((T, 6), dtype=np.float64)
    homog_all = []
    for b in range(T):
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([R_clean[b].flatten(), t_clean[b]]))
        tx.set_fixed_parameters(com)
        trans_b, rot_b, T_h = _extract_rigid_parameters(tx, 3)
        raw_params[b] = np.concatenate([trans_b, rot_b])
        homog_all.append(T_h)

    mp_clean = MotionParameters(
        raw_params,
        columns=["tx", "ty", "tz", "rx", "ry", "rz"],
        rad_to_deg_cols=["rx", "ry", "rz"],
        spatial_dim=3,
    )
    center_of_sphere = ref_img.get_center_of_mass()
    fd_power, fd_jenkinson = calculate_framewise_displacement(
        mp_clean,
        homog_all,
        radius=float(fd_radius),
        center_of_sphere=center_of_sphere,
    )

    out = type(moco)(moco) if isinstance(moco, dict) else dict(moco)
    out["fd_raw"] = np.asarray(moco.get("fd")).copy() if moco.get("fd") is not None else None
    out["motion_parameters_raw"] = moco.get("motion_parameters")
    out["motion_parameters"] = mp_clean
    out["fd"] = fd_power
    out["fd_power"] = fd_power
    out["fd_jenkinson"] = fd_jenkinson
    out["pe_b0_shift_mm"] = deltas[:, 0].copy()
    if isinstance(out.get("summary"), dict):
        summary = dict(out["summary"])
        summary["fd_raw_mean"] = summary.get("fd_mean", 0.0)
        summary["fd_mean"] = float(np.mean(fd_power[1:])) if T > 1 else 0.0
        summary["fd_median"] = float(np.median(fd_power[1:])) if T > 1 else 0.0
        summary["fd_power_mean"] = float(np.mean(fd_power[1:])) if T > 1 else 0.0
        summary["fd_jenkinson_mean"] = float(np.mean(fd_jenkinson[1:])) if T > 1 else 0.0
        summary["fd_max"] = float(np.max(fd_power))
        out["summary"] = summary

    meta = dict(meta)
    meta["slice_axis"] = {"voxel_axis": int(sl_inf.voxel_axis), "source": sl_inf.source}
    meta["sigma_ref_mm"] = meta.get("step_noise_floor_ref_mm")
    sigs_list = [meta.get("step_noise_floor_pe_mm")]
    ac1_list = [meta.get("step_lag1_autocorr_pe")]
    eta_list = [meta.get("wiener_shrinkage_eta")]
    if sagittal_nyquist:
        sigs_list.extend([meta.get("step_noise_floor_sl_mm"), meta.get("step_noise_floor_r_ro_mm")])
        ac1_list.extend([meta.get("step_lag1_autocorr_sl"), meta.get("step_lag1_autocorr_r_ro")])
        eta_list.extend([meta.get("wiener_shrinkage_eta_sl"), meta.get("wiener_shrinkage_eta_r_ro")])
    meta["sigma_epi_mm"] = sigs_list
    meta["lag1_step_autocorr"] = ac1_list
    meta["wiener_shrinkage_eta"] = eta_list
    meta["removed_pe_rms_mm"] = float(np.sqrt(np.mean(deltas[:, 0] ** 2)))
    meta["removed_slice_rms_mm"] = float(np.sqrt(np.mean(deltas[:, 1] ** 2)))
    meta["removed_pitch_rms_mm"] = float(np.sqrt(np.mean(deltas[:, 2] ** 2)))
    out["pe_pseudomotion_debias"] = meta
    return out, meta


debias_pe_pseudomotion = separate_bold_pe_pseudomotion



SURROUND_FORMS = ("charbonnier", "sparse", "cauchy")
SPARSE_KNEE_FRACTION = 0.1   # the ``sparse`` form's smoothing knee as a fraction of ``delta_mm``


def surround_rho(u_sq: torch.Tensor, weight: float, delta_mm: float, form: str = "charbonnier"):
    """The surround prior's penalty per second difference and the two coefficients of its Newton Hessian model.

    ``u_sq`` (n,) is ``|u|^2`` in mm^2 (rotations times the lever arm). Returns ``(rho, omega, kappa)``, each (n,): the
    penalty ``rho(|u|)``, the IRLS weight ``omega = rho'(|u|) / |u|`` (gradient ``omega u``) and the coefficient ``kappa`` of
    ``u u^T`` in the Hessian model ``omega I + kappa u u^T``. With ``d = delta_mm`` (the knee: second differences below it are
    measurement noise) and ``w = weight``:

    - ``charbonnier``: ``rho = w d^2 (sqrt(1 + |u|^2/d^2) - 1)``: quadratic below the knee (gradient -> 0 at 0, so sub-knee
      jitter is shrunk least), linear above with slope ``w d``. Exact Hessian ``w d (s^2 I - u u^T) / s^3``, PSD.
    - ``sparse``: ``rho = w d (sqrt(|u|^2 + e^2) - e)``, ``e = SPARSE_KNEE_FRACTION d``: the same asymptotic slope ``w d``, but the
      gradient is ``w d |u| / e`` near zero (ten times steeper), so sub-knee jitter is driven to (near) zero while a smooth
      drift (``u ~ 0``) and a real jump (one large ``u``, linear cost) are left alone. Algebraically a Charbonnier with knee
      ``e`` and weight ``w d / e``; exact PSD Hessian of the same form.
    - ``cauchy``: ``rho = (w d^2 / 2) log(1 + |u|^2/d^2)``: gradient ``w |u|`` near zero, concave beyond the knee (a jump
      costs less and less). Its exact Hessian is indefinite for ``|u| > d``, so the Newton model uses the IRLS majoriser
      ``omega I`` (``kappa = 0``), the standard choice for re-descending penalties.
    """
    if form not in SURROUND_FORMS:
        raise ValueError(f"surround_form must be one of {SURROUND_FORMS}, got {form!r}")
    w, d = float(weight), float(delta_mm)
    if form == "charbonnier":
        s = torch.sqrt(d * d + u_sq)
        return w * d * (s - d), w * d / s, -w * d / s.pow(3)
    if form == "sparse":
        e = SPARSE_KNEE_FRACTION * d
        s = torch.sqrt(e * e + u_sq)
        return w * d * (s - e), w * d / s, -w * d / s.pow(3)
    s2 = d * d + u_sq
    return 0.5 * w * d * d * torch.log1p(u_sq / (d * d)), w * d * d / s2, torch.zeros_like(u_sq)


def _surround_hessian_terms(t_param, omega, weight, delta_mm, rot_radius_mm, form):
    """(B-2, 6, 6) Hessian model of every second-difference term in mm units (``surround_rho``)."""
    u = torch.cat([t_param[1:-1] - 0.5 * (t_param[:-2] + t_param[2:]),
                   rot_radius_mm * (omega[1:-1] - 0.5 * (omega[:-2] + omega[2:]))], dim=1)        # (B-2, 6) mm
    _, om, kap = surround_rho((u * u).sum(dim=1), weight, delta_mm, form)
    eye6 = torch.eye(6, dtype=t_param.dtype, device=t_param.device)
    return om.view(-1, 1, 1) * eye6 + kap.view(-1, 1, 1) * (u.unsqueeze(2) * u.unsqueeze(1))


def surround_prior_hessian(t_param: torch.Tensor, omega: torch.Tensor, weight: float, delta_mm: float,
                           rot_radius_mm: float = 50.0, shrinkage: float = 0.0, form: str = "charbonnier") -> torch.Tensor:
    """Analytic per-frame Hessian blocks (B, 6, 6) of the surround (second-difference) head prior, in parameter units
    (mm, rad), columns ordered (t_x, t_y, t_z, omega_x, omega_y, omega_z).

    The prior is ``sum_t rho(u_t)`` with ``u_t = m_t - 0.5 (m_{t-1} + m_{t+1})`` in mm (rotations times ``rot_radius_mm``) and
    ``rho`` one of the ``surround_rho`` forms (``form``); each term's Hessian model is ``omega I + kappa u u^T`` (exact for
    ``charbonnier`` and ``sparse``, the IRLS majoriser for ``cauchy``). Frame k enters its own term with coefficient 1 and its
    neighbours' terms with 0.5, so its diagonal block is the sum of those terms' Hessians times the squared coefficients.
    The Newton stage adds this block to the image Hessian: a central difference that shifts one axis of every frame
    together cannot see a second-difference prior (its gradient is invariant to a uniform shift). Checked against
    ``torch.autograd`` in ``tests/test_motion_design.py``.
    """
    B = int(t_param.shape[0])
    f32 = dict(dtype=t_param.dtype, device=t_param.device)
    Hb = torch.zeros((B, 6, 6), **f32)
    unit = torch.tensor([1.0, 1.0, 1.0, rot_radius_mm, rot_radius_mm, rot_radius_mm], **f32)
    if float(shrinkage) > 0.0:
        # shrinkage toward identity: mu (|t|^2 + |r omega|^2) per frame, Hessian 2 mu diag(1, 1, 1, r^2, r^2, r^2)
        Hb += 2.0 * float(shrinkage) * torch.diag(unit * unit)
    if B < 3 or float(weight) <= 0.0:
        return Hb
    with torch.no_grad():
        Hu = _surround_hessian_terms(t_param, omega, weight, delta_mm, rot_radius_mm, form)
        Hs = torch.zeros((B, 6, 6), **f32)
        Hs[1:-1] += Hu                                  # own term, coefficient 1
        Hs[:-2] += 0.25 * Hu                            # as frame k-1 of the term centred on k, coefficient 0.5
        Hs[2:] += 0.25 * Hu                             # as frame k+1, coefficient 0.5
        Hb = Hb + Hs * unit.view(1, 6, 1) * unit.view(1, 1, 6)   # d/d(omega) = r d/d(u_w)
    return Hb


def surround_prior_hessian_full(t_param: torch.Tensor, omega: torch.Tensor, weight: float, delta_mm: float,
                                rot_radius_mm: float = 50.0, shrinkage: float = 0.0, form: str = "charbonnier") -> torch.Tensor:
    """The FULL Hessian (B, 6, B, 6) of the surround prior in parameter units: the diagonal blocks of
    ``surround_prior_hessian`` plus the couplings between the frames of each term (frame t with t +- 1, coefficient
    1 x -0.5; t - 1 with t + 1, coefficient 0.25). The Newton stage solves the coupled system with it: a per-frame
    step judged frame by frame is a Jacobi iteration on a strongly coupled chain and stalls."""
    B = int(t_param.shape[0])
    f32 = dict(dtype=t_param.dtype, device=t_param.device)
    H = torch.zeros((B, 6, B, 6), **f32)
    if float(shrinkage) > 0.0:
        unit0 = torch.tensor([1.0, 1.0, 1.0, rot_radius_mm, rot_radius_mm, rot_radius_mm], **f32)
        for b in range(B):
            H[b, :, b, :] += 2.0 * float(shrinkage) * torch.diag(unit0 * unit0)
    if B < 3 or float(weight) <= 0.0:
        return H
    with torch.no_grad():
        Hu = _surround_hessian_terms(t_param, omega, weight, delta_mm, rot_radius_mm, form)
        unit = torch.tensor([1.0, 1.0, 1.0, rot_radius_mm, rot_radius_mm, rot_radius_mm], **f32)
        Hu = Hu * unit.view(1, 6, 1) * unit.view(1, 1, 6)
        coef = {-1: -0.5, 0: 1.0, 1: -0.5}
        for j in range(B - 2):
            t = j + 1
            for a, ca in coef.items():
                for b, cb in coef.items():
                    H[t + a, :, t + b, :] += ca * cb * Hu[j]
    return H


def _run_converging_stage(*, optimizer, params, per_frame_params, loss_fn, points_fn, iters, B, dev, tol, max_extend,
                          lr_groups=None, lr_t=None, lr_r=0.0, point_radius=0.0, probe_fn=None, confirm_vox=None, axes=None, stage_voxel=None,
                          max_step_vox=1.0, label="", level=1, shared_params=None, reg_loss_fn=None, resample_fn=None,
                          reg_hess_fn=None, reg_frame_fn=None, reg_hess_full_fn=None):
    """One optimiser stage that ends when every frame has CONVERGED, not when an iteration count is spent.

    ``loss_fn()`` returns the (B,) per-frame loss; ``points_fn()`` the (B, S, 3) physical positions of a fixed subset of the
    sample points under the current parameters (no grad). Convergence is the ``PointStepTracker`` test (RMS point movement
    between checkpoints < ``tol`` mm twice in a row). A converged frame is FROZEN: it leaves the loss (so it cannot influence
    the other frames through the shared Adam moments' scale or the batch's single L-BFGS line search) and its per-frame
    parameters are restored after every step. ``iters`` is the size of one block; a stage not converged after one block is
    extended (Adam: a new, halved cosine cycle; L-BFGS: more blocks) up to ``max_extend`` blocks, then reported as capped.
    ``resample_fn`` (optional) re-draws the metric's sample points; it is called at every convergence checkpoint (Adam),
    after every block (L-BFGS, whose curvature history is then reset) and before every iteration (Newton), so the
    stage's optimum is not that of one fixed point subset.
    ``reg_loss_fn`` (optional) is a prior on the parameters that may couple frames; it is part of the objective in EVERY
    optimiser: Adam and L-BFGS through the loss, Newton through its gradient, its analytic per-frame Hessian block
    ``reg_hess_fn()`` (B, P, P in parameter units, ``axes`` order) and its per-frame attribution ``reg_frame_fn()`` (B,) in
    the line search. A stage that dropped the prior would move the solution off the optimum of the stated objective.
    Returns the stage record: iterations used, blocks, per-frame converged flag and final step (mm).
    """
    if int(iters) <= 0:   # a zero-iteration stage is a stage the caller switched off: nothing to converge, nothing to flag
        return {"stage": label, "level": int(level), "iterations": 0, "blocks": 0, "tol_mm": float(tol), "probe_rejections": 0, "skipped": True,
                "converged": [True] * B, "final_step_mm": [0.0] * B}
    if int(level) != 1:   # coarse levels only SEED the finest one: one block with the early stop, never extended
        max_extend = 1
    tracker = PointStepTracker(tol=tol, patience=2)
    tracker.start(points_fn())
    frozen = {id(p): p.detach().clone() for p in per_frame_params}
    shared_list = list(shared_params) if shared_params else []
    frozen_shared = {id(p): p.detach().clone() for p in shared_list}
    shared_is_frozen = [False]
    min_active_for_shared = max(6, B // 2)
    done, blocks = 0, 0

    n_rejected = [0]

    def _confirm(newly):
        """A small step is only evidence of convergence: the manifold probe measures the remaining distance (first-order Newton
        step, in voxels). Frames still far from the optimum (a slow valley, a plateau) go back to being active."""
        if probe_fn is None or not bool(newly.any()):
            return newly
        pr = probe_fn()
        far = newly & ~(pr.remaining_vox <= confirm_vox)
        if bool(far.any()):
            tracker.reject(far)
            n_rejected[0] += int(far.sum())
        return newly & ~far

    def _freeze(newly):
        newly = _confirm(newly)
        if bool(newly.any()):
            for p in per_frame_params:
                frozen[id(p)][newly] = p.detach()[newly]
        if shared_list and not shared_is_frozen[0] and int(tracker.active.sum()) < min_active_for_shared:
            for p in shared_list:
                frozen_shared[id(p)] = p.detach().clone()
            shared_is_frozen[0] = True

    def _restore():
        if not bool((~tracker.active).any()) and not shared_is_frozen[0]:
            return
        with torch.no_grad():
            if bool((~tracker.active).any()):
                for p in per_frame_params:
                    p.copy_(torch.where((~tracker.active).view(-1, *([1] * (p.dim() - 1))), frozen[id(p)], p))
            if shared_is_frozen[0]:
                for p in shared_list:
                    p.copy_(frozen_shared[id(p)])

    def _active_loss():
        base_loss = loss_fn()[tracker.active].sum()
        if reg_loss_fn is not None:
            base_loss = base_loss + reg_loss_fn()
        return base_loss

    if optimizer == "newton":
        # Saddle-free Newton with a line search, in units of the physical displacement of the sampled points. Per iteration, for
        # all active frames at once: the gradient (autograd) and the full PxP Hessian (central differences of the gradient at
        # +-0.5 voxel along each axis, P = 6 for a rigid pose), the step -|H|^-1 g from the eigendecomposition (negative or
        # tiny curvature is replaced by its magnitude, so a saddle or plateau never gives an ascent direction) with
        # Levenberg-Marquardt damping, clipped to a per-frame trust radius, and a backtracking line search on the loss itself
        # along that direction. A frame converges when its Newton step is shorter than ``confirm_vox`` voxels (the remaining
        # distance to the optimum, to second order). No learning rate, no schedule; every quantity is in voxels.
        P = len(axes)
        vox = float(stage_voxel)
        scale = torch.tensor([float(ax.scale_mm) for ax in axes], device=dev)
        to_vox = vox / scale                                           # d(loss)/d(voxels of displacement) = d(loss)/d(param) * to_vox
        leaves = []
        for ax in axes:
            if not any(ax.tensor is lf for lf in leaves):
                leaves.append(ax.tensor)

        def _grad_param(include_reg=True):
            with torch.enable_grad():
                total = loss_fn()[tracker.active].sum()
                if include_reg and reg_loss_fn is not None:
                    total = total + reg_loss_fn()
                gs = torch.autograd.grad(total, leaves, allow_unused=True)
            cols = []
            for ax in axes:
                g = gs[[i for i, lf in enumerate(leaves) if lf is ax.tensor][0]]
                cols.append(torch.zeros(B, device=dev) if g is None else g[ax.index])
            return torch.stack(cols, dim=1)                            # (B, P) in parameter units

        def _shift(a, delta):
            with torch.no_grad():
                axes[a].tensor[axes[a].index] = axes[a].tensor[axes[a].index] + delta

        trust = torch.full((B,), float(max_step_vox), device=dev)
        lam = torch.zeros(B, device=dev)
        calls = max(1, int(iters)) * max_extend
        for call in range(calls):
            if resample_fn is not None and call > 0:
                resample_fn()
            g_par = _grad_param()                                          # gradient of the TOTAL objective (image + prior)
            # the image Hessian by central differences of the IMAGE gradient alone (the prior's Hessian is analytic and is added
            # exactly once below: the diagonal block when the stage is uncoupled, the full block-tridiagonal matrix when coupled;
            # audit 2026-10-28 found the surround block counted twice and the shrinkage block three times)
            H_par = torch.zeros(B, P, P, device=dev)
            for a in range(P):
                d = 0.5 * vox / float(scale[a])
                _shift(a, +d); gp = _grad_param(include_reg=False)
                _shift(a, -2 * d); gm = _grad_param(include_reg=False)
                _shift(a, +d)
                H_par[:, :, a] = (gp - gm) / (2.0 * d)
            H_par = 0.5 * (H_par + H_par.transpose(1, 2))
            H_img_par = H_par
            if reg_hess_fn is not None and reg_hess_full_fn is None:
                H_par = H_par + reg_hess_fn()
            Gv = g_par * to_vox
            Hv = H_par * to_vox.view(1, P, 1) * to_vox.view(1, 1, P)
            ev, Q = (t.to(dev) for t in torch.linalg.eigh(Hv.detach().cpu()))   # B x 6 x 6: eigh is not implemented on MPS, and is tiny
            floor = 1e-3 * ev.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
            inv = 1.0 / (torch.maximum(ev.abs(), floor) + lam.unsqueeze(1))
            inv0 = 1.0 / torch.maximum(ev.abs(), floor)                   # undamped: the convergence test must not see the damping
            if reg_hess_full_fn is None:
                dv = -(Q @ (inv * (Q.transpose(1, 2) @ Gv.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)   # (B, P) voxels
                dv0 = -(Q @ (inv0 * (Q.transpose(1, 2) @ Gv.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
            else:
                # coupled Newton step: the image Hessian per frame (saddle-free, damped) on the block diagonal, the prior's
                # full block-tridiagonal Hessian between frames (added ONCE); solved over the ACTIVE frames (frozen ones fixed)
                Hv_img = Q @ torch.diag_embed(torch.maximum(ev.abs(), floor) + lam.unsqueeze(1)) @ Q.transpose(1, 2)
                Hv_img0 = Q @ torch.diag_embed(torch.maximum(ev.abs(), floor)) @ Q.transpose(1, 2)
                Hp = reg_hess_full_fn() * to_vox.view(1, P, 1, 1) * to_vox.view(1, 1, 1, P)       # (B,P,B,P) voxel units
                Hfull = Hp.clone(); Hfull0 = Hp.clone()
                for b in range(B):
                    Hfull[b, :, b, :] = Hfull[b, :, b, :] + Hv_img[b]
                    Hfull0[b, :, b, :] = Hfull0[b, :, b, :] + Hv_img0[b]
                act = torch.nonzero(tracker.active).flatten()
                dv = torch.zeros(B, P, device=dev); dv0 = torch.zeros(B, P, device=dev)
                if act.numel() > 0:
                    ga = Gv[act].reshape(-1).cpu().double()
                    for Hsrc, target in ((Hfull, dv), (Hfull0, dv0)):
                        Ha = Hsrc[act][:, :, act, :].reshape(act.numel() * P, act.numel() * P).cpu().double()
                        sol = torch.linalg.solve(Ha + 1e-9 * torch.eye(Ha.shape[0], dtype=Ha.dtype), -ga).float().view(act.numel(), P)
                        target[act] = sol.to(dev)
            # convergence is judged on the UNDAMPED Newton step (the remaining distance to the optimum to second order); the
            # damped step shrinks with every rejected line search and would declare a stuck frame converged (audit 2026-10-28)
            remaining = dv0.abs().max(dim=1).values
            tracker.last_step = torch.where(tracker.active, remaining * vox, tracker.last_step)
            done_now = tracker.active & (remaining <= confirm_vox)
            if bool(done_now.any()):          # the Newton step is its own evidence of convergence: no extra probe
                tracker.active = tracker.active & ~done_now
            if tracker.all_converged:
                done += 1
                break
            dv = dv * torch.clamp(trust.unsqueeze(1) / dv.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0) * tracker.active.unsqueeze(1).float()
            with torch.no_grad():
                base = [ax.tensor[ax.index].clone() for ax in axes]
                _f_frame = (lambda: loss_fn().detach() + reg_frame_fn()) if reg_frame_fn is not None else (lambda: loss_fn().detach())
                f0 = _f_frame()
                best_f, best_a = f0.clone(), torch.zeros(B, device=dev)
                coupled = reg_hess_full_fn is not None
                tot0 = float(loss_fn().detach()[tracker.active].sum() + reg_loss_fn().detach()) if coupled else None
                best_tot, best_alpha = tot0, 0.0
                for alpha in (1.0, 0.5, 0.25, 0.1):
                    for k, ax in enumerate(axes):
                        ax.tensor[ax.index] = base[k] + alpha * dv[:, k] * to_vox[k]   # voxels of displacement -> parameter units (dp = dv * dparam/dvox)
                    if coupled:
                        # one objective, one line search: the frames' steps are a joint step of the coupled system
                        tot = float(loss_fn().detach()[tracker.active].sum() + reg_loss_fn().detach())
                        if tot < best_tot:
                            best_tot, best_alpha = tot, alpha
                        continue
                    f = _f_frame()
                    better = tracker.active & (f < best_f)
                    best_f = torch.where(better, f, best_f)
                    best_a = torch.where(better, torch.full_like(best_a, alpha), best_a)
                if coupled:
                    best_a = torch.where(tracker.active, torch.full_like(best_a, best_alpha), best_a)
                moved = best_a > 0
                for k, ax in enumerate(axes):
                    step_k = best_a * dv[:, k] * to_vox[k]
                    ax.tensor[ax.index] = torch.where(moved & torch.isfinite(step_k), base[k] + step_k, base[k])
            n_rejected[0] += int((tracker.active & ~moved).sum())
            full_step = best_a >= 1.0
            ev_max = ev.abs().max(dim=1).values.clamp_min(1e-12)
            trust = torch.where(full_step, torch.clamp(trust * 2.0, max=4.0 * float(max_step_vox)), torch.where(moved, trust, torch.clamp(trust * 0.5, min=1e-4)))
            lam = torch.where(moved, lam * 0.5, torch.clamp(lam * 4.0 + 1e-3 * ev_max, max=1e6 * ev_max))
            done += 1
            if not bool(moved.any()) and resample_fn is None and float(dv[tracker.active].abs().max()) <= 1e-3:
                break
        blocks = math.ceil((call + 1) / max(1, int(iters)))
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=_CHECK_EVERY_LBFGS, history_size=10, tolerance_grad=1e-9,
                                tolerance_change=1e-11, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            l = _active_loss()
            l.backward()
            return l

        n_blocks = max(1, math.ceil(iters / _CHECK_EVERY_LBFGS))
        for call in range(n_blocks * max_extend):
            opt.step(closure)
            done += _CHECK_EVERY_LBFGS
            _freeze(tracker.update(points_fn()))
            _restore()
            if tracker.all_converged:
                break
            if resample_fn is not None:
                resample_fn()
                opt = torch.optim.LBFGS(params, lr=1.0, max_iter=_CHECK_EVERY_LBFGS, history_size=10, tolerance_grad=1e-9,
                                        tolerance_change=1e-11, line_search_fn="strong_wolfe")
        blocks = math.ceil((call + 1) / n_blocks)
    else:
        lr_scale = 1.0
        last_check = 0
        for cycle in range(max_extend):
            blocks = cycle + 1
            groups = [dict(g, lr=g["lr"] * lr_scale) for g in lr_groups]
            opt = torch.optim.Adam(groups)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(i, iters) / iters)))
            for i in range(iters):
                opt.zero_grad(set_to_none=True)
                _active_loss().backward()
                opt.step()
                sched.step()
                _restore()
                done += 1
                if done % _CHECK_EVERY_ADAM == 0 or i == iters - 1:
                    # the farthest the points could have moved in this block: iterations x (translation lr + rotation lr x radius)
                    cur = 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * (i + 1) / iters))   # cosine fraction of the cycle's lr
                    cap = (done - last_check) * cur * lr_scale * (lr_t + lr_r * point_radius)
                    last_check = done
                    _freeze(tracker.update(points_fn(), capacity=cap))
                    _restore()
                    if tracker.all_converged:
                        break
                    if resample_fn is not None:
                        resample_fn()
            if tracker.all_converged:
                break
            lr_scale *= 0.5
    return {"stage": label, "level": int(level), "iterations": int(done), "blocks": int(blocks), "tol_mm": float(tol), "probe_rejections": int(n_rejected[0]),
            "converged": (~tracker.active).cpu().numpy().tolist(),
            "final_step_mm": tracker.last_step.cpu().numpy().tolist()}


def batched_rigid_register_pass(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    max_batch_frames: Optional[int] = None,
    seed_mode: str = "wide",
    schedule_mode: str = "default",
    metric: str = "mattes",
    huber_delta: float = 0.1,
    schedule: Optional[list] = None,
    pe_affine_covariates: Optional[np.ndarray] = None,
    pe_axis: Optional[np.ndarray] = None,
    return_pe_affine: bool = False,
    sample_mask: Optional[np.ndarray] = None,
    bias_covariates: Optional[np.ndarray] = None,
    sample_fraction: float = 0.15,
    sampling: str = DEFAULT_SAMPLING,
    resample_points: bool = False,
    sample_seed: int = 42,
    convergence_tol_vox: float = 0.05,
    max_extend_factor: int = 4,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    surround_weight: Optional[float] = None,
    surround_delta_mm: float = 0.25,
    frame_levels: Optional[np.ndarray] = None,
    init_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    design_matrix: Optional[np.ndarray] = None,
    design_point_scale: bool = True,
    design_levels: Optional[np.ndarray] = None,
    design_model_live: bool = False,
    surround_scale: str = "raw",
    surround_form: str = "charbonnier",
    design_gauge: Optional[np.ndarray] = None,
    pose_shrinkage: float = 0.0,
    frame_gain: bool = False,
):
    """Rigidly register every frame in ``moving_imgs`` to the same ``reference_img``, in one
    or more batched optimisations, optionally with a per-frame eddy-current (phase-encode affine) term whose
    coefficients are tied across frames through ``pe_affine_covariates`` (see ``_batched_rigid_register_pass_core``).
    With ``return_pe_affine=True`` a fourth value, the per-frame PE-affine record (or ``None``), is returned.

    Long series are automatically split into chunks (see ``max_batch_frames``) so that no
    single internal batched tensor grows large enough to overflow MPS's gradient kernel (see
    ``_MAX_BATCH_TENSOR_ELEMENTS``); each chunk is registered to the same ``reference_img``
    independently and results are concatenated in the original frame order. Each frame's loss
    and gradient are already independent of every other frame (block-diagonal), so chunking
    doesn't change what's being optimised -- but the LBFGS stages' quasi-Newton history is
    taken over the whole batch's flattened parameter vector, so it isn't exactly
    block-diagonal in the optimiser's internal state; a chunked run can therefore converge to
    a very slightly different (not worse) solution than one giant batch, same as re-ordering
    frames across two unrelated LBFGS runs would. Transparent to callers either way -- passing
    6 frames or 600 frames both just work and recover the same shifts to the same accuracy.

    Parameters
    ----------
    max_batch_frames : int, optional
        Maximum number of frames registered in a single internal batched call. Default
        (``None``) auto-computes a safe value from ``reference_img``'s voxel count so the
        batch's total tensor size stays under ``_MAX_BATCH_TENSOR_ELEMENTS``. Pass an explicit
        smaller value (e.g. a handful of frames for a sliding-window scheme) to force smaller
        chunks than the auto-computed safe maximum.
    reference_img, moving_imgs, device, num_bins, verbose, outprefix
        See ``_batched_rigid_register_pass_core`` (this function's single-batch workhorse)
        for the full parameter/return documentation -- identical here, just chunked.
    """
    B_total = len(moving_imgs)
    if B_total == 0:
        return [], [], 0.0

    ref_is_list = isinstance(reference_img, (list, tuple))  # one reference per frame (same grid), see the core's docstring
    if ref_is_list and len(reference_img) != B_total:
        raise ValueError(f"{len(reference_img)} per-frame references for {B_total} frames")
    vol_numel = 1
    for s in (reference_img[0] if ref_is_list else reference_img).shape:
        vol_numel *= int(s)
    auto_chunk = max(1, _MAX_BATCH_TENSOR_ELEMENTS // max(vol_numel, 1))
    chunk_size = max_batch_frames if max_batch_frames is not None else min(B_total, auto_chunk)

    cov_all = None if pe_affine_covariates is None else np.asarray(pe_affine_covariates, dtype=np.float64).reshape(B_total, -1)
    if encoding_covariates is not None:
        try:
            enc_all = np.asarray(encoding_covariates, dtype=np.float64).reshape(B_total, -1)
        except (ValueError, TypeError):
            enc_all = build_encoding_covariates(state_labels=encoding_covariates, num_frames=B_total)["E"]
    else:
        enc_all = None
    if chunk_size >= B_total:
        out = _batched_rigid_register_pass_core(
            reference_img, moving_imgs, device=device, num_bins=num_bins,
            verbose=verbose, outprefix=outprefix,
            seed_mode=seed_mode, schedule_mode=schedule_mode, metric=metric, huber_delta=huber_delta, schedule=schedule,
            pe_affine_covariates=cov_all, pe_axis=pe_axis, sample_mask=sample_mask, bias_covariates=bias_covariates,
            sample_fraction=sample_fraction, sampling=sampling, resample_points=resample_points, sample_seed=sample_seed,
            convergence_tol_vox=convergence_tol_vox, max_extend_factor=max_extend_factor,
            encoding_covariates=enc_all, window_size=window_size, window_stride=window_stride,
            surround_weight=surround_weight, surround_delta_mm=surround_delta_mm, frame_levels=frame_levels,
            init_pose=init_pose, design_matrix=design_matrix, design_point_scale=design_point_scale,
            design_levels=design_levels, design_model_live=design_model_live, surround_scale=surround_scale, surround_form=surround_form,
            design_gauge=design_gauge, pose_shrinkage=pose_shrinkage, frame_gain=frame_gain,
        )
        return out if return_pe_affine else out[:3]

    n_chunks = math.ceil(B_total / chunk_size)
    if verbose:
        print(f"[syntx.motion_batched] {B_total} frames exceeds the safe single-batch size "
              f"({chunk_size}, for a {vol_numel}-voxel volume) -- splitting into {n_chunks} "
              f"chunks of <= {chunk_size} frames each, each registered independently to the "
              f"same reference.")

    fwd_all: List[List[str]] = []
    inv_all: List[List[str]] = []
    elapsed_total = 0.0
    pe_chunks = []
    starts = list(range(0, B_total, chunk_size))
    for c, start in enumerate(starts):
        chunk_imgs = moving_imgs[start:start + chunk_size]
        chunk_prefix = f"{outprefix}c{c:03d}_" if outprefix is not None else None
        fwd_c, inv_c, elapsed_c, pe_c = _batched_rigid_register_pass_core(
            list(reference_img)[start:start + chunk_size] if ref_is_list else reference_img, chunk_imgs, device=device, num_bins=num_bins,
            verbose=verbose, outprefix=chunk_prefix,
            seed_mode=seed_mode, schedule_mode=schedule_mode, metric=metric, huber_delta=huber_delta, schedule=schedule,
            pe_affine_covariates=None if cov_all is None else cov_all[start:start + chunk_size], pe_axis=pe_axis,
            sample_mask=sample_mask,
            bias_covariates=None if bias_covariates is None else np.asarray(bias_covariates)[start:start + chunk_size],
            sample_fraction=sample_fraction, sampling=sampling, resample_points=resample_points, sample_seed=sample_seed,
            convergence_tol_vox=convergence_tol_vox, max_extend_factor=max_extend_factor,
            encoding_covariates=None if enc_all is None else enc_all[start:start + chunk_size],
            window_size=window_size, window_stride=window_stride,
            surround_weight=surround_weight, surround_delta_mm=surround_delta_mm,
            frame_levels=None if frame_levels is None else np.asarray(frame_levels)[start:start + chunk_size],
            init_pose=None if init_pose is None else (np.asarray(init_pose[0])[start:start + chunk_size],
                                                      np.asarray(init_pose[1])[start:start + chunk_size]),
            # a chunk fits its design model on its own frames only (chunking starts above 1.5e9 elements; recorded)
            design_matrix=None if design_matrix is None else np.asarray(design_matrix)[start:start + chunk_size],
            design_point_scale=design_point_scale,
            design_levels=None if design_levels is None else np.asarray(design_levels)[start:start + chunk_size],
            design_model_live=design_model_live, surround_scale=surround_scale, surround_form=surround_form,
            design_gauge=None if design_gauge is None else np.asarray(design_gauge)[start:start + chunk_size],
            pose_shrinkage=pose_shrinkage, frame_gain=frame_gain,
        )
        fwd_all.extend(fwd_c)
        inv_all.extend(inv_c)
        elapsed_total += elapsed_c
        pe_chunks.append(pe_c)
    pe_info = None
    if any(pc is not None for pc in pe_chunks):
        pe_info = {}
        sizes = [len(moving_imgs[st:st + chunk_size]) for st in starts]
        first = next(pc for pc in pe_chunks if pc is not None)
        if "sampling" in first:
            pe_info["sampling"] = first["sampling"]
        if "bias" in first:
            pe_info["bias"] = first["bias"]  # per-chunk fit; the first chunk's record is reported
        enc_chunks = [None if pc is None else pc.get("encoding_pose_bias_mm_rad") for pc in pe_chunks]
        if any(ec is not None for ec in enc_chunks):
            # each chunk is an independent fit: one B_win per chunk, delta_Y concatenated in frame order (zeros where off)
            dY = np.concatenate([np.zeros((n, 6)) if ec is None else np.asarray(ec["delta_Y_mm_rad"]) for ec, n in zip(enc_chunks, sizes)])
            pe_info["encoding_pose_bias_mm_rad"] = {
                "delta_Y_mm_rad": dY,
                "delta_Y": dY.tolist(),
                "chunks": enc_chunks,
                "removed_rms_mm": float(np.sqrt((dY[:, :3] ** 2).sum(axis=1).mean())),
                "removed_rms_deg": float(np.degrees(np.sqrt((dY[:, 3:] ** 2).sum(axis=1).mean()))),
            }
            if "bias" not in pe_info:
                pe_info["bias"] = pe_info["encoding_pose_bias_mm_rad"]
        if "convergence" in first:
            pe_info["convergence"] = [pc["convergence"] for pc in pe_chunks if pc is not None]  # one stage list per chunk
        if any("a" in pc for pc in pe_chunks if pc is not None):
            # the tied eddy coefficients are estimated per chunk (each chunk is an independent fit): one matrix per chunk
            pe_info.update({
                "axis": next(pc for pc in pe_chunks if pc is not None and "axis" in pc)["axis"],
                "coefficients": [None if pc is None or "coefficients" not in pc else pc["coefficients"] for pc in pe_chunks],
                "a": np.concatenate([np.zeros((n, 4)) if pc is None or "a" not in pc else pc["a"] for pc, n in zip(pe_chunks, sizes)]),
                "jacobian": np.concatenate([np.ones(n) if pc is None or "jacobian" not in pc else pc["jacobian"] for pc, n in zip(pe_chunks, sizes)]),
            })
    return (fwd_all, inv_all, elapsed_total, pe_info) if return_pe_affine else (fwd_all, inv_all, elapsed_total)


def _batched_rigid_register_pass_core(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    seed_mode: str = "wide",
    schedule_mode: str = "default",
    _level_builder=None,
    metric: str = "mattes",
    huber_delta: float = 0.1,
    schedule: Optional[list] = None,
    pe_affine_covariates: Optional[np.ndarray] = None,
    pe_axis: Optional[np.ndarray] = None,
    sample_mask: Optional[np.ndarray] = None,
    bias_covariates: Optional[np.ndarray] = None,
    sample_fraction: float = 0.15,
    sampling: str = DEFAULT_SAMPLING,
    resample_points: bool = False,
    sample_seed: int = 42,
    convergence_tol_vox: float = 0.05,
    max_extend_factor: int = 4,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    surround_weight: Optional[float] = None,
    surround_delta_mm: float = 0.25,
    frame_levels: Optional[np.ndarray] = None,
    init_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    design_matrix: Optional[np.ndarray] = None,
    design_point_scale: bool = True,
    design_levels: Optional[np.ndarray] = None,
    design_model_live: bool = False,
    surround_scale: str = "raw",
    surround_form: str = "charbonnier",
    design_gauge: Optional[np.ndarray] = None,
    pose_shrinkage: float = 0.0,
    frame_gain: bool = False,
) -> Tuple[List[List[str]], List[List[str]], float, Optional[dict]]:
    """
    Rigidly register every frame in ``moving_imgs`` to the same ``reference_img`` in one
    batched optimisation; returns ``(fwd_transforms, inv_transforms, elapsed, pe_affine)``.

    Eddy-current term (``pe_affine_covariates`` (B, k) + ``pe_axis``; DWI: the gradient vectors and the phase-encode
    axis). The residual eddy-current field of an EPI volume displaces it along the phase-encode axis ``p`` by an amount
    affine in position -- a translation, two shears and a scale along ``p`` -- and these four coefficients depend on the
    diffusion gradient, to first order linearly: ``a_k = C @ g_k`` with one (4, k) matrix ``C`` shared by all frames of
    the call (a shell). Head motion is free per frame and independent of ``g``; the eddy term is tied through ``g`` and
    independent of time, which is what makes the two identifiable jointly. The point map of frame ``k`` is
    ``y = R_k (x - c) + c + t_k`` followed by ``y' = y + p * (a_k0 + a_k[1:4] . (y - c))``; the resampled frame is
    multiplied by the Jacobian ``1 + a_k[1:4] . p`` (signal is conserved under the distortion). Frames whose covariates
    are all zero (b0) get no eddy term. Output per frame: ``[rigid.mat, eddy_affine.mat]`` (a transform list acts on
    the reference-space point first-listed first, so this reproduces the fitted ``eddy(rigid(x))``; order corrected
    2026-10-28, before that the chain was written ``[eddy, rigid]`` and resampled ``rigid(eddy(x))``) or ``[rigid.mat]``
    when the term is zero; FD and the reported pose come from the rigid file alone, found by name
    (``rigid_transform_of``), never by position.
    ``pe_affine`` = ``{'axis', 'coefficients' (4, k), 'a' (B, 4), 'jacobian' (B,)}`` or ``None``.

    ``sampling``: the sample-point strategy from the ``syntx.sampling`` menu (default ``DEFAULT_SAMPLING`` =
    ``"gradient_floor"``: candidates above the reference's noise floor and below its saturated top, drawn proportional
    to ``0.1 + |grad n| / max``; ``"uniform"`` / ``"gradient"`` are the other entries). ``resample_points``: re-draw each stage's
    subset at every convergence checkpoint (default False: one subset per stage).
    ``sample_seed``: seed of the sample-point draws (default 42; changes only which points are drawn).
    ``sample_fraction``: the fraction of the strategy's candidate region (within ``sample_mask``) that EACH
    full-resolution metric evaluation sees. The stages draw ``_STAGE_SAMPLING`` (20 %) of a
    pool, so the pool holds ``sample_fraction / 0.2`` of the region (0.10 -> 50 %), at most ``MAX_EVAL_POINTS`` (20 000)
    per evaluation; with ``resample_points`` the stage subset is re-drawn at every convergence checkpoint. It sets the
    precision floor of every pose. Before 6.22 it sized the pool
    itself (0.15 -> 15 % pool x 20 % stage = 3 % per evaluation, fixed for the whole run), which left a one-subset
    local optimum on the slow-flow phantom. ``1.0`` evaluates every masked voxel at every full-resolution stage.

    ``sample_mask`` (fixed-space, same grid as the reference; ANTsImage or array, > 0 = use): the full-resolution stages
    evaluate the metric on sample points drawn from these voxels only (the coarse capture-range stages use the whole
    foreground).
    The points are warped into each frame by the pose being solved, so the mask needs to be aligned to the reference
    only, never to the moving frames. DWI: gray matter + CSF, whose signal does not depend on the gradient direction;
    the white-matter signal does, and registering a diffusion volume against a reference of another direction on
    white matter biases the pose (still phantom, sub-BLAST079 b3000, real contrast, measured noise: pose error RMS
    0.86 mm on all brain voxels, 0.27 mm on direction-independent voxels; b1500 0.20 -> 0.12).

    ``bias_covariates`` (B, k): a deterministic registration bias that is a known function of per-frame covariates
    (DWI: the even spherical-harmonic terms of the gradient direction, l = 2 -- what remains of the direction-dependent
    contrast once the metric is masked) is estimated after the fit from the second-order surround differences of the 6
    rigid parameters against those of the covariates, with Tukey weights (``_project_encoding_surround_tukey_np``: the
    group's mean pose and smooth drift are untouched, jumps downweighted) and REMOVED FROM THE WRITTEN POSES.
    It is registration error, not displacement, so unlike the eddy term it must not be applied to the data. Head
    motion is independent of the covariates, which is what makes it separable. Measured on the still phantom: 0.27 ->
    0.18 mm (b3000, masked), 0.115 -> 0.088 (b1500). Reported as ``pe_affine['bias']``.

    Per-frame references: ``reference_img`` may be a list of one 3-D image per frame, all on one grid (a reference
    predicted for each frame's own contrast, e.g. a spherical-harmonic prediction at the frame's gradient direction).
    The sample points, the foreground, the centre of mass and the phase-correlation seeds come from the mean of the
    references; the metric of frame b is evaluated against reference b (Mattes MI with one Parzen weight set per frame,
    Huber on the per-frame residual, Pearson per row).

    ``metric``: ``'mattes'`` (default) -- Parzen-window Mattes mutual information with ``num_bins`` bins, the right
    choice when frame and reference may differ in contrast; ``'huber'`` -- the Huber loss
    (``syntx.core.losses.huber_loss_nd``, transition ``huber_delta`` in units of the reference's [-1, 1]-scaled
    intensity) on the pointwise residual between the warped frame and the reference, for SAME-contrast series (each
    DWI shell against its own shell mean, BOLD against its mean): quadratic for residuals inside ``huber_delta``,
    linear beyond, so dropout voxels and the partial-volume rim cannot dominate the pose the way they can under MSE,
    and -- unlike MI -- the loss has no flat valley along directions that only reshuffle the joint histogram. Both are
    minimised (MI is returned negated). Single-chunk workhorse -- call ``batched_rigid_register_pass``
    (this module's public entry point) instead, which transparently chunks long series.

    Parameters
    ----------
    seed_mode : {'wide', 'narrow'}, default 'wide'
        'wide' (unchanged default): the full capture-range seed search below (6 translation
        candidates x 7 rotation seeds = 42 combos, each refined with 30 Adam steps) --
        appropriate when frames could be arbitrarily misaligned (cross-subject/modality
        template registration). 'narrow': skip the phase-correlation candidates and
        multi-rotation-seed search entirely, seeding only from the center-of-mass offset
        with zero rotation -- appropriate for TEMPORALLY COHERENT series (consecutive
        frames of the same subject/contrast) where real motion is a small perturbation, not
        an unknown large misalignment. Confirmed real speedup case: this is the dominant
        per-call cost (42x30=1260 Adam steps vs. the main schedule's 203), found while
        piloting a faster path for 356-frame real FDG-PET motion correction.
    schedule_mode : {'default', 'fast'}, default 'default'
        'default': Adam on the pyramid levels 4 and 2 (capture), then a Newton finisher at level 1 (L-BFGS when the
        eddy PE-affine terms are fitted): every stage ends when its frames CONVERGED (``convergence_tol_vox``), the iteration
        counts are the size of one block. 'fast': one coarse Adam stage and one L-BFGS stage -- for consumers that need
        only ~0.3 mm (coarse FD estimates). Pass ``schedule=`` for any other stage list.
    schedule : list of dict, optional
        Explicit stages overriding ``schedule_mode``. Each stage: ``optimizer`` ('adam' | 'lbfgs' | 'newton'), ``iters``
        (block size), ``level`` (pyramid shrink factor 4 / 2 / 1), ``sampling``, optional ``corr_weight``, ``freeze``
        ('r' | 't'), and for Adam ``step_t_vox`` / ``step_r_vox`` (displacement of the sampled points per step, in voxels of
        the finest working grid; converted to parameter units with ``core.parameter_scales``). 'newton' takes ``confirm_vox``
        (default ``convergence_tol_vox``) and needs no step size.

    Algorithm:

    1. Draw the sample-point pool with ``syntx.sampling.PointSampler`` (strategy ``sampling``, seeded, deterministic;
       ``sample_fraction / 0.2`` of the strategy's candidate region, at least 2000). Intensities are scaled with the
       reference range [0, max(reference)] for the histograms (assumes non-negative images).
    2. Seed search at the coarse level ``min(4, max_level)`` (``max_level`` keeps at least 8
       voxels on the smallest axis): translation candidates = centre-of-mass offset plus the
       top 5 phase-correlation peaks; rotation seeds = 0 and +-20 degrees about each axis. Each
       of the 6 x 7 combinations is refined with 30 Adam steps on the correlation loss; the
       best one is kept per frame.
    3. Schedule: Adam (80 iterations) at the coarse level and Adam (40) at level
       ``min(2, max_level)``, both on MI + correlation with 20 % point sampling and cosine
       learning-rate decay; two joint LBFGS stages (18, 15 iterations, strong-Wolfe line
       search, 20 % sampling) at full resolution; then translation-only and rotation-only
       LBFGS stages (15 each) on all sampled points. Coarse levels are ``avg_pool3d``
       reductions; sampling uses the analytic-gradient bilinear path of
       ``core.grid.grid_sample_nd``.

    Parameters
    ----------
    reference_img : ants.ANTsImage
        Shared fixed image (3-D).
    moving_imgs : list of ants.ANTsImage
        Frames to register; must have the reference's shape, spacing, origin and direction
        (not checked).
    device : str, default 'auto'
        'auto' picks cuda, then mps, then cpu; any other value is used as given.
    num_bins : int, default 18
        Mattes MI histogram bins (the session doc reports 18-20 bins working better than 32 on
        b0/DWI data). ``syntx.motion.motion_correction`` passes ``kwargs.get('num_bins', 32)``,
        so through that caller the default is 32.
    verbose : bool, default False
        Prints one line about the seed search.
    outprefix : str, optional
        Transform files are written as ``f"{outprefix}{t:04d}.mat"`` (t = index in
        ``moving_imgs``); without it, as ``vol{t:04d}.mat`` in a new temporary directory
        (not deleted).

    Returns
    -------
    fwd_transforms : list of list of str
        One single-element list per frame: the path of an ANTs ``AffineTransform`` (rotation
        about the reference's centre of mass, then translation) mapping reference-space
        points into the frame, i.e. usable directly in ``ants.apply_transforms``.
    inv_transforms : list of list of str
        The same paths (callers invert them with ``whichtoinvert``).
    elapsed : float
        Seconds spent from set-up to the end of the optimisation (file writing excluded).
        ``([], [], 0.0)`` for an empty ``moving_imgs``.

    Raises
    ------
    NotImplementedError
        If ``reference_img.dimension != 3``.
    ValueError
        A frame not on the reference grid.
    """
    ref_list = None
    if isinstance(reference_img, (list, tuple)):
        ref_list = list(reference_img)
        if len(ref_list) != len(moving_imgs):
            raise ValueError(f"{len(ref_list)} per-frame references for {len(moving_imgs)} frames")
        _check_frames_on_reference(ref_list[0], ref_list[1:])
        reference_img = ref_list[0]
    per_frame_ref = ref_list is not None
    dim = reference_img.dimension
    if dim != 3:
        raise NotImplementedError(
            "motion_batched.batched_rigid_register_pass only implements the 3D path. "
            "Use backend='pytorch' (per-frame robust_affine) or backend='ants' for 2D+t series."
        )
    _check_frames_on_reference(reference_img, moving_imgs)
    dev = torch.device(_auto_device() if device == "auto" else device)
    B = len(moving_imgs)
    if B == 0:
        return [], [], 0.0

    import time
    if metric not in ("mattes", "huber"):
        raise ValueError(f"metric must be 'mattes' or 'huber', got {metric!r}")
    t0 = time.time()

    # Presets below assume real head-sized volumes (hundreds of voxels/axis); on a small
    # volume, downsampling by a coarse level (e.g. 4x on a 16-voxel axis) leaves only a
    # handful of voxels per axis, an unstable estimate -- the exact degenerate-pyramid
    # failure mode found and fixed in robust_affine.py's single-frame solver earlier this
    # session (spurious scale contraction from a coarse-level Mattes-MI optimum on too few
    # voxels). Same fix here: clamp the coarsest usable level so the smallest spatial axis
    # keeps at least MIN_VOXELS_PER_AXIS voxels at every pyramid level actually used.
    MIN_VOXELS_PER_AXIS = 8
    max_level = max(1, min(int(s) for s in reference_img.shape) // MIN_VOXELS_PER_AXIS)
    # private seam (pyramid-policy comparison harness, benchmarks/pyramid_compare.py); None keeps the default avg_pool3d
    # levels bit-identically. A builder maps (role, arr (B,1,z,y,x), spacing_xyz, ctx) -> (tensor, f_xyz, off_xyz); it may
    # state ``min_shape_rule`` in {'voxel', 'mm', 'none'} for the coarsest-level clamp.
    _lvl_geom = {}
    if _level_builder is not None:
        _rule = getattr(_level_builder, 'min_shape_rule', None)
        if _rule == 'mm':
            _spf = min(float(x) for x in reference_img.spacing)
            max_level = max(1, min(int(n * float(sp_ / _spf)) for n, sp_ in zip(reference_img.shape, reference_img.spacing))
                            // MIN_VOXELS_PER_AXIS)
        elif _rule == 'none':
            max_level = 10 ** 6
        elif _rule not in (None, 'voxel'):
            raise ValueError(f"_level_builder.min_shape_rule must be 'voxel', 'mm' or 'none', got {_rule!r}")

    f32 = dict(dtype=torch.float32, device=dev)
    sp_xyz = torch.tensor(reference_img.spacing, **f32)
    orig_xyz = torch.tensor(reference_img.origin, **f32)
    dir_xyz = torch.tensor(reference_img.direction, **f32)
    com_f = np.asarray(compute_center_of_mass(reference_img, weighted=True), dtype=np.float64)
    C_phys = torch.tensor(com_f, **f32)

    # eddy-current PE-affine term, tied across frames through the covariates (see docstring)
    pe_on = False
    if pe_affine_covariates is not None:
        cov_np = np.asarray(pe_affine_covariates, dtype=np.float64).reshape(B, -1)
        if np.any(cov_np != 0):
            if pe_axis is None:
                raise ValueError("pe_affine_covariates needs pe_axis (physical unit vector of the phase-encode axis)")
            pe_on = True
            cov_t = torch.tensor(cov_np, **f32)
            p_vec = torch.tensor(np.asarray(pe_axis, dtype=np.float64) / np.linalg.norm(pe_axis), **f32)
            # unit scaling so the shear/scale coefficient rows are O(1) for the optimiser; a0 is solved in closed form
            # from t_final after optimisation so it does not create a translation gauge nullspace with t_param.
            pe_scale = torch.tensor([0.0, 0.01, 0.01, 0.01], **f32).unsqueeze(1)
            Ccoef = torch.zeros(4, cov_t.shape[1], device=dev, requires_grad=True)
            # Projector on the span of the eddy covariates over frames (the ODD-in-g part of a per-frame scalar): the
            # encoding gauge may carry a translation along p only in the complement, so it can never take the linear
            # eddy translation (that is the applied term, attributed from the head translations after the fit), while a
            # translation along p that is EVEN in g (direction contrast) stays a contrast term and never becomes motion.
            _XtX = cov_np.T @ cov_np
            P_odd_np = cov_np @ np.linalg.solve(_XtX + 1e-8 * np.trace(_XtX) / max(cov_np.shape[1], 1) * np.eye(cov_np.shape[1]), cov_np.T)
            P_odd_t = torch.tensor(P_odd_np, **f32)

    # Overlapping-window encoding-induced pose bias B_win (M, K_eff, 6) jointly optimised with pure head motion (t_param, omega)
    enc_on = False
    enc_basis = None
    B_win = None
    E_psi_t = None
    E_np = None
    psi_np = None
    K_eff = 0
    M_win = 1
    if encoding_covariates is not None:
        enc_basis = build_encoding_covariates(
            encoding_covariates=encoding_covariates,
            num_frames=B,
            window_size=16 if window_size is None else int(window_size),
            window_stride=window_stride,
        )
        K_eff = int(enc_basis["K_eff"])
        M_win = int(enc_basis["num_windows"])
        if K_eff > 0 and B >= 4:
            enc_on = True
            E_np = enc_basis["E"]
            psi_np = enc_basis["psi"]
            E_psi_t = torch.tensor(enc_basis["E_psi"], **f32)  # (B, M_win, K_eff)
            # B_win columns 3:6 are stored in mm of point displacement (rotation x rot_scale_mean) so one Adam step size
            # fits all six columns; enc_unit converts back to (mm, rad) and is set once rot_scale_mean is known.
            B_win = torch.zeros((M_win, K_eff, 6), device=dev, dtype=torch.float32, requires_grad=True)
    enc_unit = torch.ones(6, **f32)
    # In-loop Charbonnier surround penalty on the head trajectory: opt-in. On the 4-state slow-flow phantom
    # (tests/test_motion_overlapping_window.py) it made the head estimate worse as the weight grew (tz RMS error
    # 0.003 / 0.012 / 0.019 mm, rotation 0.108 / 0.133 / 0.157 deg at weights 0 / 0.02 / 0.05), so the default is 0.
    surr_w = 0.0 if surround_weight is None else float(surround_weight)
    surr_delta = max(float(surround_delta_mm), 1e-4)
    shrink_w = max(0.0, float(pose_shrinkage))   # shrinkage toward identity, same units and scale as the surround prior
    if surround_scale not in ("raw", "likelihood", "curvature"):
        raise ValueError(f"surround_scale must be 'raw', 'likelihood' or 'curvature', got {surround_scale!r}")
    if surround_form not in SURROUND_FORMS:
        raise ValueError(f"surround_form must be one of {SURROUND_FORMS}, got {surround_form!r}")
    # ``surround_scale='curvature'`` (metric-independent): the prior terms are multiplied by the measured curvature of the
    # image term, the median over frames and translation axes of the second derivative of the per-frame loss with respect
    # to a displacement (loss per mm^2, from ``probe_manifold`` at every stage start). A weight of 1 then means a prior
    # curvature equal to the data's own; it holds for the Huber and the mutual-information metrics alike.
    # ``surround_scale='likelihood'``: ``surround_weight`` is in nats per mm of second difference beyond the knee, i.e. a
    # log-prior in the units of the image log-likelihood (the Huber metric only: its loss is a mean of squared residuals). The image term per frame is a MEAN over the S sample points of
    # the Huber loss on scaled intensities, which is the log-likelihood divided by S / sigma^2 (sigma: the residual noise
    # in scaled units), so the prior enters the loss multiplied by sigma^2 / S. Both are measured at every stage start
    # on the stage's own points (``_prior_scale``). With ``'raw'`` the multiplier is 1 (the 6.25 behaviour).
    surr_scale = [1.0]
    prior_scale_log: list = []

    def pe_terms():
        """Per-frame eddy coefficients a (B, 4) and Jacobian (B,)."""
        a = cov_t @ (Ccoef * pe_scale).t()
        return a, 1.0 + a[:, 1:4] @ p_vec

    def apply_pe(y_phys, a):
        """y' = y + p (a0 + a[1:4] . (y - c)); y_phys (B, S, 3)."""
        disp = a[:, 0:1] + torch.einsum("bsi,bi->bs", y_phys - C_phys, a[:, 1:4])
        return y_phys + disp.unsqueeze(-1) * p_vec

    def to_tensor(img):
        arr = np.transpose(img.numpy().astype(np.float32), (2, 1, 0))
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dev)

    if per_frame_ref:
        fixed_stack_zyx = torch.stack([to_tensor(r) for r in ref_list], dim=0)  # (B, Z, Y, X)
        fixed_t_zyx = fixed_stack_zyx.mean(dim=0)  # foreground, seeds and centre of mass come from the mean reference
    else:
        fixed_stack_zyx = None
        fixed_t_zyx = to_tensor(reference_img)
    moving_stack = torch.stack([to_tensor(m) for m in moving_imgs], dim=0)
    fixed_t = fixed_t_zyx.permute(2, 1, 0)
    shape_xyz = torch.tensor(list(reference_img.shape), **f32)

    def _sub(vals, idx):
        """Index the sample axis of a reference quantity: (S,) shared or (B, S) per frame."""
        return vals[:, idx] if per_frame_ref else vals[idx]

    def _parzen(scaled):
        """Parzen weights (S, nb) for a shared reference, (B, S, nb) per frame."""
        if per_frame_ref:
            return parzen_weights(scaled.reshape(-1), num_bins=num_bins).reshape(*scaled.shape, num_bins)
        return parzen_weights(scaled, num_bins=num_bins)

    # Sample points: ONE implementation (syntx.sampling), strategy from its menu (default syntx.sampling.DEFAULT_SAMPLING:
    # candidates above the reference's noise floor and below its saturated top, gradient weights with a floor). The
    # caller's sample_mask restricts the full-resolution stages only (see the coarse levels below).
    from .sampling import PointSampler, draw_indices

    _ref_img = ants.from_numpy(np.ascontiguousarray(fixed_t.detach().cpu().numpy()), origin=reference_img.origin,
                               spacing=reference_img.spacing, direction=reference_img.direction)
    sampler = PointSampler(_ref_img, strategy=sampling, sample_mask=sample_mask, sample_fraction=sample_fraction,
                           stage_sampling=_STAGE_SAMPLING, min_points=2000, seed=int(sample_seed), device=dev)
    rng = sampler.generator
    _pool = sampler.pool()
    n_sample = sampler.n_pool

    def _interp_fixed(fixed_zyx, vox_xyz, shape_xyz_t):
        """Trilinear values of ``fixed_zyx`` ((Z, Y, X) shared or (B, Z, Y, X) per frame) at voxel positions ``vox_xyz`` (S, 3)."""
        norm = 2.0 * vox_xyz / (shape_xyz_t - 1.0) - 1.0
        src = fixed_zyx.unsqueeze(1) if fixed_zyx.dim() == 4 else fixed_zyx[None, None]
        out = F.grid_sample(src, norm.view(1, 1, 1, -1, 3).expand(src.shape[0], 1, 1, -1, 3), mode="bilinear", align_corners=True)
        return out.reshape(src.shape[0], -1) if fixed_zyx.dim() == 4 else out.reshape(-1)

    vox_idx, phys_X = _pool["vox_xyz"], _pool["phys"]
    fixed_vals = _interp_fixed(fixed_stack_zyx if per_frame_ref else fixed_t_zyx, vox_idx, shape_xyz)

    # Per-level point weights: with ``frame_levels`` (and per-frame references) each frame weights the shared pool by the
    # strategy's weights of ITS OWN level's reference, so a level whose edges the mean reference does not show (high-b
    # shells: CSF dark) is scored where its own contrast has edges. The stage draw uses the max over levels (every level's
    # edges can be drawn); the per-frame weight is the importance ratio level / shared (<= 1).
    lvl_of_frame = None
    _lvl_ref_img: dict = {}
    if frame_levels is not None:
        if not per_frame_ref:
            raise ValueError("frame_levels needs one reference per frame (reference_img as a list)")
        lvl_of_frame = np.asarray(frame_levels, dtype=int).reshape(-1)
        if lvl_of_frame.size != B:
            raise ValueError(f"frame_levels has {lvl_of_frame.size} entries for {B} frames")
        for k in np.unique(lvl_of_frame):
            _lvl_ref_img[int(k)] = ref_list[int(np.flatnonzero(lvl_of_frame == k)[0])]
    _lvl_w_cache: dict = {}

    def level_point_weights(sigma_mm, use_sample_mask):
        """(shared draw weights (S,), per-frame weights (B, S)) at a scale, or (None, None) without frame_levels."""
        if lvl_of_frame is None:
            return None, None
        key = (round(float(sigma_mm), 6), bool(use_sample_mask))
        if key not in _lvl_w_cache:
            w_k = {k: sampler.weights_of(img_k, sigma_mm, use_sample_mask) for k, img_k in _lvl_ref_img.items()}
            w_shared = torch.stack(list(w_k.values()), dim=0).max(dim=0).values
            w_frame = torch.stack([w_k[int(k)] for k in lvl_of_frame], dim=0) / w_shared.clamp_min(1e-12).unsqueeze(0)
            _lvl_w_cache[key] = (w_shared, w_frame)
        return _lvl_w_cache[key]

    lo = 0.0
    if per_frame_ref:
        # one intensity range PER REFERENCE: frames of a dark contrast (high-b shells) keep the full histogram range of
        # their own reference instead of being compressed into the low bins of the brightest reference (b0)
        hi_b = fixed_stack_zyx.reshape(B, -1).amax(dim=1).clamp_min(1e-8)          # (B,)
        hi = float(hi_b.max())
    else:
        hi = float(fixed_t.max())
        hi_b = None

    def _scale(vals, per_frame_rows):
        """[-1, 1] intensity scaling; ``vals`` (B, S) per frame or (S,) shared; the per-frame range when available."""
        if hi_b is not None and per_frame_rows:
            return (vals - lo) / (hi_b.unsqueeze(1) - lo + 1e-8) * 2.0 - 1.0
        return (vals - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    _warp_cache = {}

    def warp_to_moving_norm(y_phys, shape_lvl, level=1):
        """physical -> normalised grid coordinates of the moving tensor of pyramid ``level`` (prebuilt once per level through
        ``pyramid.prebuild_warp``; ``_lvl_geom`` holds the per-axis factor and offset of the level, ``f = 1, off = 0`` at level 1)."""
        fn = _warp_cache.get(level)
        if fn is None:
            f_m, o_m = _lvl_geom.get(level, ((1.0,) * 3, (0.0,) * 3))
            shp_zyx = [int(x) for x in shape_lvl.tolist()][::-1]
            fn = _warp_cache[level] = prebuild_warp(orig_xyz.cpu().numpy(), dir_xyz.cpu().numpy(), sp_xyz.cpu().numpy(),
                                                    f_m, o_m, shp_zyx, device=y_phys.device, dtype=y_phys.dtype)
        return fn(y_phys)

    _pyramid_cache = {}
    _grad_I_cache = {}

    def get_grad_I(moving_tensor):
        # Cached by tensor identity: `moving_tensor` objects here are always one of the few
        # pyramid-level tensors created ONCE by get_pyramid_level and reused across every
        # stage/iteration that references that level -- never recreated per-call. Recomputing
        # this on every LBFGS/Adam iteration (as the analytic-gradient grid_sample path would
        # otherwise force) was the dominant new per-iteration cost from switching to it;
        # caching restores the original amortized-cost design this module is built around.
        key = id(moving_tensor)
        if key not in _grad_I_cache:
            _grad_I_cache[key] = _image_spatial_gradient(moving_tensor.unsqueeze(1))
        return _grad_I_cache[key]


    def get_pyramid_level(level):
        """One scale's view of the SHARED physical pool (``sampler``): the level's fixed values interpolated at the pool
        points and the strategy's weights there at the level's scale (every eligible pool point; stages draw their share).
        Every scale uses the same physical points (no per-level grid)."""
        if level in _pyramid_cache:
            return _pyramid_cache[level]
        fixed_tensor_lvl = fixed_stack_zyx if per_frame_ref else fixed_t_zyx.unsqueeze(0)
        sp_min = float(min(reference_img.spacing))
        if level == 1:
            fixed_vals_all, moving_lvl, shape_lvl = fixed_vals, moving_stack, shape_xyz
            # the caller's sample_mask restricts the full-resolution stages only
            w_all = sampler.weights(sp_min, use_sample_mask=True)
            w_shared_lvl, w_frame_lvl = level_point_weights(sp_min, True)
        else:
            fixed_src = fixed_stack_zyx.unsqueeze(1) if per_frame_ref else fixed_t_zyx.unsqueeze(0).unsqueeze(0)
            if _level_builder is not None:
                _ctx = dict(level=level, sp_min_f=sp_min, sig_mm=0.0, budget=2 ** 24, device=dev)
                fixed_pool, f_fx, o_fx = _level_builder("fixed", fixed_src, reference_img.spacing, _ctx)[:3]
                moving_pool, f_mv, o_mv = _level_builder("moving", moving_stack.unsqueeze(1),
                                                         reference_img.spacing, _ctx)[:3]
                fixed_pool, moving_pool = fixed_pool[:, 0], moving_pool[:, 0]
                _lvl_geom[level] = (tuple(f_mv), tuple(o_mv))
            else:
                _sp = tuple(float(x) for x in reference_img.spacing)
                _lf = build_pyramid(fixed_src, _sp, [level], policy="mm_iso")[0]
                _lm = build_pyramid(moving_stack.unsqueeze(1), _sp, [level], policy="mm_iso")[0]
                fixed_pool, moving_pool = _lf.tensor[:, 0], _lm.tensor[:, 0]
                f_fx, o_fx = list(_lf.f), list(_lf.off)
                _lvl_geom[level] = (tuple(_lm.f), tuple(_lm.off))
            shape = fixed_pool.shape[1:]
            shape_lvl = torch.tensor([shape[2], shape[1], shape[0]], **f32)
            # the pool's full-resolution voxel coordinates in this level's grid (vox_full = vox_lvl * f + off)
            vox_lvl = (vox_idx - torch.tensor(o_fx, **f32)) / torch.tensor(f_fx, **f32)
            vox_lvl = torch.minimum(vox_lvl.clamp(min=0.0), shape_lvl - 1.0)
            fixed_vals_all = _interp_fixed(fixed_pool if per_frame_ref else fixed_pool[0], vox_lvl, shape_lvl)
            fixed_tensor_lvl, moving_lvl = fixed_pool, moving_pool
            # weights at this level's scale; the caller's sample_mask is NOT applied: the direction-dependent contrast
            # bias it removes is a fine-scale effect, while the coarse stages provide the capture range for real motion
            # and need the whole head (masking them cost 0.2 mm of relative-pose error under 1 mm / 1 deg motion)
            w_all = sampler.weights(sp_min * float(level), use_sample_mask=False)
            w_shared_lvl, w_frame_lvl = level_point_weights(sp_min * float(level), False)
            # every eligible pool point (stages draw their share of these with the level's weights). Thinning a coarse
            # level to its voxel count left the capture stages and the seed search a few hundred edge-concentrated points:
            # DWI phantom rotation error 0.83 deg (one seed 1.83) vs 0.36 deg with the whole pool (5 seeds, gradient_floor)
        if w_shared_lvl is not None:
            w_all = w_shared_lvl
        sel = torch.nonzero(w_all > 0, as_tuple=False).squeeze(1)
        entry = dict(X=phys_X[sel], w_y=None, moving=moving_lvl, shape=shape_lvl, fixed_vals=_sub(fixed_vals_all, sel),
                     fixed_tensor=fixed_tensor_lvl, w=w_all[sel],
                     w_frame=None if w_frame_lvl is None else w_frame_lvl[:, sel])
        _pyramid_cache[level] = entry
        return entry

    # -- translation + rotation seed search (large-jump capture-range fix; see session doc
    # Sec 4.2 for why both need seeding, and jointly, not rotation-only with translation
    # frozen) --
    t_init_com = np.zeros((B, 3), dtype=np.float32)
    for b, m in enumerate(moving_imgs):
        com_m = np.asarray(compute_center_of_mass(m, weighted=True), dtype=np.float64)
        t_init_com[b] = (com_m - com_f).astype(np.float32)

    def phase_correlation_candidates(level=4, topk=5, min_sep_vox=2):
        level = min(level, max_level)
        _sp = tuple(float(x) for x in reference_img.spacing)
        fixed_pool = build_pyramid(fixed_t_zyx.unsqueeze(0).unsqueeze(0), _sp, [level], policy="mm_iso")[0].tensor[0, 0]
        moving_pool = build_pyramid(moving_stack.unsqueeze(1), _sp, [level], policy="mm_iso")[0].tensor[:, 0]
        shape = fixed_pool.shape

        def hann(n):
            return torch.hann_window(n, periodic=False, device=dev, dtype=torch.float32)

        win = hann(shape[0]).view(-1, 1, 1) * hann(shape[1]).view(1, -1, 1) * hann(shape[2]).view(1, 1, -1)
        F_fixed = torch.fft.rfftn(fixed_pool * win)
        F_moving = torch.fft.rfftn(moving_pool * win.unsqueeze(0), dim=(-3, -2, -1))
        R = F_fixed.unsqueeze(0) * torch.conj(F_moving)
        R = R / (R.abs() + 1e-8)
        r = torch.fft.irfftn(R, s=shape, dim=(-3, -2, -1))

        def unwrap(idx, size):
            return torch.where(idx > size / 2, idx - size, idx)

        work = r.reshape(B, shape[0], shape[1], shape[2]).clone()
        cands = []
        for _ in range(topk):
            flat = work.reshape(B, -1)
            peak = flat.argmax(dim=1)
            pz = peak // (shape[1] * shape[2])
            rem = peak % (shape[1] * shape[2])
            py = rem // shape[2]
            px = rem % shape[2]
            pz_s = unwrap(pz.float(), shape[0]) * level
            py_s = unwrap(py.float(), shape[1]) * level
            px_s = unwrap(px.float(), shape[2]) * level
            vox_shift_xyz = torch.stack([px_s, py_s, pz_s], dim=-1)
            phys_shift = (vox_shift_xyz * sp_xyz) @ dir_xyz.t()
            cands.append(-phys_shift.cpu().numpy())
            for b in range(B):
                zc, yc, xc = int(pz[b]), int(py[b]), int(px[b])
                z0, z1 = max(0, zc - min_sep_vox), min(shape[0], zc + min_sep_vox + 1)
                y0, y1 = max(0, yc - min_sep_vox), min(shape[1], yc + min_sep_vox + 1)
                x0, x1 = max(0, xc - min_sep_vox), min(shape[2], xc + min_sep_vox + 1)
                work[b, z0:z1, y0:y1, x0:x1] = -1e9
        return cands

    pc_candidates = phase_correlation_candidates() if seed_mode == "wide" else []
    coarse_level = min(4, max_level)

    ref_min_sp = float(min(reference_img.spacing))   # the working grid (callers cap the resolution before this call)
    rot_scale_mean = float(np.mean([rotation_axis_scale(phys_X, C_phys, j) for j in range(3)]))   # mm of point displacement per radian
    # The pilot branch (pilot/motion-sampling-and-epi-debias) also rescaled the eddy shear / scale columns by the
    # rotation lever arm on large references. Evaluated and NOT adopted: it left a volume of a still 3-shell series
    # capped (antsxdwi test_motion_convergence) and raised the DWI phantom rotation error 0.392 -> 0.430 deg (5 seeds);
    # its one gain was the eddy-scale test (0.0322 -> 0.0300 for 0.03, one seed).

    def fit_pair(t_seed, omega_seed, iters=30, step_t_vox=0.075, step_r_vox=1.8):
        lr_t = step_t_vox * ref_min_sp                       # 0.15 mm / 0.06 rad at the geometry they were tuned on
        lr_r = step_r_vox * ref_min_sp / rot_scale_mean
        lvl = get_pyramid_level(coarse_level)
        Xc = lvl["X"] - C_phys
        t_cand = torch.as_tensor(np.broadcast_to(t_seed, (B, 3)).copy(), **f32).clone()
        omega_cand = torch.as_tensor(np.broadcast_to(omega_seed, (B, 3)).copy(), **f32).clone()
        t_cand.requires_grad_(True)
        omega_cand.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [t_cand], "lr": lr_t}, {"params": [omega_cand], "lr": lr_r}])

        def eval_loss():
            R_cand = _batched_rodrigues(omega_cand)
            y_phys = torch.einsum("bij,sj->bsi", R_cand, Xc) + C_phys + t_cand.unsqueeze(1)
            grid = warp_to_moving_norm(y_phys, lvl["shape"], level=coarse_level).view(B, 1, 1, -1, 3)
            warped = F.grid_sample(lvl["moving"].unsqueeze(1), grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True).reshape(B, -1)
            return pearson_correlation_loss(warped, lvl["fixed_vals"])

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            eval_loss().sum().backward()
            opt.step()
        with torch.no_grad():
            score = eval_loss().cpu().numpy()
        return t_cand.detach().cpu().numpy(), omega_cand.detach().cpu().numpy(), score

    if seed_mode == "wide":
        rot_seeds = [np.zeros(3)]
        for axis in range(3):
            for deg in (20.0, -20.0):
                v = np.zeros(3)
                v[axis] = np.radians(deg)
                rot_seeds.append(v)
        all_t_candidates = [t_init_com] + pc_candidates
    elif seed_mode == "narrow":
        # Temporally coherent series: real motion is a small perturbation, not an unknown
        # large misalignment -- skip the wide capture-range search (42 combos -> 1) and
        # trust the main coarse-to-fine schedule below to converge from a single
        # center-of-mass seed. This is the dominant cost cut of the 4 speed levers piloted
        # for real 356-frame FDG-PET motion correction (1260 Adam steps -> 30).
        rot_seeds = [np.zeros(3)]
        all_t_candidates = [t_init_com]
    else:
        raise ValueError(f"seed_mode must be 'wide' or 'narrow', got {seed_mode!r}")

    if init_pose is not None:
        # a known starting pose per frame (``[d_com (mm), rotvec (rad)]`` about the reference's centre of mass, e.g. the
        # composed chain of a previous pass): no seed search, the schedule refines from here
        t_init = np.asarray(init_pose[0], dtype=np.float32).reshape(B, 3)
        omega_init = np.asarray(init_pose[1], dtype=np.float32).reshape(B, 3)
        fit_results = []
    else:
        fit_results = []
        for t_cand in all_t_candidates:
            for seed in rot_seeds:
                fit_results.append(fit_pair(t_cand, seed))
        all_scores = np.stack([s for _, _, s in fit_results], axis=0)
        best_combo = all_scores.argmin(axis=0)
        t_init = np.stack([fit_results[best_combo[b]][0][b] for b in range(B)], axis=0).astype(np.float32)
        omega_init = np.stack([fit_results[best_combo[b]][1][b] for b in range(B)], axis=0).astype(np.float32)
    if verbose and fit_results:
        print(f"[motion_batched] translation+rotation init ({seed_mode}): searched "
              f"{len(all_t_candidates)} translation candidates x {len(rot_seeds)} rotation "
              f"seeds = {len(fit_results)} combos/frame")

    omega = torch.tensor(omega_init, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)

    if enc_on:
        enc_unit = torch.tensor([1.0, 1.0, 1.0] + [1.0 / max(rot_scale_mean, 1e-6)] * 3, **f32)

    def _encoding_offset():
        """(B, 6) encoding-locked pose offset E_psi . B_win in (mm, rad)."""
        dY = torch.einsum("bmk,mkd->bd", E_psi_t, B_win) * enc_unit
        if pe_on and pe_axis is not None:
            # the gauge keeps no translation along p in the span of the eddy covariates (odd in g); the even part stays
            p_comp = (dY[:, :3] * p_vec).sum(dim=-1)
            odd = P_odd_t @ p_comp
            dY_trans = dY[:, :3] - odd.unsqueeze(1) * p_vec
            dY = torch.cat([dY_trans, dY[:, 3:]], dim=-1)
        return dY

    def _total_pose():
        """Pose the image metric sees: head motion (t_param, omega) plus the encoding-locked offset."""
        if not enc_on:
            return t_param, omega
        dY = _encoding_offset()
        return t_param + dY[:, :3], omega + dY[:, 3:]

    def _surround_reg_loss():
        """Penalty (``surround_rho``, form ``surround_form``) on the second-order surround of the HEAD trajectory only (mm;
        rotations x 50 mm). Smooth drift is in its nullspace; a real jump costs ~linearly (not quadratically), so it is not
        pulled toward its neighbours; the ``sparse`` form drives sub-knee jitter to zero."""
        u_t = t_param[1:-1] - 0.5 * (t_param[:-2] + t_param[2:])
        u_w = 50.0 * (omega[1:-1] - 0.5 * (omega[:-2] + omega[2:]))
        u_sq = (u_t * u_t).sum(dim=-1) + (u_w * u_w).sum(dim=-1)
        rough = surround_rho(u_sq, surr_w, surr_delta, surround_form)[0].sum()
        shrink = shrink_w * ((t_param * t_param).sum() + 2500.0 * (omega * omega).sum())
        return surr_scale[0] * (rough + shrink)

    def _prior_scale_curvature(loss_closure, axes, stage_voxel_mm):
        """Median image curvature (loss per mm^2) over frames and translation axes at the current pose (``'curvature'``)."""
        if surround_scale != "curvature" or (surr_w <= 0.0 and shrink_w <= 0.0):
            return
        with torch.no_grad():
            pr = probe_manifold(loss_closure, axes, stage_voxel_mm, probe_vox=0.5)
            tr_axes = [i for i, ax in enumerate(axes) if ax.tensor is t_param]
            c = pr.curv[:, tr_axes] / (float(stage_voxel_mm) ** 2)          # loss per voxel^2 -> loss per mm^2
            c = c[torch.isfinite(c) & (c > 0)]
            if c.numel() == 0:
                return
            surr_scale[0] = float(c.median())
            prior_scale_log.append({"curvature_per_mm2": surr_scale[0], "points": int(cur_points[0]), "scale": surr_scale[0]})

    cur_points = [0]

    def _prior_scale(cur, moving_tensor, shape_stage, level_stage):
        """sigma^2 / S of this stage's points at the current pose (``surround_scale='likelihood'``)."""
        cur_points[0] = int(cur["X"].shape[0])
        if surround_scale != "likelihood" or (surr_w <= 0.0 and shrink_w <= 0.0):
            return
        with torch.no_grad():
            E = _warped_values(cur["X"], moving_tensor, shape_stage, level_stage)
            fv = cur["fixed_vals"]
            r = _scale(E, True) - (_scale(fv, True) if fv.dim() == 2 else _scale(fv, False).unsqueeze(0))
            mad = 1.4826 * (r - r.median(dim=1, keepdim=True).values).abs().median(dim=1).values   # (B,) per frame
            sigma2 = float(mad.median() ** 2)
            S = int(cur["X"].shape[0])
            surr_scale[0] = max(sigma2, 1e-12) / max(S, 1)
            prior_scale_log.append({"sigma_scaled": float(np.sqrt(sigma2)), "points": S, "scale": surr_scale[0]})

    def _surround_frame_terms():
        """The prior's per-term values (B-2,), one per second difference, attributed to its centre frame for the per-frame
        Newton line search: (B,) with zeros at the two ends."""
        with torch.no_grad():
            u_t = t_param[1:-1] - 0.5 * (t_param[:-2] + t_param[2:])
            u_w = 50.0 * (omega[1:-1] - 0.5 * (omega[:-2] + omega[2:]))
            u_sq = (u_t * u_t).sum(dim=-1) + (u_w * u_w).sum(dim=-1)
            terms = surr_scale[0] * surround_rho(u_sq, surr_w, surr_delta, surround_form)[0]
            out = torch.zeros(B, **f32)
            out[1:-1] = terms
            out = out + surr_scale[0] * shrink_w * ((t_param * t_param).sum(dim=1) + 2500.0 * (omega * omega).sum(dim=1))
        return out

    def _surround_reg_hess(axes):
        """Per-frame Hessian block (B, P, P) of the surround prior in the order of ``axes`` (``surround_prior_hessian``)."""
        cols = [(3 if ax.tensor is omega else 0) + int(ax.index[1]) for ax in axes]
        return surround_prior_hessian(t_param.detach(), omega.detach(), surr_scale[0] * surr_w, surr_delta, shrinkage=surr_scale[0] * shrink_w,
                                      form=surround_form)[:, cols][:, :, cols]

    # Level gauge of the ``projected`` engine: a constant pose offset of one level against the anchor is a nullspace of a
    # within-level metric (its model moves with its frames), and physics says levels differ in contrast, not in pose. The
    # gauge is fixed after every stage with the robust surround estimator on the level indicator columns (one window):
    # the level-locked component of the pose is removed, real head motion within and across levels is untouched.
    gauge_E = None
    if design_gauge is not None:
        # the full design gauge of the projected engine: every pose component in the span of the design columns (level
        # intercepts AND direction terms) is a nullspace of the design-in-metric objective and is removed from the pose
        # after every stage. Measured before this existed (BLAST031, 2026-10-28): the projected engine's readout
        # translation was 63 % explained by the even-in-g direction terms, the joint engine's (which gauges its columns) 4 %.
        G = np.asarray(design_gauge, dtype=np.float64).reshape(B, -1)
        G = G - G.mean(axis=0, keepdims=True)
        q, r_ = np.linalg.qr(G)
        keep = np.abs(np.diag(r_)) > 1e-8 * max(np.abs(np.diag(r_)).max(), 1e-30)
        gauge_E = q[:, keep] if (bool(keep.any()) and B >= 6) else None
    elif design_levels is not None:
        lev = np.asarray(design_levels, dtype=int).reshape(-1)
        if lev.size != B:
            raise ValueError(f"design_levels has {lev.size} entries for {B} frames")
        cols = [(lev == k).astype(np.float64) for k in np.unique(lev) if k != lev.min() and np.any(lev == k) and np.any(lev != k)]
        if cols and B >= 6:
            C_lev = np.stack(cols, axis=1)
            C_lev = C_lev - C_lev.mean(axis=0, keepdims=True)
            q, r_ = np.linalg.qr(C_lev)
            keep = np.abs(np.diag(r_)) > 1e-8
            gauge_E = q[:, keep] if bool(keep.any()) else None
    level_gauge_log: list = []

    def _project_level_gauge():
        if gauge_E is None:
            return None
        with torch.no_grad():
            Y = np.c_[t_param.detach().cpu().numpy().astype(np.float64), omega.detach().cpu().numpy().astype(np.float64)]
            _, dY, w_np = _project_encoding_surround_tukey_np(Y, gauge_E, np.ones((B, 1)), fd_radius=50.0, surround_weight=0.0)
            dY_t = torch.tensor(dY, **f32)
            t_param.sub_(dY_t[:, :3])
            omega.sub_(dY_t[:, 3:])
            level_gauge_log.append({"removed_rms_mm": float(np.sqrt((dY[:, :3] ** 2).sum(1).mean())),
                                    "removed_rms_deg": float(np.degrees(np.sqrt((dY[:, 3:] ** 2).sum(1).mean())))})
        return w_np

    def _surround_reg_hess_full(axes):
        """Full coupled Hessian (B, P, B, P) of the surround prior in the order of ``axes``."""
        cols = [(3 if ax.tensor is omega else 0) + int(ax.index[1]) for ax in axes]
        H = surround_prior_hessian_full(t_param.detach(), omega.detach(), surr_scale[0] * surr_w, surr_delta, shrinkage=surr_scale[0] * shrink_w,
                                        form=surround_form)
        return H[:, cols][:, :, :, cols]

    def _project_encoding_gauge():
        """Re-split the total pose into head motion and encoding offset with the closed-form Tukey surround projection.

        The offset is folded back into (t_param, omega) and B_win is re-estimated from that total, so B_win is always the
        ROBUST (Tukey-downweighted) encoding-locked component: whatever the gradient steps pushed into B_win -- e.g. part
        of a real head jump that lowered the surround penalty -- returns to the head trajectory. The total pose (and so
        the image loss) is unchanged."""
        if not enc_on:
            return None
        with torch.no_grad():
            dY_old = _encoding_offset()
            t_param.add_(dY_old[:, :3])
            omega.add_(dY_old[:, 3:])
            B_win.zero_()
            Y_tot = np.c_[t_param.detach().cpu().numpy().astype(np.float64), omega.detach().cpu().numpy().astype(np.float64)]
            Y_fit = Y_tot
            if pe_on and pe_axis is not None:
                p_arr = np.asarray(pe_axis, dtype=np.float64)
                p_norm = np.linalg.norm(p_arr)
                if p_norm > 1e-8:
                    p_unit = p_arr / p_norm
                    Y_fit = Y_tot.copy()   # the odd-in-g translation along p is the eddy term's, not the gauge's
                    Y_fit[:, :3] -= np.outer(P_odd_np @ (Y_fit[:, :3] @ p_unit), p_unit)
            B_np, _, w_np = _project_encoding_surround_tukey_np(Y_fit, E_np, psi_np, fd_radius=50.0, surround_weight=0.0)
            B_win.copy_(torch.tensor(B_np, **f32) / enc_unit)
            dY_new = _encoding_offset()            # applies the odd-only projection to the new gauge
            t_param.sub_(dY_new[:, :3])
            omega.sub_(dY_new[:, 3:])
        return w_np

    # Coarse-to-fine schedule: Adam at real avg_pool3d-downsampled levels (broad, cheap
    # capture) -> LBFGS at level 1 with point-sampling (precise polish). Dual-metric
    # (Mattes MI + correlation) at the coarse stages only -- see module docstring / session
    # doc for why correlation specifically helps at coarse resolution for this
    # intra-subject case.
    mid_level = min(2, max_level)
    if schedule is not None:
        schedule = [dict(st) for st in schedule]   # explicit stage list (see ``schedule`` in the docstring)
    elif schedule_mode == "fast":
        # Shorter schedule for temporally coherent series / precision-tolerant consumers
        # (e.g. averaging frames into a population template): one coarse Adam stage, one
        # joint LBFGS polish at level 1, no alternating translation-only/rotation-only
        # final-polish pass (the most expensive stages per iter, per the note below).
        schedule = [
            dict(optimizer="adam", iters=30, step_t_vox=0.04, step_r_vox=1.2, level=coarse_level, sampling=0.2, corr_weight=1.0),
            dict(optimizer="lbfgs", iters=10, sampling=0.2, level=1),
        ]
    elif schedule_mode in ("default", "medium"):
        # Adam on the coarse pyramid levels (broad capture: a rigid pose is recovered from offsets of ~10 mm / 20 degrees at 2 mm
        # voxels; without the L4 stage 12 mm offsets end 0.4 mm off and 20 mm offsets fail), then a Newton finisher at the finest
        # level: no learning rate, parameters scaled by the physical displacement of the sampled points, and a frame ends when
        # its Newton step is below ``convergence_tol_vox`` (the remaining distance to the optimum), so the result is the same in
        # voxels at any spacing (mouse x0.1, human, x10: 0.178 voxel at all three; L-BFGS 0.20-0.23). With the eddy
        # PE-affine (extra non-rigid shared parameters) L-BFGS fits the shared terms at L1 and Newton polishes the per-frame rigid poses.
        schedule = [
            dict(optimizer="adam", iters=80, step_t_vox=0.04, step_r_vox=1.2, level=coarse_level, sampling=0.2, corr_weight=1.0),
            dict(optimizer="adam", iters=40, step_t_vox=0.015, step_r_vox=0.45, level=mid_level, sampling=0.2, corr_weight=1.0),
            *(
                [
                    dict(optimizer="lbfgs", iters=15, sampling=0.2, level=1),
                    dict(optimizer="newton", iters=15, sampling=0.2, level=1, freeze_pe=True, confirm_vox=min(0.02, float(convergence_tol_vox))),
                ]
                if pe_on
                else [
                    dict(optimizer="newton", iters=20, sampling=0.2, level=1, confirm_vox=min(0.02, float(convergence_tol_vox))),
                ]
            ),
        ]
    elif schedule_mode == "refine":
        # From a known pose (``init_pose``): no capture stages. Adam's first steps are learning-rate sized whatever the
        # gradient, so coarse Adam stages walk a good pose away before converging back to the coarse level's own
        # (biased) optimum. L-BFGS at full resolution first (the shared design terms and the head prior enter here), then
        # the Newton finisher on the per-frame rigid poses.
        if init_pose is None:
            raise ValueError("schedule_mode='refine' needs init_pose")
        schedule = [
            dict(optimizer="lbfgs", iters=15, sampling=0.2, level=1),
            dict(optimizer="newton", iters=15, sampling=0.2, level=1, freeze_pe=True, confirm_vox=min(0.02, float(convergence_tol_vox))),
        ]
    else:
        raise ValueError(f"schedule_mode must be 'default', 'medium', 'fast' or 'refine', got {schedule_mode!r}")

    # Per-frame global gain (``frame_gain``): scanner intensity drift is a spatially uniform multiplicative change of a
    # whole frame, the one intensity term that is anchored on the spatial side (uniform pattern, far from every rigid
    # derivative) and so cannot be confused with slow head motion. One scalar per frame, the ratio of the reference's
    # robust level to the frame's at the current points and pose, refreshed at every draw (with the design model), applied
    # inside the shared warp so the metric, the design model and the prior scale all see the gain-corrected frame.
    gain_t = torch.ones(B, **f32)
    gain_log: list = []

    def _frame_gain_refresh(cur, moving_tensor, shape_stage, level_stage):
        if not frame_gain:
            return
        with torch.no_grad():
            gain_t.fill_(1.0)
            E = _warped_values(cur["X"], moving_tensor, shape_stage, level_stage)
            fv = cur["fixed_vals"]
            ref = fv if fv.dim() == 2 else fv.unsqueeze(0).expand_as(E)
            num = ref.abs().median(dim=1).values
            den = E.abs().median(dim=1).values.clamp_min(1e-8)
            g = (num / den).clamp(0.5, 2.0)
            gain_t.copy_(g)
            gain_log.append(g.detach().cpu().numpy().astype(float).tolist())

    def _warped_values(X_stage, moving_tensor, shape_stage, level_stage=1):
        """Intensities (B, S) of every frame at the physical points ``X_stage`` under the current total pose (and eddy
        term), from the moving tensor of pyramid ``level_stage``: the one warp the metric and the design model share.
        With ``frame_gain`` the frame's global gain is applied here."""
        t_eff, om_eff = _total_pose()
        R = _batched_rodrigues(om_eff)
        Xc = X_stage - C_phys
        y_phys = torch.einsum("bij,sj->bsi", R, Xc) + C_phys + t_eff.unsqueeze(1)
        if pe_on:
            a_pe, jac_pe = pe_terms()
            y_phys = apply_pe(y_phys, a_pe)
        grid_norm = warp_to_moving_norm(y_phys, shape_stage, level=level_stage)
        grid = grid_norm.view(B, 1, 1, -1, 3)
        warped = grid_sample_nd(moving_tensor.unsqueeze(1), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True,
                                 interpolator="linear", use_analytical_gradients=True,
                                 precomputed_grad_I=get_grad_I(moving_tensor)).reshape(B, -1)
        if pe_on:
            warped = warped * jac_pe.unsqueeze(1)
        if frame_gain:
            warped = warped * gain_t.unsqueeze(1)
        return warped

    # Design-model references (``design_matrix``, the ``projected`` engine): at every point the reference of frame t is the
    # prediction of the known acquisition design D (B, K) fitted across frames to the CURRENT warped values,
    # P_D e(x) with P_D = D (D'D)^-1 D'. Design-locked contrast (a level's mean, the l = 2 direction terms of a shell) is
    # in the reference, so it cannot be lowered by moving the head; what the design cannot explain is what the pose
    # explains. Refreshed at every stage start and point redraw with the model held fixed between refreshes (block
    # coordinate descent: the per-frame losses are then independent and the Newton machinery applies unchanged). The
    # per-point robust residual scale (MAD over frames) weights the points: a point whose residual stays large after
    # alignment (pulsatile CSF) carries little weight. Leverage diag(P_D) = K/B on average: a frame contributes that share
    # of its own reference; recorded, not corrected.
    design_model = None
    if design_matrix is not None:
        D_np = np.asarray(design_matrix, dtype=np.float64).reshape(B, -1)
        if D_np.shape[0] != B:
            raise ValueError(f"design_matrix has {D_np.shape[0]} rows for {B} frames")
        rank = int(np.linalg.matrix_rank(D_np))
        if rank < D_np.shape[1]:
            raise ValueError(f"design_matrix is rank deficient ({rank} < {D_np.shape[1]} columns)")
        P_np = D_np @ np.linalg.solve(D_np.T @ D_np, D_np.T)
        design_model = {"P": torch.tensor(P_np, **f32), "rank": rank, "hat_max": float(np.max(np.diag(P_np))),
                        "hat_mean": float(np.mean(np.diag(P_np))), "refreshes": 0}

    def _design_model_refs(cur, moving_tensor, shape_stage, level_stage):
        """Fill ``cur['fixed_vals']`` (B, S) with the design prediction at ``cur['X']``, ``cur['w_pts']`` with the per-point
        scale weights and the per-frame intensity range with the prediction's own range."""
        nonlocal hi_b
        with torch.no_grad():
            E = _warped_values(cur["X"], moving_tensor, shape_stage, level_stage)
            pred = design_model["P"] @ E
            resid = E - pred
            mad = 1.4826 * (resid - resid.median(dim=0).values).abs().median(dim=0).values
            glob = mad.median().clamp_min(1e-8)
            sigma = mad.clamp_min(0.5 * glob)
            cur["fixed_vals"] = pred
            cur["w_pts"] = ((glob / sigma) ** 2).unsqueeze(0).expand(B, -1).contiguous() if design_point_scale else None
            hi_b = pred.amax(dim=1).clamp_min(1e-8)
            cur["w_y"] = _parzen(_scale(pred, True)) if metric != "huber" else cur.get("w_y")
            design_model["refreshes"] += 1

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0, fixed_tensor=None, w_pts=None):
        warped = _warped_values(X_stage, moving_tensor, shape_stage, level_stage)
        if design_model is not None and design_model.get("live"):
            # the profile likelihood: the design model is re-fitted to the CURRENT warped values inside the loss, so a pose
            # change that the design explains (a constant offset of one level, a term in g) costs nothing and the head
            # prior resolves it; the gradient flows through the fit. The Newton stage holds the model fixed instead
            # (refreshed before every iteration): its per-frame Hessian must not see the frames' coupling through the fit.
            fixed_vals_stage = design_model["P"] @ warped
        w_scaled = _scale(warped, True)
        if metric == "huber":
            fixed_scaled_stage = _scale(fixed_vals_stage, fixed_vals_stage.dim() == 2)
            target = fixed_scaled_stage if fixed_scaled_stage.dim() == 2 else fixed_scaled_stage.unsqueeze(0).expand_as(w_scaled)
            elem = huber_loss_nd(w_scaled, target, delta=huber_delta, reduction="none")
            if w_pts is None:
                per_frame_loss = elem.mean(dim=1)
            else:
                per_frame_loss = (elem * w_pts).sum(dim=1) / w_pts.sum(dim=1).clamp_min(1e-8)
        else:
            w_x = parzen_weights(w_scaled.reshape(-1), num_bins=num_bins).reshape(*w_scaled.shape, num_bins)
            if w_pts is not None:
                # weighted joint histogram: each point enters with its frame's weight (sqrt on each factor)
                sw = torch.sqrt(w_pts).unsqueeze(-1)
                w_x = w_x * sw
                w_y_stage = (w_y_stage if w_y_stage.dim() == 3 else w_y_stage.unsqueeze(0).expand(B, -1, -1)) * sw
            per_frame_loss = batched_mattes_mi_from_weights(w_x, w_y_stage)
        if corr_weight > 0.0 and fixed_vals_stage is not None:
            per_frame_loss = per_frame_loss + corr_weight * pearson_correlation_loss(warped, fixed_vals_stage)
        return per_frame_loss

    convergence_log = []
    for stage_idx, stage in enumerate(schedule):
        level = stage.get("level", 1)
        samp = stage.get("sampling", 1.0) if float(sample_fraction) < 1.0 else 1.0  # dense: every sampled voxel, every stage
        freeze = stage.get("freeze", None)  # 'r' freezes omega (translation-only update);
                                             # 't' freezes t_param (rotation-only update).
        freeze_pe = bool(stage.get("freeze_pe", False))
        # The stage's pool: level 1 = the full-resolution pool drawn by the sampling strategy; coarser levels = every
        # candidate point of the level. Each metric evaluation sees ``samp`` of the pool, drawn with the strategy's weights;
        # with ``resample_points`` the subset is re-drawn at every convergence checkpoint
        # (``_run_converging_stage(resample_fn=...)``), otherwise it is fixed for the stage.
        lvl_data = get_pyramid_level(level)
        moving_tensor, shape_stage = lvl_data["moving"], lvl_data["shape"]
        n_pool = int(lvl_data["X"].shape[0])
        k_draw = sampler.eval_points(samp, n_pool, minimum=500 if level == 1 else 300)   # capped at MAX_EVAL_POINTS
        cur: dict = {}

        def _draw(lvl_data=lvl_data, cur=cur, k_draw=k_draw, n_pool=n_pool):
            sub = None if k_draw >= n_pool else draw_indices(lvl_data["w"], n_pool, k_draw, rng)
            fv = lvl_data["fixed_vals"] if sub is None else _sub(lvl_data["fixed_vals"], sub)
            cur["X"] = lvl_data["X"] if sub is None else lvl_data["X"][sub]
            cur["fixed_vals"] = fv
            wf = lvl_data.get("w_frame")
            cur["w_pts"] = None if wf is None else (wf if sub is None else wf[:, sub])
            if lvl_data["w_y"] is None:
                cur["w_y"] = _parzen(_scale(fv, fv.dim() == 2))
            else:
                cur["w_y"] = lvl_data["w_y"] if sub is None else _sub(lvl_data["w_y"], sub)
            _frame_gain_refresh(cur, moving_tensor, shape_stage, level)
            if design_model is not None:
                _design_model_refs(cur, moving_tensor, shape_stage, level)
                _frame_gain_refresh(cur, moving_tensor, shape_stage, level)   # the model moved: one more pass on the gain
            _prior_scale(cur, moving_tensor, shape_stage, level)

        if design_model is not None:
            design_model["live"] = bool(design_model_live) and stage.get("optimizer", "adam") != "newton"
        _draw()
        resample_fn = _draw if ((resample_points and k_draw < n_pool) or (design_model is not None and not design_model["live"])) else None
        X_pool = lvl_data["X"]
        corr_weight = stage.get("corr_weight", 0.0)

        if freeze == "r":
            opt_params = [t_param]
        elif freeze == "t":
            opt_params = [omega]
        else:
            opt_params = [t_param, omega]
        shared_stage_params = []
        if pe_on and not freeze_pe:
            opt_params = opt_params + [Ccoef]
            shared_stage_params.append(Ccoef)
        enc_stage = enc_on and freeze is None
        # the head prior is part of the objective in every full L-BFGS / Newton stage; never in Adam, whose per-parameter
        # normalisation turns any non-zero gradient into a full learning-rate step and so cannot weigh a prior against the data
        prior_stage = freeze is None and (surr_w > 0.0 or shrink_w > 0.0) and B >= 3 and stage.get("optimizer", "adam") != "adam"
        if enc_stage:
            opt_params = opt_params + [B_win]
            shared_stage_params.append(B_win)

        def stage_points(X_pool=X_pool):
            """Physical positions of a fixed subset of the stage's pool under the current parameters."""
            with torch.no_grad():
                Xs = X_pool[::max(1, X_pool.shape[0] // _N_CONVERGENCE_POINTS)]
                t_eff, om_eff = _total_pose()
                y = torch.einsum("bij,sj->bsi", _batched_rodrigues(om_eff), Xs - C_phys) + C_phys + t_eff.unsqueeze(1)
                if pe_on:
                    y = apply_pe(y, pe_terms()[0])
                return y

        if freeze == "r":
            per_frame_params = [t_param]
        elif freeze == "t":
            per_frame_params = [omega]
        else:
            per_frame_params = [t_param, omega]
        lr_groups = None
        if stage.get("optimizer", "adam") == "adam":
            lr_groups = []
            if freeze != "t":
                lr_groups.append({"params": [t_param], "lr": stage["step_t_vox"] * ref_min_sp})
            if freeze != "r":
                lr_groups.append({"params": [omega], "lr": stage["step_r_vox"] * ref_min_sp / rot_scale_mean})
            if pe_on and not freeze_pe:
                lr_groups.append({"params": [Ccoef], "lr": stage["step_t_vox"] * ref_min_sp})
            if enc_stage:
                lr_groups.append({"params": [B_win], "lr": stage["step_t_vox"] * ref_min_sp})
        stage_voxel = ref_min_sp * max(int(level), 1)   # the voxel of this stage's grid (pooled levels are coarser)
        _pts = X_pool[::max(1, X_pool.shape[0] // _N_CONVERGENCE_POINTS)]
        probe_axes = [ax for ax in rigid_axes(t_param, omega, _pts, C_phys)
                      if not ((freeze == "r" and ax.tensor is omega) or (freeze == "t" and ax.tensor is t_param))]
        _prior_scale_curvature(lambda: compute_loss(cur["X"], cur["w_y"], moving_tensor, shape_stage, level_stage=level,
                                                    fixed_vals_stage=cur["fixed_vals"], corr_weight=corr_weight,
                                                    fixed_tensor=lvl_data["fixed_tensor"], w_pts=cur["w_pts"]),
                               probe_axes, stage_voxel)
        rec = _run_converging_stage(
            optimizer=stage.get("optimizer", "adam"), params=opt_params, per_frame_params=per_frame_params,
            loss_fn=lambda: compute_loss(cur["X"], cur["w_y"], moving_tensor, shape_stage, level_stage=level,
                                         fixed_vals_stage=cur["fixed_vals"], corr_weight=corr_weight,
                                         fixed_tensor=lvl_data["fixed_tensor"], w_pts=cur["w_pts"]),
            resample_fn=resample_fn,
            points_fn=stage_points, iters=int(stage["iters"]), B=B, dev=dev,
            tol=float(convergence_tol_vox) * stage_voxel, max_extend=int(max_extend_factor),
            lr_groups=lr_groups, lr_t=stage.get("step_t_vox", 0.0) * ref_min_sp,
            lr_r=stage.get("step_r_vox", 0.0) * ref_min_sp / rot_scale_mean, point_radius=rot_scale_mean,
            probe_fn=lambda: probe_manifold(lambda: compute_loss(cur["X"], cur["w_y"], moving_tensor, shape_stage, level_stage=level,
                                                                 fixed_vals_stage=cur["fixed_vals"], corr_weight=corr_weight,
                                                                 fixed_tensor=lvl_data["fixed_tensor"], w_pts=cur["w_pts"]),
                                            probe_axes, stage_voxel, probe_vox=0.5),
            confirm_vox=float(stage.get("confirm_vox", _CONFIRM_FACTOR * float(convergence_tol_vox))), axes=probe_axes, stage_voxel=stage_voxel,
            label=f"{stage.get('optimizer', 'adam')}@L{level}" + (f"/{freeze}" if freeze else ""), level=level,
            shared_params=shared_stage_params if shared_stage_params else None,
            reg_loss_fn=_surround_reg_loss if prior_stage else None,
            reg_hess_fn=(lambda: _surround_reg_hess(probe_axes)) if prior_stage else None,
            reg_hess_full_fn=(lambda: _surround_reg_hess_full(probe_axes)) if prior_stage else None,
            reg_frame_fn=_surround_frame_terms if prior_stage else None)
        _project_encoding_gauge()
        _project_level_gauge()
        convergence_log.append(rec)

    elapsed = time.time() - t0

    if outprefix is None:
        work_dir = tempfile.mkdtemp(prefix="syntx_batched_moco_")
        prefix = os.path.join(work_dir, "vol")
    else:
        prefix = outprefix

    delta_Y_np = None
    enc_bias_info = None
    if enc_on:
        # (t_param, omega) is the head motion and E_psi . B_win the encoding-locked offset; their sum is the pose the image
        # metric was optimised on. The written transforms carry head motion only: the offset is registration bias
        # (contrast-driven pseudomotion), not displacement of the anatomy.
        w_tukey = _project_encoding_gauge()
        with torch.no_grad():
            delta_Y_np = _encoding_offset().detach().cpu().numpy().astype(np.float64)
            B_win_np = (B_win * enc_unit).detach().cpu().numpy().astype(np.float64)
        enc_bias_info = {
            "B_win": B_win_np.tolist(),
            "delta_Y": delta_Y_np.tolist(),
            "delta_Y_mm_rad": delta_Y_np,
            "surround_tukey_weights": w_tukey,
            "window_size": int(enc_basis["window_size"]),
            "window_stride": int(enc_basis["window_stride"]),
            "num_windows": int(M_win),
            "K_eff": int(K_eff),
            "surround_weight": float(surr_w),
            "removed_rms_mm": float(np.sqrt((delta_Y_np[:, :3] ** 2).sum(axis=1).mean())),
            "removed_rms_deg": float(np.degrees(np.sqrt((delta_Y_np[:, 3:] ** 2).sum(axis=1).mean()))),
        }
    R_final = _batched_rodrigues(omega).detach().cpu().numpy()
    t_final = t_param.detach().cpu().numpy().astype(np.float64)
    pe_info = None
    if pe_on:
        with torch.no_grad():
            a_np, jac_np = (x.detach().cpu().numpy().astype(np.float64) for x in pe_terms())
        p_np = p_vec.cpu().numpy().astype(np.float64)
        C_np = (Ccoef * pe_scale).detach().cpu().numpy().astype(np.float64)
        # Attribution of the translation along p. A per-frame rigid translation and a tied eddy translation a0 = C0 . g
        # fit the data equally well (t_k can carry p * C0 . g_k frame by frame), so the joint optimum is a family; shears
        # and scale have no rigid counterpart and are identified by the fit alone. The family is resolved by the same
        # statement that defines the model -- head motion is independent of the gradient -- as the constraint that the
        # rigid translations along p are uncorrelated with the covariates: their least-squares projection on g moves
        # into C0. Only the head-motion part is attributed: an encoding-locked offset (registration bias, not removed from
        # the data) must not reappear as an applied eddy shift. The projection is estimated on second-order SURROUND
        # differences with Tukey weights (``_project_encoding_surround_tukey_np``, covariates as given, one window): a
        # constant offset and smooth drift are in its nullspace and isolated jumps are downweighted, so real head motion
        # that happens to correlate with g over a short series is not reassigned to the eddy term (plain least squares on
        # the poses did that: simulated DWI, 20 directions, a 1.2 mm jump -> 0.12 mm head error along PE, corr(err, g_pe)
        # -0.74). Fewer than 8 frames: plain least squares (no surround statistics).
        tp = t_final @ p_np
        if B >= 8:
            Yp = np.zeros((B, 6))
            Yp[:, 0] = tp
            B0_full, dY_pe, _ = _project_encoding_surround_tukey_np(Yp, cov_np, np.ones((B, 1)))
            B0 = B0_full[0, :, 0]
        else:
            B0, *_ = np.linalg.lstsq(cov_np, tp, rcond=None)
        t_final = t_final - np.outer(cov_np @ B0, p_np)
        C_np[0] += B0
        a_np[:, 0] += cov_np @ B0
        pe_info = {"axis": p_np.tolist(), "coefficients": C_np, "a": a_np, "jacobian": jac_np}
    bias_info = None
    if bias_covariates is not None:
        # 6 rigid parameters per frame: displacement of the reference centre of mass (mm) and rotation vector (rad)
        from scipy.spatial.transform import Rotation as _Rot

        Z = np.asarray(bias_covariates, dtype=np.float64).reshape(B, -1)
        if np.any(Z != 0) and B > Z.shape[1] + 3:
            rv = _Rot.from_matrix(R_final.astype(np.float64)).as_rotvec()
            Y = np.c_[t_final, rv]
            # Surround-difference Tukey estimate (one window), not least squares on the poses: the mean pose and smooth
            # drift are in its nullspace and jumps are downweighted, so head motion that correlates with the covariates by
            # chance stays (OLS removed it: simulated DWI, l = 2 terms, 0.04 -> 0.22 mm head error).
            Zc = Z - Z.mean(axis=0, keepdims=True)
            beta_z, fit, _ = _project_encoding_surround_tukey_np(Y, Zc, np.ones((B, 1)))
            beta = np.vstack([np.zeros((1, 6)), beta_z[0]])  # row 0: the intercept (mean pose) is kept
            Y_c = Y - fit
            R_final = _Rot.from_rotvec(Y_c[:, 3:]).as_matrix().astype(R_final.dtype)
            t_final = Y_c[:, :3]
            bias_info = {"coefficients": beta[1:].tolist(), "removed_rms_mm": float(np.sqrt((fit[:, :3] ** 2).sum(1).mean())),
                         "removed_rms_deg": float(np.degrees(np.sqrt((fit[:, 3:] ** 2).sum(1).mean()))), "n_covariates": int(Z.shape[1])}
    if bias_info is not None:
        if pe_info is None:
            pe_info = {}
        pe_info["bias"] = bias_info
    if enc_bias_info is not None:
        if pe_info is None:
            pe_info = {}
        pe_info["encoding_pose_bias_mm_rad"] = enc_bias_info
        if bias_info is None:
            pe_info["bias"] = enc_bias_info
    if pe_info is None:
        pe_info = {}
    pe_info["convergence"] = convergence_log
    pe_info["frame_gain"] = gain_t.detach().cpu().numpy().astype(float).tolist() if frame_gain else None
    pe_info["prior"] = {"surround_weight": float(surr_w), "surround_delta_mm": float(surr_delta), "surround_scale": surround_scale,
                        "surround_form": surround_form,
                        "pose_shrinkage": float(shrink_w),
                        "scale_log": prior_scale_log}
    if design_model is not None:
        pe_info["design_model"] = {k: v for k, v in design_model.items() if k not in ("P", "live")}
        pe_info["design_model"]["level_gauge"] = level_gauge_log
    pe_info["sampling"] = {**sampler.provenance(), "resample_points": bool(resample_points), "sample_seed": int(sample_seed),
                           "n_sample_full_resolution": int(n_sample)}
    fwd_transforms, inv_transforms = [], []
    for b in range(B):
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([R_final[b].flatten(), t_final[b]]))
        tx.set_fixed_parameters(com_f)
        tx_path = f"{prefix}{b:04d}.mat"
        ants.write_transform(tx, tx_path)
        chain = [tx_path]
        if pe_on and np.any(a_np[b] != 0):
            # y' = M (y - c) + c + T with M = I + p a^T (a = shears/scale), T = p a0; applied AFTER the rigid, so it is
            # listed after it (a transform list acts on the point first-listed first; see ``rigid_transform_of``)
            M = np.eye(3) + np.outer(p_np, a_np[b, 1:4])
            te = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
            te.set_parameters(np.concatenate([M.flatten(), p_np * a_np[b, 0]]))
            te.set_fixed_parameters(com_f)
            te_path = f"{prefix}{b:04d}{EDDY_SUFFIX}"
            ants.write_transform(te, te_path)
            chain = [tx_path, te_path]
        fwd_transforms.append(chain)
        inv_transforms.append(list(chain))

    return fwd_transforms, inv_transforms, elapsed, pe_info


EDDY_SUFFIX = "_eddy.mat"
"""Suffix of the per-frame eddy PE-affine file written by ``_batched_rigid_register_pass_core`` (its forward-map inverse
in ``motion_reference`` is ``*_eddy_fwd.mat``); ``is_eddy_transform`` recognises both."""


def is_eddy_transform(path) -> bool:
    """True for an eddy PE-affine element of a frame's chain (``*_eddy.mat`` or its inverse ``*_eddy_fwd.mat``)."""
    import os
    return "_eddy" in os.path.basename(str(path))


def rigid_transform_of(chain) -> str:
    """The frame's own rigid file of a per-frame transform chain: the one element that is not an eddy PE-affine.

    Chains are ``[rigid]`` or ``[rigid, eddy]`` (first-listed acts first on the reference-space point); grouped results
    may prepend cross-group and debias corrections, so callers pass the frame's OWN part of the chain. Selection is by
    name, never by position: the position of the rigid changed on 2026-10-28 when the chain order was corrected."""
    chain = [chain] if isinstance(chain, str) else list(chain)
    rig = [f for f in chain if not is_eddy_transform(f)]
    if len(rig) != 1:
        raise ValueError(f"expected exactly one rigid element in the frame's own chain, got {chain}")
    return str(rig[0])


def _cap_resolution(img: ants.ANTsImage, target_spacing_mm: float) -> ants.ANTsImage:
    """Thin wrapper over the canonical ``syntx.imaging_utils.cap_resolution_for_registration``
    (added 2026-10-02 after this function was found independently duplicating that logic --
    with a real bug: this copy passed ``interp_type=1`` (nearest-neighbor) instead of the
    correct ``0`` (linear) for a continuous-intensity image feeding an intensity-based
    registration metric; now fixed by delegating to the one correct implementation)."""
    from .imaging_utils import cap_resolution_for_registration
    return cap_resolution_for_registration(img, target_spacing_mm)


def batched_rigid_register_pass_temporal(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    resolution_cap_mm: Optional[float] = 4.0,
    motion_gate_threshold: Optional[float] = 0.995,
    max_batch_frames: Optional[int] = None,
) -> Tuple[List[List[str]], List[List[str]], float]:
    """Pilot: fast rigid motion-correction pass for a TEMPORALLY COHERENT series
    (consecutive frames of the same subject/contrast, e.g. a dynamic PET series) --
    NOT a drop-in replacement for ``batched_rigid_register_pass``, which is tuned for
    cross-subject/cross-modality template registration where capture range and precision
    both need to be much larger than real inter-frame motion ever is.

    Combines 4 independent speed levers, each validated/motivated against real 356-frame
    FDG-PET data (see docs/tracking.md in antsxfunctional for the numbers this came out
    of -- the default ``batched_rigid_register_pass`` path took ~80min/subject for this
    series once chunked to avoid the MPS tensor-overflow bug; this path targets ~5min):

    1. ``resolution_cap_mm``: register on a spatially downsampled copy of ``reference_img``
       and every frame in ``moving_imgs`` (see ``_cap_resolution``) -- the recovered rigid
       transform is a continuous physical-space transform, valid at native resolution with
       no extra resampling step needed. Pass ``None`` to disable (register at native
       resolution).
    2 & 3. ``seed_mode="narrow"`` + ``schedule_mode="fast"`` are used unconditionally for
       the frames that do get registered (see ``_batched_rigid_register_pass_core``'s
       docstring for what each skips) -- appropriate here because frame-to-frame motion in
       a real dynamic series is a small perturbation, not an unknown large misalignment.
    4. ``motion_gate_threshold``: a cheap pre-pass computes each frame's voxelwise Pearson
       correlation to the reference (on the downsampled copies, so this costs almost
       nothing) and skips optimisation entirely (identity transform) for any frame already
       at or above this correlation -- in a real series, many frames need no correction at
       all. Pass ``None`` to disable (register every frame).

    Returns the same ``(fwd_transforms, inv_transforms, elapsed)`` shape as
    ``batched_rigid_register_pass``, in the original frame order, so this is a drop-in
    swap at call sites that accept the trade (lower precision / narrower capture range for
    much lower wall-clock on temporally coherent data).
    """
    import time as _time
    t0 = _time.time()

    B_total = len(moving_imgs)
    if B_total == 0:
        return [], [], 0.0

    ref_cap = _cap_resolution(reference_img, resolution_cap_mm) if resolution_cap_mm else reference_img
    moving_cap = [_cap_resolution(m, resolution_cap_mm) if resolution_cap_mm else m for m in moving_imgs]

    if motion_gate_threshold is not None:
        ref_np = ref_cap.numpy().reshape(-1).astype(np.float64)
        ref_centered = ref_np - ref_np.mean()
        ref_norm = np.linalg.norm(ref_centered) + 1e-8
        needs_correction = []
        for i, m in enumerate(moving_cap):
            m_np = m.numpy().reshape(-1).astype(np.float64)
            m_centered = m_np - m_np.mean()
            corr = float(np.dot(ref_centered, m_centered) / (ref_norm * (np.linalg.norm(m_centered) + 1e-8)))
            if corr < motion_gate_threshold:
                needs_correction.append(i)
        if verbose:
            print(f"[syntx.motion_batched] motion gate: {len(needs_correction)}/{B_total} frames "
                  f"below correlation threshold {motion_gate_threshold} -- registering only those, "
                  f"identity for the rest.")
    else:
        needs_correction = list(range(B_total))

    fwd_all: List[Optional[List[str]]] = [None] * B_total
    inv_all: List[Optional[List[str]]] = [None] * B_total

    if needs_correction:
        gated_imgs = [moving_cap[i] for i in needs_correction]
        gated_prefix = f"{outprefix}gated_" if outprefix is not None else None
        fwd_g, inv_g, _ = batched_rigid_register_pass(
            ref_cap, gated_imgs, device=device, num_bins=num_bins, verbose=verbose,
            outprefix=gated_prefix, max_batch_frames=max_batch_frames,
            seed_mode="narrow", schedule_mode="fast",
        )
        for local_i, global_i in enumerate(needs_correction):
            fwd_all[global_i] = fwd_g[local_i]
            inv_all[global_i] = inv_g[local_i]

    # Identity transform for every gated-out (already-aligned) frame.
    com_f = np.asarray(compute_center_of_mass(reference_img, weighted=True), dtype=np.float64)
    for i in range(B_total):
        if fwd_all[i] is not None:
            continue
        id_prefix = f"{outprefix}id{i:04d}_" if outprefix is not None else None
        tx_path = _create_identity_transform_file(com_f, id_prefix)
        fwd_all[i] = [tx_path]
        inv_all[i] = [tx_path]

    elapsed = _time.time() - t0
    return fwd_all, inv_all, elapsed


def estimate_coarse_fd(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    resolution_cap_mm: Optional[float] = 8.0,
    fd_radius: float = 50.0,
    device: str = "auto",
    num_bins: int = 18,
    outprefix: Optional[str] = None,
    max_batch_frames: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[str]], List[List[str]]]:
    """Cheap, registration-derived (not correlation- or center-of-mass-based) per-frame
    FD estimate -- the shared gate/ranking primitive behind
    ``batched_rigid_register_pass_adaptive`` and ``build_low_motion_reference``.

    One coarse (narrow-seed, fast-schedule, resolution-capped) batched registration
    pass, with FD computed directly from each frame's own recovered rigid parameters.
    See ``batched_rigid_register_pass_adaptive``'s docstring for why this replaced the
    correlation/center-of-mass alternatives (both resolution- or rotation-blind on real
    data).

    Returns
    -------
    (coarse_fd, fwd_transforms, inv_transforms) -- the transforms are returned too so a
    caller that ends up wanting them (e.g. a low-motion frame that never needs
    refinement) doesn't have to register that frame a second time.
    """
    from .motion import _extract_rigid_parameters

    B_total = len(moving_imgs)
    if B_total == 0:
        return np.zeros(0, dtype=np.float64), [], []

    ref_coarse = _cap_resolution(reference_img, resolution_cap_mm) if resolution_cap_mm else reference_img
    moving_coarse = [_cap_resolution(m, resolution_cap_mm) if resolution_cap_mm else m for m in moving_imgs]

    fwd_coarse, inv_coarse, _ = batched_rigid_register_pass(
        ref_coarse, moving_coarse, device=device, num_bins=num_bins, verbose=False,
        outprefix=outprefix, max_batch_frames=max_batch_frames,
        seed_mode="narrow", schedule_mode="fast",
    )

    coarse_fd = np.zeros(B_total, dtype=np.float64)
    for i, fwd in enumerate(fwd_coarse):
        tx_obj = ants.read_transform(fwd[0])
        trans, rot, _ = _extract_rigid_parameters(tx_obj, 3)
        coarse_fd[i] = float(np.sum(np.abs(trans)) + fd_radius * np.sum(np.abs(rot)))

    return coarse_fd, fwd_coarse, inv_coarse


def batched_rigid_register_pass_adaptive(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    coarse_resolution_cap_mm: Optional[float] = 8.0,
    fine_resolution_cap_mm: Optional[float] = 4.0,
    gate_fd_threshold_mm: float = 0.4,
    fd_radius: float = 50.0,
    max_batch_frames: Optional[int] = None,
) -> Tuple[List[List[str]], List[List[str]], float]:
    """Two-pass adaptive alternative to ``batched_rigid_register_pass_temporal``'s
    correlation-based motion gate.

    Real finding (2026-10-04): the correlation-based gate (``motion_gate_threshold``,
    default 0.995 raw voxelwise Pearson correlation to the reference) silently skipped
    registering EVERY frame of a real low-resolution (3.5mm isotropic) PPMI rsfMRI
    series -- reporting exactly 0.0mm FD for all 240 frames, when the trusted
    production backend found real motion (mean FD 0.28mm). Root cause: raw voxelwise
    correlation between frames of a smooth, coarse-resolution image saturates near 1.0
    even with real motion present, so a fixed correlation threshold tuned on finer data
    is not resolution-independent. A pure center-of-mass-shift alternative was tried and
    also rejected (Pearson r=0.17-0.19 against trusted FD on real SOCOM/PPMI BOLD) --
    center of mass is translation-only, and real BOLD motion in both real series checked
    had rotation contributing roughly AS MUCH as translation to FD (~1:1 ratio on real
    SOCOM data), so a translation-only proxy misses about half the real signal.

    This function gates on an actual registration-derived FD estimate instead of a proxy:

    1. Register EVERY frame at ``coarse_resolution_cap_mm`` (cheap: few voxels, narrow
       seed, fast schedule -- same levers as ``batched_rigid_register_pass_temporal``).
       Compute each frame's FD directly from this coarse transform's own recovered
       translation/rotation (not a correlation, not a center-of-mass distance -- the
       actual physical-space rigid parameters this registration pass itself produced).
    2. Frames whose coarse FD is at or below ``gate_fd_threshold_mm`` keep the coarse
       transform (already a real, if lower-precision, registration result -- not an
       identity/no-op like the old gate's skipped frames).
    3. Frames above the threshold are re-registered at ``fine_resolution_cap_mm``
       (narrow seed + fast schedule still, just less aggressively downsampled) and that
       refined transform replaces the coarse one.

    Because the gate signal IS a real rigid-transform FD estimate (mm/degrees, from the
    same Mattes-MI engine used everywhere else in this codebase), it is inherently
    resolution-independent (a physical-space quantity, not a raw-intensity statistic)
    and modality-independent (no assumption about absolute contrast/scale) -- unlike
    both of the proxies above.

    Parameters
    ----------
    coarse_resolution_cap_mm : float, optional
        Spacing (mm) for the cheap gating pass. Default 8.0 -- deliberately coarser than
        ``batched_rigid_register_pass_temporal``'s own 4.0mm default, since this pass's
        only job is a cheap, directionally-correct FD estimate, not the final result for
        low-motion frames (those still get resampled onto ``fine_resolution_cap_mm`` /
        native resolution in the caller's own final step -- this function only returns
        transforms, not resampled images).
    fine_resolution_cap_mm : float, optional
        Spacing (mm) for the refinement pass applied only to gated-in (high-motion)
        frames. Same default (4.0mm) as ``batched_rigid_register_pass_temporal``.
    gate_fd_threshold_mm : float, default 0.4
        Frames with coarse-pass FD at or below this keep the coarse (cheap) result.
    fd_radius : float, default 50.0
        Head radius (mm) for converting rotation to displacement in the gate's own FD
        computation -- same convention/default as ``calculate_framewise_displacement``.
    num_bins, verbose, outprefix, device, max_batch_frames
        See ``batched_rigid_register_pass``.

    Returns
    -------
    (fwd_transforms, inv_transforms, elapsed) -- same shape/contract as
    ``batched_rigid_register_pass``/``batched_rigid_register_pass_temporal``, so this is
    a drop-in swap at any call site that accepts one of those.
    """
    import time as _time

    t0 = _time.time()
    B_total = len(moving_imgs)
    if B_total == 0:
        return [], [], 0.0

    coarse_prefix = f"{outprefix}coarse_" if outprefix is not None else None
    t_coarse0 = _time.time()
    coarse_fd, fwd_coarse, inv_coarse = estimate_coarse_fd(
        reference_img, moving_imgs, resolution_cap_mm=coarse_resolution_cap_mm, fd_radius=fd_radius,
        device=device, num_bins=num_bins, outprefix=coarse_prefix, max_batch_frames=max_batch_frames,
    )
    elapsed_coarse = _time.time() - t_coarse0

    needs_refine = coarse_fd > gate_fd_threshold_mm
    n_refine = int(needs_refine.sum())
    if verbose:
        print(f"[syntx.motion_batched] adaptive gate: {n_refine}/{B_total} frames above "
              f"{gate_fd_threshold_mm}mm coarse FD -- refining only those at "
              f"{fine_resolution_cap_mm}mm; the rest keep the {coarse_resolution_cap_mm}mm "
              f"coarse result (coarse pass: {elapsed_coarse:.2f}s).")

    fwd_all: List[List[str]] = list(fwd_coarse)
    inv_all: List[List[str]] = list(inv_coarse)

    if n_refine:
        refine_idx = np.where(needs_refine)[0].tolist()
        ref_fine = _cap_resolution(reference_img, fine_resolution_cap_mm) if fine_resolution_cap_mm else reference_img
        moving_fine = [
            _cap_resolution(moving_imgs[i], fine_resolution_cap_mm) if fine_resolution_cap_mm else moving_imgs[i]
            for i in refine_idx
        ]
        fine_prefix = f"{outprefix}fine_" if outprefix is not None else None
        fwd_fine, inv_fine, _ = batched_rigid_register_pass(
            ref_fine, moving_fine, device=device, num_bins=num_bins, verbose=False,
            outprefix=fine_prefix, max_batch_frames=max_batch_frames,
            seed_mode="narrow", schedule_mode="fast",
        )
        for local_i, global_i in enumerate(refine_idx):
            fwd_all[global_i] = fwd_fine[local_i]
            inv_all[global_i] = inv_fine[local_i]

    elapsed = _time.time() - t0
    return fwd_all, inv_all, elapsed


def _create_identity_transform_file(com_f: np.ndarray, prefix: Optional[str]) -> str:
    if prefix is None:
        work_dir = tempfile.mkdtemp(prefix="syntx_batched_moco_identity_")
        path = os.path.join(work_dir, "identity.mat")
    else:
        path = f"{prefix}identity.mat"
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    tx.set_parameters(np.concatenate([np.eye(3).flatten(), np.zeros(3)]))
    tx.set_fixed_parameters(com_f)
    ants.write_transform(tx, path)
    return path


class GroupBiasPass(NamedTuple):
    """Result of ``batched_group_bias_register_pass``: per-frame forward / inverse transform files, the shared group rotation
    (3, 3) and translation (3,), the run time, and ``convergence`` -- one record per optimiser stage (iterations, blocks,
    per-frame converged flags and final step in mm, as in ``batched_rigid_register_pass``)."""

    fwd: List[List[str]]
    inv: List[List[str]]
    R_group: np.ndarray
    t_group: np.ndarray
    elapsed: float
    convergence: List[dict]


# Point-sampled at FULL resolution throughout; ``sampling`` = fraction of the pool each stage evaluates (see the comment in
# batched_group_bias_register_pass on why there is no pyramid).
_GROUP_BIAS_SCHEDULE = (
    dict(optimizer="adam", iters=130, step_t_vox=0.04, step_r_vox=1.2, level=1, sampling=0.18),
    dict(optimizer="adam", iters=50, step_t_vox=0.01, step_r_vox=0.18, level=1, sampling=0.18),
    dict(optimizer="lbfgs", iters=18, level=1, sampling=0.15),
    dict(optimizer="lbfgs", iters=15, level=1, sampling=0.3),
)


def batched_group_bias_register_pass(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    group_mask: List[bool],
    device: str = "auto",
    num_bins: int = 32,
    verbose: bool = False,
    outprefix: Optional[str] = None,
    metric: str = "mattes",
    huber_delta: float = 0.1,
    convergence_tol_vox: float = 0.05,
    max_extend_factor: int = 4,
    sample_fraction: float = 0.15,
    sampling: str = DEFAULT_SAMPLING,
) -> GroupBiasPass:
    """
    Like ``batched_rigid_register_pass``, but for a series from two acquisition groups that
    differ by a systematic rigid offset (e.g. b0 and DWI frames of one session).

    Jointly estimates, in one batched optimisation, a rigid transform per frame and one
    shared rigid group-bias transform (R_group, t_group) applied only to frames with
    ``group_mask[b] == True``, composed as the outer transform:
    ``R_total = R_group @ R_frame``, ``t_total = R_group @ t_frame + t_group`` (rotations about
    the reference's centre of mass). Frames with ``group_mask[b] == False`` are plain
    per-frame rigid registrations.

    Differences from ``batched_rigid_register_pass``: no phase-correlation / rotation seed
    search (translations start from centre-of-mass offsets, split into the mean over group
    frames for ``t_group`` and per-frame residuals; all rotations start at zero); no pyramid
    (all stages at full resolution on the sampled points); MI only (no correlation term);
    ``num_bins`` default 32. Schedule: Adam 130 and 50 iterations (18 % sampling), then
    LBFGS 18 (15 %) and 15 (30 %) iterations.

    Parameters
    ----------
    reference_img : ants.ANTsImage
        Shared fixed image (3-D), typically the mean of the ``group_mask == False`` frames.
    moving_imgs : list of ants.ANTsImage
        All frames of both groups, on the reference's grid (not checked).
    group_mask : list of bool
        One entry per frame; True where the shared group-bias transform applies.
    device : str, default 'auto'
        'auto' picks cuda, then mps, then cpu.
    num_bins : int, default 32
        Mattes MI histogram bins.
    verbose : bool, default False
        Unused.
    outprefix : str, optional
        Files are ``f"{outprefix}{t:04d}.mat"``; default ``vol{t:04d}.mat`` in a new temporary
        directory.
    sample_fraction, sampling
        As in ``batched_rigid_register_pass`` (``syntx.sampling``): ``sampling`` names the strategy, ``sample_fraction``
        is the fraction of its candidate region the sparsest stage evaluates.

    Returns
    -------
    fwd_transforms, inv_transforms : list of list of str
        As in ``batched_rigid_register_pass``; each file holds the total per-frame transform
        (frame transform composed with the group bias where it applies). Both lists hold the
        same paths.
    R_group : np.ndarray (3, 3)
        Estimated group-bias rotation (identity if ``group_mask`` is all False).
    t_group : np.ndarray (3,)
        Estimated group-bias translation (physical units).
    elapsed : float
        Seconds from set-up to the end of the optimisation. An empty ``moving_imgs`` returns
        ``([], [], np.eye(3), np.zeros(3), 0.0)``.

    Raises
    ------
    NotImplementedError
        If ``reference_img.dimension != 3``.
    ValueError
        If ``len(group_mask) != len(moving_imgs)``, or a frame is not on the reference grid.
    """
    dim = reference_img.dimension
    if dim != 3:
        raise NotImplementedError(
            "motion_batched.batched_group_bias_register_pass only implements the 3D path."
        )
    if len(group_mask) != len(moving_imgs):
        raise ValueError(
            f"group_mask must have one entry per moving image, got {len(group_mask)} "
            f"for {len(moving_imgs)} images."
        )
    _check_frames_on_reference(reference_img, moving_imgs)
    dev = torch.device(_auto_device() if device == "auto" else device)
    B = len(moving_imgs)
    if B == 0:
        return GroupBiasPass([], [], np.eye(3), np.zeros(3), 0.0, [])

    import time
    t0 = time.time()

    f32 = dict(dtype=torch.float32, device=dev)
    sp_xyz = torch.tensor(reference_img.spacing, **f32)
    orig_xyz = torch.tensor(reference_img.origin, **f32)
    dir_xyz = torch.tensor(reference_img.direction, **f32)
    com_f = np.asarray(compute_center_of_mass(reference_img, weighted=True), dtype=np.float64)
    C_phys = torch.tensor(com_f, **f32)

    def to_tensor(img):
        arr = np.transpose(img.numpy().astype(np.float32), (2, 1, 0))
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dev)

    fixed_t_zyx = to_tensor(reference_img)
    moving_stack = torch.stack([to_tensor(m) for m in moving_imgs], dim=0)
    fixed_t = fixed_t_zyx.permute(2, 1, 0)
    shape_xyz = torch.tensor(list(reference_img.shape), **f32)

    # Sample points: the shared implementation and menu (syntx.sampling), as in batched_rigid_register_pass.
    from .sampling import PointSampler, draw_indices

    sched_sampling = min(st["sampling"] for st in _GROUP_BIAS_SCHEDULE)
    sampler = PointSampler(reference_img, strategy=sampling, sample_fraction=sample_fraction, stage_sampling=sched_sampling,
                           min_points=2000, seed=42, device=dev)
    rng = sampler.generator
    _pool = sampler.pool()
    vox_idx, phys_X = _pool["vox_xyz"], _pool["phys"]
    fixed_vals = F.grid_sample(fixed_t_zyx[None, None], (2.0 * vox_idx / (shape_xyz - 1.0) - 1.0).view(1, 1, 1, -1, 3),
                               mode="bilinear", align_corners=True).reshape(-1)

    lo, hi = 0.0, float(fixed_t.max())
    fixed_scaled = (fixed_vals - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    w_y = parzen_weights(fixed_scaled, num_bins=num_bins)

    _warp1 = {}

    def warp_to_moving_norm(y_phys, shape_lvl, level=1):
        if level != 1:
            raise ValueError(f"batched_group_bias_register_pass has no pyramid; level {level} was requested (only level 1 exists)")
        if not _warp1:
            _warp1['fn'] = prebuild_warp(orig_xyz.cpu().numpy(), dir_xyz.cpu().numpy(), sp_xyz.cpu().numpy(), (1.0,) * 3, (0.0,) * 3,
                                         [int(x) for x in shape_lvl.tolist()][::-1], device=y_phys.device, dtype=y_phys.dtype)
        return _warp1['fn'](y_phys)

    pool_entry = dict(X=phys_X, w_y=w_y, moving=moving_stack, shape=shape_xyz, fixed_vals=fixed_vals, w=sampler.weights())

    dwi_mask_np = np.asarray(group_mask, dtype=bool)
    dwi_mask = torch.tensor(dwi_mask_np, device=dev)
    Ndwi = int(dwi_mask_np.sum())

    # Per-frame CoM offsets vs. the reference's own CoM. For non-group (b0-like) frames
    # this is the direct t_param init (group component is a no-op there via the mask).
    # For group frames, seed the SHARED t_group from the mean group-frame CoM offset (the
    # common component) and each group frame's own t_param from its RESIDUAL after
    # subtracting that mean -- starting near the right per-frame/group decomposition
    # instead of leaving the optimizer to discover the split from scratch.
    com_offsets = np.zeros((B, 3), dtype=np.float32)
    for b, m in enumerate(moving_imgs):
        com_m = np.asarray(compute_center_of_mass(m, weighted=True), dtype=np.float64)
        com_offsets[b] = (com_m - com_f).astype(np.float32)
    t_group_init = com_offsets[dwi_mask_np].mean(axis=0) if Ndwi > 0 else np.zeros(3, dtype=np.float32)
    t_init = com_offsets.copy()
    t_init[dwi_mask_np] -= t_group_init

    # Deliberately NOT using the single-group solver's phase-correlation + fit_pair coarse
    # candidate search here: that machinery fits each frame as an ISOLATED rigid transform
    # against the reference (no group concept), so for group-masked frames it converges near
    # the TOTAL offset (frame jitter + shared group bias combined). Assigning that directly
    # as t_param (meant to hold only the frame-LOCAL component) while t_group is ALSO
    # separately nonzero double-counts the shared component for every group frame --
    # confirmed empirically: adding that seeding step here made recovery WORSE (1.5-2.7mm
    # group-bias error) than the simpler CoM-decomposed init below (1.238mm, matching
    # `scripts/prototype_batched_joint_group_bias.py`'s validated result), even after
    # attempting to correct the double-count by subtracting t_group_init back out. The
    # decomposed CoM init plus the joint schedule's own coarse Adam stage is what's
    # validated to work for THIS (group-bias) model; omega starts at zero (real head
    # rotations are small enough that Adam's coarse stage captures them directly).
    omega = torch.zeros(B, 3, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)
    omega_group = torch.zeros(3, device=dev, requires_grad=True)
    t_group = torch.tensor(t_group_init, device=dev, requires_grad=True)

    # Point-sampled at FULL resolution throughout (no avg-pool coarse pyramid levels): the
    # true multi-res pyramid used by the single-group solver was tried here and confirmed
    # to HURT this decomposition task specifically (worse group-bias recovery, 1.4-2.7mm
    # vs 1.238mm) -- averaging pools the frame-local and shared-group signals together at
    # very few effective voxels/frame, exactly when the optimizer most needs to tell them
    # apart. This matches `scripts/prototype_batched_joint_group_bias.py`'s validated
    # schedule shape (point-sampling fraction only, never true downsampling).
    schedule = [dict(st) for st in _GROUP_BIAS_SCHEDULE]

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0):
        R_frame = _batched_rodrigues(omega)
        R_group = _batched_rodrigues(omega_group.unsqueeze(0))[0]
        Xc = X_stage - C_phys

        # Total per-frame transform composes the shared group-bias transform SECOND
        # (outer): R_total = R_group @ R_frame, t_total = R_group @ t_frame + t_group --
        # see the module-level docstring on `batched_group_bias_register_pass` and
        # `scripts/prototype_batched_joint_group_bias.py` for the pull-back convention
        # this matches. Non-group frames get R_group=I, t_group=0 via the mask, leaving
        # them as plain per-frame rigid registration.
        R_group_b = R_group.unsqueeze(0).expand(B, -1, -1)
        t_group_b = t_group.unsqueeze(0).expand(B, -1)
        apply_group = dwi_mask.float().view(B, 1, 1)
        eye3 = torch.eye(3, device=dev).unsqueeze(0)
        R_eff_group = apply_group * R_group_b + (1 - apply_group) * eye3
        t_eff_group = dwi_mask.float().view(B, 1) * t_group_b

        R_total = torch.bmm(R_eff_group, R_frame)
        t_total = torch.einsum("bij,bj->bi", R_eff_group, t_param) + t_eff_group

        y_phys = torch.einsum("bij,sj->bsi", R_total, Xc) + C_phys + t_total.unsqueeze(1)
        grid_norm = warp_to_moving_norm(y_phys, shape_stage, level=level_stage)
        grid = grid_norm.view(B, 1, 1, -1, 3)
        warped = grid_sample_nd(moving_tensor.unsqueeze(1), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True,
                                 interpolator="linear", use_analytical_gradients=True).reshape(B, -1)
        w_scaled = (warped - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        if metric == "huber":
            fixed_scaled_stage = (fixed_vals_stage - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
            per_frame_loss = huber_loss_nd(w_scaled, fixed_scaled_stage.unsqueeze(0).expand_as(w_scaled),
                                           delta=huber_delta, reduction="none").mean(dim=1)
        else:
            w_x = parzen_weights(w_scaled.reshape(-1), num_bins=num_bins).reshape(*w_scaled.shape, num_bins)
            per_frame_loss = batched_mattes_mi_from_weights(w_x, w_y_stage)
        if corr_weight > 0.0 and fixed_vals_stage is not None:
            per_frame_loss = per_frame_loss + corr_weight * pearson_correlation_loss(warped, fixed_vals_stage)
        return per_frame_loss

    ref_min_sp = float(min(reference_img.spacing))
    rot_scale_mean = float(np.mean([rotation_axis_scale(phys_X, C_phys, j) for j in range(3)]))
    convergence_log = []
    for stage in schedule:
        level = stage.get("level", 1)
        samp = stage.get("sampling", 1.0)
        lvl_data = pool_entry
        n_pool = int(lvl_data["X"].shape[0])
        # the pool is sized so the sparsest stage (sched_sampling) evaluates sample_fraction of the candidate region; each
        # stage draws samp of the pool with the strategy's weights
        k = sampler.eval_points(samp, n_pool, minimum=500)   # capped at MAX_EVAL_POINTS
        sub = None if k >= n_pool else draw_indices(lvl_data["w"], n_pool, k, rng)
        X_stage = lvl_data["X"] if sub is None else lvl_data["X"][sub]
        w_y_stage = lvl_data["w_y"] if sub is None else lvl_data["w_y"][sub]
        fixed_vals_stage = lvl_data["fixed_vals"] if sub is None else lvl_data["fixed_vals"][sub]
        moving_tensor, shape_stage = lvl_data["moving"], lvl_data["shape"]
        corr_weight = stage.get("corr_weight", 0.0)
        params = [t_param, omega, t_group, omega_group]

        def stage_points(X_stage=X_stage):
            """Positions of a fixed subset of the stage's points under each frame's TOTAL transform (frame o group bias)."""
            with torch.no_grad():
                Xs = X_stage[::max(1, X_stage.shape[0] // _N_CONVERGENCE_POINTS)]
                R_frame = _batched_rodrigues(omega)
                R_group = _batched_rodrigues(omega_group.unsqueeze(0))[0]
                apply_group = dwi_mask.float().view(B, 1, 1)
                R_eff = apply_group * R_group.unsqueeze(0).expand(B, -1, -1) + (1 - apply_group) * torch.eye(3, device=dev).unsqueeze(0)
                t_eff = dwi_mask.float().view(B, 1) * t_group.unsqueeze(0).expand(B, -1)
                R_total = torch.bmm(R_eff, R_frame)
                t_total = torch.einsum("bij,bj->bi", R_eff, t_param) + t_eff
                return torch.einsum("bij,sj->bsi", R_total, Xs - C_phys) + C_phys + t_total.unsqueeze(1)

        lr_groups = None
        if stage.get("optimizer", "adam") != "lbfgs":
            lr_groups = [{"params": [t_param, t_group], "lr": stage["step_t_vox"] * ref_min_sp},
                         {"params": [omega, omega_group], "lr": stage["step_r_vox"] * ref_min_sp / rot_scale_mean}]
        rec = _run_converging_stage(
            optimizer=stage.get("optimizer", "adam"), params=params, per_frame_params=[t_param, omega],
            loss_fn=lambda: compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=level,
                                         fixed_vals_stage=fixed_vals_stage, corr_weight=corr_weight),
            points_fn=stage_points, iters=int(stage["iters"]), B=B, dev=dev,
            tol=float(convergence_tol_vox) * ref_min_sp, max_extend=int(max_extend_factor),
            lr_groups=lr_groups, lr_t=stage.get("step_t_vox", 0.0) * ref_min_sp,
            lr_r=stage.get("step_r_vox", 0.0) * ref_min_sp / rot_scale_mean, point_radius=rot_scale_mean,
            label=f"{stage.get('optimizer', 'adam')}@L{level}", level=level)
        convergence_log.append(rec)

    elapsed = time.time() - t0
    if verbose:
        print(f"[motion_batched] group-bias pass: {B} frames ({int(np.sum(group_mask))} in the "
              f"group) optimised in {elapsed:.1f}s on {dev}")

    R_frame_final = _batched_rodrigues(omega).detach().cpu().numpy()
    t_frame_final = t_param.detach().cpu().numpy()
    R_group_final = _batched_rodrigues(omega_group.unsqueeze(0))[0].detach().cpu().numpy()
    t_group_final = t_group.detach().cpu().numpy()

    if outprefix is None:
        work_dir = tempfile.mkdtemp(prefix="syntx_batched_moco_group_")
        prefix = os.path.join(work_dir, "vol")
    else:
        prefix = outprefix

    # Write the TOTAL per-frame transform (jitter composed with group bias where the
    # mask applies), so callers use this exactly like `batched_rigid_register_pass`'s
    # output -- no group-bias-aware logic needed downstream.
    fwd_transforms, inv_transforms = [], []
    for b in range(B):
        if dwi_mask_np[b]:
            R_b = R_group_final @ R_frame_final[b]
            t_b = R_group_final @ t_frame_final[b] + t_group_final
        else:
            R_b = R_frame_final[b]
            t_b = t_frame_final[b]
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([R_b.flatten(), t_b]))
        tx.set_fixed_parameters(com_f)
        tx_path = f"{prefix}{b:04d}.mat"
        ants.write_transform(tx, tx_path)
        fwd_transforms.append([tx_path])
        inv_transforms.append([tx_path])

    return GroupBiasPass(fwd_transforms, inv_transforms, R_group_final, t_group_final, elapsed, convergence_log)
