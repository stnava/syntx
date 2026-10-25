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
from typing import List, NamedTuple, Optional, Tuple

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .robust_affine import compute_center_of_mass
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
# (pool 50 % and stage 20 % -> 10 %), and the stage subset is re-drawn from the pool at every convergence checkpoint.
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


def separate_bold_pe_pseudomotion(
    bold_image: ants.ANTsImage,
    moco: dict,
    pe_axis: Optional[np.ndarray] = None,
    fd_radius: float = 50.0,
    window_size: Optional[int] = None,
) -> Tuple[dict, Optional[dict]]:
    """Separate respiratory B0 phase-encode pseudomotion from the REPORTED rigid head motion of an EPI series.

    Opt-in (``motion_correction(..., debias_pe_pseudomotion=True)``). In EPI a respiratory field offset
    ``Delta B0(t)`` displaces the image along the phase-encode axis ``p_PE`` only (bandwidth along PE is low; readout
    and slice selection are not affected to first order). Only the translation along ``p_PE`` is therefore modelled:
    slice translation and pitch are dominated by real head nodding and are never modified.

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

    D = np.asarray(getattr(bold_image, "direction", np.eye(3)), dtype=np.float64)[:3, :3].copy()
    for col in range(3):
        nrm = float(np.linalg.norm(D[:, col]))
        if nrm > 1e-8:
            D[:, col] /= nrm

    if pe_axis is None:
        raise ValueError("separate_bold_pe_pseudomotion needs pe_axis (physical unit vector or voxel axis index 0/1/2)")
    arr_pe = np.asarray(pe_axis, dtype=np.float64).reshape(-1)
    if arr_pe.size == 1 and float(arr_pe[0]) in (0.0, 1.0, 2.0):
        j_pe = int(arr_pe[0])
        p_pe = D[:, j_pe].copy()
    elif arr_pe.size == 3 and float(np.linalg.norm(arr_pe)) > 1e-8:
        p_pe = arr_pe / float(np.linalg.norm(arr_pe))
        j_pe = int(np.argmax(np.abs(D.T @ p_pe)))
    else:
        raise ValueError(f"pe_axis must be a voxel axis index (0, 1, 2) or a non-zero 3-vector, got {pe_axis!r}")

    rem_axes = [a for a in range(3) if a != j_pe]
    k_sl = 2 if 2 in rem_axes else rem_axes[-1]
    p_sl_raw = D[:, k_sl] - float(np.dot(D[:, k_sl], p_pe)) * p_pe
    if float(np.linalg.norm(p_sl_raw)) < 1e-6:
        k_sl = rem_axes[0]
        p_sl_raw = D[:, k_sl] - float(np.dot(D[:, k_sl], p_pe)) * p_pe
    p_sl = p_sl_raw / max(float(np.linalg.norm(p_sl_raw)), 1e-12)
    p_ro = np.cross(p_pe, p_sl)
    p_ro = p_ro / max(float(np.linalg.norm(p_ro)), 1e-12)

    t_ro = t_all @ p_ro
    t_pe = t_all @ p_pe

    r_pe = rv_mm @ p_pe
    r_sl = rv_mm @ p_sl

    w_size = int(window_size) if (window_size is not None and int(window_size) > 0) else max(8, min(24, T // 5))
    H = build_encoding_covariates(num_frames=T, window_size=w_size)["psi"]

    Y_epi = t_pe[:, None]
    gram = H.T @ H + 1e-4 * np.eye(H.shape[1], dtype=np.float64)
    theta = np.linalg.solve(gram, H.T @ Y_epi)
    for _ in range(3):
        res = Y_epi - H @ theta
        for k in range(Y_epi.shape[1]):
            med_k = float(np.median(res[:, k]))
            mad_k = max(1.4826 * float(np.median(np.abs(res[:, k] - med_k))), 1e-6)
            u_k = (res[:, k] - med_k) / (4.685 * mad_k)
            w_k = np.where(np.abs(u_k) < 1.0, (1.0 - u_k ** 2) ** 2, 1e-4)
            Hw = H * w_k[:, None]
            theta[:, k] = np.linalg.solve(Hw.T @ H + 1e-4 * np.eye(H.shape[1], dtype=np.float64), Hw.T @ Y_epi[:, k])

    r_epi = Y_epi - H @ theta
    sig_ref = float(
        np.median([
            np.median(np.abs(np.diff(t_ro))),
            np.median(np.abs(np.diff(r_pe))),
            np.median(np.abs(np.diff(r_sl))),
        ])
    )
    sigs = [float(np.median(np.abs(np.diff(r_epi[:, k])))) for k in range(Y_epi.shape[1])]
    etas = [max(0.0, 1.0 - (sig_ref / max(s, 1e-12))) if s > 1.10 * sig_ref else 0.0 for s in sigs]
    if max(etas) <= 0.05:
        return moco, None


    d_ref_step = np.zeros(T, dtype=np.float64)
    d_ref_step[1:] = np.abs(np.diff(t_ro)) + np.abs(np.diff(r_pe)) + np.abs(np.diff(r_sl))
    d_ref_local = np.maximum(d_ref_step, np.r_[d_ref_step[1:], 0.0])
    tau_ref = max(0.25, 3.5 * float(np.median(d_ref_step[1:])))
    w_b0 = np.exp(-np.maximum(0.0, (d_ref_local - tau_ref) / max(0.15, tau_ref)) ** 2)

    deltas = np.zeros_like(r_epi)
    for k in range(Y_epi.shape[1]):
        if etas[k] <= 0.0:
            continue
        dr_k = np.zeros(T, dtype=np.float64)
        dr_k[1:] = np.abs(np.diff(r_epi[:, k]))
        dr_local = np.maximum(dr_k, np.r_[dr_k[1:], 0.0])
        scale_k = max(1.4826 * sigs[k], 0.02)
        u_step = np.maximum(0.0, dr_local - 3.5 * scale_k) / (2.5 * scale_k)
        w_step_k = np.exp(-(u_step ** 2))
        deltas[:, k] = etas[k] * w_b0 * w_step_k * r_epi[:, k]

    t_clean = t_all - np.outer(deltas[:, 0], p_pe)
    R_clean = R_all

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

    meta = {
        "method": "overlapping_window_wiener_tukey_epi_plane",
        "pe_vector_lps": [float(x) for x in p_pe],
        "slice_vector_lps": [float(x) for x in p_sl],
        "readout_vector_lps": [float(x) for x in p_ro],
        "sigma_ref_mm": sig_ref,
        "sigma_epi_mm": [float(s) for s in sigs],
        "wiener_shrinkage_eta": [float(e) for e in etas],
        "removed_pe_rms_mm": float(np.sqrt(np.mean(deltas[:, 0] ** 2))),
        "window_size": w_size,
    }
    out["pe_pseudomotion_debias"] = meta
    return out, meta



def _run_converging_stage(*, optimizer, params, per_frame_params, loss_fn, points_fn, iters, B, dev, tol, max_extend,
                          lr_groups=None, lr_t=None, lr_r=0.0, point_radius=0.0, probe_fn=None, confirm_vox=None, axes=None, stage_voxel=None,
                          max_step_vox=1.0, label="", level=1, shared_params=None, reg_loss_fn=None, resample_fn=None):
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

        def _grad_param():
            with torch.enable_grad():
                total = loss_fn()[tracker.active].sum()
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
            g_par = _grad_param()
            H_par = torch.zeros(B, P, P, device=dev)
            for a in range(P):
                d = 0.5 * vox / float(scale[a])
                _shift(a, +d); gp = _grad_param()
                _shift(a, -2 * d); gm = _grad_param()
                _shift(a, +d)
                H_par[:, :, a] = (gp - gm) / (2.0 * d)
            H_par = 0.5 * (H_par + H_par.transpose(1, 2))
            Gv = g_par * to_vox
            Hv = H_par * to_vox.view(1, P, 1) * to_vox.view(1, 1, P)
            ev, Q = (t.to(dev) for t in torch.linalg.eigh(Hv.detach().cpu()))   # B x 6 x 6: eigh is not implemented on MPS, and is tiny
            floor = 1e-3 * ev.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
            inv = 1.0 / (torch.maximum(ev.abs(), floor) + lam.unsqueeze(1))
            dv = -(Q @ (inv * (Q.transpose(1, 2) @ Gv.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)   # (B, P) voxels
            remaining = dv.abs().max(dim=1).values
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
                f0 = loss_fn().detach()
                best_f, best_a = f0.clone(), torch.zeros(B, device=dev)
                for alpha in (1.0, 0.5, 0.25, 0.1):
                    for k, ax in enumerate(axes):
                        ax.tensor[ax.index] = base[k] + alpha * dv[:, k] / to_vox[k]
                    f = loss_fn().detach()
                    better = tracker.active & (f < best_f)
                    best_f = torch.where(better, f, best_f)
                    best_a = torch.where(better, torch.full_like(best_a, alpha), best_a)
                for k, ax in enumerate(axes):
                    ax.tensor[ax.index] = base[k] + best_a * dv[:, k] / to_vox[k]
            moved = best_a > 0
            full_step = best_a >= 1.0
            trust = torch.where(full_step, torch.clamp(trust * 2.0, max=4.0 * float(max_step_vox)), torch.where(moved, trust, trust * 0.5))
            lam = torch.where(moved, lam * 0.5, lam * 4.0 + 1e-3 * ev.abs().max(dim=1).values)
            done += 1
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
    convergence_tol_vox: float = 0.05,
    max_extend_factor: int = 4,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    surround_weight: Optional[float] = None,
    surround_delta_mm: float = 0.25,
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
            sample_fraction=sample_fraction, convergence_tol_vox=convergence_tol_vox, max_extend_factor=max_extend_factor,
            encoding_covariates=enc_all, window_size=window_size, window_stride=window_stride,
            surround_weight=surround_weight, surround_delta_mm=surround_delta_mm,
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
            sample_fraction=sample_fraction, convergence_tol_vox=convergence_tol_vox, max_extend_factor=max_extend_factor,
            encoding_covariates=None if enc_all is None else enc_all[start:start + chunk_size],
            window_size=window_size, window_stride=window_stride,
            surround_weight=surround_weight, surround_delta_mm=surround_delta_mm,
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
    convergence_tol_vox: float = 0.05,
    max_extend_factor: int = 4,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    surround_weight: Optional[float] = None,
    surround_delta_mm: float = 0.25,
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
    are all zero (b0) get no eddy term. Output per frame: ``[eddy_affine.mat, rigid.mat]`` (ANTs order: the rigid is
    applied first) or ``[rigid.mat]`` when the term is zero; FD and the reported pose come from the rigid file alone.
    ``pe_affine`` = ``{'axis', 'coefficients' (4, k), 'a' (B, 4), 'jacobian' (B,)}`` or ``None``.

    ``sample_fraction``: the fraction of the head mask (``ants.get_mask`` of the reference, dilated 2 voxels; or
    ``sample_mask``) that EACH full-resolution metric evaluation sees. The stages draw ``_STAGE_SAMPLING`` (20 %) of a
    pool, so the pool holds ``sample_fraction / 0.2`` of the mask (0.10 -> 50 %), and the stage subset is re-drawn from
    the pool at every convergence checkpoint. It sets the precision floor of every pose. Before 6.22 it sized the pool
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

    1. Sample ``max(2000, 15 %)`` of the reference voxels with intensity > 0.01 (seeded,
       deterministic). Intensities are scaled with the reference range [0, max(reference)]
       for the histograms (assumes non-negative images).
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

    # Foreground = the head, from the reference: ants.get_mask (threshold at the mean, morphological cleanup, holes filled),
    # dilated by 2 voxels so the head boundary -- the strongest rigid-alignment signal -- is sampled. "Any voxel above
    # 0.01" admitted background noise and ghosting and depended on the intensity scale.
    _ref_np = np.ascontiguousarray(fixed_t.detach().cpu().numpy())
    _ref_img = ants.from_numpy(_ref_np, origin=reference_img.origin, spacing=reference_img.spacing, direction=reference_img.direction)
    head_np = ants.iMath(ants.get_mask(_ref_img, cleanup=2), "MD", 2).numpy() > 0
    if int(head_np.sum()) < 500:
        raise ValueError(f"head mask of the motion reference has {int(head_np.sum())} voxels (< 500): the reference is empty "
                         "or has no foreground; pass sample_mask explicitly")
    head_t = torch.from_numpy(head_np).to(dev)                                            # (X, Y, Z)
    head_zyx = head_t.permute(2, 1, 0).float().contiguous()
    fg = head_t.reshape(-1)
    mask_t = None
    if sample_mask is not None:
        m_np = np.asarray(sample_mask.numpy() if hasattr(sample_mask, "numpy") else sample_mask) > 0
        if tuple(m_np.shape) != tuple(reference_img.shape):
            raise ValueError(f"sample_mask shape {m_np.shape} differs from the reference grid {tuple(reference_img.shape)}")
        mask_t_zyx = torch.from_numpy(np.ascontiguousarray(np.transpose(m_np, (2, 1, 0)).astype(np.float32))).to(dev)
        mask_t = mask_t_zyx.permute(2, 1, 0)
        fg = fg & (mask_t.reshape(-1) > 0.5)
        if int(fg.sum()) < 500:
            raise ValueError(f"sample_mask leaves {int(fg.sum())} foreground voxels (< 500)")
    idx_fg = torch.nonzero(fg, as_tuple=False).squeeze(1)
    rng = torch.Generator(device="cpu").manual_seed(42)
    if not 0.0 < float(sample_fraction) <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction!r}")
    pool_fraction = 1.0 if float(sample_fraction) >= 1.0 else min(1.0, float(sample_fraction) / _STAGE_SAMPLING)
    n_sample = min(idx_fg.numel(), max(2000, int(pool_fraction * idx_fg.numel())))
    sel = idx_fg[torch.randperm(idx_fg.numel(), generator=rng)[:n_sample]].to(dev)
    vox_idx = torch.stack(torch.unravel_index(sel, fixed_t.shape), dim=-1).float()

    def _jitter(vox_xyz, shape_xyz_t):
        """Move each sample point uniformly within its voxel. On voxel centres the warped moving frame is interpolated
        exactly at a grid node when the pose is the identity (no noise averaging) and between nodes otherwise (noise
        averaged by the trilinear weights): the metric then rewards fractional-voxel shifts, a pose-dependent bias that
        reached 0.5 mm per b3000 frame on a known-truth phantom. Off-centre points make the moving frame's interpolation
        independent of the pose; the reference is interpolated at the same points (symmetric treatment)."""
        j = vox_xyz + (torch.rand(vox_xyz.shape, generator=rng) - 0.5).to(vox_xyz.device)
        return torch.minimum(j.clamp(min=0.0), shape_xyz_t - 1.0)

    def _interp_fixed(fixed_zyx, vox_xyz, shape_xyz_t):
        """Trilinear values of ``fixed_zyx`` ((Z, Y, X) shared or (B, Z, Y, X) per frame) at voxel positions ``vox_xyz`` (S, 3)."""
        norm = 2.0 * vox_xyz / (shape_xyz_t - 1.0) - 1.0
        src = fixed_zyx.unsqueeze(1) if fixed_zyx.dim() == 4 else fixed_zyx[None, None]
        out = F.grid_sample(src, norm.view(1, 1, 1, -1, 3).expand(src.shape[0], 1, 1, -1, 3), mode="bilinear", align_corners=True)
        return out.reshape(src.shape[0], -1) if fixed_zyx.dim() == 4 else out.reshape(-1)

    vox_idx = _jitter(vox_idx, shape_xyz)
    phys_X = orig_xyz + (vox_idx * sp_xyz) @ dir_xyz.t()
    fixed_vals = _interp_fixed(fixed_stack_zyx if per_frame_ref else fixed_t_zyx, vox_idx, shape_xyz)

    lo, hi = 0.0, float((fixed_stack_zyx if per_frame_ref else fixed_t).max())
    w_y = None   # level-1 Parzen weights are built per draw from the drawn fixed values (see the stage loop)

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


    def get_pyramid_level(level, point_sample_frac=None):
        cache_key = (level, point_sample_frac)
        if cache_key in _pyramid_cache:
            return _pyramid_cache[cache_key]
        if level == 1 and point_sample_frac is None:
            entry = dict(X=phys_X, w_y=w_y, moving=moving_stack, shape=shape_xyz, fixed_vals=fixed_vals,
                         fixed_tensor=(fixed_stack_zyx if per_frame_ref else fixed_t_zyx.unsqueeze(0)))
            _pyramid_cache[cache_key] = entry
            return entry
        fixed_tensor_lvl = fixed_stack_zyx if per_frame_ref else fixed_t_zyx.unsqueeze(0)
        if level == 1:
            phys_X_lvl, fixed_vals_lvl, moving_lvl, shape_lvl = phys_X, fixed_vals, moving_stack, shape_xyz
            fg_mask_lvl = torch.ones(phys_X.shape[0], dtype=torch.bool, device=dev)   # the pool is drawn inside the mask
        else:
            fixed_src = fixed_stack_zyx.unsqueeze(1) if per_frame_ref else fixed_t_zyx.unsqueeze(0).unsqueeze(0)
            if _level_builder is not None:
                _ctx = dict(level=level, sp_min_f=min(float(x) for x in reference_img.spacing), sig_mm=0.0,
                            budget=2 ** 24, device=dev)
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
            axes = [torch.arange(n, device=dev, dtype=torch.float32) for n in shape]
            mesh = torch.meshgrid(*axes, indexing="ij")
            vox_xyz_pooled = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, 3)
            shape_pooled_xyz = torch.tensor([shape[2], shape[1], shape[0]], **f32)
            vox_xyz_pooled = _jitter(vox_xyz_pooled, shape_pooled_xyz)
            vox_xyz_full = vox_xyz_pooled * torch.tensor(f_fx, **f32) + torch.tensor(o_fx, **f32)   # the level's own per-axis f, off
            phys_X_lvl = orig_xyz + (vox_xyz_full * sp_xyz) @ dir_xyz.t()
            fixed_vals_lvl = _interp_fixed(fixed_pool if per_frame_ref else fixed_pool[0], vox_xyz_pooled, shape_pooled_xyz)
            fg_mask_lvl = _interp_fixed(head_zyx, vox_xyz_full, shape_xyz) > 0.5   # the head mask at this level's points
            fixed_tensor_lvl = fixed_pool
            moving_lvl = moving_pool
            shape_lvl = torch.tensor([shape[2], shape[1], shape[0]], **f32)
        if point_sample_frac is not None:
            fg_lvl = fg_mask_lvl.clone()
            if mask_t is not None and level == 1:
                # the mask applies to the full-resolution stages only: the direction-dependent contrast bias it removes
                # is a fine-scale effect, while the coarse stages provide the capture range for real motion and need the
                # whole foreground (masking them cost 0.2 mm of relative-pose error under 1 mm / 1 deg motion on the phantom)
                fg_lvl = fg_lvl & (mask_t.reshape(-1) > 0.5)
            idx_fg_lvl = torch.nonzero(fg_lvl, as_tuple=False).squeeze(1)
            n_lvl = max(300, int(point_sample_frac * idx_fg_lvl.numel()))
            sel_lvl = idx_fg_lvl[torch.randperm(idx_fg_lvl.numel(), generator=rng)[:n_lvl]].to(dev)
            phys_X_lvl = phys_X_lvl[sel_lvl]
            fixed_vals_lvl = _sub(fixed_vals_lvl, sel_lvl)
        fixed_scaled_lvl = (fixed_vals_lvl - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_y_lvl = _parzen(fixed_scaled_lvl)
        entry = dict(X=phys_X_lvl, w_y=w_y_lvl, moving=moving_lvl, shape=shape_lvl, fixed_vals=fixed_vals_lvl,
                     fixed_tensor=fixed_tensor_lvl)
        _pyramid_cache[cache_key] = entry
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

    fit_results = []
    for t_cand in all_t_candidates:
        for seed in rot_seeds:
            fit_results.append(fit_pair(t_cand, seed))
    all_scores = np.stack([s for _, _, s in fit_results], axis=0)
    best_combo = all_scores.argmin(axis=0)
    t_init = np.stack([fit_results[best_combo[b]][0][b] for b in range(B)], axis=0).astype(np.float32)
    omega_init = np.stack([fit_results[best_combo[b]][1][b] for b in range(B)], axis=0).astype(np.float32)
    if verbose:
        print(f"[motion_batched] translation+rotation init ({seed_mode}): searched "
              f"{len(all_t_candidates)} translation candidates x {len(rot_seeds)} rotation "
              f"seeds = {len(fit_results)} combos/frame")

    omega = torch.tensor(omega_init, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)

    if enc_on:
        enc_unit = torch.tensor([1.0, 1.0, 1.0] + [1.0 / max(rot_scale_mean, 1e-6)] * 3, **f32)

    def _encoding_offset():
        """(B, 6) encoding-locked pose offset E_psi . B_win in (mm, rad)."""
        return torch.einsum("bmk,mkd->bd", E_psi_t, B_win) * enc_unit

    def _total_pose():
        """Pose the image metric sees: head motion (t_param, omega) plus the encoding-locked offset."""
        if not enc_on:
            return t_param, omega
        dY = _encoding_offset()
        return t_param + dY[:, :3], omega + dY[:, 3:]

    def _surround_reg_loss():
        """Charbonnier penalty on the second-order surround of the HEAD trajectory only (mm; rotations x 50 mm). Smooth
        drift is in its nullspace; a real jump costs ~linearly (not quadratically), so it is not pulled toward its neighbours."""
        u_t = t_param[1:-1] - 0.5 * (t_param[:-2] + t_param[2:])
        u_w = 50.0 * (omega[1:-1] - 0.5 * (omega[:-2] + omega[2:]))
        u_sq = (u_t * u_t).sum(dim=-1) + (u_w * u_w).sum(dim=-1)
        return surr_w * (surr_delta ** 2) * (torch.sqrt(1.0 + u_sq / (surr_delta ** 2)) - 1.0).sum()

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
            B_np, dY_np, w_np = _project_encoding_surround_tukey_np(Y_tot, E_np, psi_np, fd_radius=50.0, surround_weight=0.0)
            B_win.copy_(torch.tensor(B_np, **f32) / enc_unit)
            t_param.sub_(torch.tensor(dY_np[:, :3], **f32))
            omega.sub_(torch.tensor(dY_np[:, 3:], **f32))
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
    else:
        raise ValueError(f"schedule_mode must be 'default' or 'fast', got {schedule_mode!r}")

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0, fixed_tensor=None):
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
        w_scaled = (warped - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        if metric == "huber":
            fixed_scaled_stage = (fixed_vals_stage - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
            target = fixed_scaled_stage if fixed_scaled_stage.dim() == 2 else fixed_scaled_stage.unsqueeze(0).expand_as(w_scaled)
            per_frame_loss = huber_loss_nd(w_scaled, target, delta=huber_delta, reduction="none").mean(dim=1)
        else:
            w_x = parzen_weights(w_scaled.reshape(-1), num_bins=num_bins).reshape(*w_scaled.shape, num_bins)
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
        # The stage's pool: level 1 = the full-resolution pool drawn in the head mask; coarser levels = every masked voxel
        # of the level. Each metric evaluation sees ``samp`` of the pool, a subset RE-DRAWN at every convergence
        # checkpoint (see ``_run_converging_stage(resample_fn=...)``): a fixed subset makes its own sampling error a
        # deterministic local optimum (slow-flow phantom, 15 % pool x 20 % stage: one seed 0.53 deg rotation error).
        lvl_data = get_pyramid_level(1) if level == 1 else get_pyramid_level(level, point_sample_frac=1.0)
        moving_tensor, shape_stage = lvl_data["moving"], lvl_data["shape"]
        n_pool = int(lvl_data["X"].shape[0])
        k_draw = n_pool if samp >= 1.0 else min(n_pool, max(500 if level == 1 else 300, int(samp * n_pool)))
        cur: dict = {}

        def _draw(lvl_data=lvl_data, cur=cur, k_draw=k_draw, n_pool=n_pool):
            sub = None if k_draw >= n_pool else torch.randperm(n_pool, generator=rng)[:k_draw].to(dev)
            fv = lvl_data["fixed_vals"] if sub is None else _sub(lvl_data["fixed_vals"], sub)
            cur["X"] = lvl_data["X"] if sub is None else lvl_data["X"][sub]
            cur["fixed_vals"] = fv
            if lvl_data["w_y"] is None:
                cur["w_y"] = _parzen((fv - lo) / (hi - lo + 1e-8) * 2.0 - 1.0)
            else:
                cur["w_y"] = lvl_data["w_y"] if sub is None else _sub(lvl_data["w_y"], sub)

        _draw()
        resample_fn = _draw if k_draw < n_pool else None
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
        rec = _run_converging_stage(
            optimizer=stage.get("optimizer", "adam"), params=opt_params, per_frame_params=per_frame_params,
            loss_fn=lambda: compute_loss(cur["X"], cur["w_y"], moving_tensor, shape_stage, level_stage=level,
                                         fixed_vals_stage=cur["fixed_vals"], corr_weight=corr_weight,
                                         fixed_tensor=lvl_data["fixed_tensor"]),
            resample_fn=resample_fn,
            points_fn=stage_points, iters=int(stage["iters"]), B=B, dev=dev,
            tol=float(convergence_tol_vox) * stage_voxel, max_extend=int(max_extend_factor),
            lr_groups=lr_groups, lr_t=stage.get("step_t_vox", 0.0) * ref_min_sp,
            lr_r=stage.get("step_r_vox", 0.0) * ref_min_sp / rot_scale_mean, point_radius=rot_scale_mean,
            probe_fn=lambda: probe_manifold(lambda: compute_loss(cur["X"], cur["w_y"], moving_tensor, shape_stage, level_stage=level,
                                                                 fixed_vals_stage=cur["fixed_vals"], corr_weight=corr_weight,
                                                                 fixed_tensor=lvl_data["fixed_tensor"]),
                                            probe_axes, stage_voxel, probe_vox=0.5),
            confirm_vox=float(stage.get("confirm_vox", _CONFIRM_FACTOR * float(convergence_tol_vox))), axes=probe_axes, stage_voxel=stage_voxel,
            label=f"{stage.get('optimizer', 'adam')}@L{level}" + (f"/{freeze}" if freeze else ""), level=level,
            shared_params=shared_stage_params if shared_stage_params else None,
            reg_loss_fn=_surround_reg_loss if (enc_stage and surr_w > 0.0 and B >= 3) else None)
        _project_encoding_gauge()
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
    if pe_info is None:
        pe_info = {}
    pe_info["convergence"] = convergence_log
    pe_info["sampling"] = {"sample_fraction": float(sample_fraction), "pool_fraction": float(pool_fraction),
                           "head_mask_voxels": int(idx_fg.numel()), "resampled_every_checkpoint": True,
                           "n_sample_full_resolution": int(n_sample),
                           "foreground_voxels": int(idx_fg.numel())}
    fwd_transforms, inv_transforms = [], []
    for b in range(B):
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([R_final[b].flatten(), t_final[b]]))
        tx.set_fixed_parameters(com_f)
        tx_path = f"{prefix}{b:04d}.mat"
        ants.write_transform(tx, tx_path)
        chain = [tx_path]
        if pe_on and np.any(a_np[b] != 0):
            # y' = M (y - c) + c + T with M = I + p a^T (a = shears/scale), T = p a0; applied after the rigid (ANTs order)
            M = np.eye(3) + np.outer(p_np, a_np[b, 1:4])
            te = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
            te.set_parameters(np.concatenate([M.flatten(), p_np * a_np[b, 0]]))
            te.set_fixed_parameters(com_f)
            te_path = f"{prefix}{b:04d}_eddy.mat"
            ants.write_transform(te, te_path)
            chain = [te_path, tx_path]
        fwd_transforms.append(chain)
        inv_transforms.append(list(chain))

    return fwd_transforms, inv_transforms, elapsed, pe_info


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

    fg = fixed_t.reshape(-1) > 0.01
    idx_fg = torch.nonzero(fg, as_tuple=False).squeeze(1)
    rng = torch.Generator(device="cpu").manual_seed(42)
    n_sample = max(2000, int(0.15 * idx_fg.numel()))
    sel = idx_fg[torch.randperm(idx_fg.numel(), generator=rng)[:n_sample]].to(dev)
    vox_idx = torch.stack(torch.unravel_index(sel, fixed_t.shape), dim=-1).float()
    # sample points off the voxel centres, reference interpolated there (see batched_rigid_register_pass)
    vox_idx = torch.minimum((vox_idx + (torch.rand(vox_idx.shape, generator=rng) - 0.5).to(dev)).clamp(min=0.0), shape_xyz - 1.0)
    phys_X = orig_xyz + (vox_idx * sp_xyz) @ dir_xyz.t()
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

    _pyramid_cache = {}

    def get_pyramid_level(level, point_sample_frac=None):
        cache_key = (level, point_sample_frac)
        if cache_key in _pyramid_cache:
            return _pyramid_cache[cache_key]
        if level == 1 and point_sample_frac is None:
            entry = dict(X=phys_X, w_y=w_y, moving=moving_stack, shape=shape_xyz, fixed_vals=fixed_vals)
            _pyramid_cache[cache_key] = entry
            return entry
        if level == 1:
            phys_X_lvl, fixed_vals_lvl, moving_lvl, shape_lvl = phys_X, fixed_vals, moving_stack, shape_xyz
        else:
            # this solver is point-sampled at full resolution only (see the schedule below): there is no pooled level to build
            raise ValueError(f"batched_group_bias_register_pass has no pyramid; level {level} was requested (only level 1 exists)")
        if point_sample_frac is not None:
            fg_lvl = fixed_vals_lvl > 0.01
            idx_fg_lvl = torch.nonzero(fg_lvl, as_tuple=False).squeeze(1)
            n_lvl = max(300, int(point_sample_frac * idx_fg_lvl.numel()))
            sel_lvl = idx_fg_lvl[torch.randperm(idx_fg_lvl.numel(), generator=rng)[:n_lvl]].to(dev)
            phys_X_lvl = phys_X_lvl[sel_lvl]
            fixed_vals_lvl = fixed_vals_lvl[sel_lvl]
        fixed_scaled_lvl = (fixed_vals_lvl - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_y_lvl = parzen_weights(fixed_scaled_lvl, num_bins=num_bins)
        entry = dict(X=phys_X_lvl, w_y=w_y_lvl, moving=moving_lvl, shape=shape_lvl, fixed_vals=fixed_vals_lvl)
        _pyramid_cache[cache_key] = entry
        return entry

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
    schedule = [
        dict(optimizer="adam", iters=130, step_t_vox=0.04, step_r_vox=1.2, level=1, sampling=0.18),
        dict(optimizer="adam", iters=50, step_t_vox=0.01, step_r_vox=0.18, level=1, sampling=0.18),
        dict(optimizer="lbfgs", iters=18, level=1, sampling=0.15),
        dict(optimizer="lbfgs", iters=15, level=1, sampling=0.3),
    ]

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
        if level == 1:
            lvl_data = get_pyramid_level(1)
            if samp < 1.0:
                k = max(500, int(samp * lvl_data["X"].shape[0]))
                sub = torch.randperm(lvl_data["X"].shape[0], generator=rng)[:k].to(dev)
                X_stage, w_y_stage = lvl_data["X"][sub], lvl_data["w_y"][sub]
                fixed_vals_stage = lvl_data["fixed_vals"][sub]
            else:
                X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
                fixed_vals_stage = lvl_data["fixed_vals"]
        else:
            lvl_data = get_pyramid_level(level, point_sample_frac=min(samp, 1.0))
            X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
            fixed_vals_stage = lvl_data["fixed_vals"]
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
