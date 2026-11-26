"""
Joint Label Fusion (JLF) — ``syntx.joint_label_fusion``.

Patch-based multi-atlas label fusion on a PyTorch backend (CPU / MPS / CUDA). Modes:

1. ``mode="patch"``: voxel-wise local normalised cross-correlation (LNCC) weights between the target
   and each warped atlas, ``w_i ∝ LNCC_i^beta``.
2. ``mode="joint"``: joint label fusion of Wang et al., IEEE TPAMI 2013. The weights minimise the
   expected squared label error ``w' M w`` subject to ``sum(w) = 1`` (``w ∝ M^-1 1``) where the
   pairwise error dependence is estimated from the intensity residuals in the voxel's patch,
   ``M_ij(x) = ( mean_{y in patch(x)} |A_i(y)-T(y)| |A_j(y)-T(y)| )^beta``, scaled by its mean
   diagonal and ridge-regularised with ``rho``. Because each diagonal entry is a noisy patch mean, the
   residual matrix is first shrunk toward "all atlases equally good" wherever the spread between atlases
   is within sampling noise (empirical Bayes; see ``_shrink_to_equal_atlases``): without it the weights
   are arbitrary in featureless regions and a one-atlas noise blob can win the vote.
   ``patch_norm=True`` instead standardises each patch before forming the residuals (measured to
   cost ~0.06 mean Dice on the r-image validation, so it is off by default).

Speed and memory design (verified against the dense path in the tests):

* **Consensus skipping.** Where every (selected) atlas label agrees, the fused label is that label and
  its probability is exactly 1 whatever the weights, so weights are formed and solved only on the
  disagreement set dilated by the patch radius (the patches that can reach it) instead of the ROI.
* **Overlapping-patch voting (ANTs, Hongzhi's scheme).** Each voxel pools the weights of all patches
  containing it; per (atlas, offset) that is one box sum of the weight field.
* **Box sums by ``avg_pool``** (~85x faster than a conv1d separable filter on CPU, same values).
* **Elementwise unrolled Cholesky** for the small K x K systems (fast and deterministic everywhere).
* **Search radius** (``r_search``, ANTs ``-s``): per atlas and voxel the offset minimising the patch
  SSD within ``+-r_search`` is chosen; the whole shifted atlas patch then enters the weights (gathered
  only on the solve set), and the atlas label at the shifted position votes.
* **Deterministic ties:** equal class probabilities go to the smallest label.
"""

from __future__ import annotations

import itertools
import time
import warnings
from typing import Any, Literal, Mapping, Sequence

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .core.pipeline import auto_detect_device

_SOLVE_CHUNK = 1 << 21  # voxels per batched solve; bounds peak memory of the K x K systems
_UNROLLED_MAX_K = 32    # largest K solved by the unrolled elementwise Cholesky
_GATHER_ELEMS = 1 << 24  # max (voxels x K x patch) elements gathered at once
_TIE_TOL = 1e-6          # class probabilities within this of the maximum tie; the smallest label wins


def _sync(dev: torch.device) -> None:
    """Block until queued device work finishes so stage timings are real."""
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _box_sum(x: torch.Tensor, rad: int) -> torch.Tensor:
    """Sum of ``x`` over the ``(2*rad+1)**ndim`` box around each voxel, zero-padded (``x``: B,C,*spatial).

    ``avg_pool`` with ``count_include_pad=True`` times the box volume equals the zero-padded sum
    filter exactly (max relative difference vs a conv1d separable filter ~4e-7 in float32) and is
    ~85x faster on CPU for a 256x256 image.
    """
    k = 2 * rad + 1
    pool = F.avg_pool3d if x.dim() == 5 else F.avg_pool2d
    return pool(x, k, stride=1, padding=rad, count_include_pad=True) * float(k ** (x.dim() - 2))


def _search_offsets(r_search: int, dim: int) -> list:
    """Search offsets ordered by squared norm then lexicographically (index 0 is the zero offset)."""
    return sorted(itertools.product(range(-r_search, r_search + 1), repeat=dim),
                  key=lambda o: (sum(c * c for c in o), o))


def _box_sum_chunked(x: torch.Tensor, rad: int, max_channels: int = 32) -> torch.Tensor:
    """Channel-chunked ``_box_sum`` for ``(1, C, *spatial)`` tensors so MPS/CUDA pool kernels stay bounded."""
    c_tot = int(x.shape[1])
    if c_tot <= max_channels:
        return _box_sum(x, rad)
    parts = [_box_sum(x[:, s : s + max_channels], rad) for s in range(0, c_tot, max_channels)]
    return torch.cat(parts, dim=1)


def _search_best_patch_offsets(target_t, atlases_ext, offsets, margin, rad, sub_shape, roi, dev):
    """Per atlas and voxel, pick the offset with the minimum patch SSD and return ``(sel_a, sel_o, fracs)``."""
    roi_t = torch.from_numpy(np.ascontiguousarray(roi)).to(dev)
    sel_a, sel_o, moved_counts_t = [], [], [roi_t.sum()]
    for A in atlases_ext:
        best = None
        for oi, off in enumerate(offsets):
            sl = (slice(None), slice(None)) + tuple(
                slice(margin + o, margin + o + n) for o, n in zip(off, sub_shape)
            )
            a = A[sl].contiguous()
            ssd = _box_sum((a - target_t) ** 2, rad)  # patch SSD (sum; the 1/|patch| factor is common)
            if best is None:
                best, a_sel = ssd, a
                o_sel = torch.zeros_like(ssd, dtype=torch.long)
                continue
            upd = ssd < best
            best = torch.where(upd, ssd, best)
            a_sel = torch.where(upd, a, a_sel)
            o_sel = torch.where(upd, torch.full_like(o_sel, oi), o_sel)
        sel_a.append(a_sel)
        sel_o.append(o_sel)
        moved_counts_t.append(((o_sel[0, 0] != 0) & roi_t).sum())
    moved_vals = torch.stack(moved_counts_t).tolist()
    n_roi = max(int(moved_vals[0]), 1)
    fracs = [float(v) / n_roi for v in moved_vals[1:]]
    return sel_a, sel_o, fracs


def _gather_selected_labels(labels_ext, off_ids, offsets, sub_shape, margin, dev):
    """Gather centre-selected labels ``L_i(n + s_i(n))`` on ``sub_shape`` from ``labels_ext`` and ``off_ids``."""
    dim = len(sub_shape)
    ext_shape = [n + 2 * margin for n in sub_shape]
    st = [int(np.prod(ext_shape[d + 1:])) for d in range(dim)]
    N = int(np.prod(sub_shape))
    rem = torch.arange(N, dtype=torch.long, device=dev)
    base_all = torch.zeros_like(rem)
    for d in range(dim - 1, -1, -1):
        base_all = base_all + (rem % sub_shape[d] + margin) * st[d]
        rem = rem // sub_shape[d]
    shift_t = torch.tensor([int(sum(o * s for o, s in zip(off, st))) for off in offsets], dtype=torch.long, device=dev)
    sel_l = []
    for L, O in zip(labels_ext, off_ids):
        idx = base_all + shift_t[O.reshape(-1)]
        sel_l.append(L.reshape(-1)[idx].reshape(1, 1, *sub_shape))
    return sel_l


def _search_best_patch(target_t, atlases_ext, labels_ext, offsets, margin, rad, sub_shape, roi, dev):
    """Per atlas and voxel, pick the offset with the minimum patch SSD and return its intensity/label."""
    sel_a, sel_o, fracs = _search_best_patch_offsets(
        target_t, atlases_ext, offsets, margin, rad, sub_shape, roi, dev
    )
    sel_l = _gather_selected_labels(labels_ext, sel_o, offsets, sub_shape, margin, dev)
    return sel_a, sel_l, sel_o, fracs


def _spd_solve_ones_unrolled(A: torch.Tensor):
    """Solve ``A w = 1`` for ``n`` SPD ``K x K`` systems with an unrolled Cholesky.

    Every operation is an elementwise op on length-``n`` vectors (``K^3/6`` of them), so it is fast
    and deterministic on any device, where batched ``torch.linalg`` kernels are slow on MPS and CPU
    for tiny matrices. Returns ``(w (n, K), ok (n,))``; ``ok`` is False where a pivot is not
    positive (the system is not numerically SPD).
    """
    n, K, _ = A.shape
    At = A.permute(1, 2, 0).contiguous()  # (K, K, n)
    L = [[None] * K for _ in range(K)]
    ok = torch.ones(n, dtype=torch.bool, device=A.device)
    for j in range(K):
        s = At[j, j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
        ok &= s > 1e-12
        d = torch.sqrt(torch.clamp_min(s, 1e-12))
        L[j][j] = d
        for i in range(j + 1, K):
            s = At[i, j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            L[i][j] = s / d
    y = [None] * K
    for i in range(K):
        s = torch.ones(n, dtype=A.dtype, device=A.device)
        for k in range(i):
            s = s - L[i][k] * y[k]
        y[i] = s / L[i][i]
    w = [None] * K
    for i in reversed(range(K)):
        s = y[i]
        for k in range(i + 1, K):
            s = s - L[k][i] * w[k]
        w[i] = s / L[i][i]
    return torch.stack(w, dim=1), ok


def _solve_weights(M_reg: torch.Tensor, solver: str, stats: dict) -> torch.Tensor:
    """Solve ``M w = 1`` for a batch of SPD ``K x K`` systems, chunked. Returns ``(M, K)``.

    ``solver='cholesky'`` uses the unrolled elementwise Cholesky (``K <= 32``) or
    ``cholesky_ex`` beyond that; voxels that are not numerically SPD are re-solved by LU, with a
    warning and the count recorded in ``stats``. ``solver='lu'`` is the dense reference.
    """
    n, K, _ = M_reg.shape
    out = torch.empty((n, K), dtype=M_reg.dtype, device=M_reg.device)
    for s in range(0, n, _SOLVE_CHUNK):
        Mc = M_reg[s:s + _SOLVE_CHUNK]
        ones = torch.ones((Mc.shape[0], K, 1), dtype=Mc.dtype, device=Mc.device)
        if solver == "lu":
            out[s:s + Mc.shape[0]] = torch.linalg.solve(Mc, ones).squeeze(-1)
            continue
        if K <= _UNROLLED_MAX_K:
            w, ok = _spd_solve_ones_unrolled(Mc)
            bad = ~ok
        else:
            L, info = torch.linalg.cholesky_ex(Mc)
            w = torch.cholesky_solve(ones, L).squeeze(-1)
            bad = info != 0
        n_bad = int(bad.sum().item())
        if n_bad:
            warnings.warn(f"joint_label_fusion: Cholesky failed on {n_bad} voxels; re-solved with LU", RuntimeWarning)
            stats["cholesky_failed"] += n_bad
            w[bad] = torch.linalg.solve(Mc[bad], ones[bad]).squeeze(-1)
        out[s:s + Mc.shape[0]] = w
    return out


def _patch_weights(ncc: torch.Tensor, beta: float) -> torch.Tensor:
    """``w_i ∝ LNCC_i^beta`` from ``ncc`` of shape ``(n, K)`` (already clamped to >= 0)."""
    w = (ncc ** beta) + 1e-6
    return w / w.sum(dim=1, keepdim=True)


_SHRINK_C = 3.0  # shrink unless the cross-atlas spread of the diagonal exceeds _SHRINK_C x its sampling variance


def _shrink_to_equal_atlases(G: torch.Tensor, p_eff) -> torch.Tensor:
    """Shrink ``G`` (n, K, K) toward the "all atlases equally good" target.

    Each diagonal entry is a patch mean of ``p_eff`` squared residuals, so for white Gaussian
    residuals it has relative sampling variance ``2 / p_eff``. When the observed spread of the K
    diagonal entries is not larger than that noise, the differences between atlases are not
    information, and unshrunk weights are arbitrary (a one-atlas noise blob can win the vote).
    Empirical-Bayes weight ``lam = min(1, c * sampling_var / observed_var)`` moves ``G`` toward the
    target with diagonal = mean diagonal and off-diagonal = mean off-diagonal (equal weights);
    where atlases genuinely differ ``lam -> 0`` and ``G`` is untouched. ``c = 3`` because the spread
    of K diagonal values is chi-square distributed with K-1 dof and heavy tailed; c in {1, 2, 3, 5}
    left 28, 1, 0, 0 of 1280 noise-only voxels flipped (beta 4), the unshrunk estimator 143.
    """
    n, K, _ = G.shape
    if K < 2:
        return G
    d = torch.diagonal(G, dim1=-2, dim2=-1)  # (n, K)
    m = d.mean(dim=-1)
    obs = d.var(dim=-1, unbiased=True)
    samp = (2.0 / p_eff) * m * m
    lam = torch.clamp(_SHRINK_C * samp / torch.clamp_min(obs, 1e-30), 0.0, 1.0).view(n, 1, 1)
    off = (G.sum(dim=(-2, -1)) - d.sum(dim=-1)) / float(K * (K - 1))
    eye = torch.eye(K, dtype=G.dtype, device=G.device).unsqueeze(0)
    target = off.view(n, 1, 1) * (1.0 - eye) + m.view(n, 1, 1) * eye
    return (1.0 - lam) * G + lam * target


def _wang_weights(G: torch.Tensor, beta: float, rho: float, solver: str, nonnegative: bool, stats: dict,
                  p_eff=None) -> torch.Tensor:
    """Wang et al. weights from the mean patch residual products ``G`` of shape ``(n, K, K)``.

    With ``p_eff`` (patch voxels per entry, scalar or ``(n,)``) ``G`` is first shrunk toward equal
    atlases when their differences are within sampling noise (:func:`_shrink_to_equal_atlases`).
    ``M = G^beta`` (elementwise), divided by its mean diagonal (scale free), plus ``rho`` on the
    diagonal; ``w ∝ M^-1 1``, optionally clamped to ``w >= 0``, normalised to sum 1 (uniform where
    the sum vanishes).
    """
    K = G.shape[1]
    if p_eff is not None:
        G = _shrink_to_equal_atlases(G, p_eff)
    eye = torch.eye(K, dtype=G.dtype, device=G.device).unsqueeze(0)
    Mp = torch.clamp_min(G, 1e-12) ** beta
    dm = torch.diagonal(Mp, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    M_reg = Mp / torch.clamp_min(dm, 1e-30) + (rho + 1e-6) * eye
    w = _solve_weights(M_reg, solver, stats)
    if nonnegative:
        w = torch.clamp_min(w, 0.0)
    s = w.sum(dim=-1, keepdim=True)
    return torch.where(s < 1e-6, torch.full_like(w, 1.0 / K), w / torch.clamp_min(s, 1e-7))


_VAR_FLOOR = 1e-4  # floor on a patch's sum of squared deviations (intensities normalised to ~[0, 1])


def _standardise(p: torch.Tensor) -> torch.Tensor:
    """Centre each patch (last dim) and divide by the root of its sum of squared deviations.

    The deviation sum is floored at ``_VAR_FLOOR`` so featureless patches map continuously to ~0
    instead of amplifying noise; ``z_i . z_j`` is then the patch correlation ``r_ij``.
    """
    c = p - p.mean(dim=-1, keepdim=True)
    return c / torch.sqrt(torch.clamp_min((c * c).sum(dim=-1, keepdim=True), _VAR_FLOOR))


def _gather_weights(target_ext, atlases_ext, off_ids, offsets, didx, sub_shape, margin, rad, mode,
                    beta, rho, solver, nonnegative, stats, dev, patch_norm=False):
    """Weights on the disagreement set from the *whole shifted patch* of each atlas.

    For voxel ``x`` and atlas ``i`` with selected offset ``s_i(x)`` the atlas patch is
    ``A_i(y + s_i(x))``, ``y`` in the ``rad``-patch around ``x``; it is compared with ``T(y)``.
    Tensors are the margin-extended crops, indexed by flat gathers (``margin = r_search + rad``).
    """
    dim = len(sub_shape)
    K = len(atlases_ext)
    M = int(didx.numel())
    ext_shape = [n + 2 * margin for n in sub_shape]
    st = [int(np.prod(ext_shape[d + 1:])) for d in range(dim)]
    st_t = torch.tensor(st, dtype=torch.long, device=dev)
    shift_flat = (torch.tensor(offsets, dtype=torch.long, device=dev) * st_t).sum(1)  # (n_off,)
    deltas = torch.tensor(list(itertools.product(range(-rad, rad + 1), repeat=dim)), dtype=torch.long, device=dev)
    patch_flat = (deltas * st_t).sum(1)  # (P,)
    P = int(deltas.shape[0])
    T_flat = target_ext.reshape(-1)
    A_flat = [a.reshape(-1) for a in atlases_ext]
    O_flat = [o.reshape(-1) for o in off_ids]
    out = torch.empty((M, K), dtype=torch.float32, device=dev)
    chunk = max(1, _GATHER_ELEMS // (K * P))
    for s in range(0, M, chunk):
        dsel = didx[s:s + chunk]
        rem, base = dsel, torch.zeros_like(dsel)
        for d in range(dim - 1, -1, -1):
            base = base + (rem % sub_shape[d] + margin) * st[d]
            rem = rem // sub_shape[d]
        tp = T_flat[base[:, None] + patch_flat[None, :]]  # (c, P)
        ap = torch.stack([A_flat[i][(base + shift_flat[O_flat[i][dsel]])[:, None] + patch_flat[None, :]]
                          for i in range(K)], dim=1)  # (c, K, P)
        if mode == "joint":
            if patch_norm:
                E = _standardise(ap) - _standardise(tp)[:, None, :]
                G, p_eff = torch.bmm(E, E.transpose(1, 2)), None
            else:
                E = (ap - tp[:, None, :]).abs()
                G, p_eff = torch.bmm(E, E.transpose(1, 2)) / float(P), float(P)
            out[s:s + chunk] = _wang_weights(G, beta, rho, solver, nonnegative, stats, p_eff=p_eff)
        else:
            t_sum, t2_sum = tp.sum(-1, keepdim=True), (tp * tp).sum(-1, keepdim=True)
            a_sum, a2_sum, ta_sum = ap.sum(-1), (ap * ap).sum(-1), (ap * tp[:, None, :]).sum(-1)
            t_var = torch.clamp(t2_sum - t_sum * t_sum / P, min=1e-4)
            p_var = torch.clamp(a2_sum - a_sum * a_sum / P, min=1e-4)
            ncc = (ta_sum - a_sum * t_sum / P) / (torch.sqrt(t_var * p_var) + 1e-4)
            out[s:s + chunk] = _patch_weights(torch.clamp(ncc, min=0.0), beta)
    return out


def _remap_labels_compact(labels_seq: Sequence[torch.Tensor], dev: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Map arbitrary non-negative integer labels to compact ``0..C-1`` indices (with ``0`` always at index ``0``).

    Eliminates ``O(M * max_label)`` VRAM blow-up on sparse label tables such as FreeSurfer/DKT
    (``1002..2035`` -> ``63`` classes instead of ``2036``, reducing ``P`` from ~9.8 GiB to ~300 MiB).
    """
    u_labels = torch.unique(torch.cat([l.reshape(-1) for l in labels_seq]), sorted=True)
    if u_labels.numel() > 0 and int(u_labels[0].item()) < 0:
        raise ValueError(f"atlas_labels must be non-negative integers, got min={int(u_labels[0].item())}")
    if u_labels.numel() == 0 or int(u_labels[0].item()) != 0:
        u_labels = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), u_labels])
    max_lbl = int(u_labels[-1].item())
    num_compact = int(u_labels.numel())
    if max_lbl == num_compact - 1:
        return u_labels, list(labels_seq)
    if max_lbl <= 1_000_000:
        lut = torch.zeros(max_lbl + 1, dtype=torch.long, device=dev)
        lut[u_labels] = torch.arange(num_compact, dtype=torch.long, device=dev)
        return u_labels, [lut[l] for l in labels_seq]
    return u_labels, [torch.searchsorted(u_labels, l) for l in labels_seq]


def _compute_weights_on_solve_set(
    target_t: torch.Tensor,
    atlases_t: list[torch.Tensor],
    target_ext: torch.Tensor | None,
    atlases_ext: list[torch.Tensor] | None,
    off_ids: list[torch.Tensor],
    offsets: list[tuple[int, ...]],
    xidx: torch.Tensor,
    sub_shape: tuple[int, ...],
    margin: int,
    rad: int,
    r_search: int,
    mode: str,
    beta: float,
    rho: float,
    solver: str,
    nonnegative: bool,
    patch_norm: bool,
    stats: dict[str, int],
    dev: torch.device,
) -> torch.Tensor:
    """Compute ``(MX, K)`` patch or Wang JLF weights on ``xidx``."""
    K = len(atlases_t)
    dim = len(sub_shape)
    N = int(np.prod(sub_shape))
    MX = int(xidx.numel())
    kernel_vol = float((2 * rad + 1) ** dim)
    flat = lambda x: x[0, 0].reshape(-1)  # noqa: E731

    if r_search > 0:
        return _gather_weights(
            target_ext, atlases_ext, off_ids, offsets, xidx, sub_shape, margin, rad,
            mode, beta, rho, solver, nonnegative, stats, dev, patch_norm,
        )
    if mode == "patch":
        t_sum = _box_sum(target_t, rad)
        t2_sum = _box_sum(target_t * target_t, rad)
        t_var = torch.clamp(t2_sum - t_sum * t_sum / kernel_vol, min=1e-4)
        A_cat = torch.cat(atlases_t, dim=1)  # (1, K, *sub_shape)
        p_sum = _box_sum_chunked(A_cat, rad)
        p2_sum = _box_sum_chunked(A_cat * A_cat, rad)
        tp_sum = _box_sum_chunked(target_t * A_cat, rad)
        del A_cat
        cross = tp_sum - p_sum * t_sum / kernel_vol
        p_var = torch.clamp(p2_sum - p_sum * p_sum / kernel_vol, min=1e-4)
        ncc = torch.clamp(cross / (torch.sqrt(t_var * p_var) + 1e-4), min=0.0)
        ncc_X = ncc.reshape(K, N)[:, xidx].T.contiguous()
        return _patch_weights(ncc_X, beta)
    if patch_norm:
        S_t, S_tt = _box_sum(target_t, rad), _box_sum(target_t * target_t, rad)
        C_tt = S_tt - S_t * S_t / kernel_vol
        s_t = torch.sqrt(torch.clamp_min(C_tt, _VAR_FLOOR))
        A_cat = torch.cat(atlases_t, dim=1)
        S_a = _box_sum_chunked(A_cat, rad)
        C_aa = _box_sum_chunked(A_cat * A_cat, rad) - S_a * S_a / kernel_vol
        s_a = torch.sqrt(torch.clamp_min(C_aa, _VAR_FLOOR))
        r_t_all = ((_box_sum_chunked(A_cat * target_t, rad) - S_a * S_t / kernel_vol) / (s_a * s_t)).reshape(K, N)[:, xidx]
        r_tt = flat(C_tt / (s_t * s_t))[xidx]
        G = torch.empty((MX, K, K), dtype=torch.float32, device=dev)
        pairs = [(i, j) for i in range(K) for j in range(i, K)]
        for ps in range(0, len(pairs), 32):
            pchunk = pairs[ps : ps + 32]
            ii = [p[0] for p in pchunk]
            jj = [p[1] for p in pchunk]
            prod = A_cat[:, ii] * A_cat[:, jj]
            r_ij_c = ((_box_sum(prod, rad) - S_a[:, ii] * S_a[:, jj] / kernel_vol) / (s_a[:, ii] * s_a[:, jj])).reshape(len(pchunk), N)[:, xidx]
            del prod
            for ci, (i, j) in enumerate(pchunk):
                g = r_ij_c[ci] - r_t_all[i] - r_t_all[j] + r_tt
                G[:, i, j] = g
                if i != j:
                    G[:, j, i] = g
        del A_cat
        return _wang_weights(G, beta, rho, solver, nonnegative, stats)

    absd = torch.cat([(A - target_t).abs() for A in atlases_t], dim=1)  # (1, K, *sub_shape)
    G = torch.empty((MX, K, K), dtype=torch.float32, device=dev)
    pairs = [(i, j) for i in range(K) for j in range(i, K)]
    for ps in range(0, len(pairs), 32):
        pchunk = pairs[ps : ps + 32]
        ii = [p[0] for p in pchunk]
        jj = [p[1] for p in pchunk]
        prod = absd[:, ii] * absd[:, jj]
        g_chunk = (_box_sum(prod, rad) / kernel_vol).reshape(len(pchunk), N)[:, xidx]
        del prod
        for ci, (i, j) in enumerate(pchunk):
            G[:, i, j] = g_chunk[ci]
            if i != j:
                G[:, j, i] = g_chunk[ci]
    del absd
    p_eff = flat(_box_sum(torch.ones_like(target_t), rad))[xidx]  # in-image voxels per patch
    return _wang_weights(G, beta, rho, solver, nonnegative, stats, p_eff=p_eff)


def _vote_parcellation(
    labels_ext_p: list[torch.Tensor],
    labels_sel_p: list[torch.Tensor],
    didx: torch.Tensor,
    lmin_orig: torch.Tensor,
    W_full: torch.Tensor | None,
    W_pooled_full: torch.Tensor | None,
    off_ids: list[torch.Tensor],
    offsets: list[tuple[int, ...]],
    sub_shape: tuple[int, ...],
    margin: int,
    rad: int,
    r_search: int,
    return_probabilities: bool,
    shape: tuple[int, ...],
    slicer: tuple[slice, ...],
    target_image: ants.ANTsImage,
    dev: torch.device,
) -> tuple[ants.ANTsImage, ants.ANTsImage, dict[int, ants.ANTsImage]]:
    """Vote one parcellation from shared ``W_full`` (or ``W_pooled_full`` when ``r_search == 0``)."""
    K = len(labels_ext_p)
    dim = len(sub_shape)
    N = int(np.prod(sub_shape))
    M = int(didx.numel())

    u_labels, labels_ext_dense = _remap_labels_compact(labels_ext_p, dev)
    num_classes = int(u_labels.numel())
    seg_flat = lmin_orig.clone()
    P = None

    if M > 0:
        P = torch.zeros((M, num_classes), dtype=torch.float32, device=dev)
        rows = torch.arange(M, device=dev)
        if r_search == 0:
            # Exact identity at zero search offset: L_i(n) is independent of patch centre x = n + delta,
            # so sum_{delta in [-rad, rad]^d} W_i(n + delta) * 1[L_i(n)=c] = (_box_sum(W_i, rad)[n]) * 1[L_i(n)=c].
            # Row-indexed accumulation over K atlases has strictly unique rows per step -> 100% deterministic.
            assert W_pooled_full is not None
            for i in range(K):
                lab_i_d = labels_ext_dense[i].reshape(-1)[didx]
                P[rows, lab_i_d] += W_pooled_full[i, didx]
        else:
            assert W_full is not None
            ext_shape = [n + 2 * margin for n in sub_shape]
            st = [int(np.prod(ext_shape[d + 1:])) for d in range(dim)]
            rem, base = didx, torch.zeros_like(didx)
            for d in range(dim - 1, -1, -1):
                base = base + (rem % sub_shape[d] + margin) * st[d]
                rem = rem // sub_shape[d]
            shift_flat = [int(sum(o * s for o, s in zip(off, st))) for off in offsets]
            sub_st = [int(np.prod(sub_shape[d + 1:])) for d in range(dim)]
            coords, rem = [], didx
            for d in range(dim - 1, -1, -1):
                coords.insert(0, rem % sub_shape[d])
                rem = rem // sub_shape[d]
            shift_t = torch.tensor(shift_flat, dtype=torch.long, device=dev)
            off_flat = [o[0, 0].reshape(-1) for o in off_ids]
            lab_flat = [l[0, 0].reshape(-1) for l in labels_ext_dense]
            for delta in itertools.product(range(-rad, rad + 1), repeat=dim):
                xf = torch.zeros_like(didx)
                valid = torch.ones_like(didx, dtype=torch.bool)
                for d in range(dim):
                    xc = coords[d] + delta[d]
                    valid &= (xc >= 0) & (xc < sub_shape[d])
                    xf = xf + xc.clamp(0, sub_shape[d] - 1) * sub_st[d]
                for i in range(K):
                    w = W_full[i][xf] * valid
                    P[rows, lab_flat[i][base + shift_t[off_flat[i][xf]]]] += w

        P = P / torch.clamp_min(P.sum(dim=1, keepdim=True), 1e-12)
        # Ties go to the smallest compact class index, which equals the smallest original label since u_labels is sorted.
        top = P.max(dim=1, keepdim=True).values
        cls = torch.arange(num_classes, device=dev).unsqueeze(0)
        best_dense = torch.where(P >= top - _TIE_TOL, cls, num_classes).amin(dim=1)
        seg_flat[didx] = u_labels[best_dense]

    env_flat = (lmin_orig != 0)
    if M > 0:
        # u_labels[0] is guaranteed to be 0 (background).
        env_flat[didx] = (1.0 - P[:, 0]) > 0.05

    final_crop = seg_flat.reshape(sub_shape).cpu().numpy().astype(np.uint32)
    env_crop = env_flat.reshape(sub_shape).cpu().numpy().astype(np.uint32)
    u_labels_cpu = u_labels.cpu().numpy().tolist()

    prob_crop_np = None
    if return_probabilities:
        lmin_dense = torch.searchsorted(u_labels, lmin_orig)
        prob = torch.zeros((num_classes, N), dtype=torch.float32, device=dev)
        prob.scatter_(0, lmin_dense.unsqueeze(0), 1.0)
        if M > 0:
            prob[:, didx] = P.T
        prob_crop_np = prob.reshape(num_classes, *sub_shape).cpu().numpy()
        del prob

    final_arr = np.zeros(shape, dtype=np.uint32)
    final_arr[slicer] = final_crop
    env_arr = np.zeros(shape, dtype=np.uint32)
    env_arr[slicer] = env_crop

    seg_img = ants.from_numpy(
        final_arr,
        origin=target_image.origin,
        spacing=target_image.spacing,
        direction=target_image.direction,
    )
    env_img = ants.from_numpy(
        env_arr,
        origin=target_image.origin,
        spacing=target_image.spacing,
        direction=target_image.direction,
    )

    prob_images: dict[int, ants.ANTsImage] = {}
    if return_probabilities and prob_crop_np is not None:
        for c, orig_lbl in enumerate(u_labels_cpu):
            lbl_int = int(orig_lbl)
            p_full = np.zeros(shape, dtype=np.float32) if lbl_int != 0 else np.ones(shape, dtype=np.float32)
            p_full[slicer] = prob_crop_np[c]
            prob_images[lbl_int] = ants.from_numpy(
                p_full,
                origin=target_image.origin,
                spacing=target_image.spacing,
                direction=target_image.direction,
            )

    return seg_img, env_img, prob_images


def joint_label_fusion_multi(
    target_image: ants.ANTsImage,
    atlas_images: Sequence[ants.ANTsImage],
    atlas_labels_dict: Mapping[str, Sequence[ants.ANTsImage]],
    mask: ants.ANTsImage | None = None,
    rad: int = 2,
    beta: float | None = None,
    rho: float | None = None,
    mode: Literal["patch", "joint"] = "joint",
    nonnegative: bool = True,
    device: str | None = None,
    verbose: bool = False,
    r_search: int = 0,
    skip_consensus: bool = True,
    solver: Literal["cholesky", "lu"] = "cholesky",
    return_probabilities: bool = False,
    patch_norm: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fuse multiple parcellations simultaneously on one ``(target_image, atlas_images)`` cohort.

    Intensity patch search and Wang/LNCC solve-set weights depend purely on ``target_image`` and
    ``atlas_images`` (not on the label values). Computing ``w_X`` once on the union dilated
    disagreement mask across all parcellations in ``atlas_labels_dict`` yields bitwise-identical
    segmentations to calling :func:`joint_label_fusion` separately per parcellation while
    eliminating ``(P - 1)x`` redundant image uploads, patch searches, and Cholesky solves.
    """
    t0_start = time.perf_counter()
    if isinstance(r_search, bool) or not isinstance(r_search, (int, np.integer)) or r_search < 0:
        raise ValueError(f"r_search must be a non-negative integer, got {r_search!r}")
    r_search = int(r_search)
    if mode not in ("patch", "joint"):
        raise ValueError(f"mode must be 'patch' or 'joint', got {mode!r}")
    if solver not in ("cholesky", "lu"):
        raise ValueError(f"solver must be 'cholesky' or 'lu', got {solver!r}")
    if not atlas_labels_dict:
        raise ValueError("atlas_labels_dict must be non-empty")
    beta = (4.0 if mode == "joint" else 2.0) if beta is None else float(beta)
    rho = 0.01 if rho is None else float(rho)
    K = len(atlas_images)
    if K == 0:
        raise ValueError("atlas_images must be non-empty")
    for p_name, lbl_seq in atlas_labels_dict.items():
        if len(lbl_seq) != K:
            raise ValueError(
                f"atlas_images ({K}) and atlas_labels_dict[{p_name!r}] ({len(lbl_seq)}) must have equal non-zero length"
            )

    target_arr = target_image.numpy()
    shape = target_arr.shape

    if mask is not None:
        roi_mask = mask.numpy() > 0.5
    else:
        roi_mask = np.zeros(shape, dtype=bool)
        for lbl_seq in atlas_labels_dict.values():
            for lbl in lbl_seq:
                roi_mask |= (lbl.numpy() > 0.5)

    dev_name = auto_detect_device(requested_device=device)
    info_common = {"mode": mode, "r_search": r_search, "device": dev_name, "solver": solver, "beta": beta, "rho": rho}
    if not roi_mask.any():
        out_empty: dict[str, dict[str, Any]] = {}
        for p_name in atlas_labels_dict:
            zero_img = target_image.new_image_like(np.zeros(shape, dtype=np.uint32))
            probs = (
                {0: target_image.new_image_like(np.ones(shape, dtype=np.float32))}
                if return_probabilities
                else {}
            )
            out_empty[p_name] = {
                "segmentation": zero_img,
                "probability_images": probs,
                "consensus_envelope": zero_img.clone(),
                "timing_s": 0.0,
                "stage_timings_s": {"search": 0.0, "weights": 0.0, "vote": 0.0},
                "frac_shifted": None,
                "disagreement_fraction": 0.0,
                "cholesky_failed": 0,
                **info_common,
            }
        return out_empty

    pad = rad + 2
    idx = np.where(roi_mask)
    dim = len(shape)
    slicer = tuple(
        slice(max(0, int(idx[d].min()) - pad), min(shape[d], int(idx[d].max()) + pad + 1))
        for d in range(dim)
    )

    margin = (r_search + rad) if r_search > 0 else 0
    if margin == 0:
        def _crop(arr):
            return arr[slicer]
    else:
        ext_idx = [
            np.clip(np.arange(slicer[d].start - margin, slicer[d].stop + margin), 0, shape[d] - 1)
            for d in range(dim)
        ]

        def _crop(arr):
            return arr[np.ix_(*ext_idx)]

    target_crop = target_arr[slicer]
    sub_shape = target_crop.shape
    N = int(np.prod(sub_shape))

    dev = torch.device(dev_name)
    to_f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=dev)  # noqa: E731
    target_t = to_f(target_crop)
    atlases_t = [to_f(_crop(img.numpy())) for img in atlas_images]
    labels_ext_by_p: dict[str, list[torch.Tensor]] = {
        p_name: [
            torch.from_numpy(np.ascontiguousarray(_crop(lbl.numpy()))).unsqueeze(0).unsqueeze(0).to(dtype=torch.long, device=dev)
            for lbl in lbl_seq
        ]
        for p_name, lbl_seq in atlas_labels_dict.items()
    }

    _sync(dev)
    t_a = time.perf_counter()
    frac_shifted = None
    target_ext = None
    atlases_ext = None
    if r_search > 0:
        offsets = _search_offsets(r_search, dim)
        atlases_ext, target_ext = atlases_t, to_f(_crop(target_arr))
        atlases_t, off_ids, frac_shifted = _search_best_patch_offsets(
            target_t, atlases_ext, offsets, margin, rad, sub_shape, roi_mask[slicer], dev
        )
        labels_sel_by_p = {
            p_name: _gather_selected_labels(l_ext, off_ids, offsets, sub_shape, margin, dev)
            for p_name, l_ext in labels_ext_by_p.items()
        }
    else:
        offsets = [(0,) * dim]
        off_ids = [torch.zeros((1, 1, *sub_shape), dtype=torch.long, device=dev) for _ in range(K)]
        labels_sel_by_p = labels_ext_by_p
    _sync(dev)
    t_search = time.perf_counter() - t_a

    t_a = time.perf_counter()
    didx_by_p: dict[str, torch.Tensor] = {}
    lmin_by_p: dict[str, torch.Tensor] = {}
    dis_frac_by_p: dict[str, float] = {}
    dis_union = torch.zeros(N, dtype=torch.bool, device=dev)

    for p_name, l_sel in labels_sel_by_p.items():
        lab_stack = torch.cat([l[0, 0].reshape(1, -1) for l in l_sel], dim=0)
        lmin = lab_stack.amin(dim=0)
        if skip_consensus:
            dis_p = lmin != lab_stack.amax(dim=0)
        else:
            dis_p = torch.ones_like(lmin, dtype=torch.bool)
        didx_p = dis_p.nonzero().squeeze(1)
        didx_by_p[p_name] = didx_p
        lmin_by_p[p_name] = lmin
        dis_frac_by_p[p_name] = float(didx_p.numel()) / float(N)
        dis_union |= dis_p

    M_union = int(dis_union.sum().item())
    if M_union > 0:
        xmask = _box_sum(dis_union.reshape(1, 1, *sub_shape).to(torch.float32), rad) > 0.5
        xidx = xmask.reshape(-1).nonzero().squeeze(1)
    else:
        xidx = torch.empty(0, dtype=torch.long, device=dev)

    stats = {"cholesky_failed": 0}
    w_X = None
    if M_union > 0:
        w_X = _compute_weights_on_solve_set(
            target_t=target_t,
            atlases_t=atlases_t,
            target_ext=target_ext,
            atlases_ext=atlases_ext,
            off_ids=off_ids,
            offsets=offsets,
            xidx=xidx,
            sub_shape=sub_shape,
            margin=margin,
            rad=rad,
            r_search=r_search,
            mode=mode,
            beta=beta,
            rho=rho,
            solver=solver,
            nonnegative=nonnegative,
            patch_norm=patch_norm,
            stats=stats,
            dev=dev,
        )
    del target_ext, atlases_ext, target_t, atlases_t
    _sync(dev)
    t_weights = time.perf_counter() - t_a

    # Shared weight field (and pre-pooled box-summed weights when r_search == 0) across all parcellations
    W_full = None
    W_pooled_full = None
    t_pool_start = time.perf_counter()
    if M_union > 0 and w_X is not None:
        W_full = torch.zeros((K, N), dtype=torch.float32, device=dev)
        W_full[:, xidx] = w_X.T
        del w_X
        if r_search == 0:
            W_pooled_full = _box_sum_chunked(W_full.reshape(1, K, *sub_shape), rad).reshape(K, N)
            del W_full
            W_full = None
    _sync(dev)
    t_pool_shared = time.perf_counter() - t_pool_start

    results: dict[str, dict[str, Any]] = {}
    n_parc = max(len(atlas_labels_dict), 1)
    for p_name in atlas_labels_dict:
        t_v0 = time.perf_counter()
        seg_img, env_img, prob_images = _vote_parcellation(
            labels_ext_p=labels_ext_by_p[p_name],
            labels_sel_p=labels_sel_by_p[p_name],
            didx=didx_by_p[p_name],
            lmin_orig=lmin_by_p[p_name],
            W_full=W_full,
            W_pooled_full=W_pooled_full,
            off_ids=off_ids,
            offsets=offsets,
            sub_shape=sub_shape,
            margin=margin,
            rad=rad,
            r_search=r_search,
            return_probabilities=return_probabilities,
            shape=shape,
            slicer=slicer,
            target_image=target_image,
            dev=dev,
        )
        _sync(dev)
        t_vote_p = (time.perf_counter() - t_v0) + (t_pool_shared / n_parc)
        results[p_name] = {
            "segmentation": seg_img,
            "probability_images": prob_images,
            "consensus_envelope": env_img,
            "timing_s": 0.0,  # filled after loop
            "stage_timings_s": {
                "search": t_search / n_parc if len(atlas_labels_dict) > 1 else t_search,
                "weights": t_weights / n_parc if len(atlas_labels_dict) > 1 else t_weights,
                "vote": t_vote_p,
            },
            "frac_shifted": frac_shifted,
            "disagreement_fraction": dis_frac_by_p[p_name],
            "cholesky_failed": stats["cholesky_failed"],
            **info_common,
        }

    del W_full, W_pooled_full
    t_total = time.perf_counter() - t0_start
    for p_name in results:
        results[p_name]["timing_s"] = float(t_total)
    if verbose:
        print(
            f"[syntx.joint_label_fusion] Mode={mode}, K={K}, P={len(atlas_labels_dict)}, "
            f"ROI={sub_shape}, device={dev_name}, Time={t_total:.2f}s"
        )
    return results


def joint_label_fusion(
    target_image: ants.ANTsImage,
    atlas_images: Sequence[ants.ANTsImage],
    atlas_labels: Sequence[ants.ANTsImage],
    mask: ants.ANTsImage | None = None,
    rad: int = 2,
    beta: float | None = None,
    rho: float | None = None,
    mode: Literal["patch", "joint"] = "joint",
    nonnegative: bool = True,
    device: str | None = None,
    verbose: bool = False,
    r_search: int = 0,
    skip_consensus: bool = True,
    solver: Literal["cholesky", "lu"] = "cholesky",
    return_probabilities: bool = True,
    patch_norm: bool = False,
) -> dict[str, Any]:
    """Execute high-performance joint label fusion across warped atlas candidates.

    Parameters
    ----------
    target_image : ants.ANTsImage
        Target reference structural image in native or registered space.
    atlas_images : sequence of ants.ANTsImage
        Warped atlas intensity images aligned to `target_image`.
    atlas_labels : sequence of ants.ANTsImage
        Warped atlas candidate label maps aligned to `target_image` (integer labels, 0 =
        background, which is a class like any other). Inputs are never modified.
    mask : ants.ANTsImage, optional
        ROI mask defining the search space. If None, automatically computes the
        union bounding box of all atlas labels with padding. Labels are computed inside the
        bounding box of the mask; clip the result to the mask if exact masking is needed.
    rad : int, default 2
        Local patch radius (box window width = 2 * rad + 1).
    beta : float, optional
        Sharpening exponent. Default 4.0 for ``mode="joint"`` (the Wang et al. / ANTs value) and
        2.0 for ``mode="patch"``.
    rho : float, optional
        Ridge added to the diagonal of the mean-diagonal-scaled dependence matrix (``joint`` only).
        Default 0.01.
    mode : {"patch", "joint"}, default "joint"
        - "patch": diagonal LNCC patch-similarity weighting.
        - "joint": Wang et al. patch-based joint label fusion (see module docstring).
    nonnegative : bool, default True
        Whether to enforce non-negative weights (w >= 0) for ``mode="joint"``.
    device : str, optional
        Compute device ('cpu', 'mps', 'cuda'). Defaults to auto-detection (cuda > mps > cpu).
    verbose : bool, default False
        Whether to log timing and progress.
    r_search : int, default 0
        Search radius in voxels (ANTs ``-s``). With ``r_search > 0``, for each atlas and voxel the
        offset in ``[-r_search, r_search]`` per axis with the minimum patch sum of squared
        differences is selected (ties prefer the smallest displacement) and the whole shifted atlas
        patch enters the weights. Cost grows as ``K * (2*r_search+1)**ndim`` box sums.
    skip_consensus : bool, default True
        Form and solve weights only where needed (see Notes). ``False`` evaluates every voxel (the
        dense reference path, used by the tests and benchmarks).
    solver : {"cholesky", "lu"}, default "cholesky"
        Batched linear solver for ``mode="joint"``.
    return_probabilities : bool, default True
        Build the full per-class probability images (keyed by each candidate class ID present in
        ``atlas_labels`` plus background ``0``). ``False`` skips them and returns only the
        segmentation and envelope.
    patch_norm : bool, default False
        ``mode="joint"`` only: standardise each patch (zero mean, unit norm) before forming the
        residuals.

    Returns
    -------
    dict[str, Any]
        - 'segmentation': ants.ANTsImage of discrete consensus integer labels.
        - 'probability_images': dict[int, ants.ANTsImage] per-class probability maps (empty when
          ``return_probabilities=False``).
        - 'consensus_envelope': ants.ANTsImage union binary envelope of candidate labels.
        - 'timing_s': total wall-clock seconds; 'stage_timings_s': dict search / weights / vote.
        - 'mode', 'r_search', 'device', 'solver', 'beta', 'rho'.
        - 'frac_shifted': per-atlas fraction of ROI voxels with a non-zero best offset (None when
          ``r_search == 0``).
        - 'disagreement_fraction': fraction of the cropped ROI where atlases disagree.
        - 'cholesky_failed': voxels re-solved by LU after a Cholesky failure.
    """
    K = len(atlas_images)
    if K == 0 or len(atlas_labels) != K:
        raise ValueError(
            f"atlas_images ({len(atlas_images)}) and atlas_labels ({len(atlas_labels)}) must have equal non-zero length"
        )
    return joint_label_fusion_multi(
        target_image=target_image,
        atlas_images=atlas_images,
        atlas_labels_dict={"_single": atlas_labels},
        mask=mask,
        rad=rad,
        beta=beta,
        rho=rho,
        mode=mode,
        nonnegative=nonnegative,
        device=device,
        verbose=verbose,
        r_search=r_search,
        skip_consensus=skip_consensus,
        solver=solver,
        return_probabilities=return_probabilities,
        patch_norm=patch_norm,
    )["_single"]
