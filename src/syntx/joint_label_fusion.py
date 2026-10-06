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
from typing import Any, Literal, Sequence

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


def _search_best_patch(target_t, atlases_ext, labels_ext, offsets, margin, rad, sub_shape, roi, dev):
    """Per atlas and voxel, pick the offset with the minimum patch SSD and return its intensity/label.

    ``atlases_ext`` / ``labels_ext`` are ``(1, 1, *(sub_shape + 2*margin))`` tensors; ``target_t``
    is ``(1, 1, *sub_shape)``. A candidate replaces the incumbent only on strictly smaller SSD and
    offsets are visited in the order of ``offsets`` (zero first), so ties keep the smallest
    displacement and the result is deterministic. Returns, per atlas, the selected intensity,
    label and offset index (into ``offsets``), plus the fraction of ``roi`` voxels moved.
    """
    roi_t = torch.from_numpy(np.ascontiguousarray(roi)).to(dev)
    n_roi = max(int(roi_t.sum().item()), 1)
    sel_a, sel_l, sel_o, fracs = [], [], [], []
    for A, L in zip(atlases_ext, labels_ext):
        best = None
        for oi, off in enumerate(offsets):
            sl = (slice(None), slice(None)) + tuple(
                slice(margin + o, margin + o + n) for o, n in zip(off, sub_shape)
            )
            a = A[sl].contiguous()
            ssd = _box_sum((a - target_t) ** 2, rad)  # patch SSD (sum; the 1/|patch| factor is common)
            if best is None:
                best, a_sel, l_sel = ssd, a, L[sl].contiguous()
                o_sel = torch.zeros_like(ssd, dtype=torch.long)
                continue
            upd = ssd < best
            best = torch.where(upd, ssd, best)
            a_sel = torch.where(upd, a, a_sel)
            l_sel = torch.where(upd, L[sl].contiguous(), l_sel)
            o_sel = torch.where(upd, torch.full_like(o_sel, oi), o_sel)
        sel_a.append(a_sel)
        sel_l.append(l_sel)
        sel_o.append(o_sel)
        fracs.append(float(((o_sel[0, 0] != 0) & roi_t).sum().item()) / n_roi)
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
        Build the full per-class probability images. ``False`` skips them (large for many labels)
        and returns only the segmentation and envelope.
    patch_norm : bool, default False
        ``mode="joint"`` only: standardise each patch (zero mean, unit norm) before forming the
        residuals. This removes the arbitrary-weights problem in featureless regions by construction
        but discards gain/offset information; on the r-image validation it cost ~0.06 mean Dice, so
        the default is raw ``|A - T|`` residuals with empirical-Bayes shrinkage.

    Notes
    -----
    Voting follows ANTs (Hongzhi's averaging scheme): every voxel pools the weights of all patches that
    contain it, ``P_l(n) = sum_x sum_i W_i(x) 1[L_i(n + s_i(x)) = l]``, so label decisions are smoothed
    over the patch. With ``skip_consensus`` the weights are solved only on the voxels whose own selected
    votes disagree (D) dilated by ``rad``; the rest take their unanimous label. Without a search
    (``r_search=0``) this equals the dense result exactly; with a search, a voxel whose own selected votes
    are unanimous keeps that label even if neighbours' offsets would have voted differently there (an
    approximation; ``skip_consensus=False`` is the exact dense reference).

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
    t0_start = time.perf_counter()
    if isinstance(r_search, bool) or not isinstance(r_search, (int, np.integer)) or r_search < 0:
        raise ValueError(f"r_search must be a non-negative integer, got {r_search!r}")
    r_search = int(r_search)
    if mode not in ("patch", "joint"):
        raise ValueError(f"mode must be 'patch' or 'joint', got {mode!r}")
    if solver not in ("cholesky", "lu"):
        raise ValueError(f"solver must be 'cholesky' or 'lu', got {solver!r}")
    beta = (4.0 if mode == "joint" else 2.0) if beta is None else float(beta)
    rho = 0.01 if rho is None else float(rho)
    K = len(atlas_images)
    if K == 0 or len(atlas_labels) != K:
        raise ValueError(f"atlas_images ({len(atlas_images)}) and atlas_labels ({len(atlas_labels)}) must have equal non-zero length")

    target_arr = target_image.numpy()
    shape = target_arr.shape

    # 1. Determine bounding box for localized execution
    if mask is not None:
        roi_mask = mask.numpy() > 0.5
    else:
        # Compute union of all non-zero atlas labels
        roi_mask = np.zeros(shape, dtype=bool)
        for lbl in atlas_labels:
            roi_mask |= (lbl.numpy() > 0.5)

    dev_name = auto_detect_device(requested_device=device)
    info_common = {"mode": mode, "r_search": r_search, "device": dev_name, "solver": solver, "beta": beta, "rho": rho}
    if not roi_mask.any():
        # Fallback if no labels found
        zero_img = target_image.new_image_like(np.zeros(shape, dtype=np.uint32))
        return {
            "segmentation": zero_img,
            "probability_images": {0: target_image.new_image_like(np.ones(shape, dtype=np.float32))},
            "consensus_envelope": zero_img,
            "timing_s": 0.0,
            "stage_timings_s": {"search": 0.0, "weights": 0.0, "vote": 0.0},
            "frac_shifted": None,
            "disagreement_fraction": 0.0,
            "cholesky_failed": 0,
            **info_common,
        }

    # Bounding box coordinates with padding = rad + 2
    pad = rad + 2
    idx = np.where(roi_mask)
    dim = len(shape)
    slicer = tuple(
        slice(max(0, int(idx[d].min()) - pad), min(shape[d], int(idx[d].max()) + pad + 1))
        for d in range(dim)
    )

    # Crop target and atlas subvolumes. With a search radius the crop is extended by
    # ``margin = r_search + rad`` voxels per side (edge-clamped outside the image) so every shifted
    # patch has data.
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

    # 2. Convert to PyTorch tensors
    dev = torch.device(dev_name)
    to_f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=dev)  # noqa: E731
    target_t = to_f(target_crop)
    atlases_t = [to_f(_crop(img.numpy())) for img in atlas_images]
    labels_t = [torch.from_numpy(np.ascontiguousarray(_crop(lbl.numpy()))).unsqueeze(0).unsqueeze(0).to(dtype=torch.long, device=dev) for lbl in atlas_labels]
    kernel_vol = float((2 * rad + 1) ** dim)  # patch size, normalises box sums to patch means

    # 2b. Per-voxel best-patch search (ANTs ``-s``). Without a search the offsets are the single zero offset.
    _sync(dev)
    t_a = time.perf_counter()
    frac_shifted = None
    labels_ext = labels_t  # (1, 1, *(sub_shape + 2*margin)) tensors; margin == 0 without a search
    if r_search > 0:
        offsets = _search_offsets(r_search, dim)
        atlases_ext, target_ext = atlases_t, to_f(_crop(target_arr))
        atlases_t, labels_t, off_ids, frac_shifted = _search_best_patch(
            target_t, atlases_t, labels_t, offsets, margin, rad, sub_shape, roi_mask[slicer], dev
        )
    else:
        offsets = [(0,) * dim]
        off_ids = [torch.zeros_like(l, dtype=torch.long) for l in labels_t]
    _sync(dev)
    t_search = time.perf_counter() - t_a

    # 3. Disagreement set D: voxels whose own selected votes differ, so their label can depend on the
    # weights. Solve set X = D dilated by the patch radius: the voxels whose weights reach a D voxel
    # through the overlapping-patch average (every voxel pools the weights of all patches containing it).
    t_a = time.perf_counter()
    lab_stack = torch.cat([l[0, 0].reshape(1, -1) for l in labels_t], dim=0)  # (K, N), centre-selected labels
    lmin = lab_stack.amin(dim=0)
    if skip_consensus:
        dis = lmin != lab_stack.amax(dim=0)
    else:
        dis = torch.ones_like(lmin, dtype=torch.bool)
    didx = dis.nonzero().squeeze(1)  # (M,)
    M = int(didx.numel())
    disagreement_fraction = float(M) / float(N)
    if M > 0:
        xmask = _box_sum(dis.reshape(1, 1, *sub_shape).to(torch.float32), rad) > 0.5
        xidx = xmask.reshape(-1).nonzero().squeeze(1)
    else:
        xidx = didx
    MX = int(xidx.numel())
    stats = {"cholesky_failed": 0}

    # 4. Weights on the solve set, (MX, K)
    w_X = None
    if M > 0:
        flat = lambda x: x[0, 0].reshape(-1)  # noqa: E731
        if r_search > 0:
            w_X = _gather_weights(target_ext, atlases_ext, off_ids, offsets, xidx, sub_shape, margin, rad,
                                  mode, beta, rho, solver, nonnegative, stats, dev, patch_norm)
            del target_ext
        elif mode == "patch":
            t_sum = _box_sum(target_t, rad)
            t2_sum = _box_sum(target_t * target_t, rad)
            t_var = torch.clamp(t2_sum - t_sum * t_sum / kernel_vol, min=1e-4)
            ncc_X = []
            for A in atlases_t:
                p_sum = _box_sum(A, rad)
                p2_sum = _box_sum(A * A, rad)
                tp_sum = _box_sum(target_t * A, rad)
                cross = tp_sum - p_sum * t_sum / kernel_vol
                p_var = torch.clamp(p2_sum - p_sum * p_sum / kernel_vol, min=1e-4)
                ncc = cross / (torch.sqrt(t_var * p_var) + 1e-4)
                ncc_X.append(flat(torch.clamp(ncc, min=0.0))[xidx])
            w_X = _patch_weights(torch.stack(ncc_X, dim=1), beta)
        elif patch_norm:
            # G_ij = r_ij - r_it - r_jt + r_tt: patch correlations of the standardised patches, i.e.
            # mean-patch residual covariance of z_i - z_t, from box sums of first/second moments.
            S_t, S_tt = _box_sum(target_t, rad), _box_sum(target_t * target_t, rad)
            C_tt = S_tt - S_t * S_t / kernel_vol
            s_t = torch.sqrt(torch.clamp_min(C_tt, _VAR_FLOOR))
            S_a = [_box_sum(A, rad) for A in atlases_t]
            s_a, r_t = [], []
            for A, Sa in zip(atlases_t, S_a):
                C_aa = _box_sum(A * A, rad) - Sa * Sa / kernel_vol
                s_a.append(torch.sqrt(torch.clamp_min(C_aa, _VAR_FLOOR)))
                r_t.append(flat((_box_sum(A * target_t, rad) - Sa * S_t / kernel_vol) / (s_a[-1] * s_t))[xidx])
            r_tt = flat(C_tt / (s_t * s_t))[xidx]
            G = torch.empty((MX, K, K), dtype=torch.float32, device=dev)
            for i in range(K):
                for j in range(i, K):
                    if i == j:
                        r_ij = flat((_box_sum(atlases_t[i] ** 2, rad) - S_a[i] ** 2 / kernel_vol) / (s_a[i] ** 2))[xidx]
                    else:
                        r_ij = flat((_box_sum(atlases_t[i] * atlases_t[j], rad) - S_a[i] * S_a[j] / kernel_vol)
                                    / (s_a[i] * s_a[j]))[xidx]
                    g = r_ij - r_t[i] - r_t[j] + r_tt
                    G[:, i, j] = g
                    if i != j:
                        G[:, j, i] = g
            w_X = _wang_weights(G, beta, rho, solver, nonnegative, stats)
        else:
            absd = [(A - target_t).abs() for A in atlases_t]
            G = torch.empty((MX, K, K), dtype=torch.float32, device=dev)
            for i in range(K):
                for j in range(i, K):
                    g_ij = flat(_box_sum(absd[i] * absd[j], rad) / kernel_vol)[xidx]
                    G[:, i, j] = g_ij
                    if i != j:
                        G[:, j, i] = g_ij
            del absd
            p_eff = flat(_box_sum(torch.ones_like(target_t), rad))[xidx]  # in-image voxels per patch
            w_X = _wang_weights(G, beta, rho, solver, nonnegative, stats, p_eff=p_eff)
    if r_search > 0:
        del atlases_ext
    _sync(dev)
    t_weights = time.perf_counter() - t_a

    # 5. Voting with the overlapping-patch average (ANTs, Hongzhi's scheme): voxel n collects from every
    # patch x that contains it the weight W_i(x) on the label of atlas i at n + s_i(x). Per (atlas,
    # offset) that is a box sum of the weight field, so it is exact and deterministic. Unanimous voxels
    # take their shared label (probability 1).
    t_a = time.perf_counter()
    num_classes = int(max(int(l.max().item()) for l in labels_ext)) + 1
    seg_flat = lmin.clone()
    P = None
    if M > 0:
        W_full = torch.zeros((K, N), dtype=torch.float32, device=dev)
        W_full[:, xidx] = w_X.T
        del w_X
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
        lab_flat = [l[0, 0].reshape(-1) for l in labels_ext]
        P = torch.zeros((M, num_classes), dtype=torch.float32, device=dev)
        rows = torch.arange(M, device=dev)
        for delta in itertools.product(range(-rad, rad + 1), repeat=dim):  # patches x = n + delta that contain n
            xf = torch.zeros_like(didx)
            valid = torch.ones_like(didx, dtype=torch.bool)
            for d in range(dim):
                xc = coords[d] + delta[d]
                valid &= (xc >= 0) & (xc < sub_shape[d])
                xf = xf + xc.clamp(0, sub_shape[d] - 1) * sub_st[d]
            for i in range(K):
                w = W_full[i][xf] * valid
                P[rows, lab_flat[i][base + shift_t[off_flat[i][xf]]]] += w  # unique rows per call -> deterministic
        del W_full
        P = P / torch.clamp_min(P.sum(dim=1, keepdim=True), 1e-12)
        # ties (e.g. 2-vs-2 votes under equal weights) go to the smallest label: float round-off must not decide
        top = P.max(dim=1, keepdim=True).values
        cls = torch.arange(num_classes, device=dev).unsqueeze(0)
        seg_flat[didx] = torch.where(P >= top - _TIE_TOL, cls, num_classes).amin(dim=1)
    env_flat = (lmin != 0)
    if M > 0:
        env_flat[didx] = (1.0 - P[:, 0]) > 0.05

    final_crop = seg_flat.reshape(sub_shape).cpu().numpy().astype(np.uint32)
    env_crop = env_flat.reshape(sub_shape).cpu().numpy().astype(np.uint32)
    prob_crop_np = None
    if return_probabilities:
        prob = torch.zeros((num_classes, N), dtype=torch.float32, device=dev)
        prob.scatter_(0, lmin.unsqueeze(0), 1.0)
        if M > 0:
            prob[:, didx] = P.T
        prob_crop_np = prob.reshape(num_classes, *sub_shape).cpu().numpy()
        del prob
    _sync(dev)
    t_vote = time.perf_counter() - t_a

    # 6. Restore into full native image space
    final_arr = np.zeros(shape, dtype=np.uint32)
    final_arr[slicer] = final_crop
    env_arr = np.zeros(shape, dtype=np.uint32)
    env_arr[slicer] = env_crop

    # Create ANTsImage objects matching target geometry
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

    prob_images = {}
    if return_probabilities:
        for c in range(num_classes):
            p_full = np.zeros(shape, dtype=np.float32)
            p_full[slicer] = prob_crop_np[c]
            prob_images[c] = ants.from_numpy(
                p_full,
                origin=target_image.origin,
                spacing=target_image.spacing,
                direction=target_image.direction,
            )

    t_total = time.perf_counter() - t0_start
    if verbose:
        print(f"[syntx.joint_label_fusion] Mode={mode}, K={K}, ROI={sub_shape}, device={dev_name}, "
              f"disagree={disagreement_fraction:.3f}, Time={t_total:.2f}s")

    return {
        "segmentation": seg_img,
        "probability_images": prob_images,
        "consensus_envelope": env_img,
        "timing_s": float(t_total),
        "stage_timings_s": {"search": t_search, "weights": t_weights, "vote": t_vote},
        "frac_shifted": frac_shifted,
        "disagreement_fraction": disagreement_fraction,
        "cholesky_failed": stats["cholesky_failed"],
        **info_common,
    }
