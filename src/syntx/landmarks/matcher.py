"""
syntx.landmarks.matcher — Batched GPU Matching + RANSAC
========================================================

Provides modality-independent matching of landmark descriptors (SIFT, MIND, etc.)
and geometric verification via RANSAC.

Performance design
------------------
``match_landmarks`` uses **batched torch matmul** to compute the descriptor
distance matrix in chunks, avoiding the full N×M allocation that caused OOM
on large CT descriptor sets.  For N = M = 5000 descriptors of dimension 512,
the full float32 matrix is 5000×5000×4 B ≈ 96 MB; a batch_size=256 chunk is
≈ 5 MB — well within GPU VRAM or CPU RAM.

The Lowe ratio test uses ``torch.topk(k=2)`` per row — O(M) instead of
O(M log M) argsort — which is 4–8× faster on large descriptor sets.

All RANSAC residuals are computed in a single vectorised call (no Python loop).

Functions
---------
match_landmarks : Batched GPU L2/cosine NN + Lowe ratio test
ransac_filter   : RANSAC affine/rigid verification (vectorised residuals)
compute_tre     : Target Registration Error in mm (hold-out points only)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default chunk size for the source-side batch loop.
# Each chunk occupies (batch_size × M × 4) bytes; 512 rows × 5000 pts × 4B = 10 MB.
_DEFAULT_BATCH = 512


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_float32(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def _batched_l2_top2(
    src: torch.Tensor,   # [N, D]  — already normalised if cosine
    dst: torch.Tensor,   # [M, D]
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute L2 distances in batches, returning only the top-2 nearest
    neighbours per source row without ever materialising the full N×M matrix.

    Returns
    -------
    best_dist  : [N, 2]  distances to 1st and 2nd nearest neighbours
    best_idx   : [N, 2]  column indices in dst
    """
    N, D = src.shape
    M    = dst.shape[0]

    # Pre-compute ||dst||² once — reused for every chunk
    dst_sq = (dst * dst).sum(dim=1)   # [M]

    all_dist1 = torch.empty(N, dtype=torch.float32, device=src.device)
    all_dist2 = torch.empty(N, dtype=torch.float32, device=src.device)
    all_idx1  = torch.empty(N, dtype=torch.int64,   device=src.device)
    all_idx2  = torch.empty(N, dtype=torch.int64,   device=src.device)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        chunk = src[start:end]                        # [B, D]
        src_sq = (chunk * chunk).sum(dim=1, keepdim=True)  # [B, 1]
        # ||a-b||² = ||a||² + ||b||² - 2 a·b
        dists = src_sq + dst_sq.unsqueeze(0) - 2.0 * (chunk @ dst.T)
        dists = dists.clamp_min_(0.0).sqrt_()         # [B, M]

        if M >= 2:
            top2_vals, top2_idx = torch.topk(dists, k=2, dim=1, largest=False)
            all_dist1[start:end] = top2_vals[:, 0]
            all_dist2[start:end] = top2_vals[:, 1]
            all_idx1 [start:end] = top2_idx [:, 0]
            all_idx2 [start:end] = top2_idx [:, 1]
        else:
            # Only one destination descriptor
            vals, idx = dists.min(dim=1)
            all_dist1[start:end] = vals
            all_dist2[start:end] = vals * 2.0         # dummy — ratio always 0.5
            all_idx1 [start:end] = idx
            all_idx2 [start:end] = idx

    best_dist = torch.stack([all_dist1, all_dist2], dim=1)  # [N, 2]
    best_idx  = torch.stack([all_idx1,  all_idx2 ], dim=1)  # [N, 2]
    return best_dist, best_idx


def _batched_cosine_top2(
    src: torch.Tensor,   # [N, D]  L2-normalised
    dst: torch.Tensor,   # [M, D]  L2-normalised
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine distance = 1 − cosine_similarity, batched top-2."""
    N = src.shape[0]
    M = dst.shape[0]

    all_dist1 = torch.empty(N, dtype=torch.float32, device=src.device)
    all_dist2 = torch.empty(N, dtype=torch.float32, device=src.device)
    all_idx1  = torch.empty(N, dtype=torch.int64,   device=src.device)
    all_idx2  = torch.empty(N, dtype=torch.int64,   device=src.device)

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        chunk = src[start:end]                        # [B, D]
        sim   = chunk @ dst.T                         # [B, M]  cosine sim
        dists = 1.0 - sim                             # [B, M]  cosine dist

        if M >= 2:
            top2_vals, top2_idx = torch.topk(dists, k=2, dim=1, largest=False)
            all_dist1[start:end] = top2_vals[:, 0]
            all_dist2[start:end] = top2_vals[:, 1]
            all_idx1 [start:end] = top2_idx [:, 0]
            all_idx2 [start:end] = top2_idx [:, 1]
        else:
            vals, idx = dists.min(dim=1)
            all_dist1[start:end] = vals
            all_dist2[start:end] = vals * 2.0
            all_idx1 [start:end] = idx
            all_idx2 [start:end] = idx

    best_dist = torch.stack([all_dist1, all_dist2], dim=1)
    best_idx  = torch.stack([all_idx1,  all_idx2 ], dim=1)
    return best_dist, best_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_landmarks(
    kpts_src: np.ndarray,
    kpts_dst: np.ndarray,
    desc_src: np.ndarray,
    desc_dst: np.ndarray,
    ratio_thresh: float = 0.75,
    metric: Literal["l2", "cosine"] = "l2",
    batch_size: int = _DEFAULT_BATCH,
    device: Optional[str] = None,
    mutual: bool = False,
) -> np.ndarray:
    """
    Match descriptors using batched GPU nearest-neighbour + Lowe ratio test.

    Memory usage scales with ``batch_size × M × 4`` bytes per chunk rather
    than the full ``N × M × 4`` bytes — critical for large CT descriptor sets
    (e.g. 5000 × 512-D SIFT3D descriptors on a 512³ volume).

    Parameters
    ----------
    kpts_src, kpts_dst : np.ndarray, shape [N, ≥3] and [M, ≥3]
        Keypoint arrays (x_mm, y_mm, z_mm, ...).
    desc_src : np.ndarray, shape [N, D]
    desc_dst : np.ndarray, shape [M, D]
    ratio_thresh : float
        Lowe ratio threshold (keep if dist₁/dist₂ < ratio_thresh).
    metric : 'l2' | 'cosine'
    batch_size : int
        Number of source rows processed per GPU matmul chunk.
        Default 512 ≈ 10 MB per chunk for 512-D descriptors and M=5000.
    device : str | None
        Torch device string. Auto-selected (MPS → CUDA → CPU) if None.
    mutual : bool
        Additionally require the match to be a mutual nearest neighbour
        (the dst descriptor's nearest src descriptor is the same src row).
        Improves precision on inter-subject data at some cost in recall.

    Returns
    -------
    np.ndarray, shape [K, 2]
        Accepted match index pairs (src_idx, dst_idx).
    """
    if desc_src.shape[0] == 0 or desc_dst.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    dev = _get_device(device)
    src = _to_float32(desc_src, dev)   # [N, D]
    dst = _to_float32(desc_dst, dev)   # [M, D]

    if metric == "cosine":
        src = src / (src.norm(dim=1, keepdim=True) + 1e-8)
        dst = dst / (dst.norm(dim=1, keepdim=True) + 1e-8)
        best_dist, best_idx = _batched_cosine_top2(src, dst, batch_size)
    else:
        best_dist, best_idx = _batched_l2_top2(src, dst, batch_size)

    # Lowe ratio test — fully vectorised, no Python loop
    d1 = best_dist[:, 0]   # [N]  distance to 1st NN
    d2 = best_dist[:, 1]   # [N]  distance to 2nd NN
    valid = (d2 > 1e-8) & (d1 / d2.clamp_min(1e-8) < ratio_thresh)

    if mutual:
        if metric == "cosine":
            _, back_idx = _batched_cosine_top2(dst, src, batch_size)
        else:
            _, back_idx = _batched_l2_top2(dst, src, batch_size)
        rows = torch.arange(src.shape[0], device=src.device)
        valid = valid & (back_idx[best_idx[:, 0], 0] == rows)

    src_idx = torch.where(valid)[0].cpu().numpy().astype(np.int32)
    dst_idx = best_idx[valid, 0].cpu().numpy().astype(np.int32)

    if src_idx.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.stack([src_idx, dst_idx], axis=1)


def knn_matches(
    desc_src: np.ndarray,
    desc_dst: np.ndarray,
    k: int = 3,
    batch_size: int = _DEFAULT_BATCH,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    All ``k`` nearest destination descriptors for every source descriptor
    (no ratio test) — a high-recall candidate set for hypothesis voting.
    Returns ``[N*k, 2]`` int32 index pairs ordered by source row then rank.
    """
    if desc_src.shape[0] == 0 or desc_dst.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    dev = _get_device(device)
    src = _to_float32(desc_src, dev)
    dst = _to_float32(desc_dst, dev)
    k = int(min(k, dst.shape[0]))
    dst_sq = (dst * dst).sum(dim=1)
    out = []
    for start in range(0, src.shape[0], batch_size):
        chunk = src[start:start + batch_size]
        d = (chunk * chunk).sum(dim=1, keepdim=True) + dst_sq.unsqueeze(0) - 2.0 * (chunk @ dst.T)
        idx = torch.topk(d, k=k, dim=1, largest=False).indices          # [B, k]
        rows = torch.arange(start, start + chunk.shape[0], device=dev).unsqueeze(1).expand(-1, k)
        out.append(torch.stack([rows.reshape(-1), idx.reshape(-1)], dim=1))
    return torch.cat(out, dim=0).cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

def _fit_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Fit affine [4,4] from at least 4 point correspondences via least squares."""
    if src.shape[0] < 4:
        return None
    A = np.hstack([src, np.ones((src.shape[0], 1))])  # [N, 4]
    try:
        coef, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)  # [4, 3]
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = coef[:3].T
        M[:3, 3]  = coef[3]
        # Anatomical safety check: determinant must be positive and within realistic biological bounds [0.25, 4.0]
        det = float(np.linalg.det(M[:3, :3]))
        if det < 0.25 or det > 4.0:
            return None
        # Condition number bound to reject degenerate coplanar fits
        if float(np.linalg.cond(M[:3, :3])) > 6.0:
            return None
        return M
    except np.linalg.LinAlgError:
        return None


def _fit_regularized_affine(src: np.ndarray, dst: np.ndarray, lambda_reg: float = 0.05) -> Optional[np.ndarray]:
    """Tikhonov-regularized affine fit: penalizes deviation from rigid Kabsch prior."""
    if src.shape[0] < 4:
        return None
    M_rigid = _fit_rigid(src, dst)
    if M_rigid is None:
        return None
    A = np.hstack([src, np.ones((src.shape[0], 1))])  # [N, 4]
    R_prior = M_rigid[:3, :3].T
    t_prior = M_rigid[:3, 3]
    coef_prior = np.vstack([R_prior, t_prior])  # [4, 3]
    try:
        ATA = A.T @ A
        reg = lambda_reg * (np.trace(ATA) / 4.0) * np.eye(4)
        coef = np.linalg.solve(ATA + reg, A.T @ dst + reg @ coef_prior)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = coef[:3].T
        M[:3, 3]  = coef[3]
        det = float(np.linalg.det(M[:3, :3]))
        if det < 0.25 or det > 4.0 or float(np.linalg.cond(M[:3, :3])) > 6.0:
            return M_rigid
        return M
    except np.linalg.LinAlgError:
        return M_rigid


def _fit_rigid(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Fit rigid (rotation + translation) via SVD (Kabsch algorithm)."""
    if src.shape[0] < 3:
        return None
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_d - R @ mu_s
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3]  = t
    return M


def _apply_transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 4×4 homogeneous transform to [N, 3] points."""
    h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    return (M @ h.T).T[:, :3]


def ransac_filter(
    kpts_src: np.ndarray,
    kpts_dst: np.ndarray,
    matches: np.ndarray,
    model: Literal["affine", "rigid", "regularized_affine"] = "affine",
    max_iter: int = 1000,
    inlier_thresh_mm: float = 5.0,
    min_inliers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC geometric verification of landmark matches.

    Residuals are computed in a single vectorised NumPy call per iteration
    — no inner Python loop over match pairs.

    Parameters
    ----------
    kpts_src, kpts_dst : np.ndarray, shape [N, ≥3] and [M, ≥3]
        Physical coordinates (x_mm, y_mm, z_mm).
    matches : np.ndarray, shape [K, 2]
        Index pairs (src_idx, dst_idx) from ``match_landmarks``.
    model : 'affine' | 'rigid'
    max_iter : int
    inlier_thresh_mm : float
        Inlier residual threshold in mm.
    min_inliers : int

    Returns
    -------
    filtered_matches : np.ndarray, shape [M_inlier, 2]
    transform : np.ndarray, shape [4, 4]
        Best-fit transform (identity if RANSAC fails).
    """
    if matches.shape[0] < min_inliers:
        return matches, np.eye(4, dtype=np.float32)

    src_pts = kpts_src[matches[:, 0], :3].astype(np.float32)
    dst_pts = kpts_dst[matches[:, 1], :3].astype(np.float32)
    n       = src_pts.shape[0]
    n_samp  = 4 if model in ("affine", "regularized_affine") else 3

    if n < n_samp:
        return matches, np.eye(4, dtype=np.float32)

    if model == "rigid":
        fit_fn = _fit_rigid
    elif model == "regularized_affine":
        fit_fn = _fit_regularized_affine
    else:
        fit_fn = _fit_affine

    best_count   = -1
    best_inliers = np.zeros(n, dtype=bool)
    best_M       = np.eye(4, dtype=np.float32)

    rng = np.random.default_rng(seed=42)
    for _ in range(max_iter):
        idx  = rng.choice(n, n_samp, replace=False)
        M    = fit_fn(src_pts[idx], dst_pts[idx])
        if M is None:
            continue
        # Vectorised residual computation
        warped    = _apply_transform(src_pts, M)
        residuals = np.linalg.norm(warped - dst_pts, axis=1)  # [n]
        inliers   = residuals < inlier_thresh_mm
        cnt       = int(inliers.sum())
        if cnt > best_count:
            best_count   = cnt
            best_inliers = inliers
            best_M       = M

    if best_count >= min_inliers:
        # Re-fit on consensus inlier set
        M_ref = fit_fn(src_pts[best_inliers], dst_pts[best_inliers])
        if M_ref is not None:
            best_M       = M_ref
            warped       = _apply_transform(src_pts, best_M)
            best_inliers = np.linalg.norm(warped - dst_pts, axis=1) < inlier_thresh_mm

    filtered = matches[best_inliers]
    logger.debug("RANSAC: %d/%d inliers (model=%s)", int(best_inliers.sum()), n, model)
    return filtered, best_M


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_tre(
    kpts_fixed: np.ndarray,
    kpts_moving_warped: np.ndarray,
) -> float:
    """
    Compute mean Target Registration Error (TRE) in mm.

    Both arrays must be **hold-out** landmark positions not used to compute the
    transform (using landmarks that participated in the fit gives FRE, which is
    uncorrelated with TRE and should never be reported as accuracy).

    Parameters
    ----------
    kpts_fixed : np.ndarray, shape [N, ≥3]
        Fixed-space landmark positions (x_mm, y_mm, z_mm).
    kpts_moving_warped : np.ndarray, shape [N, ≥3]
        Corresponding moving-space landmarks after registration warp.

    Returns
    -------
    float
        Mean Euclidean distance in mm.
    """
    if kpts_fixed.shape[0] == 0:
        return float("nan")
    diffs = (kpts_fixed[:, :3].astype(np.float32)
             - kpts_moving_warped[:, :3].astype(np.float32))
    return float(np.mean(np.linalg.norm(diffs, axis=1)))
