"""
syntx.landmarks.matcher — Lowe ratio-test matching + RANSAC filtering
======================================================================

Provides modality-independent matching of landmark descriptors (SIFT, MIND, etc.)
and geometric verification via RANSAC.

Functions
---------
match_landmarks : L2 nearest-neighbour with Lowe ratio test
ransac_filter   : RANSAC geometric verification (affine or rigid model)
compute_tre     : Target Registration Error in mm (use hold-out points only)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_landmarks(
    kpts_src: np.ndarray,
    kpts_dst: np.ndarray,
    desc_src: np.ndarray,
    desc_dst: np.ndarray,
    ratio_thresh: float = 0.75,
    metric: Literal["l2", "cosine"] = "l2",
) -> np.ndarray:
    """
    Match descriptors using nearest-neighbour search + Lowe ratio test.

    Parameters
    ----------
    kpts_src, kpts_dst : np.ndarray, shape [N, ≥3] and [M, ≥3]
        Keypoint arrays (x_mm, y_mm, z_mm, ...).  Used only for shape checks.
    desc_src : np.ndarray, shape [N, D]
    desc_dst : np.ndarray, shape [M, D]
    ratio_thresh : float
        Lowe ratio: keep match if dist_1st / dist_2nd < ratio_thresh.
    metric : 'l2' | 'cosine'
        Distance metric.

    Returns
    -------
    np.ndarray, shape [K, 2]
        Accepted match index pairs (src_idx, dst_idx).
    """
    if desc_src.shape[0] == 0 or desc_dst.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    src = desc_src.astype(np.float32)
    dst = desc_dst.astype(np.float32)

    if metric == "cosine":
        src = src / (np.linalg.norm(src, axis=1, keepdims=True) + 1e-8)
        dst = dst / (np.linalg.norm(dst, axis=1, keepdims=True) + 1e-8)
        # cosine similarity → distance = 1 - sim
        sim = src @ dst.T          # [N, M]
        dist = 1.0 - sim
    else:
        # L2: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        s2 = (src ** 2).sum(axis=1, keepdims=True)   # [N, 1]
        d2 = (dst ** 2).sum(axis=1, keepdims=True).T # [1, M]
        dist = s2 + d2 - 2 * (src @ dst.T)           # [N, M]
        dist = np.clip(dist, 0.0, None)
        dist = np.sqrt(dist)

    matches = []
    for i in range(dist.shape[0]):
        row = dist[i]
        if dist.shape[1] < 2:
            best = int(np.argmin(row))
            matches.append([i, best])
            continue
        idx_sorted = np.argsort(row)
        d1, d2 = row[idx_sorted[0]], row[idx_sorted[1]]
        if d2 < 1e-8:
            continue
        if d1 / d2 < ratio_thresh:
            matches.append([i, int(idx_sorted[0])])

    if not matches:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(matches, dtype=np.int32)


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

def _fit_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Fit affine [4,4] from at least 4 point correspondences via least squares."""
    if src.shape[0] < 4:
        return None
    # Augment with homogeneous coordinate
    A = np.hstack([src, np.ones((src.shape[0], 1))])   # [N, 4]
    # Solve for [3, 4] affine (row per output dim)
    try:
        coef, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)  # [4, 3]
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = coef[:3].T
        M[:3, 3]  = coef[3]
        return M
    except np.linalg.LinAlgError:
        return None


def _fit_rigid(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Fit rigid (rotation + translation) via SVD (Kabsch algorithm)."""
    if src.shape[0] < 3:
        return None
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d
    H = S.T @ D
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
    """Apply 4×4 transform to [N,3] points."""
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (M @ h.T).T[:, :3]


def ransac_filter(
    kpts_src: np.ndarray,
    kpts_dst: np.ndarray,
    matches: np.ndarray,
    model: Literal["affine", "rigid"] = "affine",
    max_iter: int = 1000,
    inlier_thresh_mm: float = 5.0,
    min_inliers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC geometric verification of landmark matches.

    Parameters
    ----------
    kpts_src, kpts_dst : np.ndarray, shape [N, ≥3] and [M, ≥3]
        Physical coordinates (x_mm, y_mm, z_mm).
    matches : np.ndarray, shape [K, 2]
        Index pairs (src_idx, dst_idx) from ``match_landmarks``.
    model : 'affine' | 'rigid'
        Transformation model to fit.
    max_iter : int
        Maximum RANSAC iterations.
    inlier_thresh_mm : float
        Residual threshold in mm to classify an inlier.
    min_inliers : int
        Minimum inliers to accept the model.

    Returns
    -------
    filtered_matches : np.ndarray, shape [M_inlier, 2]
        Inlier match index pairs.
    transform : np.ndarray, shape [4, 4]
        Best-fit transform (identity if no model found).
    """
    if matches.shape[0] < min_inliers:
        return matches, np.eye(4, dtype=np.float32)

    src_pts = kpts_src[matches[:, 0], :3].astype(np.float32)
    dst_pts = kpts_dst[matches[:, 1], :3].astype(np.float32)

    n = src_pts.shape[0]
    n_sample = 4 if model == "affine" else 3
    if n < n_sample:
        return matches, np.eye(4, dtype=np.float32)

    fit_fn = _fit_affine if model == "affine" else _fit_rigid

    best_inliers = np.zeros(n, dtype=bool)
    best_M = np.eye(4, dtype=np.float32)

    rng = np.random.default_rng(seed=42)
    for _ in range(max_iter):
        idx = rng.choice(n, n_sample, replace=False)
        M = fit_fn(src_pts[idx], dst_pts[idx])
        if M is None:
            continue
        warped = _apply_transform(src_pts, M)
        residuals = np.linalg.norm(warped - dst_pts, axis=1)
        inliers = residuals < inlier_thresh_mm
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_M = M

    if best_inliers.sum() >= min_inliers:
        # Re-fit on all inliers
        M_refined = fit_fn(src_pts[best_inliers], dst_pts[best_inliers])
        if M_refined is not None:
            best_M = M_refined
            warped = _apply_transform(src_pts, best_M)
            best_inliers = np.linalg.norm(warped - dst_pts, axis=1) < inlier_thresh_mm

    filtered = matches[best_inliers]
    logger.debug("RANSAC: %d/%d inliers", best_inliers.sum(), n)
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
    diffs = kpts_fixed[:, :3].astype(np.float32) - kpts_moving_warped[:, :3].astype(np.float32)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))
