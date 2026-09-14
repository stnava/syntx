"""
syntx.landmarks.orient — Global rotation estimation and rotation-search matching
================================================================================

The axis-aligned SIFT3D descriptor is the most discriminative choice but only
tolerates ~30° of relative rotation; the frame-based (``rotation_invariant``)
descriptor tolerates any rotation but is weaker on inter-subject data.  This
module combines them:

1. detect keypoints and *frame-based* descriptors in both images, match them;
2. every match ``(i, j)`` proposes a full rotation ``R_ij = F_j F_iᵀ`` (frames
   are orthonormal, columns = local axes) — correct matches agree on the global
   fixed→moving rotation, wrong ones scatter over SO(3);
3. take the densest cluster of proposals (rotation voting, geodesic radius
   ``cluster_deg``), average it, and
4. rebuild the moving image's descriptors *axis-aligned in that rotated frame*
   so they are directly comparable with the fixed image's axis-aligned
   descriptors; match and RANSAC as usual.

Functions
---------
rotation_geodesic_deg        : pairwise geodesic angle between rotations
estimate_rotation_from_frames: voted global rotation from matched local frames (weak on real brains)
rotation_grid                : coarse SO(3) sampling (optional extra hypotheses)
pca_rotation_candidates      : descriptor-free hypotheses from within-subject landmark-cloud principal axes
refine_rotation_iteratively  : fixed-point refinement (descriptors in frame -> match -> rigid RANSAC -> rotation)
match_sift3d_with_rotation_search : full pipeline returning matches, inliers, affine, rotation
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .sift3d import detect_sift3d, sift3d_keypoints, sift3d_descriptors
from .matcher import match_landmarks, ransac_filter, knn_matches

logger = logging.getLogger(__name__)


def _project_to_so3(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def rotation_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between rotations.  ``Ra`` [N,3,3] vs ``Rb`` [M,3,3] -> [N, M]."""
    Ra = np.asarray(Ra).reshape(-1, 3, 3)
    Rb = np.asarray(Rb).reshape(-1, 3, 3)
    tr = np.einsum("nij,mij->nm", Ra, Rb)                     # trace(Ra Rbᵀ)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def estimate_rotation_from_frames(
    frames_src: np.ndarray,
    frames_dst: np.ndarray,
    matches: np.ndarray,
    cluster_deg: float = 15.0,
    min_votes: int = 5,
    n_candidates: int = 1,
):
    """
    Voted global rotation(s) src→dst from matched per-keypoint frames.

    Every match proposes ``R = F_dst F_srcᵀ``; proposals are clustered with a
    geodesic radius ``cluster_deg`` and the densest clusters are returned
    (non-maximum suppression: a later candidate must be > ``cluster_deg`` from
    every earlier one).

    Returns ``(R [3,3] or None, member_mask [K], n_votes)`` when
    ``n_candidates == 1`` (backward compatible), otherwise a list of
    ``(R, member_mask, n_votes)`` sorted by votes (possibly empty).
    """
    K = matches.shape[0]
    empty = (None, np.zeros(K, dtype=bool), 0)
    if K < min_votes:
        return empty if n_candidates == 1 else []
    Fs = frames_src[matches[:, 0]]                              # [K,3,3]
    Fd = frames_dst[matches[:, 1]]
    props = np.einsum("kij,klj->kil", Fd, Fs)                   # F_d F_sᵀ
    D = rotation_geodesic_deg(props, props)                     # [K,K]
    within = D <= cluster_deg
    votes = within.sum(axis=1)
    order = np.argsort(-votes)
    results = []
    taken = []
    for idx in order:
        if votes[idx] < min_votes or len(results) >= n_candidates:
            break
        R0 = props[idx]
        if any(rotation_geodesic_deg(R0[None], Rt[None])[0, 0] <= cluster_deg for Rt in taken):
            continue
        members = within[idx]
        R = _project_to_so3(props[members].mean(axis=0))
        members = rotation_geodesic_deg(props, R[None])[:, 0] <= cluster_deg     # refine once
        R = _project_to_so3(props[members].mean(axis=0))
        results.append((R, members, int(members.sum())))
        taken.append(R)
    if n_candidates == 1:
        return results[0] if results else empty
    return results


def _axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64); axis = axis / (np.linalg.norm(axis) + 1e-12)
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def rotation_grid(max_deg: float = 90.0, step_deg: float = 30.0, include_identity: bool = True) -> list:
    """
    Roughly uniform set of rotations with geodesic angle ≤ ``max_deg`` and
    spacing ≈ ``step_deg`` (Fibonacci-sphere axes × angle shells, deduplicated so
    no two candidates are closer than ``0.6 * step_deg``).  ~50 rotations for
    (90°, 30°); ~270 would cover all of SO(3) at that spacing.
    """
    out = [np.eye(3)] if include_identity else []
    angles = np.arange(step_deg, max_deg + 1e-6, step_deg)
    for ang in angles:
        # number of axes so that neighbouring axes are ~step apart on the shell of radius ang
        n_axes = max(6, int(round(4.0 * np.pi * np.sin(np.deg2rad(ang)) ** 2 / (np.deg2rad(step_deg) ** 2) * 1.2)))
        i = np.arange(n_axes) + 0.5
        phi = np.arccos(1 - 2 * i / n_axes)
        theta = np.pi * (1 + 5 ** 0.5) * i
        axes = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
        for ax in axes:
            R = _axis_angle(ax, ang)
            if all(rotation_geodesic_deg(R[None], Q[None])[0, 0] > 0.6 * step_deg for Q in out):
                out.append(R)
    return out


def landmark_cloud_axes(pts_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal axes of a within-subject landmark cloud: ``(centre [3], E [3,3])`` with
    eigenvector columns sorted by decreasing variance (right-handed)."""
    P = np.asarray(pts_mm, dtype=np.float64)[:, :3]
    c = P.mean(axis=0)
    C = np.cov((P - c).T)
    w, E = np.linalg.eigh(C)
    E = E[:, ::-1]
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1
    return c, E


def pca_rotation_candidates(pts_fixed: np.ndarray, pts_moving: np.ndarray) -> list:
    """
    Descriptor-free coarse rotation hypotheses fixed→moving from the principal
    axes of the two landmark clouds.  Axis signs are ambiguous, so the four
    right-handed sign combinations are returned (each a [3,3] rotation).
    """
    _, Ef = landmark_cloud_axes(pts_fixed)
    _, Em = landmark_cloud_axes(pts_moving)
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                S = np.diag([sx, sy, sz]).astype(np.float64)
                if np.linalg.det(S) < 0:
                    continue
                out.append(_project_to_so3(Em @ S @ Ef.T))
    return out


def refine_rotation_iteratively(
    sf: dict,
    sm: dict,
    df: np.ndarray,
    R0: Optional[np.ndarray],
    ratio_thresh: float = 0.9,
    mutual: bool = True,
    inlier_thresh_mm: float = 8.0,
    ransac_iter: int = 3000,
    n_iter: int = 5,
    tol_deg: float = 1.0,
    min_rotation_deg: float = 3.0,
    coarse_to_fine: tuple = (0.25, 0.5, 1.0),
    device: Optional[str] = None,
    desc_kw: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """
    Fixed-point refinement of the global rotation: build the moving descriptors
    axis-aligned in the frame ``R_k``, match against the fixed descriptors,
    estimate a rigid transform by RANSAC on the matched landmark coordinates,
    take its rotation as ``R_{k+1}``; stop when the update is below ``tol_deg``.

    ``coarse_to_fine`` schedules the fraction of keypoints used per iteration,
    strongest DoG responses first (e.g. 25% → 50% → 100%, then 100%): the early
    iterations see only the most salient, large, well-separated landmarks, whose
    axis-aligned descriptors tolerate more rotation error, and the estimate is
    then polished with the full set.  Pass ``(1.0,)`` to disable.

    The returned ``R`` is the *converged* rotation (last iterate), together with
    the matches / rigid inliers of the final full-set iteration.
    """
    desc_kw = desc_kw or {}
    cf_all, cm_all = sf["pts"], sm["pts"]
    rf = sf.get("response"); rm = sm.get("response")
    order_f = np.argsort(-rf) if rf is not None and len(rf) == len(cf_all) else np.arange(len(cf_all))
    order_m = np.argsort(-rm) if rm is not None and len(rm) == len(cm_all) else np.arange(len(cm_all))
    R = np.eye(3) if R0 is None else np.asarray(R0, dtype=np.float64)
    history = []
    last = None
    dm_full = None
    for it in range(n_iter):
        frac = coarse_to_fine[min(it, len(coarse_to_fine) - 1)] if coarse_to_fine else 1.0
        nf, nm = max(8, int(round(frac * len(cf_all)))), max(8, int(round(frac * len(cm_all))))
        sel_f, sel_m = order_f[:nf], order_m[:nm]
        use_R = None if rotation_geodesic_deg(R[None], np.eye(3)[None])[0, 0] < min_rotation_deg else R
        dm = sift3d_descriptors(sm, frame_rotation=use_R, **desc_kw)
        m_sub = match_landmarks(cf_all[sel_f], cm_all[sel_m], df[sel_f], dm[sel_m],
                                ratio_thresh=ratio_thresh, mutual=mutual, device=device)
        m = np.column_stack([sel_f[m_sub[:, 0]], sel_m[m_sub[:, 1]]]).astype(np.int32) if len(m_sub) else m_sub
        inl, M = ransac_filter(cf_all, cm_all, m, model="rigid", max_iter=ransac_iter, inlier_thresh_mm=inlier_thresh_mm)
        R_new = _project_to_so3(M[:3, :3].astype(np.float64)) if len(inl) >= 4 else R
        step = float(rotation_geodesic_deg(R_new[None], R[None])[0, 0])
        history.append(dict(iter=it, fraction=frac, rotation_deg=float(rotation_geodesic_deg(R[None], np.eye(3)[None])[0, 0]),
                            n_matches=int(len(m)), n_inliers=int(len(inl)), step_deg=step))
        if verbose:
            logger.info("  iter %d (%.0f%% kps): frame %.1f deg -> %d matches, %d rigid inliers, update %.1f deg",
                        it, 100 * frac, history[-1]["rotation_deg"], len(m), len(inl), step)
        last = dict(R=R_new if len(inl) >= 4 else R, descs_moving=dm, matches=m, inliers=inl, rigid=M, n_inliers=int(len(inl)))
        R = R_new
        if step < tol_deg and frac >= 1.0:
            break
    last["history"] = history
    return last


def match_sift3d_with_rotation_search(
    fixed,
    moving,
    ratio_thresh: float = 0.95,
    mutual: bool = True,
    ransac_model: str = "affine",
    inlier_thresh_mm: float = 8.0,
    ransac_iter: int = 5000,
    refine_ratio: float = 0.9,
    n_iter: int = 6,
    tol_deg: float = 1.0,
    coarse_to_fine: tuple = (0.25, 0.5, 1.0),
    extra_candidates: Optional[list] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    **detect_kw,
) -> dict:
    """
    Detect + match SIFT3D landmarks between two (preprocessed) images with a
    global rotation search that converges iteratively.

    1. Keypoints and the physical gradient are computed once per image
       (``sift3d_keypoints``); fixed descriptors are axis-aligned.
    2. Coarse hypotheses fixed→moving: identity plus the four principal-axis
       alignments of the two within-subject landmark clouds
       (``pca_rotation_candidates``), plus any ``extra_candidates``.
    3. Each hypothesis is refined by ``refine_rotation_iteratively`` (descriptors
       in frame → match → rigid RANSAC → new rotation, ≤ ``n_iter`` rounds),
       coarse-to-fine over the keypoint set (``coarse_to_fine`` fractions of the
       strongest responses) — PCA only seeds iteration 0, everything after is
       standard detection.
    4. The refined hypothesis with most rigid inliers wins; the final matching
       uses ``ratio_thresh`` and a ``ransac_model`` RANSAC for the returned affine.

    ``detect_kw`` is forwarded to ``sift3d_keypoints`` / ``sift3d_descriptors``
    (``preprocess`` defaults to False — pass preprocessed images).

    Returns dict: ``coords_fixed`` [N,4], ``coords_moving`` [M,4], ``descs_fixed``,
    ``descs_moving`` (moving in the winning frame), ``matches`` [K,2], ``inliers`` [L,2],
    ``affine`` [4,4], ``rotation`` [3,3] (fixed→moving directions), ``rotation_deg``,
    ``used_rotation`` (bool), ``candidates`` (per-hypothesis summaries).
    """
    detect_kw.setdefault("preprocess", False)
    desc_kw = {k: detect_kw.pop(k) for k in ("n_cells", "n_bins", "samples_per_cell") if k in detect_kw}
    sf = sift3d_keypoints(fixed, device=device, **detect_kw)
    sm = sift3d_keypoints(moving, device=device, **detect_kw)
    cf, cm = sf["pts"], sm["pts"]
    df = sift3d_descriptors(sf, **desc_kw)
    if cf.shape[0] < 4 or cm.shape[0] < 4:
        return dict(coords_fixed=cf, coords_moving=cm, descs_fixed=df,
                    descs_moving=np.zeros((cm.shape[0], df.shape[1] if df.ndim == 2 else 512), np.float32),
                    matches=np.zeros((0, 2), np.int32), inliers=np.zeros((0, 2), np.int32),
                    affine=np.eye(4, dtype=np.float32), rotation=np.eye(3), rotation_deg=0.0,
                    used_rotation=False, candidates=[])

    hyps = [("identity", None)] + [(f"pca{i}", R) for i, R in enumerate(pca_rotation_candidates(cf, cm))]
    for i, R in enumerate(extra_candidates or []):
        hyps.append((f"extra{i}", np.asarray(R, dtype=np.float64)))

    results = []
    for name, R0 in hyps:
        if verbose:
            logger.info("hypothesis %s (%.1f deg)", name, 0.0 if R0 is None else rotation_geodesic_deg(R0[None], np.eye(3)[None])[0, 0])
        r = refine_rotation_iteratively(sf, sm, df, R0, ratio_thresh=refine_ratio, mutual=mutual,
                                        inlier_thresh_mm=inlier_thresh_mm, n_iter=n_iter, tol_deg=tol_deg,
                                        coarse_to_fine=coarse_to_fine, device=device, desc_kw=desc_kw, verbose=verbose)
        r["name"] = name
        results.append(r)
    best = max(results, key=lambda r: r["n_inliers"])
    R = best["R"]
    deg = float(rotation_geodesic_deg(R[None], np.eye(3)[None])[0, 0])
    used = deg >= 3.0
    # final matching in the winning frame with the requested ratio / model
    dm = sift3d_descriptors(sm, frame_rotation=R if used else None, **desc_kw)
    m = match_landmarks(cf, cm, df, dm, ratio_thresh=ratio_thresh, mutual=mutual, device=device)
    inl, M = ransac_filter(cf, cm, m, model=ransac_model, max_iter=ransac_iter, inlier_thresh_mm=inlier_thresh_mm)
    summaries = [dict(name=r["name"], final_rotation_deg=float(rotation_geodesic_deg(r["R"][None], np.eye(3)[None])[0, 0]),
                      n_inliers=r["n_inliers"], iterations=len(r["history"]), history=r["history"]) for r in results]
    return dict(coords_fixed=cf, coords_moving=cm, descs_fixed=df, descs_moving=dm,
                matches=m, inliers=inl, affine=M,
                rotation=R if used else np.eye(3), rotation_deg=deg if used else 0.0,
                used_rotation=used, candidates=summaries, winner=best["name"])
