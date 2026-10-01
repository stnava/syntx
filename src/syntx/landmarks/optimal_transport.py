"""
Soft (optimal-transport) correspondences and sampled feature scores for rigid / similarity
initialisation.

1. ``sinkhorn_matching``: entropy-regularised optimal transport between two descriptor
   clouds, with a dustbin row / column for unmatched points, in log space. Gives a soft
   N x M correspondence matrix instead of discrete matches + RANSAC.
2. ``weighted_procrustes``: closed-form weighted Kabsch / Umeyama rigid or similarity fit
   via SVD (det(R) = +1).
3. ``sampled_optimal_transport_affine``: samples a fraction of the foreground voxels of
   both images, takes MIND-style descriptors (``syntx.landmarks.mind``) there, solves the
   Sinkhorn problem and fits a rigid / similarity transform to the soft correspondences;
   writes an ITK affine transform file.
4. ``score_rotation_candidates_sampled``: mean cosine similarity of fixed and moving
   descriptors at sampled fixed points mapped by each candidate rotation about the
   foreground centroids.

Coordinates are physical (ITK / ANTs LPS) mm via ``syntx.landmarks.spatial``; both images
must be 3-D.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import ants

from .mind import compute_mind
from .spatial import (
    vox_to_physical,
    physical_to_vox,
    get_image_affine,
)

logger = logging.getLogger(__name__)


def sinkhorn_matching(
    feat_src: torch.Tensor,
    feat_dst: torch.Tensor,
    coords_src: Optional[torch.Tensor] = None,
    coords_dst: Optional[torch.Tensor] = None,
    epsilon: float = 0.05,
    dustbin_cost: float = 1.0,
    spatial_weight: float = 0.5,
    n_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft correspondence (transport plan) between two descriptor clouds via Sinkhorn.

    Cost ``C[i, j] = 1 - cos(feat_src[i], feat_dst[j])`` (+ ``spatial_weight * |x_i - y_j| /
    100 mm`` if both coordinate sets are given). The cost matrix is augmented with a dustbin
    row and column of cost ``dustbin_cost`` (dustbin-dustbin cost 0); marginals are 1 per
    real point, M for the source dustbin and N for the destination dustbin. ``n_iters``
    log-space Sinkhorn updates with kernel ``-C / epsilon`` (epsilon floored at 1e-6).

    Parameters
    ----------
    feat_src : torch.Tensor, shape [N, D]
        Source descriptors (L2-normalised inside).
    feat_dst : torch.Tensor, shape [M, D]
        Destination descriptors (L2-normalised inside).
    coords_src : torch.Tensor, shape [N, 3], optional
        Physical coordinates (mm) for source points.
    coords_dst : torch.Tensor, shape [M, 3], optional
        Physical coordinates (mm) for destination points.
    epsilon : float, default 0.05
        Entropy regularization (in cost units); smaller = sharper plan.
    dustbin_cost : float, default 1.0
        Cost of sending a point to the dustbin; pairs with a higher cost tend to be
        routed there.
    spatial_weight : float, default 0.5
        Weight of the distance term (cost units per 100 mm). The distance is between raw
        physical coordinates of the two images, so it favours correspondences close to the
        identity mapping. 0 or missing coordinates disable it.
    n_iters : int, default 50
        Number of Sinkhorn iterations (no convergence check).

    Returns
    -------
    P_match : torch.Tensor, shape [N, M]
        Transport plan between real points (dustbin row / column dropped).
    weights : torch.Tensor, shape [N]
        Row sums of ``P_match``: the fraction (about 0..1) of each source point's unit mass
        that is matched rather than sent to the dustbin.
    """
    device = feat_src.device
    N, D = feat_src.shape
    M = feat_dst.shape[0]

    # Feature cosine cost: C_feat = 1 - cos(feat_src, feat_dst)
    feat_src_n = F.normalize(feat_src, p=2, dim=-1)
    feat_dst_n = F.normalize(feat_dst, p=2, dim=-1)
    sim = torch.matmul(feat_src_n, feat_dst_n.T)  # [N, M]
    cost = 1.0 - sim

    # Optional physical spatial distance penalty (scaled to decimetres = 100mm)
    if coords_src is not None and coords_dst is not None and spatial_weight > 0.0:
        dist_spatial = torch.cdist(coords_src, coords_dst) / 100.0
        cost = cost + spatial_weight * dist_spatial

    # Augmented cost matrix with dustbin: [N+1, M+1]
    C_aug = torch.full((N + 1, M + 1), float(dustbin_cost), device=device, dtype=cost.dtype)
    C_aug[:N, :M] = cost
    C_aug[-1, -1] = 0.0

    # Log-space kernel
    K = -C_aug / max(float(epsilon), 1e-6)

    u = torch.zeros(N + 1, device=device, dtype=cost.dtype)
    v = torch.zeros(M + 1, device=device, dtype=cost.dtype)

    # Marginal constraints: p has weight 1 for real points, M for dustbin; q has weight 1, N for dustbin
    p = torch.ones(N + 1, device=device, dtype=cost.dtype)
    p[-1] = float(M)
    q = torch.ones(M + 1, device=device, dtype=cost.dtype)
    q[-1] = float(N)

    log_p = torch.log(p)
    log_q = torch.log(q)

    # Sinkhorn iterations in log space
    for _ in range(n_iters):
        u = log_p - torch.logsumexp(K + v.unsqueeze(0), dim=1)
        v = log_q - torch.logsumexp(K + u.unsqueeze(1), dim=0)

    P = torch.exp(K + u.unsqueeze(1) + v.unsqueeze(0))
    P_match = P[:N, :M]
    weights = P_match.sum(dim=1)

    return P_match, weights


def weighted_procrustes(
    coords_src: torch.Tensor,
    coords_dst: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    allow_scaling: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Solve closed-form weighted orthogonal Procrustes (Kabsch / Umeyama) via SVD.

    Finds rigid or similarity transformation aligning coords_src -> coords_dst
    (weighted least squares):
        y ≈ s * R @ x + t

    The SVD runs on the CPU; outputs are returned on ``coords_src``'s device / dtype.

    Parameters
    ----------
    coords_src : torch.Tensor, shape [N, 3]
        Source physical coordinates (e.g. fixed points).
    coords_dst : torch.Tensor, shape [N, 3]
        Destination physical coordinates (e.g. soft-transported moving points).
    weights : torch.Tensor, shape [N], optional
        Confidence weights per point. If None, uniform weights are used.
    allow_scaling : bool, default False
        If True, estimates isotropic scale s. If False, s = 1.0 (pure rigid).

    Returns
    -------
    R : torch.Tensor, shape [3, 3]
        Orthonormal rotation matrix with det(R) = +1.
    t : torch.Tensor, shape [3]
        Translation vector in physical LPS space.
    scale : float
        Estimated scale factor: sum of singular values / weighted source variance (the
        reflection-corrected case does not flip the sign of the last singular value);
        1.0 if allow_scaling is False.
    affine_4x4 : torch.Tensor, shape [4, 4]
        ``[[s R, t], [0, 1]]``.
    """
    device = coords_src.device
    coords_dst = coords_dst.to(device=device, dtype=coords_src.dtype)
    N = coords_src.shape[0]

    if weights is None:
        w = torch.ones(N, 1, device=device, dtype=coords_src.dtype)
    else:
        w = weights.view(N, 1).to(device=device, dtype=coords_src.dtype)

    w_sum = w.sum().clamp_min(1e-8)

    # Weighted centroids
    c_src = (coords_src * w).sum(dim=0, keepdim=True) / w_sum
    c_dst = (coords_dst * w).sum(dim=0, keepdim=True) / w_sum

    src_cent = coords_src - c_src
    dst_cent = coords_dst - c_dst

    # Weighted covariance: H = (src_cent * w)^T @ dst_cent
    H = torch.matmul((src_cent * w).T, dst_cent)

    # linalg.svd on CPU to avoid MPS backend fallback quirks
    H_cpu = H.detach().cpu()
    U, S, Vt = torch.linalg.svd(H_cpu)

    R_np = (Vt.T @ U.T).numpy()

    # Reflection check: ensure right-handed rotation (det(R) = +1)
    if np.linalg.det(R_np) < 0:
        Vt[-1, :] *= -1
        R_np = (Vt.T @ U.T).numpy()

    R = torch.tensor(R_np, device=device, dtype=coords_src.dtype)

    # Scale estimation (Umeyama)
    scale = 1.0
    if allow_scaling:
        var_src = ((src_cent ** 2) * w).sum() / w_sum
        scale = float((S.sum() / (var_src.cpu() * w_sum.cpu())).item())

    # Translation: t = c_dst - scale * (c_src @ R^T)
    t = (c_dst - scale * torch.matmul(c_src, R.T)).squeeze(0)

    # Assemble 4x4 affine matrix
    affine_4x4 = torch.eye(4, device=device, dtype=coords_src.dtype)
    affine_4x4[:3, :3] = scale * R
    affine_4x4[:3, 3] = t

    return R, t, scale, affine_4x4


def sampled_optimal_transport_affine(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    sampling_percentage: float = 0.01,
    min_samples: int = 500,
    max_samples: int = 8000,
    n_mind_offsets: int = 12,
    mind_patch_size: int = 7,
    epsilon: float = 0.05,
    dustbin_cost: float = 1.0,
    spatial_weight: float = 0.5,
    n_sinkhorn_iters: int = 50,
    allow_scaling: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
    return_dict: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Rigid (or similarity) initial alignment fixed -> moving from sampled soft correspondences.

    Steps: dense ``compute_mind`` descriptors of both images (L2-normalised per voxel);
    sample voxels with raw intensity > 0.05 in each image (without replacement, seeded);
    ``sinkhorn_matching`` between the fixed and moving samples (with the spatial term);
    each fixed sample with matched mass > 0.01 gets the soft moving point
    ``sum_j P_ij y_j / sum_j P_ij``; ``weighted_procrustes`` fits fixed -> soft moving points
    (weights = matched mass). If fewer than 4 samples pass, all samples are used with unit
    weights. Only a rotation (+ scale) and translation are estimated, despite the name.

    Parameters
    ----------
    fixed : ants.ANTsImage
        Fixed reference image (3-D).
    moving : ants.ANTsImage
        Moving source image (3-D).
    sampling_percentage : float, default 0.01
        Fraction of foreground voxels to sample from each image (e.g. 0.01 = 1%).
    min_samples : int, default 500
        Minimum number of samples. Applied after the cap at the foreground size, so a
        foreground smaller than this raises ValueError from ``rng.choice``.
    max_samples : int, default 8000
        Maximum number of samples (the Sinkhorn matrices are samples_f x samples_m).
    n_mind_offsets : int, default 12
        ``compute_mind`` offsets (see ``mind._make_offsets``).
    mind_patch_size : int, default 7
        ``compute_mind`` patch size (offset distance is the default 1 mm).
    epsilon, dustbin_cost, spatial_weight : float, default 0.05, 1.0, 0.5
        ``sinkhorn_matching`` options.
    n_sinkhorn_iters : int, default 50
        Number of Sinkhorn iterations.
    allow_scaling : bool, default False
        Also estimate an isotropic scale.
    device : str, optional
        Torch computation device ('mps', 'cuda', 'cpu'). Auto-selected if None.
    seed : int, default 42
        Random seed for the sampling.
    return_dict : bool, default False
        Return a diagnostics dict instead of only the path.

    Returns
    -------
    str or dict
        Path of a new temporary ITK ``AffineTransform`` .mat file (not deleted; centre 0)
        mapping fixed physical points to moving ones (``ants.apply_transforms`` convention).
        With ``return_dict``: ``transform_path``, ``rotation`` [3,3], ``translation`` [3],
        ``scale``, ``affine_matrix`` [4,4], ``n_samples_fixed``, ``n_samples_moving``,
        ``n_valid_matches``, ``mean_weight``, ``runtime_seconds``, ``device``. If either
        image has no voxel > 0.05 an identity transform is written and the dict is only
        ``{'transform_path', 'runtime': 0.0, 'status': 'EMPTY_FG'}``.
    """
    import tempfile

    t0 = time.time()
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Compute 3D MIND-SSC features
    mind_fi = compute_mind(fixed, n_offsets=n_mind_offsets, patch_size=mind_patch_size, device=device)
    mind_mi = compute_mind(moving, n_offsets=n_mind_offsets, patch_size=mind_patch_size, device=device)

    mind_fi_norm = F.normalize(mind_fi, p=2, dim=1)
    mind_mi_norm = F.normalize(mind_mi, p=2, dim=1)

    # Step 2: Extract foreground voxel domains
    fi_arr = fixed.numpy()
    mi_arr = moving.numpy()

    fi_fg = np.argwhere(fi_arr > 0.05)
    mi_fg = np.argwhere(mi_arr > 0.05)

    n_fi_fg = len(fi_fg)
    n_mi_fg = len(mi_fg)

    if n_fi_fg == 0 or n_mi_fg == 0:
        logger.warning("Empty foreground detected in sampled_optimal_transport_affine; falling back to identity.")
        tx = ants.new_ants_transform(precision='float', dimension=3, transform_type='AffineTransform')
        tx.set_parameters(np.concatenate([np.eye(3).flatten(), np.zeros(3)]))
        tx.set_fixed_parameters(np.zeros(3))
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            tx_path = f.name
        ants.write_transform(tx, tx_path)
        return tx_path if not return_dict else {"transform_path": tx_path, "runtime": 0.0, "status": "EMPTY_FG"}

    # Dynamic percentage-based sample sizing
    k_f = max(min_samples, min(int(round(n_fi_fg * float(sampling_percentage))), max_samples, n_fi_fg))
    k_m = max(min_samples, min(int(round(n_mi_fg * float(sampling_percentage))), max_samples, n_mi_fg))

    rng = np.random.default_rng(seed)
    idx_f = rng.choice(n_fi_fg, size=k_f, replace=False)
    idx_m = rng.choice(n_mi_fg, size=k_m, replace=False)

    vox_f = fi_fg[idx_f]
    vox_m = mi_fg[idx_m]

    # Convert voxels to physical LPS mm coordinates via spatial module
    pts_f_mm = vox_to_physical(fixed, vox_f)
    pts_m_mm = vox_to_physical(moving, vox_m)

    # Extract sampled descriptors: native XYZ layout [1, C, nx, ny, nz]
    feat_f = mind_fi_norm[0, :, vox_f[:, 0], vox_f[:, 1], vox_f[:, 2]].T
    feat_m = mind_mi_norm[0, :, vox_m[:, 0], vox_m[:, 1], vox_m[:, 2]].T

    pts_f_dev = torch.tensor(pts_f_mm, dtype=torch.float32, device=device)
    pts_m_dev = torch.tensor(pts_m_mm, dtype=torch.float32, device=device)

    # Step 3: Solve Sinkhorn Optimal Transport with Dustbin
    P_match, weights = sinkhorn_matching(
        feat_src=feat_f,
        feat_dst=feat_m,
        coords_src=pts_f_dev,
        coords_dst=pts_m_dev,
        epsilon=epsilon,
        dustbin_cost=dustbin_cost,
        spatial_weight=spatial_weight,
        n_iters=n_sinkhorn_iters,
    )

    # Step 4: Compute soft moving coordinates and confidence weights
    valid_mask = weights > 0.01
    n_valid = int(valid_mask.sum().item())

    if n_valid < 4:
        logger.warning(f"Only {n_valid} valid soft matches in Sinkhorn; using unweighted centroids.")
        valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        weights = torch.ones_like(weights)

    P_norm = P_match / (weights.unsqueeze(1) + 1e-8)
    soft_pts_m = torch.matmul(P_norm, pts_m_dev)

    # Step 5: Solve Weighted Procrustes in closed-form
    R, t, scale, aff_mat = weighted_procrustes(
        coords_src=pts_f_dev[valid_mask],
        coords_dst=soft_pts_m[valid_mask],
        weights=weights[valid_mask],
        allow_scaling=allow_scaling,
    )

    R_np = R.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    aff_mat_np = aff_mat.detach().cpu().numpy()

    # Step 6: Package into ITK ANTs Transform
    # ITK convention: y = A(x - c) + c + t => with c=0, y = A x + t
    tx = ants.new_ants_transform(precision='float', dimension=3, transform_type='AffineTransform')
    matrix_param = (scale * R_np).flatten()
    tx.set_parameters(np.concatenate([matrix_param, t_np]))
    tx.set_fixed_parameters(np.zeros(3))

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        tx_path = f.name
    ants.write_transform(tx, tx_path)

    elapsed = time.time() - t0

    if return_dict:
        return {
            "transform_path": tx_path,
            "rotation": R_np,
            "translation": t_np,
            "scale": scale,
            "affine_matrix": aff_mat_np,
            "n_samples_fixed": k_f,
            "n_samples_moving": k_m,
            "n_valid_matches": n_valid,
            "mean_weight": float(weights.mean().item()),
            "runtime_seconds": elapsed,
            "device": device,
        }

    return tx_path


def score_rotation_candidates_sampled(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    candidate_rotations: Union[List[np.ndarray], np.ndarray],
    sampling_percentage: float = 0.01,
    min_samples: int = 500,
    max_samples: int = 4000,
    feature_type: str = "mind",
    device: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Score candidate rotations fixed->moving by sampled feature agreement.

    Evaluates:
        score(R) = (1 / N) * sum_k < F_fixed(x_k), F_moving( R (x_k - c_f) + c_m ) >

    over N fixed voxels with raw intensity > 0.05 (sampled without replacement, seeded);
    c_f is the mean of those samples and c_m the mean of a random subset of the moving
    voxels > 0.05 (origin if none). Moving features are trilinearly sampled (zeros outside
    the image) and re-normalised to unit length. All candidates share one sample set.

    Parameters
    ----------
    fixed : ants.ANTsImage
        Fixed target image (3-D).
    moving : ants.ANTsImage
        Moving source image (3-D).
    candidate_rotations : list of [3, 3] np.ndarray
        Candidate rotation matrices (fixed -> moving directions) to evaluate.
    sampling_percentage : float, default 0.01
        Fraction of fixed foreground voxels to sample.
    min_samples : int, default 500
        Minimum sample points (as in ``sampled_optimal_transport_affine``, a smaller
        foreground raises ValueError).
    max_samples : int, default 4000
        Maximum sample points.
    feature_type : str, default 'mind'
        'mind': unit-normalised ``compute_mind`` descriptors (12 offsets, patch 7), score =
        mean cosine similarity in [-1, 1]. Any other value: intensity / image max as a single
        channel; after re-normalisation the moving value is just 1 (inside, positive) or 0,
        so the score is the mean normalised fixed intensity of the samples that land on
        positive moving voxels.
    device : str, optional
        Torch device; auto-selected (mps > cuda > cpu) if None.
    seed : int, default 42
        Sampling seed.

    Returns
    -------
    results : list of dict
        List of dicts sorted by score (descending), each with:
        {"index": int (position in ``candidate_rotations``), "rotation": the input matrix,
        "score": float}. If the fixed image has no voxel > 0.05, all scores are 0.0 in input
        order.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    fi_arr = fixed.numpy()
    fi_fg = np.argwhere(fi_arr > 0.05)
    n_fg = len(fi_fg)

    if n_fg == 0:
        return [{"index": i, "rotation": R, "score": 0.0} for i, R in enumerate(candidate_rotations)]

    k = max(min_samples, min(int(round(n_fg * float(sampling_percentage))), max_samples, n_fg))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_fg, size=k, replace=False)
    vox_f = fi_fg[idx]

    pts_f_mm = vox_to_physical(fixed, vox_f)
    cf = pts_f_mm.mean(axis=0)

    # Compute moving foreground centroid in physical space
    mi_arr = moving.numpy()
    mi_fg = np.argwhere(mi_arr > 0.05)
    if len(mi_fg) > 0:
        cm = vox_to_physical(moving, mi_fg[rng.choice(len(mi_fg), size=min(k, len(mi_fg)), replace=False)]).mean(axis=0)
    else:
        cm = np.zeros(3)

    # Feature representations
    if feature_type == "mind":
        mind_fi = compute_mind(fixed, n_offsets=12, patch_size=7, device=device)
        mind_mi = compute_mind(moving, n_offsets=12, patch_size=7, device=device)
        feat_fi_norm = F.normalize(mind_fi, p=2, dim=1)
        feat_mi_norm = F.normalize(mind_mi, p=2, dim=1)
        feat_f = feat_fi_norm[0, :, vox_f[:, 0], vox_f[:, 1], vox_f[:, 2]].T  # [k, C]
    else:
        # Intensity feature fallback
        f_tens = torch.tensor(fi_arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        m_tens = torch.tensor(mi_arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        feat_fi_norm = f_tens / (f_tens.max().clamp_min(1e-6))
        feat_mi_norm = m_tens / (m_tens.max().clamp_min(1e-6))
        feat_f = feat_fi_norm[0, :, vox_f[:, 0], vox_f[:, 1], vox_f[:, 2]].T

    feat_f_dev = feat_f.to(device)
    pts_f_centered = torch.tensor(pts_f_mm - cf, dtype=torch.float32, device=device)  # [k, 3]
    cm_dev = torch.tensor(cm, dtype=torch.float32, device=device)

    nx, ny, nz = moving.shape
    scores = []

    for i, R_np in enumerate(candidate_rotations):
        R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
        # Transformed physical coordinates: x'_mm = R @ (x_mm - cf) + cm
        pts_m_pred = (pts_f_centered @ R_torch.T) + cm_dev
        pts_m_np = pts_m_pred.detach().cpu().numpy()

        # Convert to moving voxel coordinates
        vox_m_pred = physical_to_vox(moving, pts_m_np)  # [k, 3]

        # Normalized coordinates for grid_sample: [ix, iy, iz] -> [-1, 1]
        grid_x = 2.0 * vox_m_pred[:, 0] / (nx - 1) - 1.0
        grid_y = 2.0 * vox_m_pred[:, 1] / (ny - 1) - 1.0
        grid_z = 2.0 * vox_m_pred[:, 2] / (nz - 1) - 1.0

        # grid_sample convention for [1, C, nx, ny, nz]: (iz, iy, ix)
        grid_np = np.stack([grid_z, grid_y, grid_x], axis=-1)
        grid = torch.tensor(grid_np, dtype=torch.float32, device=device).view(1, 1, 1, k, 3)

        sampled_feat_m = F.grid_sample(feat_mi_norm, grid, mode='bilinear', align_corners=True)  # [1, C, 1, 1, k]
        sampled_feat_m = sampled_feat_m.squeeze().T  # [k, C]
        sampled_feat_m = F.normalize(sampled_feat_m, p=2, dim=-1)

        # Mean cosine similarity
        cos_sim = float((feat_f_dev * sampled_feat_m).sum(dim=-1).mean().item())
        scores.append({"index": i, "rotation": R_np, "score": cos_sim})

    scores.sort(key=lambda s: s["score"], reverse=True)
    return scores
