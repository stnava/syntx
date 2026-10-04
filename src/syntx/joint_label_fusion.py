"""
Joint Label Fusion (JLF) — ``syntx.joint_label_fusion``.

High-performance, patch-based multi-atlas consensus label fusion. Supports:
1. Fast patch-based similarity weighting (``mode="patch"``):
   Voxel-wise Local Normalized Cross-Correlation (LNCC) or Local MSE weights between target
   and warped atlas patches, computed with sub-second separable box filters.
2. Full Joint Label Fusion (``mode="joint"``):
   Solves the batched K x K linear system with pairwise atlas error covariance
   to explicitly penalize correlated atlas errors (Wang et al., IEEE PAMI 2013).

Optimized with automated bounding-box ROI extraction so large 3D full-head volumes
execute in seconds on CPU or GPU/MPS without redundant full-field computation.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import ants
import numpy as np
import torch

from .core.pipeline import auto_detect_device
from .core.smoothing import separable_1d_filter


def joint_label_fusion(
    target_image: ants.ANTsImage,
    atlas_images: Sequence[ants.ANTsImage],
    atlas_labels: Sequence[ants.ANTsImage],
    mask: ants.ANTsImage | None = None,
    rad: int = 2,
    beta: float = 2.0,
    rho: float = 0.05,
    mode: Literal["patch", "joint"] = "joint",
    nonnegative: bool = True,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute high-performance joint label fusion across warped atlas candidates.

    Parameters
    ----------
    target_image : ants.ANTsImage
        Target reference structural image in native or registered space.
    atlas_images : sequence of ants.ANTsImage
        Warped atlas intensity images aligned to `target_image`.
    atlas_labels : sequence of ants.ANTsImage
        Warped atlas candidate label maps aligned to `target_image`.
    mask : ants.ANTsImage, optional
        ROI mask defining the search space. If None, automatically computes the
        union bounding box of all atlas labels with padding.
    rad : int, default 2
        Local patch radius (box window width = 2 * rad + 1).
    beta : float, default 2.0
        Similarity sharpening exponent for patch-based weighting.
    rho : float, default 0.05
        Ridge regularization parameter added to the diagonal of the covariance matrix.
    mode : {"patch", "joint"}, default "joint"
        Fusion algorithm:
        - "patch": Fast diagonal LNCC patch-similarity weighting.
        - "joint": Full JLF solving the K x K pairwise error covariance system.
    nonnegative : bool, default True
        Whether to enforce non-negative weights (w >= 0).
    device : str, optional
        Compute device ('cpu', 'mps', 'cuda'). Defaults to auto-detection.
    verbose : bool, default False
        Whether to log timing and progress.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'segmentation': ants.ANTsImage of discrete consensus integer labels.
        - 'probability_images': dict[int, ants.ANTsImage] per-class probability maps.
        - 'consensus_envelope': ants.ANTsImage union binary envelope of candidate labels.
        - 'timing_s': Wall-clock runtime in seconds.
        - 'mode': 'patch' or 'joint'.
    """
    t0_start = time.time()
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

    if not roi_mask.any():
        # Fallback if no labels found
        zero_img = target_image.new_image_like(np.zeros(shape, dtype=np.uint32))
        return {
            "segmentation": zero_img,
            "probability_images": {0: target_image.new_image_like(np.ones(shape, dtype=np.float32))},
            "consensus_envelope": zero_img,
            "timing_s": 0.0,
            "mode": mode,
        }

    # Bounding box coordinates with padding = rad + 2
    pad = rad + 2
    idx = np.where(roi_mask)
    x_min, x_max = max(0, int(idx[0].min()) - pad), min(shape[0], int(idx[0].max()) + pad + 1)
    y_min, y_max = max(0, int(idx[1].min()) - pad), min(shape[1], int(idx[1].max()) + pad + 1)
    z_min, z_max = max(0, int(idx[2].min()) - pad), min(shape[2], int(idx[2].max()) + pad + 1)

    slicer = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))

    # Crop target and atlas subvolumes
    target_crop = target_arr[slicer]
    atlases_crop = [img.numpy()[slicer] for img in atlas_images]
    labels_crop = [lbl.numpy()[slicer] for lbl in atlas_labels]
    sub_shape = target_crop.shape

    # 2. Convert to PyTorch tensors
    dev = auto_detect_device(device)
    target_t = torch.from_numpy(target_crop).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=dev)
    atlases_t = [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=dev) for a in atlases_crop]
    labels_t = [torch.from_numpy(l).unsqueeze(0).unsqueeze(0).to(dtype=torch.long, device=dev) for l in labels_crop]

    # Setup separable filter kernels
    kernel_size = 2 * rad + 1
    dim = 3
    k = torch.ones(kernel_size, dtype=torch.float32, device=dev)
    kernels = [k] * dim
    kernel_vol = float(kernel_size ** dim)

    # 3. Compute weights
    t_sum = separable_1d_filter(target_t, kernels)
    t2_sum = separable_1d_filter(target_t * target_t, kernels)
    t_var = torch.clamp(t2_sum - t_sum * t_sum / kernel_vol, min=1e-4)

    lncc_list = []
    diff_list = [A - target_t for A in atlases_t]
    d_sq_list = []

    for i in range(K):
        A = atlases_t[i]
        p_sum = separable_1d_filter(A, kernels)
        p2_sum = separable_1d_filter(A * A, kernels)
        tp_sum = separable_1d_filter(target_t * A, kernels)
        cross = tp_sum - p_sum * t_sum / kernel_vol
        p_var = torch.clamp(p2_sum - p_sum * p_sum / kernel_vol, min=1e-4)
        ncc = cross / (torch.sqrt(t_var * p_var) + 1e-4)
        lncc_list.append(torch.clamp(ncc, min=0.0))
        if mode == "joint":
            d_sq = separable_1d_filter(diff_list[i] * diff_list[i], kernels) / kernel_vol
            d_sq_list.append(torch.clamp_min(d_sq, 1e-4))

    if mode == "joint":
        # Wang & Yushkevich (TPAMI 2013) Joint Error Model:
        # M_ij(x) = [ (1 - LNCC_i) * (1 - LNCC_j) ]^(beta/2) * ( (1 + Corr(D_i, D_j)) / 2 )
        err = [(torch.clamp(1.0 - ncc, min=0.01)) ** (beta / 2.0) for ncc in lncc_list]
        N_vox = sub_shape[0] * sub_shape[1] * sub_shape[2]
        M_mat = torch.zeros((K, K, *sub_shape), dtype=torch.float32, device=dev)

        for i in range(K):
            for j in range(i, K):
                if i == j:
                    corr = torch.ones_like(target_t[0, 0])
                else:
                    prod = separable_1d_filter(diff_list[i] * diff_list[j], kernels) / kernel_vol
                    raw_corr = prod / (torch.sqrt(d_sq_list[i] * d_sq_list[j]) + 1e-4)
                    corr = (torch.clamp(raw_corr[0, 0], min=-1.0, max=1.0) + 1.0) * 0.5
                m_ij = err[i][0, 0] * err[j][0, 0] * corr
                M_mat[i, j] = m_ij
                if i != j:
                    M_mat[j, i] = m_ij

        M_vox = M_mat.permute(2, 3, 4, 0, 1).reshape(N_vox, K, K)
        diag_mean = torch.diagonal(M_vox, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(K, dtype=torch.float32, device=dev).unsqueeze(0)
        M_reg = M_vox + (rho * torch.clamp_min(diag_mean, 1e-4) + 1e-5) * eye

        ones = torch.ones((N_vox, K, 1), dtype=torch.float32, device=dev)
        w_vox = torch.linalg.solve(M_reg, ones).squeeze(-1)

        if nonnegative:
            w_vox = torch.clamp_min(w_vox, 0.0)

        sum_w = torch.sum(w_vox, dim=-1, keepdim=True)
        zero_mask = sum_w < 1e-6
        uniform = torch.full_like(w_vox, 1.0 / K)
        w_vox = torch.where(zero_mask, uniform, w_vox / torch.clamp_min(sum_w, 1e-7))
        weights = w_vox.reshape(*sub_shape, K).permute(3, 0, 1, 2).unsqueeze(1) # (K, 1, *sub_shape)
    else:
        # Patch-based local cross-correlation similarity
        weights_list = [(ncc ** beta) + 1e-6 for ncc in lncc_list]
        weights = torch.cat(weights_list, dim=0) # (K, 1, *sub_shape)
        sum_w = torch.sum(weights, dim=0, keepdim=True)
        weights = weights / sum_w

    # 4. Multi-class probability voting
    max_label = max(int(l.max().item()) for l in labels_t)
    num_classes = max_label + 1
    prob_crop = torch.zeros((num_classes, *sub_shape), dtype=torch.float32, device=dev)

    for c in range(num_classes):
        c_votes = torch.zeros_like(target_t[0, 0])
        for i in range(K):
            c_mask = (labels_t[i][0, 0] == c).float()
            c_votes += weights[i, 0] * c_mask
        prob_crop[c] = c_votes

    final_crop = torch.argmax(prob_crop, dim=0).cpu().numpy().astype(np.uint32)
    prob_crop_np = prob_crop.cpu().numpy()

    # 5. Restore into full native image space
    final_arr = np.zeros(shape, dtype=np.uint32)
    final_arr[slicer] = final_crop

    # Envelope
    env_arr = np.zeros(shape, dtype=np.uint32)
    env_crop = (np.sum(prob_crop_np[1:], axis=0) > 0.05).astype(np.uint32)
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
    for c in range(num_classes):
        p_full = np.zeros(shape, dtype=np.float32)
        p_full[slicer] = prob_crop_np[c]
        prob_images[c] = ants.from_numpy(
            p_full,
            origin=target_image.origin,
            spacing=target_image.spacing,
            direction=target_image.direction,
        )

    t_total = time.time() - t0_start
    if verbose:
        print(f"[syntx.joint_label_fusion] Mode={mode}, K={K}, ROI={sub_shape}, Time={t_total:.2f}s")

    return {
        "segmentation": seg_img,
        "probability_images": prob_images,
        "consensus_envelope": env_img,
        "timing_s": float(t_total),
        "mode": mode,
    }
