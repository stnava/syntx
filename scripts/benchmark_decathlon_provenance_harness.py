#!/usr/bin/env python3
"""
scripts/benchmark_decathlon_provenance_harness.py
=================================================

Provenance-Oriented, No-Regression Medical Segmentation Decathlon (MSD) Evaluation Harness.
Conforms strictly to GEMINI.md Section 6, Section 5, and Section 2:
1. Whole-organ vs. focal lesion metrics:
   - Stratum A (6 organ tasks: Heart, Liver, Hippocampus, Prostate, Pancreas, Spleen):
     Evaluates organ parenchymal overlap (Sørensen-Dice) with combined parenchyma+tumor masks.
   - Stratum B (4 focal lesion tasks: BrainTumour, Lung, HepaticVessel, Colon):
     Evaluates surrogate whole-organ / parenchymal segmentation overlap (brain mask, bilateral lung cavity,
     abdominal soft tissue) and structural intensity correlation (3D LNCC), never scoring inter-subject
     alignment solely on focal pathology masks without anatomical co-localization.
2. Hold-out landmark Target Registration Error (TRE in mm) under controlled physical perturbations,
   evaluated strictly on disjoint hold-out points via syntx.landmarks.matcher.compute_tre (no FRE leakage).
3. Fair 4-Arm End-to-End Method Roster (canonical [100, 100, 20] deformable iterations):
   - ANTs Affine (ants.registration type_of_transform='Affine')
   - syntx Native Affine (portfolio tournament via whole-domain Mattes MI, zero test-label leakage)
   - ANTs SyN (ants.registration type_of_transform='SyNOnly' seeded from ANTs Affine, [100, 100, 20])
   - syntx SyN Beltrami (syntx.syn canonical beltrami: grad_step=0.7, beltrami_dilation=0.1, [100, 100, 20])
4. Registration QC & Light Theme:
   - Diffeomorphism integrity evaluated via Liouville determinant (syntx.liouville_determinant)
   - Symmetric consistency evaluated via inverse identity error scores
   - Light-theme visualizers (render_standard_4panel theme='light')
5. Comprehensive provenance & incremental checkpointing:
   - Live per-task JSON + Markdown updates, --merge-existing support for targeted task re-runs,
     and --smoke CLI gate (<60s).
"""

import os
import sys
import time
import json
import shutil
import datetime
import argparse
import tempfile
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import ants

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import syntx
from syntx.core.utils import normalize_image
from syntx.core.losses import mattes_mi_loss_nd
from syntx.landmarks import (
    preprocess_for_landmarks,
    is_ct_image,
    detect_sift3d,
    match_landmarks,
    ransac_filter,
)
from syntx.landmarks.matcher import compute_tre
from syntx.landmarks.optimal_transport import sampled_optimal_transport_affine
from syntx.landmarks import spatial as S
from syntx.robust_affine import robust_affine
from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.pyramid import cap_ants_image
from syntx.registration_metrics import evaluate_registration_qc
from syntx.liouville import liouville_determinant
from syntx.viz import render_standard_4panel
from syntx.data.surrogates import extract_brain_parenchyma, extract_ct_lung_parenchyma

DEFAULT_DECATHLON_DIR = os.environ.get("DECATHLON_DIR", os.path.expanduser("~/data/decathlon"))
OUT_JSON = "docs/provenance/decathlon_provenance_benchmark.json"
OUT_MD = "docs/reports/decathlon_provenance_benchmark.md"

# Classification of MSD Tasks per GEMINI.md Section 6 (Stratum A: Organ tasks, Stratum B: Focal tasks)
TASK_SPECS = [
    {
        "name": "Task01_BrainTumour",
        "anatomy": "Brain",
        "modality": "MRI",
        "is_ct": False,
        "is_organ_task": False,
        "stratum": "B",
        "label_description": "Focal Glioma (surrogate: whole-brain mask, T1w structural channel)",
    },
    {
        "name": "Task02_Heart",
        "anatomy": "Heart",
        "modality": "MRI",
        "is_ct": False,
        "is_organ_task": True,
        "stratum": "A",
        "label_description": "Left Atrium Myocardium/Cavity",
    },
    {
        "name": "Task03_Liver",
        "anatomy": "Liver",
        "modality": "CT",
        "is_ct": True,
        "is_organ_task": True,
        "stratum": "A",
        "label_description": "Liver Parenchyma + Lesions (labels 1+2)",
    },
    {
        "name": "Task04_Hippocampus",
        "anatomy": "Hippocampus",
        "modality": "MRI",
        "is_ct": False,
        "is_organ_task": True,
        "stratum": "A",
        "label_description": "Hippocampus Anterior/Posterior",
    },
    {
        "name": "Task05_Prostate",
        "anatomy": "Prostate",
        "modality": "MRI",
        "is_ct": False,
        "is_organ_task": True,
        "stratum": "A",
        "label_description": "Prostate Peripheral + Transition Zones (whole gland 1+2)",
    },
    {
        "name": "Task06_Lung",
        "anatomy": "Lung",
        "modality": "CT",
        "is_ct": True,
        "is_organ_task": False,
        "stratum": "B",
        "label_description": "Focal Lung Nodule (surrogate: bilateral lung parenchyma)",
    },
    {
        "name": "Task07_Pancreas",
        "anatomy": "Pancreas",
        "modality": "CT",
        "is_ct": True,
        "is_organ_task": True,
        "stratum": "A",
        "label_description": "Pancreas Parenchyma + Tumor (labels 1+2)",
    },
    {
        "name": "Task08_HepaticVessel",
        "anatomy": "Hepatic Vessel",
        "modality": "CT",
        "is_ct": True,
        "is_organ_task": False,
        "stratum": "B",
        "label_description": "Vessels & Tumor (surrogate: abdominal soft tissue)",
    },
    {
        "name": "Task09_Spleen",
        "anatomy": "Spleen",
        "modality": "CT",
        "is_ct": True,
        "is_organ_task": True,
        "stratum": "A",
        "label_description": "Spleen Parenchyma",
    },
    {
        "name": "Task10_Colon",
        "anatomy": "Colon",
        "modality": "CT",
        "is_ct": True,
        "is_organ_task": False,
        "stratum": "B",
        "label_description": "Focal Colon Cancer (surrogate: abdominal soft tissue)",
    },
]

TASK_ORDER = {spec["name"]: idx for idx, spec in enumerate(TASK_SPECS)}


def extract_surrogate_mask(
    img: ants.ANTsImage,
    task_spec: Dict[str, Any],
    raw_label: Optional[ants.ANTsImage] = None,
) -> ants.ANTsImage:
    """
    Extracts anatomical parenchymal mask adhering to Bug H5 task semantics:
    - If organ task with labeled organ:
      - Task03 (Liver) & Task07 (Pancreas): combine parenchyma + tumor (lbl == 1) | (lbl == 2)
      - Task05 (Prostate): combine peripheral + transition zones (lbl == 1) | (lbl == 2)
      - Other organ tasks (Heart, Hippocampus, Spleen): binarize organ mask (lbl > 0)
    - If focal lesion task:
      - Brain: Whole-brain intracranial mask using structural T1w channel via extract_brain_parenchyma(img, channel=1)
      - Lung: Bilateral lung cavity parenchyma via extract_ct_lung_parenchyma(img)
      - Abdomen (HepaticVessel, Colon): Soft-tissue foreground mask [-100, 300 HU]
    """
    t_name = task_spec["name"]
    if task_spec["is_organ_task"] and raw_label is not None:
        arr_lbl = raw_label.numpy()
        if t_name in ("Task03_Liver", "Task07_Pancreas", "Task05_Prostate"):
            comb = ((arr_lbl == 1) | (arr_lbl == 2)).astype(np.float32)
        else:
            comb = (arr_lbl > 0).astype(np.float32)
        return ants.from_numpy(comb, origin=raw_label.origin, spacing=raw_label.spacing, direction=raw_label.direction)

    # Focal lesion task surrogates
    if t_name == "Task01_BrainTumour":
        return extract_brain_parenchyma(img, channel=1)
    elif t_name == "Task06_Lung":
        return extract_ct_lung_parenchyma(img)
    else:
        # Abdominal CT (Task08_HepaticVessel, Task10_Colon)
        arr_img = img.numpy()
        if task_spec["is_ct"]:
            mask_arr = ((arr_img >= -100) & (arr_img <= 300)).astype(np.float32)
        else:
            mask_arr = ants.get_mask(img, cleanup=2).numpy().astype(np.float32)
        return ants.from_numpy(mask_arr, origin=img.origin, spacing=img.spacing, direction=img.direction)


def compute_structural_lncc(
    fi: ants.ANTsImage,
    warped_mi: ants.ANTsImage,
    mask: Optional[ants.ANTsImage] = None,
    window_size: int = 7,
) -> float:
    """Computes 3-D Local Normalized Cross Correlation (LNCC, 1e-6 variance floor per GEMINI.md §3)."""
    t_f = S.image_to_tensor(fi).to(dtype=torch.float32)
    t_m = S.image_to_tensor(warped_mi).to(dtype=torch.float32)
    while t_f.dim() < 5:
        t_f = t_f.unsqueeze(0)
    while t_m.dim() < 5:
        t_m = t_m.unsqueeze(0)

    k = int(window_size)
    pad = k // 2
    with torch.no_grad():
        mu_f = F.avg_pool3d(t_f, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        mu_m = F.avg_pool3d(t_m, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        mu_ff = F.avg_pool3d(t_f * t_f, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        mu_mm = F.avg_pool3d(t_m * t_m, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        mu_fm = F.avg_pool3d(t_f * t_m, kernel_size=k, stride=1, padding=pad, count_include_pad=False)

        var_f = torch.clamp(mu_ff - mu_f * mu_f, min=0.0)
        var_m = torch.clamp(mu_mm - mu_m * mu_m, min=0.0)
        cov = mu_fm - mu_f * mu_m

        active = (var_f > 1e-6) | (var_m > 1e-6)
        if mask is not None:
            t_mask = S.image_to_tensor(mask).reshape_as(active) > 0.5
            active = active & t_mask

        if int(active.sum().item()) < 16:
            return 0.0

        denom = torch.sqrt(torch.clamp(var_f, min=1e-6) * torch.clamp(var_m, min=1e-6))
        local_corr = torch.clamp(cov / denom, min=-1.0, max=1.0)
        both_var = (var_f > 1e-6) & (var_m > 1e-6)
        local_corr = torch.where(both_var, local_corr, torch.zeros_like(local_corr))
        corr = float(local_corr[active].mean().item())
    return max(-1.0, min(1.0, corr))


def score_affine_candidate_mattes(
    eval_fixed_norm: ants.ANTsImage,
    moving_norm: ants.ANTsImage,
    moving_ones: ants.ANTsImage,
    expected_ov: float,
    tx_list: List[str],
) -> float:
    """Scores an affine candidate via whole-domain Mattes MI (GEMINI.md §2 & §3) + slab-overlap guard."""
    flags = [False] * len(tx_list)
    w_img = ants.apply_transforms(eval_fixed_norm, moving_norm, transformlist=tx_list, whichtoinvert=flags)
    w_in = ants.apply_transforms(eval_fixed_norm, moving_ones, transformlist=tx_list, whichtoinvert=flags)
    in_frac = float((w_in.numpy() > 0.5).mean())
    t0 = S.image_to_tensor(eval_fixed_norm).reshape(-1)
    tw = S.image_to_tensor(w_img).reshape(-1)
    with torch.no_grad():
        mi_loss = float(
            mattes_mi_loss_nd(tw, t0, mask=None, num_bins=32, auto_mask=False, fixed_range=(0.0, 1.0)).item()
        )
    ov_deficit = max(0.0, 0.55 * expected_ov - in_frac)
    return -mi_loss - 2.0 * ov_deficit


def evaluate_decathlon_task(
    task_spec: Dict[str, Any],
    decathlon_dir: str,
    pair_idx: int = 0,
    pair_indices: Optional[Tuple[int, int]] = None,
    device: str = "mps",
    return_visuals: bool = False,
    smoke: bool = False,
) -> Optional[Dict[str, Any]]:
    """Evaluates a single Decathlon task pair across all 4 registration arms."""
    task_name = task_spec["name"]
    t_dir = os.path.join(decathlon_dir, task_name)
    if not os.path.isdir(t_dir):
        return None

    meta_path = os.path.join(t_dir, "dataset.json")
    if not os.path.isfile(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    training = meta.get("training", [])
    if len(training) < 2:
        return None

    n_tr = len(training)
    if pair_indices is not None:
        idx0, idx1 = pair_indices
    else:
        if 2 * pair_idx + 1 < n_tr:
            idx0, idx1 = 2 * pair_idx, 2 * pair_idx + 1
        else:
            idx0 = (2 * pair_idx) % n_tr
            idx1 = (2 * pair_idx + 1) % n_tr
            if idx0 == idx1:
                idx1 = (idx0 + 1) % n_tr

    c0 = training[idx0]
    c1 = training[idx1]
    p0 = os.path.join(t_dir, c0["image"] if isinstance(c0, dict) else c0)
    p1 = os.path.join(t_dir, c1["image"] if isinstance(c1, dict) else c1)
    l0_p = os.path.join(t_dir, c0["label"]) if isinstance(c0, dict) and "label" in c0 else None
    l1_p = os.path.join(t_dir, c1["label"]) if isinstance(c1, dict) and "label" in c1 else None

    # 1. Load Images & Squeeze 4D if multi-channel using dataset.json modality metadata (zero task-name hardcoding)
    img0_raw = ants.image_read(p0)
    img1_raw = ants.image_read(p1)

    mod_map = meta.get("modality", {}) if isinstance(meta.get("modality"), dict) else {}
    ch_sel = next((int(k) for k, v in mod_map.items() if str(v).strip().upper() in ("T1W", "T1")), 0)

    if img0_raw.dimension == 4:
        ch0 = min(ch_sel, img0_raw.shape[-1] - 1)
        img0_raw = ants.from_numpy(
            img0_raw.numpy()[..., ch0],
            origin=img0_raw.origin[:3],
            spacing=img0_raw.spacing[:3],
            direction=img0_raw.direction[:3, :3],
        )
    if img1_raw.dimension == 4:
        ch1 = min(ch_sel, img1_raw.shape[-1] - 1)
        img1_raw = ants.from_numpy(
            img1_raw.numpy()[..., ch1],
            origin=img1_raw.origin[:3],
            spacing=img1_raw.spacing[:3],
            direction=img1_raw.direction[:3, :3],
        )

    raw_l0 = ants.image_read(l0_p) if (l0_p and os.path.exists(l0_p)) else None
    raw_l1 = ants.image_read(l1_p) if (l1_p and os.path.exists(l1_p)) else None
    if raw_l0 is not None and raw_l0.dimension == 4:
        raw_l0 = ants.from_numpy(
            raw_l0.numpy()[..., 0],
            origin=raw_l0.origin[:3],
            spacing=raw_l0.spacing[:3],
            direction=raw_l0.direction[:3, :3],
        )
    if raw_l1 is not None and raw_l1.dimension == 4:
        raw_l1 = ants.from_numpy(
            raw_l1.numpy()[..., 0],
            origin=raw_l1.origin[:3],
            spacing=raw_l1.spacing[:3],
            direction=raw_l1.direction[:3, :3],
        )

    # Bug H5: Generate verified parenchymal masks on native full-resolution images
    mask0_native = extract_surrogate_mask(img0_raw, task_spec, raw_l0)
    mask1_native = extract_surrogate_mask(img1_raw, task_spec, raw_l1)

    # Bug H2: Working image downsampling using cap_ants_image with deformable policy preserving native z-spacing
    # Ground-truth labels are NEVER downsampled.
    img0 = img0_raw
    img1 = img1_raw
    if np.prod(img0.shape) > 5_000_000:
        img0 = cap_ants_image(img0, max_resolution_mm=1.5, policy="deformable")
    if np.prod(img1.shape) > 5_000_000:
        img1 = cap_ants_image(img1, max_resolution_mm=1.5, policy="deformable")

    # Autonomous image-driven modality & anatomy diagnosis via syntx.diagnose_pair + synthesize_policy (zero task-name hints)
    pair_diag = syntx.diagnose_pair(img0, img1, fast=False)
    auto_policy = syntx.synthesize_policy(pair_diag)
    diag_is_ct = bool(pair_diag.fixed.is_ct() and pair_diag.moving.is_ct())
    ct_window = None
    if diag_is_ct and auto_policy.ct_window is not None:
        ct_window = "lung" if auto_policy.ct_window[0] <= -500.0 else "soft_tissue"

    pre0 = preprocess_for_landmarks(img0, is_ct=diag_is_ct, ct_window=ct_window)
    pre1 = preprocess_for_landmarks(img1, is_ct=diag_is_ct, ct_window=ct_window)

    # 2. Initial Unaligned Baseline Overlap on Native Labels (Bug H1 & H2)
    _, _, dice_init = compute_bidirectional_dice(mask0_native, mask1_native, img0_raw, img1_raw, [], [], [])
    warped_init_img = ants.apply_transforms(pre0, pre1, transformlist=[])
    lncc_init = compute_structural_lncc(pre0, warped_init_img, mask=None)

    # 3. Bug H4: Controlled Ground-Truth Landmark TRE (Strictly Disjoint Hold-Out Points)
    ctr = np.array(ants.get_center_of_mass(pre0))
    a1, a2 = np.deg2rad(7.5), np.deg2rad(4.0)
    R_gt = (
        np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    )
    t_gt = np.array([3.0, -2.5, 1.5])
    tx_gt = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx_gt.set_parameters(np.concatenate([R_gt.ravel(), t_gt]))
    tx_gt.set_fixed_parameters(ctr)
    img_gt_warped = tx_gt.apply_to_image(pre0, pre0)

    kpts0_tre, desc0_tre = detect_sift3d(pre0, preprocess=False, max_keypoints=300)
    kpts_gt, desc_gt = detect_sift3d(img_gt_warped, preprocess=False, max_keypoints=300)
    m_gt = match_landmarks(kpts0_tre, kpts_gt, desc0_tre, desc_gt, ratio_thresh=0.85, mutual=True)

    tre_median = -1.0
    tre_mean = -1.0
    if len(m_gt) >= 8:
        m_fit = m_gt[::2]
        m_hold = m_gt[1::2]
        mf_gt, M_gt = ransac_filter(kpts0_tre, kpts_gt, m_fit, model="rigid", inlier_thresh_mm=3.0)
        if len(mf_gt) >= 4 and len(m_hold) > 0:
            R_inv = np.linalg.inv(R_gt)
            pts_hold_f0 = kpts0_tre[m_hold[:, 0], :3]
            pts_true = (R_inv @ (pts_hold_f0 - ctr - t_gt).T).T + ctr
            pts_pred = (M_gt[:3, :3] @ pts_hold_f0.T).T + M_gt[:3, 3]
            tre_mean = compute_tre(pts_true, pts_pred)
            tre_vals = np.linalg.norm(pts_pred - pts_true, axis=1)
            tre_median = float(np.median(tre_vals))
    elif len(m_gt) >= 4:
        mf_gt, M_gt = ransac_filter(kpts0_tre, kpts_gt, m_gt, model="rigid", inlier_thresh_mm=3.0)
        if len(mf_gt) >= 4:
            R_inv = np.linalg.inv(R_gt)
            used_idx = set(mf_gt[:, 0])
            holdout_idx = [i for i in range(len(kpts0_tre)) if i not in used_idx]
            if holdout_idx:
                pts_hold_f0 = kpts0_tre[holdout_idx, :3]
                pts_true = (R_inv @ (pts_hold_f0 - ctr - t_gt).T).T + ctr
                pts_pred = (M_gt[:3, :3] @ pts_hold_f0.T).T + M_gt[:3, 3]
                tre_mean = compute_tre(pts_true, pts_pred)
                tre_vals = np.linalg.norm(pts_pred - pts_true, axis=1)
                tre_median = float(np.median(tre_vals))

    # 4. Multi-Start Tournament Portfolio (Bug H3: zero test-label leakage, whole-domain Mattes MI per GEMINI.md §2 & §3)
    tx_id = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx_id.set_parameters(np.concatenate([np.eye(3).ravel(), np.zeros(3)]))
    tx_id.set_fixed_parameters(np.array(ants.get_center_of_mass(pre0)))
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f_id:
        id_mat = f_id.name
    ants.write_transform(tx_id, id_mat)

    t0_aff = time.time()
    res_aff_default = robust_affine(pre0, pre1, mode="auto", device=device, verbose=False)
    t_aff_default = time.time() - t0_aff

    # Candidate C: SIFT3D + RANSAC keypoint candidate (consistent scale range on both volumes)
    sift_mat = None
    res_aff_sift = None
    if not smoke:
        try:
            kpts0_p, desc0_p = detect_sift3d(pre0, preprocess=False, max_keypoints=500, sigma_min=1.5, sigma_max=6.0)
            kpts1_p, desc1_p = detect_sift3d(pre1, preprocess=False, max_keypoints=500, sigma_min=1.5, sigma_max=6.0)
            m_pair = match_landmarks(kpts0_p, kpts1_p, desc0_p, desc1_p, ratio_thresh=0.90, mutual=True)
            mf_pair, M_pair = ransac_filter(kpts0_p, kpts1_p, m_pair, model="rigid", inlier_thresh_mm=15.0)
            if len(mf_pair) >= 4:
                tx_sift = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
                tx_sift.set_parameters(np.concatenate([M_pair[:3, :3].ravel(), M_pair[:3, 3]]))
                tx_sift.set_fixed_parameters(np.zeros(3))
                with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp_s:
                    sift_mat = tmp_s.name
                ants.write_transform(tx_sift, sift_mat)
                res_aff_sift = robust_affine(pre0, pre1, mode="auto", initial_transform=sift_mat, device=device, verbose=False)
        except Exception:
            pass

    # Candidate D: Continuous Soft Optimal Transport (Sinkhorn OT)
    ot_mat = None
    res_aff_ot = None
    if not smoke:
        try:
            ot_mat = sampled_optimal_transport_affine(pre0, pre1, sampling_percentage=0.08, device=device)
            res_aff_ot = robust_affine(pre0, pre1, mode="auto", initial_transform=ot_mat, device=device, verbose=False)
        except Exception:
            pass

    n0 = normalize_image(pre0, method="auto", force=True)
    n1 = normalize_image(pre1, method="auto", force=True)
    eval_fixed = n0
    ones1 = ants.from_numpy(
        np.ones(pre1.shape, dtype=np.float32),
        origin=pre1.origin,
        spacing=pre1.spacing,
        direction=pre1.direction,
    )
    ext0 = np.abs(np.asarray(eval_fixed.direction).reshape(3, 3)) @ (
        np.asarray(eval_fixed.shape, dtype=np.float64) * np.asarray(eval_fixed.spacing, dtype=np.float64)
    )
    ext1 = np.abs(np.asarray(pre1.direction).reshape(3, 3)) @ (
        np.asarray(pre1.shape, dtype=np.float64) * np.asarray(pre1.spacing, dtype=np.float64)
    )
    expected_ov = float(np.clip(np.prod(np.minimum(1.0, ext1 / np.maximum(ext0, 1e-3))), 0.05, 1.0))

    def _score_candidate(res_dict: Dict[str, Any]) -> float:
        return score_affine_candidate_mattes(eval_fixed, n1, ones1, expected_ov, res_dict["fwdtransforms"])

    candidates = [
        ("Identity", {"fwdtransforms": [id_mat], "invtransforms": [id_mat]}, _score_candidate({"fwdtransforms": [id_mat]})),
        ("Default", res_aff_default, _score_candidate(res_aff_default)),
    ]
    if res_aff_sift is not None:
        candidates.append(("SIFT3D", res_aff_sift, _score_candidate(res_aff_sift)))
    if res_aff_ot is not None:
        candidates.append(("Sinkhorn_OT", res_aff_ot, _score_candidate(res_aff_ot)))

    winning_name, winning_aff, _ = max(candidates, key=lambda c: c[2])

    # Post-hoc evaluation of winning native affine on native full-resolution labels (Bug H1: whichtoinvert_inv=[True])
    _, _, dice_syntx_aff = compute_bidirectional_dice(
        mask0_native,
        mask1_native,
        img0_raw,
        img1_raw,
        winning_aff["fwdtransforms"],
        winning_aff["invtransforms"],
        whichtoinvert_inv=[True],
    )
    warped_aff_img = ants.apply_transforms(pre0, pre1, winning_aff["fwdtransforms"], whichtoinvert=[False])
    lncc_aff = compute_structural_lncc(pre0, warped_aff_img, mask=None)

    # 5a. Autonomous syntx.auto_reg deformable registration (diagnose_pair -> synthesize_policy -> scale-space Beltrami SyN)
    reg_iters = [10, 5] if smoke else None
    t0_syn_belt = time.time()
    res_syn_beltrami = syntx.auto_reg(
        fixed=img0,
        moving=img1,
        type_of_transform="SyN",
        guided=False,
        initial_transform=winning_aff["fwdtransforms"][0],
        regularizer="beltrami",
        grad_step=0.7,
        dilatation_weight=0.1,
        reg_iterations=reg_iters,
        device=device,
        verbose=False,
    )
    t_syn_belt = time.time() - t0_syn_belt
    _, _, dice_syntx_syn = compute_bidirectional_dice(
        mask0_native,
        mask1_native,
        img0_raw,
        img1_raw,
        res_syn_beltrami["fwdtransforms"],
        res_syn_beltrami["invtransforms"],
        whichtoinvert_inv=[True, False],
    )
    warped_def_img = ants.apply_transforms(pre0, pre1, res_syn_beltrami["fwdtransforms"], whichtoinvert=[False, False])
    lncc_def = compute_structural_lncc(pre0, warped_def_img, mask=None)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Restore default multi-core ITK threading before running external ANTs C++ baselines (GEMINI.md §4)
    os.environ.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)

    # 5b. ANTs Affine reference baseline
    t0_ants_aff = time.time()
    res_ants_aff = ants.registration(fixed=pre0, moving=pre1, type_of_transform="Affine")
    t_ants_aff = time.time() - t0_ants_aff
    _, _, dice_ants_aff = compute_bidirectional_dice(
        mask0_native,
        mask1_native,
        img0_raw,
        img1_raw,
        res_ants_aff["fwdtransforms"],
        res_ants_aff["invtransforms"],
        whichtoinvert_inv=[True],
    )

    # 5c. ANTs SyN reference baseline (SyNOnly seeded from ANTs's own Affine, matched [100, 100, 20] iterations)
    ants_iters = [10, 5, 0] if smoke else [100, 100, 20]
    t0_ants_syn = time.time()
    res_ants_syn = ants.registration(
        fixed=pre0,
        moving=pre1,
        type_of_transform="SyNOnly",
        initial_transform=res_ants_aff["fwdtransforms"][0],
        reg_iterations=ants_iters,
    )
    t_ants_syn = time.time() - t0_ants_syn
    _, _, dice_ants_syn = compute_bidirectional_dice(
        mask0_native,
        mask1_native,
        img0_raw,
        img1_raw,
        res_ants_syn["fwdtransforms"],
        res_ants_syn["invtransforms"],
        whichtoinvert_inv=[True, False],
    )

    # 6. Registration QC: Liouville determinant and inverse identity error scores (Bug H6)
    qc_report = evaluate_registration_qc(
        fixed=pre0,
        moving=pre1,
        registration_result=res_syn_beltrami,
        fixed_labels=mask0_native,
        moving_labels=mask1_native,
    )
    folding_pct = qc_report.folding_pct
    min_jac = qc_report.min_jacobian
    inv_err_mean = qc_report.inverse_consistency_error_mean_mm or 0.0
    inv_err_max = qc_report.inverse_consistency_error_max_mm or 0.0
    is_diffeomorphic = bool(qc_report.is_diffeomorphic)

    # 7. Light-Theme Visualizations (Bug H6 & GEMINI.md Section 5)
    visual_payload = None
    if return_visuals:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fig:
                fig_path = tmp_fig.name
            render_standard_4panel(
                fixed=pre0,
                warped=res_syn_beltrami,
                theme="light",
                filename=fig_path,
            )
            visual_payload = {"fig_path": fig_path, "task_name": task_name}
        except Exception:
            visual_payload = None

    # Cleanup temporary transforms
    all_txs = (
        res_aff_default.get("fwdtransforms", [])
        + res_syn_beltrami.get("fwdtransforms", [])
        + res_syn_beltrami.get("invtransforms", [])
        + res_ants_aff.get("fwdtransforms", [])
        + res_ants_aff.get("invtransforms", [])
        + res_ants_syn.get("fwdtransforms", [])
        + res_ants_syn.get("invtransforms", [])
    )
    if res_aff_sift:
        all_txs += res_aff_sift.get("fwdtransforms", [])
    if res_aff_ot:
        all_txs += res_aff_ot.get("fwdtransforms", [])
    for p in (id_mat, sift_mat, ot_mat):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    for tx in all_txs:
        if os.path.exists(tx) and (tx.endswith(".mat") or tx.endswith(".nii.gz")):
            try:
                os.remove(tx)
            except Exception:
                pass

    out_res = {
        "task_name": task_name,
        "stratum": task_spec["stratum"],
        "anatomy": task_spec["anatomy"],
        "modality": task_spec["modality"],
        "is_organ_task": task_spec["is_organ_task"],
        "label_description": task_spec["label_description"],
        "pair_idx": pair_idx,
        "cases": [os.path.basename(p0), os.path.basename(p1)],
        "dice_initial": float(dice_init),
        "dice_ants_affine": float(dice_ants_aff),
        "time_ants_affine_s": float(t_ants_aff),
        "dice_syntx_affine": float(dice_syntx_aff),
        "time_syntx_affine_s": float(t_aff_default),
        "winning_candidate": winning_name,
        "dice_ants_syn": float(dice_ants_syn),
        "time_ants_syn_s": float(t_ants_syn + t_ants_aff),
        "dice_syntx_syn": float(dice_syntx_syn),
        "time_syntx_syn_s": float(t_syn_belt + t_aff_default),
        "gain_syntx_affine": float(dice_syntx_aff - dice_init),
        "gain_syntx_syn": float(dice_syntx_syn - dice_init),
        "gain_ants_syn": float(dice_ants_syn - dice_init),
        "delta_vs_ants_syn": float(dice_syntx_syn - dice_ants_syn),
        "lncc_initial": float(lncc_init),
        "lncc_affine": float(lncc_aff),
        "lncc_deformable": float(lncc_def),
        "landmark_tre_mm": float(tre_median),
        "landmark_tre_mean_mm": float(tre_mean),
        "folding_pct": float(folding_pct),
        "min_jacobian": float(min_jac),
        "inverse_identity_mean_error_mm": float(inv_err_mean),
        "inverse_identity_max_error_mm": float(inv_err_max),
        "is_diffeomorphic": is_diffeomorphic,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if visual_payload is not None:
        out_res["visuals"] = visual_payload
    return out_res


def compute_cohort_stats(values: List[float]) -> Dict[str, Any]:
    """Computes mean, standard deviation, median, and 95% confidence interval."""
    vals = np.array(values, dtype=np.float64)
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    median = float(np.median(vals))
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if n >= 2 and std > 1e-9:
        try:
            import scipy.stats
            t_crit = float(scipy.stats.t.ppf(0.975, df=n - 1))
            margin = t_crit * (std / np.sqrt(n))
        except Exception:
            margin = 1.96 * (std / np.sqrt(n))
        ci95 = [float(mean - margin), float(mean + margin)]
    else:
        ci95 = [mean, mean]
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": vmin,
        "max": vmax,
        "ci95": ci95,
    }


def compute_paired_pvalue(d_initial: List[float], d_final: List[float]) -> float:
    """Computes paired statistical test p-value comparing initial vs deformable Dice."""
    if len(d_initial) < 2:
        return 1.0
    import scipy.stats
    diffs = np.array(d_final) - np.array(d_initial)
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        if len(diffs) >= 5:
            _, p = scipy.stats.wilcoxon(d_final, d_initial, alternative="greater")
        else:
            _, p = scipy.stats.ttest_rel(d_final, d_initial, alternative="greater")
        return float(p)
    except Exception:
        return 1.0


def evaluate_decathlon_cohort(
    task_spec: Dict[str, Any],
    decathlon_dir: str,
    num_pairs: int = 1,
    device: str = "mps",
    return_visuals: bool = False,
    smoke: bool = False,
) -> Optional[Dict[str, Any]]:
    """Evaluates multiple pairs across a task, computing statistical power metrics."""
    pairs = []
    visuals = None
    for p_idx in range(num_pairs):
        print(f"  [Pair {p_idx + 1}/{num_pairs}] Running registration...", flush=True)
        res = evaluate_decathlon_task(
            task_spec,
            decathlon_dir,
            pair_idx=p_idx,
            device=device,
            return_visuals=(return_visuals and p_idx == 0),
            smoke=smoke,
        )
        if res is None:
            break
        if "visuals" in res:
            visuals = res.pop("visuals")
        pairs.append(res)
        print(
            f"    Pair {p_idx + 1} Result: Init={res['dice_initial']:.4f} -> "
            f"ANTsAff={res['dice_ants_affine']:.4f} | syntxAff={res['dice_syntx_affine']:.4f} ({res['winning_candidate']}) | "
            f"ANTsSyN={res['dice_ants_syn']:.4f} | syntxSyN={res['dice_syntx_syn']:.4f} "
            f"(ΔANTs: {res['dice_syntx_syn'] - res['dice_ants_syn']:+.4f}) | TRE={res['landmark_tre_mm']:.3f} mm",
            flush=True,
        )

    if not pairs:
        return None

    if num_pairs == 1:
        out = dict(pairs[0])
        out["num_pairs"] = 1
        out["pairs"] = pairs
        if visuals:
            out["visuals"] = visuals
        return out

    d_inits = [p["dice_initial"] for p in pairs]
    d_ants_affs = [p["dice_ants_affine"] for p in pairs]
    d_syntx_affs = [p["dice_syntx_affine"] for p in pairs]
    d_ants_syns = [p["dice_ants_syn"] for p in pairs]
    d_syntx_syns = [p["dice_syntx_syn"] for p in pairs]
    gains_syntx = [p["gain_syntx_syn"] for p in pairs]
    tres = [p["landmark_tre_mm"] for p in pairs if p["landmark_tre_mm"] > 0]
    folds = [p["folding_pct"] for p in pairs]
    min_jacs = [p["min_jacobian"] for p in pairs]
    inv_errs = [p["inverse_identity_mean_error_mm"] for p in pairs]

    stats_init = compute_cohort_stats(d_inits)
    stats_ants_aff = compute_cohort_stats(d_ants_affs)
    stats_syntx_aff = compute_cohort_stats(d_syntx_affs)
    stats_ants_syn = compute_cohort_stats(d_ants_syns)
    stats_syntx_syn = compute_cohort_stats(d_syntx_syns)
    stats_gain = compute_cohort_stats(gains_syntx)
    stats_tre = compute_cohort_stats(tres) if tres else {"mean": 0.0, "std": 0.0, "median": 0.0, "ci95": [0.0, 0.0]}

    p_val = compute_paired_pvalue(d_inits, d_syntx_syns)
    win_rate = float(np.mean([g >= 0 for g in gains_syntx]) * 100.0)
    wins_vs_ants = int(sum(1 for ds, da in zip(d_syntx_syns, d_ants_syns) if ds >= da - 0.002))

    winners = [p["winning_candidate"] for p in pairs]
    top_winner = max(set(winners), key=winners.count)

    out = {
        "task_name": task_spec["name"],
        "stratum": task_spec["stratum"],
        "anatomy": task_spec["anatomy"],
        "modality": task_spec["modality"],
        "is_organ_task": task_spec["is_organ_task"],
        "label_description": task_spec["label_description"],
        "num_pairs": len(pairs),
        "winning_candidate": top_winner,
        "dice_initial": stats_init["mean"],
        "dice_initial_std": stats_init["std"],
        "dice_ants_affine": stats_ants_aff["mean"],
        "dice_ants_affine_std": stats_ants_aff["std"],
        "dice_syntx_affine": stats_syntx_aff["mean"],
        "dice_syntx_affine_std": stats_syntx_aff["std"],
        "dice_ants_syn": stats_ants_syn["mean"],
        "dice_ants_syn_std": stats_ants_syn["std"],
        "dice_syntx_syn": stats_syntx_syn["mean"],
        "dice_syntx_syn_std": stats_syntx_syn["std"],
        "delta_vs_ants_syn": stats_syntx_syn["mean"] - stats_ants_syn["mean"],
        "wins_vs_ants": f"{wins_vs_ants}/{len(pairs)}",
        "gain_syntx_affine": stats_syntx_aff["mean"] - stats_init["mean"],
        "gain_syntx_syn": stats_gain["mean"],
        "gain_syntx_syn_std": stats_gain["std"],
        "gain_syntx_syn_ci95": stats_gain["ci95"],
        "landmark_tre_mm": stats_tre["mean"],
        "landmark_tre_std": stats_tre["std"],
        "folding_pct": float(np.mean(folds)),
        "min_jacobian": float(np.min(min_jacs)),
        "inverse_identity_mean_error_mm": float(np.mean(inv_errs)),
        "win_rate": win_rate,
        "p_value": p_val,
        "pairs": pairs,
    }
    if visuals:
        out["visuals"] = visuals
    return out


def _write_provenance_outputs(
    results: List[Dict[str, Any]],
    output_json: str,
    output_md: str,
    device: str,
    num_pairs: int,
    smoke: bool,
) -> None:
    """Atomically writes sorted structured JSON provenance and Stratum A / B Markdown report."""
    ordered = sorted(results, key=lambda r: TASK_ORDER.get(r.get("task_name", ""), 999))
    save_results = []
    for r in ordered:
        r_copy = dict(r)
        r_copy.pop("visuals", None)
        if "pairs" in r_copy:
            clean_pairs = []
            for p in r_copy["pairs"]:
                p_c = dict(p)
                p_c.pop("visuals", None)
                clean_pairs.append(p_c)
            r_copy["pairs"] = clean_pairs
        save_results.append(r_copy)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    tmp_json = output_json + ".tmp"
    with open(tmp_json, "w") as f:
        json.dump(save_results, f, indent=2)
    os.replace(tmp_json, output_json)

    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    lines_md = [
        "# Medical Segmentation Decathlon (MSD) Provenance & Evaluation Report\n\n",
        f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n",
        f"**Device**: `{device}`  \n",
        f"**Cohort Scale**: `{num_pairs} pair(s) per task` (`{len(ordered)}/{len(TASK_SPECS)}` tasks recorded)  \n",
        f"**Deformable Schedule**: `{'[10, 5]' if smoke else '[100, 100, 20]'}`  \n",
        f"**Smoke Mode**: `{smoke}`  \n",
        "**Standard**: GEMINI.md Section 6 (Whole-Organ vs Focal Lesion Separation & Hold-Out Landmark TRE)\n\n",
    ]

    res_stratum_a = [r for r in ordered if r.get("stratum") == "A"]
    if res_stratum_a:
        lines_md.extend([
            "## Stratum A: Anatomical Organ Alignment (6 Tasks)\n\n",
            "Evaluates whole-organ parenchymal overlap (Sørensen-Dice) with combined parenchymal and lesion coverage.\n\n",
            "| Task Name | Anatomy | Modality | Init Dice | ANTs Affine | syntx Native Affine | Winner | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |\n",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        ])
        for r in res_stratum_a:
            delta_ants = r["dice_syntx_syn"] - r["dice_ants_syn"]
            lines_md.append(
                f"| **{r['task_name']}** | {r['anatomy']} | {r['modality']} | "
                f"{r['dice_initial']:.4f} | {r['dice_ants_affine']:.4f} | {r['dice_syntx_affine']:.4f} | "
                f"**{r['winning_candidate']}** | {r['dice_ants_syn']:.4f} | **{r['dice_syntx_syn']:.4f}** | "
                f"**{delta_ants:+.4f}** | **{r['gain_syntx_syn']:+.4f}** | "
                f"{r['landmark_tre_mm']:.3f} mm | {r['folding_pct']:.4f}% | {r['min_jacobian']:.4f} |\n"
            )
        lines_md.append("\n")

    res_stratum_b = [r for r in ordered if r.get("stratum") == "B"]
    if res_stratum_b:
        lines_md.extend([
            "## Stratum B: Surrogate Parenchymal Alignment for Focal Pathology (4 Tasks)\n\n",
            "Evaluates surrogate parenchymal overlap (brain mask, lung cavity, abdominal soft tissue) and structural correlation.\n\n",
            "| Task Name | Anatomy | Modality | Surrogate Definition | Init Dice | ANTs Affine | syntx Native Affine | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |\n",
            "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        ])
        for r in res_stratum_b:
            delta_ants = r["dice_syntx_syn"] - r["dice_ants_syn"]
            lines_md.append(
                f"| **{r['task_name']}** | {r['anatomy']} | {r['modality']} | {r['label_description'].split('(')[-1].rstrip(')')} | "
                f"{r['dice_initial']:.4f} | {r['dice_ants_affine']:.4f} | {r['dice_syntx_affine']:.4f} | "
                f"{r['dice_ants_syn']:.4f} | **{r['dice_syntx_syn']:.4f}** | **{delta_ants:+.4f}** | "
                f"**{r['gain_syntx_syn']:+.4f}** | {r['landmark_tre_mm']:.3f} mm | {r['folding_pct']:.4f}% | {r['min_jacobian']:.4f} |\n"
            )
        lines_md.append("\n")

    lines_md.append("## Stratum Summaries & Power Analysis\n\n")
    if res_stratum_a:
        mean_init_a = float(np.mean([r["dice_initial"] for r in res_stratum_a]))
        mean_ants_aff_a = float(np.mean([r["dice_ants_affine"] for r in res_stratum_a]))
        mean_aff_a = float(np.mean([r["dice_syntx_affine"] for r in res_stratum_a]))
        mean_ants_syn_a = float(np.mean([r["dice_ants_syn"] for r in res_stratum_a]))
        mean_syn_a = float(np.mean([r["dice_syntx_syn"] for r in res_stratum_a]))
        mean_gain_a = float(np.mean([r["gain_syntx_syn"] for r in res_stratum_a]))
        tres_a = [r["landmark_tre_mm"] for r in res_stratum_a if r["landmark_tre_mm"] > 0]
        mean_tre_a = float(np.mean(tres_a)) if tres_a else 0.0
        lines_md.extend([
            "### Stratum A Summary (Anatomical Organ Tasks):\n",
            f"- **Mean Initial Dice**: `{mean_init_a:.4f}`\n",
            f"- **Mean Affine Dice**: `syntx` **`{mean_aff_a:.4f}`** vs `ANTs` `{mean_ants_aff_a:.4f}` (`{mean_aff_a - mean_ants_aff_a:+.4f}`)\n",
            f"- **Mean Deformable Dice**: `syntx SyN Beltrami` **`{mean_syn_a:.4f}`** vs `ANTs SyN` `{mean_ants_syn_a:.4f}` (**`{mean_syn_a - mean_ants_syn_a:+.4f}`**, net gain **`{mean_gain_a:+.4f}`**)\n",
            f"- **Mean Hold-Out Target Registration Error**: **`{mean_tre_a:.3f} mm`**\n\n",
        ])
    if res_stratum_b:
        mean_init_b = float(np.mean([r["dice_initial"] for r in res_stratum_b]))
        mean_ants_aff_b = float(np.mean([r["dice_ants_affine"] for r in res_stratum_b]))
        mean_aff_b = float(np.mean([r["dice_syntx_affine"] for r in res_stratum_b]))
        mean_ants_syn_b = float(np.mean([r["dice_ants_syn"] for r in res_stratum_b]))
        mean_syn_b = float(np.mean([r["dice_syntx_syn"] for r in res_stratum_b]))
        mean_gain_b = float(np.mean([r["gain_syntx_syn"] for r in res_stratum_b]))
        tres_b = [r["landmark_tre_mm"] for r in res_stratum_b if r["landmark_tre_mm"] > 0]
        mean_tre_b = float(np.mean(tres_b)) if tres_b else 0.0
        lines_md.extend([
            "### Stratum B Summary (Focal Lesion Tasks with Surrogates):\n",
            f"- **Mean Initial Surrogate Dice**: `{mean_init_b:.4f}`\n",
            f"- **Mean Affine Dice**: `syntx` **`{mean_aff_b:.4f}`** vs `ANTs` `{mean_ants_aff_b:.4f}` (`{mean_aff_b - mean_ants_aff_b:+.4f}`)\n",
            f"- **Mean Deformable Dice**: `syntx SyN Beltrami` **`{mean_syn_b:.4f}`** vs `ANTs SyN` `{mean_ants_syn_b:.4f}` (**`{mean_syn_b - mean_ants_syn_b:+.4f}`**, net gain **`{mean_gain_b:+.4f}`**)\n",
            f"- **Mean Hold-Out Target Registration Error**: **`{mean_tre_b:.3f} mm`**\n\n",
        ])

    tmp_md = output_md + ".tmp"
    with open(tmp_md, "w") as f:
        f.writelines(lines_md)
    os.replace(tmp_md, output_md)


def _compose_light_theme_mosaic(panel_paths: List[Tuple[str, str]], output_viz: str) -> None:
    """Stitches pre-rendered syntx.viz 4-panel PNGs (never raw medical arrays) into a light-theme report figure."""
    if not panel_paths:
        return
    try:
        from PIL import Image, ImageDraw
        imgs = []
        for tname, p in panel_paths:
            if p and os.path.exists(p):
                imgs.append((tname, Image.open(p).convert("RGB")))
        if not imgs:
            return
        w_max = max(im.width for _, im in imgs)
        pad_top = 36
        total_h = sum(im.height + pad_top for _, im in imgs)
        canvas = Image.new("RGB", (w_max, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        y = 0
        for tname, im in imgs:
            draw.text((16, y + 10), f"{tname} — syntx.viz Standard 4-Panel QC (Light Theme)", fill=(30, 41, 59))
            canvas.paste(im, (0, y + pad_top))
            y += im.height + pad_top
        os.makedirs(os.path.dirname(os.path.abspath(output_viz)), exist_ok=True)
        canvas.save(output_viz)
    finally:
        for _, p in panel_paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Medical Decathlon Provenance & Evaluation Harness")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DECATHLON_DIR)
    parser.add_argument("--output-json", type=str, default=OUT_JSON)
    parser.add_argument("--output-md", type=str, default=OUT_MD)
    parser.add_argument("--output-viz", type=str, default="docs/reports/decathlon_multi_organ_visualization.png")
    parser.add_argument("--task", type=str, default=None, help="Filter to specific task name")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task names")
    parser.add_argument("--num-pairs", "-n", "--pairs-per-task", dest="num_pairs", type=int, default=1, help="Pairs per task")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps' or 'cpu')")
    parser.add_argument("--smoke", action="store_true", help="Run fast single-pair smoke test (<60s)")
    parser.add_argument("--render-viz", action="store_true", help="Render light-theme visual figure")
    parser.add_argument("--merge-existing", action="store_true", help="Merge newly evaluated tasks into existing output JSON")
    args = parser.parse_args()

    print("=" * 85)
    print("MEDICAL DECATHLON PROVENANCE EVALUATION HARNESS")
    print("Compliant with GEMINI.md Section 6 (Whole-Organ & Landmark TRE Scoring)")
    print(f"Directory: {args.data_dir} | Pairs per task: {args.num_pairs} | Smoke: {args.smoke}")
    print("=" * 85, flush=True)

    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.smoke:
        smoke_tasks = ["Task04_Hippocampus", "Task01_BrainTumour"]
        specs_to_run = [s for s in TASK_SPECS if s["name"] in smoke_tasks]
        if args.task:
            specs_to_run = [s for s in TASK_SPECS if s["name"].lower() == args.task.lower()]
        args.num_pairs = 1
        if args.output_json == OUT_JSON:
            args.output_json = os.path.join(tempfile.gettempdir(), "decathlon_smoke_provenance.json")
        if args.output_md == OUT_MD:
            args.output_md = os.path.join(tempfile.gettempdir(), "decathlon_smoke_provenance.md")
    else:
        specs_to_run = TASK_SPECS
        if args.task:
            specs_to_run = [s for s in TASK_SPECS if s["name"].lower() == args.task.lower()]
        elif args.tasks:
            task_list = [t.strip().lower() for t in args.tasks.split(",")]
            specs_to_run = [s for s in TASK_SPECS if s["name"].lower() in task_list]

    results_by_task: Dict[str, Dict[str, Any]] = {}
    if args.merge_existing and not args.smoke and os.path.isfile(args.output_json):
        try:
            with open(args.output_json) as f:
                existing = json.load(f)
            if isinstance(existing, list):
                for entry in existing:
                    if isinstance(entry, dict) and "task_name" in entry:
                        results_by_task[entry["task_name"]] = entry
            print(f"[+] Loaded {len(results_by_task)} existing task record(s) from {args.output_json}", flush=True)
        except Exception as exc:
            print(f"[!] Warning: could not read existing {args.output_json} ({exc})", flush=True)

    panel_paths: List[Tuple[str, str]] = []
    evaluated_any = False
    for spec in specs_to_run:
        t_name = spec["name"]
        print(f"\n--- Evaluating {t_name} (Stratum {spec['stratum']}, {spec['anatomy']}, {spec['modality']}) ---", flush=True)
        res = evaluate_decathlon_cohort(
            spec,
            args.data_dir,
            num_pairs=args.num_pairs,
            device=device,
            return_visuals=args.render_viz,
            smoke=args.smoke,
        )
        if res is None:
            print(f"[-] Skipping {t_name}: data not found or incomplete.")
            continue

        evaluated_any = True
        vis = res.pop("visuals", None)
        if vis and vis.get("fig_path"):
            panel_paths.append((t_name, vis["fig_path"]))

        results_by_task[t_name] = res
        print(
            f"  [Summary] Init={res['dice_initial']:.4f} | "
            f"ANTsAff={res['dice_ants_affine']:.4f} | syntxAff={res['dice_syntx_affine']:.4f} | "
            f"ANTsSyN={res['dice_ants_syn']:.4f} | syntxSyN={res['dice_syntx_syn']:.4f} "
            f"(ΔANTs: {res['dice_syntx_syn'] - res['dice_ants_syn']:+.4f}, Gain: {res['gain_syntx_syn']:+.4f}) | "
            f"TRE={res['landmark_tre_mm']:.3f} mm | MinJac={res['min_jacobian']:.4f} | Folds={res['folding_pct']:.4f}%",
            flush=True,
        )

        # Checkpoint immediately after every completed task
        _write_provenance_outputs(
            list(results_by_task.values()),
            args.output_json,
            args.output_md,
            device=device,
            num_pairs=args.num_pairs,
            smoke=args.smoke,
        )

    if not evaluated_any:
        print("\n[!] No tasks evaluated.")
        return

    if args.render_viz and panel_paths:
        _compose_light_theme_mosaic(panel_paths, args.output_viz)
        print(f"Saved light-theme visualization figure to: {args.output_viz}")

    print(f"\nSaved structured provenance JSON to: {args.output_json}")
    print(f"Generated Markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
