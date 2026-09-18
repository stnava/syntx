#!/usr/bin/env python3
"""
Generates publication-grade dark-theme 6-panel composite figures for real data:
1. Task 01: Brain Tumour — Visually Distinct Inter-Subject Inter-Modality MRI (BRATS_003 T1w -> BRATS_036 T2w)
   with Intracranial Parenchyma Surrogate and prominent bilateral tumor pathology.
2. Task 09: Spleen — Thick-Slice (5.0mm) CT with Condition-Bounded Regularized Affine.

Theme: Deep space dark theme (#0f172a / #020617) for real radiological imaging data.
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ants
import torch

from syntx.landmarks import (
    preprocess_for_landmarks,
    detect_sift3d,
    detect_blobs_dog,
    compute_mind,
    extract_mind_at_points,
    match_landmarks,
    ransac_filter,
)
from syntx.landmarks import spatial as S
from syntx.robust_affine import robust_affine
from syntx.data.surrogates import extract_brain_parenchyma

DECATHLON_DIR = "/Users/stnava/data/decathlon"
OUTPUT_DIR = "/Users/stnava/code/syntx/docs/reports/images"
BRAIN_ARTIFACTS_DIR = "/Users/stnava/.gemini/antigravity-cli/brain/ed9af813-1310-43df-bc86-a32aeec400a0/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BRAIN_ARTIFACTS_DIR, exist_ok=True)


def add_anatomical_labels_dark(ax, spec):
    """Add standard anatomical direction letters (R, L, A, P, S, I) to the borders."""
    lab = spec["labels"]
    kw = dict(
        color="#facc15",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="#0f172a", alpha=0.75, edgecolor="none"),
    )
    ax.text(0.015, 0.5, lab["left"], transform=ax.transAxes, ha="left", va="center", **kw)
    ax.text(0.985, 0.5, lab["right"], transform=ax.transAxes, ha="right", va="center", **kw)
    ax.text(0.5, 0.02, lab["bottom"], transform=ax.transAxes, ha="center", va="bottom", **kw)
    ax.text(0.5, 0.98, lab["top"], transform=ax.transAxes, ha="center", va="top", **kw)


def render_dark_composite(
    title: str,
    fi: ants.ANTsImage,
    mi: ants.ANTsImage,
    warped_mi: ants.ANTsImage,
    cf: np.ndarray,
    cm: np.ndarray,
    mf: np.ndarray,
    lbl_f: ants.ANTsImage = None,
    lbl_m: ants.ANTsImage = None,
    warped_lbl_m: ants.ANTsImage = None,
    dice_init: float = None,
    dice_warp: float = None,
    center_mm: np.ndarray = None,
    out_paths: list[str] = None,
    fixed_label_name: str = "Fixed Target",
    moving_label_name: str = "Moving Image",
):
    """
    Renders the canonical 6-panel composite figure under the dark theme (#0f172a / #020617).
    """
    if center_mm is not None:
        sl_f = S.extract_ortho_slices(fi, center_mm=center_mm)
        sl_m = S.extract_ortho_slices(mi, center_mm=center_mm)
        sl_w = S.extract_ortho_slices(warped_mi, center_mm=center_mm)
    else:
        sl_f = S.extract_ortho_slices(fi)
        sl_m = S.extract_ortho_slices(mi)
        sl_w = S.extract_ortho_slices(warped_mi, center_mm=sl_f["center_mm"])

    fig = plt.figure(figsize=(20, 11), facecolor="#0f172a")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.18, hspace=0.22)

    spec_ax = sl_f["views"]["ax"]
    ext_ax = [0, spec_ax["n_u"] * spec_ax["spacing_u"], 0, spec_ax["n_v"] * spec_ax["spacing_v"]]
    spec_cor = sl_f["views"]["cor"]
    ext_cor = [0, spec_cor["n_u"] * spec_cor["spacing_u"], 0, spec_cor["n_v"] * spec_cor["spacing_v"]]

    def norm_arr(a):
        vmin, vmax = np.percentile(a, [2, 98])
        return np.clip((a - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Panel 1: Fixed Target Axial with Keypoints
    # -------------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0], facecolor="#020617")
    ax0.imshow(sl_f["ax"], cmap="gray", origin="lower", extent=ext_ax, interpolation="nearest")
    u_f, v_f, in_sl_ax = S.project_to_slice(fi, cf[:, :3], view=spec_ax, slab_half_mm=8.0)
    if in_sl_ax.sum() > 0:
        ax0.scatter(
            (u_f[in_sl_ax] + 0.5) * spec_ax["spacing_u"],
            (v_f[in_sl_ax] + 0.5) * spec_ax["spacing_v"],
            c="#38bdf8",
            s=28,
            edgecolors="#1d4ed8",
            linewidths=0.8,
            alpha=0.9,
            label=f"Keypoints ({in_sl_ax.sum()} pts)",
        )
        ax0.legend(loc="lower left", fontsize=8.5, facecolor="#1e293b", edgecolor="#475569", labelcolor="#ffffff")
    add_anatomical_labels_dark(ax0, spec_ax)
    ax0.set_title(f"(A) {fixed_label_name}: Axial (LPS Space, ±8mm Slab)", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax0.set_xlim(ext_ax[0], ext_ax[1])
    ax0.set_ylim(ext_ax[2], ext_ax[3])
    ax0.set_xticks([])
    ax0.set_yticks([])

    # -------------------------------------------------------------------------
    # Panel 2: Fixed Target Coronal with Keypoints
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1], facecolor="#020617")
    ax1.imshow(sl_f["cor"], cmap="gray", origin="lower", extent=ext_cor, interpolation="nearest")
    u_cor, v_cor, in_sl_cor = S.project_to_slice(fi, cf[:, :3], view=spec_cor, slab_half_mm=8.0)
    if in_sl_cor.sum() > 0:
        ax1.scatter(
            (u_cor[in_sl_cor] + 0.5) * spec_cor["spacing_u"],
            (v_cor[in_sl_cor] + 0.5) * spec_cor["spacing_v"],
            c="#f43f5e",
            s=28,
            edgecolors="#881337",
            linewidths=0.8,
            alpha=0.9,
            label=f"Keypoints ({in_sl_cor.sum()} pts)",
        )
        ax1.legend(loc="lower left", fontsize=8.5, facecolor="#1e293b", edgecolor="#475569", labelcolor="#ffffff")
    add_anatomical_labels_dark(ax1, spec_cor)
    ax1.set_title(f"(B) {fixed_label_name}: Coronal (LPS Space, ±8mm Slab)", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax1.set_xlim(ext_cor[0], ext_cor[1])
    ax1.set_ylim(ext_cor[2], ext_cor[3])
    ax1.set_xticks([])
    ax1.set_yticks([])

    # -------------------------------------------------------------------------
    # Panel 3: Inter-Subject Inlier Matches
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2], facecolor="#020617")
    spec_m_ax = sl_m["views"]["ax"]
    wf = spec_ax["n_u"] * spec_ax["spacing_u"]
    hf = spec_ax["n_v"] * spec_ax["spacing_v"]
    wm = spec_m_ax["n_u"] * spec_m_ax["spacing_u"]
    hm = spec_m_ax["n_v"] * spec_m_ax["spacing_v"]
    gap = 20.0

    ax2.imshow(sl_f["ax"], cmap="gray", origin="lower", extent=[0, wf, 0, hf], interpolation="nearest")
    ax2.imshow(sl_m["ax"], cmap="gray", origin="lower", extent=[wf + gap, wf + gap + wm, 0, hm], interpolation="nearest")

    if len(mf) > 0:
        uf_in, vf_in, _ = S.project_to_slice(fi, cf[mf[:, 0], :3], view=spec_ax, slab_half_mm=200.0)
        um_in, vm_in, _ = S.project_to_slice(mi, cm[mf[:, 1], :3], view=spec_m_ax, slab_half_mm=200.0)

        for i in range(len(mf)):
            x0_p = (uf_in[i] + 0.5) * spec_ax["spacing_u"]
            y0_p = (vf_in[i] + 0.5) * spec_ax["spacing_v"]
            x1_p = wf + gap + (um_in[i] + 0.5) * spec_m_ax["spacing_u"]
            y1_p = (vm_in[i] + 0.5) * spec_m_ax["spacing_v"]
            ax2.plot([x0_p, x1_p], [y0_p, y1_p], color="#4ade80", lw=1.3, alpha=0.85)
            ax2.plot(x0_p, y0_p, "o", color="#facc15", ms=4.5, mec="#000000", mew=0.5)
            ax2.plot(x1_p, y1_p, "o", color="#38bdf8", ms=4.5, mec="#000000", mew=0.5)

    add_anatomical_labels_dark(ax2, spec_ax)
    ax2.text(
        wf + gap + 4,
        hm / 2,
        spec_m_ax["labels"]["left"],
        color="#facc15",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"),
    )
    ax2.text(
        wf + gap + wm - 4,
        hm / 2,
        spec_m_ax["labels"]["right"],
        color="#facc15",
        fontsize=9,
        fontweight="bold",
        ha="right",
        bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"),
    )

    ax2.set_xlim(0, wf + gap + wm)
    ax2.set_ylim(0, max(hf, hm))
    match_title = f"(C) Inlier Matches ({len(mf)} Pairs): Left={fixed_label_name}, Right={moving_label_name}"
    ax2.set_title(match_title, fontsize=10.5, fontweight="bold", color="#ffffff", pad=6)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # -------------------------------------------------------------------------
    # Panel 4: Before Registration False-Color Overlay (Unaligned)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#020617")
    unaligned_mi = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[], interpolator="linear")
    sl_unaligned = S.extract_ortho_slices(unaligned_mi, center_mm=sl_f["center_mm"])

    f_norm = norm_arr(sl_f["ax"])
    u_norm = norm_arr(sl_unaligned["ax"])

    rgb_before = np.zeros((*f_norm.shape, 3))
    rgb_before[..., 0] = f_norm               # Red = Fixed Target
    rgb_before[..., 1] = u_norm * 0.95        # Green = Unaligned Moving
    rgb_before[..., 2] = u_norm * 0.95        # Cyan tint
    ax3.imshow(rgb_before, origin="lower", extent=ext_ax, interpolation="nearest")
    add_anatomical_labels_dark(ax3, spec_ax)
    raw_str = f"Dice = {dice_init:.3f}" if dice_init is not None else "Unaligned"
    ax3.set_title(f"(D) Before Alignment ({raw_str}): Red=Target, Cyan=Moving", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax3.set_xlim(ext_ax[0], ext_ax[1])
    ax3.set_ylim(ext_ax[2], ext_ax[3])
    ax3.set_xticks([])
    ax3.set_yticks([])

    # -------------------------------------------------------------------------
    # Panel 5: After Alignment False-Color Overlay
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#020617")
    w_norm = norm_arr(sl_w["ax"])
    rgb_after = np.zeros((*f_norm.shape, 3))
    rgb_after[..., 0] = f_norm                # Red = Fixed Target
    rgb_after[..., 1] = w_norm * 0.95         # Green = Warped Moving
    rgb_after[..., 2] = w_norm * 0.95         # Cyan tint
    ax4.imshow(rgb_after, origin="lower", extent=ext_ax, interpolation="nearest")
    add_anatomical_labels_dark(ax4, spec_ax)
    reg_str = f"Dice = {dice_warp:.3f}" if dice_warp is not None else "Aligned"
    ax4.set_title(f"(E) After Alignment ({reg_str}): Red=Target, Cyan=Warped", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax4.set_xlim(ext_ax[0], ext_ax[1])
    ax4.set_ylim(ext_ax[2], ext_ax[3])
    ax4.set_xticks([])
    ax4.set_yticks([])

    # -------------------------------------------------------------------------
    # Panel 6: Segmentation Contours / Alignment Verification
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2], facecolor="#020617")
    has_labels = (lbl_f is not None and warped_lbl_m is not None and dice_warp is not None and dice_warp >= 0.05)

    if has_labels:
        sl_lf = S.extract_ortho_slices(lbl_f, center_mm=sl_f["center_mm"])
        sl_lu = S.extract_ortho_slices(
            ants.apply_transforms(fixed=fi, moving=lbl_m, transformlist=[], interpolator="nearestNeighbor"),
            center_mm=sl_f["center_mm"],
        )
        sl_lw = S.extract_ortho_slices(warped_lbl_m, center_mm=sl_f["center_mm"])

        ax5.imshow(sl_f["ax"], cmap="gray", origin="lower", extent=ext_ax, interpolation="nearest")
        if sl_lf["ax"].max() > 0:
            ax5.contour(sl_lf["ax"] > 0, levels=[0.5], colors=["#38bdf8"], linewidths=[2.5], extent=ext_ax)
        if sl_lu["ax"].max() > 0:
            ax5.contour(sl_lu["ax"] > 0, levels=[0.5], colors=["#f43f5e"], linewidths=[1.8], linestyles=["dashed"], extent=ext_ax)
        if sl_lw["ax"].max() > 0:
            ax5.contour(sl_lw["ax"] > 0, levels=[0.5], colors=["#4ade80"], linewidths=[2.5], extent=ext_ax)

        custom_lines = [
            Line2D([0], [0], color="#38bdf8", lw=2.5, label="Fixed Target Mask"),
            Line2D([0], [0], color="#f43f5e", lw=1.8, linestyle="dashed", label=f"Unaligned (Dice={dice_init:.3f})" if dice_init is not None else "Unaligned"),
            Line2D([0], [0], color="#4ade80", lw=2.5, label=f"Warped Mask (Dice={dice_warp:.3f})" if dice_warp is not None else "Warped"),
        ]
        ax5.legend(handles=custom_lines, loc="lower left", fontsize=8.5, facecolor="#1e293b", edgecolor="#475569", labelcolor="#ffffff")
        add_anatomical_labels_dark(ax5, spec_ax)
        ax5.set_title(f"(F) Anatomical Contours [Dice: {dice_init:.3f} → {dice_warp:.3f}]", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    else:
        # High-contrast checkerboard
        n_chk = 8
        chk = np.indices((n_chk, n_chk)).sum(axis=0) % 2
        chk_full = np.repeat(
            np.repeat(chk, f_norm.shape[0] // n_chk + 1, axis=0)[:f_norm.shape[0], :],
            f_norm.shape[1] // n_chk + 1,
            axis=1,
        )[:, :f_norm.shape[1]]
        composite = np.where(chk_full, f_norm, w_norm)
        ax5.imshow(composite, cmap="gray", origin="lower", extent=ext_ax, interpolation="nearest")
        add_anatomical_labels_dark(ax5, spec_ax)
        ax5.set_title("(F) Checkerboard: Target vs Warped Image", fontsize=11, fontweight="bold", color="#ffffff", pad=6)

    ax5.set_xlim(ext_ax[0], ext_ax[1])
    ax5.set_ylim(ext_ax[2], ext_ax[3])
    ax5.set_xticks([])
    ax5.set_yticks([])

    fig.suptitle(title, fontsize=13.5, fontweight="bold", color="#38bdf8", y=0.98)

    if out_paths:
        for p in out_paths:
            fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="#0f172a")
            print(f"[+] Saved dark figure: {p}")
    plt.close(fig)


def generate_task01_dark():
    print("\n--- Generating Task 01: Distinct Inter-Subject Inter-Modality MRI (BRATS_003 T1w -> BRATS_036 T2w) ---")
    p0 = os.path.join(DECATHLON_DIR, "Task01_BrainTumour", "imagesTr", "BRATS_003.nii.gz")
    p1 = os.path.join(DECATHLON_DIR, "Task01_BrainTumour", "imagesTr", "BRATS_036.nii.gz")
    l0 = os.path.join(DECATHLON_DIR, "Task01_BrainTumour", "labelsTr", "BRATS_003.nii.gz")
    l1 = os.path.join(DECATHLON_DIR, "Task01_BrainTumour", "labelsTr", "BRATS_036.nii.gz")

    img0_4d = ants.image_read(p0)
    img1_4d = ants.image_read(p1)
    lbl0 = ants.image_read(l0)
    lbl1 = ants.image_read(l1)

    # Subject 0 = BRATS_003 T1w (idx 1, 1.41M voxels, large brain, right parietal lesion)
    # Subject 1 = BRATS_036 T2w (idx 3, 0.89M voxels, small brain, left frontal hyperintense lesion)
    fi_t1 = ants.slice_image(img0_4d, axis=3, idx=1)
    mi_t2 = ants.slice_image(img1_4d, axis=3, idx=3)

    pre0 = preprocess_for_landmarks(fi_t1, use_n4=False)
    pre1 = preprocess_for_landmarks(mi_t2, use_n4=False)

    surr0 = extract_brain_parenchyma(fi_t1, min_volume_voxels=50000)
    surr1 = extract_brain_parenchyma(mi_t2, min_volume_voxels=50000)

    # Raw unaligned Dice
    raw_surr = ants.apply_transforms(fixed=surr0, moving=surr1, transformlist=[], interpolator="nearestNeighbor")
    dice_raw = float(ants.label_overlap_measures(surr0, raw_surr).loc[lambda df: df["Label"] == "All", "MeanOverlap"].iloc[0])

    # Inter-Modality Mattes-MI robust_affine
    res = robust_affine(pre0, pre1, mode="auto", verbose=False)
    warped_mi = ants.apply_transforms(fixed=pre0, moving=pre1, transformlist=res["fwdtransforms"], whichtoinvert=[False], interpolator="linear")
    warped_surr = ants.apply_transforms(fixed=surr0, moving=surr1, transformlist=res["fwdtransforms"], whichtoinvert=[False], interpolator="nearestNeighbor")
    dice_aligned = float(ants.label_overlap_measures(surr0, warped_surr).loc[lambda df: df["Label"] == "All", "MeanOverlap"].iloc[0])

    print(f"Task 01 Inter-Subject Inter-Modality Dice: Raw={dice_raw:.4f} -> Aligned={dice_aligned:.4f} (+{dice_aligned-dice_raw:+.4f})")

    # Detect multi-modal MIND descriptors & match
    pts0 = detect_blobs_dog(pre0, max_keypoints=300)
    pts1 = detect_blobs_dog(pre1, max_keypoints=300)
    mind0 = compute_mind(pre0)
    mind1 = compute_mind(pre1)
    desc0 = extract_mind_at_points(pre0, pts0[:, :3], mind_vol=mind0)
    desc1 = extract_mind_at_points(pre1, pts1[:, :3], mind_vol=mind1)
    m = match_landmarks(pts0, pts1, desc0, desc1, ratio_thresh=0.92, mutual=True)
    mf, M = ransac_filter(pts0, pts1, m, model="rigid", inlier_thresh_mm=12.0)
    print(f"Task 01 Multi-Modal MIND Matches: {len(m)}, RANSAC inliers: {len(mf)}")

    out_p1 = os.path.join(OUTPUT_DIR, "task01_braintumour_spatial_composite.png")
    out_p2 = os.path.join(BRAIN_ARTIFACTS_DIR, "task01_braintumour_spatial_composite.png")

    # Slice at z=98mm where both distinct lesions and ventricles are prominent
    ctr_tumor_mm = np.array([120.0, 120.0, 98.0])

    render_dark_composite(
        title="Task 01: Brain Tumour — Inter-Subject Inter-Modality MRI (BRATS_003 T1w vs BRATS_036 T2w) with Intracranial Surrogate",
        fi=pre0,
        mi=pre1,
        warped_mi=warped_mi,
        cf=pts0,
        cm=pts1,
        mf=mf,
        lbl_f=surr0,
        lbl_m=surr1,
        warped_lbl_m=warped_surr,
        dice_init=dice_raw,
        dice_warp=dice_aligned,
        center_mm=ctr_tumor_mm,
        out_paths=[out_p1, out_p2],
        fixed_label_name="BRATS_003 (T1w Target, 1.41L Brain)",
        moving_label_name="BRATS_036 (T2w Moving, 0.89L Brain)",
    )


def generate_task09_dark():
    print("\n--- Generating Task 09: Spleen CT Thick-Slice Regularized Affine (Dark Theme) ---")
    p0 = os.path.join(DECATHLON_DIR, "Task09_Spleen", "imagesTr", "spleen_19.nii.gz")
    p1 = os.path.join(DECATHLON_DIR, "Task09_Spleen", "imagesTr", "spleen_31.nii.gz")
    l0 = os.path.join(DECATHLON_DIR, "Task09_Spleen", "labelsTr", "spleen_19.nii.gz")
    l1 = os.path.join(DECATHLON_DIR, "Task09_Spleen", "labelsTr", "spleen_31.nii.gz")

    img0 = ants.image_read(p0)
    img1 = ants.image_read(p1)
    lbl0 = ants.image_read(l0)
    lbl1 = ants.image_read(l1)

    pre0 = preprocess_for_landmarks(img0, ct_window="soft_tissue")
    pre1 = preprocess_for_landmarks(img1, ct_window="soft_tissue")

    c0, d0 = detect_sift3d(pre0, preprocess=False, max_keypoints=500)
    c1, d1 = detect_sift3d(pre1, preprocess=False, max_keypoints=500)
    m = match_landmarks(c0, c1, d0, d1, ratio_thresh=0.90, mutual=True)

    # Use regularized affine to protect against 5.0mm thick-slice coplanar shear collapse
    mf, M = ransac_filter(c0, c1, m, model="regularized_affine", inlier_thresh_mm=15.0)

    with tempfile.NamedTemporaryFile(suffix=".mat") as tmp_mat:
        tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
        tx.set_parameters(np.concatenate([M[:3, :3].ravel(), M[:3, 3]]))
        ants.write_transform(tx, tmp_mat.name)

        # Compute raw Dice
        raw_lbl = ants.apply_transforms(fixed=lbl0, moving=lbl1, transformlist=[], interpolator="nearestNeighbor")
        dice_raw = float(ants.label_overlap_measures(lbl0, raw_lbl).loc[lambda df: df["Label"] == "All", "MeanOverlap"].iloc[0])

        # Warp images with regularized affine
        warped_mi = ants.apply_transforms(fixed=pre0, moving=pre1, transformlist=[tmp_mat.name], whichtoinvert=[False], interpolator="linear")
        warped_lbl = ants.apply_transforms(fixed=lbl0, moving=lbl1, transformlist=[tmp_mat.name], whichtoinvert=[False], interpolator="nearestNeighbor")
        dice_warp = float(ants.label_overlap_measures(lbl0, warped_lbl).loc[lambda df: df["Label"] == "All", "MeanOverlap"].iloc[0])

    print(f"Task 09 Regularized Affine Dice: Raw={dice_raw:.4f} -> Landmark Regularized Affine={dice_warp:.4f} (+{dice_warp-dice_raw:+.4f})")

    out_p1 = os.path.join(OUTPUT_DIR, "task09_spleen_spatial_composite.png")
    out_p2 = os.path.join(BRAIN_ARTIFACTS_DIR, "task09_spleen_spatial_composite.png")

    render_dark_composite(
        title="Task 09: Spleen — Thick-Slice (5.0mm) CT Registration under Condition-Bounded Regularized Affine",
        fi=pre0,
        mi=pre1,
        warped_mi=warped_mi,
        cf=c0,
        cm=c1,
        mf=mf,
        lbl_f=lbl0,
        lbl_m=lbl1,
        warped_lbl_m=warped_lbl,
        dice_init=dice_raw,
        dice_warp=dice_warp,
        out_paths=[out_p1, out_p2],
        fixed_label_name="Fixed CT Target (spleen_19)",
        moving_label_name="Moving CT (spleen_31)",
    )


if __name__ == "__main__":
    generate_task01_dark()
    generate_task09_dark()
    print("\n[+] All dark-theme composites generated successfully!")
