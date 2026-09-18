#!/usr/bin/env python3
"""
Regenerate Task 06 (Lung CT) and Task 08 (Hepatic Vessel CT) composite figures
using true syntx robust_affine / tournament registration, eliminating degenerate
sparse RANSAC artifacts and showing valid anatomical alignment.
"""

import os
import time
import shutil
import numpy as np
import ants
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from syntx.landmarks.preprocess import preprocess_for_landmarks
from syntx.robust_affine import robust_affine
from syntx.landmarks import (
    spatial as S,
    detect_sift3d,
    match_landmarks,
    ransac_filter,
    sampled_optimal_transport_affine,
)

def add_anatomical_labels(ax, view_spec):
    labels = view_spec["labels"]
    w = view_spec["n_u"] * view_spec["spacing_u"]
    h = view_spec["n_v"] * view_spec["spacing_v"]
    pad = 4.0
    ax.text(w / 2, h - pad, labels["top"], color="#facc15", fontsize=9, fontweight="bold",
            ha="center", va="top", bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"))
    ax.text(w / 2, pad, labels["bottom"], color="#facc15", fontsize=9, fontweight="bold",
            ha="center", va="bottom", bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"))
    ax.text(pad, h / 2, labels["left"], color="#facc15", fontsize=9, fontweight="bold",
            ha="left", va="center", bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"))
    ax.text(w - pad, h / 2, labels["right"], color="#facc15", fontsize=9, fontweight="bold",
            ha="right", va="center", bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"))

def generate_valid_task_figure(task_name, anatomy, modality, fi, mi, cf, cm, mf, warped_mi, out_path, cc_init, cc_warped):
    sl_f = S.extract_ortho_slices(fi)
    sl_m = S.extract_ortho_slices(mi)
    sl_w = S.extract_ortho_slices(warped_mi, center_mm=sl_f["center_mm"])

    fig = plt.figure(figsize=(20, 11), facecolor="#0f172a")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.18, hspace=0.22)

    # Panel 1: Fixed Axial
    ax0 = fig.add_subplot(gs[0, 0], facecolor="#020617")
    spec_ax = sl_f["views"]["ax"]
    ext_ax = [0, spec_ax["n_u"] * spec_ax["spacing_u"], 0, spec_ax["n_v"] * spec_ax["spacing_v"]]
    ax0.imshow(sl_f["ax"], cmap="gray", origin="lower", extent=ext_ax, interpolation="nearest")
    u_f, v_f, in_sl_ax = S.project_to_slice(fi, cf[:, :3], view=spec_ax, slab_half_mm=8.0)
    if in_sl_ax.sum() > 0:
        ax0.scatter((u_f[in_sl_ax] + 0.5) * spec_ax["spacing_u"],
                    (v_f[in_sl_ax] + 0.5) * spec_ax["spacing_v"],
                    c="#38bdf8", s=28, edgecolors="#1d4ed8", linewidths=0.8, alpha=0.9,
                    label=f"Landmarks ({in_sl_ax.sum()} pts)")
        ax0.legend(loc="lower left", fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="#ffffff")
    add_anatomical_labels(ax0, spec_ax)
    ax0.set_title(f"(A) Fixed Target: Axial (LPS Space, ±8mm Slab)", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax0.set_xlim(ext_ax[0], ext_ax[1]); ax0.set_ylim(ext_ax[2], ext_ax[3])
    ax0.set_xticks([]); ax0.set_yticks([])

    # Panel 2: Fixed Coronal
    ax1 = fig.add_subplot(gs[0, 1], facecolor="#020617")
    spec_cor = sl_f["views"]["cor"]
    ext_cor = [0, spec_cor["n_u"] * spec_cor["spacing_u"], 0, spec_cor["n_v"] * spec_cor["spacing_v"]]
    ax1.imshow(sl_f["cor"], cmap="gray", origin="lower", extent=ext_cor, interpolation="nearest")
    u_cor, v_cor, in_sl_cor = S.project_to_slice(fi, cf[:, :3], view=spec_cor, slab_half_mm=8.0)
    if in_sl_cor.sum() > 0:
        ax1.scatter((u_cor[in_sl_cor] + 0.5) * spec_cor["spacing_u"],
                    (v_cor[in_sl_cor] + 0.5) * spec_cor["spacing_v"],
                    c="#f43f5e", s=28, edgecolors="#881337", linewidths=0.8, alpha=0.9,
                    label=f"Landmarks ({in_sl_cor.sum()} pts)")
        ax1.legend(loc="lower left", fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="#ffffff")
    add_anatomical_labels(ax1, spec_cor)
    ax1.set_title(f"(B) Fixed Target: Coronal (LPS Space, ±8mm Slab)", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax1.set_xlim(ext_cor[0], ext_cor[1]); ax1.set_ylim(ext_cor[2], ext_cor[3])
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 3: Inter-subject Matches
    ax2 = fig.add_subplot(gs[0, 2], facecolor="#020617")
    spec_m_ax = sl_m["views"]["ax"]
    wf = spec_ax["n_u"] * spec_ax["spacing_u"]
    hf = spec_ax["n_v"] * spec_ax["spacing_v"]
    wm = spec_m_ax["n_u"] * spec_m_ax["spacing_u"]
    hm = spec_m_ax["n_v"] * spec_m_ax["spacing_v"]
    gap = 20.0

    ax2.imshow(sl_f["ax"], cmap="gray", origin="lower", extent=[0, wf, 0, hf], interpolation="nearest")
    ax2.imshow(sl_m["ax"], cmap="gray", origin="lower", extent=[wf + gap, wf + gap + wm, 0, hm], interpolation="nearest")

    uf_in, vf_in, _ = S.project_to_slice(fi, cf[mf[:, 0], :3], view=spec_ax, slab_half_mm=200.0)
    um_in, vm_in, _ = S.project_to_slice(mi, cm[mf[:, 1], :3], view=spec_m_ax, slab_half_mm=200.0)

    for i in range(len(mf)):
        x0_p = (uf_in[i] + 0.5) * spec_ax["spacing_u"]
        y0_p = (vf_in[i] + 0.5) * spec_ax["spacing_v"]
        x1_p = wf + gap + (um_in[i] + 0.5) * spec_m_ax["spacing_u"]
        y1_p = (vm_in[i] + 0.5) * spec_m_ax["spacing_v"]
        ax2.plot([x0_p, x1_p], [y0_p, y1_p], color="#4ade80", lw=1.2, alpha=0.85)
        ax2.plot(x0_p, y0_p, "o", color="#facc15", ms=4, mec="#000000", mew=0.5)
        ax2.plot(x1_p, y1_p, "o", color="#38bdf8", ms=4, mec="#000000", mew=0.5)

    add_anatomical_labels(ax2, spec_ax)
    ax2.text(wf + gap + 4, hm / 2, spec_m_ax["labels"]["left"], color="#facc15", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"))
    ax2.text(wf + gap + wm - 4, hm / 2, spec_m_ax["labels"]["right"], color="#facc15", fontsize=9, fontweight="bold",
             ha="right", bbox=dict(boxstyle="square,pad=0.1", facecolor="#0f172a", alpha=0.7, edgecolor="none"))
    ax2.set_xlim(0, wf + gap + wm)
    ax2.set_ylim(0, max(hf, hm))
    ax2.set_title(f"(C) Inlier Matches ({len(mf)} Pairs)", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 4: Before Alignment False-Color Overlay
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#020617")
    unaligned_mi = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[], interpolator="linear")
    sl_unaligned = S.extract_ortho_slices(unaligned_mi, center_mm=sl_f["center_mm"])

    def norm_arr(a):
        vmin, vmax = np.percentile(a, [2, 98])
        return np.clip((a - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    f_norm = norm_arr(sl_f["ax"])
    u_norm = norm_arr(sl_unaligned["ax"])
    w_norm = norm_arr(sl_w["ax"])

    rgb_before = np.zeros((*f_norm.shape, 3))
    rgb_before[..., 0] = f_norm
    rgb_before[..., 1] = u_norm * 0.95
    rgb_before[..., 2] = u_norm * 0.95
    ax3.imshow(rgb_before, origin="lower", extent=ext_ax, interpolation="nearest")
    add_anatomical_labels(ax3, spec_ax)
    ax3.set_title(f"(D) Before Alignment (Corr = {cc_init:.3f})", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax3.set_xlim(ext_ax[0], ext_ax[1]); ax3.set_ylim(ext_ax[2], ext_ax[3])
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 5: After Alignment False-Color Overlay
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#020617")
    rgb_after = np.zeros((*f_norm.shape, 3))
    rgb_after[..., 0] = f_norm
    rgb_after[..., 1] = w_norm * 0.95
    rgb_after[..., 2] = w_norm * 0.95
    ax4.imshow(rgb_after, origin="lower", extent=ext_ax, interpolation="nearest")
    add_anatomical_labels(ax4, spec_ax)
    ax4.set_title(f"(E) Our Best Alignment: Red=Target, Cyan=Warped (Corr = {cc_warped:.3f})", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    ax4.set_xlim(ext_ax[0], ext_ax[1]); ax4.set_ylim(ext_ax[2], ext_ax[3])
    ax4.set_xticks([]); ax4.set_yticks([])

    # Panel 6: Checkerboard
    ax5 = fig.add_subplot(gs[1, 2], facecolor="#020617")
    n_tiles = 6
    chk = np.zeros_like(f_norm)
    sy, sx = f_norm.shape[0] // n_tiles, f_norm.shape[1] // n_tiles
    for i in range(n_tiles):
        for j in range(n_tiles):
            y0_s, y1_s = i * sy, (i + 1) * sy if i < n_tiles - 1 else f_norm.shape[0]
            x0_s, x1_s = j * sx, (j + 1) * sx if j < n_tiles - 1 else f_norm.shape[1]
            if (i + j) % 2 == 0:
                chk[y0_s:y1_s, x0_s:x1_s] = f_norm[y0_s:y1_s, x0_s:x1_s]
            else:
                chk[y0_s:y1_s, x0_s:x1_s] = w_norm[y0_s:y1_s, x0_s:x1_s]
    ax5.imshow(chk, cmap="gray", origin="lower", extent=ext_ax, interpolation="nearest")
    ax5.set_title("(F) Checkerboard: Target vs Robust Affine Warped", fontsize=11, fontweight="bold", color="#ffffff", pad=6)
    add_anatomical_labels(ax5, spec_ax)
    ax5.set_xlim(ext_ax[0], ext_ax[1]); ax5.set_ylim(ext_ax[2], ext_ax[3])
    ax5.set_xticks([]); ax5.set_yticks([])

    fig.suptitle(f"{task_name} ({anatomy}, {modality}) — Full Anatomical Alignment via syntx robust_affine",
                 fontsize=15, fontweight="bold", color="#38bdf8", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[+] Saved valid composite to {out_path}")

def run():
    # ----------------------------------------------------
    # Task 06: Lung CT
    # ----------------------------------------------------
    print("\n=== Processing Task 06: Lung CT ===")
    p0_lung = os.path.expanduser("~/data/decathlon/Task06_Lung/imagesTr/lung_001.nii.gz")
    p1_lung = os.path.expanduser("~/data/decathlon/Task06_Lung/imagesTr/lung_003.nii.gz")
    f_l = ants.image_read(p0_lung)
    m_l = ants.image_read(p1_lung)

    # Subsample slightly for fast high-quality convergence on 512x512x300 CT
    f_l_sub = ants.resample_image(f_l, (256, 256, 152), use_voxels=True, interp_type=1)
    m_l_sub = ants.resample_image(m_l, (256, 256, 144), use_voxels=True, interp_type=1)

    pre_f_l = preprocess_for_landmarks(f_l_sub, ct_window="lung")
    pre_m_l = preprocess_for_landmarks(m_l_sub, ct_window="lung")

    # Detect keypoints
    cf_l, df_l = detect_sift3d(pre_f_l, preprocess=False, max_keypoints=300)
    cm_l, dm_l = detect_sift3d(pre_m_l, preprocess=False, max_keypoints=300)
    m_l_pairs = match_landmarks(cf_l, cm_l, df_l, dm_l, ratio_thresh=0.92, mutual=True)
    mf_l, _ = ransac_filter(cf_l, cm_l, m_l_pairs, model="rigid", inlier_thresh_mm=15.0)

    # Run robust_affine
    print("Running robust_affine on Task 06...")
    res_l = robust_affine(pre_f_l, pre_m_l, mode="auto", verbose=False)
    warped_m_l = ants.apply_transforms(fixed=pre_f_l, moving=pre_m_l, transformlist=res_l["fwdtransforms"])

    # Correlations
    unaligned_l = ants.apply_transforms(fixed=pre_f_l, moving=pre_m_l, transformlist=[])
    cc_init_l = float(np.corrcoef(pre_f_l.numpy().ravel(), unaligned_l.numpy().ravel())[0, 1])
    cc_warp_l = float(np.corrcoef(pre_f_l.numpy().ravel(), warped_m_l.numpy().ravel())[0, 1])
    print(f"Task 06 Correlation: {cc_init_l:.4f} -> {cc_warp_l:.4f}")

    out_lung = "docs/reports/images/task06_lung_spatial_composite.png"
    generate_valid_task_figure("Task06_Lung", "Lung", "CT", pre_f_l, pre_m_l, cf_l, cm_l, mf_l, warped_m_l, out_lung, cc_init_l, cc_warp_l)

    # ----------------------------------------------------
    # Task 08: Hepatic Vessel CT
    # ----------------------------------------------------
    print("\n=== Processing Task 08: Hepatic Vessel CT ===")
    p0_hep = os.path.expanduser("~/data/decathlon/Task08_HepaticVessel/imagesTr/hepaticvessel_001.nii.gz")
    p1_hep = os.path.expanduser("~/data/decathlon/Task08_HepaticVessel/imagesTr/hepaticvessel_002.nii.gz")
    f_h = ants.image_read(p0_hep)
    m_h = ants.image_read(p1_hep)

    f_h_sub = ants.resample_image(f_h, (256, 256, 49), use_voxels=True, interp_type=1)
    m_h_sub = ants.resample_image(m_h, (256, 256, 66), use_voxels=True, interp_type=1)

    pre_f_h = preprocess_for_landmarks(f_h_sub, ct_window="soft_tissue")
    pre_m_h = preprocess_for_landmarks(m_h_sub, ct_window="soft_tissue")

    cf_h, df_h = detect_sift3d(pre_f_h, preprocess=False, max_keypoints=300)
    cm_h, dm_h = detect_sift3d(pre_m_h, preprocess=False, max_keypoints=300)
    m_h_pairs = match_landmarks(cf_h, cm_h, df_h, dm_h, ratio_thresh=0.92, mutual=True)
    mf_h, _ = ransac_filter(cf_h, cm_h, m_h_pairs, model="rigid", inlier_thresh_mm=15.0)

    print("Running robust_affine on Task 08...")
    res_h = robust_affine(pre_f_h, pre_m_h, mode="auto", verbose=False)
    warped_m_h = ants.apply_transforms(fixed=pre_f_h, moving=pre_m_h, transformlist=res_h["fwdtransforms"])

    unaligned_h = ants.apply_transforms(fixed=pre_f_h, moving=pre_m_h, transformlist=[])
    cc_init_h = float(np.corrcoef(pre_f_h.numpy().ravel(), unaligned_h.numpy().ravel())[0, 1])
    cc_warp_h = float(np.corrcoef(pre_f_h.numpy().ravel(), warped_m_h.numpy().ravel())[0, 1])
    print(f"Task 08 Correlation: {cc_init_h:.4f} -> {cc_warp_h:.4f}")

    out_hep = "docs/reports/images/task08_hepaticvessel_spatial_composite.png"
    generate_valid_task_figure("Task08_HepaticVessel", "Vessels", "CT", pre_f_h, pre_m_h, cf_h, cm_h, mf_h, warped_m_h, out_hep, cc_init_h, cc_warp_h)

    # Copy to brain artifacts
    brain_fig_dir = "/Users/stnava/.gemini/antigravity-cli/brain/ed9af813-1310-43df-bc86-a32aeec400a0/figures"
    os.makedirs(brain_fig_dir, exist_ok=True)
    shutil.copy(out_lung, os.path.join(brain_fig_dir, "task06_lung_spatial_composite.png"))
    shutil.copy(out_hep, os.path.join(brain_fig_dir, "task08_hepaticvessel_spatial_composite.png"))
    print("[+] Successfully updated and synchronized Task 06 and Task 08 figures!")

if __name__ == "__main__":
    run()
