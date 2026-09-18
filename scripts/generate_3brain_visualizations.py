#!/usr/bin/env python3
"""
scripts/generate_3brain_visualizations.py

Generates 3 detailed real-data brain examples with affine registration
and landmark projections for the master affine report:
1. Brain Example 1: Mindboggle Hard Pair 44 (NKI-TRT-20-2 -> MMRR-21-2)
2. Brain Example 2: Mindboggle Intra Pair 00 (OASIS-TRT-20-17 -> OASIS-TRT-20-16)
3. Brain Example 3: Mindboggle Inter Pair 76 (NKI-RS-22-4 -> MMRR-21-8)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ants
import torch

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.evaluate import normalize_intensity, compute_bidirectional_dice
from syntx.landmarks import detect_sift3d, match_landmarks, ransac_filter
from syntx.landmarks import spatial as S
from syntx.robust_affine import robust_affine

OUT_DIR = "/Users/stnava/.gemini/antigravity-cli/brain/ed9af813-1310-43df-bc86-a32aeec400a0/figures"
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
    {"idx": 44, "name": "brain_example_01_mbhard_pair44.png", "title": "Brain Example 1: Mindboggle Hard Pair 44 (Inter-Cohort: NKI-TRT vs MMRR)"},
    {"idx": 0,  "name": "brain_example_02_oasis_pair00.png",  "title": "Brain Example 2: Mindboggle Intra Pair 00 (Intra-Cohort: OASIS-TRT)"},
    {"idx": 76, "name": "brain_example_03_inter_pair76.png",  "title": "Brain Example 3: Mindboggle Inter Pair 76 (Inter-Cohort: NKI-RS vs MMRR)"},
]

def render_pair_composite(p_info):
    pair_idx = p_info["idx"]
    out_file = os.path.join(OUT_DIR, p_info["name"])
    print(f"Generating visualization for Pair {pair_idx} -> {out_file}...")

    # Load pair
    p = load_mindboggle_pair(pair_idx=pair_idx, use_n4=True)
    fi = normalize_intensity(p["fixed"])
    mi = normalize_intensity(p["moving"])
    fl = p["fixed_label"]
    ml = p["moving_label"]

    # Compute or load affine
    aff_mat = f"results/canonical_affines/pair_{pair_idx:03d}_pt7_affine.mat"
    if not os.path.exists(aff_mat):
        aff_mat = f"results/canonical_affines/pair_{pair_idx:03d}_affine.mat"
    if not os.path.exists(aff_mat):
        res_aff = robust_affine(fi, mi, mode="auto", seed=42)
        aff_mat = res_aff["fwdtransforms"][0]

    # Warped moving image and label under affine
    mi_warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[aff_mat], interpolator="linear")
    ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[aff_mat], interpolator="nearestNeighbor")

    # Evaluate Dice
    _, _, aff_dice = compute_bidirectional_dice(fl, ml, fi, mi, [aff_mat], [aff_mat], [True])

    # Extract SIFT3D landmarks
    cf, df = detect_sift3d(fi, preprocess=False, max_keypoints=600, n_scales=6)
    cm, dm = detect_sift3d(mi_warped, preprocess=False, max_keypoints=600, n_scales=6)
    matches = match_landmarks(cf, cm, df, dm, ratio_thresh=0.95, mutual=True)

    # Slices via syntx.landmarks.spatial
    sl_fi = S.extract_ortho_slices(fi)
    sl_mi = S.extract_ortho_slices(mi)
    sl_warp = S.extract_ortho_slices(mi_warped)

    # Project keypoints to axial slice
    u_f, v_f, in_f = S.project_to_slice(fi, cf[:, :3], view=sl_fi["views"]["ax"], slab_half_mm=4.0)
    u_m, v_m, in_m = S.project_to_slice(mi_warped, cm[:, :3], view=sl_warp["views"]["ax"], slab_half_mm=4.0)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.patch.set_facecolor('#0d1117')

    views = ['ax', 'cor', 'sag']
    view_names = ['Axial Slice', 'Coronal Slice', 'Sagittal Slice']

    for r, (v_key, v_title) in enumerate(zip(views, view_names)):
        # Col 0: Fixed Image
        ax0 = axes[r, 0]
        ax0.imshow(sl_fi[v_key], cmap='gray', origin='lower')
        ax0.set_title(f"Fixed Target ({v_title})", color='white', fontsize=11, pad=6)
        ax0.axis('off')

        # Col 1: Moving Image (Unaligned)
        ax1 = axes[r, 1]
        ax1.imshow(sl_mi[v_key], cmap='gray', origin='lower')
        ax1.set_title(f"Moving Unaligned ({v_title})", color='white', fontsize=11, pad=6)
        ax1.axis('off')

        # Col 2: Affine Aligned
        ax2 = axes[r, 2]
        ax2.imshow(sl_warp[v_key], cmap='gray', origin='lower')
        ax2.set_title(f"Affine Aligned ({v_title})", color='white', fontsize=11, pad=6)
        ax2.axis('off')

        # Col 3: Difference / Overlay or Landmarks
        ax3 = axes[r, 3]
        diff = np.abs(sl_fi[v_key] - sl_warp[v_key])
        im_diff = ax3.imshow(diff, cmap='magma', origin='lower', vmin=0.0, vmax=0.5)
        ax3.set_title(f"Residual |Fixed - Affine| ({v_title})", color='white', fontsize=11, pad=6)
        ax3.axis('off')

    # Overlay SIFT3D landmarks on row 0 (axial)
    axes[0, 0].scatter(u_f[in_f], v_f[in_f], s=25, c='#00ffcc', edgecolors='black', linewidths=0.5, alpha=0.85, label='SIFT3D Keypoints')
    axes[0, 0].legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=8)

    axes[0, 2].scatter(u_m[in_m], v_m[in_m], s=25, c='#ff007f', edgecolors='black', linewidths=0.5, alpha=0.85, label='Warped Keypoints')
    axes[0, 2].legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=8)

    plt.suptitle(f"{p_info['title']}\nCanonical Affine Sørensen-Dice: {aff_dice:.4f} | SIFT3D Matches: {len(matches)}",
                 color='white', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    plt.savefig(out_file, dpi=160, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Successfully saved {out_file}")

def main():
    for p in PAIRS:
        render_pair_composite(p)
    print("All 3 brain visualizations created successfully!")

if __name__ == "__main__":
    main()
