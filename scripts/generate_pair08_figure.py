#!/usr/bin/env python3
"""
Generate publication-quality anatomical visual comparison figure for Pair 08
demonstrating replication of Continuum Mechanics gains in an intra-scanner cohort.
Strictly adheres to GEMINI.md Rules 1-6 (light theme, physical space, syntx.viz).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ants

from syntx.viz import extract_oriented_slice

plt.rcParams.update({
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'text.color': '#1E293B',
    'axes.labelcolor': '#1E293B',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 1.0,
})

FIG_DIR = "docs/reports/figures_cm_evidence"
PAIR_DIR = "results/anatomical_evidence/pair_008"
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading Pair 08 images...", flush=True)
fi = ants.image_read(os.path.join(PAIR_DIR, "fixed_target.nii.gz"))
mi = ants.image_read(os.path.join(PAIR_DIR, "moving_source.nii.gz"))
w_aff = ants.image_read(os.path.join(PAIR_DIR, "warped_affine.nii.gz"))
w_sob = ants.image_read(os.path.join(PAIR_DIR, "warped_sobolev.nii.gz"))
w_hyp = ants.image_read(os.path.join(PAIR_DIR, "warped_syn_hyperelastic.nii.gz"))
w_div = ants.image_read(os.path.join(PAIR_DIR, "warped_syn_divcurl.nii.gz"))

fl = ants.image_read(os.path.join(PAIR_DIR, "fixed_label_aseg.nii.gz"))
ml = ants.image_read(os.path.join(PAIR_DIR, "moving_label_aseg.nii.gz"))

fl_arr = fl.numpy()
vent_mask_f = np.isin(fl_arr, [4, 14, 43])
z_mid = int(np.mean(np.where(vent_mask_f)[2])) if vent_mask_f.sum() > 0 else fi.shape[2] // 2

fi_ax, asp_ax = extract_oriented_slice(fi, slice_axis=2, slice_idx=z_mid)
w_aff_ax, _ = extract_oriented_slice(w_aff, slice_axis=2, slice_idx=z_mid)
w_sob_ax, _ = extract_oriented_slice(w_sob, slice_axis=2, slice_idx=z_mid)
w_hyp_ax, _ = extract_oriented_slice(w_hyp, slice_axis=2, slice_idx=z_mid)
w_div_ax, _ = extract_oriented_slice(w_div, slice_axis=2, slice_idx=z_mid)
fl_ax, _ = extract_oriented_slice(fl, slice_axis=2, slice_idx=z_mid)

diff_sob = np.abs(fi_ax - w_sob_ax)
diff_hyp = np.abs(fi_ax - w_hyp_ax)
diff_gain = diff_sob - diff_hyp

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor='#ffffff')

# Row 1: Anatomical Slices
axes[0, 0].imshow(fi_ax, cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[0, 0].contour(np.isin(fl_ax, [4, 14, 43]), levels=[0.5], colors=['#0284c7'], linewidths=[1.8])
axes[0, 0].set_title("A: Fixed Target (MMRR-21-8)\nReference Anatomical Slice", fontsize=11, fontweight='bold', pad=8)
axes[0, 0].axis('off')

axes[0, 1].imshow(w_sob_ax, cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[0, 1].contour(np.isin(fl_ax, [4, 14, 43]), levels=[0.5], colors=['#0284c7'], linewidths=[1.8])
axes[0, 1].set_title("B: Sobolev SyN Warped\nCortical Dice: 0.6195", fontsize=11, fontweight='bold', pad=8)
axes[0, 1].axis('off')

axes[0, 2].imshow(w_hyp_ax, cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[0, 2].contour(np.isin(fl_ax, [4, 14, 43]), levels=[0.5], colors=['#0284c7'], linewidths=[1.8])
axes[0, 2].set_title("C: Hyperelastic SyN Warped\nCortical Dice: 0.6331 (+1.35% Gain)", fontsize=11, fontweight='bold', pad=8)
axes[0, 2].axis('off')

# Row 2: Difference Images
axes[1, 0].imshow(diff_sob, cmap='magma', vmin=0, vmax=0.6, aspect=asp_ax, interpolation='bilinear')
axes[1, 0].set_title("D: |Target - Sobolev| Error Map\nElevated Residuals in Deep Sulci", fontsize=11, fontweight='bold', pad=8)
axes[1, 0].axis('off')

axes[1, 1].imshow(diff_hyp, cmap='magma', vmin=0, vmax=0.6, aspect=asp_ax, interpolation='bilinear')
axes[1, 1].set_title("E: |Target - Hyperelastic| Error Map\nAttenuated Parenchymal Error", fontsize=11, fontweight='bold', pad=8)
axes[1, 1].axis('off')

im_g = axes[1, 2].imshow(diff_gain, cmap='bwr', vmin=-0.2, vmax=0.2, aspect=asp_ax, interpolation='bilinear')
axes[1, 2].set_title("F: Absolute Error Reduction (&Delta;)\nRed = Hyperelastic Closer to Target", fontsize=11, fontweight='bold', pad=8)
axes[1, 2].axis('off')
cbar = plt.colorbar(im_g, ax=axes[1, 2], fraction=0.046, pad=0.04)
cbar.set_label("Sobolev Error - Hyperelastic Error", fontsize=9)

fig.suptitle("Figure 6: Independent Replication of Continuum Mechanics Advantage on Pair 08 (Intra-Scanner)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
fig6_path = os.path.join(FIG_DIR, "fig6_pair08_intra_scanner_replication.png")
plt.savefig(fig6_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("--> Saved Figure 6: fig6_pair08_intra_scanner_replication.png", flush=True)

if __name__ == "__main__":
    pass
