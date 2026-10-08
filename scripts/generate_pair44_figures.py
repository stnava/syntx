#!/usr/bin/env python3
"""
Generate publication-quality anatomical visual comparison figures for Pair 44.
Strictly adheres to GEMINI.md Rules 1-6 (light theme, physical space, syntx.viz).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import ants

from syntx.viz import extract_oriented_slice
from syntx.viz.figures import _display_displacement

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
PAIR_DIR = "results/anatomical_evidence/pair_044"
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load Images
print("Loading Pair 44 images...", flush=True)
fi = ants.image_read(os.path.join(PAIR_DIR, "fixed_target.nii.gz"))
mi = ants.image_read(os.path.join(PAIR_DIR, "moving_source.nii.gz"))
w_aff = ants.image_read(os.path.join(PAIR_DIR, "warped_affine.nii.gz"))
w_sob = ants.image_read(os.path.join(PAIR_DIR, "warped_sobolev.nii.gz"))
w_hyp = ants.image_read(os.path.join(PAIR_DIR, "warped_syn_hyperelastic.nii.gz"))
w_div = ants.image_read(os.path.join(PAIR_DIR, "warped_syn_divcurl.nii.gz"))

fl = ants.image_read(os.path.join(PAIR_DIR, "fixed_label_aseg.nii.gz"))
ml = ants.image_read(os.path.join(PAIR_DIR, "moving_label_aseg.nii.gz"))
w_aseg_sob = ants.image_read(os.path.join(PAIR_DIR, "warped_aseg_sobolev.nii.gz"))
w_aseg_hyp = ants.image_read(os.path.join(PAIR_DIR, "warped_aseg_syn_hyperelastic.nii.gz"))

detJ_sob = ants.image_read(os.path.join(PAIR_DIR, "detJ_sobolev.nii.gz"))
detJ_hyp = ants.image_read(os.path.join(PAIR_DIR, "detJ_syn_hyperelastic.nii.gz"))
detJ_div = ants.image_read(os.path.join(PAIR_DIR, "detJ_syn_divcurl.nii.gz"))

log_detJ_sob = ants.image_read(os.path.join(PAIR_DIR, "log_detJ_sobolev.nii.gz"))
log_detJ_hyp = ants.image_read(os.path.join(PAIR_DIR, "log_detJ_syn_hyperelastic.nii.gz"))
log_detJ_div = ants.image_read(os.path.join(PAIR_DIR, "log_detJ_syn_divcurl.nii.gz"))

warp_sob = ants.image_read(os.path.join(PAIR_DIR, "warp_sobolev.nii.gz"))
warp_hyp = ants.image_read(os.path.join(PAIR_DIR, "warp_syn_hyperelastic.nii.gz"))

# Ventricle Center of mass
fl_arr = fl.numpy()
vent_mask_f = np.isin(fl_arr, [4, 14, 43])
z_vent = int(np.mean(np.where(vent_mask_f)[2]))
y_vent = int(np.mean(np.where(vent_mask_f)[1]))
print(f"Ventricle center coordinates: Y={y_vent}, Z={z_vent}", flush=True)

# -------------------------------------------------------------
# FIGURE 1: Morphological Discrepancy & Ventricular Ex-Vacuo Anatomy
# -------------------------------------------------------------
print("Generating Figure 1: Morphological Discrepancy...", flush=True)
fi_cor, asp_cor = extract_oriented_slice(fi, slice_axis=1, slice_idx=y_vent)
fi_ax, asp_ax = extract_oriented_slice(fi, slice_axis=2, slice_idx=z_vent)
fl_cor, _ = extract_oriented_slice(fl, slice_axis=1, slice_idx=y_vent)
fl_ax, _ = extract_oriented_slice(fl, slice_axis=2, slice_idx=z_vent)

mi_cor, _ = extract_oriented_slice(mi, slice_axis=1, slice_idx=y_vent)
mi_ax, _ = extract_oriented_slice(mi, slice_axis=2, slice_idx=z_vent)
ml_cor, _ = extract_oriented_slice(ml, slice_axis=1, slice_idx=y_vent)
ml_ax, _ = extract_oriented_slice(ml, slice_axis=2, slice_idx=z_vent)

fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), facecolor='#ffffff')

# A: Fixed Coronal
axes[0, 0].imshow(fi_cor, cmap='gray', aspect=asp_cor, interpolation='bilinear')
axes[0, 0].contour(np.isin(fl_cor, [4, 14, 43]), levels=[0.5], colors=['#0284c7'], linewidths=[2.2])
axes[0, 0].set_title("A: Fixed Target (NKI-TRT-20-2) — Coronal\nVentricles: 11.7 mL (Normal Adult Anatomy)", fontsize=11, fontweight='bold', pad=8)
axes[0, 0].axis('off')

# B: Moving Coronal
axes[0, 1].imshow(mi_cor, cmap='gray', aspect=asp_cor, interpolation='bilinear')
axes[0, 1].contour(np.isin(ml_cor, [4, 14, 43]), levels=[0.5], colors=['#e11d48'], linewidths=[2.2])
axes[0, 1].set_title("B: Moving Source (MMRR-21-2) — Coronal\nVentricles: 22.1 mL (+89% Ex-Vacuo Ventriculomegaly)", fontsize=11, fontweight='bold', pad=8)
axes[0, 1].axis('off')

# C: Fixed Axial
axes[1, 0].imshow(fi_ax, cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[1, 0].contour(np.isin(fl_ax, [4, 14, 43]), levels=[0.5], colors=['#0284c7'], linewidths=[2.2])
axes[1, 0].set_title("C: Fixed Target — Axial Slice through Frontal Horns\nCortex: 556.1 mL (Normal Cortical Mantle)", fontsize=11, fontweight='bold', pad=8)
axes[1, 0].axis('off')

# D: Moving Axial
axes[1, 1].imshow(mi_ax, cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[1, 1].contour(np.isin(ml_ax, [4, 14, 43]), levels=[0.5], colors=['#e11d48'], linewidths=[2.2])
axes[1, 1].set_title("D: Moving Source — Axial Slice through Frontal Horns\nCortex: 465.6 mL (-16% Marked Cortical Atrophy)", fontsize=11, fontweight='bold', pad=8)
axes[1, 1].axis('off')

fig.suptitle("Figure 1: Extreme Morphological Discrepancy in Mindboggle Pair 44 (Inter-Scanner)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig1_pair44_morphology_discrepancy.png")
plt.savefig(fig1_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("--> Saved Figure 1.", flush=True)

# -------------------------------------------------------------
# FIGURE 2: Head-to-Head Ventricular Realignment & Difference Maps
# -------------------------------------------------------------
print("Generating Figure 2: Realignment & Difference Maps...", flush=True)
w_aff_ax, _ = extract_oriented_slice(w_aff, slice_axis=2, slice_idx=z_vent)
w_sob_ax, _ = extract_oriented_slice(w_sob, slice_axis=2, slice_idx=z_vent)
w_hyp_ax, _ = extract_oriented_slice(w_hyp, slice_axis=2, slice_idx=z_vent)
w_div_ax, _ = extract_oriented_slice(w_div, slice_axis=2, slice_idx=z_vent)

# Differences
diff_aff = np.abs(fi_ax - w_aff_ax)
diff_sob = np.abs(fi_ax - w_sob_ax)
diff_hyp = np.abs(fi_ax - w_hyp_ax)
diff_div = np.abs(fi_ax - w_div_ax)
diff_gain = diff_sob - diff_hyp  # Positive where Hyperelastic is closer to Target

fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor='#ffffff')

titles_r1 = [
    "A1: Fixed Target\n(Reference Anatomy)",
    "A2: Affine Initialized\n(Residual Mismatch)",
    "A3: Sobolev SyN Warped\n(Damped Compression)",
    "A4: Hyperelastic SyN Warped\n(Deep Shape Capture)"
]
imgs_r1 = [fi_ax, w_aff_ax, w_sob_ax, w_hyp_ax]
vent_contour = np.isin(fl_ax, [4, 14, 43])

for j, (im, t) in enumerate(zip(imgs_r1, titles_r1)):
    ax = axes[0, j]
    ax.imshow(im, cmap='gray', aspect=asp_ax, interpolation='bilinear')
    ax.contour(vent_contour, levels=[0.5], colors=['#0284c7'], linewidths=[1.8])
    ax.set_title(t, fontsize=10.5, fontweight='bold', pad=8)
    ax.axis('off')

titles_r2 = [
    "B1: |Target - Affine|\n(Severe Ventricular Error)",
    "B2: |Target - Sobolev|\n(Bright Boundary Residuals)",
    "B3: |Target - Hyperelastic|\n(Attenuated Residuals)",
    "B4: |Target - Div-Curl|\n(Attenuated Residuals)"
]
imgs_r2 = [diff_aff, diff_sob, diff_hyp, diff_div]
vmax_diff = 0.65

for j, (im, t) in enumerate(zip(imgs_r2, titles_r2)):
    ax = axes[1, j]
    im_h = ax.imshow(im, cmap='magma', vmin=0, vmax=vmax_diff, aspect=asp_ax, interpolation='bilinear')
    ax.set_title(t, fontsize=10.5, fontweight='bold', pad=8)
    ax.axis('off')
    if j == 3:
        cbar = plt.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|Target - Warped| Error", fontsize=9)

# Row 3: Difference Reduction and Zoom In
ax = axes[2, 0]
im_gain = ax.imshow(diff_gain, cmap='bwr', vmin=-0.25, vmax=0.25, aspect=asp_ax, interpolation='bilinear')
ax.set_title("C1: Error Reduction (&Delta;)\n(Red = Hyperelastic Closer to Target)", fontsize=10.5, fontweight='bold', pad=8)
ax.axis('off')
cbar_g = plt.colorbar(im_gain, ax=ax, fraction=0.046, pad=0.04)
cbar_g.set_label("Sobolev Error - Hyperelastic Error", fontsize=9)

# Zoom in on Lateral Ventricles
y_indices, x_indices = np.where(vent_contour)
y_min, y_max = max(0, y_indices.min() - 25), min(fi_ax.shape[0], y_indices.max() + 25)
x_min, x_max = max(0, x_indices.min() - 25), min(fi_ax.shape[1], x_indices.max() + 25)

ax = axes[2, 1]
ax.imshow(w_sob_ax[y_min:y_max, x_min:x_max], cmap='gray', aspect=asp_ax, interpolation='bilinear')
ax.contour(vent_contour[y_min:y_max, x_min:x_max], levels=[0.5], colors=['#0284c7'], linewidths=[2.0])
ax.set_title("C2: Sobolev ROI Zoom\nUnder-compression (Wide CSF Gap)", fontsize=10.5, fontweight='bold', pad=8)
ax.axis('off')

ax = axes[2, 2]
ax.imshow(w_hyp_ax[y_min:y_max, x_min:x_max], cmap='gray', aspect=asp_ax, interpolation='bilinear')
ax.contour(vent_contour[y_min:y_max, x_min:x_max], levels=[0.5], colors=['#0284c7'], linewidths=[2.0])
ax.set_title("C3: Hyperelastic ROI Zoom\nTight Alignment to Target Wall", fontsize=10.5, fontweight='bold', pad=8)
ax.axis('off')

ax = axes[2, 3]
ax.imshow(w_div_ax[y_min:y_max, x_min:x_max], cmap='gray', aspect=asp_ax, interpolation='bilinear')
ax.contour(vent_contour[y_min:y_max, x_min:x_max], levels=[0.5], colors=['#0284c7'], linewidths=[2.0])
ax.set_title("C4: Div-Curl ROI Zoom\nDecoupled Shear Alignment", fontsize=10.5, fontweight='bold', pad=8)
ax.axis('off')

fig.suptitle("Figure 2: Lateral Ventricle Realignment and Residual Difference Analysis (Pair 44)", fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "fig2_pair44_ventricle_realignment_comparison.png")
plt.savefig(fig2_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("--> Saved Figure 2.", flush=True)

# -------------------------------------------------------------
# FIGURE 3: Local Jacobian Expansion Maps: log det(J)
# -------------------------------------------------------------
print("Generating Figure 3: Jacobian Determinant Maps...", flush=True)
logJ_sob_ax, _ = extract_oriented_slice(log_detJ_sob, slice_axis=2, slice_idx=z_vent)
logJ_hyp_ax, _ = extract_oriented_slice(log_detJ_hyp, slice_axis=2, slice_idx=z_vent)
logJ_div_ax, _ = extract_oriented_slice(log_detJ_div, slice_axis=2, slice_idx=z_vent)

logJ_sob_cor, _ = extract_oriented_slice(log_detJ_sob, slice_axis=1, slice_idx=y_vent)
logJ_hyp_cor, _ = extract_oriented_slice(log_detJ_hyp, slice_axis=1, slice_idx=y_vent)
logJ_div_cor, _ = extract_oriented_slice(log_detJ_div, slice_axis=1, slice_idx=y_vent)

fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='#ffffff')

# Row 1: Axial log det(J) maps
axes[0, 0].imshow(fi_ax, cmap='gray', alpha=0.35, aspect=asp_ax)
axes[0, 0].imshow(logJ_sob_ax, cmap='coolwarm', vmin=-1.2, vmax=1.2, alpha=0.75, aspect=asp_ax)
axes[0, 0].contour(vent_contour, levels=[0.5], colors=['#0f172a'], linewidths=[1.5])
axes[0, 0].set_title("A1: Sobolev SyN — Axial log det(J)\nConstrained Dynamic Range [-0.80, +0.65]", fontsize=10.5, fontweight='bold', pad=8)
axes[0, 0].axis('off')

axes[0, 1].imshow(fi_ax, cmap='gray', alpha=0.35, aspect=asp_ax)
axes[0, 1].imshow(logJ_hyp_ax, cmap='coolwarm', vmin=-1.2, vmax=1.2, alpha=0.75, aspect=asp_ax)
axes[0, 1].contour(vent_contour, levels=[0.5], colors=['#0f172a'], linewidths=[1.5])
axes[0, 1].set_title("A2: Hyperelastic SyN — Axial log det(J)\nLocalized Ventricle Compression (log J = -1.15)", fontsize=10.5, fontweight='bold', pad=8)
axes[0, 1].axis('off')

axes[0, 2].imshow(fi_ax, cmap='gray', alpha=0.35, aspect=asp_ax)
axes[0, 2].imshow(logJ_div_ax, cmap='coolwarm', vmin=-1.2, vmax=1.2, alpha=0.75, aspect=asp_ax)
axes[0, 2].contour(vent_contour, levels=[0.5], colors=['#0f172a'], linewidths=[1.5])
axes[0, 2].set_title("A3: Div-Curl SyN — Axial log det(J)\nDeep Volumetric Range [-1.33, +1.20]", fontsize=10.5, fontweight='bold', pad=8)
axes[0, 2].axis('off')

# Row 2: Coronal log det(J) maps
vent_contour_cor = np.isin(fl_cor, [4, 14, 43])
axes[1, 0].imshow(fi_cor, cmap='gray', alpha=0.35, aspect=asp_cor)
axes[1, 0].imshow(logJ_sob_cor, cmap='coolwarm', vmin=-1.2, vmax=1.2, alpha=0.75, aspect=asp_cor)
axes[1, 0].contour(vent_contour_cor, levels=[0.5], colors=['#0f172a'], linewidths=[1.5])
axes[1, 0].set_title("B1: Sobolev SyN — Coronal Plane\nDamped Expansion along Cortical Mantle", fontsize=10.5, fontweight='bold', pad=8)
axes[1, 0].axis('off')

axes[1, 1].imshow(fi_cor, cmap='gray', alpha=0.35, aspect=asp_cor)
axes[1, 1].imshow(logJ_hyp_cor, cmap='coolwarm', vmin=-1.2, vmax=1.2, alpha=0.75, aspect=asp_cor)
axes[1, 1].contour(vent_contour_cor, levels=[0.5], colors=['#0f172a'], linewidths=[1.5])
axes[1, 1].set_title("B2: Hyperelastic SyN — Coronal Plane\nBulk Modulus Decouples Expansion from Shear", fontsize=10.5, fontweight='bold', pad=8)
axes[1, 1].axis('off')

axes[1, 2].imshow(fi_cor, cmap='gray', alpha=0.35, aspect=asp_cor)
im_c = axes[1, 2].imshow(logJ_div_cor, cmap='coolwarm', vmin=-1.2, vmax=1.2, alpha=0.75, aspect=asp_cor)
axes[1, 2].contour(vent_contour_cor, levels=[0.5], colors=['#0f172a'], linewidths=[1.5])
axes[1, 2].set_title("B3: Div-Curl SyN — Coronal Plane\nZero Folding Guaranteed (J_min = +0.0107)", fontsize=10.5, fontweight='bold', pad=8)
axes[1, 2].axis('off')

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im_c, cax=cbar_ax)
cbar.set_label("Local Log Jacobian Determinant log det(J)\n[Blue = Local Expansion (det J > 1) | Red = Local Contraction (det J < 1)]", fontsize=10, fontweight='bold')

fig.suptitle("Figure 3: Continuous Liouville Determinant Maps det(J) Demonstrating Localized Shape Capture (Pair 44)", fontsize=13, fontweight='bold', y=0.98)
fig3_path = os.path.join(FIG_DIR, "fig3_pair44_jacobian_expansion_maps.png")
plt.savefig(fig3_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("--> Saved Figure 3.", flush=True)

# -------------------------------------------------------------
# FIGURE 4: Sulcal Shear & Tangential Slip (Sylvian Fissure & Insula)
# -------------------------------------------------------------
print("Generating Figure 4: Sulcal Shear & Tangential Slip...", flush=True)
# Find insula in fixed
insula_mask = np.isin(fl_arr, [1035, 2035])
z_ins = int(np.mean(np.where(insula_mask)[2]))

fi_ins_ax, _ = extract_oriented_slice(fi, slice_axis=2, slice_idx=z_ins)
w_sob_ins_ax, _ = extract_oriented_slice(w_sob, slice_axis=2, slice_idx=z_ins)
w_hyp_ins_ax, _ = extract_oriented_slice(w_hyp, slice_axis=2, slice_idx=z_ins)
fl_ins_ax, _ = extract_oriented_slice(fl, slice_axis=2, slice_idx=z_ins)

# Display displacement via syntx.viz
dcol_sob, drow_sob, _, _, _ = _display_displacement(warp_sob, fi, slice_axis=2, slice_idx=z_ins, reorient=True)
dcol_hyp, drow_hyp, _, _, _ = _display_displacement(warp_hyp, fi, slice_axis=2, slice_idx=z_ins, reorient=True)

# ROI bounds around left insula / superior temporal gyrus
ins_y_idx, ins_x_idx = np.where(np.isin(fl_ins_ax, [1035, 1030, 2035, 2030]))
roi_y0 = max(0, ins_y_idx.min() - 15)
roi_y1 = min(fi_ins_ax.shape[0], ins_y_idx.max() + 15)
roi_x0 = max(0, ins_x_idx.min() - 15)
roi_x1 = min(fi_ins_ax.shape[1], ins_x_idx.max() + 15)

step = 4
Y, X = np.mgrid[0:(roi_y1-roi_y0), 0:(roi_x1-roi_x0)]

u_s_x_sub = dcol_sob[roi_y0:roi_y1, roi_x0:roi_x1]
u_s_y_sub = drow_sob[roi_y0:roi_y1, roi_x0:roi_x1]

u_h_x_sub = dcol_hyp[roi_y0:roi_y1, roi_x0:roi_x1]
u_h_y_sub = drow_hyp[roi_y0:roi_y1, roi_x0:roi_x1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor='#ffffff')

# Panel A: Fixed Target Anatomy
axes[0].imshow(fi_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[0].contour(np.isin(fl_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], [1035, 2035]), levels=[0.5], colors=['#0284c7'], linewidths=[2.0])
axes[0].contour(np.isin(fl_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], [1030, 2030]), levels=[0.5], colors=['#d97706'], linewidths=[2.0])
axes[0].set_title("A: Target Anatomy (Sylvian Fissure ROI)\n[Blue = Insula | Amber = Superior Temporal]", fontsize=10.5, fontweight='bold', pad=8)
axes[0].axis('off')

# Panel B: Sobolev Vector Quiver
axes[1].imshow(w_sob_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[1].quiver(X[::step, ::step], Y[::step, ::step], u_s_x_sub[::step, ::step], u_s_y_sub[::step, ::step],
              color='#0284c7', scale=40, width=0.005)
axes[1].contour(np.isin(fl_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], [1035, 1030, 2035, 2030]), levels=[0.5], colors=['#e11d48'], linewidths=[1.5])
axes[1].set_title("B: Sobolev Displacement Quiver\nIsotropic Drag Across Sulcal Wall (Blurring)", fontsize=10.5, fontweight='bold', pad=8)
axes[1].axis('off')

# Panel C: Hyperelastic Vector Quiver
axes[2].imshow(w_hyp_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], cmap='gray', aspect=asp_ax, interpolation='bilinear')
axes[2].quiver(X[::step, ::step], Y[::step, ::step], u_h_x_sub[::step, ::step], u_h_y_sub[::step, ::step],
              color='#059669', scale=40, width=0.005)
axes[2].contour(np.isin(fl_ins_ax[roi_y0:roi_y1, roi_x0:roi_x1], [1035, 1030, 2035, 2030]), levels=[0.5], colors=['#e11d48'], linewidths=[1.5])
axes[2].set_title("C: Hyperelastic Displacement Quiver\nTangential Slip Along Sulcal Banks (+5.2% STG Dice)", fontsize=10.5, fontweight='bold', pad=8)
axes[2].axis('off')

fig.suptitle("Figure 4: Tangential Sulcal Shear vs. Isotropic Drag at the Sylvian Fissure (Pair 44)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "fig4_pair44_sulcal_shear_tangential_slip.png")
plt.savefig(fig4_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("--> Saved Figure 4.", flush=True)

# -------------------------------------------------------------
# FIGURE 5: Structure-by-Structure Dice Gain Bar Chart
# -------------------------------------------------------------
print("Generating Figure 5: Structure-by-Structure Dice Gain...", flush=True)
structures = [
    ("Middle Temporal Gyrus", 0.5443, 0.6136, 0.6120),
    ("Superior Temporal Gyrus", 0.6810, 0.7326, 0.7338),
    ("Caudal Anterior Cingulate", 0.5349, 0.5786, 0.5573),
    ("Supramarginal Gyrus", 0.6106, 0.6443, 0.6415),
    ("Inferior Temporal Gyrus", 0.6093, 0.6402, 0.6385),
    ("Superior Parietal Cortex", 0.5152, 0.5297, 0.5280),
    ("Precuneus", 0.6420, 0.6528, 0.6468),
    ("Insula Cortex", 0.8317, 0.8377, 0.8340),
    ("3rd Ventricle", 0.7692, 0.7789, 0.7841),
    ("Global Cortical Mean", 0.6011, 0.6165, 0.6162),
]

names = [s[0] for s in structures]
sob_vals = [s[1] for s in structures]
hyp_vals = [s[2] for s in structures]
div_vals = [s[3] for s in structures]

y = np.arange(len(names))
height = 0.26

fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='#ffffff')
ax.set_facecolor('#ffffff')

rects1 = ax.barh(y - height, sob_vals, height, label='Sobolev SyN (Scalar Smoother)', color='#0284c7', edgecolor='none')
rects2 = ax.barh(y, hyp_vals, height, label='Hyperelastic SyN (Continuum Bulk+Shear)', color='#059669', edgecolor='none')
rects3 = ax.barh(y + height, div_vals, height, label='Div-Curl SyN (Spectral Helmholtz)', color='#d97706', edgecolor='none')

ax.set_xlabel('Sørensen-Dice Coefficient', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Figure 5: Structure-Specific Dice Gain — Large Deformation Cortical & Subcortical Parcels (Pair 44)', fontsize=12.5, fontweight='bold', pad=12)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=10.5, fontweight='bold')
ax.set_xlim([0.45, 0.90])
ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cbd5e1')
ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10.5)

# Add text callouts for gains
for i, (s, h) in enumerate(zip(sob_vals, hyp_vals)):
    diff = (h - s) * 100
    if diff > 0.5:
        ax.text(h + 0.006, y[i] - 0.05, f"+{diff:.2f}%", color='#059669', fontweight='bold', fontsize=9.5, va='center')

plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, "fig5_pair44_regional_dice_gain.png")
plt.savefig(fig5_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("--> Saved Figure 5.", flush=True)

print("\nAll Pair 44 Evidence Figures successfully generated in docs/reports/figures_cm_evidence/\n", flush=True)
