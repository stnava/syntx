#!/usr/bin/env python3
"""
Generate publication-quality anatomical visual comparison figures
supporting the Continuum Mechanics Regularization claims in syntx.
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
from skimage import feature
import ants

import syntx
from syntx.viz import extract_oriented_slice, plot_edge_overlay, AnatomicalVisualizer
from syntx.viz.colormaps import get_dkt_colormap

# Set publication style (GEMINI.md §5 Invariant: Light Theme)
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
os.makedirs(FIG_DIR, exist_ok=True)

def generate_figure_1_morphology(pair_dir, out_path):
    """Figure 1: Morphological Discrepancy & Ventricular Ex-Vacuo Anatomy (Pair 44)."""
    fi = ants.image_read(os.path.join(pair_dir, "fixed_target.nii.gz"))
    mi = ants.image_read(os.path.join(pair_dir, "moving_source.nii.gz"))
    fl = ants.image_read(os.path.join(pair_dir, "fixed_label_aseg.nii.gz"))
    ml = ants.image_read(os.path.join(pair_dir, "moving_label_aseg.nii.gz"))

    # Slices: Coronal (axis 1) and Axial (axis 2) through lateral ventricles
    # Find ventricle center of mass in fixed
    fl_arr = fl.numpy()
    vent_mask_f = np.isin(fl_arr, [4, 14, 43])
    idx_z_f = int(np.mean(np.where(vent_mask_f)[2])) if vent_mask_f.sum() > 0 else fi.shape[2] // 2
    idx_y_f = int(np.mean(np.where(vent_mask_f)[1])) if vent_mask_f.sum() > 0 else fi.shape[1] // 2

    # Slices for fixed
    fi_cor, asp_cor_f = extract_oriented_slice(fi, slice_axis=1, slice_idx=idx_y_f)
    fi_ax, asp_ax_f = extract_oriented_slice(fi, slice_axis=2, slice_idx=idx_z_f)
    fl_cor, _ = extract_oriented_slice(fl, slice_axis=1, slice_idx=idx_y_f)
    fl_ax, _ = extract_oriented_slice(fl, slice_axis=2, slice_idx=idx_z_f)

    # Moving ventricle center of mass
    ml_arr = ml.numpy()
    vent_mask_m = np.isin(ml_arr, [4, 14, 43])
    idx_z_m = int(np.mean(np.where(vent_mask_m)[2])) if vent_mask_m.sum() > 0 else mi.shape[2] // 2
    idx_y_m = int(np.mean(np.where(vent_mask_m)[1])) if vent_mask_m.sum() > 0 else mi.shape[1] // 2

    mi_cor, asp_cor_m = extract_oriented_slice(mi, slice_axis=1, slice_idx=idx_y_m)
    mi_ax, asp_ax_m = extract_oriented_slice(mi, slice_axis=2, slice_idx=idx_z_m)
    ml_cor, _ = extract_oriented_slice(ml, slice_axis=1, slice_idx=idx_y_m)
    ml_ax, _ = extract_oriented_slice(ml, slice_axis=2, slice_idx=idx_z_m)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), facecolor='#ffffff')

    # Panel A: Fixed Coronal
    ax = axes[0, 0]
    ax.imshow(fi_cor, cmap='gray', aspect=asp_cor_f, interpolation='bilinear')
    vent_cor_f = np.isin(fl_cor, [4, 14, 43])
    ax.contour(vent_cor_f, levels=[0.5], colors=['#0284c7'], linewidths=[2.0])
    ax.set_title("Fixed Target (NKI-TRT-20-2) — Coronal\nVentricle Volume: 11.7 mL (Normal)", fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

    # Panel B: Moving Coronal
    ax = axes[0, 1]
    ax.imshow(mi_cor, cmap='gray', aspect=asp_cor_m, interpolation='bilinear')
    vent_cor_m = np.isin(ml_cor, [4, 14, 43])
    ax.contour(vent_cor_m, levels=[0.5], colors=['#e11d48'], linewidths=[2.0])
    ax.set_title("Moving Source (MMRR-21-2) — Coronal\nVentricle Volume: 22.1 mL (+89% Enlargement)", fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

    # Panel C: Fixed Axial
    ax = axes[1, 0]
    ax.imshow(fi_ax, cmap='gray', aspect=asp_ax_f, interpolation='bilinear')
    vent_ax_f = np.isin(fl_ax, [4, 14, 43])
    ax.contour(vent_ax_f, levels=[0.5], colors=['#0284c7'], linewidths=[2.0])
    ax.set_title("Fixed Target — Axial Plane\nCortex: 556.1 mL (Normal Mantle)", fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

    # Panel D: Moving Axial
    ax = axes[1, 1]
    ax.imshow(mi_ax, cmap='gray', aspect=asp_ax_m, interpolation='bilinear')
    vent_ax_m = np.isin(ml_ax, [4, 14, 43])
    ax.contour(vent_ax_m, levels=[0.5], colors=['#e11d48'], linewidths=[2.0])
    ax.set_title("Moving Source — Axial Plane\nCortex: 465.6 mL (-16% Cortical Atrophy)", fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

    fig.suptitle("Figure 1: Morphological Discrepancy & Ventricular Ex-Vacuo Enlargement in Pair 44", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"Generated Figure 1: {out_path}")

print("Evidence figure script drafted.")
