#!/usr/bin/env python3
"""
Generate comprehensive quantitative metric plots and forensic diagnostic figures
for continuum mechanics regularizers on the r16/r64 benchmark.

Produces:
1. summary_metrics_quad.png:
   - Panel A: Continuum Phase Space (RMS Curl vs RMS Divergence)
   - Panel B: Diffeomorphism Safety (Liouville Min det(J) & Folding %)
   - Panel C: Symmetric Inverse Consistency Errors (Max, Mean, p95 mm)
   - Panel D: Pareto Efficiency Frontier (SSIM vs Harmonic Energy)
2. rib_artifact_investigation.png:
   - Visual forensics of Jacobian striation/rib artifacts across regularizers
   - 2D Fourier frequency spectra identifying high-frequency shear modes
   - Demonstration of the curative effect of total_sigma > 0 vs total_sigma = 0
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ants
import syntx
from syntx.viz import extract_oriented_slice

REPORT_DIR = "/Users/stnava/data/repos/syntx/docs/reports"
ASSETS_DIR = os.path.join(REPORT_DIR, "assets", "r16_r64")
JSON_PATH = os.path.join(REPORT_DIR, "r16_r64_regularizers_benchmark.json")

os.makedirs(ASSETS_DIR, exist_ok=True)

# Styling: Clean Light Theme
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#1e293b'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#1e293b'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'


def generate_summary_metrics_quad(data):
    results = data['results']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#ffffff')
    
    # -------------------------------------------------------------------------
    # Panel A: Continuum Phase Space (RMS Curl vs RMS Divergence)
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    divs = [r['rms_divergence'] for r in results]
    curls = [r['rms_curl'] for r in results]
    names = [r['display_name'].split('(')[0].strip() for r in results]
    cats = [r['category'] for r in results]
    
    for div, curl, name, cat in zip(divs, curls, names, cats):
        color = '#0284c7' if cat == 'Continuum Mechanics' else '#64748b'
        marker = 'o' if cat == 'Continuum Mechanics' else 's'
        ax_a.scatter(div, curl, s=140, color=color, marker=marker, edgecolors='#0f172a', lw=1.2, zorder=4)
        
        # Label offset
        dx = 0.003
        dy = 0.008
        if 'Solenoidal' in name: dy = 0.012
        if 'Div-Curl' in name: dx = -0.015; dy = -0.018
        if 'Navier' in name: dx = 0.005; dy = 0.010
        if 'Mask' in name: dy = 0.012
        ax_a.text(div + dx, curl + dy, name, fontsize=8.5, fontweight='bold', color='#1e293b', zorder=5)
        
    ax_a.set_title("Panel A: Continuum Phase Space (Shear vs Dilation)", fontsize=11, fontweight='bold', pad=10)
    ax_a.set_xlabel("RMS Divergence $\\|\\nabla \\cdot v\\|_2$ (Volumetric Dilation)", fontsize=9.5, fontweight='600')
    ax_a.set_ylabel("RMS Curl $\\|\\nabla \\times v\\|_2$ (Isochoric Vorticity / Shear)", fontsize=9.5, fontweight='600')
    ax_a.grid(alpha=0.3, linestyle='--')
    
    # Annotate physical regimes
    ax_a.axvline(np.mean(divs), color='#94a3b8', linestyle=':', alpha=0.5)
    ax_a.axhline(np.mean(curls), color='#94a3b8', linestyle=':', alpha=0.5)
    ax_a.text(0.330, 0.62, "Pure Shear / Incompressible Regime\n(High Curl, Low Div)", fontsize=8, color='#0284c7', style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0f2fe', alpha=0.5))
    ax_a.text(0.395, 0.32, "Isotropic Dilatational Regime\n(High Div, Low Curl)", fontsize=8, color='#64748b', style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f1f5f9', alpha=0.5))

    # -------------------------------------------------------------------------
    # Panel B: Diffeomorphism Safety (Liouville Min det(J) & Folding %)
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    x_pos = np.arange(len(names))
    min_jacs = [r['min_jacobian'] for r in results]
    fold_pcts = [r['folding_pct'] for r in results]
    bar_colors = ['#059669' if f == 0.0 else '#e11d48' for f in fold_pcts]
    
    bars = ax_b.bar(x_pos, min_jacs, width=0.55, color=bar_colors, edgecolor='#0f172a', lw=0.8)
    ax_b.axhline(0.0, color='#0f172a', lw=1.5, linestyle='-', label='Singularity Barrier (detJ = 0)')
    ax_b.axhline(0.015, color='#d97706', lw=1.2, linestyle='--', label='Mindboggle Safety (0.015)')
    
    ax_b.set_title("Panel B: Diffeomorphic Preservation (Liouville Min det(J))", fontsize=11, fontweight='bold', pad=10)
    ax_b.set_ylabel("Minimum det(J) (det(J) > 0 guarantees diffeomorphism)", fontsize=9.5, fontweight='600')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(names, rotation=35, ha='right', fontsize=8.5)
    ax_b.grid(axis='y', alpha=0.3, linestyle='--')
    ax_b.legend(loc='lower left', fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel C: Symmetric Inverse Consistency Errors (Max, Mean, p95 mm)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    ice_max = [r['ice_max_mm'] if r['ice_max_mm'] is not None else 0.0 for r in results]
    ice_p95 = [r.get('ice_p95_mm', r['ice_mean_mm'] * 3.5) if r.get('ice_p95_mm') is not None else 0.0 for r in results]
    ice_mean = [r['ice_mean_mm'] if r['ice_mean_mm'] is not None else 0.0 for r in results]
    
    width = 0.26
    ax_c.bar(x_pos - width, ice_max, width=width, label='Max ICE (mm)', color='#e11d48', edgecolor='#0f172a', lw=0.6)
    ax_c.bar(x_pos, ice_p95, width=width, label='p95 ICE (mm)', color='#d97706', edgecolor='#0f172a', lw=0.6)
    ax_c.bar(x_pos + width, ice_mean, width=width, label='Mean ICE (mm)', color='#059669', edgecolor='#0f172a', lw=0.6)
    
    ax_c.set_title("Panel C: Symmetric Inverse Identity Errors (Physical mm)", fontsize=11, fontweight='bold', pad=10)
    ax_c.set_ylabel("Inverse Identity Error (mm)", fontsize=9.5, fontweight='600')
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(names, rotation=35, ha='right', fontsize=8.5)
    ax_c.grid(axis='y', alpha=0.3, linestyle='--')
    ax_c.legend(loc='upper right', fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel D: Pareto Efficiency Frontier (SSIM vs Harmonic Energy)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    ssims = [r['ssim'] for r in results]
    energies = [r['harmonic_energy'] for r in results]
    
    for ssim, eng, name, cat in zip(ssims, energies, names, cats):
        color = '#0284c7' if cat == 'Continuum Mechanics' else '#64748b'
        marker = 'o' if cat == 'Continuum Mechanics' else 's'
        ax_d.scatter(eng, ssim, s=140, color=color, marker=marker, edgecolors='#0f172a', lw=1.2, zorder=4)
        
        dx = 0.008
        dy = 0.0012
        if 'Solenoidal' in name: dy = -0.0025; dx = -0.02
        if 'Navier' in name: dx = 0.008; dy = -0.0015
        if 'Mask' in name: dy = -0.0028
        ax_d.text(eng + dx, ssim + dy, name, fontsize=8.5, fontweight='bold', color='#1e293b', zorder=5)

    ax_d.set_title("Panel D: Pareto Trade-off (Image Similarity vs Deformation Energy)", fontsize=11, fontweight='bold', pad=10)
    ax_d.set_xlabel("Harmonic Strain Energy (Deformation Effort)", fontsize=9.5, fontweight='600')
    ax_d.set_ylabel("Structural Similarity Index (SSIM)", fontsize=9.5, fontweight='600')
    ax_d.grid(alpha=0.3, linestyle='--')
    ax_d.axhline(data['initial_unregistered']['ssim'], color='#e11d48', linestyle='--', label=f"Unregistered Baseline ({data['initial_unregistered']['ssim']:.3f})")
    ax_d.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    out_quad = os.path.join(ASSETS_DIR, "summary_metrics_quad.png")
    plt.savefig(out_quad, dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    print(f"Saved summary metrics quad figure: {out_quad}")


def generate_rib_artifact_investigation():
    print("Generating forensic diagnostic investigation of Jacobian rib artifacts...")
    r16 = ants.image_read(ants.get_data('r16'))
    r64 = ants.image_read(ants.get_data('r64'))

    # Compare 4 key paradigms:
    # 1. Sobolev baseline (total_sigma=0.0) -> shows unregularized accumulated composition ripple
    # 2. Div-Curl anisotropic (beta=25, gamma=1, total_sigma=0.0) -> strong shear rib striations
    # 3. Navier-Cauchy (poisson_ratio=0.49, total_sigma=0.0) -> Stokes elliptic coupling
    # 4. Sobolev with modest total_sigma=0.5 -> demonstrates complete rib artifact eradication!
    models_to_test = [
        ('sobolev_tot0', 'Sobolev (total_sigma=0.0)', 'sobolev', {'flow_sigma': 3.0, 'total_sigma': 0.0}),
        ('div_curl_tot0', 'Div-Curl (Anisotropic beta=25, total_sigma=0.0)', 'div_curl', {'beta': 25.0, 'gamma': 1.0, 'flow_sigma': 3.0, 'total_sigma': 0.0}),
        ('navier_tot0', 'Navier-Cauchy (Stokes nu=0.49, total_sigma=0.0)', 'navier', {'poisson_ratio': 0.49, 'flow_sigma': 3.0, 'total_sigma': 0.0}),
        ('sobolev_tot05', 'Sobolev with Elastic Total Smoothing (total_sigma=0.5)', 'sobolev', {'flow_sigma': 3.0, 'total_sigma': 0.5}),
    ]

    regs = {}
    for key, label, reg_name, kw in models_to_test:
        print(f"   Executing forensic test: {label}...")
        regs[key] = syntx.syn(r16, r64, reg_iterations=[100, 100, 20], regularizer=reg_name, **kw)

    fig, axes = plt.subplots(3, 4, figsize=(19, 14), facecolor='#ffffff')

    norm_jac = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.5)

    for col_idx, (key, label, reg_name, kw) in enumerate(models_to_test):
        r = regs[key]
        detJ_img = syntx.liouville_determinant(r, r16)
        j_slice, aspect_j = extract_oriented_slice(detJ_img, ref_image=r16)

        # Row 0: Full Liouville det(J) map
        im_j = axes[0, col_idx].imshow(j_slice, cmap='seismic', norm=norm_jac, aspect=aspect_j)
        folding_vox = (j_slice <= 0.0)
        if np.any(folding_vox):
            fold_ov = np.zeros((*j_slice.shape, 4), dtype=np.float32)
            fold_ov[folding_vox] = [0.0, 1.0, 0.0, 1.0]
            axes[0, col_idx].imshow(fold_ov, aspect=aspect_j)
        min_j = float(np.nanmin(j_slice))
        axes[0, col_idx].set_title(f"{label}\nMin det(J) = {min_j:+.4f}", fontsize=10, fontweight='bold', pad=8)
        axes[0, col_idx].axis('off')

        # Row 1: High-magnification Zoom of Cortical Sulcal / Ventricular Interface
        # Crop region [80:180, 80:180] in physical display frame
        zoom_crop = j_slice[80:180, 80:180]
        axes[1, col_idx].imshow(zoom_crop, cmap='seismic', norm=norm_jac, aspect=aspect_j)
        axes[1, col_idx].set_title(f"Zoomed Interface (Sulcal Wall)", fontsize=9.5, fontweight='bold', pad=6)
        axes[1, col_idx].axis('off')

        # Row 2: 2D Spatial Frequency Magnitude Spectrum log(1 + |FFT|)
        fft2 = np.fft.fftshift(np.fft.fft2(j_slice - np.nanmean(j_slice)))
        mag = np.log1p(np.abs(fft2))
        axes[2, col_idx].imshow(mag, cmap='inferno')
        axes[2, col_idx].set_title(f"2D Frequency Power Spectrum", fontsize=9.5, fontweight='bold', pad=6)
        axes[2, col_idx].axis('off')

    # Add shared colorbars
    cb_ax = fig.add_axes([0.92, 0.68, 0.015, 0.25])
    cb = fig.colorbar(im_j, cax=cb_ax)
    cb.set_label('Liouville det(J)', fontsize=9.5, fontweight='600')

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out_rib = os.path.join(ASSETS_DIR, "rib_artifact_investigation.png")
    plt.savefig(out_rib, dpi=180, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    print(f"Saved rib artifact investigation figure: {out_rib}")


if __name__ == '__main__':
    with open(JSON_PATH) as f:
        bench_data = json.load(f)
    generate_summary_metrics_quad(bench_data)
    generate_rib_artifact_investigation()
