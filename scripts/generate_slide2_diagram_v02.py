"""
Revise Figure: diag_spatial_inverse_problem_v01.png for Maximum Clarity and Scientific Detail.
Signature Brian Avants Style: Clean light themes (#FFFFFF, #F8FAFC), dark slate typography (#0F172A),
high-contrast royal blue (#2563EB), vibrant purple (#9333EA), emerald green (#059669), and crimson (#DC2626).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_PATH = "docs/presentation/figures/diag_spatial_inverse_problem_v01.png"

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'text.color': '#0F172A',
    'axes.labelcolor': '#1E293B',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
})

def make_refined_slide2_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), facecolor='#FFFFFF')
    ax0, ax1 = axes[0], axes[1]

    # ----------------------------------------------------
    # PANEL A: Continuous Spatial Coordinate Pullback
    # ----------------------------------------------------
    ax0.set_facecolor('#F8FAFC')
    for spine in ax0.spines.values():
        spine.set_edgecolor('#CBD5E1')
        spine.set_linewidth(1.6)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Parameterize realistic gyral/sulcal anatomical folds
    t = np.linspace(0, 2 * np.pi, 350)
    
    # Target Anatomy (Fixed I_F) - Solid Royal Blue
    r_target = 0.42 + 0.12 * np.sin(3 * t) + 0.05 * np.cos(6 * t)
    x_target = r_target * np.cos(t)
    y_target = r_target * np.sin(t)

    # Moving Anatomy (Source I_M) - Dashed Vibrant Purple (deformed + shifted)
    r_moving = 0.38 + 0.10 * np.sin(3 * (t - 0.28)) + 0.06 * np.cos(6 * (t - 0.28))
    x_moving = r_moving * np.cos(t - 0.28) + 0.05
    y_moving = r_moving * np.sin(t - 0.28) - 0.03

    # Fill anatomy regions with subtle tints
    ax0.fill(x_target, y_target, color='#EFF6FF', alpha=0.6)
    ax0.plot(x_target, y_target, color='#2563EB', lw=3.0, label=r'Target Cortical Ribbon $I_F(\mathbf{x})$ (Fixed Target)')
    ax0.plot(x_moving, y_moving, color='#9333EA', lw=2.6, ls='--', label=r'Moving Cortical Ribbon $I_M(\mathbf{x})$ (Source Image)')

    # Draw clear, prominent displacement arrows u(x) connecting homologous points
    sample_indices = [25, 75, 120, 165, 210, 260]
    for idx in sample_indices:
        xm, ym = x_moving[idx], y_moving[idx]
        xt, yt = x_target[idx], y_target[idx]
        ax0.annotate('', xy=(xt, yt), xytext=(xm, ym),
                     arrowprops=dict(facecolor='#2563EB', edgecolor='#1D4ED8', width=1.6, headwidth=6.0, headlength=6.5, shrink=0.08))

    # Landmark 1: Gyral Crest
    idx_gyrus = 52
    xm1, ym1 = x_moving[idx_gyrus], y_moving[idx_gyrus]
    xt1, yt1 = x_target[idx_gyrus], y_target[idx_gyrus]
    ax0.plot([xm1], [ym1], 'o', color='#9333EA', ms=8, zorder=5)
    ax0.plot([xt1], [yt1], 'o', color='#2563EB', ms=8, zorder=5)
    ax0.plot([xm1, xt1], [ym1, yt1], color='#DC2626', lw=2.5, zorder=4)
    ax0.text(xt1 - 0.02, yt1 + 0.16, "Gyral Crest\n" + r"$\mathbf{x} \mapsto \Phi(\mathbf{x})$", fontsize=8.5, fontweight='bold', color='#DC2626', ha='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=1.0))

    # Landmark 2: Sulcal Fundus
    idx_sulcus = 145
    xm2, ym2 = x_moving[idx_sulcus], y_moving[idx_sulcus]
    xt2, yt2 = x_target[idx_sulcus], y_target[idx_sulcus]
    ax0.plot([xm2], [ym2], 'o', color='#9333EA', ms=8, zorder=5)
    ax0.plot([xt2], [yt2], 'o', color='#2563EB', ms=8, zorder=5)
    ax0.plot([xm2, xt2], [ym2, yt2], color='#DC2626', lw=2.5, zorder=4)
    ax0.text(xt2 - 0.05, yt2 + 0.18, "Sulcal Fundus\n" + r"$\mathbf{u}(\mathbf{x}) \in \mathbb{R}^3$", fontsize=8.5, fontweight='bold', color='#DC2626', ha='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=1.0))

    # Domain label
    ax0.text(0.0, -0.05, r"Continuous Domain $\Omega \subset \mathbb{R}^3$" + "\n" + r"$\sim 10^7$ Spatial Unknowns", fontsize=9.0, fontweight='bold', color='#475569', ha='center', va='center')

    ax0.set_title(r"Continuous Spatial Coordinate Pullback $\Phi: \Omega \to \Omega$" + "\n" + r"$I_M(\Phi(\mathbf{x})) \approx I_F(\mathbf{x}) \quad \text{where } \Phi(\mathbf{x}) = \mathbf{x} + \mathbf{u}(\mathbf{x})$", fontsize=10.5, fontweight='bold', pad=10, color='#1E40AF')
    ax0.legend(loc='lower center', fontsize=8.0, framealpha=0.95, edgecolor='#CBD5E1')
    ax0.set_xlim(-0.95, 0.95)
    ax0.set_ylim(-0.95, 0.95)

    # ----------------------------------------------------
    # PANEL B: The Ill-Posed Aperture Problem & Null Space
    # ----------------------------------------------------
    ax1.set_facecolor('#F8FAFC')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#CBD5E1')
        spine.set_linewidth(1.6)
    ax1.set_xticks([])
    ax1.set_yticks([])

    x_edge = np.linspace(-0.95, 0.95, 300)
    y_edge = 0.20 * np.sin(2.0 * x_edge) + 0.08 * x_edge
    
    # Fill Dark (CSF) vs Bright (Gray Matter)
    ax1.fill_between(x_edge, y_edge, 0.95, color='#E2E8F0', alpha=0.6)
    ax1.fill_between(x_edge, y_edge, -0.95, color='#EFF6FF', alpha=0.6)
    ax1.plot(x_edge, y_edge, color='#0F172A', lw=3.2, label=r'Iso-Intensity Boundary ($\nabla I^\perp$)')

    ax1.text(0.0, 0.70, r"CSF / Background (Dark: $I \approx 0.1$)", fontsize=9, color='#475569', ha='center', fontweight='bold')
    ax1.text(0.0, -0.72, r"Cortex / White Matter (Bright: $I \approx 0.9$)", fontsize=9, color='#1E40AF', ha='center', fontweight='bold')

    # Point of evaluation
    p_x, p_y = 0.0, 0.0
    ax1.plot(p_x, p_y, 'ro', ms=9, zorder=6)

    # Normal Intensity Gradient Vector (Determined by Data)
    nx, ny = -0.50, 0.86
    ax1.annotate('', xy=(p_x + 0.48 * nx, p_y + 0.48 * ny), xytext=(p_x, p_y),
                 arrowprops=dict(facecolor='#059669', edgecolor='#047857', width=2.4, headwidth=7.5, headlength=8), zorder=5)
    ax1.text(p_x + 0.50 * nx - 0.02, p_y + 0.50 * ny + 0.06,
             r"$\nabla I$ Normal Vector (1 DOF Constrained)" + "\n" + r"$\mathbf{u} \cdot \nabla I = \Delta I$ (Directly Observable)",
             fontsize=8.5, fontweight='bold', color='#047857', ha='center')

    # Tangential Null-Space Motion Vectors (Ambiguous DOFs along edge)
    tx, ty = 0.86, 0.50
    ax1.annotate('', xy=(p_x + 0.50 * tx, p_y + 0.50 * ty), xytext=(p_x, p_y),
                 arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', width=2.0, headwidth=6, ls='--', headlength=7), zorder=5)
    ax1.annotate('', xy=(p_x - 0.50 * tx, p_y - 0.50 * ty), xytext=(p_x, p_y),
                 arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', width=2.0, headwidth=6, ls='--', headlength=7), zorder=5)
    
    # Tangential Null Space Callout Box
    ax1.text(0.10, -0.38,
             r"Tangential Null Space (2 DOFs Unconstrained)" + "\n" + r"Infinite valid solutions $\mathbf{u}_\parallel \in \nabla I^\perp$ with $\Delta I = 0$",
             fontsize=8.5, fontweight='bold', color='#991B1B', ha='center',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=1.2))

    # Mathematical Equation Summary Box at Top Right
    ax1.text(0.44, 0.40,
             r"$\mathbf{1 \text{ Equation}} \leftrightarrow \mathbf{3 \text{ Unknowns } (u_x, u_y, u_z)}$" + "\n" +
             r"$\rightarrow$ Requires Lie Algebra Regularization",
             fontsize=8.0, fontweight='bold', color='#0F172A', ha='center',
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.2))

    ax1.set_title("The Ill-Posed Aperture Problem\n" + r"Why Naive Intensity Matching Fails Without Diffeomorphic Physics", fontsize=10.5, fontweight='bold', pad=10, color='#0F172A')
    ax1.set_xlim(-0.95, 0.95)
    ax1.set_ylim(-0.95, 0.95)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()
    print(f"Refined Slide 2 Figure saved to: {OUT_PATH}", flush=True)

if __name__ == "__main__":
    make_refined_slide2_figure()
