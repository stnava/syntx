"""
Generate Focused Educational Visualizations for the 20-Slide PhD Masterclass Presentation.
Signature Brian Avants Style: Clean light themes (#FFFFFF), crisp vectors, dark-slate typography (#0F172A),
royal blue (#2563EB), emerald (#10B981), and red (#EF4444) accents.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "docs/presentation/figures"
os.makedirs(OUT_DIR, exist_ok=True)

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

def make_slide3_topology_folding():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='#FFFFFF')
    
    # Diffeomorphic (det(J) > 0)
    ax0 = axes[0]
    ax0.set_facecolor('#F8FAFC')
    y, x = np.mgrid[0:1:11j, 0:1:11j]
    dx = 0.08 * np.sin(2 * np.pi * y) * np.cos(np.pi * x)
    dy = 0.08 * np.cos(2 * np.pi * x) * np.sin(np.pi * y)
    gx = x + dx
    gy = y + dy
    for i in range(11):
        ax0.plot(gx[i, :], gy[i, :], color='#2563EB', lw=1.8)
        ax0.plot(gx[:, i], gy[:, i], color='#2563EB', lw=1.8)
    ax0.set_title(r"Smooth Diffeomorphism: $\det(J) > 0$" + "\n(Orientation Preserved, Bijective)", fontsize=11, fontweight='bold', pad=8, color='#047857')
    ax0.text(0.5, 0.02, "Preserved Topology & Exact Invertibility", fontsize=9.5, fontweight='bold', color='#047857', ha='center', transform=ax0.transAxes)
    ax0.axis('off')

    # Folded / Torn (det(J) <= 0)
    ax1 = axes[1]
    ax1.set_facecolor('#FEF2F2')
    gx_fold = gx.copy()
    gy_fold = gy.copy()
    gx_fold[4:7, 4:7] += 0.28 * np.array([[-1, 1, -1], [1, -1.5, 1], [-1, 1, -1]])
    gy_fold[4:7, 4:7] += 0.28 * np.array([[1, -1, 1], [-1.5, 1, -1.5], [1, -1, 1]])
    for i in range(11):
        ax1.plot(gx_fold[i, :], gy_fold[i, :], color='#EF4444', lw=1.8)
        ax1.plot(gx_fold[:, i], gy_fold[:, i], color='#EF4444', lw=1.8)
    circle = patches.Circle((0.5, 0.5), 0.22, edgecolor='#DC2626', facecolor='#FEE2E2', alpha=0.6, lw=2, ls='--')
    ax1.add_patch(circle)
    ax1.text(0.5, 0.5, "det(J) <= 0\nGrid Self-Intersection\n(Coordinate Tearing)", fontsize=9.5, fontweight='bold', color='#991B1B', ha='center', va='center')
    ax1.set_title(r"Classical Collapse: $\det(J) \leq 0$" + "\n(Non-Invertible Singularity)", fontsize=11, fontweight='bold', pad=8, color='#B91C1C')
    ax1.text(0.5, 0.02, "Non-Physical Singularity & Irreversible Loss", fontsize=9.5, fontweight='bold', color='#B91C1C', ha='center', transform=ax1.transAxes)
    ax1.axis('off')

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_topology_preservation.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}", flush=True)

def make_slide6_variance_singularity():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    v = np.linspace(0, 1e-4, 300)
    g_unfloored = 1.0 / np.maximum(v, 1e-9)
    g_floored = 1.0 / np.maximum(v, 1e-6)

    ax.semilogy(v * 1e4, g_unfloored, color='#EF4444', lw=2.5, ls='--', label=r"Unfloored Autograd $\frac{1}{\mathrm{Var}}$ ($\mathcal{O}(\mathrm{Var}^{-1/2}) \to \infty$)")
    ax.semilogy(v * 1e4, g_floored, color='#2563EB', lw=2.8, label=r"Safe Floor $\mathrm{Var}_{\mathrm{safe}} = \max(\mathrm{Var}(I), 10^{-6})$")
    
    ax.axvline(0.01, color='#10B981', ls=':', lw=2, label="Homogeneous Tissue Boundary (White Matter / Ventricles)")
    ax.annotate("Derivative Explosion\n(Drives Local Grid Folding)", xy=(0.002, 1e8), xytext=(0.02, 3e8),
                arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', color='#DC2626')
    ax.annotate("Bounded Descent Dynamics\n(Strictly Diffeomorphic)", xy=(0.002, 1e6), xytext=(0.03, 2e5),
                arrowprops=dict(facecolor='#2563EB', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', color='#2563EB')

    ax.set_title("LNCC Analytical Gradient Gain vs Local Image Variance", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(r"Local Patch Variance $\mathrm{Var}(I) \times 10^{-4}$", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"Gradient Gain $\|\partial \mathcal{L}_{\mathrm{LNCC}} / \partial I\|$", fontsize=10, fontweight='bold')
    ax.grid(True, ls=':', alpha=0.6, color='#CBD5E1')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
    ax.set_xlim(-0.002, 1.0)
    ax.set_ylim(1e4, 5e9)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_variance_floor_proof.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}", flush=True)

def make_slide16_sobolev_adam_concept():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor='#FFFFFF')
    
    # Pointwise Adam
    ax0 = axes[0]
    ax0.set_facecolor('#FEF2F2')
    np.random.seed(42)
    x = np.linspace(-1, 1, 14)
    y = np.linspace(-1, 1, 14)
    X, Y = np.meshgrid(x, y)
    U_adam = np.random.randn(14, 14) * 0.35
    V_adam = np.random.randn(14, 14) * 0.35
    ax0.quiver(X, Y, U_adam, V_adam, color='#DC2626', scale=7, width=0.012)
    ax0.set_title(r"Standard Pointwise Adam on $\mathfrak{X}(\Omega)$" + "\n" + r"(Pointwise $m_t / \sqrt{v_t}$ Amplifies Noise $\to \mathcal{O}(1)$)", fontsize=10.0, fontweight='bold', pad=8, color='#B91C1C')
    ax0.set_ylim(-1.3, 1.15)
    ax0.text(0.5, 0.04, "Metric Collapse: High-Frequency Shears", fontsize=9, fontweight='bold', color='#B91C1C', ha='center', transform=ax0.transAxes)
    ax0.axis('off')

    # SobolevAdam
    ax1 = axes[1]
    ax1.set_facecolor('#F0FDF4')
    U_sob = np.sin(np.pi * Y) * 0.45
    V_sob = np.cos(np.pi * X) * 0.45
    ax1.quiver(X, Y, U_sob, V_sob, color='#047857', scale=5.5, width=0.014)
    ax1.set_title(r"Riemannian SobolevAdam Preconditioning" + "\n" + r"$(\mathcal{G}_{\mathrm{Sobolev}} = (I - \alpha \Delta)^{-s} \in H^s(\Omega))$", fontsize=10.0, fontweight='bold', pad=8, color='#047857')
    ax1.set_ylim(-1.3, 1.15)
    ax1.text(0.5, 0.04, "Strict Sobolev Regularity: Smooth Flow", fontsize=9, fontweight='bold', color='#047857', ha='center', transform=ax1.transAxes)
    ax1.axis('off')

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_sobolev_adam_comparison.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}", flush=True)

if __name__ == "__main__":
    make_slide3_topology_folding()
    make_slide6_variance_singularity()
    make_slide16_sobolev_adam_concept()
    print("ALL PRESENTATION DIAGRAMS GENERATED SUCCESSFULLY!")
