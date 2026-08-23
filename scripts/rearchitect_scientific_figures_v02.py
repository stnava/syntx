"""
Rearchitect Slide 7, Slide 8, Slide 10, Slide 13, and Slide 15 for Maximum Didactic Impact.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

OUT_DIR = "docs/presentation/figures"

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

def format_card_axis(ax):
    ax.set_facecolor('#F8FAFC')
    for spine in ax.spines.values():
        spine.set_edgecolor('#CBD5E1')
        spine.set_linewidth(1.6)

# ----------------------------------------------------
# SLIDE 7: 3D Lie Group SO(3) & Taylor Continuity
# ----------------------------------------------------
def rearchitect_slide7():
    fig = plt.figure(figsize=(12.8, 5.4), facecolor='#FFFFFF')
    
    # Left: 3D Projection of SO(3) Sphere & Tangent Plane
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.set_facecolor('#F8FAFC')
    
    # Draw Sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax0.plot_surface(xs, ys, zs, color='#EFF6FF', alpha=0.35, edgecolor='#BFDBFE', lw=0.5)

    # Tangent Plane at Identity (0, 0, 1)
    xp, yp = np.meshgrid(np.linspace(-0.8, 0.8, 8), np.linspace(-0.8, 0.8, 8))
    zp = np.ones_like(xp) * 1.0
    ax0.plot_surface(xp, yp, zp, color='#ECFDF5', alpha=0.6, edgecolor='#10B981', lw=1.0)

    # Identity point
    ax0.scatter([0], [0], [1], color='#DC2626', s=100, label=r'Identity $I \in \mathrm{SO}(3)$')

    # Tangent Vector in Lie Algebra
    ax0.quiver(0, 0, 1, 0.6, 0.6, 0, color='#059669', lw=3.0, label=r'Lie Algebra $\boldsymbol{\omega} \in \mathfrak{so}(3)$')

    # Geodesic Curve along Sphere Surface
    t_geo = np.linspace(0, 0.8, 40)
    xg = np.sin(t_geo) * 0.707
    yg = np.sin(t_geo) * 0.707
    zg = np.cos(t_geo)
    ax0.plot(xg, yg, zg, color='#9333EA', lw=3.5, label=r'Geodesic $\exp([\boldsymbol{\omega}]_\times)$')

    ax0.set_title(r"$\mathrm{SO}(3)$ Manifold & $\mathfrak{so}(3)$ Tangent Plane" + "\n" + r"$R(\boldsymbol{\omega}) = I + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2$", fontsize=9.5, fontweight='bold', pad=8, color='#1E40AF')
    ax0.legend(loc='lower left', fontsize=7.5, framealpha=0.92)
    ax0.set_axis_off()

    # Right: Taylor Limit Derivative Transmission
    ax1 = fig.add_subplot(1, 2, 2)
    format_card_axis(ax1)

    theta = np.linspace(-0.03, 0.03, 300)
    naive_grad = np.where(np.abs(theta) < 1e-4, 0.0, np.cos(theta*100))
    taylor_grad = 1.0 - (theta*100)**2 / 6.0

    ax1.plot(theta * 100, naive_grad, color='#EF4444', lw=2.2, ls='--', label='Conditional Branch (`if theta==0` Zero-Grad Trap)')
    ax1.plot(theta * 100, taylor_grad, color='#2563EB', lw=3.0, label=r'4th-Order Taylor Limit $\lim_{\theta \to 0} R(\boldsymbol{\omega}) = I + [\boldsymbol{\omega}]_\times$')
    ax1.plot([0], [1.0], 'ro', ms=8, label='Continuous Backpropagation at Identity')

    ax1.set_title("First-Order Taylor Continuity at Origin\n" + r"Smooth Gradient Flow at Identity Initialization $\theta \to 0$", fontsize=10.5, fontweight='bold', pad=10, color='#0F172A')
    ax1.set_xlabel(r"Rotation Magnitude $\theta = \|\boldsymbol{\omega}\|_2 \times 10^{-2}$ (rad)", fontsize=9.5, fontweight='bold')
    ax1.set_ylabel("Autograd Derivative Transmission", fontsize=9.5, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=8.0, framealpha=0.92)
    ax1.set_xlim(-3, 3); ax1.set_ylim(-0.2, 1.2)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_so3_lie_algebra_v02.png")
    plt.savefig(p, dpi=300); plt.close()
    print(f"Saved: {p}", flush=True)

# ----------------------------------------------------
# SLIDE 8: 18-Cone Multi-Start Search & Landscape
# ----------------------------------------------------
def rearchitect_slide8():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), facecolor='#FFFFFF')
    ax0, ax1 = axes[0], axes[1]
    format_card_axis(ax0)
    format_card_axis(ax1)

    # Left: Non-Convex Rotational Loss Landscape with Basins
    angles = np.linspace(-60, 60, 300)
    # Landscape with false local minimum at -15 deg and true global minimum at +28 deg
    loss = -0.55 * np.exp(-((angles - 28)/12)**2) - 0.32 * np.exp(-((angles + 15)/10)**2) + 0.0003 * angles**2 + 0.85
    ax0.plot(angles, loss, color='#0F172A', lw=2.8, label=r'Cost Landscape $\mathcal{E}(\theta)$')
    
    # False Basin
    ax0.plot(-15, loss[112], 'ro', ms=9, label=r'False Local Basin (Traps $0^\circ$ Start)')
    ax0.annotate('False Local Trap\n(Stalls with >25° Misalignment)', xy=(-15, loss[112]), xytext=(-56, 0.95),
                 arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#DC2626')

    # True Global Basin
    ax0.plot(28, loss[220], 'go', ms=10, label=r'Global True Minimum ($+28^\circ$)')
    ax0.annotate('True Anatomical Basin\n(Captured by 18-Cone Probes)', xy=(28, loss[220]), xytext=(10, 0.10),
                 arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#047857')

    ax0.set_title("Non-Convex Rotational Energy Landscape\n" + r"Why Gradient Descent from Identity Fails Without Multi-Start", fontsize=10.5, fontweight='bold', pad=10, color='#0F172A')
    ax0.set_xlabel(r"Euler Rotation Angle $\theta_x$ (degrees)", fontsize=9.0, fontweight='bold')
    ax0.set_ylabel(r"Registration Cost $\mathcal{L}_{\mathrm{MI}}$", fontsize=9.0, fontweight='bold')
    ax0.legend(loc='upper right', fontsize=7.5, framealpha=0.92)
    ax0.set_xlim(-60, 60); ax0.set_ylim(-0.05, 1.4)

    # Right: 18-Cone Spatial Probing Lattice
    ax1.set_xticks([]); ax1.set_yticks([])
    t_cones = np.linspace(0, 2*np.pi, 18, endpoint=False)
    cx, cy = np.cos(t_cones) * 15, np.sin(t_cones) * 15
    
    # Draw brain silhouette in center
    t_b = np.linspace(0, 2*np.pi, 100)
    bx = (5.5 + 1.2*np.sin(2*t_b)) * np.cos(t_b)
    by = (7.0 + 1.0*np.cos(2*t_b)) * np.sin(t_b)
    ax1.fill(bx, by, color='#E2E8F0', alpha=0.7)
    ax1.plot(bx, by, color='#475569', lw=2.0)

    ax1.plot(cx, cy, 'o', color='#2563EB', ms=9, label=r'18 Geodesic Cones ($\pm 15^\circ$)')
    ax1.plot(0, 0, 'ks', ms=9, label=r'Center of Mass $T_0$')
    for i in range(18):
        ax1.plot([0, cx[i]], [0, cy[i]], color='#94A3B8', ls='--', lw=1.2)
    
    circle15 = patches.Circle((0, 0), 15, edgecolor='#2563EB', facecolor='#EFF6FF', alpha=0.25, lw=1.8, ls=':')
    ax1.add_patch(circle15)

    ax1.text(0.0, -21, "100% (16/16) Global Basin Recovery Rate\nForeground-Masked Mutual Information Scoring",
             fontsize=8.5, fontweight='bold', color='#047857', ha='center',
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.2))

    ax1.set_title("Deterministic 18-Cone Search Lattice\n" + r"Parallel Geodesic Shooting in $\mathfrak{so}(3)$ Space", fontsize=10.5, fontweight='bold', pad=10, color='#1E40AF')
    ax1.set_xlim(-25, 25); ax1.set_ylim(-26, 25)
    ax1.legend(loc='upper center', fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_18cone_multistart_v02.png")
    plt.savefig(p, dpi=300); plt.close()
    print(f"Saved: {p}", flush=True)

# ----------------------------------------------------
# SLIDE 13: Antithetic Bootstrapping on Real Boundary
# ----------------------------------------------------
def rearchitect_slide13():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), facecolor='#FFFFFF')
    ax0, ax1 = axes[0], axes[1]
    format_card_axis(ax0)
    format_card_axis(ax1)
    ax0.set_xticks([]); ax0.set_yticks([])

    # Left: Sharp Cortical Boundary Edge with Sub-Voxel Triplet
    y_b, x_b = np.mgrid[-1:1:100j, -1:1:100j]
    edge = np.tanh((x_b + 0.5*y_b) * 12.0)
    ax0.imshow(edge, cmap='bone', extent=[-1, 1, -1, 1], origin='lower')

    # Discrete Grid lines
    for g in np.linspace(-1, 1, 9):
        ax0.axvline(g, color='#94A3B8', alpha=0.35, ls=':')
        ax0.axhline(g, color='#94A3B8', alpha=0.35, ls=':')

    # Triplet sample
    p_x, p_y = 0.0, 0.0
    dx, dy = 0.28, 0.22
    ax0.plot([p_x], [p_y], 'yo', ms=10, label=r'Native Grid Sample $\mathbf{X}$')
    ax0.plot([p_x + dx], [p_y + dy], 'ro', ms=8, label=r'Positive Jitter $\mathbf{X} + \boldsymbol{\delta}$')
    ax0.plot([p_x - dx], [p_y - dy], 'co', ms=8, label=r'Antithetic Jitter $\mathbf{X} - \boldsymbol{\delta}$')
    ax0.plot([p_x - dx, p_x + dx], [p_y - dy, p_y + dy], color='#F59E0B', lw=2.2, ls='--')

    ax0.text(0.0, -0.75, r"$\mathbb{E}[\boldsymbol{\delta} + (-\boldsymbol{\delta})] \equiv \mathbf{0}$ (Zero Bias)" + "\n" + "Destructively Cancels Sub-Voxel Discretization Noise",
             fontsize=8.5, fontweight='bold', color='#047857', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.2))

    ax0.set_title("Antithetic Coordinate Triplet Sampling\n" + r"$\boldsymbol{\delta} \sim \mathcal{U}(-0.25, 0.25) \odot \mathbf{s}_{\mathrm{phys}}$", fontsize=10.5, fontweight='bold', pad=10, color='#1E40AF')
    ax0.legend(loc='upper left', fontsize=7.5, framealpha=0.92)
    ax0.set_xlim(-1, 1); ax0.set_ylim(-1, 1)

    # Right: Thin-Plate Bending Energy Reduction
    methods = ['ANTs C++ SyN', 'Standard Autograd', 'Antithetic Bootstrapping']
    bnd_vals = [0.0169, 0.0125, 0.0067]
    colors = ['#94A3B8', '#EF4444', '#059669']
    bars = ax1.bar(methods, bnd_vals, color=colors, width=0.50, edgecolor='#0F172A', lw=1.4)
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0006, f"{b.get_height():.4f}", ha='center', fontsize=9.0, fontweight='bold')
    
    ax1.set_title("Thin-Plate Bending Energy Reduction\n" + r"$\mathrm{Bnd}(v) = \frac{1}{|\Omega|}\int (\|\nabla^2 v_x\|^2 + \|\nabla^2 v_y\|^2) dx dy$", fontsize=10.0, fontweight='bold', pad=10, color='#0F172A')
    ax1.set_ylabel("Thin-Plate Bending Energy", fontsize=9.0, fontweight='bold')
    ax1.set_ylim(0, 0.022)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_antithetic_bootstrapping_v02.png")
    plt.savefig(p, dpi=300); plt.close()
    print(f"Saved: {p}", flush=True)

if __name__ == "__main__":
    rearchitect_slide7()
    rearchitect_slide8()
    rearchitect_slide13()
    print("REARCHITECTED FIGURES 7, 8, AND 13 SUCCESSFULLY!")
