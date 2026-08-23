"""
Generate a dedicated, complementary diagram for Slide 2:
"The Spatial Correspondence Problem on Continuous Domains & The Aperture Problem"
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

def make_slide2_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), facecolor='#FFFFFF')
    
    # Left Panel: Continuous Coordinate Mapping & Flow
    ax0 = axes[0]
    ax0.set_facecolor('#F8FAFC')
    for spine in ax0.spines.values():
        spine.set_edgecolor('#CBD5E1')
        spine.set_linewidth(1.5)
    ax0.set_xticks([])
    ax0.set_yticks([])

    y, x = np.mgrid[-0.95:0.95:100j, -0.95:0.95:100j]
    z_f = np.sin(2.5 * x) * np.cos(2.5 * y) + 0.3 * np.sin(5 * x * y)
    z_m = np.sin(2.5 * (x - 0.2)) * np.cos(2.5 * (y + 0.15)) + 0.3 * np.sin(5 * (x - 0.2) * (y + 0.15))

    ax0.contour(x, y, z_f, levels=7, colors='#2563EB', alpha=0.35, linewidths=1.2)
    ax0.contour(x, y, z_m, levels=7, colors='#9333EA', alpha=0.35, linewidths=1.2, linestyles='--')

    x_sub = np.linspace(-0.75, 0.75, 7)
    y_sub = np.linspace(-0.75, 0.75, 7)
    X_sub, Y_sub = np.meshgrid(x_sub, y_sub)
    U = 0.22 * np.cos(np.pi * Y_sub / 2)
    V = -0.18 * np.sin(np.pi * X_sub / 2)
    
    ax0.quiver(X_sub, Y_sub, U, V, color='#2563EB', angles='xy', scale_units='xy', scale=1.3, width=0.013, headwidth=4, headlength=5, label=r'Pullback $\Phi(\mathbf{x}) = \mathbf{x} + \mathbf{u}(\mathbf{x})$')
    ax0.plot([0.25, 0.25 - 0.20], [-0.15, -0.15 + 0.16], 'o-', color='#DC2626', lw=2.5, ms=7, label=r'Homologous Point $\mathbf{x} \mapsto \Phi(\mathbf{x})$')

    ax0.set_title(r"Continuous Coordinate Pullback $\Phi: \Omega \to \Omega$" + "\n" + r"$I_M(\Phi(\mathbf{x})) \approx I_F(\mathbf{x}) \quad (\mathbf{x} \in \mathbb{R}^3)$", fontsize=10.5, fontweight='bold', pad=8, color='#1E40AF')
    ax0.text(0.5, 0.04, r"Target $I_F$ (Blue solid) $\leftrightarrow$ Source $I_M$ (Purple dashed)", fontsize=9.0, fontweight='bold', color='#1E293B', ha='center', transform=ax0.transAxes)
    ax0.legend(loc='upper right', fontsize=8.0, framealpha=0.92)
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)

    # Right Panel: The Ill-Posed Aperture Problem
    ax1 = axes[1]
    ax1.set_facecolor('#F8FAFC')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#CBD5E1')
        spine.set_linewidth(1.5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    s = np.linspace(-0.96, 0.96, 200)
    edge_y = 0.32 * np.sin(2.5 * s)
    ax1.plot(s, edge_y, color='#0F172A', lw=2.8, label='Iso-Intensity Edge')
    ax1.fill_between(s, edge_y, 0.98, color='#E2E8F0', alpha=0.5)
    ax1.text(0.0, 0.70, r"Dark Tissue (CSF / Ventricles)", fontsize=8.5, color='#475569', ha='center', fontweight='bold')
    ax1.text(0.0, -0.70, r"Bright Tissue (Cortex / White Matter)", fontsize=8.5, color='#475569', ha='center', fontweight='bold')

    px, py = 0.0, 0.0
    ax1.plot(px, py, 'ro', ms=7)

    # Normal gradient vector (1 Constrained DOF)
    ax1.annotate('', xy=(px, py + 0.48), xytext=(px, py),
                 arrowprops=dict(facecolor='#059669', edgecolor='#059669', width=2.2, headwidth=6))
    ax1.text(px + 0.08, py + 0.28, r"$\nabla I$ (1 Constrained DOF)" + "\n" + r"$\mathbf{u} \cdot \nabla I = \Delta I$", fontsize=8.5, fontweight='bold', color='#047857')

    # Tangential ambiguous vectors (2 Unconstrained DOFs)
    ax1.annotate('', xy=(px + 0.50, py + 0.0), xytext=(px, py),
                 arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', width=1.8, headwidth=5, ls='--'))
    ax1.annotate('', xy=(px - 0.50, py + 0.0), xytext=(px, py),
                 arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', width=1.8, headwidth=5, ls='--'))
    ax1.text(px, py - 0.28, r"Tangential Ambiguity (2 Unconstrained DOFs)" + "\n" + r"Infinite valid solutions along $\nabla I^\perp$", fontsize=8.5, fontweight='bold', color='#991B1B', ha='center')

    ax1.set_title("The Ill-Posed Aperture Problem\n" + r"1 Intensity Equation vs 3 Unknown Displacements $\mathbf{u}(\mathbf{x})$", fontsize=10.5, fontweight='bold', pad=8, color='#0F172A')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "diag_spatial_inverse_problem.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"Saved: {p}", flush=True)

if __name__ == "__main__":
    make_slide2_diagram()
