import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_figure9():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=300)
    plt.subplots_adjust(left=0.06, right=0.92, wspace=0.22, top=0.88, bottom=0.10)

    # High-resolution mesh for Jacobian field
    n_fine = 250
    x_fine = np.linspace(-2.2, 2.2, n_fine)
    y_fine = np.linspace(-2.2, 2.2, n_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Coarse mesh for grid lines
    n_lines = 23
    x_grid = np.linspace(-2.0, 2.0, n_lines)
    y_grid = np.linspace(-2.0, 2.0, n_lines)

    # Dense resolution along grid lines for smooth curve plotting
    n_line_pts = 200
    line_t = np.linspace(-2.0, 2.0, n_line_pts)

    # -------------------------------------------------------------
    # Panel A: Diffeomorphic Transformation (Smooth Gaussian expansion/warp)
    # -------------------------------------------------------------
    def warp_A(x, y):
        # Smooth localized expansion
        r2 = x**2 + y**2
        sigma = 1.1
        amp = 0.45
        factor = amp * np.exp(-r2 / (2 * sigma**2))
        ux = factor * x
        uy = factor * y
        return x + ux, y + uy

    def jacobian_A(x, y):
        r2 = x**2 + y**2
        sigma = 1.1
        amp = 0.45
        g = amp * np.exp(-r2 / (2 * sigma**2))
        
        dg_dx = g * (-x / sigma**2)
        dg_dy = g * (-y / sigma**2)
        
        dux_dx = g + x * dg_dx
        dux_dy = x * dg_dy
        duy_dx = y * dg_dx
        duy_dy = g + y * dg_dy

        d11 = 1.0 + dux_dx
        d12 = dux_dy
        d21 = duy_dx
        d22 = 1.0 + duy_dy

        J = d11 * d22 - d12 * d21
        return J

    J_A = jacobian_A(X_fine, Y_fine)

    # -------------------------------------------------------------
    # Panel B: Non-Diffeomorphic Transformation (Severe compression leading to grid folding)
    # -------------------------------------------------------------
    def warp_B(x, y):
        r2 = x**2 + y**2
        sigma = 0.95
        amp = -1.65 # Severe negative amplitude causes inversion
        factor = amp * np.exp(-r2 / (2 * sigma**2))
        ux = factor * x
        uy = factor * y
        return x + ux, y + uy

    def jacobian_B(x, y):
        r2 = x**2 + y**2
        sigma = 0.95
        amp = -1.65
        g = amp * np.exp(-r2 / (2 * sigma**2))
        
        dg_dx = g * (-x / sigma**2)
        dg_dy = g * (-y / sigma**2)
        
        dux_dx = g + x * dg_dx
        dux_dy = x * dg_dy
        duy_dx = y * dg_dx
        duy_dy = g + y * dg_dy

        d11 = 1.0 + dux_dx
        d12 = dux_dy
        d21 = duy_dx
        d22 = 1.0 + duy_dy

        J = d11 * d22 - d12 * d21
        return J

    J_B = jacobian_B(X_fine, Y_fine)

    # Setup Custom Diverging Colormap centered at J=1.0, with red for J <= 0
    vmin, vmax = -0.6, 2.2
    cmap = plt.cm.coolwarm

    # Plot Panel A
    ax_a = axes[0]
    im_a = ax_a.contourf(X_fine, Y_fine, J_A, levels=np.linspace(vmin, vmax, 100), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    
    # Draw deformed grid lines for Panel A
    for x_c in x_grid:
        pts_x, pts_y = warp_A(x_c, line_t)
        ax_a.plot(pts_x, pts_y, color='black', alpha=0.6, linewidth=0.9)
    for y_c in y_grid:
        pts_x, pts_y = warp_A(line_t, y_c)
        ax_a.plot(pts_x, pts_y, color='black', alpha=0.6, linewidth=0.9)

    ax_a.set_title("(a) Diffeomorphic Mapping (Topology Preserving)\n$J(\\mathbf{x}) > 0$ Everywhere", fontsize=11, fontweight='bold', pad=10, color='#111111')
    ax_a.set_xlim(-2.1, 2.1)
    ax_a.set_ylim(-2.1, 2.1)
    ax_a.set_aspect('equal')
    ax_a.set_xlabel("Spatial Coordinate $x_1$", fontsize=10)
    ax_a.set_ylabel("Spatial Coordinate $x_2$", fontsize=10)
    ax_a.grid(False)

    # Annotation box for Panel A
    text_a = (
        "Properties:\n"
        "• Smooth, one-to-one mapping $\\phi(\\mathbf{x})$\n"
        "• Inverse $\\phi^{-1}$ exists & smooth everywhere\n"
        "• Jacobian $J(\\mathbf{x}) = \\det(D\\phi) > 0$\n"
        "• Grid Folding Rate = 0.0000%"
    )
    ax_a.text(0.03, 0.03, text_a, transform=ax_a.transAxes, fontsize=8.5,
              verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#2b5c8f', lw=1.5))

    # Plot Panel B
    ax_b = axes[1]
    im_b = ax_b.contourf(X_fine, Y_fine, J_B, levels=np.linspace(vmin, vmax, 100), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    
    # Contour line for J(x) = 0 singularity boundary
    ax_b.contour(X_fine, Y_fine, J_B, levels=[0.0], colors=['#cc0000'], linewidths=[2.2], linestyles=['--'])

    # Draw deformed grid lines for Panel B
    for x_c in x_grid:
        pts_x, pts_y = warp_B(x_c, line_t)
        ax_b.plot(pts_x, pts_y, color='black', alpha=0.65, linewidth=0.9)
    for y_c in y_grid:
        pts_x, pts_y = warp_B(line_t, y_c)
        ax_b.plot(pts_x, pts_y, color='black', alpha=0.65, linewidth=0.9)

    ax_b.set_title("(b) Non-Diffeomorphic Mapping (Grid Folding Singularity)\nLocal Jacobian $J(\\mathbf{x}) \\leq 0$ (Tangled Grid)", fontsize=11, fontweight='bold', pad=10, color='#880000')
    ax_b.set_xlim(-2.1, 2.1)
    ax_b.set_ylim(-2.1, 2.1)
    ax_b.set_aspect('equal')
    ax_b.set_xlabel("Spatial Coordinate $x_1$", fontsize=10)
    ax_b.set_ylabel("Spatial Coordinate $x_2$", fontsize=10)
    ax_b.grid(False)

    # Annotation box for Panel B
    text_b = (
        "Violations:\n"
        "• Tangled & self-intersecting grid lines\n"
        "• Negative Jacobian $J(\\mathbf{x}) \\leq 0$ region (Red)\n"
        "• Non-invertible spatial singularity\n"
        "• Topological breakdown (cell overlap)"
    )
    ax_b.text(0.03, 0.03, text_b, transform=ax_b.transAxes, fontsize=8.5,
              verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#b30000', lw=1.5))

    # Add arrow annotation pointing to J <= 0 region
    ax_b.annotate('Singularity Zone\n$J(\\mathbf{x}) \\leq 0$', xy=(0, 0), xytext=(0.70, 0.70),
                arrowprops=dict(facecolor='#cc0000', edgecolor='#880000', shrink=0.08, width=1.5, headwidth=7),
                fontsize=9, fontweight='bold', color='#aa0000',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeeee', edgecolor='#cc0000', lw=1))

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(im_b, cax=cbar_ax)
    cbar.set_label('Jacobian Determinant $J(\\mathbf{x}) = \\det(D\\phi)$', fontsize=10, labelpad=8)
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    cbar.ax.axhline(0.0, color='red', linewidth=2, linestyle='--')

    # Overall Figure Title
    fig.suptitle("Diffeomorphic Invertibility vs. Non-Diffeomorphic Grid Folding in Image Registration",
                 fontsize=13, fontweight='bold', y=0.98)

    # Ensure output directory exists
    out_dir = "/Users/stnava/code/syntx/docs/manuscript/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig9_diffeomorphic_invertibility_concept.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated Figure 9 at: {out_path}")

if __name__ == "__main__":
    generate_figure9()
