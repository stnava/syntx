"""
Interactive Visualization Gallery Generator for Syntx.

Builds self-contained HTML gallery showcasing all standard figure generators and statistical displays:
- Figure 1: Input Fixed & Moving Pair (2D/3D)
- Figure 2: Standard 4-Panel Registration Quality Report
- Mindboggle DKT Anatomical Label Alignment
- Statistical Distributions (Dice overlap distributions & det(J) histograms)
"""

import os
import base64
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from .figures import (
    render_input_pair_figure,
    render_standard_4panel,
    render_label_alignment_figure,
    plot_deformation_grid,
    plot_edge_overlay
)
from .stats import plot_label_overlap_stats, plot_jacobian_distribution
from .reports import build_engine_provenance


def fig_to_base64_png(fig) -> str:
    """Converts matplotlib Figure to base64 data URI."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def create_visualization_gallery(
    fixed,
    moving,
    warped=None,
    warp=None,
    detJ=None,
    inv_err_map=None,
    fixed_labels=None,
    warped_labels=None,
    dice_scores=None,
    output_path: str = "syntx_visualization_gallery.html",
    provenance: Optional[Dict[str, Any]] = None,
    title: str = "Syntx Medical Image Registration Visualization Gallery"
) -> str:
    """
    Creates publication-grade self-contained interactive HTML visualization gallery.
    
    Args:
        fixed: Fixed target image.
        moving: Moving source image.
        warped: Warped moving image (optional).
        warp: Deformation field (optional).
        detJ: Jacobian determinant map (optional).
        inv_err_map: Inverse identity error map (optional).
        fixed_labels: Fixed target segmentation labels (optional).
        warped_labels: Warped moving segmentation labels (optional).
        dice_scores: Dice evaluation dictionary (optional).
        output_path: Output HTML filepath.
        provenance: Optional provenance dictionary.
        title: Gallery title.
        
    Returns:
        str: Absolute path to generated HTML gallery.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if provenance is None:
        provenance = build_engine_provenance()

    # Imports for figures
    from .figures import (
        plot_correspondence_vectors,
        plot_vector_field,
        plot_deformation_tensor_rgb,
    )

    # Generate Figure 1 (Input Pair) in Dark & Light themes
    fig1_dark = render_input_pair_figure(fixed, moving, theme="dark", title="Figure 1: Pre-Registration Input Images (Dark Theme)")
    uri_fig1_dark = fig_to_base64_png(fig1_dark)

    fig1_light = render_input_pair_figure(fixed, moving, theme="light", title="Figure 1: Pre-Registration Input Images (Light Theme)")
    uri_fig1_light = fig_to_base64_png(fig1_light)

    # Generate Figure 2 (Standard 4-Panel) if warped/warp/detJ available
    uri_fig2 = None
    if warped is not None and warp is not None and detJ is not None and inv_err_map is not None:
        fig2 = render_standard_4panel(fixed, warped, warp, detJ, inv_err_map, title_prefix="Syntx Registration Quality")
        uri_fig2 = fig_to_base64_png(fig2)

    # Generate Vector & Tensor Displays
    uri_corr_vec = None
    uri_vector_field = None
    uri_tensor_rgb = None
    if warp is not None:
        fig_corr = plot_correspondence_vectors(warp, fixed=fixed, theme="dark")
        uri_corr_vec = fig_to_base64_png(fig_corr)

        fig_vec = plot_vector_field(warp, fixed=fixed, theme="dark")
        uri_vector_field = fig_to_base64_png(fig_vec)

        try:
            fig_dt = plot_deformation_tensor_rgb(warp, fixed=fixed, theme="dark")
            uri_tensor_rgb = fig_to_base64_png(fig_dt)
        except Exception:
            uri_tensor_rgb = None

    # Generate Anatomical Label Alignment if segmentations provided
    uri_label_dark = None
    uri_label_light = None
    if fixed_labels is not None and warped_labels is not None:
        fig_l_dark = render_label_alignment_figure(fixed_labels, warped_labels, fixed_image=fixed, theme="dark")
        uri_label_dark = fig_to_base64_png(fig_l_dark)

        fig_l_light = render_label_alignment_figure(fixed_labels, warped_labels, fixed_image=fixed, theme="light")
        uri_label_light = fig_to_base64_png(fig_l_light)

    # Generate Statistical Distribution Displays
    uri_stats_dice = None
    if dice_scores is not None:
        fig_d = plot_label_overlap_stats(dice_scores, theme="dark")
        uri_stats_dice = fig_to_base64_png(fig_d)

    uri_stats_jac = None
    if detJ is not None:
        fig_j = plot_jacobian_distribution(detJ, theme="dark")
        uri_stats_jac = fig_to_base64_png(fig_j)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --card-bg: #161b22;
            --border: #30363d;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-green: #3fb950;
            --accent-orange: #fb923c;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 32px 0 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 32px;
        }}
        .header h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0 0 10px;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}
        .header p {{
            color: var(--text-secondary);
            font-size: 1.05rem;
            margin: 0;
        }}
        .provenance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 36px;
        }}
        .prov-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .prov-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .prov-val {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--accent);
        }}
        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 36px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .section-card h2 {{
            font-size: 1.4rem;
            margin-top: 0;
            margin-bottom: 16px;
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }}
        .fig-img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
            display: block;
            margin-top: 12px;
        }}
        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .grid-2col {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Syntx Medical Image Registration Gallery</h1>
            <p>Standardized 2D/3D Visualization Suite, Anatomical Labeling & Statistical Quality Benchmark</p>
        </div>

        <div class="provenance-grid">
            <div class="prov-card"><div class="prov-label">Algorithm</div><div class="prov-val">{provenance.get("algorithm", "syntx")}</div></div>
            <div class="prov-card"><div class="prov-label">Backend Engine</div><div class="prov-val">{provenance.get("backend", "pytorch").upper()}</div></div>
            <div class="prov-card"><div class="prov-label">Compute Device</div><div class="prov-val">{provenance.get("device", "cpu").upper()}</div></div>
            <div class="prov-card"><div class="prov-label">Syntx Version</div><div class="prov-val">v{provenance.get("syntx_version", "1.1.8")}</div></div>
        </div>

        <div class="section-card">
            <h2>Figure 1: Pre-Registration Input Images (Standard Layout & Themes)</h2>
            <p style="color: var(--text-secondary);">Axial (Anterior UP), Coronal (Superior UP), Sagittal (Superior UP) orthographic views in canonical LPI space with physical anisotropy scaling and exactly 1 colorbar per image.</p>
            <div class="grid-2col">
                <div>
                    <h3 style="color: var(--accent);">Dark Theme</h3>
                    <img class="fig-img" src="{uri_fig1_dark}" alt="Figure 1 Dark Theme">
                </div>
                <div>
                    <h3 style="color: var(--accent-orange);">Light Theme</h3>
                    <img class="fig-img" src="{uri_fig1_light}" alt="Figure 1 Light Theme">
                </div>
            </div>
        </div>

        {"<div class='section-card'><h2>Figure 2: Standard 4-Panel Registration Quality Report</h2><p style='color: var(--text-secondary);'>Panel A: Deformed Mesh Grid | Panel B: Jacobian det(J) Map | Panel C: Inverse Error Map (mm) | Panel D: Canny Edge Alignment Overlap</p><img class='fig-img' src='" + uri_fig2 + "' alt='Figure 2 Report'></div>" if uri_fig2 else ""}

        <div class="section-card">
            <h2>Physical Vector & Deformation Tensor Field Displays</h2>
            <p style="color: var(--text-secondary);">Correspondence Vector Quiver, Displacement Vector Magnitude (mm), and Deformation Gradient Tensor RGB Strain Direction (Red: Left-Right, Green: Anterior-Posterior, Blue: Superior-Inferior).</p>
            <div class="grid-2col">
                {"<div><h3>Physical Correspondence Vectors</h3><img class='fig-img' src='" + uri_corr_vec + "' alt='Correspondence Vectors'></div>" if uri_corr_vec else ""}
                {"<div><h3>Deformation Vector Field Overlay</h3><img class='fig-img' src='" + uri_vector_field + "' alt='Deformation Vector Field'></div>" if uri_vector_field else ""}
            </div>
            {"<div style='margin-top: 20px;'><h3>Deformation Gradient Tensor RGB Strain Map</h3><p style='color: var(--text-secondary); font-size: 0.9rem;'>Eigen-direction of maximum spatial strain computed from physical deformation gradient matrix F = I + &nabla;u via ants.deformation_gradient.</p><img class='fig-img' src='" + uri_tensor_rgb + "' alt='Deformation Tensor RGB'></div>" if uri_tensor_rgb else ""}
        </div>

        {"<div class='section-card'><h2>Anatomical Label Segmentations (Mindboggle DKT Overlays)</h2><p style='color: var(--text-secondary);'>Fixed Target Labels (Top) vs Warped Moving Labels (Bottom) with high-contrast discrete qualitative colormapping in canonical LPI space.</p><img class='fig-img' src='" + uri_label_dark + "' alt='Label Alignment Dark'></div>" if uri_label_dark else ""}

        <div class="section-card">
            <h2>Statistical Quality Distributions & Benchmark Metrics</h2>
            <div class="grid-2col">
                {"<div><h3>Mindboggle DKT Dice Overlap</h3><img class='fig-img' src='" + uri_stats_dice + "' alt='Dice Stats'></div>" if uri_stats_dice else ""}
                {"<div><h3>Jacobian det(J) Regularization</h3><img class='fig-img' src='" + uri_stats_jac + "' alt='Jacobian Stats'></div>" if uri_stats_jac else ""}
            </div>
        </div>
    </div>
</body>
</html>
"""

    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return abs_path
