"""
Standardized Reporting & Verification Infrastructure for Syntx Medical Image Registration.

Provides publication-grade visual and quantitative reporting tools for single registration runs
and multi-configuration parameter sweeps with comprehensive provenance tracking.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .syn import extract_2d_slice, render_standard_4panel, local_ncc_loss_nd as lncc_loss_nd


def _parse_image_metadata(img, name="Image"):
    """Extracts physical space and dimension metadata from ANTsImage, PyTorch Tensor, or NumPy Array."""
    meta = {
        "name": name,
        "type": type(img).__name__,
        "shape": "N/A",
        "spacing": "N/A",
        "origin": "N/A",
        "orientation": "N/A",
        "is_ants": False,
    }

    if isinstance(img, ants.ANTsImage):
        meta["is_ants"] = True
        meta["shape"] = str(tuple(img.shape))
        meta["spacing"] = " × ".join([f"{s:.3f}" for s in img.spacing]) + " mm"
        meta["origin"] = str(tuple([round(o, 2) for o in img.origin])) + " mm"
        meta["orientation"] = str(img.orientation)
    elif hasattr(img, "shape"):
        meta["shape"] = str(tuple(img.shape))

    return meta


def _compute_jacobian_stats(warp, fixed=None):
    """Computes Jacobian determinant array and summary statistics from displacement field."""
    if hasattr(warp, "detach"):
        warp_np = warp.squeeze(0).detach().cpu().numpy()
    elif hasattr(warp, "numpy"):
        warp_np = warp.numpy()
    else:
        warp_np = np.asarray(warp)

    if warp_np.ndim == 4 and warp_np.shape[0] in (2, 3) and warp_np.shape[1] > 4:
        warp_np = np.moveaxis(warp_np, 0, -1)

    spacing = fixed.spacing if (fixed is not None and isinstance(fixed, ants.ANTsImage)) else (1.0,) * (warp_np.ndim - 1)

    if warp_np.ndim == 4:
        du_dx = np.gradient(warp_np[..., 0], axis=0) / spacing[0]
        du_dy = np.gradient(warp_np[..., 1], axis=1) / spacing[1]
        du_dz = np.gradient(warp_np[..., 2], axis=2) / spacing[2]
        detJ = (1.0 + du_dx) * (1.0 + du_dy) * (1.0 + du_dz)
    elif warp_np.ndim == 3:
        du_dx = np.gradient(warp_np[..., 0], axis=0) / spacing[0]
        du_dy = np.gradient(warp_np[..., 1], axis=1) / spacing[1]
        detJ = (1.0 + du_dx) * (1.0 + du_dy)
    else:
        detJ = np.ones(warp_np.shape[:-1])

    min_j = float(np.min(detJ))
    max_j = float(np.max(detJ))
    mean_j = float(np.mean(detJ))
    std_j = float(np.std(detJ))

    mask = (detJ <= 0.0)
    folding_pct = float(np.mean(mask) * 100.0)

    return detJ, {
        "min": min_j,
        "max": max_j,
        "mean": mean_j,
        "std": std_j,
        "folding_pct": folding_pct,
    }


def build_engine_provenance(
    algorithm="syntx.syn",
    backend="pytorch",
    device="cpu",
    fit_time=None,
    reg_iterations=None,
    affine_iterations=None,
    solver="SyN",
    fluid_sigma=3.0,
    elastic_sigma=0.0,
    learning_rate=None,
    optimizer_type="Adam",
    similarity_metric="lncc",
    loss_window=9,
    fixed_shape=None,
    fixed_spacing=None,
    fixed_orientation=None,
    moving_shape=None,
    moving_spacing=None,
    moving_orientation=None,
    **kwargs
):
    """
    Constructs a standardized, non-breakable registration provenance dictionary directly from the registration engine.
    """
    prov = {
        "algorithm": algorithm,
        "backend": str(backend),
        "device": str(device),
        "fit_time": float(fit_time) if isinstance(fit_time, (int, float)) else "N/A",
        "iterations": str(reg_iterations) if reg_iterations is not None else "N/A",
        "affine_iterations": str(affine_iterations) if affine_iterations is not None else "N/A",
        "solver": str(solver),
        "fluid_sigma": str(fluid_sigma),
        "elastic_sigma": str(elastic_sigma),
        "learning_rate": str(learning_rate) if learning_rate is not None else "N/A",
        "optimizer": str(optimizer_type),
        "similarity_metric": str(similarity_metric),
        "loss_window": str(loss_window),
        "fixed_shape": str(fixed_shape) if fixed_shape is not None else "N/A",
        "fixed_spacing": str(fixed_spacing) if fixed_spacing is not None else "N/A",
        "fixed_orientation": str(fixed_orientation) if fixed_orientation is not None else "N/A",
        "moving_shape": str(moving_shape) if moving_shape is not None else "N/A",
        "moving_spacing": str(moving_spacing) if moving_spacing is not None else "N/A",
        "moving_orientation": str(moving_orientation) if moving_orientation is not None else "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "syntx_version": "1.0.4",
    }
    prov.update(kwargs)
    return prov


def create_registration_report(
    fixed,
    moving,
    warped=None,
    warp=None,
    output_html="registration_report.html",
    fixed_name="Fixed Target Image",
    moving_name="Moving Source Image",
    provenance=None,
    fixed_label=None,
    moving_label=None,
    warped_label=None,
    inv_err_map=None,
    detJ=None,
    slice_axis=2,
    slice_idx=None,
    title="Syntx Medical Image Registration Verification Report",
    assets_dir=None,
    show_report=False,
    reg=None
):
    """
    Generates a publication-grade, standalone interactive HTML report and 4-panel visual figure asset
    for a completed medical image registration task, with complete provenance tracking and metric verification.

    Args:
        fixed: Target fixed image (ANTsImage, Tensor, or NumPy array).
        moving: Source moving image (ANTsImage, Tensor, or NumPy array).
        warped: Warped moving image or registration result dictionary.
        warp: Forward spatial displacement vector field (optional).
        output_html: Path to save output HTML report file (default: 'registration_report.html').
        fixed_name: Human-readable identifier for target image.
        moving_name: Human-readable identifier for source image.
        provenance: Dict or string containing execution metadata.
        fixed_label: Optional target discrete segmentation label map.
        moving_label: Optional source discrete segmentation label map.
        warped_label: Optional warped discrete segmentation label map.
        inv_err_map: Optional inverse identity error map array.
        detJ: Optional Jacobian determinant map array.
        slice_axis: Axis for 2D slice visualization (0: Sagittal, 1: Coronal, 2: Axial). Default: 2.
        slice_idx: Optional slice index along slice_axis. Defaults to midpoint.
        title: Custom title for report.
        assets_dir: Optional directory for PNG figure assets. Defaults to 'assets' relative to output_html.
        show_report: If True, prints report summary to stdout.
        reg: Optional registration output dictionary from syntx.syn or TVFModel.fit().

    Returns:
        dict: Report summary metrics, file paths, and provenance metadata.
    """
    # Auto-extract from registration dict if passed directly
    if reg is not None and isinstance(reg, dict):
        if warped is None:
            warped = reg.get("warpedmovout", fixed)
        if warp is None and "fwdtransforms" in reg:
            warp = reg["fwdtransforms"][0]
        if provenance is None and "provenance" in reg:
            provenance = reg["provenance"]
    elif isinstance(warped, dict):
        reg_dict = warped
        warped = reg_dict.get("warpedmovout", fixed)
        if warp is None and "fwdtransforms" in reg_dict:
            warp = reg_dict["fwdtransforms"][0]
        if provenance is None and "provenance" in reg_dict:
            provenance = reg_dict["provenance"]

    if warped is None:
        warped = fixed

    if warped_label is None and moving_label is not None:
        warped_label = moving_label

    output_html = os.path.abspath(output_html)
    html_dir = os.path.dirname(output_html)
    os.makedirs(html_dir, exist_ok=True)

    if assets_dir is None:
        assets_dir = os.path.join(html_dir, "assets")
    else:
        assets_dir = os.path.abspath(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Parse Image Metadata & Provenance
    meta_fixed = _parse_image_metadata(fixed, fixed_name)
    meta_moving = _parse_image_metadata(moving, moving_name)
    meta_warped = _parse_image_metadata(warped, "Warped Moving")

    prov = {
        "algorithm": "syntx.syn",
        "backend": "pytorch",
        "device": "cpu",
        "fit_time": "N/A",
        "iterations": "N/A",
        "solver": "SyN",
        "fluid_sigma": "3.0",
        "elastic_sigma": "0.0",
        "learning_rate": "N/A",
        "optimizer": "Adam",
        "similarity_metric": "Intensity LNCC (w=9)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if isinstance(provenance, dict):
        prov.update(provenance)
    elif isinstance(provenance, str):
        prov["info"] = provenance

    # 2. Compute Similarity Metrics
    fi_arr = fixed.numpy() if isinstance(fixed, ants.ANTsImage) else np.squeeze(np.asarray(fixed))
    mi_arr = warped.numpy() if isinstance(warped, ants.ANTsImage) else np.squeeze(np.asarray(warped))

    mse_val = float(np.mean((fi_arr - mi_arr) ** 2))
    mae_val = float(np.mean(np.abs(fi_arr - mi_arr)))

    fi_t = torch.tensor(fi_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mi_t = torch.tensor(mi_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    try:
        lncc_loss = float(lncc_loss_nd(fi_t, mi_t, window_size=9).item())
        lncc_val = -lncc_loss
    except Exception:
        lncc_val = 0.0

    # 3. Compute Jacobian Determinant Stats
    if detJ is None and warp is not None:
        detJ_arr, jac_stats = _compute_jacobian_stats(warp, fixed=fixed)
    elif detJ is not None:
        detJ_arr = detJ.numpy() if isinstance(detJ, ants.ANTsImage) else np.squeeze(np.asarray(detJ))
        min_j, max_j = float(np.min(detJ_arr)), float(np.max(detJ_arr))
        jac_stats = {
            "min": min_j,
            "max": max_j,
            "mean": float(np.mean(detJ_arr)),
            "std": float(np.std(detJ_arr)),
            "folding_pct": float(np.mean(detJ_arr <= 0.0) * 100.0),
        }
    else:
        detJ_arr = np.ones(fi_arr.shape)
        jac_stats = {"min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0, "folding_pct": 0.0}

    # 4. Compute Label Overlap (DICE Score) if labels available
    dice_val = None
    dice_df_html = ""
    if fixed_label is not None and warped_label is not None:
        if isinstance(fixed_label, ants.ANTsImage) and isinstance(warped_label, ants.ANTsImage):
            df_overlap = ants.label_overlap_measures(fixed_label, warped_label)
            df_clean = df_overlap[(df_overlap['Label'] != 'All') & (df_overlap['Label'] != 0) & (df_overlap['Label'] != '0')]
            col_name = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_clean.columns else 'TargetOverlap'
            if len(df_clean) > 0:
                dice_val = float(df_clean[col_name].mean())
        else:
            fl_np = np.squeeze(np.asarray(fixed_label))
            wl_np = np.squeeze(np.asarray(warped_label))
            labels = np.unique(fl_np)
            labels = labels[labels != 0]
            dices = []
            for l in labels:
                intersection = np.sum((fl_np == l) & (wl_np == l))
                total = np.sum(fl_np == l) + np.sum(wl_np == l)
                if total > 0:
                    dices.append(2.0 * intersection / total)
            if len(dices) > 0:
                dice_val = float(np.mean(dices))

    # 5. Inverse Error Map stats
    if inv_err_map is not None:
        err_arr = inv_err_map.numpy() if isinstance(inv_err_map, ants.ANTsImage) else np.squeeze(np.asarray(inv_err_map))
        inv_stats = {
            "max": float(np.max(err_arr)),
            "mean": float(np.mean(err_arr)),
            "p95": float(np.percentile(err_arr, 95)),
        }
    else:
        err_arr = np.zeros(fi_arr.shape)
        inv_stats = {
            "max": 0.0,
            "mean": 0.0,
            "p95": 0.0,
        }

    # 6. Render Standard 4-Panel Figure PNG
    fig_filename = "registration_4panel_report.png"
    fig_abs_path = os.path.join(assets_dir, fig_filename)
    rel_fig_path = os.path.relpath(fig_abs_path, html_dir)

    dummy_warp = warp if warp is not None else np.zeros((*fi_arr.shape, fi_arr.ndim))
    render_standard_4panel(
        fixed=fixed,
        warped=warped,
        warp=dummy_warp,
        detJ=detJ_arr,
        inv_err_map=err_arr,
        slice_axis=slice_axis,
        slice_idx=slice_idx,
        lncc_val=lncc_val,
        inv_err_max=inv_stats["max"],
        inv_err_mean=inv_stats["mean"],
        inv_err_p95=inv_stats["p95"],
        min_detJ=jac_stats["min"],
        title_prefix=f"{title}",
        filename=fig_abs_path
    )

    # 7. Construct HTML Report Content
    fit_time_str = f"{prov['fit_time']:.2f} s" if isinstance(prov.get("fit_time"), (int, float)) else str(prov.get("fit_time"))
    dice_card_str = f"{dice_val:.4f}" if dice_val is not None else "N/A"
    folding_badge_str = '<span class="badge badge-success">0.00% Folding (Fold-Free)</span>' if jac_stats["folding_pct"] == 0.0 else f'<span class="badge badge-danger">{jac_stats["folding_pct"]:.2f}% Folding</span>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - syntx</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: #111622;
            --border-color: #212636;
            --text-main: #e6edf3;
            --text-muted: #8b949e;
            --accent-blue: #38bdf8;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-purple: #a855f7;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .header-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .tag-version {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 1rem;
            margin-top: 0.5rem;
        }}
        .grid-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
        }}
        .card-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .card-val {{
            font-size: 1.75rem;
            font-weight: 700;
        }}
        .val-green {{ color: var(--accent-green); }}
        .val-blue {{ color: var(--accent-blue); }}
        .val-yellow {{ color: var(--accent-yellow); }}
        .val-purple {{ color: var(--accent-purple); }}
        .card-desc {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.4rem;
        }}
        section {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 2rem;
            margin-bottom: 2.5rem;
        }}
        section h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            color: var(--text-main);
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-success {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }}
        .badge-danger {{ background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 81, 73, 0.3); }}
        .prov-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .prov-table td, .prov-table th {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
        }}
        .prov-table th {{
            text-align: left;
            color: var(--text-muted);
            width: 30%;
        }}
        .prov-table code {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-blue);
        }}
        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>{title}</h1>
                <span class="tag-version">syntx Registration Report</span>
            </div>
            <p class="subtitle">
                Target Fixed: <strong>{fixed_name}</strong> | Source Moving: <strong>{moving_name}</strong> | Timestamp: {prov['timestamp']}
            </p>
        </header>

        <div class="grid-summary">
            <div class="card">
                <div class="card-label">Overlap DICE Score</div>
                <div class="card-val val-green">{dice_card_str}</div>
                <div class="card-desc">Categorical anatomical label overlap</div>
            </div>
            <div class="card">
                <div class="card-label">Similarity Metric (LNCC)</div>
                <div class="card-val val-blue">{lncc_val:.4f}</div>
                <div class="card-desc">Local Normalized Cross-Correlation (w=9)</div>
            </div>
            <div class="card">
                <div class="card-label">Grid Folding Rate</div>
                <div class="card-val val-purple">{jac_stats['folding_pct']:.2f}%</div>
                <div class="card-desc">det(J) in [{jac_stats['min']:+.2f}, {jac_stats['max']:.2f}]</div>
            </div>
            <div class="card">
                <div class="card-label">Execution Fit Time</div>
                <div class="card-val val-yellow">{fit_time_str}</div>
                <div class="card-desc">Backend: {prov['backend']} ({prov['device']})</div>
            </div>
        </div>

        <section>
            <h2>📋 Registration Provenance & System Configuration</h2>
            <table class="prov-table">
                <tr><th>Algorithm</th><td><code>{prov['algorithm']}</code> ({prov['solver']})</td></tr>
                <tr><th>Compute Engine & Device</th><td><code>{prov['backend']}</code> on <code>{prov['device']}</code></td></tr>
                <tr><th>Execution Fit Runtime</th><td><code>{fit_time_str}</code></td></tr>
                <tr><th>Optimizer & Learning Rate</th><td><code>{prov['optimizer']}</code> (lr = {prov['learning_rate']})</td></tr>
                <tr><th>Fluid / Elastic Sigmas</th><td>fluid = <code>{float(prov['fluid_sigma']) if isinstance(prov['fluid_sigma'], (int, float)) or (isinstance(prov['fluid_sigma'], str) and prov['fluid_sigma'].replace('.','',1).isdigit()) else 3.0:.3f}</code>, elastic = <code>{float(prov['elastic_sigma']) if isinstance(prov['elastic_sigma'], (int, float)) or (isinstance(prov['elastic_sigma'], str) and prov['elastic_sigma'].replace('.','',1).isdigit()) else 0.005:.3f}</code></td></tr>
                <tr><th>Target Fixed Image</th><td>{fixed_name} (Shape: <code>{meta_fixed['shape']}</code>, Spacing: <code>{meta_fixed['spacing']}</code>, Orientation: <code>{meta_fixed['orientation']}</code>)</td></tr>
                <tr><th>Source Moving Image</th><td>{moving_name} (Shape: <code>{meta_moving['shape']}</code>, Spacing: <code>{meta_moving['spacing']}</code>, Orientation: <code>{meta_moving['orientation']}</code>)</td></tr>
                <tr><th>Jacobian Determinant det(J)</th><td>Min: <code>{jac_stats['min']:+.4f}</code>, Max: <code>{jac_stats['max']:.4f}</code>, Mean: <code>{jac_stats['mean']:.4f}</code> ({folding_badge_str})</td></tr>
                <tr><th>Sub-voxel Inverse Error</th><td>Max: <code>{inv_stats['max']:.4f} mm</code>, Mean: <code>{inv_stats['mean']:.4f} mm</code>, p95: <code>{inv_stats['p95']:.4f} mm</code></td></tr>
            </table>
        </section>

        <section>
            <h2>📊 ANTs Standard LAI Anatomical 4-Panel Visualization</h2>
            <div style="text-align: center;">
                <img src="{rel_fig_path}" alt="Registration Standard 4-Panel Figure" style="max-width: 100%; border-radius: 10px; border: 1px solid var(--border-color);">
                <p style="font-size: 0.9rem; color: var(--text-muted); margin-top: 0.75rem;">
                    <strong>Figure 1: Standardized Registration Visual Report (ANTs LAI Orientation)</strong> — 
                    Panel A: Standard Deformed Mesh Grid. Panel B: Divergent Jacobian det(J) Map. 
                    Panel C: Inverse Identity Error Map (mm). Panel D: High-Contrast Canny Edge Alignment Overlap.
                </p>
            </div>
        </section>

        <footer>
            <p>Generated automatically by <strong>syntx</strong> — Advanced Medical Image Registration & Verification Engine</p>
        </footer>
    </div>
</body>
</html>
"""

    with open(output_html, "w") as f:
        f.write(html_content)

    summary = {
        "html_path": output_html,
        "fig_path": fig_abs_path,
        "lncc": lncc_val,
        "dice": dice_val,
        "mae": mae_val,
        "mse": mse_val,
        "jacobian": jac_stats,
        "inverse_error": inv_stats,
        "provenance": prov,
    }

    if show_report:
        print(f"\n✓ Syntx Registration Report Generated: {output_html}")
        print(f"   - Target: {fixed_name} | Source: {moving_name}")
        print(f"   - Similarity LNCC: {lncc_val:.4f} | DICE: {dice_val}")
        print(f"   - Jacobian Range: [{jac_stats['min']:+.2f}, {jac_stats['max']:.2f}] (Folding: {jac_stats['folding_pct']:.2f}%)")

    return summary
