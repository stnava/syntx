"""HTML report generation for N-dimensional continuous elastic curve matching.

Follows syntx light-theme guidelines:
- Self-contained single-file HTML report with embedded responsive figures.
- KPI summary scorecards and interactive metric breakdowns.
- Methodological documentation and clinical/anatomical interpretation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

from syntx.curves.matching import CurveMatch
from syntx.curves.viz import fig_to_base64, plot_curve_match


def generate_curve_matching_html_report(
    output_path: str | Path,
    *,
    matches: Sequence[tuple],
    title: str = "Continuous Elastic Curve Matching Evaluation Report",
    subtitle: str = "syntx.curves — Multi-Scale Elastic Matching in R^d",
    extra_html: str = "",
    execution_info: str = "",
) -> str:
    """Generate a self-contained, publication-ready HTML evaluation report.

    Parameters
    ----------
    output_path : str or Path
        Destination HTML file path.
    matches : sequence of (name, CurveMatch) or (name, CurveMatch, gt_error)
        Curve matches to render. ``gt_error`` is the dict returned by
        :func:`syntx.curves.correspondence_error`; when given, ground-truth landmark error is
        shown as a KPI (the primary accuracy metric for simulated data).
    title : str
        Report header title.
    subtitle : str
        Report subtitle.
    extra_html : str
        Additional HTML sections (e.g. sweep figures) inserted before the methodology block.
    execution_info : str
        Device / runtime / seed conditions line shown in the header.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections_html = []
    for item in matches:
        match_name, m = item[0], item[1]
        gt = item[2] if len(item) > 2 else None
        fig = plot_curve_match(m, title=f"Curve Match Diagnostic: {match_name}")
        b64_img = fig_to_base64(fig, dpi=130)

        if gt is not None:
            acc_label = "Ground-Truth Landmark Error"
            acc_value = f"{gt['mean_error_fraction']:.4f} <span class='unit'>frac</span>"
            acc_sub = f"max {gt['max_error_fraction']:.4f} frac &bull; mean {gt['mean_error_mm']:.2f} mm"
        else:
            acc_label = "Feature-Space Gain"
            acc_value = f"{m.feature_gain_rel * 100:.0f} <span class='unit'>%</span>"
            acc_sub = "Reduction of curvature+tangent energy vs ramp"
        orient_badge = "Reversed" if m.reversed else "Direct"
        mode_badge = f"{m.mode.capitalize()} Curve (R^{m.dimension})"
        clamp_badge = '<span class="badge badge-warning">Rotation Clamped</span>' if m.rotation_clamped else ""

        inv_val = f"{m.inverse_consistency_rms:.4f} <span class='unit'>frac</span>" if m.inverse_consistency_rms is not None else "N/A"
        inv_sub = f"Physical: {m.inverse_consistency_rms_mm:.2f} mm" if m.inverse_consistency_rms_mm is not None else "Round-trip error"

        gain_sign = "+" if m.elastic_gain_mm >= 0 else ""
        gain_class = "text-success" if m.elastic_gain_mm >= 0 else "text-warning"

        sections_html.append(f"""
        <div class="bundle-card">
            <div class="bundle-header">
                <h2>{match_name}</h2>
                <div>
                    <span class="badge badge-info">{mode_badge}</span>
                    <span class="badge badge-success">{orient_badge} (gap {m.orientation_cost_gap_rel * 100:.0f}%)</span>
                    {clamp_badge}
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">{acc_label}</div>
                    <div class="kpi-value text-accent">{acc_value}</div>
                    <div class="kpi-sub">{acc_sub}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Elastic RMS Distance</div>
                    <div class="kpi-value">{m.elastic_rms_mm:.2f} <span class="unit">mm</span></div>
                    <div class="kpi-sub">Baseline RMS: {m.baseline_rms_mm:.2f} mm</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Spatial Gain (secondary)</div>
                    <div class="kpi-value {gain_class}">{gain_sign}{m.elastic_gain_mm:.2f} <span class="unit">mm</span></div>
                    <div class="kpi-sub">Can be negative for a correct match (features move apart)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Max Speed Stretch</div>
                    <div class="kpi-value">{m.energies.get('max_speed_stretch', 1.0):.2f} <span class="unit">x</span></div>
                    <div class="kpi-sub">Membrane energy: {m.energies.get('membrane', 0.0):.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Selected Scale &sigma;</div>
                    <div class="kpi-value">{m.sigma_selected_mm:.1f} <span class="unit">mm</span></div>
                    <div class="kpi-sub">Extrema topology scale</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Inverse Consistency</div>
                    <div class="kpi-value">{inv_val}</div>
                    <div class="kpi-sub">{inv_sub}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Sample Count & Spacing</div>
                    <div class="kpi-value">{m.n_points} <span class="unit">pts</span></div>
                    <div class="kpi-sub">{m.spacing_mm:.2f} mm/sample</div>
                </div>
            </div>

            <div class="figure-container">
                <img src="{b64_img}" alt="Curve Match Diagnostic" class="report-fig" />
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #ffffff;
            --surface: #f8fafc;
            --surface-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --accent: #0284c7;
            --accent-light: #e0f2fe;
            --success: #10b981;
            --success-light: #d1fae5;
            --warning: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding: 2.5rem 3rem;
        }}
        .header {{
            border-bottom: 2px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2.5rem;
        }}
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }}
        .header p {{
            font-size: 1.05rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .header-meta {{
            margin-top: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        .bundle-card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.75rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }}
        .bundle-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }}
        .bundle-header h2 {{
            font-size: 1.35rem;
            font-weight: 600;
            color: var(--text-main);
        }}
        .badge {{
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}
        .badge-success {{ background: var(--success-light); color: #065f46; }}
        .badge-info {{ background: var(--accent-light); color: #0369a1; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.75rem;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.25rem;
        }}
        .kpi-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }}
        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        .kpi-value .unit {{
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }}
        .kpi-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}
        .text-accent {{ color: var(--accent); }}
        .text-success {{ color: var(--success); }}
        .text-warning {{ color: #b45309; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .sweep-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
        .sweep-note {{ font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem; }}
        .figure-container {{
            margin-top: 1.5rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .report-fig {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
        }}
        .methodology {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.75rem;
            margin-top: 2.5rem;
        }}
        .methodology h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        .methodology p, .methodology ul {{
            font-size: 0.9rem;
            color: var(--text-main);
            margin-bottom: 0.75rem;
        }}
        .methodology ul {{ padding-left: 1.5rem; }}
        .methodology li {{ margin-bottom: 0.4rem; }}
        code {{
            background: #e2e8f0;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85em;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <div class="header-meta">Generated: {timestamp} &bull; Library: <code>syntx.curves</code>{(' &bull; ' + execution_info) if execution_info else ''}</div>
    </div>

    {''.join(sections_html)}

    {extra_html}

    <div class="methodology">
        <h2>Methodological Reference & Architecture</h2>
        <p>
            This module ports and advances the continuous parametric curve matching framework of 
            <strong>Avants & Gee (2003, Scale-Space LNCS 2695)</strong> to arbitrary <em>N</em>-dimensional Euclidean spaces:
        </p>
        <ul>
            <li><strong>Dimension Invariance (R^d)</strong>: Supports 2-D planar contours, 3-D spatial curves, and higher-dimensional trajectories.</li>
            <li><strong>Dual-Mode Dynamic Programming</strong>: Unified engine for open curves (strictly pinned endpoints) and closed curves (cyclic offset sweep).</li>
            <li><strong>Direction-Encoded Tangent Field</strong>: Multi-scale unit tangent inner product (<code>1 - T_P &bull; T_Q</code>) providing full directional sensitivity in S^(d-1).</li>
            <li><strong>Orientation Resolution (Direct vs. Flipped)</strong>: Resolves tracking and traversal orientation by evaluating both directions under dynamic time warping.</li>
            <li><strong>SO(d) Geodesic Rotation Bounds</strong>: Optimal Procrustes alignment bounded on Lie group manifolds to prevent unphysical coordinate flips.</li>
            <li><strong>Membrane & Bending Regularization</strong>: Penalizes non-linear stretching (&alpha;&int;(g'&minus;1)&sup2;) and path acceleration (&beta;&int;g''&sup2;), ensuring diffeomorphic correspondence.</li>
        </ul>
    </div>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")
    return str(out_file)
