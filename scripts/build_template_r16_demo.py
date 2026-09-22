#!/usr/bin/env python
"""
scripts/build_template_r16_demo.py
===================================
Demonstrates syntx.build_template and syntx.reflect_image on the r16 brain
phantom from ANTsPy.

Pipeline
--------
1.  Load r16 (a 2D coronal brain slice, 256×256, 1mm isotropic).
2.  Create r16_reflected = syntx.reflect_image(r16, axis='x')  (L↔R flip).
3.  Run syntx.build_template([r16, r16_reflected], iterations=8, ...).
4.  Produce a publication-grade HTML + PNG report using syntx.viz conventions:
    - All slice display via extract_oriented_slice / plot_edge_overlay /
      plot_deformation_grid (physical-space, no ad-hoc .numpy().T).
    - Row 0: r16 | r16_reflected | initial mean template
    - Row 1: warped_r16 | warped_r16_reflected | final template
    - Row 2: edge overlays (warped on template, both subjects) + |diff|
    - Row 3: deformation grids + Jacobian det(J)
    - Row 4: convergence curve (template MAE)
    - Row 5: shape residual curves (L2, membrane, bending)
    - Row 6: numeric metrics table

Output
------
  docs/reports/build_template_r16_demo.png
  docs/reports/build_template_r16_demo.html
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ants

# ---- Ensure syntx is importable from repo root ----------------------------
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src = os.path.join(_repo, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

import syntx
from syntx.viz.figures import extract_oriented_slice, plot_edge_overlay, plot_deformation_grid

# ---------------------------------------------------------------------------
# Theme (GEMINI.md §5 — light theme, no dark backgrounds)
# ---------------------------------------------------------------------------
BG      = '#ffffff'
SLATE   = '#1e293b'
CYAN    = '#06b6d4'
EMERALD = '#10b981'
AMBER   = '#f59e0b'
ROSE    = '#f43f5e'

TITLE_FS = 10
LABEL_FS = 8
TICK_FS  = 7

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.edgecolor':   SLATE,
    'axes.labelcolor':  SLATE,
    'text.color':       SLATE,
    'xtick.color':      SLATE,
    'ytick.color':      SLATE,
    'grid.color':       '#e2e8f0',
    'grid.linewidth':   0.6,
    'font.family':      'sans-serif',
})


# ---------------------------------------------------------------------------
# Physical-space display helpers
# ---------------------------------------------------------------------------

def _imshow(ax, img, title='', cmap='gray', vmin=None, vmax=None):
    """
    Display an ANTsImage using extract_oriented_slice for physical-space
    correct orientation and aspect ratio. For 2D images uses slice_axis=2
    (no actual slicing — returns full array with correct aspect).
    """
    arr, aspect = extract_oriented_slice(img, slice_axis=2, reorient=True)
    vmin_ = vmin if vmin is not None else float(np.percentile(arr[arr > 0], 1)) if (arr > 0).any() else arr.min()
    vmax_ = vmax if vmax is not None else float(np.percentile(arr[arr > 0], 99)) if (arr > 0).any() else arr.max()
    ax.imshow(arr, cmap=cmap, aspect=aspect, vmin=vmin_, vmax=vmax_,
              interpolation='bilinear')
    ax.set_title(title, fontsize=TITLE_FS, color=SLATE, pad=3)
    ax.axis('off')


def _edge_overlay_ax(ax, fixed_img, warped_img, title=''):
    """
    Edge overlay using syntx.viz.plot_edge_overlay — physical-space correct,
    light theme.
    """
    plot_edge_overlay(
        fixed=fixed_img,
        warped=warped_img,
        slice_axis=2,
        edge_color=CYAN,
        fixed_edge_color=AMBER,
        alpha=0.80,
        theme='light',
        ax=ax,
        title=title,
    )


def _warp_grid_ax(ax, warp_file, ref_img, title=''):
    """
    Deformation grid using syntx.viz.plot_deformation_grid — physical-space
    correct, light theme.
    """
    if warp_file is None or not os.path.isfile(warp_file):
        ax.set_title(title + ' (no warp file)', fontsize=TITLE_FS, color=SLATE)
        ax.axis('off')
        return
    warp_img = ants.image_read(warp_file)
    plot_deformation_grid(
        warp=warp_img,
        fixed=ref_img,
        slice_axis=2,
        grid_spacing=12,
        line_color=CYAN,
        theme='light',
        ax=ax,
        title=title,
    )


def _jacobian_ax(ax, warp_file, template, title='Jacobian det(J)'):
    """
    Jacobian determinant overlay — physical-space via extract_oriented_slice.
    """
    if warp_file is None or not os.path.isfile(warp_file):
        ax.set_title(title + ' (no warp)', fontsize=TITLE_FS, color=SLATE)
        ax.axis('off')
        return
    try:
        warp_img = ants.image_read(warp_file)
        jac = ants.create_jacobian_determinant_image(template, warp_img, do_log=False)
        arr, aspect = extract_oriented_slice(jac, slice_axis=2, reorient=True)
        vabs = max(abs(float(arr.min()) - 1.0), abs(float(arr.max()) - 1.0), 0.15)
        im = ax.imshow(arr, cmap='RdBu_r', aspect=aspect,
                       vmin=1.0 - vabs, vmax=1.0 + vabs, interpolation='bilinear')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=TITLE_FS, color=SLATE, pad=3)
        ax.axis('off')
    except Exception as e:
        ax.set_title(f'{title} (err: {e})', fontsize=TICK_FS, color=SLATE)
        ax.axis('off')


def _diff_ax(ax, img_a, img_b, title=''):
    """Absolute difference between two images, physical-space correct."""
    arr_a, aspect = extract_oriented_slice(img_a, slice_axis=2, reorient=True)
    arr_b, _      = extract_oriented_slice(img_b, slice_axis=2, reorient=True)
    diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
    ax.imshow(diff, cmap='hot', aspect=aspect, vmin=0,
              vmax=float(np.percentile(diff, 99.5)), interpolation='bilinear')
    ax.set_title(title, fontsize=TITLE_FS, color=SLATE, pad=3)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='build_template r16 demo')
    parser.add_argument('--iterations',      type=int,   default=8)
    parser.add_argument('--gradient-step',   type=float, default=0.20)
    parser.add_argument('--blending-weight', type=float, default=0.75)
    parser.add_argument('--output-dir',      default=None)
    parser.add_argument('--report-dir',      default='docs/reports')
    parser.add_argument('--syn-metric',      default='mattes')
    parser.add_argument('--flow-sigma',      type=float, default=3.0)
    parser.add_argument('--reg-iterations',  default='30,10,0')
    args = parser.parse_args()

    reg_iters = [int(x) for x in args.reg_iterations.split(',')]
    os.makedirs(args.report_dir, exist_ok=True)
    out_png  = os.path.join(args.report_dir, 'build_template_r16_demo.png')
    out_html = os.path.join(args.report_dir, 'build_template_r16_demo.html')

    # ---- 1. Load r16 -------------------------------------------------------
    print("[demo] Loading r16 …")
    r16 = ants.image_read(ants.get_ants_data('r16'))
    print(f"  r16: shape={r16.shape}  spacing={r16.spacing}  origin={r16.origin}")

    # ---- 2. Reflect --------------------------------------------------------
    print("[demo] Reflecting r16 along X axis (L↔R) …")
    r16_ref = syntx.reflect_image(r16, axis='x')
    print(f"  r16_reflected: shape={r16_ref.shape}  origin={r16_ref.origin}")

    # ---- 3. Initial template (exact symmetric mean) ------------------------
    init_template = (r16 + r16_ref) * 0.5

    # ---- 4. build_template -------------------------------------------------
    print(f"\n[demo] Running build_template ({args.iterations} iterations) …")
    t0 = time.time()
    result = syntx.build_template(
        image_list=[r16, r16_ref],
        iterations=args.iterations,
        gradient_step=args.gradient_step,
        blending_weight=args.blending_weight,
        type_of_transform='SyN',
        syn_metric=args.syn_metric,
        flow_sigma=args.flow_sigma,
        reg_iterations=reg_iters,
        output_dir=args.output_dir,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[demo] Finished in {elapsed:.1f}s")

    template    = result['template']
    warped_r16  = result['warped_images'][0]
    warped_ref  = result['warped_images'][1]
    convergence = result['convergence']
    shape_res   = result['shape_residuals']
    n_iters     = result['n_iterations']
    fwd_r16     = result['fwdtransforms'][0]
    fwd_ref     = result['fwdtransforms'][1]

    # Template asymmetry
    tmpl_reflected = syntx.reflect_image(template, axis='x')
    tmpl_asym_val = float((template - tmpl_reflected).abs().mean())
    print(f"\n[demo] Final template L/R asymmetry MAE: {tmpl_asym_val:.4f}")

    # ---- 5. Print table ----------------------------------------------------
    print("\n  Iter │  Template MAE  │  L2 (mm)  │  Membrane  │  Bending")
    print("  " + "─" * 62)
    for i, (mae, sr) in enumerate(zip(convergence, shape_res)):
        lbl = f"Iter {i}" + (" (init)" if i == 0 else "")
        print(f"  {lbl:>9s} │  {mae:>12.6f}  │  {sr['l2_norm']:>9.4f}  │"
              f"  {sr['membrane_energy']:>10.4f}  │  {sr['bending_energy']:>10.6f}")

    # ---- 6. Build figure ---------------------------------------------------
    print("\n[demo] Building figure …")

    warp_r16_file = next(
        (t for t in fwd_r16 if t.endswith('.nii.gz') or t.endswith('.nii')), None)
    warp_ref_file = next(
        (t for t in fwd_ref if t.endswith('.nii.gz') or t.endswith('.nii')), None)

    fig = plt.figure(figsize=(18, 24), facecolor=BG)
    gs  = gridspec.GridSpec(
        7, 3, figure=fig,
        hspace=0.38, wspace=0.06,
        top=0.965, bottom=0.03, left=0.04, right=0.97,
    )

    # ── Row 0: Inputs ────────────────────────────────────────────────────────
    _imshow(fig.add_subplot(gs[0, 0]), r16,           'Input: r16')
    _imshow(fig.add_subplot(gs[0, 1]), r16_ref,       'Input: r16 reflected (L↔R)')
    _imshow(fig.add_subplot(gs[0, 2]), init_template, 'Initial template (mean)')

    # ── Row 1: Warped + final template ───────────────────────────────────────
    _imshow(fig.add_subplot(gs[1, 0]), warped_r16, 'Warped r16 → template')
    _imshow(fig.add_subplot(gs[1, 1]), warped_ref, 'Warped r16_reflected → template')
    _imshow(fig.add_subplot(gs[1, 2]), template,   f'Final template ({n_iters} iters)')

    # ── Row 2: Edge overlays + template asymmetry ────────────────────────────
    _edge_overlay_ax(fig.add_subplot(gs[2, 0]), template, warped_r16,
                     'Edge overlay: warped r16 on template')
    _edge_overlay_ax(fig.add_subplot(gs[2, 1]), template, warped_ref,
                     'Edge overlay: warped r16_reflected on template')
    _diff_ax(fig.add_subplot(gs[2, 2]), template, tmpl_reflected,
             f'Template asymmetry: |T − reflect(T)| (MAE={tmpl_asym_val:.3f})')

    # ── Row 3: Deformation grids + Jacobian ──────────────────────────────────
    _warp_grid_ax(fig.add_subplot(gs[3, 0]), warp_r16_file, template,
                  'Deformation grid: r16 → template')
    _warp_grid_ax(fig.add_subplot(gs[3, 1]), warp_ref_file, template,
                  'Deformation grid: r16_reflected → template')
    _jacobian_ax(fig.add_subplot(gs[3, 2]), warp_r16_file, template,
                 'Jacobian det(J): r16 warp')

    # ── Row 4: Convergence curve ─────────────────────────────────────────────
    ax_conv = fig.add_subplot(gs[4, :])
    iters_x = list(range(len(convergence)))
    ax_conv.plot(iters_x, convergence, 'o-', color=CYAN, lw=2, ms=6,
                 label='Template sharpen-blend MAE')
    ax_conv.set_xlabel('Iteration', fontsize=LABEL_FS)
    ax_conv.set_ylabel('MAE', fontsize=LABEL_FS)
    ax_conv.set_title(
        'Template Convergence — Mean Absolute Change per Iteration (sharpen-blend MAE)',
        fontsize=TITLE_FS)
    ax_conv.legend(fontsize=LABEL_FS)
    ax_conv.grid(True)
    ax_conv.tick_params(labelsize=TICK_FS)
    ax_conv.set_xticks(iters_x)

    # ── Row 5: Shape residual curves ─────────────────────────────────────────
    ax_sr  = fig.add_subplot(gs[5, :])
    ax_sr2 = ax_sr.twinx()

    l2s  = [sr['l2_norm']         for sr in shape_res]
    mems = [sr['membrane_energy'] for sr in shape_res]
    bnds = [sr['bending_energy']  for sr in shape_res]

    ax_sr.plot( iters_x, l2s,  'o-',  color=EMERALD, lw=2,   ms=6, label='L2 norm (mm)')
    ax_sr2.plot(iters_x, mems, 's--', color=AMBER,   lw=1.5, ms=5, label='Membrane energy')
    ax_sr2.plot(iters_x, bnds, '^-.', color=ROSE,    lw=1.5, ms=5, label='Bending energy')

    ax_sr.set_xlabel('Iteration', fontsize=LABEL_FS)
    ax_sr.set_ylabel('L2 norm / mm',   fontsize=LABEL_FS, color=EMERALD)
    ax_sr2.set_ylabel('Energy',        fontsize=LABEL_FS, color=SLATE)
    ax_sr.set_title('Shape Residual of Mean Warp Field ψ̄ — L2, Membrane & Bending Energy',
                    fontsize=TITLE_FS)

    lines1, lbls1 = ax_sr.get_legend_handles_labels()
    lines2, lbls2 = ax_sr2.get_legend_handles_labels()
    ax_sr.legend(lines1 + lines2, lbls1 + lbls2, fontsize=LABEL_FS, loc='upper left')
    ax_sr.grid(True)
    ax_sr.tick_params(labelsize=TICK_FS)
    ax_sr2.tick_params(labelsize=TICK_FS)
    ax_sr.set_xticks(iters_x)

    # ── Row 6: Metrics table ─────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[6, :])
    ax_tbl.axis('off')
    col_labels = ['Iter', 'Template MAE', 'L2 norm (mm)', 'Membrane energy', 'Bending energy']
    rows = []
    for i, (mae, sr) in enumerate(zip(convergence, shape_res)):
        rows.append([
            f"Iter {i}" + (" (init)" if i == 0 else ""),
            f"{mae:.6f}",
            f"{sr['l2_norm']:.4f}",
            f"{sr['membrane_energy']:.4f}",
            f"{sr['bending_energy']:.6f}",
        ])
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(TICK_FS)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_facecolor('#f8fafc' if r % 2 == 0 else BG)
        if r == 0:
            cell.set_facecolor('#e0f2fe')
            cell.set_text_props(weight='bold', color=SLATE)
    ax_tbl.set_title('Per-Iteration Metrics', fontsize=TITLE_FS, pad=10)

    fig.suptitle(
        f'syntx.build_template — r16 & r16_reflected  '
        f'({n_iters} iters, gradient_step={args.gradient_step:.2f}, '
        f'blending_weight={args.blending_weight:.2f})',
        fontsize=13, color=SLATE, y=0.986,
    )

    print(f"[demo] Saving PNG → {out_png}")
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)

    # ---- 7. HTML report ----------------------------------------------------
    print(f"[demo] Generating HTML → {out_html}")
    png_basename = os.path.basename(out_png)

    provenance = {
        'script': os.path.basename(__file__),
        'iterations': n_iters,
        'gradient_step': args.gradient_step,
        'blending_weight': args.blending_weight,
        'syn_metric': args.syn_metric,
        'flow_sigma': args.flow_sigma,
        'reg_iterations': reg_iters,
        'elapsed_sec': round(elapsed, 2),
        'convergence': [round(v, 8) for v in convergence],
        'shape_residuals': [
            {k: round(v, 8) for k, v in sr.items()} for sr in shape_res
        ],
    }

    table_rows = ''
    for i, (mae, sr) in enumerate(zip(convergence, shape_res)):
        bg = '#f8fafc' if i % 2 == 0 else '#ffffff'
        table_rows += (
            f'<tr style="background:{bg}">'
            f'<td>{i+1}</td>'
            f'<td>{mae:.8f}</td>'
            f'<td>{sr["l2_norm"]:.6f}</td>'
            f'<td>{sr["membrane_energy"]:.6f}</td>'
            f'<td>{sr["bending_energy"]:.8f}</td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>syntx.build_template — r16 Demo Report</title>
<style>
  body {{font-family:system-ui,sans-serif;background:#fff;color:{SLATE};margin:0;padding:24px}}
  h1   {{font-size:1.6rem;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
  h2   {{font-size:1.1rem;color:#0369a1;margin-top:32px}}
  .fig {{text-align:center;margin:20px 0}}
  .fig img {{max-width:100%;border:1px solid #e2e8f0;border-radius:6px}}
  table {{border-collapse:collapse;width:100%;font-size:0.88rem}}
  th    {{background:#e0f2fe;padding:8px 12px;text-align:center}}
  td    {{padding:6px 12px;text-align:center;border-top:1px solid #e2e8f0}}
  pre   {{background:#f1f5f9;padding:12px;border-radius:6px;font-size:0.8rem;overflow:auto}}
</style>
</head>
<body>
<h1>syntx.build_template — r16 &amp; r16_reflected Demo</h1>

<h2>Configuration</h2>
<table style="width:auto">
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Iterations</td><td>{n_iters}</td></tr>
<tr><td>gradient_step</td><td>{args.gradient_step}</td></tr>
<tr><td>blending_weight</td><td>{args.blending_weight} — ANTsPy sharpen-blend convention</td></tr>
<tr><td>type_of_transform</td><td>SyN</td></tr>
<tr><td>syn_metric</td><td>{args.syn_metric}</td></tr>
<tr><td>flow_sigma</td><td>{args.flow_sigma}</td></tr>
<tr><td>reg_iterations per level</td><td>{reg_iters}</td></tr>
<tr><td>Total elapsed</td><td>{elapsed:.1f}s</td></tr>
</table>

<h2>Visualization</h2>
<div class="fig">
  <img src="{png_basename}" alt="build_template r16 demo">
</div>

<h2>Per-Iteration Metrics</h2>
<table>
<tr>
  <th>Iteration</th><th>Template MAE</th>
  <th>Mean Warp L2 (mm)</th><th>Membrane Energy</th><th>Bending Energy</th>
</tr>
{table_rows}
</table>

<h2>Algorithm Notes</h2>
<ul>
<li><strong>Visualization:</strong> All image display uses
    <code>syntx.viz.extract_oriented_slice</code>,
    <code>plot_edge_overlay</code>, and <code>plot_deformation_grid</code>
    — physical-space correct, radiological convention, no ad-hoc numpy transposition.</li>
<li><strong>Blending (ANTsPy convention):</strong>
    <code>T = blending_weight · T_shape + (1−blending_weight) · iMath(T_shape,"Sharpen")</code>.
    The complement is a sharpened version of the new template, not the old one.</li>
<li><strong>Shape residual:</strong> L2 norm (RMS displacement magnitude in mm),
    membrane energy (Frobenius norm of Jacobian of ψ̄), and bending energy
    (Frobenius norm of Hessian of ψ̄) of the weighted-mean warp field ψ̄.</li>
<li><strong>Single interpolation (GEMINI.md §1):</strong> No pre-warping to disk.
    Warped images come from a single <code>ants.apply_transforms</code> call.</li>
<li><strong>reflect_image:</strong> Full ITK physical-space reflection —
    direction column negation + origin shift to preserve bounding box.</li>
</ul>

<h2>Provenance JSON</h2>
<pre>{json.dumps(provenance, indent=2)}</pre>
</body>
</html>
"""

    with open(out_html, 'w') as f:
        f.write(html)

    print(f"\n✓  Report:  {out_html}")
    print(f"✓  Figure:  {out_png}")
    print(f"✓  Elapsed: {elapsed:.1f}s  ({n_iters} iterations)")


if __name__ == '__main__':
    main()
