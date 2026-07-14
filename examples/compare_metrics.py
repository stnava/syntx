import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import ants
import pandas as pd
import base64
import matplotlib.pyplot as plt
from skimage import feature
from matplotlib.colors import ListedColormap

# Add src to sys.path to allow importing syntx locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from syntx import registration, SyNTo
import antstorch


# =====================================================================
# Diagnostics & Plotting Helpers
# =====================================================================

def get_lai_slice(image, axis, slice_index):
    """
    Reorients an ANTs image to LAI space and extracts a 2D slice,
    correctly transposing spatial dimensions for matplotlib while leaving
    multivariate component axes alone.
    """
    image_lai = image.reorient_image2("LAI")
    img_arr = image_lai.numpy()
    img_arr = np.rollaxis(img_arr, axis)
    
    if isinstance(slice_index, float) and slice_index < 1.0:
        slice_idx = int(slice_index * img_arr.shape[0])
    else:
        slice_idx = int(slice_index)
        
    imslice = img_arr[slice_idx]
    
    def mirror_matrix(x):
        return x[::-1, ...]
        
    def rotate270_matrix(x):
        if x.ndim == 3:
            return mirror_matrix(x.transpose(1, 0, 2))
        return mirror_matrix(x.T)
        
    def rotate90_matrix(x):
        if x.ndim == 3:
            return x.transpose(1, 0, 2)
        return x.T
        
    if axis != 2:
        imslice = rotate90_matrix(imslice)
    else:
        imslice = rotate270_matrix(imslice)
    imslice = mirror_matrix(imslice)
    return imslice


def save_label_overlay_plot(fixed_t1_slice, warped_dkt_slice, filename):
    """Saves a transparent label segmentation overlay over the grayscale target T1w image."""
    np.random.seed(42)
    colors = np.random.rand(256, 3)
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#08080a')
    ax.imshow(fixed_t1_slice, cmap='gray')
    
    masked_dkt = np.ma.masked_where(warped_dkt_slice == 0, warped_dkt_slice)
    ax.imshow(masked_dkt, cmap=cmap, alpha=0.4, interpolation='nearest')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def save_edge_overlay_plot(fixed_t1_slice, warped_t1_slice, filename):
    """Saves Canny edges of the warped moving T1w image overlaid as red contours on the fixed T1w."""
    norm_warped = (warped_t1_slice - warped_t1_slice.min()) / (warped_t1_slice.max() - warped_t1_slice.min() + 1e-8)
    edges = feature.canny(norm_warped, sigma=1.0)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#08080a')
    ax.imshow(fixed_t1_slice, cmap='gray')
    ax.contour(edges, colors='#ff7675', linewidths=0.6, alpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def save_warped_grid_overlay(t1_slice, displ_slice, spacing, filename, direction=None):
    """Saves a continuous 2D warped coordinate grid overlaid on the grayscale target T1w."""
    H_lai, W_lai = t1_slice.shape
    
    dx = displ_slice[:, :, 0]
    dy = displ_slice[:, :, 1]
    
    dx_vox = dx / spacing[0]
    dy_vox = dy / spacing[1]
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#08080a')
    ax.imshow(t1_slice, cmap='gray')
    
    grid_step = 15
    for x_c in range(0, W_lai, grid_step):
        y_points = np.arange(H_lai)
        x_points = x_c + dx_vox[:, x_c]
        y_points_warped = y_points + dy_vox[:, x_c]
        ax.plot(x_points, y_points_warped, color='#55efc4', linewidth=0.8, alpha=0.8)
        
    for y_c in range(0, H_lai, grid_step):
        x_points = np.arange(W_lai)
        x_points_warped = x_points + dx_vox[y_c, :]
        y_points = y_c + dy_vox[y_c, :]
        ax.plot(x_points_warped, y_points, color='#55efc4', linewidth=0.8, alpha=0.8)
        
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def save_jacobian_slice(t1_slice, jac_slice, filename):
    """Saves a colorbar-enabled Jacobian determinant overlay over the T1w image."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#08080a')
    ax.imshow(t1_slice, cmap='gray')
    
    masked_jac = np.ma.masked_where(np.abs(jac_slice - 1.0) < 0.03, jac_slice)
    im = ax.imshow(masked_jac, cmap='bwr', alpha=0.6, vmin=0.5, vmax=1.5)
    ax.axis('off')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Jacobian Determinant', color='white')
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


# =====================================================================
# Registration metrics & Logic
# =====================================================================

def compute_dice(mask1, mask2):
    m1 = (mask1 > 0).astype(float)
    m2 = (mask2 > 0).astype(float)
    intersection = np.sum(m1 * m2)
    vol1 = np.sum(m1)
    vol2 = np.sum(m2)
    if vol1 + vol2 == 0:
        return 0.0
    return 2.0 * intersection / (vol1 + vol2)


def compute_multiregion_dice(dkt1, dkt2):
    labels = np.unique(dkt1.numpy())
    labels = labels[labels > 0]
    
    dices = {}
    for label in labels:
        m1 = (dkt1.numpy() == label).astype(np.float32)
        m2 = (dkt2.numpy() == label).astype(np.float32)
        dices[label] = compute_dice(m1, m2)
        
    return dices


def main():
    parser = argparse.ArgumentParser(
        description="Compare SyN metric parameters (Mattes MI, LNCC, LNCC + VGG) on T1w scans with DKT parcellation target."
    )
    parser.add_argument("-f", "--fixed", type=str, default="~/.antspyt1w/28523-00000000-T1w-05.nii.gz")
    parser.add_argument("-m", "--moving", type=str, default="~/.antspyt1w/28497-00000000-T1w-04.nii.gz")
    parser.add_argument("-c", "--cache-dir", type=str, default="cache", help="Cache directory for brain extraction and DKT segmentations")
    parser.add_argument("-o", "--output-dir", type=str, default="outputs_comparison", help="Output directory for visual results")
    parser.add_argument("-r", "--report-name", type=str, default="metrics_comparison_report.html", help="HTML report output file name")
    
    parser.add_argument("--epochs-per-level", type=int, nargs='+', default=[30, 20, 10], help="Epochs per level for SyN deformable model")
    parser.add_argument("--affine-epochs", type=int, nargs='+', default=[30, 20, 10], help="Epochs per level for Affine model")
    parser.add_argument("--levels", type=int, nargs='+', default=[8, 4, 2], help="Resolution levels (excluding level 1 to save memory/time)")
    parser.add_argument("--device", type=str, default="mps", help="Training device (mps or cpu)")
    
    args = parser.parse_args()

    args.fixed = os.path.expanduser(args.fixed)
    args.moving = os.path.expanduser(args.moving)
    
    print("=== Starting Quantitative Registration Metric Comparison ===")
    print(f"Fixed:  {args.fixed}")
    print(f"Moving: {args.moving}")

    # Set up directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Caching brain extraction & DKT parcellations
    base1 = os.path.basename(args.fixed).replace('.nii.gz', '')
    base2 = os.path.basename(args.moving).replace('.nii.gz', '')
    
    img1_brain_path = os.path.join(args.cache_dir, f"{base1}_brain.nii.gz")
    img2_brain_path = os.path.join(args.cache_dir, f"{base2}_brain.nii.gz")
    dktseg1_path = os.path.join(args.cache_dir, f"{base1}_dktseg.nii.gz")
    dktseg2_path = os.path.join(args.cache_dir, f"{base2}_dktseg.nii.gz")

    if (os.path.exists(img1_brain_path) and os.path.exists(img2_brain_path) and
        os.path.exists(dktseg1_path) and os.path.exists(dktseg2_path)):
        print("\n--- Loading Preprocessed Data from Cache ---")
        img1_brain = ants.image_read(img1_brain_path)
        img2_brain = ants.image_read(img2_brain_path)
        dktseg1 = ants.image_read(dktseg1_path)
        dktseg2 = ants.image_read(dktseg2_path)
    else:
        print("\n--- Cache Miss: Running Deep Learning Preprocessing (Brain Extraction & DKT Labeling) ---")
        img1 = ants.image_read(args.fixed)
        img2 = ants.image_read(args.moving)
        
        # Brain extraction
        print("Extracting brain from Image 1...")
        bxt1 = antstorch.brain_extraction(img1, 't1', verbose=True).threshold_image(0.5, 1.0)
        img1_brain = img1 * bxt1
        ants.image_write(img1_brain, img1_brain_path)
        
        print("Extracting brain from Image 2...")
        bxt2 = antstorch.brain_extraction(img2, 't1', verbose=True).threshold_image(0.5, 1.0)
        img2_brain = img2 * bxt2
        ants.image_write(img2_brain, img2_brain_path)
        
        # DKT segmentations
        print("DKT labeling on Image 1...")
        dktseg1 = antstorch.desikan_killiany_tourville_labeling(img1, do_preprocessing=True, verbose=True)
        ants.image_write(dktseg1, dktseg1_path)
        
        print("DKT labeling on Image 2...")
        dktseg2 = antstorch.desikan_killiany_tourville_labeling(img2, do_preprocessing=True, verbose=True)
        ants.image_write(dktseg2, dktseg2_path)

    # Re-read raw fixed image for report visuals
    fixed_raw = ants.image_read(args.fixed)
    fixed_lai = fixed_raw.reorient_image2("LAI")
    z_mid = int(fixed_lai.shape[2] * 0.55)
    fixed_t1_slice = get_lai_slice(fixed_raw, 2, z_mid)
    spacing_info = fixed_lai.spacing

    # Evaluate Baseline (unregistered) DKT overlap
    print("\n--- Evaluating Baseline DKT Overlap ---")
    base_dices = compute_multiregion_dice(dktseg1, dktseg2)
    base_mean = np.mean(list(base_dices.values()))
    print(f"Baseline Mean DICE: {base_mean:.4f}")

    # Set up experimental configurations
    configs = {
        'mattes_mi': {
            'syn_metric': 'mattes_mi',
            'syn_sampling': 32,
            'kwargs': {}
        },
        'lncc_5x5x5': {
            'syn_metric': 'lncc',
            'syn_sampling': 2,
            'kwargs': {}
        },
        'vgg_lncc_3d': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc_3d'}
        },
        'vgg_lncc': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'lncc'}
        },
        'vgg_patch_walk': {
            'syn_metric': 'vgg19',
            'syn_sampling': 4,
            'kwargs': {'vgg_mode': 'patch_walk'}
        }
    }

    results = {}

    for name, conf in configs.items():
        print(f"\n=========================================")
        print(f" Running Registration Configuration: {name}")
        print(f"=========================================")
        
        start_time = time.time()
        res = registration(
            fixed=img1_brain,
            moving=img2_brain,
            type_of_transform='SyNTo',
            backend='pytorch',
            syn_metric=conf['syn_metric'],
            syn_sampling=conf['syn_sampling'],
            levels=args.levels,
            affine_iterations=args.affine_epochs,
            reg_iterations=args.epochs_per_level,
            verbose=True,
            **conf['kwargs']
        )
        runtime = time.time() - start_time
        print(f"Finished in {runtime:.2f} seconds.")

        # Re-load or evaluate warped outputs
        # Warping DKT label map
        print("Warping DKT label map...")
        warped_dktseg2 = ants.apply_transforms(
            fixed=img1_brain,
            moving=dktseg2,
            transformlist=res['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        dices = compute_multiregion_dice(dktseg1, warped_dktseg2)
        mean_dice = np.mean(list(dices.values()))
        print(f"Mean DKT DICE Overlap: {mean_dice:.4f}")

        # Compute warped T1w moving image
        warped_moving = ants.apply_transforms(
            fixed=img1_brain,
            moving=img2_brain,
            transformlist=res['fwdtransforms']
        )

        # Compute Jacobian folding rate
        composite_warp_path = os.path.join(args.output_dir, f"{name}_CompositeWarp.nii.gz")
        
        # In order to compute Jacobian, we need to extract the PyTorch transform object
        # Wait, the registration helper returns 'fwdtransforms' which contains paths.
        # We can construct the SyNToTransform and compute it.
        # Actually, let's read the composite warp field back into ants and compute folding rate
        ants.image_write(ants.image_read(res['fwdtransforms'][0]), composite_warp_path)
        comp_warp_img = ants.image_read(composite_warp_path)
        
        # Or compute folding rate from the Jacobian image
        jac_img = ants.create_jacobian_determinant_image(img1_brain, composite_warp_path, do_log=False)
        jac_np = jac_img.numpy()
        folds = np.sum(jac_np <= 0) / jac_np.size * 100
        print(f"Folding Rate (J <= 0): {folds:.4f}%")

        # Generate visual slices
        lbl_png = os.path.join(args.output_dir, f"{name}_lbl.png")
        edge_png = os.path.join(args.output_dir, f"{name}_edge.png")
        grid_png = os.path.join(args.output_dir, f"{name}_grid.png")
        jac_png = os.path.join(args.output_dir, f"{name}_jacobian.png")

        # Extracts
        warped_dkt_slice = get_lai_slice(warped_dktseg2, 2, z_mid)
        warped_t1_slice = get_lai_slice(warped_moving, 2, z_mid)
        displ_slice = get_lai_slice(comp_warp_img, 2, z_mid)
        jac_slice = get_lai_slice(jac_img, 2, z_mid)

        # Plotting
        save_label_overlay_plot(fixed_t1_slice, warped_dkt_slice, lbl_png)
        save_edge_overlay_plot(fixed_t1_slice, warped_t1_slice, edge_png)
        save_warped_grid_overlay(fixed_t1_slice, displ_slice, spacing_info, grid_png)
        save_jacobian_slice(fixed_t1_slice, jac_slice, jac_png)

        results[name] = {
            'mean_dice': mean_dice,
            'dices': dices,
            'folds': folds,
            'runtime': runtime,
            'affine_losses': res.get('affine_losses', []),
            'syn_losses': res.get('syn_losses', []),
            'lbl_png': lbl_png,
            'edge_png': edge_png,
            'grid_png': grid_png,
            'jac_png': jac_png
        }

    # Generate HTML report
    report_path = os.path.join(args.output_dir, args.report_name)
    generate_comparison_report(base_dices, results, report_path)


def generate_comparison_report(base_dices, results, output_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 1. Base64 helper
    def get_b64(path):
        if not path or not os.path.exists(path): return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    # 2. Plotly Grouped DICE Barplot
    labels = sorted(list(base_dices.keys()))
    fig_data = [
        go.Bar(name='Baseline', x=[f"Region {int(l)}" for l in labels], y=[base_dices[l] for l in labels], marker_color='#7f8c8d')
    ]
    
    colors = ['#ff7675', '#55efc4', '#74b9ff', '#ffeaa7', '#a29bfe']
    for idx, (name, res) in enumerate(results.items()):
        fig_data.append(
            go.Bar(
                name=name.upper().replace('_', ' '),
                x=[f"Region {int(l)}" for l in labels],
                y=[res['dices'].get(l, 0.0) for l in labels],
                marker_color=colors[idx % len(colors)]
            )
        )
        
    fig_dice = go.Figure(data=fig_data)
    fig_dice.update_layout(
        title=dict(text='DKT Cortical Region DICE Overlap Comparison', font=dict(size=18, color='#f1f2f6')),
        xaxis=dict(tickangle=45, tickfont=dict(color='#a4b0be'), gridcolor='#2f3542'),
        yaxis=dict(title='DICE Score', tickfont=dict(color='#a4b0be'), gridcolor='#2f3542', range=[0, 1]),
        barmode='group',
        paper_bgcolor='#1e1e24',
        plot_bgcolor='#1e1e24',
        legend=dict(font=dict(color='#f1f2f6')),
        height=500,
        margin=dict(l=50, r=30, t=60, b=100)
    )
    barplot_html = fig_dice.to_html(full_html=False, include_plotlyjs='cdn')

    # 3. Training Loss Curves
    fig_loss = make_subplots(rows=len(results), cols=2, subplot_titles=[
        f"{name.upper()} Affine Loss" if i == 0 else f"{name.upper()} SyN Loss"
        for name in results.keys() for i in range(2)
    ])
    
    for row_idx, (name, res) in enumerate(results.items()):
        fig_loss.add_trace(go.Scatter(y=res['affine_losses'], mode='lines', name=f'{name} Affine', line=dict(color='#55efc4')), row=row_idx+1, col=1)
        fig_loss.add_trace(go.Scatter(y=res['syn_losses'], mode='lines', name=f'{name} SyN', line=dict(color='#00b894')), row=row_idx+1, col=2)
        
    fig_loss.update_layout(
        title=dict(text='Optimizer Loss Convergence History', font=dict(size=18, color='#f1f2f6')),
        paper_bgcolor='#1e1e24',
        plot_bgcolor='#1e1e24',
        showlegend=False,
        height=250 * len(results),
        margin=dict(l=50, r=30, t=60, b=50)
    )
    for annotation in fig_loss['layout']['annotations']:
        annotation['font'] = dict(color='#f1f2f6', size=12)
    fig_loss.update_xaxes(tickfont=dict(color='#a4b0be'), gridcolor='#2f3542')
    fig_loss.update_yaxes(tickfont=dict(color='#a4b0be'), gridcolor='#2f3542')
    loss_html = fig_loss.to_html(full_html=False, include_plotlyjs='cdn')

    # 4. Building Summary Table & Stat Cards
    table_rows = []
    summary_items = [
        f"""<div class="summary-stat">
            <div class="summary-value">{np.mean(list(base_dices.values())):.4f}</div>
            <div class="summary-label">Baseline DICE</div>
        </div>"""
    ]
    
    for idx, (name, res) in enumerate(results.items()):
        table_rows.append(f"""
        <tr>
            <td style="color: {colors[idx]}">{name.upper()}</td>
            <td>{res['mean_dice']:.4f}</td>
            <td>{res['folds']:.4f}%</td>
            <td>{res['runtime']:.2f}s</td>
        </tr>
        """)
        summary_items.append(f"""
        <div class="summary-stat">
            <div class="summary-value" style="color: {colors[idx]}">{res['mean_dice']:.4f}</div>
            <div class="summary-label">{name} DICE</div>
        </div>
        """)
        
    summary_table_html = f"""
    <table class="styled-table">
        <thead>
            <tr>
                <th>Registration Metric</th>
                <th>Mean DICE Score</th>
                <th>Jacobian Folding Rate (J &le; 0)</th>
                <th>Runtime</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
    """

    # 5. Diagnostic grid overlays
    diag_sections = []
    for name, res in results.items():
        diag_sections.append(f"""
        <h2>Visual Alignment: {name.upper().replace('_', ' ')}</h2>
        <div class="img-grid">
            <div class="img-box">
                <h3>Warped DKT Labels</h3>
                <img src="data:image/png;base64,{get_b64(res['lbl_png'])}">
            </div>
            <div class="img-box">
                <h3>Warped Canny Edges</h3>
                <img src="data:image/png;base64,{get_b64(res['edge_png'])}">
            </div>
            <div class="img-box">
                <h3>Deformed Gaussian Grid</h3>
                <img src="data:image/png;base64,{res['grid_png']}">
            </div>
            <div class="img-box">
                <h3>Jacobian Determinant Map</h3>
                <img src="data:image/png;base64,{res['jac_png']}">
            </div>
        </div>
        """)

    # 6. HTML Template Assembly
    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Registration Metrics Quantitative Report</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; max-width: 1450px; margin: 0 auto; padding: 40px; background: #0f0f12; color: #f1f2f6; }}
            h1 {{ font-size: 32px; background: -webkit-linear-gradient(#55efc4, #00b894); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }}
            p.subtitle {{ color: #a4b0be; font-size: 15px; margin-top: 0; margin-bottom: 40px; }}
            h2 {{ color: #ffffff; border-bottom: 1px solid #2f3542; padding-bottom: 8px; margin-top: 50px; font-weight: 500; }}
            .img-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 45px; }}
            .img-box {{ background: #1e1e24; padding: 10px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid #2f3542; text-align: center; }}
            .img-box img {{ width: 100%; border-radius: 6px; }}
            .img-box h3 {{ color: #ffffff; font-size: 13px; margin-top: 0; margin-bottom: 10px; font-weight: 400; }}
            .chart-container {{ background: #1e1e24; padding: 25px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid #2f3542; margin-bottom: 40px; }}
            .summary-card {{ background: #1e1e24; padding: 20px; border-radius: 12px; border: 1px solid #2f3542; margin-bottom: 40px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; }}
            .summary-stat {{ text-align: center; flex: 1; min-width: 120px; margin: 10px; }}
            .summary-value {{ font-size: 24px; font-weight: bold; color: #55efc4; }}
            .summary-label {{ font-size: 11px; color: #a4b0be; text-transform: uppercase; margin-top: 5px; }}
            .styled-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 15px; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid #2f3542; }}
            .styled-table th, .styled-table td {{ padding: 12px 15px; text-align: left; }}
            .styled-table thead tr {{ background-color: #1e1e24; color: #ffffff; border-bottom: 2px solid #2f3542; font-weight: bold; }}
            .styled-table tbody tr {{ border-bottom: 1px solid #2f3542; background: #151518; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #1e1e24; }}
            .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #55efc4; }}
        </style>
    </head>
    <body>
        <h1>Registration Metrics Quantitative Performance Report</h1>
        <p class="subtitle">Documents alignment metrics, regional overlap, warp smoothness, and topology across Mattes MI, LNCC, and LNCC+VGG configurations.</p>
        
        <h2>Performance Summary</h2>
        <div class="summary-card">
            {"".join(summary_items)}
        </div>
        
        {summary_table_html}

        <h2>Regional Overlap Over DKT Cortical Subdivisions</h2>
        <div class="chart-container">{barplot_html}</div>
        
        {"".join(diag_sections)}
        
        <h2>Optimizer Convergence History</h2>
        <div class="chart-container">
            {loss_html}
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Generated HTML report at {output_path}")


if __name__ == "__main__":
    main()
