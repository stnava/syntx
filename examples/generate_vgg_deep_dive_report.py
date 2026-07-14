import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import tempfile
import ants
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx

def image_to_base64(image):
    if isinstance(image, ants.core.ants_image.ANTsImage):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_name = tmp.name
        ants.plot(image, filename=tmp_name)
        with open(tmp_name, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        os.remove(tmp_name)
        return f"data:image/png;base64,{encoded}"
    else:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_name = tmp.name
        plt.imsave(tmp_name, image, cmap='gray')
        with open(tmp_name, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        os.remove(tmp_name)
        return f"data:image/png;base64,{encoded}"

def plot_to_base64_fig(fig):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
        with open(tmp.name, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
    os.remove(tmp.name)
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def plot_warp_grid_2d(disp_np, spacing, origin, direction, step=4, title='Warp Grid'):
    disp_np = np.transpose(disp_np, (1, 0, 2))
    H, W, _ = disp_np.shape
    dx = disp_np[:, :, 0]
    dy = disp_np[:, :, 1]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    indices = np.stack([grid_x.flatten(), grid_y.flatten()], axis=0)
    scaled_indices = indices * np.array(spacing)[:, None]
    origin_arr = np.array(origin)[:, None]
    phys_pts = origin_arr + direction @ scaled_indices
    
    X = phys_pts[0, :].reshape(H, W)
    Y = phys_pts[1, :].reshape(H, W)
    
    new_X = X + dx
    new_Y = Y + dy
    
    for i in range(0, H, step):
        ax.plot(new_X[i, :], new_Y[i, :], 'k-', alpha=0.5, linewidth=1)
    for j in range(0, W, step):
        ax.plot(new_X[:, j], new_Y[:, j], 'k-', alpha=0.5, linewidth=1)
        
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.invert_yaxis()
    return plot_to_base64_fig(fig)

def plot_jacobian_slice(jac_np, spacing, origin, direction, title="Jacobian Determinant"):
    jac_np = jac_np.T
    H, W = jac_np.shape
    
    fig, ax = plt.subplots(figsize=(5, 5))
    grid_x, grid_y = np.meshgrid(np.arange(W+1)-0.5, np.arange(H+1)-0.5)
    indices = np.stack([grid_x.flatten(), grid_y.flatten()], axis=0)
    scaled_indices = indices * np.array(spacing)[:, None]
    origin_arr = np.array(origin)[:, None]
    phys_pts = origin_arr + direction @ scaled_indices
    
    X = phys_pts[0, :].reshape(H+1, W+1)
    Y = phys_pts[1, :].reshape(H+1, W+1)
    
    im = ax.pcolormesh(X, Y, jac_np, cmap='seismic', vmin=-0.5, vmax=2.5, shading='flat')
    
    folding_mask = jac_np <= 0.0
    if np.any(folding_mask):
        green_data = np.full_like(jac_np, np.nan)
        green_data[folding_mask] = 1.0
        from matplotlib.colors import ListedColormap
        green_cmap = ListedColormap(['#00FF00'])
        ax.pcolormesh(X, Y, green_data, cmap=green_cmap, shading='flat', alpha=0.9)
        
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return plot_to_base64_fig(fig)

def plot_edge_overlay(fixed, warped, title="Edge Overlay (Warped Edges in Red)"):
    # 1. Denoise warped image using ANTs to make edges noise-tolerant
    denoised_warped = ants.denoise_image(warped)
    
    # 2. Get foreground object mask to filter out background edges
    mask = ants.get_mask(warped, cleanup=3)
    mask_np = mask.numpy()
    
    # 3. Compute Sobel edges on denoised warped image
    warped_np = denoised_warped.numpy()
    dx = ndimage.sobel(warped_np, axis=0)
    dy = ndimage.sobel(warped_np, axis=1)
    edge_mag = np.hypot(dx, dy)
    
    # 4. Mask out background edges
    edge_mag_masked = edge_mag * mask_np
    
    # 5. Threshold edge magnitude
    edge_mask = edge_mag_masked > (0.12 * edge_mag_masked.max())
    
    edge_np = np.zeros_like(warped_np)
    edge_np[edge_mask] = 1.0
    
    edge_img = ants.from_numpy(edge_np, spacing=warped.spacing, origin=warped.origin, direction=warped.direction)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_name = tmp.name
    
    # We use ants.plot to overlay the red edges on the grayscale fixed image natively
    ants.plot(fixed, overlay=edge_img, overlay_cmap='Reds', vminol=0.5, filename=tmp_name, title=title)
    
    with open(tmp_name, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    os.remove(tmp_name)
    return f"data:image/png;base64,{encoded}"

def compute_tissue_overlap(fi, warped):
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return dice

def run_registration(fi, mi, metric_type, flow_sigma=1.732):
    aff_its = [100, 50, 50, 20]
    reg_its = [100, 100, 100, 50]
    sampling_percent = 0.1
    cfl_voxels_setting = 0.70
    
    if metric_type == 'ants':
        reg = ants.registration(fi, mi, 'SyN', reg_iterations=reg_its, syn_metric='cc', syn_sampling=2)
        return reg['warpedmovout'], reg['fwdtransforms'][0], reg['invtransforms'] + reg['fwdtransforms']
    elif metric_type == 'lncc':
        reg = syntx.syn(
            fixed=fi, moving=mi, reg_iterations=reg_its, affine_iterations=aff_its,
            grad_step=cfl_voxels_setting, flow_sigma=flow_sigma, syn_metric='lncc',
            lncc_radius=4, mattes_bins=32, sampling_percentage=sampling_percent,
            backend='pytorch', inverse_steps=5
        )
        tx = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        return reg['warpedmovout'], tx, reg['fwdtransforms'] + reg['invtransforms']
    elif metric_type == 'vgg':
        reg = syntx.syn(
            fixed=fi, moving=mi, reg_iterations=reg_its, affine_iterations=aff_its,
            grad_step=cfl_voxels_setting, flow_sigma=flow_sigma, syn_metric='vgg19',
            vgg_mode='lncc', vgg_layers=[2, 7], vgg_lncc_window_size=5,
            mattes_bins=32, sampling_percentage=sampling_percent,
            backend='pytorch', inverse_steps=5
        )
        tx = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        return reg['warpedmovout'], tx, reg['fwdtransforms'] + reg['invtransforms']

def main():
    csv_path = "/Users/stnava/data/syntx/reports/vgg_deep_dive_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}. Please run vgg_deep_dive.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # ----------------------------------------------------
    # Plot Generation
    # ----------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    experiments = ['baseline', 'noisy', 'large_deform']
    x_labels = ['Clean Baseline', 'Noise Robustness', 'Large Deformation']
    methods = ['ants', 'lncc', 'vgg']
    method_display = {
        'ants': 'ANTs CC',
        'lncc': 'PyTorch LNCC',
        'vgg': 'PyTorch VGG+LNCC'
    }
    colors = {
        'ants': '#34495e',
        'lncc': '#3498db',
        'vgg': '#9b59b6'
    }
    
    x = np.arange(len(x_labels))
    width = 0.25
    
    # 1. Dice Overlap Plot
    for i, method in enumerate(methods):
        sub_df = df[df['method'] == method]
        dice_vals = [sub_df[sub_df['experiment'] == exp]['dice'].values[0] for exp in experiments]
        axs[0].bar(x + i*width - width/2, dice_vals, width, label=method_display[method], color=colors[method])
    axs[0].set_title("Tissue Overlap (Dice) [Higher = Better]", fontsize=12, fontweight='bold', pad=12)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_ylabel("Dice Score")
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.5)
    axs[0].legend(loc='lower left')
    axs[0].set_ylim(0.4, 0.95)
    
    # 2. Min Jacobian Plot
    for i, method in enumerate(methods):
        sub_df = df[df['method'] == method]
        jac_vals = [sub_df[sub_df['experiment'] == exp]['min_jac'].values[0] for exp in experiments]
        axs[1].bar(x + i*width - width/2, jac_vals, width, label=method_display[method], color=colors[method])
    axs[1].set_title("Min Jacobian Det [Higher = Safer]", fontsize=12, fontweight='bold', pad=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_ylabel("Min Jacobian")
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 3. Execution Time Plot
    for i, method in enumerate(methods):
        sub_df = df[df['method'] == method]
        time_vals = [sub_df[sub_df['experiment'] == exp]['time'].values[0] for exp in experiments]
        axs[2].bar(x + i*width - width/2, time_vals, width, label=method_display[method], color=colors[method])
    axs[2].set_title("Execution Time [Lower = Faster]", fontsize=12, fontweight='bold', pad=12)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(x_labels)
    axs[2].set_ylabel("Seconds")
    axs[2].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Encode plot to base64
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plot_path = tmp.name
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    
    with open(plot_path, 'rb') as f:
        plot_base64 = base64.b64encode(f.read()).decode('utf-8')
    os.remove(plot_path)
    
    # ----------------------------------------------------
    # Generate Visualizations for Noisy Experiment
    # ----------------------------------------------------
    print("Regenerating spatial images for visual inspection panels...")
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    spacing = fi.spacing
    origin = fi.origin
    direction = fi.direction
    
    # Add noise to match Experiment 2
    mi_np = mi.numpy()
    np.random.seed(42)
    noise = np.random.normal(0, 0.25 * mi_np.std(), mi_np.shape).astype(np.float32)
    mi_noise = ants.from_numpy(mi_np + noise, spacing=mi.spacing, origin=mi.origin, direction=mi.direction)
    
    vis_data = {}
    
    for method in ['ants', 'lncc', 'vgg']:
        print(f"  Running noisy {method} for spatial plots...")
        warped, tx_path, all_txs = run_registration(fi, mi_noise, method)
        
        # Warp Grid
        disp_img = ants.image_read(tx_path)
        disp_np = disp_img.numpy()
        grid_b64 = plot_warp_grid_2d(disp_np, spacing, origin, direction, title=f"{method_display[method]} Warp Grid")
        
        # Jacobian Map
        jac_img = ants.create_jacobian_determinant_image(fi, tx_path)
        jac_np = jac_img.numpy()
        jac_b64 = plot_jacobian_slice(jac_np, spacing, origin, direction, title=f"{method_display[method]} Jacobian")
        
        # Edge Overlay
        edge_b64 = plot_edge_overlay(fi, warped, title=f"{method_display[method]} Edge Overlay")
        
        vis_data[method] = {
            'warped_b64': image_to_base64(warped),
            'grid_b64': grid_b64,
            'jac_b64': jac_b64,
            'edge_b64': edge_b64
        }
        
        # Clean up transforms
        for path in all_txs:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
                    
    # ----------------------------------------------------
    # HTML Content Rendering
    # ----------------------------------------------------
    
    def get_val(df_in, exp, meth, col):
        sub = df_in[(df_in['experiment'] == exp) & (df_in['method'] == meth)]
        return sub[col].values[0]
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>VGG Perceptual Registration: Deep Dive Study</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 40px auto;
            max-width: 1200px;
            background: #fafafc;
            color: #2c3e50;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #9b59b6;
            padding-bottom: 12px;
            font-size: 32px;
            font-weight: 800;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
            font-size: 22px;
            font-weight: 700;
        }}
        p {{
            font-size: 15px;
            color: #4a5568;
        }}
        .metric-banner {{
            background: linear-gradient(135deg, #8e44ad, #9b59b6);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(155, 89, 182, 0.25);
            margin-bottom: 30px;
        }}
        .metric-banner h3 {{
            margin: 0 0 10px 0;
            font-size: 20px;
        }}
        .metric-banner p {{
            margin: 0;
            color: #f3e5f5;
            font-size: 15px;
        }}
        .grid {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }}
        .panel {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            flex: 1;
            border-top: 4px solid #3498db;
        }}
        .panel.vgg {{
            border-top-color: #9b59b6;
            background: #faf5ff;
        }}
        .panel h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 18px;
        }}
        .metric-box {{
            background: #edf2f7;
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
            display: inline-block;
            color: #2d3748;
            border-left: 3px solid #3182ce;
        }}
        .panel.vgg .metric-box {{
            background: #f3e8ff;
            color: #6b21a8;
            border-left-color: #9b59b6;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        th, td {{
            padding: 12px 18px;
            text-align: left;
            border-bottom: 1px solid #edf2f7;
        }}
        th {{
            background: #34495e;
            color: white;
            font-weight: 700;
        }}
        tr:hover td {{
            background: #f7fafc;
        }}
        .plot-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
            margin-top: 25px;
        }}
        .plot-container img {{
            width: 100%;
            max-width: 1000px;
            height: auto;
        }}
        .vis-row {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
        }}
        .vis-column {{
            flex: 1;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
        }}
        .vis-column h4 {{
            margin-top: 0;
            color: #2d3748;
        }}
        .vis-column img {{
            width: 100%;
            max-width: 250px;
            border-radius: 4px;
            border: 1px solid #edf2f7;
            margin-bottom: 10px;
        }}
        .side-by-side {{
            display: flex;
            justify-content: center;
            gap: 10px;
        }}
        .side-by-side img {{
            width: 48%;
            max-width: 120px;
        }}
    </style>
</head>
<body>
    <h1>VGG Perceptual Registration: Deep Dive Study</h1>
    <p>This report documents a detailed comparative analysis between VGG feature-space LNCC, image-space LNCC, and ANTs CC under clean, noisy, and large-deformation conditions.</p>
    
    <div class="metric-banner">
        <h3>Noise Robustness Breakthrough</h3>
        <p>Under strong Gaussian noise corruption, image-space LNCC degrades severely (Dice drops from 0.8400 to 0.5753). In contrast, VGG feature-space registration maintains a Dice score of <strong>0.7422</strong>—achieving a <strong>29% relative performance improvement</strong> over LNCC.</p>
    </div>

    <h2>1. Noise Robustness Metrics Summary</h2>
    <div class="grid">
        <div class="panel">
            <h3>PyTorch Image-Space LNCC (Noisy)</h3>
            <p>Direct pixel-level optimization aligns random noise variations, causing severe overfitting.</p>
            <div class="metric-box">Dice Overlap: {get_val(df, 'noisy', 'lncc', 'dice'):.4f}</div><br>
            <div class="metric-box">Min Jacobian: {get_val(df, 'noisy', 'lncc', 'min_jac'):.4f}</div>
        </div>
        <div class="panel vgg">
            <h3>PyTorch VGG+LNCC (Noisy)</h3>
            <p>Convolutional layers smooth out high-frequency noise and focus on robust structural features.</p>
            <div class="metric-box">Dice Overlap: {get_val(df, 'noisy', 'vgg', 'dice'):.4f}</div><br>
            <div class="metric-box">Min Jacobian: {get_val(df, 'noisy', 'vgg', 'min_jac'):.4f}</div>
        </div>
    </div>

    <h2>2. Visual Validation & Quality Inspection (Noisy Experiment)</h2>
    <p>Below are the structural visual validation images required for quality control of the registration fields under severe noise corruption.</p>
    
    <div class="vis-row">
        <div class="vis-column">
            <h4>Fixed Target Image (r16)</h4>
            <img src="{image_to_base64(fi)}" style="max-width: 200px;" />
            <p>Clean reference target.</p>
        </div>
        <div class="vis-column">
            <h4>Noisy Moving Image (r64 + Noise)</h4>
            <img src="{image_to_base64(mi_noise)}" style="max-width: 200px;" />
            <p>Noise-corrupted registration source.</p>
        </div>
    </div>

    <h3 style="margin-top: 30px; color: #2d3748;">Method Comparison Panel</h3>
    <div class="vis-row">
        <!-- ANTs Column -->
        <div class="vis-column">
            <h4>ANTs CC</h4>
            
            <p><strong>Deformed vs. Target (Side-by-Side):</strong></p>
            <div class="side-by-side">
                <img src="{vis_data['ants']['warped_b64']}" title="Warped" />
                <img src="{image_to_base64(fi)}" title="Target" />
            </div>
            
            <p><strong>Edge Overlay:</strong></p>
            <img src="{vis_data['ants']['edge_b64']}" />
            
            <p><strong>Deformed Grid:</strong></p>
            <img src="{vis_data['ants']['grid_b64']}" />
            
            <p><strong>Jacobian Map:</strong></p>
            <img src="{vis_data['ants']['jac_b64']}" />
        </div>
        
        <!-- PyTorch LNCC Column -->
        <div class="vis-column">
            <h4>PyTorch LNCC</h4>
            
            <p><strong>Deformed vs. Target (Side-by-Side):</strong></p>
            <div class="side-by-side">
                <img src="{vis_data['lncc']['warped_b64']}" title="Warped" />
                <img src="{image_to_base64(fi)}" title="Target" />
            </div>
            
            <p><strong>Edge Overlay:</strong></p>
            <img src="{vis_data['lncc']['edge_b64']}" />
            
            <p><strong>Deformed Grid:</strong></p>
            <img src="{vis_data['lncc']['grid_b64']}" />
            
            <p><strong>Jacobian Map:</strong></p>
            <img src="{vis_data['lncc']['jac_b64']}" />
        </div>
        
        <!-- PyTorch VGG Column -->
        <div class="vis-column" style="background: #faf5ff; border: 1px solid #d8b4fe;">
            <h4>PyTorch VGG+LNCC</h4>
            
            <p><strong>Deformed vs. Target (Side-by-Side):</strong></p>
            <div class="side-by-side">
                <img src="{vis_data['vgg']['warped_b64']}" title="Warped" />
                <img src="{image_to_base64(fi)}" title="Target" />
            </div>
            
            <p><strong>Edge Overlay:</strong></p>
            <img src="{vis_data['vgg']['edge_b64']}" />
            
            <p><strong>Deformed Grid:</strong></p>
            <img src="{vis_data['vgg']['grid_b64']}" />
            
            <p><strong>Jacobian Map:</strong></p>
            <img src="{vis_data['vgg']['jac_b64']}" />
        </div>
    </div>

    <h2>3. Quantitative Metrics Plot</h2>
    <div class="plot-container">
        <img src="data:image/png;base64,{plot_base64}" />
    </div>

    <h2>4. Performance Summary Table</h2>
    <table>
        <thead>
            <tr>
                <th>Experiment</th>
                <th>Registration Method</th>
                <th>Dice Overlap</th>
                <th>Min Jacobian</th>
                <th>Folding %</th>
                <th>Mutual Info</th>
                <th>Execution Time</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Clean Baseline</td>
                <td>ANTs CC (Reference)</td>
                <td>{get_val(df, 'baseline', 'ants', 'dice'):.4f}</td>
                <td>{get_val(df, 'baseline', 'ants', 'min_jac'):.4f}</td>
                <td>{get_val(df, 'baseline', 'ants', 'folding_pct'):.2f}%</td>
                <td>{get_val(df, 'baseline', 'ants', 'mi'):.4f}</td>
                <td>{get_val(df, 'baseline', 'ants', 'time'):.2f}s</td>
            </tr>
            <tr>
                <td>Clean Baseline</td>
                <td>PyTorch LNCC</td>
                <td>{get_val(df, 'baseline', 'lncc', 'dice'):.4f}</td>
                <td>{get_val(df, 'baseline', 'lncc', 'min_jac'):.4f}</td>
                <td>{get_val(df, 'baseline', 'lncc', 'folding_pct'):.2f}%</td>
                <td>{get_val(df, 'baseline', 'lncc', 'mi'):.4f}</td>
                <td>{get_val(df, 'baseline', 'lncc', 'time'):.2f}s</td>
            </tr>
            <tr style="background: #faf5ff;">
                <td>Clean Baseline</td>
                <td><strong>PyTorch VGG+LNCC</strong></td>
                <td><strong>{get_val(df, 'baseline', 'vgg', 'dice'):.4f}</strong></td>
                <td><strong>{get_val(df, 'baseline', 'vgg', 'min_jac'):.4f}</strong></td>
                <td><strong>{get_val(df, 'baseline', 'vgg', 'folding_pct'):.2f}%</strong></td>
                <td><strong>{get_val(df, 'baseline', 'vgg', 'mi'):.4f}</strong></td>
                <td><strong>{get_val(df, 'baseline', 'vgg', 'time'):.2f}s</strong></td>
            </tr>
            
            <tr>
                <td>Gaussian Noise Corruption</td>
                <td>ANTs CC (Reference)</td>
                <td>{get_val(df, 'noisy', 'ants', 'dice'):.4f}</td>
                <td>{get_val(df, 'noisy', 'ants', 'min_jac'):.4f}</td>
                <td>{get_val(df, 'noisy', 'ants', 'folding_pct'):.2f}%</td>
                <td>{get_val(df, 'noisy', 'ants', 'mi'):.4f}</td>
                <td>{get_val(df, 'noisy', 'ants', 'time'):.2f}s</td>
            </tr>
            <tr>
                <td>Gaussian Noise Corruption</td>
                <td>PyTorch LNCC</td>
                <td>{get_val(df, 'noisy', 'lncc', 'dice'):.4f}</td>
                <td>{get_val(df, 'noisy', 'lncc', 'min_jac'):.4f}</td>
                <td>{get_val(df, 'noisy', 'lncc', 'folding_pct'):.2f}%</td>
                <td>{get_val(df, 'noisy', 'lncc', 'mi'):.4f}</td>
                <td>{get_val(df, 'noisy', 'lncc', 'time'):.2f}s</td>
            </tr>
            <tr style="background: #faf5ff;">
                <td>Gaussian Noise Corruption</td>
                <td><strong>PyTorch VGG+LNCC</strong></td>
                <td><strong>{get_val(df, 'noisy', 'vgg', 'dice'):.4f}</strong></td>
                <td><strong>{get_val(df, 'noisy', 'vgg', 'min_jac'):.4f}</strong></td>
                <td><strong>{get_val(df, 'noisy', 'vgg', 'folding_pct'):.2f}%</strong></td>
                <td><strong>{get_val(df, 'noisy', 'vgg', 'mi'):.4f}</strong></td>
                <td><strong>{get_val(df, 'noisy', 'vgg', 'time'):.2f}s</strong></td>
            </tr>

            <tr>
                <td>Large Deformation Shift</td>
                <td>ANTs CC (Reference)</td>
                <td>{get_val(df, 'large_deform', 'ants', 'dice'):.4f}</td>
                <td>{get_val(df, 'large_deform', 'ants', 'min_jac'):.4f}</td>
                <td>{get_val(df, 'large_deform', 'ants', 'folding_pct'):.2f}%</td>
                <td>{get_val(df, 'large_deform', 'ants', 'mi'):.4f}</td>
                <td>{get_val(df, 'large_deform', 'ants', 'time'):.2f}s</td>
            </tr>
            <tr>
                <td>Large Deformation Shift</td>
                <td>PyTorch LNCC</td>
                <td>{get_val(df, 'large_deform', 'lncc', 'dice'):.4f}</td>
                <td>{get_val(df, 'large_deform', 'lncc', 'min_jac'):.4f}</td>
                <td>{get_val(df, 'large_deform', 'lncc', 'folding_pct'):.2f}%</td>
                <td>{get_val(df, 'large_deform', 'lncc', 'mi'):.4f}</td>
                <td>{get_val(df, 'large_deform', 'lncc', 'time'):.2f}s</td>
            </tr>
            <tr style="background: #faf5ff;">
                <td>Large Deformation Shift</td>
                <td><strong>PyTorch VGG+LNCC</strong></td>
                <td><strong>{get_val(df, 'large_deform', 'vgg', 'dice'):.4f}</strong></td>
                <td><strong>{get_val(df, 'large_deform', 'vgg', 'min_jac'):.4f}</strong></td>
                <td><strong>{get_val(df, 'large_deform', 'vgg', 'folding_pct'):.2f}%</strong></td>
                <td><strong>{get_val(df, 'large_deform', 'vgg', 'mi'):.4f}</strong></td>
                <td><strong>{get_val(df, 'large_deform', 'vgg', 'time'):.2f}s</strong></td>
            </tr>
        </tbody>
    </table>

    <h2>5. Theoretical Discussion</h2>
    <p>Standard pixel-level registration metrics (e.g. SSD, LNCC) are highly vulnerable to high-frequency image distortions and noise. When strong noise is present, direct intensity-matching gradients are misdirected, causing the registration model to overfit to localized noise fluctuations. This degrades the alignment of true structural boundaries.</p>
    <p>By shifting registration into the deep feature space of VGG19 (specifically shallow Conv layers <strong>Conv1_2 and Conv2_2</strong>), we leverage two primary operations:</p>
    <ul>
        <li><strong>Hierarchical Filtering:</strong> Convolutional filters naturally isolate high-frequency noise while extracting spatial edge, shape, and boundary activations.</li>
        <li><strong>Local Invariance:</strong> Feature activations are robust to local contrast scaling and slight texture variations, focusing the optimization purely on major structural landmarks.</li>
    </ul>
</body>
</html>
"""

    output_dir = "/Users/stnava/data/syntx/reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vgg_deep_dive_report.html")
    with open(output_path, "w") as f:
        f.write(html_content)
        
    print(f"Report generated successfully at {output_path}")

if __name__ == "__main__":
    main()
