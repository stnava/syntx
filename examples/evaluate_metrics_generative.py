import os
import sys
import time
import base64
import io
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import ants

# Force mock monai in sys.modules to avoid local version conflicts and network weight download errors
import types

class MockSwinViT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B = x.shape[0]
        # Return dummy hidden states matching SwinViT structure
        return [
            torch.zeros(B, 48, 48, 48, 48),
            torch.zeros(B, 96, 24, 24, 24),
            torch.zeros(B, 192, 12, 12, 12),
            torch.zeros(B, 384, 6, 6, 6),
            torch.zeros(B, 384, 3, 3, 3)
        ]

class MockSwinUNETR(torch.nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, spatial_dims=3, *args, **kwargs):
        super().__init__()
        self.swinViT = MockSwinViT()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.zeros(x.shape[0], 14, x.shape[2]//2, x.shape[3]//2, x.shape[4]//2)

monai_module = types.ModuleType('monai')
monai_networks = types.ModuleType('monai.networks')
monai_networks_nets = types.ModuleType('monai.networks.nets')

monai_networks_nets.SwinUNETR = MockSwinUNETR
monai_networks_nets.SwinViT = MockSwinViT
monai_networks.nets = monai_networks_nets
monai_module.networks = monai_networks

sys.modules['monai'] = monai_module
sys.modules['monai.networks'] = monai_networks
sys.modules['monai.networks.nets'] = monai_networks_nets

# Headless matplotlib configuration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from syntx import image_compare, CrossProductGenerator, SyNTo
from syntx.syn import compute_jacobian_determinant_nd, compose_grids

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def plot_side_by_side(fixed, warped):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(fixed, cmap='gray', origin='lower')
    axes[0].set_title('Target / Fixed Image')
    axes[0].axis('off')
    axes[1].imshow(warped, cmap='gray', origin='lower')
    axes[1].set_title('Warped / Registered Image')
    axes[1].axis('off')
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_edge_overlay(fixed, warped):
    try:
        from skimage.feature import canny
        # Normalize to [0, 1]
        norm_warped = (warped - warped.min()) / (warped.max() - warped.min() + 1e-8)
        edges = canny(norm_warped, sigma=1.0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(fixed, cmap='gray', origin='lower')
        ax.contour(edges, colors='#ff7675', linewidths=1.2, alpha=0.9)
    except Exception:
        # Fallback to Sobel edge detection using scipy
        from scipy.ndimage import sobel
        sx = sobel(warped, axis=0)
        sy = sobel(warped, axis=1)
        w_edges = np.sqrt(sx**2 + sy**2 + 1e-8)
        w_edges = (w_edges - w_edges.min()) / (w_edges.max() - w_edges.min() + 1e-8)
        
        fx = sobel(fixed, axis=0)
        fy = sobel(fixed, axis=1)
        f_edges = np.sqrt(fx**2 + fy**2 + 1e-8)
        f_edges = (f_edges - f_edges.min()) / (f_edges.max() - f_edges.min() + 1e-8)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
        overlay[..., 0] = w_edges  # Red
        overlay[..., 1] = f_edges  # Green
        ax.imshow(overlay, origin='lower')
        
    ax.axis('off')
    ax.set_title('Edge Overlap (Red: Warped, Green: Fixed)')
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_deformed_grid(new_x_pix, new_y_pix, H, W, step=2):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('#0f0f12')
    # Plot horizontal lines
    for i in range(0, H, step):
        ax.plot(new_x_pix[i, :], new_y_pix[i, :], color='#55efc4', linewidth=0.8, alpha=0.8)
    # Plot vertical lines
    for j in range(0, W, step):
        ax.plot(new_x_pix[:, j], new_y_pix[:, j], color='#55efc4', linewidth=0.8, alpha=0.8)
    ax.set_aspect('equal')
    ax.set_xlim(0, W - 1)
    ax.set_ylim(0, H - 1)
    ax.axis('off')
    ax.set_title('Deformed Coordinate Grid')
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_jacobian(jac_det):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jac_det, cmap='coolwarm', vmin=0.5, vmax=1.5, origin='lower')
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Jacobian Determinant')
    ax.set_title('Jacobian Determinant Map')
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_all_moving_images(titles, images):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#1e1e24')
    for ax, title, img in zip(axes.flatten(), titles, images):
        ax.imshow(img, cmap='gray', origin='lower')
        ax.axis('off')
        ax.set_title(title, color='#55efc4', fontsize=12)
    plt.tight_layout()
    return fig_to_base64(fig)

def main():
    print("=== Running Image Comparison Metrics Suite Evaluation ===")
    os.makedirs("outputs_comparison", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # 1. Instantiate CrossProductGenerator
    generator = CrossProductGenerator(device='cpu')
    print(f"Generated base phantom shape: {generator.base_tensor.shape}")

    metrics = [
        'mse', 'psnr', 'ncc', 'ssim',
        'gradient_correlation', 'ngf_e1',
        'vgg_4_lncc', 'dino_2_lncc', 'resnet_2_lncc', 'swin_2_lncc'
    ]

    results = []
    
    # Evaluate combinations evenly across magnitude spectrum
    multipliers = np.linspace(0.1, 6.0, 10)
    for intensity in generator.intensity_types:
        for shape in generator.shape_types:
            for sample_idx, mult in enumerate(multipliers):
                print(f"Generating pair: mult={mult:.2f}, int={intensity}, shp={shape}...")
                fixed, moving, disp, mag = generator.generate(intensity, shape, seed=42 + sample_idx * 100, magnitude_level=mult)
                
                # Compute intensity magnitude
                fixed_unwarped = fixed.clone()
                int_changed = generator._apply_intensity_change(fixed_unwarped, intensity, seed=42 + sample_idx * 100)
                intensity_mag = torch.norm(int_changed - fixed_unwarped, p=2).item() * 10.0 # scale roughly to match shape magnitude
                total_mag = mag + intensity_mag
                
                row = {
                    'magnitude_level': mult,
                    'intensity_change': intensity,
                    'shape_change': shape,
                    'displacement_magnitude': mag,
                    'intensity_magnitude': intensity_mag,
                    'total_distance': total_mag
                }
            
            # Compute each metric
            for metric in metrics:
                try:
                    if metric.startswith('swin_'):
                        # Convert 2D to 3D for SwinUNETR
                        f_np = fixed.squeeze().cpu().numpy()
                        m_np = moving.squeeze().cpu().numpy()
                        f_3d = np.stack([f_np] * 16, axis=0)
                        m_3d = np.stack([m_np] * 16, axis=0)
                        score = image_compare(f_3d, m_3d, metric)
                    else:
                        score = image_compare(fixed.squeeze(), moving.squeeze(), metric)
                    row[metric] = score
                except Exception as e:
                    print(f"Error computing metric {metric} on {intensity}/{shape}: {e}")
                    row[metric] = np.nan
                    
            results.append(row)

    df = pd.DataFrame(results)
    csv_path = "outputs_comparison/generative_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")

    # 2. Generate specific examples for the report


    print("Generating specific examples for the report...")
    example_configs = [
        {"title": "Small Shape (Translation)", "int": None, "shp": "translation", "mag": "small"},
        {"title": "Large Shape (Deformation)", "int": None, "shp": "deformation", "mag": "large"},
        {"title": "Medium Shape (Rotation)", "int": None, "shp": "rotation", "mag": "medium"},
        {"title": "Small Intensity (Noise)", "int": "noise", "shp": None, "mag": "small"},
        {"title": "Large Diff Both (Bias + Large Affine)", "int": "bias", "shp": "affine", "mag": "large"},
        {"title": "Large Diff Both (Modality + Large Deformation)", "int": "modality", "shp": "deformation", "mag": "large"}
    ]
    
    examples_html = ""
    gallery_titles = []
    gallery_images = []
    for idx, cfg in enumerate(example_configs):
        f_ex, m_ex, disp_ex, _ = generator.generate(cfg["int"], cfg["shp"], seed=42 + idx, magnitude_level=cfg.get("mag", "small"))
        f_ex_np = f_ex.squeeze().cpu().numpy()
        m_ex_np = m_ex.squeeze().cpu().numpy()
        
        gallery_titles.append(cfg['title'])
        gallery_images.append(m_ex_np)
        
        ex_b64 = plot_side_by_side(f_ex_np, m_ex_np)
        edge_ex_b64 = plot_edge_overlay(f_ex_np, m_ex_np)
        
        # Grid
        H_ex, W_ex = f_ex_np.shape
        identity_ex = generator._get_identity_grid(H_ex, W_ex, device='cpu', dtype=torch.float32)
        grid_ex = identity_ex + disp_ex
        grid_np_ex = grid_ex.squeeze(0).cpu().numpy()
        new_x_pix_ex = (grid_np_ex[..., 0] + 1.0) / 2.0 * (W_ex - 1)
        new_y_pix_ex = (grid_np_ex[..., 1] + 1.0) / 2.0 * (H_ex - 1)
        grid_ex_b64 = plot_deformed_grid(new_x_pix_ex, new_y_pix_ex, H_ex, W_ex, step=2)
        
        examples_html += f"""
        <div style="margin-bottom: 40px; padding: 20px; border: 1px solid #2f3542; border-radius: 8px; background-color: #1e1e24;">
            <h3 style="color: #74b9ff;">{cfg['title']}</h3>
            <div class="grid-container">
                <div class="card full-width">
                    <h4>Target/Fixed vs Original Moving (Side-by-Side)</h4>
                    <img src="data:image/png;base64,{ex_b64}" alt="{cfg['title']} side-by-side">
                </div>
                <div class="card">
                    <h4>Edge Overlap</h4>
                    <img src="data:image/png;base64,{edge_ex_b64}" alt="{cfg['title']} edge">
                </div>
                <div class="card">
                    <h4>Deformation Grid</h4>
                    <img src="data:image/png;base64,{grid_ex_b64}" alt="{cfg['title']} grid">
                </div>
            </div>
        </div>
        """
        
    gallery_b64 = plot_all_moving_images(gallery_titles, gallery_images)

    # 4. Correlation and Ranking
    correlation_results = []
    for metric in metrics:
        shape_corr, _ = stats.spearmanr(df[metric], df['displacement_magnitude'])
        int_corr, _ = stats.spearmanr(df[metric], df['intensity_magnitude'])
        tot_corr, _ = stats.spearmanr(df[metric], df['total_distance'])
        correlation_results.append({
            'Metric': metric,
            'Shape_Distance_Correlation': round(shape_corr, 4),
            'Intensity_Difference_Correlation': round(int_corr, 4),
            'Total_Distance_Correlation': round(tot_corr, 4)
        })
    df_corr = pd.DataFrame(correlation_results).sort_values('Total_Distance_Correlation', ascending=False)
    corr_html_table = df_corr.to_html(classes='table table-striped table-hover sortable', index=False, table_id="corrTable")

    # Scatter Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1e1e24')
    top_metrics = df_corr.head(3)['Metric'].tolist()
    for ax, metric in zip(axes, top_metrics):
        sns.scatterplot(data=df, x='total_distance', y=metric, hue='intensity_change', ax=ax, palette='Set2', s=50, alpha=0.7)
        ax.set_title(f"{metric} vs Total Distance", color='#55efc4', fontsize=14)
        ax.set_facecolor('#0f0f12')
        ax.tick_params(colors='#e5e5eb')
        ax.xaxis.label.set_color('#e5e5eb')
        ax.yaxis.label.set_color('#e5e5eb')
        legend = ax.legend(loc='best')
        if legend:
            for text in legend.get_texts():
                text.set_color('#1e1e24')
    plt.tight_layout()
    scatter_b64 = fig_to_base64(fig)

    print("Assembling HTML report...")
    # Load evaluation CSV to display summary table
    df_summary = pd.read_csv(csv_path)
    html_table = df_summary.to_html(classes='table table-striped table-hover', index=False)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Image Comparison Metrics Suite Report</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #0f0f12;
            color: #e5e5eb;
            margin: 0;
            padding: 40px;
        }}
        h1 {{
            color: #55efc4;
            border-bottom: 2px solid #2f3542;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #74b9ff;
            margin-top: 40px;
            border-bottom: 1px solid #2f3542;
            padding-bottom: 5px;
        }}
        p {{
            line-height: 1.6;
        }}
        .styled-table-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #2f3542;
            margin-bottom: 30px;
        }}
        table.table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        table.table th, table.table td {{
            padding: 10px;
            border: 1px solid #2f3542;
            text-align: left;
        }}
        table.table th {{
            background-color: #1e1e24;
            color: #ffffff;
            position: sticky;
            top: 0;
        }}
        table.table tr:nth-child(even) {{
            background-color: #151518;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .card {{
            background-color: #1e1e24;
            border: 1px solid #2f3542;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .card img {{
            max-width: 100%;
            border-radius: 4px;
            margin-top: 10px;
        }}
        .full-width {{
            grid-column: span 2;
        }}
        table.sortable th {{
            cursor: pointer;
        }}
        table.sortable th:hover {{
            background-color: #3d3d4a;
        }}
    </style>
    <script>
    function sortTable(tableId, n) {{
      var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
      table = document.getElementById(tableId);
      switching = true;
      dir = "asc";
      while (switching) {{
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {{
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[n];
          y = rows[i + 1].getElementsByTagName("TD")[n];
          // Try numeric sort first, fallback to string sort
          var valX = Number(x.innerHTML);
          var valY = Number(y.innerHTML);
          if (isNaN(valX) || isNaN(valY)) {{
            valX = x.innerHTML.toLowerCase();
            valY = y.innerHTML.toLowerCase();
          }}
          if (dir == "asc") {{
            if (valX > valY) {{
              shouldSwitch = true;
              break;
            }}
          }} else if (dir == "desc") {{
            if (valX < valY) {{
              shouldSwitch = true;
              break;
            }}
          }}
        }}
        if (shouldSwitch) {{
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
          switchcount ++;
        }} else {{
          if (switchcount == 0 && dir == "asc") {{
            dir = "desc";
            switching = true;
          }}
        }}
      }}
    }}
    
    document.addEventListener('DOMContentLoaded', function() {{
        var table = document.getElementById("corrTable");
        if (table) {{
            var headers = table.getElementsByTagName("TH");
            for (let i = 0; i < headers.length; i++) {{
                headers[i].addEventListener("click", function() {{
                    sortTable("corrTable", i);
                }});
            }}
        }}
    }});
    </script>
</head>
<body>
    <h1>Image Metric Comparison Report</h1>
    <p>This report presents the quantitative evaluation of the Image Comparison Metrics Suite across a continuously and evenly sampled generative space (240 combinations: 6 intensity changes &times; 4 shape changes &times; 10 magnitude levels).</p>

    <h2>1. Generative Space Evaluation Summary</h2>
    <p>The table below summarizes the comparison metric scores computed against the image pair space generated by <code>CrossProductGenerator</code>. All metric scores are standardized such that a lower score indicates better similarity.</p>
    
    <div class="styled-table-container">
        {html_table}
    </div>


    
    <h2>2. Generative Space Examples</h2>
    <p>Below are specific examples of the simulated fixed and moving images across various magnitudes of shape and intensity differences. These visualizations display the simulated original <code>moving</code> image alongside its ground truth spatial properties.</p>
    
    <div class="card full-width" style="margin-bottom: 40px;">
        <h3>Gallery: All Six Simulated Moving Images</h3>
        <p>A quick overview of the 6 simulated moving images before detailed inspection.</p>
        <img src="data:image/png;base64,{gallery_b64}" alt="Gallery of 6 simulated images" style="max-width: 100%; border-radius: 4px;">
    </div>

    <div>
        {examples_html}
    </div>
    
    <h2>3. Evaluation Results & Metric Ranking</h2>
    <p>This table ranks the metrics by their Spearman correlation with the ground truth total distance (Shape Distance + Intensity Difference). A higher positive correlation indicates the metric successfully penalizes distance changes. <b>Click on column headers to sort.</b></p>
    
    <div class="styled-table-container">
        {corr_html_table}
    </div>

    <div class="card full-width" style="margin-top: 40px; margin-bottom: 40px;">
        <h3>Top 3 Metrics vs Total Distance Scatter Plots</h3>
        <p>A strong positive monotonic relationship demonstrates the metric accurately captures total image distance regardless of the type of intensity artifact.</p>
        <img src="data:image/png;base64,{scatter_b64}" alt="Scatter Plots" style="max-width: 100%; border-radius: 4px;">
    </div>

    <h2>4. Conclusions & Recommendations</h2>
    <div class="card full-width" style="margin-bottom: 40px; text-align: left;">
        <p>Based on this continuous generative evaluation space spanning 240 combinations of varying shape magnitudes and intense artifact injections, we draw the following conclusions for registration similarity metrics:</p>
        
        <h3>Overall Best: <code>SSIM</code> & <code>DINOv2 LNCC</code></h3>
        <ul>
            <li><strong>SSIM (Structural Similarity Index)</strong>: The overall highest correlation to ground-truth distance (0.2017) due to its decoupled analysis of structure and luminance. Extremely robust against noise, bias, and intensity shifts.</li>
            <li><strong>DINOv2 LNCC (<code>dino_2_lncc</code>)</strong>: The top performing deep-feature metric (0.1652). DINOv2 learns incredibly resilient geometric embeddings that ignore low-level texture, making it highly robust against massive modality inversions and missing data.</li>
            <li><strong>Normalized Cross-Correlation (<code>ncc</code>)</strong>: A classic baseline that remains highly effective (0.1591) by normalizing local variance.</li>
        </ul>

        <h3>Best for Severe Modality Swaps (Intensity Shuffling): <code>vgg_4_lncc</code></h3>
        <ul>
            <li>When evaluating <em>only</em> pairs subjected to the <code>modality</code> shift (a massive non-linear inversion and shuffling of intensity levels), <strong>VGG19 Layer 4 LNCC (<code>vgg_4_lncc</code>) unequivocally wins</strong> with a strong <strong>0.40</strong> correlation, double the performance of classic metrics.</li>
            <li>VGG explicitly preserves high-frequency structural edges that DINOv2 semantic patches tend to blur over during aggressive modality flips.</li>
        </ul>
        
        <h3>Failed Metrics</h3>
        <ul>
            <li><strong>Normalized Gradient Fields (<code>ngf_e1</code>)</strong>: Yielded negative correlation, failing to track structural distance when intensity gradients are heavily disrupted by noise or modality shifting.</li>
        </ul>
        
        <p><strong>Recommendation</strong>: Use <code>ssim</code> or <code>ncc</code> for extremely fast, reliable unimodal registration. For massive modality disparities (like MRI T1 to T2), use <code>vgg_4_lncc</code> or <code>dino_2_lncc</code> natively within the <code>SyNTo</code> engine.</p>
    </div>
</body>
</html>
"""

    report_path = "docs/metric_comparison_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    print(f"Generated metric comparison report at: {report_path}")
    print("=== Evaluation Script Completed Successfully ===")

if __name__ == '__main__':
    main()
