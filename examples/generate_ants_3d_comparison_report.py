import ants
import base64
import os
import tempfile
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from syntx.syn import SyNTo
from syntx.syn_jax import SyNTo as SyNToJax
import syntx
# ff='examples/generate_ants_3d_comparison_report.py'
# exec(open(ff).read())

def plot_to_base64_fig(fig):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
        with open(tmp.name, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
    os.remove(tmp.name)
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def generate_plot_b64(base_image, overlay=None, title="", cmap='gray', overlay_cmap='jet', alpha=0.5):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_name = tmp.name
    
    ants.plot(
        base_image, 
        overlay=overlay, 
        cmap=cmap,
        overlay_cmap=overlay_cmap, 
        overlay_alpha=alpha,
        title=title, 
        axis=2,
        filename=tmp_name
    )
    
    with open(tmp_name, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    os.remove(tmp_name)
    return f"data:image/png;base64,{encoded}"

def compute_dkt_dice(fixed_labels, warped_moving_labels):
    overlap = ants.label_overlap_measures(fixed_labels, warped_moving_labels)
    # Filter out background (Label 0) and 'All'
    overlap = overlap[(overlap['Label'] != 0) & (overlap['Label'] != 'All')]
    if 'TotalOrTargetOverlap' in overlap.columns:
        return float(overlap['TotalOrTargetOverlap'].mean())
    elif 'TargetOverlap' in overlap.columns:
        return float(overlap['TargetOverlap'].mean())
    elif 'MeanOverlap' in overlap.columns:
        return float(overlap['MeanOverlap'].mean())
    return 0.0

def plot_metrics_comparison(
    mi_ants, mi_py, mi_jax, mi_vgg,
    dice_ants, dice_py, dice_jax, dice_vgg,
    min_jac_ants, min_jac_py, min_jac_jax, min_jac_vgg,
    time_ants, time_py, time_jax, time_vgg
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    methods = ['ANTs', 'PyTorch LNCC', 'JAX LNCC', 'PyTorch VGG']
    colors = ['#34495e', '#3498db', '#2ecc71', '#9b59b6']
    
    # 1. MI (L2R)
    axs[0, 0].bar(methods, [mi_ants, mi_py, mi_jax, mi_vgg], color=colors, width=0.4)
    axs[0, 0].set_title("Mutual Information (L2R) [Lower = Better]", fontsize=11, fontweight='bold', pad=10)
    axs[0, 0].set_ylabel("MI Value")
    axs[0, 0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 2. Tissue Overlap (Dice)
    axs[0, 1].bar(methods, [dice_ants, dice_py, dice_jax, dice_vgg], color=colors, width=0.4)
    axs[0, 1].set_title("Structural DKT Overlap (Mean Dice) [Higher = Better]", fontsize=11, fontweight='bold', pad=10)
    axs[0, 1].set_ylabel("Dice Coefficient")
    axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.5)

    # 3. Min Jacobian L2R
    axs[1, 0].bar(methods, [min_jac_ants, min_jac_py, min_jac_jax, min_jac_vgg], color=colors, width=0.4)
    axs[1, 0].set_title("Min Jacobian Det (L2R) [Higher = Safer]", fontsize=11, fontweight='bold', pad=10)
    axs[1, 0].set_ylabel("Min Jacobian")
    axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 4. Execution Time (s)
    axs[1, 1].bar(methods, [time_ants, time_py, time_jax, time_vgg], color=colors, width=0.4)
    axs[1, 1].set_title("Execution Time [Lower = Faster]", fontsize=11, fontweight='bold', pad=10)
    axs[1, 1].set_ylabel("Seconds")
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return plot_to_base64_fig(fig)

if True:
    print("Loading 3D Mindboggle images...")
    base_path = '/Users/stnava/data/mindboggle/volumes'
    base_path = '/Users/stnava/Downloads/mindboggle_data/'
    fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
    mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
    fl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')
    ml_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')
    
    fi_full = ants.image_read(fi_path)
    mi_full = ants.image_read(mi_path)
    fl_full = ants.image_read(fl_path)
    ml_full = ants.image_read(ml_path)
    
    print("Cropping fixed image to bounding box to save time...")
    mask = ants.get_mask(fi_full)
    mask_dilated = ants.iMath(mask, "MD", 12)
    fi = ants.crop_image(fi_full, mask_dilated)
    fl = ants.crop_image(fl_full, mask_dilated)
    
    # Keep moving image at full native size to stress-test physical coordinate tracking
    mi = mi_full
    ml = ml_full
    
    # Set moderate deformation iterations since we cropped
    syn_iters = [50, 20, 0]
    
    # --- 2. ANTs Registration ---
    
    print("Running ANTs SyN registration...")
    t0 = time.time()
    reg_ants = ants.registration(
        fi, mi, 'SyN', 
        reg_iterations=syn_iters, 
        syn_metric='mattes', syn_sampling=32
    )
    t1 = time.time()
    ants_time = t1 - t0
    
    warped_ants_mov = reg_ants['warpedmovout']
    mi_ants_mov = ants.image_mutual_information(fi, warped_ants_mov)
    
    warped_py_img_mov = ants.apply_transforms(fi, mi, reg_ants['fwdtransforms'][1])
    deenk
    reg_py2 = syntx.syn(
            fixed=fi, moving=warped_py_img_mov, type_of_transform='SyN', backend='jax',
            affine_iterations=[200, 20, 5], reg_iterations=[100,50,5],
            syn_metric='mattes_mi', sampling_percentage=0.2, verbose=2
        )
    ants.image_write( fi, '/tmp/tempf.nii.gz')
    ants.image_write( reg_py2['warpedmovout'], '/tmp/tempm.nii.gz')
    mi_py_mov = ants.image_mutual_information(fi, reg_py2['warpedmovout'])
    mi_py_mov


    # --- 3. Syntx Registration ---
    sampling_percent = 0.2
    
    # Run PyTorch LNCC
    print("Running Syntx SyNTo PyTorch registration (composed affine + SyN)...")
    t0 = time.time()
    m='mattes_mi'
    reg_py = syntx.syn(
            fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
            affine_iterations=[200, 20, 5], reg_iterations=[20,20,5],
            syn_metric='mattes_mi',
            sampling_percentage=sampling_percent, verbose=2
        )
    t1 = time.time()
    py_time = t1 - t0
    ants.image_mutual_information(fi, reg_py['warpedmovout'])
    ants.image_write( warped_py_img_mov, '/tmp/temp2.nii.gz')
    ants.image_write( reg_py['warpedmovout'], '/tmp/temp3.nii.gz')
    reg_py3 = ants.registration(
            fi, warped_py_img_mov, 'SyN', 
            reg_iterations=[10,0], 
            syn_metric='mattes', verbose=True )
    ants.image_write( reg_py3['warpedmovout'], '/tmp/tempm.nii.gz')
    mi_py_mov = ants.image_mutual_information(fi, reg_py3['warpedmovout'])
    mi_py_mov
    doink2
    # Run JAX LNCC
    print("Running Syntx SyNTo JAX registration (composed affine + SyN)...")
    t0 = time.time()
    reg_jax = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='SyN', backend='jax',
        affine_iterations=[100, 50, 20], reg_iterations=syn_iters,
        syn_metric='lncc', lncc_window_size=5, sampling_percentage=sampling_percent
    )
    t1 = time.time()
    jax_time = t1 - t0
    warped_jax_img_mov = ants.apply_transforms(fi, mi, reg_jax['fwdtransforms'])
    mi_jax_mov = ants.image_mutual_information(fi, warped_jax_img_mov)
    
    # Run PyTorch VGG
    print("Running Syntx SyNTo PyTorch registration (composed affine + SyN) with VGG+LNCC metric...")
    t0 = time.time()
    reg_vgg = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
        affine_iterations=[100, 50, 20], reg_iterations=syn_iters,
        syn_metric='vgg19', vgg_mode='lncc_3d', vgg_layers=[4], sampling_percentage=sampling_percent
    )
    t1 = time.time()
    vgg_time = t1 - t0
    warped_vgg_img_mov = ants.apply_transforms(fi, mi, reg_vgg['fwdtransforms'])
    mi_vgg_mov = ants.image_mutual_information(fi, warped_vgg_img_mov)
    
    print("Computing DKT Label DICE overlap metrics...")
    warped_ants_ml = ants.apply_transforms(fi, ml, reg_ants['fwdtransforms'], interpolator='nearestNeighbor')
    warped_py_ml = ants.apply_transforms(fi, ml, reg_py['fwdtransforms'], interpolator='nearestNeighbor')
    warped_jax_ml = ants.apply_transforms(fi, ml, reg_jax['fwdtransforms'], interpolator='nearestNeighbor')
    warped_vgg_ml = ants.apply_transforms(fi, ml, reg_vgg['fwdtransforms'], interpolator='nearestNeighbor')
    
    dice_ants = compute_dkt_dice(fl, warped_ants_ml)
    dice_py = compute_dkt_dice(fl, warped_py_ml)
    dice_jax = compute_dkt_dice(fl, warped_jax_ml)
    dice_vgg = compute_dkt_dice(fl, warped_vgg_ml)
    
    print(f"  ANTs MI:         {mi_ants_mov:.6f} | Mean DKT Dice: {dice_ants:.4f}")
    print(f"  PyTorch LNCC MI: {mi_py_mov:.6f} | Mean DKT Dice: {dice_py:.4f}")
    print(f"  JAX LNCC MI:     {mi_jax_mov:.6f} | Mean DKT Dice: {dice_jax:.4f}")
    print(f"  PyTorch VGG MI:  {mi_vgg_mov:.6f} | Mean DKT Dice: {dice_vgg:.4f}")
    
    print("Calculating Jacobian determinants for topological folding...")
    
    l2r_path = next(tx for tx in reg_py['fwdtransforms'] if tx.endswith('.nii.gz'))
    l2r_path_jax = next(tx for tx in reg_jax['fwdtransforms'] if tx.endswith('.nii.gz'))
    l2r_path_vgg = next(tx for tx in reg_vgg['fwdtransforms'] if tx.endswith('.nii.gz'))
    
    jac_ants_img = ants.create_jacobian_determinant_image(fi, reg_ants['fwdtransforms'][0])
    jac_ants_np = jac_ants_img.numpy()
    folding_ants = 100.0 * np.mean(jac_ants_np <= 0)
    min_jac_ants = float(jac_ants_np.min())
    
    jac_py_img = ants.create_jacobian_determinant_image(fi, l2r_path)
    jac_py_np = jac_py_img.numpy()
    folding_py = 100.0 * np.mean(jac_py_np <= 0)
    min_jac_py = float(jac_py_np.min())
    
    jac_jax_img = ants.create_jacobian_determinant_image(fi, l2r_path_jax)
    jac_jax_np = jac_jax_img.numpy()
    folding_jax = 100.0 * np.mean(jac_jax_np <= 0)
    min_jac_jax = float(jac_jax_np.min())
    
    jac_vgg_img = ants.create_jacobian_determinant_image(fi, l2r_path_vgg)
    jac_vgg_np = jac_vgg_img.numpy()
    folding_vgg = 100.0 * np.mean(jac_vgg_np <= 0)
    min_jac_vgg = float(jac_vgg_np.min())
    
    print("Generating physically accurate report visualizations using ants.plot...")
    
    # Input Images
    fi_b64 = generate_plot_b64(fi, title="Fixed Target (Cropped)")
    mi_b64 = generate_plot_b64(mi, title="Moving Source (Full)")
    
    # Image Overlays
    ants_img_ov = generate_plot_b64(fi, warped_ants_mov, title="ANTs SyN Overlay", overlay_cmap='hot', alpha=0.5)
    py_img_ov = generate_plot_b64(fi, warped_py_img_mov, title="PyTorch LNCC Overlay", overlay_cmap='hot', alpha=0.5)
    jax_img_ov = generate_plot_b64(fi, warped_jax_img_mov, title="JAX LNCC Overlay", overlay_cmap='hot', alpha=0.5)
    vgg_img_ov = generate_plot_b64(fi, warped_vgg_img_mov, title="PyTorch VGG Overlay", overlay_cmap='hot', alpha=0.5)
    
    # Label Overlays
    fl_native_ov = generate_plot_b64(fi, fl, title="Fixed Native Labels", overlay_cmap='tab20', alpha=0.7)
    ants_lbl_ov = generate_plot_b64(fi, warped_ants_ml, title="ANTs Warped Labels", overlay_cmap='tab20', alpha=0.7)
    py_lbl_ov = generate_plot_b64(fi, warped_py_ml, title="PyTorch LNCC Warped Labels", overlay_cmap='tab20', alpha=0.7)
    jax_lbl_ov = generate_plot_b64(fi, warped_jax_ml, title="JAX LNCC Warped Labels", overlay_cmap='tab20', alpha=0.7)
    vgg_lbl_ov = generate_plot_b64(fi, warped_vgg_ml, title="PyTorch VGG Warped Labels", overlay_cmap='tab20', alpha=0.7)
    
    # Jacobian Overlays
    ants_jac_ov = generate_plot_b64(fi, jac_ants_img, title="ANTs Jacobian", overlay_cmap='seismic', alpha=0.6)
    py_jac_ov = generate_plot_b64(fi, jac_py_img, title="PyTorch LNCC Jacobian", overlay_cmap='seismic', alpha=0.6)
    jax_jac_ov = generate_plot_b64(fi, jac_jax_img, title="JAX LNCC Jacobian", overlay_cmap='seismic', alpha=0.6)
    vgg_jac_ov = generate_plot_b64(fi, jac_vgg_img, title="PyTorch VGG Jacobian", overlay_cmap='seismic', alpha=0.6)
    
    # Clean up temp transform files from registration runs
    for r in [reg_py, reg_jax, reg_vgg]:
        for path in r['fwdtransforms'] + r['invtransforms']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    # --- Generate Plots ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(reg_py['syn_losses'], color='blue', label='PyTorch LNCC')
    axs[0].set_title("Syntx PyTorch LNCC Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Similarity Loss")
    axs[0].legend()
    
    axs[1].plot(reg_jax['syn_losses'], color='green', label='JAX LNCC')
    axs[1].set_title("Syntx JAX LNCC Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Similarity Loss")
    axs[1].legend()

    axs[2].plot(reg_vgg['syn_losses'], color='purple', label='PyTorch VGG')
    axs[2].set_title("Syntx PyTorch VGG Loss")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Similarity Loss")
    axs[2].legend()
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_name = tmp.name
    plt.savefig(tmp_name)
    with open(tmp_name, 'rb') as f:
        encoded_plots = base64.b64encode(f.read()).decode('utf-8')
    os.remove(tmp_name)
    plot_base64 = f"data:image/png;base64,{encoded_plots}"
    
    comparison_plot_base64 = plot_metrics_comparison(
        mi_ants_mov, mi_py_mov, mi_jax_mov, mi_vgg_mov,
        dice_ants, dice_py, dice_jax, dice_vgg,
        min_jac_ants, min_jac_py, min_jac_jax, min_jac_vgg,
        ants_time, py_time, jax_time, vgg_time
    )
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ANTs vs Syntx 3D SyN Comparison</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f4f4f9; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
            .panel {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex: 1; min-width: 250px; max-width: 400px; }}
            .panel img {{ width: 100%; border-radius: 4px; }}
            .metric-box {{ background: #eef2f5; padding: 10px 20px; border-radius: 5px; font-size: 15px; font-weight: bold; margin-top: 15px; display: inline-block; color: #2c3e50; border-left: 4px solid #3498db; margin-bottom: 5px; text-align: left; }}
        </style>
    </head>
    <body>
        <h1>3D Mindboggle DKT Alignment (Syntx vs ANTs)</h1>
        <p>This report documents the performance of classic ANTs SyN compared to PyTorch and JAX backends of the Syntx package, running at native resolution with robust physical coordinate tracking.</p>
        
        <h2>1. The Raw Input Data (Native Resolution)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Fixed Target (Cropped to FOV)</h3>
                <img src="{fi_b64}" />
                <div class="metric-box">Size: {fi.shape}</div>
            </div>
            <div class="panel">
                <h3>Moving Target (Full Size)</h3>
                <img src="{mi_b64}" />
                <div class="metric-box">Size: {mi.shape}</div>
            </div>
            <div class="panel">
                <h3>Fixed Ground Truth Labels</h3>
                <img src="{fl_native_ov}" />
            </div>
        </div>

        <h2>2. Classic ANTs Registration (Reference)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Image Match Overlay</h3>
                <img src="{ants_img_ov}" />
                <br>
                <div class="metric-box">MI: {mi_ants_mov:.4f}</div>
            </div>
            <div class="panel">
                <h3>Warped DKT Labels</h3>
                <img src="{ants_lbl_ov}" />
                <br>
                <div class="metric-box">Mean Struct Dice: {dice_ants:.4f}</div>
            </div>
            <div class="panel">
                <h3>Jacobian Topology Map</h3>
                <img src="{ants_jac_ov}" />
                <br>
                <div class="metric-box">Folding: {folding_ants:.4f}%<br>Min Jac: {min_jac_ants:.4f}</div>
            </div>
        </div>

        <h2>3. Syntx PyTorch Registration (SyNTo + LNCC)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Image Match Overlay</h3>
                <img src="{py_img_ov}" />
                <br>
                <div class="metric-box">MI: {mi_py_mov:.4f}</div>
            </div>
            <div class="panel">
                <h3>Warped DKT Labels</h3>
                <img src="{py_lbl_ov}" />
                <br>
                <div class="metric-box">Mean Struct Dice: {dice_py:.4f}</div>
            </div>
            <div class="panel">
                <h3>Jacobian Topology Map</h3>
                <img src="{py_jac_ov}" />
                <br>
                <div class="metric-box">Folding: {folding_py:.4f}%<br>Min Jac: {min_jac_py:.4f}</div>
            </div>
        </div>

        <h2>4. Syntx JAX Registration (SyNTo + LNCC)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Image Match Overlay</h3>
                <img src="{jax_img_ov}" />
                <br>
                <div class="metric-box">MI: {mi_jax_mov:.4f}</div>
            </div>
            <div class="panel">
                <h3>Warped DKT Labels</h3>
                <img src="{jax_lbl_ov}" />
                <br>
                <div class="metric-box">Mean Struct Dice: {dice_jax:.4f}</div>
            </div>
            <div class="panel">
                <h3>Jacobian Topology Map</h3>
                <img src="{jax_jac_ov}" />
                <br>
                <div class="metric-box">Folding: {folding_jax:.4f}%<br>Min Jac: {min_jac_jax:.4f}</div>
            </div>
        </div>

        <h2>5. Syntx PyTorch Registration (SyNTo + VGG-LNCC)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Image Match Overlay</h3>
                <img src="{vgg_img_ov}" />
                <br>
                <div class="metric-box">MI: {mi_vgg_mov:.4f}</div>
            </div>
            <div class="panel">
                <h3>Warped DKT Labels</h3>
                <img src="{vgg_lbl_ov}" />
                <br>
                <div class="metric-box">Mean Struct Dice: {dice_vgg:.4f}</div>
            </div>
            <div class="panel">
                <h3>Jacobian Topology Map</h3>
                <img src="{vgg_jac_ov}" />
                <br>
                <div class="metric-box">Folding: {folding_vgg:.4f}%<br>Min Jac: {min_jac_vgg:.4f}</div>
            </div>
        </div>

        <h2>6. Metrics and Convergence Summary</h2>
        <div style="background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 20px;">
            <img src="{comparison_plot_base64}" style="width: 100%; max-width: 900px;" />
            <br>
            <img src="{plot_base64}" style="width: 100%; max-width: 900px; margin-top: 20px;" />
        </div>
    </body>
    </html>
    """
    
    output_dir = "/Users/stnava/code/syntx/docs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "parity_report_3d.html")
    with open(output_path, "w") as f:
        f.write(html_content)
        
    print(f"Report generated successfully at {output_path}")

if __name__ == "__main__":
    main()
