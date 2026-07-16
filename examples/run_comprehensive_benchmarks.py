import os
import sys
import time
import base64
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import feature
import ants

# Add src to sys.path to allow importing syntx locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx

# Set devices
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Auto-detected device: {device}")

# =====================================================================
# Diagnostics & Plotting Helpers
# =====================================================================

def get_lai_slice(image, axis, slice_index):
    """
    Reorients an ANTs image to LAI space and extracts a 2D slice.
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
    """Saves transparent label segmentation overlay over target T1w image."""
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
    """Saves Canny edges of warped moving T1w image overlaid on fixed T1w."""
    norm_warped = (warped_t1_slice - warped_t1_slice.min()) / (warped_t1_slice.max() - warped_t1_slice.min() + 1e-8)
    edges = feature.canny(norm_warped, sigma=1.0)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#08080a')
    ax.imshow(fixed_t1_slice, cmap='gray')
    ax.contour(edges, colors='#ff7675', linewidths=0.6, alpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

def save_warped_grid_overlay(t1_slice, displ_slice, spacing, filename):
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

def save_side_by_side(fixed_slice, warped_slice, filename):
    """Saves the target and registered images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#08080a')
    axes[0].imshow(fixed_slice, cmap='gray')
    axes[0].set_title("Target / Fixed Image", color='white')
    axes[0].axis('off')
    axes[1].imshow(warped_slice, cmap='gray')
    axes[1].set_title("Warped / Registered Image", color='white')
    axes[1].axis('off')
    plt.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

# =====================================================================
# Metrics Computation Helpers
# =====================================================================

def compute_tissue_overlap(fi, warped):
    """Computes the DICE overlap on foreground tissue classes."""
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return dice

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

# =====================================================================
# Objective 1: Systematic 2D Sweep
# =====================================================================

def run_2d_sweep():
    print("\n=======================================================")
    print("Objective 1: Running Systematic 2D Sweep on Phantom Pairs")
    print("=======================================================")
    
    pairs = [
        ('r16', 'r27'),
        ('r16', 'r64'),
        ('r27', 'r64')
    ]
    
    csv_path = "outputs_comparison/r1_2d_sweep_results.csv"
    os.makedirs("outputs_comparison", exist_ok=True)
    
    results = []
    
    for f_name, m_name in pairs:
        fi = ants.image_read(ants.get_data(f_name))
        mi = ants.image_read(ants.get_data(m_name))
        
        # 1. ANTs SyN Baseline
        print(f"Running ANTs SyN for pair ({f_name}, {m_name})...")
        t0 = time.time()
        reg_ants = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
        runtime = time.time() - t0
        warped = reg_ants['warpedmovout']
        l2r_path = next(tx for tx in reg_ants['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        dice = compute_tissue_overlap(fi, warped)
        results.append({
            'fixed': f_name, 'moving': m_name, 'metric': 'ants_syn', 'backend': 'ants',
            'dice': dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass
                
        # 2. Raw intensity LNCC PyTorch
        print(f"Running LNCC PyTorch for pair ({f_name}, {m_name})...")
        t0 = time.time()
        reg = syntx.syn(
            fixed=fi, moving=mi, backend='pytorch', type_of_transform='SyNTo',
            syn_metric='lncc', levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
            grad_step=0.5, flow_sigma=1.732
        )
        runtime = time.time() - t0
        warped = reg['warpedmovout']
        l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        dice = compute_tissue_overlap(fi, warped)
        results.append({
            'fixed': f_name, 'moving': m_name, 'metric': 'lncc', 'backend': 'pytorch',
            'dice': dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg['fwdtransforms'] + reg['invtransforms']:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass

        # 3. Raw intensity LNCC JAX
        print(f"Running LNCC JAX for pair ({f_name}, {m_name})...")
        t0 = time.time()
        reg = syntx.syn(
            fixed=fi, moving=mi, backend='jax', type_of_transform='SyNTo',
            syn_metric='lncc', levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
            grad_step=0.5, flow_sigma=1.732
        )
        runtime = time.time() - t0
        warped = reg['warpedmovout']
        l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        dice = compute_tissue_overlap(fi, warped)
        results.append({
            'fixed': f_name, 'moving': m_name, 'metric': 'lncc', 'backend': 'jax',
            'dice': dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg['fwdtransforms'] + reg['invtransforms']:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass

        # 4. ResNet-10 PyTorch
        print(f"Running ResNet-10 for pair ({f_name}, {m_name})...")
        t0 = time.time()
        reg = syntx.syn(
            fixed=fi, moving=mi, backend='pytorch', type_of_transform='SyNTo',
            syn_metric='resnet10', levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
            vgg_mode='lncc', vgg_layers=[4], grad_step=0.5, flow_sigma=1.732
        )
        runtime = time.time() - t0
        warped = reg['warpedmovout']
        l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        dice = compute_tissue_overlap(fi, warped)
        results.append({
            'fixed': f_name, 'moving': m_name, 'metric': 'resnet10', 'backend': 'pytorch',
            'dice': dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg['fwdtransforms'] + reg['invtransforms']:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass

        # 5. VGG19 (vgg_mode='lncc')
        print(f"Running VGG19 for pair ({f_name}, {m_name})...")
        t0 = time.time()
        reg = syntx.syn(
            fixed=fi, moving=mi, backend='pytorch', type_of_transform='SyNTo',
            syn_metric='vgg19', levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
            vgg_mode='lncc', vgg_layers=[4], grad_step=0.5, flow_sigma=1.732
        )
        runtime = time.time() - t0
        warped = reg['warpedmovout']
        l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        dice = compute_tissue_overlap(fi, warped)
        results.append({
            'fixed': f_name, 'moving': m_name, 'metric': 'vgg19', 'backend': 'pytorch',
            'dice': dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg['fwdtransforms'] + reg['invtransforms']:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass

        # 6. DINOv2
        print(f"Running DINOv2 for pair ({f_name}, {m_name})...")
        t0 = time.time()
        reg = syntx.syn(
            fixed=fi, moving=mi, backend='pytorch', type_of_transform='SyNTo',
            syn_metric='dinov2', levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
            vgg_mode='lncc', vgg_layers=[4], grad_step=0.5, flow_sigma=1.732
        )
        runtime = time.time() - t0
        warped = reg['warpedmovout']
        l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fi, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        dice = compute_tissue_overlap(fi, warped)
        results.append({
            'fixed': f_name, 'moving': m_name, 'metric': 'dinov2', 'backend': 'pytorch',
            'dice': dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg['fwdtransforms'] + reg['invtransforms']:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass

    df_2d = pd.DataFrame(results)
    df_2d.to_csv(csv_path, index=False)
    print("2D sweep complete!")
    return df_2d

# =====================================================================
# Objectives 2 & 3: Establish 3D Parity and Evaluate Deep Features
# =====================================================================

def run_3d_evaluation():
    print("\n=======================================================")
    print("Objectives 2 & 3: Running 3D Native Scans Evaluations")
    print("=======================================================")
    
    scans = [
        '28364-00000000-T1w-00',
        '28386-00000000-T1w-01',
        '28405-00000000-T1w-02',
        '28478-00000000-T1w-03'
    ]
    
    fixed_brain_path = "cache/T_template0_brain.nii.gz"
    fixed_dkt_path = "cache/T_template0_dktseg.nii.gz"
    
    fixed_brain = ants.image_read(fixed_brain_path)
    fixed_dkt = ants.image_read(fixed_dkt_path)
    
    results = []
    csv_path = "outputs_comparison/r2_3d_sweep_results.csv"
    
    for scan in scans:
        print(f"\n--- Processing Scan: {scan} ---")
        moving_brain_path = f"cache/{scan}_brain.nii.gz"
        moving_dkt_path = f"cache/{scan}_dktseg.nii.gz"
        
        moving_brain = ants.image_read(moving_brain_path)
        moving_dkt = ants.image_read(moving_dkt_path)
        
        # Compute rigid initialization
        print(f"  Computing initial rigid transform...")
        init_tx = ants.registration(fixed=fixed_brain, moving=moving_brain, type_of_transform='Rigid')
        tx_path_orig = init_tx['fwdtransforms'][0]
        tx_path = f"cache/{scan}_initial_rigid.mat"
        shutil.copy(tx_path_orig, tx_path)
        
        # Clean up temp files from rigid
        for p in init_tx['fwdtransforms'] + init_tx['invtransforms']:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
                
        # 1. ANTs SyN Baseline
        print(f"  Running 3D ANTs SyN baseline...")
        t0 = time.time()
        reg_ants = ants.registration(
            fixed=fixed_brain, moving=moving_brain, type_of_transform='SyN', initial_transform=tx_path
        )
        runtime = time.time() - t0
        warped_dkt = ants.apply_transforms(
            fixed=fixed_brain, moving=moving_dkt, transformlist=reg_ants['fwdtransforms'], interpolator='genericLabel'
        )
        dices = compute_multiregion_dice(fixed_dkt, warped_dkt)
        mean_dice = np.mean(list(dices.values()))
        l2r_path = next(tx for tx in reg_ants['fwdtransforms'] if tx.endswith('.nii.gz'))
        jac_img = ants.create_jacobian_determinant_image(fixed_brain, l2r_path)
        folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
        results.append({
            'scan': scan, 'metric': 'ants_syn', 'backend': 'ants', 'dice': mean_dice, 'folding_rate': folding, 'runtime': runtime
        })
        for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
            if path != tx_path and os.path.exists(path):
                try: os.remove(path)
                except: pass

        # Configs to establish parity and evaluate deep features
        # We use levels=[8, 4, 2, 1], with 0 iterations on level 8 and 1 to speed up execution
        # affine_iterations=[50, 30, 10, 0] gives robust affine alignment
        # reg_iterations=[20, 10, 5, 0] gives rapid deformable alignment
        # We auto-detect device (MPS/CPU)
        configs = [
            # Parity check LNCC (PyTorch)
            {'metric': 'lncc', 'backend': 'pytorch', 'mode': 'lncc_3d', 'layers': [4]},
            # Parity check LNCC (JAX)
            {'metric': 'lncc', 'backend': 'jax', 'mode': 'lncc_3d', 'layers': [4]},
            # Perceptual VGG19 (3D LNCC with Layer 4)
            {'metric': 'vgg19', 'backend': 'pytorch', 'mode': 'lncc_3d', 'layers': [4]},
            # Perceptual DINOv2
            {'metric': 'dinov2', 'backend': 'pytorch', 'mode': 'lncc_3d', 'layers': [4]},
            # Perceptual ResNet-10
            {'metric': 'resnet10', 'backend': 'pytorch', 'mode': 'lncc_3d', 'layers': [4]}
        ]
        
        for cfg in configs:
            print(f"  Running 3D {cfg['metric']} ({cfg['backend']})...")
            t0 = time.time()
            try:
                reg = syntx.syn(
                    fixed=fixed_brain,
                    moving=moving_brain,
                    type_of_transform='SyNTo',
                    backend=cfg['backend'],
                    syn_metric=cfg['metric'],
                    levels=[8, 4, 2, 1],
                    affine_iterations=[50, 30, 10, 0],
                    reg_iterations=[20, 10, 5, 0],
                    vgg_mode=cfg['mode'],
                    vgg_layers=cfg['layers'],
                    initial_transform=tx_path,
                    grad_step=0.75,
                    flow_sigma=3.0
                )
                runtime = time.time() - t0
                
                warped_dkt = ants.apply_transforms(
                    fixed=fixed_brain, moving=moving_dkt, transformlist=reg['fwdtransforms'], interpolator='genericLabel'
                )
                dices = compute_multiregion_dice(fixed_dkt, warped_dkt)
                mean_dice = np.mean(list(dices.values()))
                
                l2r_path = next(tx for tx in reg['fwdtransforms'] if tx.endswith('.nii.gz'))
                jac_img = ants.create_jacobian_determinant_image(fixed_brain, l2r_path)
                folding = float(100.0 * np.mean(jac_img.numpy() <= 0))
                
                results.append({
                    'scan': scan, 'metric': cfg['metric'], 'backend': cfg['backend'],
                    'dice': mean_dice, 'folding_rate': folding, 'runtime': runtime
                })
                
                # Save visual slices for the report for a representative scan (e.g. T1w-00)
                if scan == '28364-00000000-T1w-00':
                    fixed_lai = fixed_brain.reorient_image2("LAI")
                    z_mid = int(fixed_lai.shape[2] * 0.55)
                    fixed_t1_slice = get_lai_slice(fixed_brain, 2, z_mid)
                    warped_moving = ants.apply_transforms(
                        fixed=fixed_brain, moving=moving_brain, transformlist=reg['fwdtransforms']
                    )
                    warped_t1_slice = get_lai_slice(warped_moving, 2, z_mid)
                    warped_dkt_slice = get_lai_slice(warped_dkt, 2, z_mid)
                    comp_warp_img = ants.image_read(l2r_path)
                    displ_slice = get_lai_slice(comp_warp_img, 2, z_mid)
                    jac_slice = get_lai_slice(jac_img, 2, z_mid)
                    spacing_info = fixed_lai.spacing
                    
                    save_label_overlay_plot(fixed_t1_slice, warped_dkt_slice, f"outputs_comparison/{scan}_{cfg['metric']}_{cfg['backend']}_lbl.png")
                    save_edge_overlay_plot(fixed_t1_slice, warped_t1_slice, f"outputs_comparison/{scan}_{cfg['metric']}_{cfg['backend']}_edge.png")
                    save_warped_grid_overlay(fixed_t1_slice, displ_slice, spacing_info, f"outputs_comparison/{scan}_{cfg['metric']}_{cfg['backend']}_grid.png")
                    save_jacobian_slice(fixed_t1_slice, jac_slice, f"outputs_comparison/{scan}_{cfg['metric']}_{cfg['backend']}_jacobian.png")
                    save_side_by_side(fixed_t1_slice, warped_t1_slice, f"outputs_comparison/{scan}_{cfg['metric']}_{cfg['backend']}_sidebyside.png")
                
                # Clean up fwd / inv transforms
                for path in reg['fwdtransforms'] + reg['invtransforms']:
                    if path != tx_path and os.path.exists(path):
                        try: os.remove(path)
                        except: pass
                        
            except Exception as e:
                print(f"    Failed: {e}")
                results.append({
                    'scan': scan, 'metric': cfg['metric'], 'backend': cfg['backend'],
                    'dice': np.nan, 'folding_rate': np.nan, 'runtime': np.nan
                })
                
        # Clean up initial rigid transform
        if os.path.exists(tx_path):
            try: os.remove(tx_path)
            except: pass
            
    df_3d = pd.DataFrame(results)
    df_3d.to_csv(csv_path, index=False)
    print("3D native scans evaluations complete!")
    return df_3d

# =====================================================================
# Objective 4: Compile HTML Report docs/deep_feature_impact_report.html
# =====================================================================

def compile_html_report(df_2d, df_3d):
    print("\n=======================================================")
    print("Objective 4: Compiling HTML Dashboard Report to docs/deep_feature_impact_report.html")
    print("=======================================================")
    
    def get_b64(path):
        if not path or not os.path.exists(path): 
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
            
    summary_2d = df_2d.groupby(['metric', 'backend']).mean(numeric_only=True).reset_index()
    summary_3d = df_3d.groupby(['metric', 'backend']).mean(numeric_only=True).reset_index()
    
    rows_2d = []
    for _, r in summary_2d.iterrows():
        rows_2d.append(f"""
        <tr>
            <td style="font-weight: bold; color: #55efc4;">{r['metric'].upper()} ({r['backend'].upper()})</td>
            <td>{r['dice']:.4f}</td>
            <td>{r['folding_rate']:.4f}%</td>
            <td>{r['runtime']:.2f}s</td>
        </tr>
        """)
        
    rows_3d = []
    for _, r in summary_3d.iterrows():
        rows_3d.append(f"""
        <tr>
            <td style="font-weight: bold; color: #74b9ff;">{r['metric'].upper()} ({r['backend'].upper()})</td>
            <td>{r['dice']:.4f}</td>
            <td>{r['folding_rate']:.4f}%</td>
            <td>{r['runtime']:.2f}s</td>
        </tr>
        """)
        
    top_scan = "28364-00000000-T1w-00"
    top_metric = "vgg19"
    top_backend = "pytorch"
    
    lbl_b64 = get_b64(f"outputs_comparison/{top_scan}_{top_metric}_{top_backend}_lbl.png")
    edge_b64 = get_b64(f"outputs_comparison/{top_scan}_{top_metric}_{top_backend}_edge.png")
    grid_b64 = get_b64(f"outputs_comparison/{top_scan}_{top_metric}_{top_backend}_grid.png")
    jac_b64 = get_b64(f"outputs_comparison/{top_scan}_{top_metric}_{top_backend}_jacobian.png")
    sidebyside_b64 = get_b64(f"outputs_comparison/{top_scan}_{top_metric}_{top_backend}_sidebyside.png")
    
    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Deep Feature Registration Performance & Parity Report</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 40px; background: #0c0c0e; color: #f1f2f6; }}
            h1 {{ font-size: 34px; background: -webkit-linear-gradient(#55efc4, #74b9ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }}
            p.subtitle {{ color: #a4b0be; font-size: 15px; margin-top: 0; margin-bottom: 40px; }}
            h2 {{ color: #ffffff; border-bottom: 1px solid #2f3542; padding-bottom: 8px; margin-top: 50px; font-weight: 500; }}
            .styled-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 15px; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 16px rgba(0,0,0,0.4); border: 1px solid #2f3542; }}
            .styled-table th, .styled-table td {{ padding: 14px 18px; text-align: left; }}
            .styled-table thead tr {{ background-color: #1e1e24; color: #ffffff; border-bottom: 2px solid #2f3542; font-weight: bold; }}
            .styled-table tbody tr {{ border-bottom: 1px solid #2f3542; background: #131316; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #1a1a20; }}
            .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #55efc4; }}
            .summary-card {{ background: #1e1e24; padding: 25px; border-radius: 12px; border: 1px solid #2f3542; margin-bottom: 40px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; }}
            .summary-stat {{ text-align: center; flex: 1; min-width: 150px; margin: 10px; }}
            .summary-value {{ font-size: 28px; font-weight: bold; color: #55efc4; }}
            .summary-label {{ font-size: 12px; color: #a4b0be; text-transform: uppercase; margin-top: 5px; }}
            .layout-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px; }}
            .chart-box {{ background: #1e1e24; padding: 25px; border-radius: 12px; border: 1px solid #2f3542; box-shadow: 0 8px 16px rgba(0,0,0,0.3); text-align: center; }}
            .chart-box img {{ max-width: 100%; border-radius: 8px; }}
            .img-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 45px; }}
            .img-box {{ background: #1e1e24; padding: 12px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid #2f3542; text-align: center; }}
            .img-box img {{ width: 100%; border-radius: 6px; }}
            .img-box h3 {{ color: #ffffff; font-size: 14px; margin-top: 0; margin-bottom: 10px; font-weight: 400; }}
            .side-by-side-box {{ background: #1e1e24; padding: 20px; border-radius: 12px; border: 1px solid #2f3542; text-align: center; margin-bottom: 40px; }}
            .side-by-side-box img {{ max-width: 90%; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>Deep Feature Registration Performance & Parity Report</h1>
        <p class="subtitle">Systematic validation and comparison of PyTorch-based JAX/PyTorch SyNTo models against ANTs SyN baseline.</p>
        
        <h2>Quantitative Summary</h2>
        <div class="summary-card">
            <div class="summary-stat">
                <div class="summary-value" style="color: #55efc4;">{df_2d[df_2d['metric'] == 'lncc']['dice'].mean():.4f}</div>
                <div class="summary-label">2D LNCC DICE</div>
            </div>
            <div class="summary-stat">
                <div class="summary-value" style="color: #74b9ff;">{df_3d[df_3d['metric'] == 'lncc']['dice'].mean():.4f}</div>
                <div class="summary-label">3D LNCC DICE</div>
            </div>
            <div class="summary-stat">
                <div class="summary-value" style="color: #ffeaa7;">{df_3d[df_3d['metric'] == 'vgg19']['dice'].mean():.4f}</div>
                <div class="summary-label">3D VGG19 DICE</div>
            </div>
            <div class="summary-stat">
                <div class="summary-value" style="color: #ff7675;">{df_3d[df_3d['metric'] == 'ants_syn']['dice'].mean():.4f}</div>
                <div class="summary-label">ANTs SyN 3D DICE</div>
            </div>
        </div>

        <div class="layout-grid">
            <div>
                <h2>R1: 2D Sweep Results (Average)</h2>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Registration Metric</th>
                            <th>Mean Overlap (DICE)</th>
                            <th>Folding Rate (J &le; 0)</th>
                            <th>Runtime</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_2d)}
                    </tbody>
                </table>
            </div>
            <div>
                <h2>R2: 3D Scans Results (Average)</h2>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Registration Metric</th>
                            <th>DKT Label DICE</th>
                            <th>Jacobian Folding Rate</th>
                            <th>Runtime</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_3d)}
                    </tbody>
                </table>
            </div>
        </div>

        <h2>Visual Inspection: Top-Performing Config ({top_metric.upper()} on Scan {top_scan})</h2>
        <div class="side-by-side-box">
            <h3>Side-by-Side Registered (Warped) vs Target (Fixed) Images</h3>
            <img src="data:image/png;base64,{sidebyside_b64}">
        </div>

        <div class="img-grid">
            <div class="img-box">
                <h3>Warped DKT Label Overlay</h3>
                <img src="data:image/png;base64,{lbl_b64}">
            </div>
            <div class="img-box">
                <h3>Warped Canny Edges</h3>
                <img src="data:image/png;base64,{edge_b64}">
            </div>
            <div class="img-box">
                <h3>Deformed Coordinate Grid</h3>
                <img src="data:image/png;base64,{grid_b64}">
            </div>
            <div class="img-box">
                <h3>Jacobian Determinant Map</h3>
                <img src="data:image/png;base64,{jac_b64}">
            </div>
        </div>
    </body>
    </html>
    """
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/deep_feature_impact_report.html", "w") as f:
        f.write(html)
        
    print("Report compiled successfully and saved to docs/deep_feature_impact_report.html")

# =====================================================================
# Main Execution Entrypoint
# =====================================================================

def main():
    t_start = time.time()
    
    df_2d = run_2d_sweep()
    df_3d = run_3d_evaluation()
    compile_html_report(df_2d, df_3d)
    
    print(f"\nAll benchmarks completed in {(time.time() - t_start) / 60.0:.2f} minutes.")

if __name__ == '__main__':
    main()
