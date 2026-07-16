import ants
import base64
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from syntx.syn import SyNTo
from syntx.syn_jax import SyNTo as SyNToJax
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

def compute_smoothness_metrics(disp_np, spacing):
    sp_y, sp_x = spacing
    du_dy = np.gradient(disp_np, axis=0) / sp_y
    du_dx = np.gradient(disp_np, axis=1) / sp_x
    smooth_1st = np.mean(du_dy**2 + du_dx**2)
    
    d2u_dy2 = np.gradient(du_dy, axis=0) / sp_y
    d2u_dxdy = np.gradient(du_dy, axis=1) / sp_x
    d2u_dx2 = np.gradient(du_dx, axis=1) / sp_x
    smooth_2nd = np.mean(d2u_dy2**2 + 2 * d2u_dxdy**2 + d2u_dx2**2)
    return float(smooth_1st), float(smooth_2nd)

def compute_tissue_overlap(fi, warped):
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return dice

def plot_metrics_comparison(
    mi_ants, mi_py, mi_jax, mi_vgg,
    dice_ants, dice_py, dice_jax, dice_vgg,
    min_jac_ants, min_jac_py, min_jac_jax, min_jac_vgg,
    smooth_2nd_ants, smooth_2nd_py, smooth_2nd_jax, smooth_2nd_vgg,
    time_ants, time_py, time_jax, time_vgg
):
    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    methods = ['ANTs', 'PyTorch LNCC', 'JAX LNCC', 'PyTorch VGG']
    colors = ['#34495e', '#3498db', '#2ecc71', '#9b59b6']
    
    # 1. MI (L2R)
    axs[0, 0].bar(methods, [mi_ants, mi_py, mi_jax, mi_vgg], color=colors, width=0.4)
    axs[0, 0].set_title("Mutual Information (L2R) [Lower = Better]", fontsize=11, fontweight='bold', pad=10)
    axs[0, 0].set_ylabel("MI Value")
    axs[0, 0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 2. Tissue Overlap (Dice)
    axs[0, 1].bar(methods, [dice_ants, dice_py, dice_jax, dice_vgg], color=colors, width=0.4)
    axs[0, 1].set_title("Tissue Overlap (Dice) [Higher = Better]", fontsize=11, fontweight='bold', pad=10)
    axs[0, 1].set_ylabel("Dice Coefficient")
    axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.5)

    # 3. Min Jacobian L2R
    axs[1, 0].bar(methods, [min_jac_ants, min_jac_py, min_jac_jax, min_jac_vgg], color=colors, width=0.4)
    axs[1, 0].set_title("Min Jacobian Det (L2R) [Higher = Safer]", fontsize=11, fontweight='bold', pad=10)
    axs[1, 0].set_ylabel("Min Jacobian")
    axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 4. 2nd-Order Smoothness (Bending Energy)
    axs[1, 1].bar(methods, [smooth_2nd_ants, smooth_2nd_py, smooth_2nd_jax, smooth_2nd_vgg], color=colors, width=0.4)
    axs[1, 1].set_title("2nd-Order Smoothness (L2R) [Lower = Smoother]", fontsize=11, fontweight='bold', pad=10)
    axs[1, 1].set_ylabel("Bending Energy")
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 5. Execution Time (s)
    axs[2, 0].bar(methods, [time_ants, time_py, time_jax, time_vgg], color=colors, width=0.4)
    axs[2, 0].set_title("Execution Time [Lower = Faster]", fontsize=11, fontweight='bold', pad=10)
    axs[2, 0].set_ylabel("Seconds")
    axs[2, 0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Remove the unused 6th subplot
    fig.delaxes(axs[2, 1])
    
    plt.tight_layout()
    base64_img = plot_to_base64_fig(fig)
    return base64_img

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

def plot_edge_overlay_2d(fixed, warped):
    from skimage.feature import canny
    
    fixed_np = fixed.numpy() if isinstance(fixed, ants.core.ants_image.ANTsImage) else fixed
    warped_np = warped.numpy() if isinstance(warped, ants.core.ants_image.ANTsImage) else warped
    
    # Transpose to (H, W) for plotting (since ANTs uses (W, H) convention)
    fixed_np = fixed_np.T
    warped_np = warped_np.T
    
    # Normalize images to [0, 1] range for Canny edge detector
    f_min, f_max = fixed_np.min(), fixed_np.max()
    if f_max > f_min:
        fixed_norm = (fixed_np - f_min) / (f_max - f_min)
    else:
        fixed_norm = fixed_np.copy()
        
    w_min, w_max = warped_np.min(), warped_np.max()
    if w_max > w_min:
        warped_norm = (warped_np - w_min) / (w_max - w_min)
    else:
        warped_norm = warped_np.copy()
        
    # Detect edges of the warped image
    edges = canny(warped_norm)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(fixed_norm, cmap='gray')
    
    # Create an RGBA overlay for the red edges
    overlay = np.zeros((*fixed_norm.shape, 4))
    overlay[edges] = [1.0, 0.0, 0.0, 1.0] # Red contour
    ax.imshow(overlay)
    
    ax.axis('off')
    return plot_to_base64_fig(fig)

def main():
    print("Loading images...")
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    spacing_y, spacing_x = fi.spacing
    
    # --- 2. ANTs Registration ---
    print("Running ANTs SyN registration...")
    t0 = time.time()
    reg_ants = ants.registration(
        fi, mi, 'SyN', 
        reg_iterations=[100, 100, 100, 50], 
        syn_metric='cc', syn_sampling=2
    )
    t1 = time.time()
    ants_time = t1 - t0
    
    # Extract ANTs transforms
    ants_fwd = reg_ants['fwdtransforms'][0]
    ants_inv = reg_ants['invtransforms'][1]
    
    warped_ants_mov = reg_ants['warpedmovout']
    warped_ants_fix = reg_ants['warpedfixout']
    
    mi_ants_mov = ants.image_mutual_information(fi, warped_ants_mov)
    mi_ants_fix = ants.image_mutual_information(mi, warped_ants_fix)
    
    sampling_percent = 0.1
    
    # --- 2b. Affine-Only Registrations ---
    print("Running ANTs Affine-only registration...")
    reg_ants_affine = ants.registration(fi, mi, 'Affine')
    mi_ants_affine = ants.image_mutual_information(fi, reg_ants_affine['warpedmovout'])
    
    print("Running Syntx Affine PyTorch registration...")
    reg_py_affine = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='Affine', backend='pytorch',
        affine_iterations=[100, 50, 50, 20], mattes_bins=32,
        sampling_percentage=sampling_percent
    )
    mi_py_affine = ants.image_mutual_information(fi, reg_py_affine['warpedmovout'])
    
    print("Running Syntx Affine JAX registration...")
    reg_jax_affine = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='Affine', backend='jax',
        affine_iterations=[100, 50, 50, 20], mattes_bins=32,
        sampling_percentage=sampling_percent
    )
    mi_jax_affine = ants.image_mutual_information(fi, reg_jax_affine['warpedmovout'])
    
    # --- 2c. Registrations Initialized with ANTs Affine ---
    print("Running Syntx PyTorch SyN initialized with ANTs Affine...")
    reg_py_init = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
        initial_transform=reg_ants_affine['fwdtransforms'],
        affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
        grad_step=0.75, flow_sigma=1.732, inverse_steps=5
    )
    mi_py_init = ants.image_mutual_information(fi, reg_py_init['warpedmovout'])
    
    print("Running Syntx JAX SyN initialized with ANTs Affine...")
    reg_jax_init = syntx.syn(
        fixed=fi, moving=mi, type_of_transform='SyN', backend='jax',
        initial_transform=reg_ants_affine['fwdtransforms'],
        affine_iterations=[0], reg_iterations=[100, 100, 100, 50],
        grad_step=0.75, flow_sigma=1.732, inverse_steps=5
    )
    mi_jax_init = ants.image_mutual_information(fi, reg_jax_init['warpedmovout'])
    
    # Clean up temp transform files from initial-transform and affine-only runs
    for r in [reg_py_init, reg_jax_init, reg_py_affine, reg_jax_affine]:
        for path in r['fwdtransforms'] + r['invtransforms']:
            if os.path.exists(path):
                os.remove(path)
    
    # --- 3. Syntx SyNTo Registration (PyTorch & JAX & PyTorch VGG) ---
    print("Running Syntx SyNTo PyTorch registration (composed affine + SyN)...")
    cfl_voxels_setting = 0.70
    flow_sigma = 1.732
    aff_its = [200, 200, 200, 20]
    
    t0 = time.time()
    reg_py = syntx.syn(
        fixed=fi,
        moving=mi,
        reg_iterations=[100, 100, 100, 50],
        affine_iterations=aff_its,
        grad_step=cfl_voxels_setting,
        flow_sigma=flow_sigma,
        syn_metric='lncc',
        lncc_radius=4,
        mattes_bins=32,
        sampling_percentage=sampling_percent,
        backend='pytorch',
        inverse_steps=5
    )
    t1 = time.time()
    synto_time = t1 - t0
    
    print("Running Syntx SyNTo JAX registration (composed affine + SyN)...")
    t0 = time.time()
    reg_jax = syntx.syn(
        fixed=fi,
        moving=mi,
        reg_iterations=[100, 100, 100, 50],
        affine_iterations=aff_its,
        grad_step=cfl_voxels_setting,
        flow_sigma=flow_sigma,
        syn_metric='lncc',
        lncc_radius=4,
        mattes_bins=32,
        sampling_percentage=sampling_percent,
        backend='jax',
        inverse_steps=5
    )
    t1 = time.time()
    synto_jax_time = t1 - t0

    print("Running Syntx SyNTo PyTorch registration (composed affine + SyN) with VGG+LNCC metric...")
    t0 = time.time()
    reg_py_vgg = syntx.syn(
        fixed=fi,
        moving=mi,
        reg_iterations=[100, 100, 100, 50],
        affine_iterations=aff_its,
        grad_step=cfl_voxels_setting,
        flow_sigma=flow_sigma,
        syn_metric='vgg19',
        vgg_mode='lncc',
        vgg_layers=[2, 7],
        vgg_lncc_window_size=5,
        mattes_bins=32,
        sampling_percentage=sampling_percent,
        backend='pytorch',
        inverse_steps=5
    )
    t1 = time.time()
    vgg_time = t1 - t0
    
    # Extract the composed transform paths
    l2r_path = next(tx for tx in reg_py['fwdtransforms'] if tx.endswith('.nii.gz'))
    r2l_path = next(tx for tx in reg_py['invtransforms'] if tx.endswith('.nii.gz'))
    
    l2r_path_jax = next(tx for tx in reg_jax['fwdtransforms'] if tx.endswith('.nii.gz'))
    r2l_path_jax = next(tx for tx in reg_jax['invtransforms'] if tx.endswith('.nii.gz'))
    
    l2r_path_vgg = next(tx for tx in reg_py_vgg['fwdtransforms'] if tx.endswith('.nii.gz'))
    r2l_path_vgg = next(tx for tx in reg_py_vgg['invtransforms'] if tx.endswith('.nii.gz'))
    
    disp_l2r_np = ants.image_read(l2r_path).numpy()
    disp_r2l_np = ants.image_read(r2l_path).numpy()
    disp_l2r_np_jax = ants.image_read(l2r_path_jax).numpy()
    disp_r2l_np_jax = ants.image_read(r2l_path_jax).numpy()
    disp_l2r_np_vgg = ants.image_read(l2r_path_vgg).numpy()
    disp_r2l_np_vgg = ants.image_read(r2l_path_vgg).numpy()
    
    warped_sulceye_img_mov = reg_py['warpedmovout']
    warped_sulceye_img_fix = reg_py['warpedfixout']
    warped_sulceye_img_mov_jax = reg_jax['warpedmovout']
    warped_sulceye_img_fix_jax = reg_jax['warpedfixout']
    warped_vgg_img_mov = reg_py_vgg['warpedmovout']
    warped_vgg_img_fix = reg_py_vgg['warpedfixout']
    
    mi_sulceye_mov = ants.image_mutual_information(fi, warped_sulceye_img_mov)
    mi_sulceye_fix = ants.image_mutual_information(mi, warped_sulceye_img_fix)
    mi_sulceye_mov_jax = ants.image_mutual_information(fi, warped_sulceye_img_mov_jax)
    mi_sulceye_fix_jax = ants.image_mutual_information(mi, warped_sulceye_img_fix_jax)
    mi_vgg_mov = ants.image_mutual_information(fi, warped_vgg_img_mov)
    mi_vgg_fix = ants.image_mutual_information(mi, warped_vgg_img_fix)
    
    # Compute tissue overlaps (anatomical metric)
    print("Computing Otsu tissue overlap metrics...")
    dice_ants = compute_tissue_overlap(fi, warped_ants_mov)
    dice_py = compute_tissue_overlap(fi, warped_sulceye_img_mov)
    dice_jax = compute_tissue_overlap(fi, warped_sulceye_img_mov_jax)
    dice_vgg = compute_tissue_overlap(fi, warped_vgg_img_mov)
    
    print(f"  ANTs MI:         {mi_ants_mov:.6f} | Dice: {dice_ants:.4f}")
    print(f"  PyTorch LNCC MI: {mi_sulceye_mov:.6f} | Dice: {dice_py:.4f}")
    print(f"  JAX LNCC MI:     {mi_sulceye_mov_jax:.6f} | Dice: {dice_jax:.4f}")
    print(f"  PyTorch VGG MI:  {mi_vgg_mov:.6f} | Dice: {dice_vgg:.4f}")
    
    # --- 4. Evaluate Jacobian Folding ---
    print("Calculating Jacobian determinants for topological folding...")
    jac_ants_img = ants.create_jacobian_determinant_image(fi, reg_ants['fwdtransforms'][0])
    jac_ants_np = jac_ants_img.numpy()
    folding_ants = 100.0 * np.mean(jac_ants_np < 0)
    folding_ants_le = 100.0 * np.mean(jac_ants_np <= 0)
    min_jac_ants = float(jac_ants_np.min())
    
    jac_sulceye_img = ants.create_jacobian_determinant_image(fi, l2r_path)
    jac_sulceye_np = jac_sulceye_img.numpy()
    folding_sulceye = 100.0 * np.mean(jac_sulceye_np < 0)
    folding_sulceye_le = 100.0 * np.mean(jac_sulceye_np <= 0)
    min_jac_sulceye = float(jac_sulceye_np.min())
    
    jac_sulceye_img_jax = ants.create_jacobian_determinant_image(fi, l2r_path_jax)
    jac_sulceye_np_jax = jac_sulceye_img_jax.numpy()
    folding_sulceye_jax = 100.0 * np.mean(jac_sulceye_np_jax < 0)
    folding_sulceye_jax_le = 100.0 * np.mean(jac_sulceye_np_jax <= 0)
    min_jac_sulceye_jax = float(jac_sulceye_np_jax.min())
    
    jac_vgg_img = ants.create_jacobian_determinant_image(fi, l2r_path_vgg)
    jac_vgg_np = jac_vgg_img.numpy()
    folding_vgg = 100.0 * np.mean(jac_vgg_np < 0)
    folding_vgg_le = 100.0 * np.mean(jac_vgg_np <= 0)
    min_jac_vgg = float(jac_vgg_np.min())
    
    # ANTs displacement fields are (X, Y) ordered, transpose to (Y, X) for numpy
    disp_ants_l2r = np.transpose(ants.image_read(ants_fwd).numpy(), (1, 0, 2))
    
    # Smoothness Metrics
    smooth_1st_ants_l2r, smooth_2nd_ants_l2r = compute_smoothness_metrics(disp_ants_l2r, (spacing_y, spacing_x))
    smooth_1st_sulceye_l2r, smooth_2nd_sulceye_l2r = compute_smoothness_metrics(disp_l2r_np, (spacing_y, spacing_x))
    smooth_1st_sulceye_l2r_jax, smooth_2nd_sulceye_l2r_jax = compute_smoothness_metrics(disp_l2r_np_jax, (spacing_y, spacing_x))
    smooth_1st_vgg_l2r, smooth_2nd_vgg_l2r = compute_smoothness_metrics(disp_l2r_np_vgg, (spacing_y, spacing_x))
    
    print("Generating report visualizations (grids, jacobians, edges)...")
    # ANTs
    ants_grid_b64 = plot_warp_grid_2d(ants.image_read(ants_fwd).numpy(), fi.spacing, fi.origin, fi.direction, title="ANTs Warp Grid")
    ants_jac_b64 = plot_jacobian_slice(jac_ants_np, fi.spacing, fi.origin, fi.direction, title="ANTs Jacobian")
    ants_edge_b64 = plot_edge_overlay_2d(fi, warped_ants_mov)
    
    # PyTorch LNCC
    py_grid_b64 = plot_warp_grid_2d(disp_l2r_np, fi.spacing, fi.origin, fi.direction, title="PyTorch LNCC Warp Grid")
    py_jac_b64 = plot_jacobian_slice(jac_sulceye_np, fi.spacing, fi.origin, fi.direction, title="PyTorch LNCC Jacobian")
    py_edge_b64 = plot_edge_overlay_2d(fi, warped_sulceye_img_mov)
    
    # JAX LNCC
    jax_grid_b64 = plot_warp_grid_2d(disp_l2r_np_jax, fi.spacing, fi.origin, fi.direction, title="JAX LNCC Warp Grid")
    jax_jac_b64 = plot_jacobian_slice(jac_sulceye_np_jax, fi.spacing, fi.origin, fi.direction, title="JAX LNCC Jacobian")
    jax_edge_b64 = plot_edge_overlay_2d(fi, warped_sulceye_img_mov_jax)
    
    # PyTorch VGG
    vgg_grid_b64 = plot_warp_grid_2d(disp_l2r_np_vgg, fi.spacing, fi.origin, fi.direction, title="PyTorch VGG Warp Grid")
    vgg_jac_b64 = plot_jacobian_slice(jac_vgg_np, fi.spacing, fi.origin, fi.direction, title="PyTorch VGG Jacobian")
    vgg_edge_b64 = plot_edge_overlay_2d(fi, warped_vgg_img_mov)
    
    # Clean up temp transform files from registration runs
    for r in [reg_py, reg_jax, reg_py_vgg]:
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

    axs[2].plot(reg_py_vgg['syn_losses'], color='purple', label='PyTorch VGG')
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
        mi_ants_mov, mi_sulceye_mov, mi_sulceye_mov_jax, mi_vgg_mov,
        dice_ants, dice_py, dice_jax, dice_vgg,
        min_jac_ants, min_jac_sulceye, min_jac_sulceye_jax, min_jac_vgg,
        smooth_2nd_ants_l2r, smooth_2nd_sulceye_l2r, smooth_2nd_sulceye_l2r_jax, smooth_2nd_vgg_l2r,
        ants_time, synto_time, synto_jax_time, vgg_time
    )
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ANTs vs Syntx SyN Comparison</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f4f4f9; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
            .panel {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex: 1; min-width: 250px; }}
            .panel img {{ max-width: 100%; border-radius: 4px; }}
            pre {{ background: #282c34; color: #abb2bf; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 14px; line-height: 1.5; }}
            .metric-box {{ background: #eef2f5; padding: 10px 20px; border-radius: 5px; font-size: 16px; font-weight: bold; margin-top: 15px; display: inline-block; color: #2c3e50; border-left: 4px solid #3498db; margin-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #34495e; color: white; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>SyN Alignment Algorithm Comparison (Syntx vs ANTs)</h1>
        <p>This report documents the performance of classic ANTs SyN compared to PyTorch and JAX backends of the Syntx package.</p>
        
        <h2>1. The Raw Input Data (r16 vs r64)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Fixed Target (r16)</h3>
                <img src="{image_to_base64(fi)}" />
            </div>
            <div class="panel">
                <h3>Moving Source (r64)</h3>
                <img src="{image_to_base64(mi)}" />
            </div>
        </div>

        <h2>2. Affine-Only Registration Performance</h2>
        <div class="grid">
            <div class="panel" style="text-align: left;">
                <h3>Classic ANTs Affine</h3>
                <div class="metric-box">MI: {mi_ants_affine:.6f}</div>
            </div>
            <div class="panel" style="text-align: left;">
                <h3>PyTorch Affine (Syntx)</h3>
                <div class="metric-box">MI: {mi_py_affine:.6f}</div>
            </div>
            <div class="panel" style="text-align: left;">
                <h3>JAX Affine (Syntx)</h3>
                <div class="metric-box">MI: {mi_jax_affine:.6f}</div>
            </div>
        </div>

        <h2>3. Classic ANTs Registration (Reference)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Fixed Target</h3>
                <img src="{image_to_base64(fi)}" />
            </div>
            <div class="panel">
                <h3>Warped Moving (L2R)</h3>
                <img src="{image_to_base64(warped_ants_mov)}" />
                <br>
                <div class="metric-box">MI: {mi_ants_mov:.4f}</div>
                <div class="metric-box">Tissue Overlap (Dice): {dice_ants:.4f}</div>
                <div class="metric-box">Folding: {folding_ants:.4f}%</div>
                <div class="metric-box">Min Jac: {min_jac_ants:.4f}</div>
            </div>
            <div class="panel">
                <h3>Edge Overlay</h3>
                <img src="{ants_edge_b64}" />
            </div>
            <div class="panel">
                <h3>Warp Grid</h3>
                <img src="{ants_grid_b64}" />
            </div>
            <div class="panel">
                <h3>Jacobian Det</h3>
                <img src="{ants_jac_b64}" />
            </div>
        </div>

        <h2>4. Syntx PyTorch Registration (SyNTo composed with LNCC)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Fixed Target</h3>
                <img src="{image_to_base64(fi)}" />
            </div>
            <div class="panel">
                <h3>Warped Moving (L2R) - PyTorch LNCC</h3>
                <img src="{image_to_base64(warped_sulceye_img_mov)}" />
                <br>
                <div class="metric-box">MI: {mi_sulceye_mov:.4f}</div>
                <div class="metric-box">Tissue Overlap (Dice): {dice_py:.4f}</div>
                <div class="metric-box">Folding: {folding_sulceye:.4f}%</div>
                <div class="metric-box">Min Jac: {min_jac_sulceye:.4f}</div>
            </div>
            <div class="panel">
                <h3>Edge Overlay</h3>
                <img src="{py_edge_b64}" />
            </div>
            <div class="panel">
                <h3>Warp Grid</h3>
                <img src="{py_grid_b64}" />
            </div>
            <div class="panel">
                <h3>Jacobian Det</h3>
                <img src="{py_jac_b64}" />
            </div>
        </div>

        <h2>5. Syntx JAX Registration (SyNTo composed with LNCC)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Fixed Target</h3>
                <img src="{image_to_base64(fi)}" />
            </div>
            <div class="panel">
                <h3>Warped Moving (L2R) - JAX LNCC</h3>
                <img src="{image_to_base64(warped_sulceye_img_mov_jax)}" />
                <br>
                <div class="metric-box">MI: {mi_sulceye_mov_jax:.4f}</div>
                <div class="metric-box">Tissue Overlap (Dice): {dice_jax:.4f}</div>
                <div class="metric-box">Folding: {folding_sulceye_jax:.4f}%</div>
                <div class="metric-box">Min Jac: {min_jac_sulceye_jax:.4f}</div>
            </div>
            <div class="panel">
                <h3>Edge Overlay</h3>
                <img src="{jax_edge_b64}" />
            </div>
            <div class="panel">
                <h3>Warp Grid</h3>
                <img src="{jax_grid_b64}" />
            </div>
            <div class="panel">
                <h3>Jacobian Det</h3>
                <img src="{jax_jac_b64}" />
            </div>
        </div>

        <h2>6. Syntx PyTorch Registration (SyNTo composed with VGG+LNCC)</h2>
        <div class="grid">
            <div class="panel">
                <h3>Fixed Target</h3>
                <img src="{image_to_base64(fi)}" />
            </div>
            <div class="panel">
                <h3>Warped Moving (L2R) - PyTorch VGG</h3>
                <img src="{image_to_base64(warped_vgg_img_mov)}" />
                <br>
                <div class="metric-box">MI: {mi_vgg_mov:.4f}</div>
                <div class="metric-box">Tissue Overlap (Dice): {dice_vgg:.4f}</div>
                <div class="metric-box">Folding: {folding_vgg:.4f}%</div>
                <div class="metric-box">Min Jac: {min_jac_vgg:.4f}</div>
            </div>
            <div class="panel">
                <h3>Edge Overlay</h3>
                <img src="{vgg_edge_b64}" />
            </div>
            <div class="panel">
                <h3>Warp Grid</h3>
                <img src="{vgg_grid_b64}" />
            </div>
            <div class="panel">
                <h3>Jacobian Det</h3>
                <img src="{vgg_jac_b64}" />
            </div>
        </div>

        <h2>7. Metrics and Timing Summary</h2>
        <div style="background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 20px;">
            <img src="{comparison_plot_base64}" style="width: 100%; max-width: 900px;" />
        </div>
    </body>
    </html>
    """
    
    output_dir = "/Users/stnava/code/syntx/docs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "parity_report_2d.html")
    with open(output_path, "w") as f:
        f.write(html_content)
        
    print(f"Report generated successfully at {output_path}")

if __name__ == "__main__":
    main()
