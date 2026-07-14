import os
import sys
import argparse
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
from syntx import SyNTo, SyNToJax


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
    
    # Match ANTs slice reorientation logic
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


def save_warped_grid_overlay(t1_slice, displ_slice, spacing, filename, direction=None, origin=None, is_3d=False):
    """Saves a continuous 2D warped coordinate grid overlaid on the grayscale target T1w."""
    if direction is not None:
        t1_slice = np.flipud(t1_slice)
        displ_slice = np.flipud(displ_slice)
        H, W = t1_slice.shape
        grid_r, grid_c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        if is_3d:
            D = np.array(direction)
            D1 = D[:, 0]
            D2 = D[:, 1]
            
            X_phys = spacing[0] * grid_c
            Y_phys = spacing[1] * grid_r
            
            du_proj = np.dot(displ_slice, D1)
            dv_proj = np.dot(displ_slice, D2)
            
            X_deformed = X_phys + du_proj
            Y_deformed = Y_phys + dv_proj
        else:
            D = np.array(direction)
            S = np.array(spacing)
            O = np.array(origin) if origin is not None else np.zeros(2)
            
            X_phys = D[0, 0] * S[0] * grid_c + D[0, 1] * S[1] * grid_r + O[0]
            Y_phys = D[1, 0] * S[0] * grid_c + D[1, 1] * S[1] * grid_r + O[1]
            
            X_deformed = X_phys + displ_slice[..., 0]
            Y_deformed = Y_phys + displ_slice[..., 1]
            
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#08080a')
        ax.pcolormesh(X_phys, Y_phys, t1_slice, cmap='gray', shading='auto', zorder=1)
        
        grid_step = 15
        for x_c in range(0, W, grid_step):
            ax.plot(X_deformed[:, x_c], Y_deformed[:, x_c], color='#55efc4', linewidth=0.8, alpha=0.8, zorder=2)
            
        for y_c in range(0, H, grid_step):
            ax.plot(X_deformed[y_c, :], Y_deformed[y_c, :], color='#55efc4', linewidth=0.8, alpha=0.8, zorder=2)
            
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return

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


def save_jacobian_slice(t1_slice, jac_slice, filename, direction=None, spacing=None, origin=None, is_3d=False):
    """Saves a colorbar-enabled Jacobian determinant overlay over the T1w image."""
    if direction is not None:
        t1_slice = np.flipud(t1_slice)
        jac_slice = np.flipud(jac_slice)
        H, W = t1_slice.shape
        grid_r, grid_c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        if is_3d:
            X_phys = spacing[0] * grid_c
            Y_phys = spacing[1] * grid_r
        else:
            D = np.array(direction)
            S = np.array(spacing)
            O = np.array(origin) if origin is not None else np.zeros(2)
            
            X_phys = D[0, 0] * S[0] * grid_c + D[0, 1] * S[1] * grid_r + O[0]
            Y_phys = D[1, 0] * S[0] * grid_c + D[1, 1] * S[1] * grid_r + O[1]
            
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='#08080a')
        ax.pcolormesh(X_phys, Y_phys, t1_slice, cmap='gray', shading='auto', zorder=1)
        
        masked_jac = np.ma.masked_where(np.abs(jac_slice - 1.0) < 0.03, jac_slice)
        im = ax.pcolormesh(X_phys, Y_phys, masked_jac, cmap='bwr', alpha=0.6, vmin=0.5, vmax=1.5, shading='auto', zorder=2)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Jacobian Determinant', color='white')
        plt.tight_layout()
        fig.savefig(filename, dpi=120, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return

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
        description="Compare SyN registration backends (ANTs, PyTorch, JAX) on 3D or 2D images and evaluate DKT DICE overlap."
    )
    parser.add_argument("-f", "--fixed", type=str, required=True, help="Path to fixed/target subject image")
    parser.add_argument("-m", "--moving", type=str, required=True, help="Path to moving subject image")
    parser.add_argument("-fd", "--fixed-dkt", type=str, default=None, help="Path to fixed subject DKT label map")
    parser.add_argument("-md", "--moving-dkt", type=str, default=None, help="Path to moving subject DKT label map")
    parser.add_argument("-o", "--output-dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("-r", "--report-name", type=str, default="registration_report.html", help="HTML report output file name")
    
    parser.add_argument("--epochs-per-level", type=int, nargs='+', default=[100, 200, 100, 0], help="Epochs per level for SyN deformable model")
    parser.add_argument("--affine-epochs", type=int, nargs='+', default=[400, 200, 100, 0], help="Epochs per level for Affine model")
    parser.add_argument("--levels", type=int, nargs='+', default=[8, 4, 2, 1], help="Resolution levels")
    parser.add_argument("--similarity-metric", type=str, default="mattes_mi", choices=["mattes_mi", "lncc", "meansquares"], help="Similarity metric")
    parser.add_argument("--sampling-percentage", type=float, default=0.10, help="Sampling percentage for the metric")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch training device")
    
    parser.add_argument("--no-ants", action="store_false", dest="run_ants", help="Disable ANTs registration comparison")
    parser.add_argument("--no-jax", action="store_false", dest="run_jax", help="Disable JAX backend registration comparison")
    
    args = parser.parse_args()

    print("=== Starting Inter-Subject Registration and Comparison ===")
    
    # Load Input Images
    print(f"Loading Fixed Subject: {args.fixed}")
    fixed_img = ants.image_read(args.fixed)
    dim = fixed_img.dimension
    
    print(f"Loading Moving Subject: {args.moving}")
    moving_img = ants.image_read(args.moving)
    
    # Load DKT maps if specified
    fixed_dkt = None
    moving_dkt = None
    if args.fixed_dkt and args.moving_dkt:
        print(f"Loading Fixed Subject DKT: {args.fixed_dkt}")
        fixed_dkt = ants.image_read(args.fixed_dkt)
        print(f"Loading Moving Subject DKT: {args.moving_dkt}")
        moving_dkt = ants.image_read(args.moving_dkt)
    else:
        print("DKT label maps not provided; skipping multi-region overlap DICE evaluation.")

    # --- Baseline Evaluation ---
    base_dices = None
    base_mean = 0.0
    if fixed_dkt is not None and moving_dkt is not None:
        print("\n--- Baseline (Unregistered) ---")
        base_dices = compute_multiregion_dice(fixed_dkt, moving_dkt)
        base_mean = np.mean(list(base_dices.values()))
        print(f"Mean DKT DICE Overlap: {base_mean:.4f}")

    # --- ANTs SyN Registration ---
    ants_reg = None
    warped_dkt_ants_affine = None
    ants_affine_dices = None
    warped_dkt_ants = None
    ants_dices = None
    ants_mean = 0.0
    warped_t1_ants_affine = None
    warped_t1_ants = None
    
    if args.run_ants:
        print("\n--- ANTs SyN Registration ---")
        reg_its = (100, 200, 100, 0) if dim == 3 else (100, 100, 100, 0)
        ants_reg = ants.registration(
            fixed=fixed_img, 
            moving=moving_img, 
            type_of_transform='SyN',
            reg_iterations=reg_its
        )
        
        warped_t1_ants_affine = ants.apply_transforms(
            fixed=fixed_img,
            moving=moving_img,
            transformlist=ants_reg['fwdtransforms'][1:]  # Affine only
        )
        warped_t1_ants = ants_reg['warpedmovout']
        
        if moving_dkt is not None:
            warped_dkt_ants_affine = ants.apply_transforms(
                fixed=fixed_img,
                moving=moving_dkt,
                transformlist=ants_reg['fwdtransforms'][1:], # Affine only
                interpolator='nearestNeighbor'
            )
            ants_affine_dices = compute_multiregion_dice(fixed_dkt, warped_dkt_ants_affine)
            print(f"Mean DKT DICE Overlap (ANTs Affine): {np.mean(list(ants_affine_dices.values())):.4f}")
            
            warped_dkt_ants = ants.apply_transforms(
                fixed=fixed_img,
                moving=moving_dkt,
                transformlist=ants_reg['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            ants_dices = compute_multiregion_dice(fixed_dkt, warped_dkt_ants)
            ants_mean = np.mean(list(ants_dices.values()))
            print(f"Mean DKT DICE Overlap (ANTs SyN): {ants_mean:.4f}")

    # --- SyNTo (PyTorch) Registration ---
    print("\n--- SyNTo (PyTorch) Registration ---")
    device = torch.device(args.device)
    
    fixed_tensor = torch.from_numpy(fixed_img.numpy()).unsqueeze(0).unsqueeze(0).to(device)
    moving_tensor = torch.from_numpy(moving_img.numpy()).unsqueeze(0).unsqueeze(0).to(device)
    
    engine = SyNTo(dim=dim, grid_shape=fixed_img.shape, elastic_sigma=0.0, transform_type='Affine').to(device)
    
    print("Fitting SyNTo (PyTorch)...")
    engine.fit(
        fixed_tensor, 
        moving_tensor, 
        levels=args.levels,
        epochs_per_level=args.epochs_per_level,
        affine_epochs=args.affine_epochs,
        similarity_metric=args.similarity_metric,
        sampling_percentage=args.sampling_percentage
    )
    
    print("Extracting SyNTo (PyTorch) Transform Object...")
    metadata = {'origin': fixed_img.origin, 'spacing': fixed_img.spacing, 'direction': fixed_img.direction}
    tx_fwd = engine.get_forward_transform(metadata)
    
    pt_affine_mean = 0.0
    pt_mean = 0.0
    warped_dkt_pt_affine_img = None
    warped_dkt_img_pt = None
    pt_affine_dices = None
    pt_dices = None
    
    if moving_dkt is not None:
        print("Warping moving DKT and evaluating overlaps...")
        moving_dkt_tensor = torch.from_numpy(moving_dkt.numpy()).unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            warped_dkt_tensor = tx_fwd.apply(moving_dkt_tensor, mode='nearest')
            grid_affine = engine.get_affine_grid(moving_dkt_tensor.shape[2:], device)
            warped_dkt_pt_affine = F.grid_sample(moving_dkt_tensor, grid_affine, mode='nearest', padding_mode='border', align_corners=True)
            
        warped_dkt_pt_affine_np = warped_dkt_pt_affine.squeeze().cpu().numpy()
        warped_dkt_pt_affine_img = fixed_img.new_image_like(warped_dkt_pt_affine_np)
        pt_affine_dices = compute_multiregion_dice(fixed_dkt, warped_dkt_pt_affine_img)
        pt_affine_mean = np.mean(list(pt_affine_dices.values()))
            
        warped_dkt_pt = warped_dkt_tensor.squeeze().cpu().numpy()
        warped_dkt_img_pt = fixed_img.new_image_like(warped_dkt_pt)
        pt_dices = compute_multiregion_dice(fixed_dkt, warped_dkt_img_pt)
        pt_mean = np.mean(list(pt_dices.values()))
        
        print(f"Mean DKT DICE Overlap (SyNTo PyTorch Affine): {pt_affine_mean:.4f}")
        print(f"Mean DKT DICE Overlap (SyNTo PyTorch SyN): {pt_mean:.4f}")

    # Generate Warped T1 Images for PyTorch
    with torch.no_grad():
        grid_affine = engine.get_affine_grid(moving_tensor.shape[2:], device)
        warped_t1_pt_affine = F.grid_sample(moving_tensor, grid_affine, mode='bilinear', padding_mode='border', align_corners=True)
        warped_t1_pt_tensor = tx_fwd.apply(moving_tensor, mode='bilinear')
        
    warped_t1_pt_affine_img = fixed_img.new_image_like(warped_t1_pt_affine.squeeze().cpu().numpy())
    warped_t1_pt_img = fixed_img.new_image_like(warped_t1_pt_tensor.squeeze().cpu().numpy())

    # --- SyNTo (JAX) Registration ---
    engine_jax = None
    tx_fwd_jax = None
    jax_affine_mean = 0.0
    jax_mean = 0.0
    warped_dkt_jax_affine_img = None
    warped_dkt_img_jax = None
    jax_affine_dices = None
    jax_dices = None
    warped_t1_jax_affine_img = None
    warped_t1_jax_img = None
    
    if args.run_jax:
        print("\n--- SyNTo (JAX) Registration ---")
        engine_jax = SyNToJax(dim=dim, grid_shape=fixed_img.shape, elastic_sigma=0.0, transform_type='Affine')
        
        print("Fitting SyNTo (JAX)...")
        engine_jax.fit(
            fixed_tensor, 
            moving_tensor, 
            levels=args.levels,
            epochs_per_level=args.epochs_per_level,
            affine_epochs=args.affine_epochs,
            similarity_metric=args.similarity_metric,
            sampling_percentage=args.similarity_metric
        )
        
        print("Extracting SyNTo (JAX) Transform Object...")
        tx_fwd_jax = engine_jax.get_forward_transform(metadata).to(device)
        
        if moving_dkt is not None:
            print("Warping moving DKT for JAX...")
            moving_dkt_tensor = torch.from_numpy(moving_dkt.numpy()).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad():
                warped_dkt_tensor_jax = tx_fwd_jax.apply(moving_dkt_tensor, mode='nearest')
                grid_affine_jax = engine_jax.get_affine_grid(moving_dkt_tensor.shape[2:], device)
                warped_dkt_jax_affine = F.grid_sample(moving_dkt_tensor, grid_affine_jax, mode='nearest', padding_mode='border', align_corners=True)
                
            warped_dkt_jax_affine_np = warped_dkt_jax_affine.squeeze().cpu().numpy()
            warped_dkt_jax_affine_img = fixed_img.new_image_like(warped_dkt_jax_affine_np)
            jax_affine_dices = compute_multiregion_dice(fixed_dkt, warped_dkt_jax_affine_img)
            jax_affine_mean = np.mean(list(jax_affine_dices.values()))
                
            warped_dkt_jax_np = warped_dkt_tensor_jax.squeeze().cpu().numpy()
            warped_dkt_img_jax = fixed_img.new_image_like(warped_dkt_jax_np)
            jax_dices = compute_multiregion_dice(fixed_dkt, warped_dkt_img_jax)
            jax_mean = np.mean(list(jax_dices.values()))
            
            print(f"Mean DKT DICE Overlap (SyNTo JAX Affine): {jax_affine_mean:.4f}")
            print(f"Mean DKT DICE Overlap (SyNTo JAX SyN): {jax_mean:.4f}")
            
        with torch.no_grad():
            grid_affine_jax = engine_jax.get_affine_grid(moving_tensor.shape[2:], device)
            warped_t1_jax_affine = F.grid_sample(moving_tensor, grid_affine_jax, mode='bilinear', padding_mode='border', align_corners=True)
            warped_t1_jax_tensor = tx_fwd_jax.apply(moving_tensor, mode='bilinear')
            
        warped_t1_jax_affine_img = fixed_img.new_image_like(warped_t1_jax_affine.squeeze().cpu().numpy())
        warped_t1_jax_img = fixed_img.new_image_like(warped_t1_jax_tensor.squeeze().cpu().numpy())

    # --- Generate Diagnostic Visuals ---
    print("\n--- Generating Diagnostic Visuals ---")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Export Transformation parameters
    tx_fwd.export_classic(os.path.join(out_dir, 'pt_'))
    tx_fwd.to_composite_warp(os.path.join(out_dir, 'pt_CompositeWarp.nii.gz'))
    
    if args.run_jax:
        tx_fwd_jax.export_classic(os.path.join(out_dir, 'jax_'))
        tx_fwd_jax.to_composite_warp(os.path.join(out_dir, 'jax_CompositeWarp.nii.gz'))

    # 2. Slice Coordinates & Extraction
    if dim == 3:
        fixed_lai = fixed_img.reorient_image2("LAI")
        z_mid = int(fixed_lai.shape[2] * 0.55)
        
        fixed_t1_slice = get_lai_slice(fixed_img, 2, z_mid)
        base_t1_slice = get_lai_slice(moving_img, 2, z_mid)
        pt_aff_t1_slice = get_lai_slice(warped_t1_pt_affine_img, 2, z_mid)
        pt_syn_t1_slice = get_lai_slice(warped_t1_pt_img, 2, z_mid)
        
        ants_aff_t1_slice = get_lai_slice(warped_t1_ants_affine, 2, z_mid) if args.run_ants else None
        ants_syn_t1_slice = get_lai_slice(warped_t1_ants, 2, z_mid) if args.run_ants else None
        
        jax_aff_t1_slice = get_lai_slice(warped_t1_jax_affine_img, 2, z_mid) if args.run_jax else None
        jax_syn_t1_slice = get_lai_slice(warped_t1_jax_img, 2, z_mid) if args.run_jax else None
        
        base_dkt_slice = get_lai_slice(moving_dkt, 2, z_mid) if moving_dkt is not None else None
        pt_aff_dkt_slice = get_lai_slice(warped_dkt_pt_affine_img, 2, z_mid) if moving_dkt is not None else None
        pt_syn_dkt_slice = get_lai_slice(warped_dkt_img_pt, 2, z_mid) if moving_dkt is not None else None
        
        ants_aff_dkt_slice = get_lai_slice(warped_dkt_ants_affine, 2, z_mid) if (moving_dkt is not None and args.run_ants) else None
        ants_syn_dkt_slice = get_lai_slice(warped_dkt_ants, 2, z_mid) if (moving_dkt is not None and args.run_ants) else None
        
        jax_aff_dkt_slice = get_lai_slice(warped_dkt_jax_affine_img, 2, z_mid) if (moving_dkt is not None and args.run_jax) else None
        jax_syn_dkt_slice = get_lai_slice(warped_dkt_img_jax, 2, z_mid) if (moving_dkt is not None and args.run_jax) else None
        
        spacing_info = fixed_lai.spacing
    else:
        # dim == 2
        fixed_t1_slice = fixed_img.numpy()
        base_t1_slice = moving_img.numpy()
        pt_aff_t1_slice = warped_t1_pt_affine_img.numpy()
        pt_syn_t1_slice = warped_t1_pt_img.numpy()
        
        ants_aff_t1_slice = warped_t1_ants_affine.numpy() if args.run_ants else None
        ants_syn_t1_slice = warped_t1_ants.numpy() if args.run_ants else None
        
        jax_aff_t1_slice = warped_t1_jax_affine_img.numpy() if args.run_jax else None
        jax_syn_t1_slice = warped_t1_jax_img.numpy() if args.run_jax else None
        
        base_dkt_slice = moving_dkt.numpy() if moving_dkt is not None else None
        pt_aff_dkt_slice = warped_dkt_pt_affine_img.numpy() if moving_dkt is not None else None
        pt_syn_dkt_slice = warped_dkt_img_pt.numpy() if moving_dkt is not None else None
        
        ants_aff_dkt_slice = warped_dkt_ants_affine.numpy() if (moving_dkt is not None and args.run_ants) else None
        ants_syn_dkt_slice = warped_dkt_ants.numpy() if (moving_dkt is not None and args.run_ants) else None
        
        jax_aff_dkt_slice = warped_dkt_jax_affine_img.numpy() if (moving_dkt is not None and args.run_jax) else None
        jax_syn_dkt_slice = warped_dkt_img_jax.numpy() if (moving_dkt is not None and args.run_jax) else None
        
        spacing_info = fixed_img.spacing

    # 3. Label overlays
    label_paths = {}
    if moving_dkt is not None:
        label_paths['base'] = os.path.join(out_dir, 'lbl_base.png')
        save_label_overlay_plot(fixed_t1_slice, base_dkt_slice, label_paths['base'])
        
        label_paths['pt_aff'] = os.path.join(out_dir, 'lbl_pt_aff.png')
        save_label_overlay_plot(fixed_t1_slice, pt_aff_dkt_slice, label_paths['pt_aff'])
        
        label_paths['pt_syn'] = os.path.join(out_dir, 'lbl_pt_syn.png')
        save_label_overlay_plot(fixed_t1_slice, pt_syn_dkt_slice, label_paths['pt_syn'])
        
        if args.run_ants:
            label_paths['ants_aff'] = os.path.join(out_dir, 'lbl_ants_aff.png')
            save_label_overlay_plot(fixed_t1_slice, ants_aff_dkt_slice, label_paths['ants_aff'])
            
            label_paths['ants_syn'] = os.path.join(out_dir, 'lbl_ants_syn.png')
            save_label_overlay_plot(fixed_t1_slice, ants_syn_dkt_slice, label_paths['ants_syn'])
            
        if args.run_jax:
            label_paths['jax_aff'] = os.path.join(out_dir, 'lbl_jax_aff.png')
            save_label_overlay_plot(fixed_t1_slice, jax_aff_dkt_slice, label_paths['jax_aff'])
            
            label_paths['jax_syn'] = os.path.join(out_dir, 'lbl_jax_syn.png')
            save_label_overlay_plot(fixed_t1_slice, jax_syn_dkt_slice, label_paths['jax_syn'])

    # 4. Edge Overlays
    edge_paths = {
        'base': os.path.join(out_dir, 'edge_base.png'),
        'pt_aff': os.path.join(out_dir, 'edge_pt_aff.png'),
        'pt_syn': os.path.join(out_dir, 'edge_pt_syn.png')
    }
    save_edge_overlay_plot(fixed_t1_slice, base_t1_slice, edge_paths['base'])
    save_edge_overlay_plot(fixed_t1_slice, pt_aff_t1_slice, edge_paths['pt_aff'])
    save_edge_overlay_plot(fixed_t1_slice, pt_syn_t1_slice, edge_paths['pt_syn'])
    
    if args.run_ants:
        edge_paths['ants_aff'] = os.path.join(out_dir, 'edge_ants_aff.png')
        save_edge_overlay_plot(fixed_t1_slice, ants_aff_t1_slice, edge_paths['ants_aff'])
        
        edge_paths['ants_syn'] = os.path.join(out_dir, 'edge_ants_syn.png')
        save_edge_overlay_plot(fixed_t1_slice, ants_syn_t1_slice, edge_paths['ants_syn'])
        
    if args.run_jax:
        edge_paths['jax_aff'] = os.path.join(out_dir, 'edge_jax_aff.png')
        save_edge_overlay_plot(fixed_t1_slice, jax_aff_t1_slice, edge_paths['jax_aff'])
        
        edge_paths['jax_syn'] = os.path.join(out_dir, 'edge_jax_syn.png')
        save_edge_overlay_plot(fixed_t1_slice, jax_syn_t1_slice, edge_paths['jax_syn'])

    # 5. Jacobian Determinant Maps
    jac_np = tx_fwd.get_jacobian_determinant()
    folds = np.sum(jac_np <= 0) / jac_np.size * 100
    print(f"\n--- Topology Diagnostics ---")
    print(f"SyNTo PyTorch Folding Rate (J <= 0): {folds:.4f}%")
    
    if dim == 3:
        jac_img = fixed_img.new_image_like(jac_np)
        pt_jac_slice = get_lai_slice(jac_img, 2, z_mid)
    else:
        pt_jac_slice = jac_np
        
    pt_jac_png = os.path.join(out_dir, 'pt_jacobian.png')
    save_jacobian_slice(fixed_t1_slice, pt_jac_slice, pt_jac_png)
    
    folds_jax = 0.0
    jax_jac_png = None
    if args.run_jax:
        jac_jax_np = tx_fwd_jax.get_jacobian_determinant()
        folds_jax = np.sum(jac_jax_np <= 0) / jac_jax_np.size * 100
        print(f"SyNTo JAX Folding Rate (J <= 0): {folds_jax:.4f}%")
        
        if dim == 3:
            jac_jax_img = fixed_img.new_image_like(jac_jax_np)
            jax_jac_slice = get_lai_slice(jac_jax_img, 2, z_mid)
        else:
            jax_jac_slice = jac_jax_np
            
        jax_jac_png = os.path.join(out_dir, 'jax_jacobian.png')
        save_jacobian_slice(fixed_t1_slice, jax_jac_slice, jax_jac_png)

    # 6. Deformed Grid Maps
    comp_warp_path = os.path.join(out_dir, 'pt_CompositeWarp.nii.gz')
    comp_warp_img = ants.image_read(comp_warp_path)
    displ_slice = get_lai_slice(comp_warp_img, 2, z_mid) if dim == 3 else comp_warp_img.numpy()
    pt_grid_png = os.path.join(out_dir, 'pt_deformed_grid.png')
    save_warped_grid_overlay(fixed_t1_slice, displ_slice, spacing_info, pt_grid_png)
    
    jax_grid_png = None
    if args.run_jax:
        comp_warp_path_jax = os.path.join(out_dir, 'jax_CompositeWarp.nii.gz')
        comp_warp_img_jax = ants.image_read(comp_warp_path_jax)
        displ_slice_jax = get_lai_slice(comp_warp_img_jax, 2, z_mid) if dim == 3 else comp_warp_img_jax.numpy()
        jax_grid_png = os.path.join(out_dir, 'jax_deformed_grid.png')
        save_warped_grid_overlay(fixed_t1_slice, displ_slice_jax, spacing_info, jax_grid_png)

    # Output Summary
    print("\n=== Summary ===")
    if moving_dkt is not None:
        print(f"Baseline:     {base_mean:.4f}")
        if args.run_ants:
            print(f"ANTs Affine:  {np.mean(list(ants_affine_dices.values())):.4f}")
            print(f"ANTs SyN:     {ants_mean:.4f}")
        print(f"PT Affine:    {pt_affine_mean:.4f}")
        print(f"PT SyNTo:     {pt_mean:.4f}")
        if args.run_jax:
            print(f"JAX Affine:   {jax_affine_mean:.4f}")
            print(f"JAX SyNTo:    {jax_mean:.4f}")
    else:
        print("DICE overlap evaluation skipped (no DKT labels provided).")
    print(f"PT Folds:     {folds:.4f}%")
    if args.run_jax:
        print(f"JAX Folds:    {folds_jax:.4f}%")

    # Generate visual HTML report
    report_path = os.path.join(out_dir, args.report_name)
    generate_phase4_report(
        base_dices, ants_affine_dices, pt_affine_dices, jax_affine_dices,
        ants_dices, pt_dices, jax_dices,
        label_paths, edge_paths,
        pt_jac_png, pt_grid_png, jax_jac_png, jax_grid_png,
        engine.affine_losses, engine.syn_losses,
        engine_jax.affine_losses if args.run_jax else [], engine_jax.syn_losses if args.run_jax else [],
        report_path
    )


def generate_phase4_report(
    base_dices, ants_affine_dices, pt_affine_dices, jax_affine_dices,
    ants_dices, pt_dices, jax_dices, 
    label_paths, edge_paths,
    pt_jac_png, pt_grid_png, jax_jac_png, jax_grid_png,
    pt_affine_losses, pt_syn_losses,
    jax_affine_losses, jax_syn_losses,
    output_path
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import base64
    
    has_dkt = base_dices is not None
    run_ants = ants_dices is not None
    run_jax = jax_dices is not None

    def get_b64(path):
        if not path or not os.path.exists(path): return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    barplot_html = ""
    quantitative_summary_card = ""
    
    if has_dkt:
        labels = sorted(list(base_dices.keys()))
        df_dict = {
            'Label': labels,
            'Baseline': [base_dices[l] for l in labels],
            'SyNTo_PT_Affine': [pt_affine_dices.get(l, 0) for l in labels],
            'SyNTo_PT_SyN': [pt_dices.get(l, 0) for l in labels],
        }
        if run_ants:
            df_dict['ANTs_Affine'] = [ants_affine_dices.get(l, 0) for l in labels]
            df_dict['ANTs_SyN'] = [ants_dices.get(l, 0) for l in labels]
        if run_jax:
            df_dict['SyNTo_JAX_Affine'] = [jax_affine_dices.get(l, 0) for l in labels]
            df_dict['SyNTo_JAX_SyN'] = [jax_dices.get(l, 0) for l in labels]
            
        df = pd.DataFrame(df_dict)
        
        # Build grouped DICE barplot
        fig_data = [
            go.Bar(name='Baseline', x=[f"Region {int(l)}" for l in df['Label']], y=df['Baseline'], marker_color='#7f8c8d'),
        ]
        if run_ants:
            fig_data.append(go.Bar(name='ANTs Affine', x=[f"Region {int(l)}" for l in df['Label']], y=df['ANTs_Affine'], marker_color='#ff7675'))
        fig_data.append(go.Bar(name='AffineTo PT', x=[f"Region {int(l)}" for l in df['Label']], y=df['SyNTo_PT_Affine'], marker_color='#55efc4'))
        if run_jax:
            fig_data.append(go.Bar(name='AffineTo JAX', x=[f"Region {int(l)}" for l in df['Label']], y=df['SyNTo_JAX_Affine'], marker_color='#74b9ff'))
        if run_ants:
            fig_data.append(go.Bar(name='ANTs SyN', x=[f"Region {int(l)}" for l in df['Label']], y=df['ANTs_SyN'], marker_color='#d63031'))
        fig_data.append(go.Bar(name='SyNTo PT', x=[f"Region {int(l)}" for l in df['Label']], y=df['SyNTo_PT_SyN'], marker_color='#00b894'))
        if run_jax:
            fig_data.append(go.Bar(name='SyNTo JAX', x=[f"Region {int(l)}" for l in df['Label']], y=df['SyNTo_JAX_SyN'], marker_color='#0984e3'))
            
        fig_dice = go.Figure(data=fig_data)
        fig_dice.update_layout(
            title=dict(text='Regional DICE Overlap Comparison', font=dict(size=18, color='#f1f2f6')),
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
        
        # Build Quantitative Summary Card
        summary_items = [
            f"""<div class="summary-stat">
                <div class="summary-value">{np.mean([base_dices[l] for l in labels]):.4f}</div>
                <div class="summary-label">Baseline Mean DICE</div>
            </div>"""
        ]
        if run_ants:
            summary_items.append(f"""<div class="summary-stat">
                <div class="summary-value ants">{np.mean([ants_affine_dices.get(l,0) for l in labels]):.4f}</div>
                <div class="summary-label">ANTs Affine DICE</div>
            </div>""")
        summary_items.append(f"""<div class="summary-stat">
            <div class="summary-value">{np.mean([pt_affine_dices.get(l,0) for l in labels]):.4f}</div>
            <div class="summary-label">PT Affine DICE</div>
        </div>""")
        if run_jax:
            summary_items.append(f"""<div class="summary-stat">
                <div class="summary-value jax">{np.mean([jax_affine_dices.get(l,0) for l in labels]):.4f}</div>
                <div class="summary-label">JAX Affine DICE</div>
            </div>""")
        if run_ants:
            summary_items.append(f"""<div class="summary-stat">
                <div class="summary-value ants">{np.mean([ants_dices.get(l,0) for l in labels]):.4f}</div>
                <div class="summary-label">ANTs SyN DICE</div>
            </div>""")
        summary_items.append(f"""<div class="summary-stat">
            <div class="summary-value" style="color: #00b894;">{np.mean([pt_dices.get(l,0) for l in labels]):.4f}</div>
            <div class="summary-label">PT SyN DICE</div>
        </div>""")
        if run_jax:
            summary_items.append(f"""<div class="summary-stat">
                <div class="summary-value" style="color: #0984e3;">{np.mean([jax_dices.get(l,0) for l in labels]):.4f}</div>
                <div class="summary-label">JAX SyN DICE</div>
            </div>""")
        
        quantitative_summary_card = f"""
        <h2>Quantitative Summary</h2>
        <div class="summary-card">
            {"".join(summary_items)}
        </div>
        """

    # Optimizer convergence curves
    subplot_titles = ["PT Affine Convergence (Mattes MI)", "PT SyN Convergence (LNCC)"]
    if run_jax:
        subplot_titles.extend(["JAX Affine Convergence (Mattes MI)", "JAX SyN Convergence (LNCC)"])
        
    fig_loss = make_subplots(rows=2 if run_jax else 1, cols=2, subplot_titles=subplot_titles)
    fig_loss.add_trace(go.Scatter(y=pt_affine_losses, mode='lines', name='PT Affine Loss', line=dict(color='#55efc4', width=2)), row=1, col=1)
    fig_loss.add_trace(go.Scatter(y=pt_syn_losses, mode='lines', name='PT SyN Loss', line=dict(color='#00b894', width=2)), row=1, col=2)
    
    if run_jax:
        fig_loss.add_trace(go.Scatter(y=jax_affine_losses, mode='lines', name='JAX Affine Loss', line=dict(color='#74b9ff', width=2)), row=2, col=1)
        fig_loss.add_trace(go.Scatter(y=jax_syn_losses, mode='lines', name='JAX SyN Loss', line=dict(color='#0984e3', width=2)), row=2, col=2)
        
    fig_loss.update_layout(
        title=dict(text='Training Loss Convergence History', font=dict(size=18, color='#f1f2f6')),
        paper_bgcolor='#1e1e24',
        plot_bgcolor='#1e1e24',
        showlegend=False,
        height=700 if run_jax else 400,
        margin=dict(l=50, r=30, t=60, b=50)
    )
    for annotation in fig_loss['layout']['annotations']:
        annotation['font'] = dict(color='#f1f2f6', size=14)
    fig_loss.update_xaxes(tickfont=dict(color='#a4b0be'), gridcolor='#2f3542')
    fig_loss.update_yaxes(tickfont=dict(color='#a4b0be'), gridcolor='#2f3542')
    loss_html = fig_loss.to_html(full_html=False, include_plotlyjs='cdn')

    # Load b64 images
    pt_jac_b64 = get_b64(pt_jac_png)
    pt_grid_b64 = get_b64(pt_grid_png)
    jax_jac_b64 = get_b64(jax_jac_png) if run_jax else ""
    jax_grid_b64 = get_b64(jax_grid_png) if run_jax else ""
    
    # Label HTML Box grid
    lbl_html = ""
    if has_dkt:
        lbl_cols = 3 + (2 if run_ants else 0) + (2 if run_jax else 0)
        lbl_boxes = [
            f"""<div class="img-box">
                <h3>1. Baseline</h3>
                <img src="data:image/png;base64,{get_b64(label_paths.get('base'))}">
            </div>"""
        ]
        if run_ants:
            lbl_boxes.append(f"""<div class="img-box">
                <h3>2. ANTs Affine</h3>
                <img src="data:image/png;base64,{get_b64(label_paths.get('ants_aff'))}">
            </div>""")
        lbl_boxes.append(f"""<div class="img-box">
            <h3>3. PT Affine</h3>
            <img src="data:image/png;base64,{get_b64(label_paths.get('pt_aff'))}">
        </div>""")
        if run_jax:
            lbl_boxes.append(f"""<div class="img-box">
                <h3>4. JAX Affine</h3>
                <img src="data:image/png;base64,{get_b64(label_paths.get('jax_aff'))}">
            </div>""")
        if run_ants:
            lbl_boxes.append(f"""<div class="img-box">
                <h3>5. ANTs SyN</h3>
                <img src="data:image/png;base64,{get_b64(label_paths.get('ants_syn'))}">
            </div>""")
        lbl_boxes.append(f"""<div class="img-box">
            <h3>6. PT SyN</h3>
            <img src="data:image/png;base64,{get_b64(label_paths.get('pt_syn'))}">
        </div>""")
        if run_jax:
            lbl_boxes.append(f"""<div class="img-box">
                <h3>7. JAX SyN</h3>
                <img src="data:image/png;base64,{get_b64(label_paths.get('jax_syn'))}">
            </div>""")
            
        lbl_html = f"""
        <h2>Concept 1: Deformed Label Overlays on Target</h2>
        <div style="display: grid; grid-template-columns: repeat({lbl_cols}, 1fr); gap: 10px; margin-bottom: 45px;">
            {"".join(lbl_boxes)}
        </div>
        """

    # Edge HTML Box grid
    edge_cols = 3 + (2 if run_ants else 0) + (2 if run_jax else 0)
    edge_boxes = [
        f"""<div class="img-box">
            <h3>1. Baseline</h3>
            <img src="data:image/png;base64,{get_b64(edge_paths.get('base'))}">
        </div>"""
    ]
    if run_ants:
        edge_boxes.append(f"""<div class="img-box">
            <h3>2. ANTs Affine</h3>
            <img src="data:image/png;base64,{get_b64(edge_paths.get('ants_aff'))}">
        </div>""")
    edge_boxes.append(f"""<div class="img-box">
        <h3>3. PT Affine</h3>
        <img src="data:image/png;base64,{get_b64(edge_paths.get('pt_aff'))}">
    </div>""")
    if run_jax:
        edge_boxes.append(f"""<div class="img-box">
            <h3>4. JAX Affine</h3>
            <img src="data:image/png;base64,{get_b64(edge_paths.get('jax_aff'))}">
        </div>""")
    if run_ants:
        edge_boxes.append(f"""<div class="img-box">
            <h3>5. ANTs SyN</h3>
            <img src="data:image/png;base64,{get_b64(edge_paths.get('ants_syn'))}">
        </div>""")
    edge_boxes.append(f"""<div class="img-box">
        <h3>6. PT SyN</h3>
        <img src="data:image/png;base64,{get_b64(edge_paths.get('pt_syn'))}">
    </div>""")
    if run_jax:
        edge_boxes.append(f"""<div class="img-box">
            <h3>7. JAX SyN</h3>
            <img src="data:image/png;base64,{get_b64(edge_paths.get('jax_syn'))}">
        </div>""")
        
    edge_html = f"""
    <h2>Concept 2: Deformed Image Edges on Target</h2>
    <div style="display: grid; grid-template-columns: repeat({edge_cols}, 1fr); gap: 10px; margin-bottom: 45px;">
        {"".join(edge_boxes)}
    </div>
    """

    jax_warps_html = ""
    if run_jax:
        jax_warps_html = f"""
        <h2>Warp Field & Topography Mappings (JAX)</h2>
        <div class="img-grid-2">
            <div class="img-box">
                <h3>JAX Deformed Gaussian Grid</h3>
                <img src="data:image/png;base64,{jax_grid_b64}">
            </div>
            <div class="img-box">
                <h3>JAX Jacobian Determinant</h3>
                <img src="data:image/png;base64,{jax_jac_b64}">
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Registration Diagnostics Report</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; max-width: 1450px; margin: 0 auto; padding: 40px; background: #0f0f12; color: #f1f2f6; }}
            h1 {{ font-size: 32px; background: -webkit-linear-gradient(#55efc4, #00b894); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px; }}
            p.subtitle {{ color: #a4b0be; font-size: 15px; margin-top: 0; margin-bottom: 40px; }}
            h2 {{ color: #ffffff; border-bottom: 1px solid #2f3542; padding-bottom: 8px; margin-top: 50px; font-weight: 500; }}
            .img-grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 45px; }}
            .img-box {{ background: #1e1e24; padding: 10px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid #2f3542; text-align: center; }}
            .img-box img {{ width: 100%; border-radius: 6px; }}
            .img-box h3 {{ color: #ffffff; font-size: 13px; margin-top: 0; margin-bottom: 10px; font-weight: 400; }}
            .chart-container {{ background: #1e1e24; padding: 25px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid #2f3542; margin-bottom: 40px; }}
            .summary-card {{ background: #1e1e24; padding: 20px; border-radius: 12px; border: 1px solid #2f3542; margin-bottom: 40px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; }}
            .summary-stat {{ text-align: center; flex: 1; min-width: 120px; margin: 10px; }}
            .summary-value {{ font-size: 24px; font-weight: bold; color: #55efc4; }}
            .summary-value.ants {{ color: #ff7675; }}
            .summary-value.jax {{ color: #74b9ff; }}
            .summary-label {{ font-size: 11px; color: #a4b0be; text-transform: uppercase; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <h1>Registration Diagnostics & Quality Control Report</h1>
        <p class="subtitle">Documents alignment metrics, warp fields, and convergence comparison.</p>
        
        {quantitative_summary_card}

        {lbl_html}

        {edge_html}
        
        <h2>Warp Field & Topography Mappings (PyTorch)</h2>
        <div class="img-grid-2">
            <div class="img-box">
                <h3>PT Deformed Gaussian Grid</h3>
                <img src="data:image/png;base64,{pt_grid_b64}">
            </div>
            <div class="img-box">
                <h3>PT Jacobian Determinant</h3>
                <img src="data:image/png;base64,{pt_jac_b64}">
            </div>
        </div>

        {jax_warps_html}
        
        {f'<h2>Regional Overlap Over DKT Cortical Subdivisions</h2><div class="chart-container">{barplot_html}</div>' if barplot_html else ''}
        
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
