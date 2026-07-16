import os
import time
import numpy as np
import torch
import jax
import ants

from syntx import registration

def compute_otsu_dice(fi, warped):
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    if 'MeanOverlap' in overlap.columns:
        return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return 0.0

def evaluate_jacobian_folding(fi, fwd_warp_file):
    jac_img = ants.create_jacobian_determinant_image(fi, fwd_warp_file)
    jac_np = jac_img.numpy()
    min_jac = float(jac_np.min())
    folding_rate = float(np.mean(jac_np <= 0.0))
    return min_jac, folding_rate

def run_2d_registration():
    print("\n--- Running 2D Registration (r16 to r64) ---")
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    
    # PyTorch
    t0 = time.time()
    res_py = registration(
        fixed=fi, moving=mi, type_of_transform='SyNTo', backend='pytorch',
        levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
        grad_step=0.5, flow_sigma=1.0
    )
    t_py = time.time() - t0
    dice_py = compute_otsu_dice(fi, res_py['warpedmovout'])
    warp_py = next(tx for tx in res_py['fwdtransforms'] if tx.endswith('.nii.gz'))
    min_jac_py, folding_py = evaluate_jacobian_folding(fi, warp_py)
    
    # JAX
    t0 = time.time()
    res_jax = registration(
        fixed=fi, moving=mi, type_of_transform='SyNTo', backend='jax',
        levels=[2, 1], reg_iterations=[30, 20], affine_iterations=[30, 20],
        grad_step=0.5, flow_sigma=1.0
    )
    t_jax = time.time() - t0
    dice_jax = compute_otsu_dice(fi, res_jax['warpedmovout'])
    warp_jax = next(tx for tx in res_jax['fwdtransforms'] if tx.endswith('.nii.gz'))
    min_jac_jax, folding_jax = evaluate_jacobian_folding(fi, warp_jax)
    
    print(f"2D PyTorch: Time={t_py:.2f}s, Dice={dice_py:.4f}, MinJac={min_jac_py:.4f}, Folding={folding_py*100:.4f}%")
    print(f"2D JAX:     Time={t_jax:.2f}s, Dice={dice_jax:.4f}, MinJac={min_jac_jax:.4f}, Folding={folding_jax*100:.4f}%")
    
    # Clean up fwd / inv transforms
    for r in [res_py, res_jax]:
        for tx in r['fwdtransforms'] + r['invtransforms']:
            if os.path.exists(tx):
                try:
                    os.remove(tx)
                except Exception:
                    pass
                    
    return {
        'pytorch': {'time': t_py, 'dice': dice_py, 'min_jac': min_jac_py, 'folding': folding_py},
        'jax': {'time': t_jax, 'dice': dice_jax, 'min_jac': min_jac_jax, 'folding': folding_jax}
    }

def run_3d_registration():
    print("\n--- Running 3D Registration (img1_brain to img2_brain downsampled) ---")
    fi = ants.image_read("cache/img1_brain.nii.gz")
    mi = ants.image_read("cache/img2_brain.nii.gz")
    
    # Resample to 32^3 to keep it fast
    fi_small = ants.resample_image(fi, (32, 32, 32), use_voxels=True)
    mi_small = ants.resample_image(mi, (32, 32, 32), use_voxels=True)
    
    # PyTorch
    t0 = time.time()
    res_py = registration(
        fixed=fi_small, moving=mi_small, type_of_transform='SyNTo', backend='pytorch',
        levels=[2, 1], reg_iterations=[15, 10], affine_iterations=[15, 10],
        grad_step=0.5, flow_sigma=1.0
    )
    t_py = time.time() - t0
    dice_py = compute_otsu_dice(fi_small, res_py['warpedmovout'])
    warp_py = next(tx for tx in res_py['fwdtransforms'] if tx.endswith('.nii.gz'))
    min_jac_py, folding_py = evaluate_jacobian_folding(fi_small, warp_py)
    
    # JAX
    t0 = time.time()
    res_jax = registration(
        fixed=fi_small, moving=mi_small, type_of_transform='SyNTo', backend='jax',
        levels=[2, 1], reg_iterations=[15, 10], affine_iterations=[15, 10],
        grad_step=0.5, flow_sigma=1.0
    )
    t_jax = time.time() - t0
    dice_jax = compute_otsu_dice(fi_small, res_jax['warpedmovout'])
    warp_jax = next(tx for tx in res_jax['fwdtransforms'] if tx.endswith('.nii.gz'))
    min_jac_jax, folding_jax = evaluate_jacobian_folding(fi_small, warp_jax)
    
    print(f"3D PyTorch: Time={t_py:.2f}s, Dice={dice_py:.4f}, MinJac={min_jac_py:.4f}, Folding={folding_py*100:.4f}%")
    print(f"3D JAX:     Time={t_jax:.2f}s, Dice={dice_jax:.4f}, MinJac={min_jac_jax:.4f}, Folding={folding_jax*100:.4f}%")
    
    # Clean up fwd / inv transforms
    for r in [res_py, res_jax]:
        for tx in r['fwdtransforms'] + r['invtransforms']:
            if os.path.exists(tx):
                try:
                    os.remove(tx)
                except Exception:
                    pass
                    
    return {
        'pytorch': {'time': t_py, 'dice': dice_py, 'min_jac': min_jac_py, 'folding': folding_py},
        'jax': {'time': t_jax, 'dice': dice_jax, 'min_jac': min_jac_jax, 'folding': folding_jax}
    }

if __name__ == "__main__":
    results_2d = run_2d_registration()
    results_3d = run_3d_registration()
    print("\n--- Summary 2D vs 3D ---")
    print(f"2D Dice parity between JAX and PyTorch: JAX={results_2d['jax']['dice']:.4f}, PyTorch={results_2d['pytorch']['dice']:.4f}")
    print(f"3D Dice parity between JAX and PyTorch: JAX={results_3d['jax']['dice']:.4f}, PyTorch={results_3d['pytorch']['dice']:.4f}")
