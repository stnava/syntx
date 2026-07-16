import ants
import os
import time
import numpy as np
from syntx.syn import registration

def compute_mean_dice(fixed_labels, moving_labels, transforms):
    warped_labels = ants.apply_transforms(
        fixed=fixed_labels,
        moving=moving_labels,
        transformlist=transforms,
        interpolator='nearestNeighbor'
    )
    overlap = ants.label_overlap_measures(
        source_image=fixed_labels, 
        target_image=warped_labels
    )
    # Exclude background label (usually 0) and the 'All' summary row
    overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
    
    if 'TargetOverlap' in overlap_valid.columns:
        mean_dice = overlap_valid['TargetOverlap'].mean()
    else:
        mean_dice = overlap_valid['MeanOverlap'].mean()
        
    return mean_dice

def run_mindboggle_syn_test():
    base_path = '/Users/stnava/data/mindboggle/volumes'

    # Fixed Image: OASIS-TRT-20-1
    fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
    fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

    # Moving Image: MMRR-21-1
    moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
    moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')
    
    fi = ants.image_read(fixed_img_path)
    fl = ants.image_read(fixed_lbl_path)
    mi = ants.image_read(moving_img_path)
    ml = ants.image_read(moving_lbl_path)
    
    # Resample images to 2x2x2 for speed
    fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
    mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)
    
    print("\n--- 1. ANTs SyN (Deformable) ---")
    t0 = time.time()
    reg_ants = ants.registration(fixed=fi_low, moving=mi_low, type_of_transform='SyN')
    t_ants = time.time() - t0
    dice_ants = compute_mean_dice(fl, ml, reg_ants['fwdtransforms'])
    print(f"ANTs SyN DICE: {dice_ants:.4f} ({t_ants:.2f}s)")
    
    # Clean up ANTs transforms
    for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
            
    print("\n--- 2. SyNTo (PyTorch) SyN ---")
    t0 = time.time()
    reg_torch = registration(
        fixed=fi_low, moving=mi_low, type_of_transform='SyN',
        backend='pytorch', syn_metric='mattes_mi',
        affine_iterations=[200, 100, 50],
        reg_iterations=[100, 50, 20],
        verbose=True
    )
    t_torch = time.time() - t0
    dice_torch = compute_mean_dice(fl, ml, reg_torch['fwdtransforms'])
    print(f"PyTorch SyN DICE: {dice_torch:.4f} ({t_torch:.2f}s)")
    
    # Clean up PyTorch transforms
    for path in reg_torch['fwdtransforms'] + reg_torch['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
            
    print("\n--- 3. SyNTo (JAX) SyN ---")
    t0 = time.time()
    reg_jax = registration(
        fixed=fi_low, moving=mi_low, type_of_transform='SyN',
        backend='jax', syn_metric='mattes_mi',
        affine_iterations=[200, 100, 50],
        reg_iterations=[100, 50, 20],
        verbose=True
    )
    t_jax = time.time() - t0
    dice_jax = compute_mean_dice(fl, ml, reg_jax['fwdtransforms'])
    print(f"JAX SyN DICE: {dice_jax:.4f} ({t_jax:.2f}s)")
    
    # Clean up JAX transforms
    for path in reg_jax['fwdtransforms'] + reg_jax['invtransforms']:
        if os.path.exists(path):
            os.remove(path)

if __name__ == '__main__':
    run_mindboggle_syn_test()
