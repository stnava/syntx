import ants
import os
import time
import numpy as np
import sys

sys.path.insert(0, 'src')
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
    overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
    if 'TargetOverlap' in overlap_valid.columns:
        dice = overlap_valid['TargetOverlap'].mean()
    else:
        dice = overlap_valid['MeanOverlap'].mean()
    return dice

base_path = '/Users/stnava/data/mindboggle/volumes'

fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

print("Loading dataset...")
fi = ants.image_read(fixed_img_path)
fl = ants.image_read(fixed_lbl_path)
mi = ants.image_read(moving_img_path)
ml = ants.image_read(moving_lbl_path)

fi_low = ants.resample_image(fi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)

# 1. Run ANTs Affine to get starting transform
print("\n--- Running ANTs Affine to establish starting transform ---")
reg_aff = ants.registration(fixed=fi_low, moving=mi_low, type_of_transform='Affine')
aff_tx = reg_aff['fwdtransforms'][0]

# Compute initial/affine Dice
dice_init = compute_mean_dice(fl, ml, [aff_tx])
print(f"Initial Affine DICE: {dice_init:.4f}")

# 2. Run JAX SyN
print("\n--- Running JAX SyN (levels=[4, 2, 1], reg_iterations=[100, 70, 50]) ---")
t0 = time.time()
reg_jax = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    backend='jax', syn_metric='lncc', syn_sampling=4,
    initial_transform=aff_tx,
    levels=[4, 2, 1],
    reg_iterations=[100, 70, 50],
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=False
)
t_jax = time.time() - t0
dice_jax = compute_mean_dice(fl, ml, reg_jax['fwdtransforms'])
print(f"JAX DICE: {dice_jax:.4f} (Time: {t_jax:.2f}s)")

# 3. Run PyTorch SyN
print("\n--- Running PyTorch SyN (levels=[4, 2, 1], reg_iterations=[100, 70, 50]) ---")
t0 = time.time()
reg_py = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    backend='pytorch', syn_metric='lncc', syn_sampling=4,
    initial_transform=aff_tx,
    levels=[4, 2, 1],
    reg_iterations=[100, 70, 50],
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=False
)
t_py = time.time() - t0
dice_py = compute_mean_dice(fl, ml, reg_py['fwdtransforms'])
print(f"PyTorch DICE: {dice_py:.4f} (Time: {t_py:.2f}s)")

# Cleanup temporary transform files
for path in reg_jax['fwdtransforms'] + reg_jax['invtransforms'] + reg_py['fwdtransforms'] + reg_py['invtransforms'] + reg_aff['fwdtransforms'] + reg_aff['invtransforms']:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

print("\n--- Summary Table ---")
print(f"{'Backend':<15} | {'DICE':<8} | {'Time (s)':<10}")
print("-" * 40)
print(f"{'Initial Affine':<15} | {dice_init:<8.4f} | {'N/A':<10}")
print(f"{'JAX':<15} | {dice_jax:<8.4f} | {t_jax:<10.2f}")
print(f"{'PyTorch':<15} | {dice_py:<8.4f} | {t_py:<10.2f}")
print(f"{'ANTs (Baseline)':<15} | {0.4296:<8.4f} | ~1500s (deformable portion)")
