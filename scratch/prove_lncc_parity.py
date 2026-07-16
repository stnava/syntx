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

# 2. Run ANTs SyN (CC / LNCC)
print("\n--- Running ANTs SyN with CC (LNCC) Metric ---")
t0 = time.time()
reg_ants = ants.registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    syn_metric='cc', syn_sampling=4,
    initial_transform=aff_tx,
    reg_iterations=[100, 70, 50],
    grad_step=0.1,
    flow_sigma=3.0
)
t_ants = time.time() - t0
dice_ants = compute_mean_dice(fl, ml, reg_ants['fwdtransforms'])
mi_ants = ants.image_mutual_information(fi_low, reg_ants['warpedmovout'])
print(f"ANTs DICE: {dice_ants:.4f} | MI: {mi_ants:.6f} (Time: {t_ants:.2f}s)")

# 3. Run JAX SyN (LNCC)
print("\n--- Running JAX SyN with LNCC Metric ---")
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
# Load JAX warped image using ANTs fwdtransforms
warped_jax_img = ants.apply_transforms(fi_low, mi_low, reg_jax['fwdtransforms'])
mi_jax = ants.image_mutual_information(fi_low, warped_jax_img)
print(f"JAX DICE: {dice_jax:.4f} | MI: {mi_jax:.6f} (Time: {t_jax:.2f}s)")

# Cleanup temporary transform files
for path in reg_jax['fwdtransforms'] + reg_jax['invtransforms'] + reg_ants['fwdtransforms'] + reg_ants['invtransforms'] + reg_aff['fwdtransforms'] + reg_aff['invtransforms']:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

print("\n--- Summary Table (Identical Metric: LNCC / CC) ---")
print(f"{'Backend':<15} | {'DICE':<8} | {'Mutual Info':<12} | {'Time (s)':<10}")
print("-" * 55)
print(f"{'ANTs (LNCC/CC)':<15} | {dice_ants:<8.4f} | {mi_ants:<12.6f} | {t_ants:<10.2f}")
print(f"{'JAX (LNCC)':<15} | {dice_jax:<8.4f} | {mi_jax:<12.6f} | {t_jax:<10.2f}")
