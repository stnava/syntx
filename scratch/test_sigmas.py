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

fi = ants.image_read(fixed_img_path)
fl = ants.image_read(fixed_lbl_path)
mi = ants.image_read(moving_img_path)
ml = ants.image_read(moving_lbl_path)

fi_low = ants.resample_image(fi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)

# Test ANTs with explicit flow_sigma and elastic_sigma
print("\n--- Running ANTs with explicit flow_sigma=3.0, elastic_sigma=0.0 ---")
reg_ants = ants.registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    syn_metric='cc', syn_sampling=4,
    flow_sigma=3.0, elastic_sigma=0.0
)
dice_ants = compute_mean_dice(fl, ml, reg_ants['fwdtransforms'])
print(f"ANTs DICE: {dice_ants:.4f}")

# Test JAX with spacing=None (voxel-space smoothing!)
print("\n--- Running JAX with spacing=None (voxel-space smoothing) ---")
reg_jax_voxel = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    backend='jax', syn_metric='lncc', syn_sampling=4,
    levels=[8, 4, 2, 1],
    affine_iterations=[200, 100, 50, 20],
    reg_iterations=[100, 70, 50, 20],
    grad_step=0.1,
    flow_sigma=3.0,
    spacing=None, # This will trigger spacing=None voxel-space smoothing!
    verbose=False
)
dice_jax_voxel = compute_mean_dice(fl, ml, reg_jax_voxel['fwdtransforms'])
print(f"JAX Voxel-Space DICE: {dice_jax_voxel:.4f}")

# Clean up
for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
for path in reg_jax_voxel['fwdtransforms'] + reg_jax_voxel['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
