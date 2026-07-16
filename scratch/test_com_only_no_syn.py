import os
import sys
sys.path.insert(0, 'src')
import ants
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
        mean_dice = overlap_valid['TargetOverlap'].mean()
    else:
        mean_dice = overlap_valid['MeanOverlap'].mean()
    return mean_dice

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

fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)

# Run registration with CoM initialization and 0 affine iterations and 0 deformable iterations
print("Running SyNTo (PyTorch) with 0 affine and 0 deformable iterations...")
reg_torch = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    backend='pytorch', similarity_metric='lncc',
    affine_iterations=[0, 0, 0],
    reg_iterations=[0, 0, 0],
    verbose=True
)
dice_torch = compute_mean_dice(fl, ml, reg_torch['fwdtransforms'])
print(f"PyTorch SyN (CoM-only, no SyN) DICE: {dice_torch:.4f}")
