import os
import sys
sys.path.insert(0, 'src')
import ants
import numpy as np

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

reg_ants = ants.registration(fixed=fi_low, moving=mi_low, type_of_transform='Affine')
overlap = ants.label_overlap_measures(fl, ants.apply_transforms(fl, ml, reg_ants['fwdtransforms'], interpolator='nearestNeighbor'))
overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
if 'TargetOverlap' in overlap_valid.columns:
    dice = overlap_valid['TargetOverlap'].mean()
else:
    dice = overlap_valid['MeanOverlap'].mean()
print(f"ANTs Affine DICE: {dice:.4f}")

# Clean up
for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
