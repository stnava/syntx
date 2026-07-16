import os
import sys
sys.path.insert(0, 'src')
import ants
import torch
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'

# Fixed Image: OASIS-TRT-20-1
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

# Moving Image: MMRR-21-1
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)

print(f"fi_low shape: {fi_low.shape}, spacing: {fi_low.spacing}, origin: {fi_low.origin}")
print(f"mi_low shape: {mi_low.shape}, spacing: {mi_low.spacing}, origin: {mi_low.origin}")

# Run a dummy registration in PyTorch to trigger print statements in CoM
import syntx
reg = syntx.syn(fi_low, mi_low, 'Affine', backend='pytorch', affine_iterations=[1, 0, 0])
