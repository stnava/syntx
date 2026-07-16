import ants
import os
import sys
sys.path.insert(0, 'src')
import numpy as np
from syntx.syn import registration

base_path = '/Users/stnava/data/mindboggle/volumes'
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=4)

print("Running JAX SyN with verbose=True to check displacement field evolution...")
reg_jax = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    backend='jax', syn_metric='lncc',
    affine_iterations=[0, 0, 0], # Focus on deformable stage
    reg_iterations=[10, 0, 0], # Just 10 iterations at coarsest resolution
    verbose=True
)
