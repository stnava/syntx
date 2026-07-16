import ants
import os
import numpy as np
from syntx.syn import registration

base_path = '/Users/stnava/data/mindboggle/volumes'
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

fi_low = ants.resample_image(fi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)

print("--- Running ANTs SyN ---")
reg_ants = ants.registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    syn_metric='cc', syn_sampling=4,
    reg_iterations=[10, 0, 0] # 10 iterations at level 0
)

warp_ants = ants.image_read(reg_ants['fwdtransforms'][0])
warp_ants_arr = warp_ants.numpy()
print("ANTs Warp min:", warp_ants_arr.min(axis=(0, 1, 2)))
print("ANTs Warp max:", warp_ants_arr.max(axis=(0, 1, 2)))
print("ANTs Warp mean:", warp_ants_arr.mean(axis=(0, 1, 2)))

print("\n--- Running JAX SyN ---")
reg_jax = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    backend='jax', syn_metric='lncc', syn_sampling=4,
    levels=[8, 4, 2], # Run same 3 levels
    affine_iterations=[0, 0, 0], # No affine to focus purely on deformation starting from same state
    reg_iterations=[10, 0, 0], # 10 iterations at level 0
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=False
)

warp_jax = ants.image_read(reg_jax['fwdtransforms'][0])
warp_jax_arr = warp_jax.numpy()
print("JAX Warp min:", warp_jax_arr.min(axis=(0, 1, 2)))
print("JAX Warp max:", warp_jax_arr.max(axis=(0, 1, 2)))
print("JAX Warp mean:", warp_jax_arr.mean(axis=(0, 1, 2)))

# Clean up
for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
for path in reg_jax['fwdtransforms'] + reg_jax['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
