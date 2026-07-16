import ants
import os
import numpy as np
import sys

sys.path.insert(0, 'src')
from syntx.syn import registration

base_path = '/Users/stnava/data/mindboggle/volumes'
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

# Use 3.0mm spacing to make it fast
fi_low = ants.resample_image(fi, (3.0, 3.0, 3.0), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (3.0, 3.0, 3.0), use_voxels=False, interp_type=4)

# 1. Run ANTs Affine Only to get a common starting transform
print("--- Running ANTs Affine to get starting transform ---")
reg_aff = ants.registration(
    fixed=fi_low, moving=mi_low, type_of_transform='Affine'
)
aff_tx = reg_aff['fwdtransforms'][0]

# 2. Run ANTs SyN for 10 iterations starting from aff_tx
print("--- Running ANTs SyN starting from Affine ---")
reg_ants = ants.registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    syn_metric='cc', syn_sampling=4,
    initial_transform=aff_tx,
    reg_iterations=[10, 0, 0]
)
warp_ants = ants.image_read(reg_ants['fwdtransforms'][0])
warp_ants_arr = warp_ants.numpy()

# 3. Run JAX SyN for 10 iterations starting from aff_tx
print("--- Running JAX SyN starting from Affine ---")
reg_jax = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyN',
    backend='jax', syn_metric='lncc', syn_sampling=4,
    initial_transform=aff_tx,
    affine_iterations=[0, 0], # Disable JAX affine stage
    reg_iterations=[10, 0], # 10 iterations at Level 0
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=False
)
warp_jax = ants.image_read(reg_jax['fwdtransforms'][0])
warp_jax_arr = warp_jax.numpy()

print("\nFull 3x3 correlation matrix between ANTs components (rows) and JAX components (cols):")
for i in range(3):
    row_corrs = []
    for j in range(3):
        corr = np.corrcoef(warp_ants_arr[..., i].ravel(), warp_jax_arr[..., j].ravel())[0, 1]
        row_corrs.append(f"{corr: .4f}")
    print(f"ANTs component {i}: " + ", ".join(row_corrs))

# Clean up
for path in reg_aff['fwdtransforms'] + reg_aff['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
for path in reg_jax['fwdtransforms'] + reg_jax['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
