import os
import sys
sys.path.insert(0, 'src')
import ants
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'

fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

fi_low = ants.resample_image(fi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)

reg_ants = ants.registration(fixed=fi_low, moving=mi_low, type_of_transform='Affine')

# Load ANTs transform parameters
tx_file = reg_ants['fwdtransforms'][0]
tx = ants.read_transform(tx_file)
params = tx.parameters
fixed_params = tx.fixed_parameters

print("ANTs Affine Parameters:")
print("Parameters (rotation/scale upper 3x3 and translation):")
print("Matrix:\n", params[:9].reshape(3, 3))
print("Translation:\n", params[9:])
print("Fixed Parameters (center of rotation):")
print(fixed_params)

# Clean up
for path in reg_ants['fwdtransforms'] + reg_ants['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
