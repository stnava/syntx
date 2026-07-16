import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
from syntx.syn import physical_to_grid_affine

base_path = '/Users/stnava/data/mindboggle/volumes'
fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi_full = ants.image_read(fi_path)
mi_full = ants.image_read(mi_path)

mask = ants.get_mask(fi_full)
mask_dilated = ants.iMath(mask, "MD", 12)
fi = ants.crop_image(fi_full, mask_dilated)
mi = mi_full

reg_ants = ants.registration(fi, mi, 'Affine')
tx = ants.read_transform(reg_ants['fwdtransforms'][0])

M_ants = tx.parameters[:9].reshape(3,3)
t_ants = tx.parameters[9:]

T_grid_ants = physical_to_grid_affine(M_ants, t_ants, fi, mi)
print("ANTs grid matrix T_grid (XYZ order):")
print(T_grid_ants)

# Clean up
for path in reg_ants['fwdtransforms']:
    if os.path.exists(path):
        os.remove(path)
