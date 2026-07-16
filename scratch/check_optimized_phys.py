import numpy as np
import os
import ants

base_path = '/Users/stnava/data/mindboggle/volumes'
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

fi_low = ants.resample_image(fi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)

T_grid_opt = np.array([
    [ 0.00418562, -0.09849785, -0.650417,    0.01256688],
    [-0.85926807,  0.24997216, -0.0430854,  -0.01460676],
    [ 0.2111487,   0.8898485,  -0.06614452, -0.13308059],
    [ 0.,          0.,          0.,          1.        ]
], dtype=np.float32)

from syntx.syn import grid_to_physical_affine
M_phys, t_phys = grid_to_physical_affine(T_grid_opt, fi_low, mi_low)
print("Optimized Physical Affine:")
print("M_phys:\n", M_phys)
print("t_phys:\n", t_phys)
