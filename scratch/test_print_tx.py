import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import syntx

base_path = '/Users/stnava/data/mindboggle/volumes'
fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
fl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')
ml_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi_full = ants.image_read(fi_path)
mi_full = ants.image_read(mi_path)
fl_full = ants.image_read(fl_path)
ml_full = ants.image_read(ml_path)

mask = ants.get_mask(fi_full)
mask_dilated = ants.iMath(mask, "MD", 12)
fi = ants.crop_image(fi_full, mask_dilated)
mi = mi_full

reg = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[0], reg_iterations=[0]
)

# Print transforms from the output dict
tx_file = reg['fwdtransforms'][0]
tx = ants.read_transform(tx_file)
print("Saved matrix (M_phys_xyz):\n", tx.parameters[:9].reshape(3,3))
print("Saved translation (t_phys_xyz):\n", tx.parameters[9:])

# Calculate ground-truth CoM translation
com_fi = np.array(ants.get_center_of_mass(fi))
com_mi = np.array(ants.get_center_of_mass(mi))
print("\ncom_fi (XYZ):", com_fi)
print("com_mi (XYZ):", com_mi)
print("com_mi - com_fi (XYZ):", com_mi - com_fi)

# Clean up
for path in reg['fwdtransforms'] + reg['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
