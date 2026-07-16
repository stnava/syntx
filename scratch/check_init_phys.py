import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))

rs = syntx.syn(
    fixed=fi_cropped, moving=mi, type_of_transform='Affine', backend='pytorch',
    affine_iterations=[0], reg_iterations=[0]
)

# Read the exported transform
tx = ants.read_transform(rs['fwdtransforms'][0])

# Compute fixed center
fixed_center = np.array(fi_cropped.origin) + np.array(fi_cropped.spacing) * (np.array(fi_cropped.shape) - 1) / 2.0
print("Fixed Center:", fixed_center)

# Map fixed center using exported transform
# ITK: y = M @ (x - center) + center + t
# Since fixed_parameters (center) is [0,0,0], it's: y = M @ x + t
M = tx.parameters[:9].reshape(3, 3)
t = tx.parameters[9:]
mapped_center = M @ fixed_center + t
print("Mapped Center (Syntx):", mapped_center)

# ANTs reference registration parameters
reg_ants = ants.registration(fixed=fi_cropped, moving=mi, type_of_transform='Affine')
tx_ants = ants.read_transform(reg_ants['fwdtransforms'][0])

M_ants = tx_ants.parameters[:9].reshape(3, 3)
t_ants = tx_ants.parameters[9:]
center_ants = tx_ants.fixed_parameters
mapped_center_ants = M_ants @ (fixed_center - center_ants) + center_ants + t_ants
print("Mapped Center (ANTs):", mapped_center_ants)

# Print target center of mass of moving image (foreground)
mask_m = ants.get_mask(mi)
moving_center_fov = np.array(mi.origin) + np.array(mi.spacing) * (np.array(mi.shape) - 1) / 2.0
print("Moving FOV Center:", moving_center_fov)
