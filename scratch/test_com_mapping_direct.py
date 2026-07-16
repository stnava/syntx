import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")

from syntx.syn import _physical_to_normalized_torch_yfirst

def physical_to_normalized_torch_corrected(phys_coords, target_shape, spacing, origin, direction):
    target_shape_rev = tuple(reversed(target_shape))
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape_rev, spacing_rev, origin_rev, direction_rev)

# Compute ANTs Fixed and Moving COMs (foreground)
com_fi = np.array(ants.get_center_of_mass(fi_cropped))
com_mi = np.array(ants.get_center_of_mass(mi))

print("fi cropped COM (XYZ):", com_fi)
print("mi COM (XYZ):", com_mi)

# Map com_fi using y_phys = com_fi + (com_mi - com_fi) = com_mi
# So the physical point in moving image space is exactly com_mi!
p_phys_xyz = torch.tensor(com_mi, dtype=torch.float32)
p_phys_zyx = torch.flip(p_phys_xyz, dims=[0]).view(1, 1, 1, 1, 3)

y_norm = physical_to_normalized_torch_corrected(
    p_phys_zyx, mi.shape, mi.spacing, mi.origin, mi.direction
)
print("y_norm (XYZ):", y_norm.numpy().flatten())

# Map y_norm back to moving image voxel index
# Since y_norm is [x_norm, y_norm, z_norm]:
# x_voxel = (x_norm + 1) / 2 * (nx - 1)
shape_xyz = torch.tensor(mi.shape, dtype=torch.float32)
voxel_coords = (y_norm.view(-1) + 1.0) / 2.0 * (shape_xyz - 1)
print("Mapped Voxel Index (XYZ):", voxel_coords.numpy())

# ANTs ground-truth index for com_mi
com_mi_index = ants.transform_physical_point_to_index(mi, list(com_mi))
print("ANTs Ground-Truth Index (XYZ):", com_mi_index)
