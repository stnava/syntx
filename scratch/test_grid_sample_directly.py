import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_lbl = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")
mi_lbl = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/labels.DKT31.manual.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))
fi_lbl_cropped = ants.crop_image(fi_lbl, ants.iMath(ants.get_mask(fi), "MD", 12))

# Convert to PyTorch tensors
fixed_lbl = torch.tensor(fi_lbl_cropped.numpy())[None, None].to(torch.float32)
moving_lbl = torch.tensor(mi_lbl.numpy())[None, None].to(torch.float32)

from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

X_phys = get_physical_grid_torch(
    fixed_lbl.shape[2:], fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)

# ANTs COMs
com_fi = np.array(ants.get_center_of_mass(fi_cropped))
com_mi = np.array(ants.get_center_of_mass(mi))
best_t = com_mi - com_fi
best_t_zyx = torch.tensor(best_t[::-1].copy(), dtype=torch.float32)

y_phys = X_phys + best_t_zyx
y_norm = physical_to_normalized_torch(
    y_phys, mi.shape, mi.spacing, mi.origin, mi.direction
)

# Permute moving_lbl to (Z, Y, X)
moving_lbl_perm = moving_lbl.permute(0, 1, 4, 3, 2)
warped_lbl_torch = F.grid_sample(moving_lbl_perm, y_norm, mode='nearest', padding_mode='border', align_corners=True)

# Find center index of fixed COM in voxel indices
com_fi_index = np.round(ants.transform_physical_point_to_index(fi_cropped, list(com_fi))).astype(int)
com_mi_index = np.round(ants.transform_physical_point_to_index(mi, list(com_mi))).astype(int)

print("Fixed COM index (XYZ):", com_fi_index)
print("Moving COM index (XYZ):", com_mi_index)

# Get labels at COM
val_fixed_com = fi_lbl_cropped[int(com_fi_index[0]), int(com_fi_index[1]), int(com_fi_index[2])]
val_moving_com = mi_lbl[int(com_mi_index[0]), int(com_mi_index[1]), int(com_mi_index[2])]
val_warped_com = warped_lbl_torch[0, 0, int(com_fi_index[0]), int(com_fi_index[1]), int(com_fi_index[2])].item()

print("Label at Fixed COM:", val_fixed_com)
print("Label at Moving COM:", val_moving_com)
print("Label at Warped COM (should match Moving COM):", val_warped_com)

# Let's count how many non-zero elements are in fixed, moving, and warped
print("Fixed non-zero count:", (fixed_lbl > 0).sum().item())
print("Moving non-zero count:", (moving_lbl > 0).sum().item())
print("Warped non-zero count:", (warped_lbl_torch > 0).sum().item())

# Compute intersection of non-zero elements
print("Overlap non-zero count:", ((fixed_lbl > 0) & (warped_lbl_torch > 0)).sum().item())
