import sys, os
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))

fixed_image = torch.tensor(fi_cropped.numpy())[None, None]
moving_image = torch.tensor(mi.numpy())[None, None]

from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

# Get fixed physical grid
X_phys = get_physical_grid_torch(
    fixed_image.shape[2:], fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)

com_fi = np.array(ants.get_center_of_mass(fi_cropped))
com_mi = np.array(ants.get_center_of_mass(mi))
best_t = com_mi - com_fi

best_t_tensor = torch.tensor(best_t, dtype=torch.float32)
best_t_zyx = torch.flip(best_t_tensor, dims=[0])

# Compute initial grid using physical mapping
y_phys = X_phys + best_t_zyx
y_norm = physical_to_normalized_torch(
    y_phys, mi.shape, mi.spacing, mi.origin, mi.direction
)

print("y_phys center (ZYX):", y_phys[0, 103, 80, 80].numpy())
print("y_norm center (XY):", y_norm[0, 103, 80, 80].numpy())

print("y_norm min:", y_norm.min().item(), "max:", y_norm.max().item())
