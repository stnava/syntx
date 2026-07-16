import os
import sys
sys.path.insert(0, 'src')
import ants
import torch
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'

# Fixed Image: OASIS-TRT-20-1
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

# Moving Image: MMRR-21-1
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi = ants.image_read(fixed_img_path)
mi = ants.image_read(moving_img_path)

fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)

# Extract physical metadata
fixed_spacing = fi_low.spacing
fixed_origin = fi_low.origin
fixed_direction = fi_low.direction

moving_spacing = mi_low.spacing
moving_origin = mi_low.origin
moving_direction = mi_low.direction

# Convert to tensors
device = 'cpu'
dtype = torch.float32

fixed_image_shape = fi_low.shape
# PyTorch treats shape as (C, Z, Y, X) or similar, but let's look at what syn.py gets:
# fixed_image has shape (1, 1, 80, 128, 128)
# moving_image has shape (1, 1, 128, 128, 102)
fixed_shape_py = (80, 128, 128)
moving_shape_py = (128, 128, 102)

Nx_t = torch.tensor(list(reversed(fixed_shape_py)), device=device, dtype=dtype)
Sx_t = torch.tensor(list(fixed_spacing), device=device, dtype=dtype)
Ox_t = torch.tensor(list(fixed_origin), device=device, dtype=dtype)
Dx_t = torch.tensor(np.asarray(fixed_direction), device=device, dtype=dtype)

com_fixed_fov = Dx_t @ (Sx_t * (Nx_t - 1) / 2.0) + Ox_t

Ny_t = torch.tensor(list(reversed(moving_shape_py)), device=device, dtype=dtype)
Sy_t = torch.tensor(list(moving_spacing), device=device, dtype=dtype)
Oy_t = torch.tensor(list(moving_origin), device=device, dtype=dtype)
Dy_t = torch.tensor(np.asarray(moving_direction), device=device, dtype=dtype)

com_moving_fov = Dy_t @ (Sy_t * (Ny_t - 1) / 2.0) + Oy_t

t_fov = com_moving_fov - com_fixed_fov

print(f"Nx_t: {Nx_t.numpy()}")
print(f"Sx_t: {Sx_t.numpy()}")
print(f"Ox_t: {Ox_t.numpy()}")
print(f"Dx_t:\n{Dx_t.numpy()}")
print(f"com_fixed_fov: {com_fixed_fov.numpy()}")
print("")
print(f"Ny_t: {Ny_t.numpy()}")
print(f"Sy_t: {Sy_t.numpy()}")
print(f"Oy_t: {Oy_t.numpy()}")
print(f"Dy_t:\n{Dy_t.numpy()}")
print(f"com_moving_fov: {com_moving_fov.numpy()}")
print("")
print(f"t_fov: {t_fov.numpy()}")
print("")

# Test physical translation for identity T_grid
from syntx.syn import grid_to_physical_affine
T_grid_id = np.eye(4, dtype=np.float32)
M_phys_id, t_phys_id = grid_to_physical_affine(T_grid_id, fi_low, mi_low)
print(f"For Identity T_grid:")
print(f"M_phys_id:\n{M_phys_id}")
print(f"t_phys_id: {t_phys_id}")

