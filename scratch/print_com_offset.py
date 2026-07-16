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

fixed_spacing = fi_cropped.spacing
fixed_origin = fi_cropped.origin
fixed_direction = fi_cropped.direction

moving_spacing = mi.spacing
moving_origin = mi.origin
moving_direction = mi.direction

dim = 3
device = 'cpu'
dtype = torch.float32

Nx_t = torch.tensor(list(reversed(fixed_image.shape[2:])), dtype=dtype)
Sx_t = torch.tensor(list(reversed(fixed_spacing)), dtype=dtype)
Ox_t = torch.tensor(list(reversed(fixed_origin)), dtype=dtype)
Dx_t = torch.tensor(np.asarray(fixed_direction)[::-1, ::-1].copy(), dtype=dtype)

Ny_t = torch.tensor(list(reversed(moving_image.shape[2:])), dtype=dtype)
Sy_t = torch.tensor(list(reversed(moving_spacing)), dtype=dtype)
Oy_t = torch.tensor(list(reversed(moving_origin)), dtype=dtype)
Dy_t = torch.tensor(np.asarray(moving_direction)[::-1, ::-1].copy(), dtype=dtype)

Kx = torch.diag((Nx_t - 1) / 2.0)
Cx = (Nx_t - 1) / 2.0
Ky = torch.diag((Ny_t - 1) / 2.0)
Cy = (Ny_t - 1) / 2.0

Kx_inv = torch.inverse(Kx)
Sx_inv = torch.inverse(torch.diag(Sx_t))
Wx = Kx_inv @ Sx_inv @ Dx_t.t()
bx = - Kx_inv @ Sx_inv @ Dx_t.t() @ Ox_t - Kx_inv @ Cx

Vy = Dy_t @ torch.diag(Sy_t) @ Ky
cy = Dy_t @ torch.diag(Sy_t) @ Cy + Oy_t

com_fixed_fov = Dx_t @ (Sx_t * (Nx_t - 1) / 2.0) + Ox_t

# best_t from fg
com_fixed_pos = torch.clamp(fixed_image, min=0.0)
com_moving_pos = torch.clamp(moving_image, min=0.0)
grids_f = [torch.arange(s, dtype=dtype) for s in fixed_image.shape[2:]]
idxs_f = torch.stack(list(reversed(torch.meshgrid(*grids_f, indexing='ij'))), dim=-1)
grids_m = [torch.arange(s, dtype=dtype) for s in moving_image.shape[2:]]
idxs_m = torch.stack(list(reversed(torch.meshgrid(*grids_m, indexing='ij'))), dim=-1)
com_fixed_voxel = torch.sum(com_fixed_pos.squeeze(0).squeeze(0).unsqueeze(-1) * idxs_f, dim=list(range(dim))) / com_fixed_pos.sum()
com_moving_voxel = torch.sum(com_moving_pos.squeeze(0).squeeze(0).unsqueeze(-1) * idxs_m, dim=list(range(dim))) / com_moving_pos.sum()
com_fixed_fg = Dx_t @ (Sx_t * com_fixed_voxel) + Ox_t
com_moving_fg = Dy_t @ (Sy_t * com_moving_voxel) + Oy_t
best_t = com_moving_fg - com_fixed_fg
t_fov = cy - com_fixed_fov

t_grid = torch.inverse(Vy) @ (best_t - t_fov)

# Let's print Vy @ bx
Vy_bx = Vy @ bx
print("Vy @ bx (ZYX):", Vy_bx.numpy())
print("cy (com_moving_fov, ZYX):", cy.numpy())
print("Vy @ t_grid (ZYX):", (Vy @ t_grid).numpy())

t_phys = Vy @ bx + cy + Vy @ t_grid
print("t_phys (ZYX):", t_phys.numpy())

# Flipped to XYZ
P = torch.flip(torch.eye(dim), dims=[0])
t_phys_xyz = P @ t_phys
print("t_phys_xyz (XYZ):", t_phys_xyz.numpy())

# Mmapped center using y = M_phys @ x + t_phys
# M_phys = Vy @ Wx
M_phys = Vy @ Wx
M_phys_xyz = P @ M_phys @ P
fixed_center_xyz = torch.flip(com_fixed_fg, dims=[0]) # center of mass in XYZ
mapped_center_xyz = M_phys_xyz @ fixed_center_xyz + t_phys_xyz
print("\nMapped fixed center of mass (XYZ):", mapped_center_xyz.numpy())
print("Target moving center of mass (com_moving_fg, XYZ):", torch.flip(com_moving_fg, dims=[0]).numpy())
