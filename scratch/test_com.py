import torch
import numpy as np
from syntx.syn import SyNTo
import ants

fixed_np = np.zeros((32, 32, 32), dtype=np.float32)
fixed_np[10:22, 10:22, 10:22] = 1.0
fixed_img = ants.from_numpy(fixed_np, spacing=(1, 1, 1), origin=(10, 20, 30))

moving_np = np.zeros((32, 32, 32), dtype=np.float32)
moving_np[12:24, 12:24, 12:24] = 1.0
moving_img = ants.from_numpy(moving_np, spacing=(1, 1, 1), origin=(-50, -40, -30))

fixed_tensor = torch.tensor(fixed_img.numpy()).unsqueeze(0).unsqueeze(0)
moving_tensor = torch.tensor(moving_img.numpy()).unsqueeze(0).unsqueeze(0)

# Let's inspect CoM initialization
model = SyNTo(dim=3, grid_shape=(32, 32, 32))
model.fit(
    fixed_tensor, moving_tensor,
    levels=[1],
    epochs_per_level=0,
    affine_epochs=0,
    fixed_spacing=fixed_img.spacing,
    fixed_origin=fixed_img.origin,
    fixed_direction=fixed_img.direction,
    moving_spacing=moving_img.spacing,
    moving_origin=moving_img.origin,
    moving_direction=moving_img.direction,
)

T_grid = model.affine.get_affine_grid_matrix().detach()
print("T_grid matrix:\n", T_grid.numpy())

from syntx.syn import grid_to_physical_affine_torch
fixed_spacing = fixed_img.spacing
fixed_origin = fixed_img.origin
fixed_direction = fixed_img.direction
moving_spacing = moving_img.spacing
moving_origin = moving_img.origin
moving_direction = moving_img.direction

M_phys, t_phys = grid_to_physical_affine_torch(
    T_grid,
    fixed_img.shape, fixed_spacing, fixed_origin, fixed_direction,
    moving_img.shape, moving_spacing, moving_origin, moving_direction
)

print("Physical translation t_phys:", t_phys.numpy())
print("Expected physical translation (com_moving - com_fixed):")
# Compute actual centers of mass using coordinate indexing
z, y, x = np.ogrid[:32, :32, :32]
# reversed to match the (z, y, x) order in physical coordinate grids
com_fixed_voxel = np.array([z[10:22].mean(), y[10:22].mean(), x[10:22].mean()])
com_fixed_phys = np.array(fixed_direction) @ (np.array(fixed_spacing) * com_fixed_voxel) + np.array(fixed_origin)

com_moving_voxel = np.array([z[12:24].mean(), y[12:24].mean(), x[12:24].mean()])
com_moving_phys = np.array(moving_direction) @ (np.array(moving_spacing) * com_moving_voxel) + np.array(moving_origin)

print("Fixed CoM physical:", com_fixed_phys)
print("Moving CoM physical:", com_moving_phys)
print("Difference (com_moving - com_fixed):", com_moving_phys - com_fixed_phys)
