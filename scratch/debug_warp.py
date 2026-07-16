import ants
import numpy as np
import torch
import torch.nn.functional as F

fixed = ants.image_read(ants.get_ants_data('r16'))
fixed_label = ants.threshold_image(fixed, 'Otsu', 3)

tx = ants.new_ants_transform(precision='float', dimension=2)
tx.set_parameters([1.0, 0.0, 0.0, 1.0, 5.0, -3.0])

import tempfile
import os
fd, temp_tx_path = tempfile.mkstemp(suffix='.mat')
os.close(fd)
try:
    ants.write_transform(tx, temp_tx_path)
    warped_ants = ants.apply_transforms(
        fixed=fixed, 
        moving=fixed_label, 
        transformlist=[temp_tx_path], 
        interpolator='nearestNeighbor'
    )
finally:
    if os.path.exists(temp_tx_path):
        os.remove(temp_tx_path)

# Let's inspect warped_ants non-zero pixels
ants_arr = warped_ants.numpy()
y_indices, x_indices = np.where(ants_arr > 0)
print("ANTs non-zero pixels count:", len(y_indices))
if len(y_indices) > 0:
    print("ANTs first 5 non-zero pixels (y, x):", list(zip(y_indices[:5], x_indices[:5])))

# Let's do the PyTorch warp
device = 'cpu'
dtype = torch.float32
label_tensor = torch.tensor(fixed_label.numpy(), dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

# Build grid
def get_physical_grid_torch(shape, spacing, origin, direction):
    dim = len(shape)
    grids = [torch.arange(s, dtype=dtype) for s in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    meshgrid_reversed = list(reversed(meshgrid))
    idxs = torch.stack(meshgrid_reversed, dim=-1)
    spacing_t = torch.tensor(spacing, dtype=dtype)
    origin_t = torch.tensor(origin, dtype=dtype)
    direction_t = torch.tensor(direction, dtype=dtype)
    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)

def physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction):
    dim = len(target_shape)
    spacing_t = torch.tensor(spacing, dtype=dtype)
    origin_t = torch.tensor(origin, dtype=dtype)
    direction_t = torch.tensor(direction, dtype=dtype)
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    shape_t = torch.tensor(list(reversed(target_shape)), dtype=dtype)
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    return norm_coords.view(phys_coords.shape)

identity_phys = get_physical_grid_torch(fixed.shape, fixed.spacing, fixed.origin, fixed.direction)
flat_phys = identity_phys.view(-1, 2)
params = tx.parameters
M_phys = params[:4].reshape(2, 2)
t_phys = params[4:]

# Print parameters
print("M_phys:\n", M_phys)
print("t_phys:", t_phys)

M_phys_t = torch.tensor(M_phys, dtype=dtype)
t_phys_t = torch.tensor(t_phys, dtype=dtype)

# Check: apply transform. Let's try y = M @ x + t vs y = x + t or other combinations.
# In ITK, standard mapping: y = M @ (x - center) + center + t
center = torch.tensor(tx.fixed_parameters, dtype=dtype)
y_phys = (flat_phys - center) @ M_phys_t.t() + center + t_phys_t
y_phys = y_phys.view(identity_phys.shape)

norm_grid = physical_to_normalized_torch(y_phys, fixed_label.shape, fixed_label.spacing, fixed_label.origin, fixed_label.direction)
warped_torch_tensor = F.grid_sample(label_tensor, norm_grid, mode='nearest', padding_mode='border', align_corners=True)
torch_arr = warped_torch_tensor.squeeze().numpy()

y_t, x_t = np.where(torch_arr > 0)
print("PyTorch non-zero pixels count:", len(y_t))
if len(y_t) > 0:
    print("PyTorch first 5 non-zero pixels (y, x):", list(zip(y_t[:5], x_t[:5])))

# Compute Dice
overlap = ants.label_overlap_measures(warped_ants, ants.from_numpy(torch_arr, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction))
print("Dice:", overlap['MeanOverlap'].iloc[0])
