import torch

def new_physical_to_normalized(phys_coords, target_shape, spacing, origin, direction):
    device = phys_coords.device
    dtype = phys_coords.dtype
    dim = len(target_shape)
    
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    target_shape_t = torch.tensor(target_shape, device=device, dtype=dtype)
    
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    
    norm_coords = (voxel_coords / (target_shape_t - 1)) * 2.0 - 1.0
    norm_coords_flipped = torch.flip(norm_coords, dims=[-1])
    return norm_coords_flipped.view(phys_coords.shape)

import numpy as np
phys = torch.tensor([[255.0, 0.0]])
res = new_physical_to_normalized(phys, (256, 256), (1.0, 1.0), (0.0, 0.0), np.eye(2))
print("new:", res)
