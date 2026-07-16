import torch
import sys
import numpy as np
sys.path.insert(0, 'src')
from syntx.syn import physical_to_normalized_torch

# Suppose a physical point at X=255, Y=0
phys_coords = torch.tensor([[255.0, 0.0]])
target_shape = (256, 256)
spacing = (1.0, 1.0)
origin = (0.0, 0.0)
direction = np.eye(2)

coords_norm = physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction)
print("coords_norm:", coords_norm)
