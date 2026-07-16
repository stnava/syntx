import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import _get_physical_grid_torch_yfirst

spacing = (2.5, 1.5) # sp_ordered: (spacing_y, spacing_x)
origin = (0.0, 0.0)
direction = np.eye(2)
device = torch.device('cpu')

X_phys = _get_physical_grid_torch_yfirst((100, 200), spacing, origin, direction, device, torch.float32)
print("X_phys shape:", X_phys.shape)
print("X_phys[0, 1, 0]:", X_phys[0, 1, 0]) # Moving along width
print("X_phys[0, 0, 1]:", X_phys[0, 0, 1]) # Moving along height
