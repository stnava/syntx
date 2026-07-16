import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

width, height = 256, 256
spacing = (1.0, 1.0)
origin = (0.0, 0.0)
direction = np.eye(2)
grid = get_physical_grid_torch((width, height), spacing, origin, direction)
norm_grid = physical_to_normalized_torch(grid, (width, height), spacing, origin, direction)

print("grid[0, 50, 200]:", grid[0, 50, 200].tolist())
print("norm_grid[0, 50, 200]:", norm_grid[0, 50, 200].tolist())
