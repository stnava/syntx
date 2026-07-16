import torch
import sys
import numpy as np
sys.path.insert(0, 'src')
from syntx.syn import get_physical_grid_torch

shape = (256, 256)
spacing = (1.0, 1.0)
origin = (0.0, 0.0)
direction = np.eye(2)

grid = get_physical_grid_torch(shape, spacing, origin, direction)
print(grid.shape)
print("grid[255, 0]:", grid[0, 255, 0])
