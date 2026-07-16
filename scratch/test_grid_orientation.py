import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import get_physical_grid_torch

shape = (10, 20) # width=10, height=20
spacing = (2.0, 3.0) # sx=2, sy=3
origin = (5.0, 7.0) # ox=5, oy=7
direction = np.eye(2)

grid = get_physical_grid_torch(shape, spacing, origin, direction)
# grid shape should be (1, 10, 20, 2)

print("Grid shape:", grid.shape)
print("grid[0, 0, 0]:", grid[0, 0, 0].tolist()) # Expected: (x0, y0)
print("grid[0, 1, 0]:", grid[0, 1, 0].tolist()) # Move along dim 1 (width)
print("grid[0, 0, 1]:", grid[0, 0, 1].tolist()) # Move along dim 2 (height)
