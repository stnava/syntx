import torch
import torch.nn.functional as F
import numpy as np

# Image of shape (1, 1, 3, 5) -> X_size = 3, Y_size = 5
img = torch.zeros(1, 1, 3, 5)
img[0, 0, 2, 4] = 1.0 # The element at X=2, Y=4 is 1.0

# If we want to sample the element at X=2, Y=4
# X_norm = (2 / (3-1)) * 2 - 1 = (2/2)*2 - 1 = 1.0
# Y_norm = (4 / (5-1)) * 2 - 1 = (4/4)*2 - 1 = 1.0
# Let's test a point at X=2, Y=0
# X_norm = 1.0, Y_norm = -1.0
img[0, 0, 2, 0] = 2.0

grid = torch.tensor([[[[1.0, -1.0]]]]) # grid shape (1, 1, 1, 2)
# Does grid[..., 0] = 1.0 mean X_norm=1.0 or Y_norm=1.0?
sampled = F.grid_sample(img, grid, align_corners=True)
print("Sampled with grid=[1.0, -1.0]:", sampled.item())

grid = torch.tensor([[[[-1.0, 1.0]]]])
sampled = F.grid_sample(img, grid, align_corners=True)
print("Sampled with grid=[-1.0, 1.0]:", sampled.item())

