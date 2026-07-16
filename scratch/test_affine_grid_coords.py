import torch
import torch.nn.functional as F

theta = torch.tensor([[[1.0, 0.0, 0.1],
                       [0.0, 1.0, 0.2]]])
grid = F.affine_grid(theta, (1, 1, 3, 5), align_corners=True)
# grid shape is (1, 3, 5, 2)
print("grid shape:", grid.shape)
print("grid[0, 0, 0]:", grid[0, 0, 0])
print("grid[0, 0, 1]:", grid[0, 0, 1])
print("grid[0, 1, 0]:", grid[0, 1, 0])
