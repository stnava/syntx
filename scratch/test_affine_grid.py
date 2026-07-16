import torch
import torch.nn.functional as F

# 1x1x3x5
theta = torch.tensor([[[1.0, 0.0, 0.1],
                       [0.0, 1.0, 0.2]]])
grid = F.affine_grid(theta, (1, 1, 3, 5), align_corners=True)
print(grid[0, 0, 0])
