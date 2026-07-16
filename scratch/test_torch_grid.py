import torch
import torch.nn.functional as F

# Create a 2x3 image: H=2 (width), W=3 (height)
img = torch.tensor([[[
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]]])

grid = torch.zeros(1, 1, 1, 2)
# grid[..., 0] is x (last dim), grid[..., 1] is y (first dim)
grid[0, 0, 0, 0] = -1.0 # x = left (0)
grid[0, 0, 0, 1] = -1.0 # y = top (0)

out = F.grid_sample(img, grid, align_corners=True)
print("(-1, -1):", out[0, 0, 0, 0].item()) # Expect 1.0

grid[0, 0, 0, 0] = 1.0 # x = right (W-1) -> column 2
grid[0, 0, 0, 1] = -1.0 # y = top (H-1) -> row 0
out = F.grid_sample(img, grid, align_corners=True)
print("(1, -1):", out[0, 0, 0, 0].item()) # Expect 3.0 (if x is last dim)

grid[0, 0, 0, 0] = -1.0 # x = left (W-1) -> column 0
grid[0, 0, 0, 1] = 1.0 # y = bottom (H-1) -> row 1
out = F.grid_sample(img, grid, align_corners=True)
print("(-1, 1):", out[0, 0, 0, 0].item()) # Expect 4.0
