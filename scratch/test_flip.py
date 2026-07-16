import torch
import torch.nn.functional as F

img = torch.zeros(1, 1, 3, 5) # (N, C, X, Y)
img[0, 0, 2, 0] = 2.0         # Element at X=2, Y=0

# X_norm = 1.0, Y_norm = -1.0
coords_xy = torch.tensor([[[[1.0, -1.0]]]]) # (X_norm, Y_norm)

# F.grid_sample expects (Y_norm, X_norm) because X is dim 2 and Y is dim 3 (last dim)
sampled = F.grid_sample(img, coords_xy.flip(-1), align_corners=True)
print("Sampled with flip:", sampled.item())
