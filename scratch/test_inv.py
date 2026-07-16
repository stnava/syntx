import torch
import sys
sys.path.insert(0, 'src')
from syntx.syn import update_inverse_field_nd, get_physical_grid_torch

W_disp = torch.zeros(1, 64, 64, 2)
W_disp[..., 0] = 10.0 # X translation
W_disp[..., 1] = 0.0

W_inv = torch.zeros(1, 64, 64, 2)
W_inv = update_inverse_field_nd(W_disp, W_inv, spacing=(4.0, 4.0), origin=(0.0, 0.0), direction=[[1, 0], [0, 1]])

print("W_inv X:", W_inv[0, 32, 32, 0].item())
print("W_inv Y:", W_inv[0, 32, 32, 1].item())
