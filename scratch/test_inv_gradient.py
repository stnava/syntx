import torch
import sys
sys.path.insert(0, 'src')
from syntx.syn import update_inverse_field_nd

W_disp = torch.zeros(1, 64, 64, 2)
# Make X translation depend on Y
for y in range(64):
    W_disp[0, :, y, 0] = y * 0.1

W_inv = torch.zeros(1, 64, 64, 2)
W_inv = update_inverse_field_nd(W_disp, W_inv, spacing=(4.0, 4.0), origin=(0.0, 0.0), direction=[[1, 0], [0, 1]])

print("W_disp X sum:", W_disp[..., 0].sum().item())
print("W_inv X sum:", W_inv[..., 0].sum().item())
