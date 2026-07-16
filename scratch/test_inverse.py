import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from syntx.syn import update_inverse_field_nd

shape = (64, 64)
spacing = (1.0, 1.0)
origin = (0.0, 0.0)
direction = np.eye(2)

w_disp = torch.zeros(1, *shape, 2)
w_disp[..., 0] = 2.0
w_disp[..., 1] = 3.0

w_inv = torch.zeros_like(w_disp)
w_inv = update_inverse_field_nd(w_disp, w_inv, steps=20, spacing=spacing, origin=origin, direction=direction)

print("w_disp at center:", w_disp[0, 32, 32])
print("w_inv at center:", w_inv[0, 32, 32])
