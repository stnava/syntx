import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F
from syntx.syn import grid_to_physical_affine, get_physical_grid_torch, physical_to_normalized_torch

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

T_grid = np.array([
    [1.0, 0.0, 0.15686],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# PyTorch Warp
I_curr = torch.tensor(fi.numpy()).unsqueeze(0).unsqueeze(0)
J_curr = torch.tensor(mi.numpy()).unsqueeze(0).unsqueeze(0)
spatial = I_curr.shape[2:]

X_phys = get_physical_grid_torch(spatial, fi.spacing, fi.origin, fi.direction, device='cpu', dtype=torch.float32)
from syntx.syn import grid_to_physical_affine_torch
M_phys_t, t_phys_t = grid_to_physical_affine_torch(
    torch.tensor(T_grid), spatial, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction
)
y_phys = X_phys @ M_phys_t.t() + t_phys_t
y_norm = physical_to_normalized_torch(y_phys, mi.shape, mi.spacing, mi.origin, mi.direction)
warped_py = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)

py_np = warped_py.squeeze().numpy()
mi_np = mi.numpy()

# Find the best shift of py_np along axis 0 (X) and axis 1 (Y)
best_shift_0 = None
min_err_0 = float('inf')
for shift in range(-50, 50):
    if shift < 0:
        err = np.abs(py_np[:shift, 128] - mi_np[-shift:, 128]).mean()
    elif shift > 0:
        err = np.abs(py_np[shift:, 128] - mi_np[:-shift, 128]).mean()
    else:
        err = np.abs(py_np[:, 128] - mi_np[:, 128]).mean()
    
    if err < min_err_0:
        min_err_0 = err
        best_shift_0 = shift

best_shift_1 = None
min_err_1 = float('inf')
for shift in range(-50, 50):
    if shift < 0:
        err = np.abs(py_np[128, :shift] - mi_np[128, -shift:]).mean()
    elif shift > 0:
        err = np.abs(py_np[128, shift:] - mi_np[128, :-shift]).mean()
    else:
        err = np.abs(py_np[128, :] - mi_np[128, :]).mean()
    
    if err < min_err_1:
        min_err_1 = err
        best_shift_1 = shift

print(f"For PyTorch: best shift along Axis 0 (X) is {best_shift_0} pixels with error {min_err_0:.4f}")
print(f"For PyTorch: best shift along Axis 1 (Y) is {best_shift_1} pixels with error {min_err_1:.4f}")
