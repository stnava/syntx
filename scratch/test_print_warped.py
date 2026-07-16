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

# Let's apply a simple translation: shift moving image in +X direction by 20 pixels
# T_grid = [1, 0, tx], [0, 1, ty]
# If we want a +20 pixel shift in X in voxel space, tx = 20 / 127.5 = 0.15686
T_grid = np.array([
    [1.0, 0.0, 0.15686],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)
print("t_phys:", t_phys)

import tempfile
affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
tx = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
tx.set_fixed_parameters(np.zeros(2))
ants.write_transform(tx, affine_file)

warped_ants = ants.apply_transforms(fi, mi, [affine_file])

# PyTorch
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

ants_np = warped_ants.numpy()
py_np = warped_py.squeeze().numpy()

# Print slices of the center line (row 128)
print("\nRow 128 of ANTs warped:")
print(ants_np[128, 120:140])

print("\nRow 128 of PyTorch warped:")
print(py_np[128, 120:140])

# Clean up
if os.path.exists(affine_file):
    os.remove(affine_file)
