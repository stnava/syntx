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

# Define a grid matrix in XYZ order
T_grid = np.array([
    [1.0, 0.0, 0.1],
    [0.0, 1.0, -0.2],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Convert to physical
M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)
print("Computed M_phys:\n", M_phys)
print("Computed t_phys:", t_phys)

# Warp in ANTs using this physical transform
import tempfile
affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
tx = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
tx.set_fixed_parameters(np.zeros(2))
ants.write_transform(tx, affine_file)

warped_ants = ants.apply_transforms(fi, mi, [affine_file])

# Warp in PyTorch using T_grid
I_curr = torch.tensor(fi.numpy()).unsqueeze(0).unsqueeze(0)
J_curr = torch.tensor(mi.numpy()).unsqueeze(0).unsqueeze(0)

device = 'cpu'
dtype = torch.float32
spatial = I_curr.shape[2:]

X_phys = get_physical_grid_torch(spatial, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)
# T_grid to tensor
T_tensor = torch.tensor(T_grid, device=device, dtype=dtype)
from syntx.syn import grid_to_physical_affine_torch
M_phys_t, t_phys_t = grid_to_physical_affine_torch(
    T_tensor, spatial, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction
)
y_phys = X_phys @ M_phys_t.t() + t_phys_t
y_norm = physical_to_normalized_torch(y_phys, mi.shape, mi.spacing, mi.origin, mi.direction)
warped_py = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)

# Compare warped images
ants_np = warped_ants.numpy()
py_np = warped_py.squeeze().numpy()

diff = np.abs(ants_np - py_np)
print("\nMax absolute difference between ANTs warp and PyTorch warp:", diff.max())

# Clean up
if os.path.exists(affine_file):
    os.remove(affine_file)
