import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

T_grid = np.array([
    [1.0, 0.0, 0.15686],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

from syntx.syn import grid_to_physical_affine

M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)

# ANTs Warp
import tempfile
affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
tx = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
tx.set_fixed_parameters(np.zeros(2))
ants.write_transform(tx, affine_file)
warped_ants = ants.apply_transforms(fi, mi, [affine_file])

# --- Reverted Physical Grid Generation ---
def patched_get_physical_grid_torch(spatial_shape, spacing, origin, direction, device=None, dtype=None):
    dim = len(spatial_shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in spatial_shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    
    # STACK WITHOUT REVERSING!
    idxs = torch.stack(meshgrid, dim=-1)
    
    shape_t = torch.tensor(list(spatial_shape), device=device, dtype=dtype)
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    
    flat_idxs = idxs.view(-1, dim)
    scaled = flat_idxs * spacing_t
    rotated = scaled @ direction_t.t()
    phys = rotated + origin_t
    return phys.view(*spatial_shape, dim)

# PyTorch Warp
I_curr = torch.tensor(fi.numpy()).unsqueeze(0).unsqueeze(0)
J_curr = torch.tensor(mi.numpy()).unsqueeze(0).unsqueeze(0)
spatial = I_curr.shape[2:]

X_phys = patched_get_physical_grid_torch(spatial, fi.spacing, fi.origin, fi.direction, device='cpu', dtype=torch.float32)

from syntx.syn import grid_to_physical_affine_torch, physical_to_normalized_torch
M_phys_t, t_phys_t = grid_to_physical_affine_torch(
    torch.tensor(T_grid), spatial, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction
)
X_phys_batch = X_phys.unsqueeze(0)
y_phys = X_phys_batch @ M_phys_t.t() + t_phys_t
y_norm = physical_to_normalized_torch(y_phys, mi.shape, mi.spacing, mi.origin, mi.direction)
warped_py = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)

ants_np = warped_ants.numpy()
py_np = warped_py.squeeze().numpy()

# Find the best shift between py_np and ants_np along X axis (row 128)
row = 128
sig_ants = ants_np[row, :]
sig_py = py_np[row, :]

best_shift = None
min_err = float('inf')
for shift in range(-50, 50):
    if shift < 0:
        err = np.abs(sig_py[:shift] - sig_ants[-shift:]).mean()
    elif shift > 0:
        err = np.abs(sig_py[shift:] - sig_ants[:-shift]).mean()
    else:
        err = np.abs(sig_py - sig_ants).mean()
    
    if err < min_err:
        min_err = err
        best_shift = shift

print(f"Best shift along X is {best_shift} pixels with error {min_err:.4f}")

# Clean up
if os.path.exists(affine_file):
    os.remove(affine_file)
