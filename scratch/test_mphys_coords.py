import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import SyNTo, grid_to_physical_affine
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Force non-square to see the difference
fi = ants.resample_image(fi, (100, 200), use_voxels=True)
mi = ants.resample_image(mi, (100, 200), use_voxels=True)

model = SyNTo(dim=2, grid_shape=(100, 200), spacing=(1.0, 1.0), direction=np.eye(2), transform_type='Affine')
T_grid = model.affine.get_matrix().detach().cpu().numpy()
print("T_grid:\n", T_grid)

M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)
print("M_phys:\n", M_phys)
print("t_phys:\n", t_phys)

# Let's check X_phys
device = torch.device('cpu')
grid = model.get_affine_grid((100, 200), device)
print("Affine grid shape:", grid.shape)

from syntx.syn import _get_physical_grid_torch_yfirst
X_phys = _get_physical_grid_torch_yfirst((100, 200), (1.0, 1.0), (0.0, 0.0), np.eye(2), device, torch.float32)
print("X_phys shape:", X_phys.shape)
print("X_phys[0, 1, 0]:", X_phys[0, 1, 0]) # Moving along dimension 0 (width/X_size)
print("X_phys[0, 0, 1]:", X_phys[0, 0, 1]) # Moving along dimension 1 (height/Y_size)
