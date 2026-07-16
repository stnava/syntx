import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import SyNTo, grid_to_physical_affine_torch
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Load the affine we got earlier
T_grid = torch.tensor([[ 1.05179489e+00, -3.35011780e-02,  1.04148746e-01],
 [ 5.22676855e-05,  9.33740437e-01, -2.24237919e-01],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
T_grid = T_grid[:2, :]

M_phys, t_phys = grid_to_physical_affine_torch(
    T_grid, tuple(fi.shape), tuple(fi.spacing), tuple(fi.origin), np.array(fi.direction),
    tuple(mi.shape), tuple(mi.spacing), tuple(mi.origin), np.array(mi.direction)
)
print("T_grid:\n", T_grid)
print("M_phys:\n", M_phys)
print("t_phys:\n", t_phys)

