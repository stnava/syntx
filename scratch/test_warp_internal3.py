import sys
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'src')
import syntx
import ants
from syntx.syn import get_physical_grid_torch, grid_to_physical_affine, physical_to_normalized_torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch', reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi', verbose=False, grad_step=0.1, fluid_sigma=np.sqrt(3.0))

print("Forward transforms:", reg_pt['fwdtransforms'])
warped_ants = ants.apply_transforms(fi, mi_affine, reg_pt['fwdtransforms'])
print("PyTorch SyN MI (ANTs apply):", ants.image_mutual_information(fi, warped_ants))

# Warp internally!
model = reg_pt['model']
total_fwd = model.warp_l2r.data.unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction, device=total_fwd.device, dtype=total_fwd.dtype)

# total_fwd maps fixed to moving_curr
Z = X_phys + total_fwd

T_grid = model.affine.get_matrix().detach().cpu().numpy()
M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi_affine)
M_phys = torch.tensor(M_phys, device=total_fwd.device, dtype=total_fwd.dtype)
t_phys = torch.tensor(t_phys, device=total_fwd.device, dtype=total_fwd.dtype)

y_phys = Z @ M_phys.t() + t_phys

y_norm = physical_to_normalized_torch(y_phys, mi_affine.shape, mi_affine.spacing, mi_affine.origin, mi_affine.direction)
mi_affine_t = torch.tensor(mi_affine.numpy(), device=total_fwd.device, dtype=total_fwd.dtype).unsqueeze(0).unsqueeze(0)
warped_internal_t = F.grid_sample(mi_affine_t, y_norm, padding_mode='border', align_corners=True)
warped_internal = ants.from_numpy(warped_internal_t[0, 0].cpu().numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction)

print("PyTorch SyN MI (Internal apply):", ants.image_mutual_information(fi, warped_internal))
