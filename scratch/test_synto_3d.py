import torch
import torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, 'src')
import syntx
import ants
from syntx.syn import physical_to_normalized_torch, get_physical_grid_torch

fi = ants.image_read(ants.get_ants_data('mni')).resample_image((64, 64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('mni')).resample_image((64, 64, 64), use_voxels=True)
tx = ants.create_ants_transform(transform_type='Euler3DTransform', translation=(60, 0, 0)) # Shift by 60mm!
ants.write_transform(tx, 'scratch/tx.mat')
mi = ants.apply_transforms(fi, mi, transformlist=['scratch/tx.mat'])

reg_py = syntx.syn(fi, mi, 'SyNTo', backend='pytorch', reg_iterations=[0, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi')
model = reg_py['model']
T_grid = model.affine.get_matrix()
print("PyTorch Affine matrix:\n", T_grid.detach().numpy())

# Test MI manually
fi_tensor = torch.tensor(fi.numpy()).unsqueeze(0).unsqueeze(0)
mi_tensor = torch.tensor(mi.numpy()).unsqueeze(0).unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction, device='cpu', dtype=torch.float32)

t_fg = torch.tensor([0.4995041, -0.10627747, -48.177307])
y_phys_fg = X_phys + t_fg
y_norm_fg = physical_to_normalized_torch(y_phys_fg, mi.shape, mi.spacing, mi.origin, mi.direction)
J_warped_fg = F.grid_sample(mi_tensor, y_norm_fg, padding_mode='border', align_corners=True)

diff_fg = np.abs(fi.numpy() - J_warped_fg[0, 0].numpy()).mean()
print("L1 Diff (fi, PyTorch warped_fg):", diff_fg)

from syntx.syn import mattes_mi_loss_nd
print("PyTorch MI (FG WARP):", mattes_mi_loss_nd(J_warped_fg, fi_tensor, num_bins=32).item())

