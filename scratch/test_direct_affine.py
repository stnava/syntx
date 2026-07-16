import torch
import torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, 'src')
import syntx
from syntx.syn import physical_to_normalized_torch, get_physical_grid_torch, grid_to_physical_affine_torch
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Just apply a known translation
T_grid = torch.tensor([[1.0, 0.0, 0.1],
                       [0.0, 1.0, 0.0]])
device = torch.device('cpu')
dtype = torch.float32

M_phys, t_phys = grid_to_physical_affine_torch(
    T_grid, fi.shape, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction
)

X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)
y_phys = torch.matmul(X_phys, M_phys.t()) + t_phys
y_norm = physical_to_normalized_torch(y_phys, mi.shape, mi.spacing, mi.origin, mi.direction)

J_curr = torch.tensor(mi.numpy(), device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
moving_warped = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)
moving_warped_np = moving_warped[0, 0].detach().cpu().numpy()

warped_img = fi.new_image_like(moving_warped_np)
mi_internal = ants.image_mutual_information(fi, warped_img)

affine_tx = ants.create_ants_transform(transform_type='AffineTransform', dimension=2)
params = affine_tx.parameters
params[4] = t_phys[0].item() # X translation
params[5] = t_phys[1].item() # Y translation
affine_tx.set_parameters(params)
ants.write_transform(affine_tx, 'scratch/affine.mat')
warped_ants = ants.apply_transforms(fi, mi, ['scratch/affine.mat'])
mi_ants = ants.image_mutual_information(fi, warped_ants)

print(f"Internal PyTorch MI: {mi_internal:.4f}")
print(f"ANTs MI: {mi_ants:.4f}")
