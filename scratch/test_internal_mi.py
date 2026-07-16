import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx
from syntx.syn import physical_to_normalized_torch, get_physical_grid_torch
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Run SyN registration
reg_iterations = [40, 20, 0]
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)

# Compute MI using PyTorch's internal grid sample
model = reg_py['model']
device = model.warp_l2r.device
dtype = model.warp_l2r.dtype

# Reconstruct everything
I_curr = torch.tensor(fi.numpy(), device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
J_curr = torch.tensor(mi.numpy(), device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

# 1. Total forward field (already computed in model.warp_l2r)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)
phi_phys = X_phys + model.warp_l2r

# 2. Apply affine matrix
T_grid = model.affine.get_matrix().detach()
# wait, phi_phys is in pre-affine moving space.
# We need to map it to normalized coordinates, then apply T_grid!
# Wait! y_phys = phi_phys @ M_phys.t() + t_phys
from syntx.syn import grid_to_physical_affine_torch
M_phys, t_phys = grid_to_physical_affine_torch(
    T_grid, fi.shape, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction
)
y_phys = torch.matmul(phi_phys, M_phys.t()) + t_phys

# 3. Sample moving image
y_norm = physical_to_normalized_torch(
    y_phys, torch.tensor(mi.shape, device=device, dtype=torch.int32), 
    torch.tensor(mi.spacing, device=device, dtype=dtype),
    torch.tensor(mi.origin, device=device, dtype=dtype),
    torch.tensor(mi.direction, device=device, dtype=dtype)
)

moving_warped = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)
moving_warped_np = moving_warped[0, 0].detach().cpu().numpy()

# 4. Compute MI
warped_img = fi.new_image_like(moving_warped_np)
mi_internal = ants.image_mutual_information(fi, warped_img)
print(f"Internal PyTorch MI: {mi_internal:.4f}")
print(f"External ANTs MI: {ants.image_mutual_information(fi, ants.apply_transforms(fi, mi, [reg_py['fwdtransforms'][0], reg_py['fwdtransforms'][1]])):.4f}")

