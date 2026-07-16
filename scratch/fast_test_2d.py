import torch
import torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, 'src')
import syntx
from syntx.syn import physical_to_normalized_torch, get_physical_grid_torch
import ants
import time

# 1. Fast inputs
fi = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), use_voxels=True)

print(f"Image shape: {fi.shape}")

# 2. ANTs reference (SyNOnly)
t0 = time.time()
reg_ants = ants.registration(fixed=fi, moving=mi, type_of_transform='SyNOnly', reg_iterations=(40, 0, 0))
t1 = time.time()
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
print(f"ANTs SyNOnly MI: {mi_ants:.4f} (Time: {t1-t0:.2f}s)")

# 3. PyTorch SyN (SyNOnly)
t0 = time.time()
reg_py = syntx.syn(fi, mi, 'SyNOnly', backend='pytorch', reg_iterations=[40, 0, 0])
t1 = time.time()

# 4. Explicit Total Forward Composition
model = reg_py['model']
device = model.warp_l2r.device
dtype = model.warp_l2r.dtype

X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)

# Compose phi_1 (Fixed to Midpoint)
phi_1 = X_phys + model.warp_l2r_inv

# Evaluate phi_2_inv (warp_r2l) AT phi_1
coords_norm_phi_1 = physical_to_normalized_torch(phi_1, fi.shape, fi.spacing, fi.origin, fi.direction)
disp_2_at_phi_1 = F.grid_sample(
    model.warp_r2l.movedim(-1, 1), 
    coords_norm_phi_1, # F.grid_sample expects Y_norm, X_norm, which physical_to_normalized_torch outputs!
    padding_mode='border', align_corners=True
).movedim(1, -1)

# Full map: phi_2_inv( phi_1(x) )
phi_fwd = phi_1 + disp_2_at_phi_1
total_fwd_disp = phi_fwd - X_phys

# Apply this composed field manually in PyTorch
y_norm_fwd = physical_to_normalized_torch(phi_fwd, mi.shape, mi.spacing, mi.origin, mi.direction)
J_curr = torch.tensor(mi.numpy(), device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
moving_warped = F.grid_sample(J_curr, y_norm_fwd, padding_mode='border', align_corners=True)
warped_img = fi.new_image_like(moving_warped[0, 0].detach().cpu().numpy())
mi_py_internal = ants.image_mutual_information(fi, warped_img)
print(f"PyTorch Internal Composed MI: {mi_py_internal:.4f} (Time: {t1-t0:.2f}s)")

# 5. Export to ITK and test
from syntx.syn import save_displacement_field_nd
ants_disp = fi.new_image_like(total_fwd_disp[0].detach().cpu().numpy())
ants.image_write(ants_disp, 'scratch/fwd_composed.nii.gz')

warped_itk = ants.apply_transforms(fi, mi, ['scratch/fwd_composed.nii.gz'])
mi_py_itk = ants.image_mutual_information(fi, warped_itk)
print(f"PyTorch Exported Composed MI: {mi_py_itk:.4f}")

