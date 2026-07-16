import torch
import torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, 'src')
import syntx
from syntx.syn import physical_to_normalized_torch, get_physical_grid_torch
import ants

fi = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), use_voxels=True)

reg_py = syntx.syn(fi, mi, 'SyNOnly', backend='pytorch', reg_iterations=[40, 20, 0], affine_iterations=[0, 0, 0])
model = reg_py['model']
device = model.warp_l2r.device
dtype = model.warp_l2r.dtype

X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)
phi_fwd = X_phys + model.warp_l2r
total_fwd_disp = model.warp_l2r

y_norm_fwd = physical_to_normalized_torch(phi_fwd, mi.shape, mi.spacing, mi.origin, mi.direction)
J_curr = torch.tensor(mi.numpy(), device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
moving_warped = F.grid_sample(J_curr, y_norm_fwd, padding_mode='border', align_corners=True)
warped_img = fi.new_image_like(moving_warped[0, 0].detach().cpu().numpy())
mi_py_internal = ants.image_mutual_information(fi, warped_img)

fwd_file = 'scratch/fwd_composed.nii.gz'
fwd_img = ants.from_numpy(total_fwd_disp[0].detach().cpu().numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, fwd_file)

warped_itk = ants.apply_transforms(fi, mi, [fwd_file])
mi_py_itk = ants.image_mutual_information(fi, warped_itk)

reg_ants = ants.registration(fixed=fi, moving=mi, type_of_transform='SyNOnly', reg_iterations=(40, 20, 0))
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])

print(f"Internal PyTorch MI: {mi_py_internal:.4f}")
print(f"ITK Application MI: {mi_py_itk:.4f}")
print(f"ANTs SyNOnly MI: {mi_ants:.4f}")
