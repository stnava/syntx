import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx
from syntx.syn import physical_to_normalized_torch, _get_physical_grid_torch_yfirst
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Run SyN registration
reg_iterations = [40, 20, 0]
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)

# Evaluate Current (which just uses warp_l2r)
fwd_file = reg_py['fwdtransforms'][0]
affine_file = reg_py['fwdtransforms'][1]
w_syn = ants.apply_transforms(fi, mi, [fwd_file, affine_file])
mi_syn = ants.image_mutual_information(fi, w_syn)
print(f"PyTorch SyN MI (Current - warp_l2r): {mi_syn:.4f}")

# Now let's extract the PyTorch model's internal fields and compose them properly
model = reg_py['model']
device = model.warp_l2r.device
dtype = model.warp_l2r.dtype

# Create X_phys
sp_ordered = tuple(reversed(fi.spacing))
orig_ordered = tuple(reversed(fi.origin))
dir_ordered = np.array(fi.direction)[::-1, ::-1].copy()

X_phys = _get_physical_grid_torch_yfirst(
    model.grid_shape, sp_ordered, orig_ordered, dir_ordered, device, dtype
)

# physical_to_normalized expects un-reversed arguments
spacing_t = torch.tensor(fi.spacing, device=device, dtype=dtype)
origin_t = torch.tensor(fi.origin, device=device, dtype=dtype)
direction_t = torch.tensor(fi.direction, device=device, dtype=dtype)
fixed_shape_t = torch.tensor(fi.shape, device=device, dtype=torch.int32)

# Compute full forward field
phi_phys = X_phys + model.warp_l2r_inv
phi_norm = physical_to_normalized_torch(phi_phys, fixed_shape_t, spacing_t, origin_t, direction_t)

warp_r2l_sampled = F.grid_sample(
    model.warp_r2l.movedim(-1, 1), 
    phi_norm, 
    padding_mode='border', 
    align_corners=True
).movedim(1, -1)

total_fwd_deformable = model.warp_l2r_inv + warp_r2l_sampled

# Save to ITK file
disp_l2r_t = total_fwd_deformable.detach().cpu().numpy()[0].astype(np.float32)
# No need to swap, it's already in (X, Y)
fwd_img = ants.from_numpy(disp_l2r_t, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, 'scratch/fwd_total.nii.gz')

w_syn_total = ants.apply_transforms(fi, mi, ['scratch/fwd_total.nii.gz', affine_file])
mi_syn_total = ants.image_mutual_information(fi, w_syn_total)
print(f"PyTorch SyN MI (Total composed): {mi_syn_total:.4f}")

