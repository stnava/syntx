"""Check if internal inverse warp works."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, 'src')
import syntx, ants
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

m = rs['model']
X = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Forward: move mi to fi space using warp_l2r + affine
# This is what the export does correctly
phi_fwd = X + m.warp_l2r.data
coords_fwd = physical_to_normalized_torch(phi_fwd, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped = F.grid_sample(J_t, coords_fwd, align_corners=True, padding_mode='border')
mi_fwd_internal = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))
print(f"Forward (internal, no affine): MI={mi_fwd_internal:.4f}")

# Inverse: move fi to mi space using warp_r2l + affine_inv
# warp_r2l is defined on the FIXED grid (it moves points from fixed midpoint toward moving)
phi_inv = X + m.warp_r2l.data
coords_inv = physical_to_normalized_torch(phi_inv, fi.shape, fi.spacing, fi.origin, fi.direction)
I_warped = F.grid_sample(I_t, coords_inv, align_corners=True, padding_mode='border')
mi_inv_internal = ants.image_mutual_information(mi,
    ants.from_numpy(I_warped[0,0].numpy(), origin=mi.origin, spacing=mi.spacing, direction=mi.direction))
print(f"Inverse (internal, no affine): MI={mi_inv_internal:.4f}")

# The warp_r2l should be in the right direction?
# In SyN: warp_l2r maps fixed→mid, warp_r2l maps mid←moving
# After composition: total_r2l = warp_l2r ∘ warp_r2l_inv
# This should map fixed_space → moving_space
# So applying warp_r2l to FIXED image should produce something close to MOVING

# Check warp_r2l_inv
phi_r2l_inv = X + m.warp_r2l_inv.data
coords_r2l_inv = physical_to_normalized_torch(phi_r2l_inv, fi.shape, fi.spacing, fi.origin, fi.direction)
I_warped2 = F.grid_sample(I_t, coords_r2l_inv, align_corners=True, padding_mode='border')
mi_inv2 = ants.image_mutual_information(mi,
    ants.from_numpy(I_warped2[0,0].numpy(), origin=mi.origin, spacing=mi.spacing, direction=mi.direction))
print(f"warp_r2l_inv (should map fixed→mid): MI={mi_inv2:.4f}")

print(f"\nwarp_l2r max: {m.warp_l2r.data.abs().max():.2f}")
print(f"warp_r2l max: {m.warp_r2l.data.abs().max():.2f}")
print(f"warp_l2r_inv max: {m.warp_l2r_inv.data.abs().max():.2f}")
print(f"warp_r2l_inv max: {m.warp_r2l_inv.data.abs().max():.2f}")
