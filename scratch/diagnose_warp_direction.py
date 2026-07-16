"""Check if the warp field actually improves registration when applied internally."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'src')
import syntx
import ants
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch, mattes_mi_loss_nd

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

print("Running syntx SyN with grad_step=0.1 (stable)...")
reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.1)

model = reg_pt['model']
warp_l2r = model.warp_l2r.data  # (1, H, W, 2) -- physical mm, YX order
warp_r2l = model.warp_r2l.data

I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_affine.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)

print(f"\nwarp_l2r range: [{warp_l2r.min():.3f}, {warp_l2r.max():.3f}]")
print(f"warp_r2l range: [{warp_r2l.min():.3f}, {warp_r2l.max():.3f}]")

# Option 1: Apply warp_l2r as forward (current code behavior)
phi_fwd = X_phys + warp_l2r
coords_fwd = physical_to_normalized_torch(phi_fwd, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped_1 = F.grid_sample(J_t, coords_fwd, align_corners=True, padding_mode='border')
mi_1 = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_1[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))

# Option 2: Apply -warp_l2r
phi_neg = X_phys - warp_l2r
coords_neg = physical_to_normalized_torch(phi_neg, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped_2 = F.grid_sample(J_t, coords_neg, align_corners=True, padding_mode='border')
mi_2 = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_2[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))

# Option 3: Apply warp_r2l
phi_r2l = X_phys + warp_r2l
coords_r2l = physical_to_normalized_torch(phi_r2l, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped_3 = F.grid_sample(J_t, coords_r2l, align_corners=True, padding_mode='border')
mi_3 = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_3[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))

# Option 4: Apply -warp_r2l
phi_neg_r2l = X_phys - warp_r2l
coords_neg_r2l = physical_to_normalized_torch(phi_neg_r2l, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped_4 = F.grid_sample(J_t, coords_neg_r2l, align_corners=True, padding_mode='border')
mi_4 = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_4[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))

# Option 5: No warp (identity)
coords_id = physical_to_normalized_torch(X_phys, fi.shape, fi.spacing, fi.origin, fi.direction)
J_id = F.grid_sample(J_t, coords_id, align_corners=True, padding_mode='border')
mi_id = ants.image_mutual_information(fi,
    ants.from_numpy(J_id[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))

print(f"\nInternal PyTorch warping results (lower MI = better):")
print(f"  Identity (no warp): {mi_id:.4f}")
print(f"  +warp_l2r:          {mi_1:.4f}")
print(f"  -warp_l2r:          {mi_2:.4f}")
print(f"  +warp_r2l:          {mi_3:.4f}")
print(f"  -warp_r2l:          {mi_4:.4f}")

# Also check the PyTorch MI directly (same units as internal loss)
for name, warped in [('identity', J_id), ('+warp_l2r', J_warped_1), ('-warp_l2r', J_warped_2),
                      ('+warp_r2l', J_warped_3), ('-warp_r2l', J_warped_4)]:
    mi_pt = mattes_mi_loss_nd(I_t, warped).item()
    print(f"  PyTorch MI ({name}): {mi_pt:.4f}")
