"""Diagnose the even/odd oscillation pattern in loss values.
This suggests a systematic issue with the gradient direction."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'src')
import ants
from syntx.syn import (get_physical_grid_torch, physical_to_normalized_torch,
                        _physical_to_normalized_torch_yfirst,
                        mattes_mi_loss_nd, local_ncc_loss_nd,
                        prepare_mid_images_and_gradients_torch,
                        separable_gaussian_filter,
                        update_inverse_field_nd, get_boundary_mask)

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

# Set up at level 8 (32x32) resolution
scale = 8
down_shape = tuple(max(1, d // scale) for d in fi.shape)
curr_spacing = tuple(sp * (N-1)/(n-1) if n > 1 else sp 
                     for sp, N, n in zip(fi.spacing, fi.shape, down_shape))

I_full = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_full = torch.tensor(mi_affine.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

I_curr = F.interpolate(I_full, size=down_shape, mode='bilinear', align_corners=True)
J_curr = F.interpolate(J_full, size=down_shape, mode='bilinear', align_corners=True)

print(f"Working at {down_shape} resolution, spacing={[f'{s:.2f}' for s in curr_spacing]}")

# Initialize
warp_l2r = torch.zeros(1, *down_shape, 2, dtype=torch.float32, requires_grad=True)
warp_r2l = torch.zeros(1, *down_shape, 2, dtype=torch.float32, requires_grad=True)
warp_l2r_inv = torch.zeros_like(warp_l2r, requires_grad=False)
warp_r2l_inv = torch.zeros_like(warp_r2l, requires_grad=False)

X_phys = get_physical_grid_torch(down_shape, curr_spacing, fi.origin, fi.direction)
b_mask = get_boundary_mask(down_shape, 'cpu', torch.float32)
M_phys = torch.eye(2, dtype=torch.float32)
t_phys = torch.zeros(2, dtype=torch.float32)

fixed_shape_t = torch.tensor(list(down_shape), dtype=torch.float32)
spacing_rev = tuple(reversed(curr_spacing))
origin_rev = tuple(reversed(fi.origin))
direction_rev = np.asarray(fi.direction)[::-1, ::-1].copy()
fixed_spacing_t = torch.tensor(spacing_rev, dtype=torch.float32)
fixed_origin_t = torch.tensor(origin_rev, dtype=torch.float32)
fixed_direction_t = torch.tensor(direction_rev, dtype=torch.float32)

moving_shape_t = fixed_shape_t.clone()
moving_spacing_t = fixed_spacing_t.clone()
moving_origin_t = fixed_origin_t.clone()
moving_direction_t = fixed_direction_t.clone()

curr_spacing_fixed_t = torch.tensor(list(reversed(curr_spacing)), dtype=torch.float32)

cfl_voxels = 0.1
fluid_sigma = 3.0

print(f"\nManual SyN loop (cfl={cfl_voxels}):")
print(f"{'Epoch':>5} {'Loss':>10} {'warp_l2r_max':>14} {'delta_max':>12} {'lr':>10}")

for epoch in range(10):
    if warp_l2r.grad is not None: warp_l2r.grad.zero_()
    if warp_r2l.grad is not None: warp_r2l.grad.zero_()
    
    I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_torch(
        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
        X_phys,
        fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
        moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
        curr_spacing, curr_spacing,
        M_phys, t_phys, None
    )
    
    loss = mattes_mi_loss_nd(J_mid, I_mid)
    loss.backward()
    loss_val = loss.item()
    
    with torch.no_grad():
        grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, fluid_sigma, spacing=curr_spacing)
        grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, fluid_sigma, spacing=curr_spacing)
        
        grad_l_voxel = grad_l * curr_spacing_fixed_t
        grad_r_voxel = grad_r * curr_spacing_fixed_t
        max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max() + 1e-8
        
        lr_l = cfl_voxels / max_norm_l
        
        delta_l = lr_l * grad_l_voxel * curr_spacing_fixed_t
        delta_r = lr_l * grad_r_voxel * curr_spacing_fixed_t  # Using same lr
        
        delta_max_mm = delta_l.abs().max().item()
        
        print(f"{epoch:>5d} {loss_val:>10.4f} {warp_l2r.abs().max().item():>14.4f} {delta_max_mm:>12.4f} {lr_l.item():>10.4f}")
        
        # Greedy SyN composition
        coords_phys_l = X_phys - delta_l
        coords_norm_l = physical_to_normalized_torch(
            coords_phys_l, down_shape, curr_spacing, fi.origin, fi.direction
        )
        warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
        warp_l2r.copy_(warp_l2r_sampled - delta_l)
        
        coords_phys_r = X_phys - delta_r
        coords_norm_r = physical_to_normalized_torch(
            coords_phys_r, down_shape, curr_spacing, fi.origin, fi.direction
        )
        warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
        warp_r2l.copy_(warp_r2l_sampled - delta_r)
        
        warp_l2r.mul_(b_mask)
        warp_r2l.mul_(b_mask)
        
        # Diffeomorphic projection
        warp_l2r_inv = update_inverse_field_nd(
            warp_l2r, warp_l2r_inv.detach(), steps=5,
            spacing=curr_spacing, origin=fi.origin, direction=fi.direction
        )
        warp_l2r.copy_(update_inverse_field_nd(
            warp_l2r_inv, warp_l2r.detach(), steps=5,
            spacing=curr_spacing, origin=fi.origin, direction=fi.direction
        ))
        
        warp_r2l_inv = update_inverse_field_nd(
            warp_r2l, warp_r2l_inv.detach(), steps=5,
            spacing=curr_spacing, origin=fi.origin, direction=fi.direction
        )
        warp_r2l.copy_(update_inverse_field_nd(
            warp_r2l_inv, warp_r2l.detach(), steps=5,
            spacing=curr_spacing, origin=fi.origin, direction=fi.direction
        ))

print(f"\nFinal warp_l2r max: {warp_l2r.abs().max().item():.4f}mm")
print(f"Final warp_r2l max: {warp_r2l.abs().max().item():.4f}mm")

# Now check: does the midpoint loss actually measure what we think?
# The loss computed at the midpoint should correlate with the full-resolution MI
print(f"\n=== Key insight about the oscillation ===")
print(f"Even epochs: loss is lower (e.g., -1.08)")
print(f"Odd epochs:  loss is higher (e.g., -1.19)")
print(f"This 2-cycle pattern suggests the gradient update OVERSHOOTS")
print(f"and the diffeomorphic projection doesn't fully correct it")
print(f"So the warp field alternates between two states.")

