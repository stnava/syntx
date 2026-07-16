import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F
import tempfile

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

# 1. ANTs Affine registration
print("Pre-aligning images...")
reg_ants_affine = ants.registration(fi, mi, 'Affine')
mi_affine = ants.apply_transforms(fi, mi, reg_ants_affine['fwdtransforms'])
print("Initial ANTs MI:", ants.image_mutual_information(fi, mi_affine))

# 2. Initialize Syntx PyTorch SyN model
from syntx.syn import SyNTo
device = 'cpu'
dtype = torch.float32
dim = 2

# Normalize images to match syn.py
fi_np = fi.numpy()
mi_np = mi_affine.numpy()
fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

I_tensor = torch.tensor(fi_norm, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
J_tensor = torch.tensor(mi_norm, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

model = SyNTo(
    dim=dim, grid_shape=fi.shape, spacing=fi.spacing, direction=fi.direction,
    fluid_sigma=3.0, elastic_sigma=0.0, transform_type='SyN'
)

# Initialize warp fields
spatial = J_tensor.shape[2:]
warp_l2r = torch.zeros(1, *spatial, dim, device=device, dtype=dtype).requires_grad_(True)
warp_r2l = torch.zeros(1, *spatial, dim, device=device, dtype=dtype).requires_grad_(True)
warp_l2r_inv = torch.zeros_like(warp_l2r)
warp_r2l_inv = torch.zeros_like(warp_r2l)

from syntx.syn import (
    get_physical_grid_torch, prepare_mid_images_and_gradients_torch,
    local_ncc_loss_nd, separable_gaussian_filter, get_boundary_mask,
    update_inverse_field_nd, physical_to_normalized_torch
)

X_phys = get_physical_grid_torch(spatial, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)
b_mask = get_boundary_mask(spatial, device, dtype)

fixed_shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
fixed_spacing_t = torch.tensor(list(reversed(fi.spacing)), device=device, dtype=dtype)
fixed_origin_t = torch.tensor(list(reversed(fi.origin)), device=device, dtype=dtype)
fixed_direction_t = torch.tensor(np.eye(2), device=device, dtype=dtype)

moving_shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
moving_spacing_t = torch.tensor(list(reversed(mi_affine.spacing)), device=device, dtype=dtype)
moving_origin_t = torch.tensor(list(reversed(mi_affine.origin)), device=device, dtype=dtype)
moving_direction_t = torch.tensor(np.eye(2), device=device, dtype=dtype)

M_phys = torch.eye(2, device=device, dtype=dtype)
t_phys = torch.zeros(2, device=device, dtype=dtype)

cfl_voxels = 0.25
spacing_t = torch.tensor(list(reversed(fi.spacing)), device=device, dtype=dtype)

print("\nStarting optimization loop...")
for step in range(20):
    if warp_l2r.grad is not None: warp_l2r.grad.zero_()
    if warp_r2l.grad is not None: warp_r2l.grad.zero_()
    
    I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled = prepare_mid_images_and_gradients_torch(
        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_tensor, J_tensor,
        X_phys,
        fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
        moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
        fi.spacing, mi_affine.spacing,
        M_phys, t_phys, None
    )
    
    I_mid.retain_grad()
    J_mid.retain_grad()
    
    loss = local_ncc_loss_nd(J_mid, I_mid, window_size=5)
    loss.backward()
    
    with torch.no_grad():
        # Custom analytical gradients
        grad_l_raw = (I_mid.grad.movedim(1, -1) * grad_I_mid_sampled).squeeze(0)
        grad_r_raw = (J_mid.grad.movedim(1, -1) * grad_J_mid_sampled).squeeze(0)
        
        grad_l = separable_gaussian_filter(grad_l_raw * b_mask, 3.0, spacing=fi.spacing)
        grad_r = separable_gaussian_filter(grad_r_raw * b_mask, 3.0, spacing=fi.spacing)
        
        grad_l_voxel = grad_l * spacing_t
        grad_r_voxel = grad_r * spacing_t
        max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max() + 1e-8
        max_norm_r = torch.sqrt(torch.sum(grad_r_voxel**2, dim=-1)).max() + 1e-8
        
        lr_l = cfl_voxels / max_norm_l
        lr_r = cfl_voxels / max_norm_r
        
        delta_l = lr_l * grad_l_voxel * spacing_t
        delta_r = lr_r * grad_r_voxel * spacing_t
        
        # Greedy composition
        coords_phys_l = X_phys - delta_l
        coords_norm_l = physical_to_normalized_torch(coords_phys_l, spatial, fi.spacing, fi.origin, fi.direction)
        warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
        warp_l2r.copy_(warp_l2r_sampled - delta_l)
        
        coords_phys_r = X_phys - delta_r
        coords_norm_r = physical_to_normalized_torch(coords_phys_r, spatial, fi.spacing, fi.origin, fi.direction)
        warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
        warp_r2l.copy_(warp_r2l_sampled - delta_r)
        
        warp_l2r.mul_(b_mask)
        warp_r2l.mul_(b_mask)
        
        # Double inversion to update inverses
        warp_l2r_inv = update_inverse_field_nd(warp_l2r, warp_l2r_inv, steps=5, spacing=fi.spacing)
        warp_r2l_inv = update_inverse_field_nd(warp_r2l, warp_r2l_inv, steps=5, spacing=fi.spacing)
        
    # Evaluate current warped image using ANTs image_mutual_information!
    with torch.no_grad():
        disp_l2r_np = warp_l2r[0].numpy()
        fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
        fwd_img = ants.from_numpy(disp_l2r_np, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
        ants.image_write(fwd_img, fwd_file)
        
        warped_curr = ants.apply_transforms(fi, mi_affine, [fwd_file])
        mi_val = ants.image_mutual_information(fi, warped_curr)
        print(f"Step {step+1:02d} | Loss: {loss.item():.6f} | ANTs MI: {mi_val:.6f}")
        
        os.remove(fwd_file)
