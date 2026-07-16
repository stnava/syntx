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

# 2. Initialize Syntx PyTorch SyN model in ZYX/YX-transposed pipeline
device = 'cpu'
dtype = torch.float32
dim = 2

# Normalize images
fi_np = fi.numpy()
mi_np = mi_affine.numpy()
fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

I_tensor = torch.tensor(fi_norm, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
J_tensor = torch.tensor(mi_norm, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

# Transpose spatial dimensions to YX order!
I_tensor_zyx = I_tensor.permute(0, 1, 3, 2)
J_tensor_zyx = J_tensor.permute(0, 1, 3, 2)

spatial = J_tensor_zyx.shape[2:] # (ny, nx)

warp_l2r = torch.zeros(1, *spatial, dim, device=device, dtype=dtype).requires_grad_(True)
warp_r2l = torch.zeros(1, *spatial, dim, device=device, dtype=dtype).requires_grad_(True)
warp_l2r_inv = torch.zeros_like(warp_l2r)
warp_r2l_inv = torch.zeros_like(warp_r2l)

# Reverse physical parameters to YX order
spacing_zyx = tuple(reversed(fi.spacing))
origin_zyx = tuple(reversed(fi.origin))
direction_zyx = np.asarray(fi.direction)[::-1, ::-1].copy()

spacing_moving_zyx = tuple(reversed(mi_affine.spacing))
origin_moving_zyx = tuple(reversed(mi_affine.origin))
direction_moving_zyx = np.asarray(mi_affine.direction)[::-1, ::-1].copy()

# Physical Grid in YX order
def get_physical_grid_zyx(shape, spacing, origin, direction, device=None, dtype=None):
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij') # (y, x)
    idxs = torch.stack(meshgrid, dim=-1) # YX component order!
    
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    flat_idxs = idxs.view(-1, dim)
    scaled = flat_idxs * spacing_t
    rotated = scaled @ direction_t.t()
    phys = rotated + origin_t
    return phys.view(*shape, dim)

# Physical to normalized coordinates in YX order (flips components to XY for grid_sample)
def physical_to_normalized_zyx(phys_coords, target_shape, spacing, origin, direction):
    device = phys_coords.device
    dtype = phys_coords.dtype
    dim = len(target_shape)
    
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    
    direction_inv = torch.linalg.inv(direction_t)
    voxel_coords = diff @ direction_inv.t() / spacing_t
    
    shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    
    # Flip components from YX to XY for grid_sample!
    norm_coords_reversed = torch.flip(norm_coords, dims=[-1])
    return norm_coords_reversed.view(phys_coords.shape)

from syntx.syn import (
    local_ncc_loss_nd, separable_gaussian_filter, get_boundary_mask,
    update_inverse_field_nd
)

X_phys = get_physical_grid_zyx(spatial, spacing_zyx, origin_zyx, direction_zyx, device=device, dtype=dtype)
b_mask = get_boundary_mask(spatial, device, dtype)

cfl_voxels = 0.25
spacing_t = torch.tensor(spacing_zyx, device=device, dtype=dtype)

def spatial_jacobian_zyx(image, spacing):
    dim = len(spacing)
    grad_list = []
    for d in range(dim):
        grad_d = torch.gradient(image.squeeze(0).squeeze(0), spacing=spacing[d], dim=d)[0]
        grad_list.append(grad_d)
    return torch.stack(grad_list, dim=-1).unsqueeze(0)

print("\nRunning ZYX optimization loop...")
for step in range(30):
    if warp_l2r.grad is not None: warp_l2r.grad.zero_()
    if warp_r2l.grad is not None: warp_r2l.grad.zero_()
    
    # 1. Warp images to midpoint
    coords_phys_l = X_phys.unsqueeze(0) + warp_l2r
    coords_norm_l = physical_to_normalized_zyx(coords_phys_l, spatial, spacing_zyx, origin_zyx, direction_zyx)
    I_mid = F.grid_sample(I_tensor_zyx, coords_norm_l, padding_mode='border', align_corners=True)
    
    coords_phys_r = X_phys.unsqueeze(0) + warp_r2l
    coords_norm_r = physical_to_normalized_zyx(coords_phys_r, spatial, spacing_moving_zyx, origin_moving_zyx, direction_moving_zyx)
    J_mid = F.grid_sample(J_tensor_zyx, coords_norm_r, padding_mode='border', align_corners=True)
    
    I_mid.retain_grad()
    J_mid.retain_grad()
    
    loss = local_ncc_loss_nd(J_mid, I_mid, window_size=5)
    loss.backward()
    
    with torch.no_grad():
        grad_I = spatial_jacobian_zyx(I_tensor_zyx, spacing_zyx)
        grad_J = spatial_jacobian_zyx(J_tensor_zyx, spacing_moving_zyx)
        
        grad_I_mid = F.grid_sample(grad_I.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1).squeeze(0)
        grad_J_mid = F.grid_sample(grad_J.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1).squeeze(0)
        
        grad_l_raw = I_mid.grad.squeeze(0).squeeze(0).unsqueeze(-1) * grad_I_mid
        grad_r_raw = J_mid.grad.squeeze(0).squeeze(0).unsqueeze(-1) * grad_J_mid
        
        grad_l = separable_gaussian_filter(grad_l_raw * b_mask, 3.0, spacing=spacing_zyx)
        grad_r = separable_gaussian_filter(grad_r_raw * b_mask, 3.0, spacing=spacing_zyx)
        
        grad_l_voxel = grad_l * spacing_t
        grad_r_voxel = grad_r * spacing_t
        max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max() + 1e-8
        max_norm_r = torch.sqrt(torch.sum(grad_r_voxel**2, dim=-1)).max() + 1e-8
        
        lr_l = cfl_voxels / max_norm_l
        lr_r = cfl_voxels / max_norm_r
        
        delta_l = lr_l * grad_l_voxel * spacing_t
        delta_r = lr_r * grad_r_voxel * spacing_t
        
        coords_phys_update_l = X_phys.unsqueeze(0) - delta_l
        coords_norm_update_l = physical_to_normalized_zyx(coords_phys_update_l, spatial, spacing_zyx, origin_zyx, direction_zyx)
        warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_norm_update_l, padding_mode='border', align_corners=True).movedim(1, -1)
        warp_l2r.copy_(warp_l2r_sampled - delta_l)
        
        coords_phys_update_r = X_phys.unsqueeze(0) - delta_r
        coords_norm_update_r = physical_to_normalized_zyx(coords_phys_update_r, spatial, spacing_moving_zyx, origin_moving_zyx, direction_moving_zyx)
        warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_update_r, padding_mode='border', align_corners=True).movedim(1, -1)
        warp_r2l.copy_(warp_r2l_sampled - delta_r)
        
        warp_l2r.mul_(b_mask)
        warp_r2l.mul_(b_mask)
        
        warp_l2r_inv = update_inverse_field_nd(warp_l2r, warp_l2r_inv, steps=5, spacing=spacing_zyx)
        warp_r2l_inv = update_inverse_field_nd(warp_r2l, warp_r2l_inv, steps=5, spacing=spacing_zyx)

print("\nFinished optimization. Evaluating ZYX composed warp...")

with torch.no_grad():
    coords_norm_inv = physical_to_normalized_zyx(X_phys.unsqueeze(0) + warp_l2r_inv, spatial, spacing_zyx, origin_zyx, direction_zyx)
    warp_r2l_warped = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_inv, padding_mode='border', align_corners=True).movedim(1, -1)
    warp_fwd = warp_l2r_inv + warp_r2l_warped
    
    # Transpose spatial dimensions back from YX to XY order!
    disp_l2r_np = warp_fwd[0].numpy().transpose(1, 0, 2)
    # Reverse components from YX to XY order!
    disp_l2r_t = disp_l2r_np[..., ::-1].copy()
    
    fwd_file_comp = tempfile.NamedTemporaryFile(suffix='_comp.nii.gz', delete=False).name
    fwd_img_comp = ants.from_numpy(disp_l2r_t, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
    ants.image_write(fwd_img_comp, fwd_file_comp)
    
    warped_comp = ants.apply_transforms(fi, mi_affine, [fwd_file_comp])
    mi_comp = ants.image_mutual_information(fi, warped_comp)
    print(f"Composed Warp ZYX | ANTs MI: {mi_comp:.6f}")
    
    # Calculate Otsu Dice
    fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
    warped_seg = ants.threshold_image(warped_comp, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    dice = float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    print(f"Composed Warp ZYX | Otsu Dice: {dice:.4f}")
    
    os.remove(fwd_file_comp)
