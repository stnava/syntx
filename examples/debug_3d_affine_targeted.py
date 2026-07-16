#!/usr/bin/env python
"""
Targeted diagnostics for 3D affine: 
1. lr=5e-2 divergence investigation
2. Convergence behavior analysis  
3. Parameter unlocking schedule analysis
"""
import sys
import os
import numpy as np
import ants
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from syntx.syn import SyNTo, grid_to_physical_affine, mattes_mi_loss_nd
import tempfile

# ---- Load images ----
fi = ants.resample_image(ants.image_read(ants.get_ants_data('mni')), (2, 2, 2), use_voxels=False, interp_type=0)
mi = ants.resample_image(ants.image_read(ants.get_ants_data('ch2')), (2, 2, 2), use_voxels=False, interp_type=0)

dim = fi.dimension
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

fi_np = fi.numpy()
mi_np = mi.numpy()
fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def compute_multilabel_dice(img1, img2, num_classes=3):
    t1 = ants.threshold_image(img1, 'Otsu', num_classes)
    t2 = ants.threshold_image(img2, 'Otsu', num_classes)
    arr1 = t1.numpy().astype(int)
    arr2 = t2.numpy().astype(int)
    dices = []
    for label in range(1, num_classes + 1):
        m1 = (arr1 == label).astype(np.float32)
        m2 = (arr2 == label).astype(np.float32)
        intersection = np.sum(m1 * m2)
        total = np.sum(m1) + np.sum(m2)
        if total > 0:
            dices.append(2.0 * intersection / total)
    return np.mean(dices) if dices else 0.0

def apply_syntx_affine_and_get_dice(model, fi, mi):
    with torch.no_grad():
        T_grid = model.affine.get_matrix().cpu().numpy()
    M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)
    tx = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
    tx.set_fixed_parameters(np.zeros(dim))
    tx_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    ants.write_transform(tx, tx_file)
    warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[tx_file])
    dice = compute_multilabel_dice(fi, warped)
    os.unlink(tx_file)
    return dice

# =====================================================
# Test 1: Manual affine optimization with loss tracking
# =====================================================
print("=" * 70)
print("TEST 1: DETAILED LOSS TRACKING AT DIFFERENT LRs")
print("=" * 70)

for lr in [1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1]:
    sp_ordered = tuple(reversed(fi.spacing))
    model = SyNTo(
        dim=dim, grid_shape=fi.shape, spacing=sp_ordered,
        direction=fi.direction, transform_type='Affine'
    ).to(device)
    
    # Manual CoM init
    with torch.no_grad():
        fixed_pos = torch.clamp(I_tensor, min=0.0)
        moving_pos = torch.clamp(J_tensor, min=0.0)
        sum_f = fixed_pos.sum()
        sum_m = moving_pos.sum()
        theta_id = torch.eye(dim, dim + 1, device=device).unsqueeze(0)
        grid_id = F.affine_grid(theta_id, size=I_tensor.shape, align_corners=True)
        grid_id_m = F.affine_grid(theta_id, size=J_tensor.shape, align_corners=True)
        com_f = torch.stack([torch.sum(fixed_pos[0, 0] * grid_id[0, ..., k]) / sum_f for k in range(dim)])
        com_m = torch.stack([torch.sum(moving_pos[0, 0] * grid_id_m[0, ..., k]) / sum_m for k in range(dim)])
        model.affine.translation.data.copy_(com_m - com_f)
    
    # All params active (no hierarchical unlocking)
    all_params = list(model.affine.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
    losses = []
    for epoch in range(300):
        optimizer.zero_grad()
        grid = model.get_affine_grid(fi.shape, device)
        moving_warped = F.grid_sample(J_tensor, grid, padding_mode='border', align_corners=True)
        loss = mattes_mi_loss_nd(moving_warped, I_tensor, num_bins=32)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    dice = apply_syntx_affine_and_get_dice(model, fi, mi)
    
    print(f"lr={lr:.0e}: final_loss={losses[-1]:.6f}, DICE={dice:.6f}, "
          f"loss[0]={losses[0]:.6f}, min_loss={min(losses):.6f}@{np.argmin(losses)}, "
          f"loss[50]={losses[50] if len(losses)>50 else 'N/A':.6f}")

# =====================================================
# Test 2: Hierarchical vs Non-hierarchical unlocking
# =====================================================
print("\n" + "=" * 70)
print("TEST 2: HIERARCHICAL vs FLAT PARAMETER UNLOCKING")
print("=" * 70)

lr = 1e-2

# Test A: Flat - all params active from start
model_flat = SyNTo(
    dim=dim, grid_shape=fi.shape, spacing=tuple(reversed(fi.spacing)),
    direction=fi.direction, transform_type='Affine'
).to(device)

with torch.no_grad():
    model_flat.affine.translation.data.copy_(com_m - com_f)

all_params = list(model_flat.affine.parameters())
optimizer = torch.optim.Adam(all_params, lr=lr)

flat_losses = []
for epoch in range(500):
    optimizer.zero_grad()
    grid = model_flat.get_affine_grid(fi.shape, device)
    moving_warped = F.grid_sample(J_tensor, grid, padding_mode='border', align_corners=True)
    loss = mattes_mi_loss_nd(moving_warped, I_tensor, num_bins=32)
    loss.backward()
    optimizer.step()
    flat_losses.append(loss.item())

flat_dice = apply_syntx_affine_and_get_dice(model_flat, fi, mi)
print(f"Flat (all params, 500 iters): DICE={flat_dice:.6f}, final_loss={flat_losses[-1]:.6f}")

# Test B: Default hierarchical via model.fit()
model_hier = SyNTo(
    dim=dim, grid_shape=fi.shape, spacing=tuple(reversed(fi.spacing)),
    direction=fi.direction, transform_type='Affine'
).to(device)

model_hier.fit(
    I_tensor, J_tensor,
    levels=[4, 2, 1],
    epochs_per_level=[0, 0, 0],
    affine_epochs=[200, 200, 200],
    affine_lr=lr,
    similarity_metric='mattes_mi',
    mattes_bins=32
)

hier_dice = apply_syntx_affine_and_get_dice(model_hier, fi, mi)
hier_losses = [l.item() if hasattr(l, 'item') else float(l) for l in model_hier.affine_losses]
print(f"Hierarchical [4,2,1]: DICE={hier_dice:.6f}, final_loss={hier_losses[-1]:.6f}, n_iters={len(hier_losses)}")

# Test C: Multi-resolution with all params active
model_mr = SyNTo(
    dim=dim, grid_shape=fi.shape, spacing=tuple(reversed(fi.spacing)),
    direction=fi.direction, transform_type='Affine'
).to(device)

with torch.no_grad():
    model_mr.affine.translation.data.copy_(com_m - com_f)

levels = [4, 2, 1]
iters_per = [200, 200, 200]
all_params = list(model_mr.affine.parameters())
mr_losses = []

for level_idx, scale in enumerate(levels):
    I_curr = F.interpolate(I_tensor, scale_factor=1.0/scale, mode='trilinear', align_corners=False) if scale > 1 else I_tensor
    J_curr = F.interpolate(J_tensor, scale_factor=1.0/scale, mode='trilinear', align_corners=False) if scale > 1 else J_tensor
    
    if level_idx == 0:
        optimizer = torch.optim.Adam(all_params, lr=lr)
    # Don't recreate optimizer — reuse with momentum
    
    curr_spatial = I_curr.shape[2:]
    for epoch in range(iters_per[level_idx]):
        optimizer.zero_grad()
        grid = model_mr.get_affine_grid(curr_spatial, device)
        moving_warped = F.grid_sample(J_curr, grid, padding_mode='border', align_corners=True)
        loss = mattes_mi_loss_nd(moving_warped, I_curr, num_bins=32)
        loss.backward()
        optimizer.step()
        mr_losses.append(loss.item())

mr_dice = apply_syntx_affine_and_get_dice(model_mr, fi, mi)
print(f"Multi-res, all params: DICE={mr_dice:.6f}, final_loss={mr_losses[-1]:.6f}, n_iters={len(mr_losses)}")

# Test D: Multi-resolution with LR decay per level
model_lrd = SyNTo(
    dim=dim, grid_shape=fi.shape, spacing=tuple(reversed(fi.spacing)),
    direction=fi.direction, transform_type='Affine'
).to(device)

with torch.no_grad():
    model_lrd.affine.translation.data.copy_(com_m - com_f)

lrd_losses = []
for level_idx, scale in enumerate(levels):
    I_curr = F.interpolate(I_tensor, scale_factor=1.0/scale, mode='trilinear', align_corners=False) if scale > 1 else I_tensor
    J_curr = F.interpolate(J_tensor, scale_factor=1.0/scale, mode='trilinear', align_corners=False) if scale > 1 else J_tensor
    
    level_lr = lr / (2 ** level_idx)
    all_params = list(model_lrd.affine.parameters())
    optimizer = torch.optim.Adam(all_params, lr=level_lr)
    
    curr_spatial = I_curr.shape[2:]
    for epoch in range(iters_per[level_idx]):
        optimizer.zero_grad()
        grid = model_lrd.get_affine_grid(curr_spatial, device)
        moving_warped = F.grid_sample(J_curr, grid, padding_mode='border', align_corners=True)
        loss = mattes_mi_loss_nd(moving_warped, I_curr, num_bins=32)
        loss.backward()
        optimizer.step()
        lrd_losses.append(loss.item())

lrd_dice = apply_syntx_affine_and_get_dice(model_lrd, fi, mi)
print(f"Multi-res, LR decay: DICE={lrd_dice:.6f}, final_loss={lrd_losses[-1]:.6f}, n_iters={len(lrd_losses)}")

# =====================================================
# Test 3: Check if the grid_to_physical_affine roundtrip is correct
# =====================================================
print("\n" + "=" * 70)
print("TEST 3: GRID-TO-PHYSICAL ROUNDTRIP VERIFICATION")
print("=" * 70)

# Create a known ANTs affine transform and verify roundtrip
# 1. Run ANTs and get its transform
ants_result = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine')
ants_tx = ants.read_transform(ants_result['fwdtransforms'][0])
ants_params = ants_tx.parameters
print(f"ANTs transform params: {ants_params}")

M_ants = ants_params[:dim*dim].reshape(dim, dim)
t_ants = ants_params[dim*dim:]
print(f"ANTs M:\n{M_ants}")
print(f"ANTs t: {t_ants}")
print(f"ANTs DICE: {compute_multilabel_dice(fi, ants_result['warpedmovout']):.6f}")

# 2. Check the syntx best model's physical transform
T_grid = model_flat.affine.get_matrix().detach().cpu().numpy()
M_syntx, t_syntx = grid_to_physical_affine(T_grid, fi, mi)
print(f"\nsyntx best M:\n{M_syntx}")
print(f"syntx best t: {t_syntx}")
print(f"syntx best DICE: {flat_dice:.6f}")

# 3. Compare M matrices
print(f"\nM difference (ANTs - syntx):\n{M_ants - M_syntx}")
print(f"t difference: {t_ants - t_syntx}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
