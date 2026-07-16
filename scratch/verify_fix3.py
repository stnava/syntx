"""Check if loss is decreasing and if the internal warp is correct."""
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
tx_aff = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx_aff['fwdtransforms'])

# Run 20 iterations at level 8 (32x32)
reg_pt = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                    reg_iterations=[50, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.2)

model = reg_pt['model']
losses = model.syn_losses
print(f"Loss trajectory (first 10): {[f'{l:.4f}' for l in losses[:10]]}")
print(f"Loss trajectory (last 10):  {[f'{l:.4f}' for l in losses[-10:]]}")
print(f"Total iterations: {len(losses)}")

# Check warp field magnitudes
print(f"\nwarp_l2r max: {model.warp_l2r.data.abs().max():.3f}mm")
print(f"warp_r2l max: {model.warp_r2l.data.abs().max():.3f}mm")

# External MI evaluation
warped = ants.apply_transforms(fi, mi_aff, reg_pt['fwdtransforms'])
mi_before = ants.image_mutual_information(fi, mi_aff)
mi_after = ants.image_mutual_information(fi, warped)
print(f"\nMI before: {mi_before:.4f}")
print(f"MI after:  {mi_after:.4f}")
print(f"Change:    {mi_after - mi_before:.4f}  {'BETTER' if mi_after < mi_before else 'WORSE'}")

# Check internal warping produces same result as ANTs apply
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_aff.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)

# Apply total fwd warp
phi_fwd = X_phys + model.warp_l2r.data
coords_fwd = physical_to_normalized_torch(phi_fwd, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped_internal = F.grid_sample(J_t, coords_fwd, align_corners=True, padding_mode='border')

# Compare
mi_internal = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_internal[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))
print(f"\nMI (internal PyTorch apply): {mi_internal:.4f}")
print(f"MI (ANTs apply_transforms): {mi_after:.4f}")

