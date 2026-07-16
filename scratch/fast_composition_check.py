"""Check if midpoint → total composition is losing quality."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'src')
import syntx, ants
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

r = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
               reg_iterations=[10, 5, 0], affine_iterations=[0, 0, 0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
m = r['model']

# Internal warps at GRID resolution (not downsampled)
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_aff.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
X = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)

# Method 1: Use the total composed warp (what gets exported)
phi_total = X + m.warp_l2r.data
coords_total = physical_to_normalized_torch(phi_total, fi.shape, fi.spacing, fi.origin, fi.direction)
J_total = F.grid_sample(J_t, coords_total, align_corners=True, padding_mode='border')
mi_total = ants.image_mutual_information(fi,
    ants.from_numpy(J_total[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))

# Method 2: Apply the warp via ANTs (exported file)
w_ants = ants.apply_transforms(fi, mi_aff, r['fwdtransforms'])
mi_ants_apply = ants.image_mutual_information(fi, w_ants)

# Method 3: Warp using just warp_l2r (half-warp, midpoint arm)
# This would move the moving image halfway to alignment
# (not a valid comparison but shows the midpoint field magnitude)

print(f"Total composed warp (internal): MI={mi_total:.4f}")
print(f"Total composed warp (ANTs apply): MI={mi_ants_apply:.4f}")
print(f"Consistency: {abs(mi_total - mi_ants_apply):.6f}")
print(f"Pre-aligned MI: {ants.image_mutual_information(fi, mi_aff):.4f}")
print(f"warp_l2r max: {m.warp_l2r.data.abs().max():.2f}mm")
print(f"warp_r2l max: {m.warp_r2l.data.abs().max():.2f}mm")

# Check: does the internal midpoint loss match what we expect?
losses = m.syn_losses
print(f"\nInternal midpoint losses: [{losses[0]:.4f}→{losses[-1]:.4f}]")
print(f"The midpoint loss improved by {losses[0]-losses[-1]:.4f}")

# Compare ANTs SyN warp magnitude
ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[10, 5, 0], syn_metric='mattes')
ants_warp = ants.image_read(ra['fwdtransforms'][0])
print(f"\nANTs warp max: {np.abs(ants_warp.numpy()).max():.2f}mm")
print(f"syntx warp max: {m.warp_l2r.data.abs().max():.2f}mm")
print(f"ANTs MI: {ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
