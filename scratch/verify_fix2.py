"""Quick verification of the coordinate fixes and SyN registration."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'src')
import syntx
import ants
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_aff = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx_aff['fwdtransforms'])
mi_before = ants.image_mutual_information(fi, mi_aff)

# Identity check
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)
coords = physical_to_normalized_torch(X_phys, fi.shape, fi.spacing, fi.origin, fi.direction)
I_id = F.grid_sample(I_t, coords, align_corners=True)
print(f"Identity warp max diff: {(I_t - I_id).abs().max().item():.6f} (expected ~0 for bilinear)")

# SyN with different settings
for gs, iters in [(0.2, 20), (0.2, 50), (0.2, 100)]:
    reg_pt = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                        reg_iterations=[iters, 0, 0], affine_iterations=[0, 0, 0],
                        similarity_metric='mattes_mi', verbose=False, grad_step=gs)
    warped = ants.apply_transforms(fi, mi_aff, reg_pt['fwdtransforms'])
    mi_pt = ants.image_mutual_information(fi, warped)
    n_iters = len(reg_pt['model'].syn_losses)
    print(f"syntx SyN (gs={gs}, req={iters}, ran={n_iters}): MI={mi_pt:.4f}")

# ANTs baseline
reg_ants = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
print(f"\nANTs SyNOnly (20 iters): MI={mi_ants:.4f}")
print(f"Initial MI: {mi_before:.4f}")
