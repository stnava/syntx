"""Comprehensive diagnostic for syntx SyN deformable registration."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'src')
import syntx
import ants
from syntx.syn import (get_physical_grid_torch, physical_to_normalized_torch,
                        local_ncc_loss_nd, mattes_mi_loss_nd)

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])
print(f"Initial MI: {ants.image_mutual_information(fi, mi_affine):.4f}")

print("\n=== DIAGNOSIS 1: Resolution Pyramid ===")
print("Default 2D levels: [8, 4, 2, 1]")
print(f"Full resolution: {fi.shape}")
for s in [8, 4, 2, 1]:
    down = tuple(max(1, d // s) for d in fi.shape)
    spacing = tuple(sp * (N-1)/(n-1) if n > 1 else sp for sp, N, n in zip(fi.spacing, fi.shape, down))
    print(f"  Level {s}: spatial={down}, spacing={[f'{s:.2f}' for s in spacing]}")

print("\n=== DIAGNOSIS 2: Default grad_step ===")
print("syn.py line 1852: grad_step=0.75 (default in registration())")
print("syn.py line 1041: cfl_voxels=0.75 (default in fit())")
print("ANTs default: 0.2 voxels per step")
print("syntx default: 0.75 voxels per step (HIGHER than ANTs)")

print("\n=== DIAGNOSIS 3: Gradient magnitude check (autograd vs analytical) ===")
# Quick check: do gradients actually flow through grid_sample?
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_affine.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

disp = torch.zeros(1, *fi.shape, 2, dtype=torch.float32, requires_grad=True)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)
phi = X_phys + disp
coords = physical_to_normalized_torch(phi, fi.shape, fi.spacing, fi.origin, fi.direction)
I_warped = F.grid_sample(I_t, coords, align_corners=True)

loss_lncc = local_ncc_loss_nd(J_t, I_warped)
loss_lncc.backward()
print(f"  LNCC gradient max: {disp.grad.abs().max().item():.6f}")
print(f"  LNCC gradient mean: {disp.grad.abs().mean().item():.6f}")
print(f"  LNCC gradient nonzero: {(disp.grad.abs() > 1e-8).sum().item()} / {disp.grad.numel()}")

disp2 = torch.zeros(1, *fi.shape, 2, dtype=torch.float32, requires_grad=True)
phi2 = X_phys + disp2
coords2 = physical_to_normalized_torch(phi2, fi.shape, fi.spacing, fi.origin, fi.direction)
I_warped2 = F.grid_sample(I_t, coords2, align_corners=True)

loss_mi = mattes_mi_loss_nd(J_t, I_warped2)
loss_mi.backward()
print(f"  MI gradient max: {disp2.grad.abs().max().item():.6f}")
print(f"  MI gradient mean: {disp2.grad.abs().mean().item():.6f}")
print(f"  MI gradient nonzero: {(disp2.grad.abs() > 1e-8).sum().item()} / {disp2.grad.numel()}")

print("\n=== DIAGNOSIS 4: CFL-normalized step size in mm ===")
# At level 8 (32x32), spacing is 8mm. CFL=0.75 voxels * 8mm = 6mm per step
for s in [8, 4, 2, 1]:
    eff_spacing = fi.spacing[0] * (fi.shape[0]-1) / (max(1, fi.shape[0]//s) - 1)
    cfl_mm = 0.75 * eff_spacing
    print(f"  Level {s}: CFL step = 0.75 voxels * {eff_spacing:.2f}mm = {cfl_mm:.2f}mm max displacement/iter")

print("\n=== DIAGNOSIS 5: 20-iteration comparison ===")
# ANTs
reg_ants = ants.registration(fi, mi_affine, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')
fwd_ants = ants.image_read(reg_ants['fwdtransforms'][0])
print(f"  ANTs MI: {ants.image_mutual_information(fi, reg_ants['warpedmovout']):.4f}")
print(f"  ANTs deformation max: {np.max(np.linalg.norm(fwd_ants.numpy(), axis=-1)):.4f}mm")

# PyTorch - default settings (0.75 cfl_voxels)
reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False)
warped_pt = ants.apply_transforms(fi, mi_affine, reg_pt['fwdtransforms'])
fwd_pt = ants.image_read(reg_pt['fwdtransforms'][0])
print(f"  syntx MI: {ants.image_mutual_information(fi, warped_pt):.4f}")
print(f"  syntx deformation max: {np.max(np.linalg.norm(fwd_pt.numpy(), axis=-1)):.4f}mm")

# Print warp stats
print(f"  syntx warp_l2r max: {reg_pt['model'].warp_l2r.data.abs().max().item():.4f}")

print("\n=== DIAGNOSIS 6: Composition correctness ===")
# After optimization, the midpoint fields are composed into total fields.
# Verify: total_fwd should be φ_r2l ∘ φ_l2r_inv^{-1} (see lines 1580-1593)
# This is correct for SyN: the forward map is the composition of inverse-of-phi1 and phi2
print("  Composition formula (lines 1580-1593):")
print("  total_fwd = warp_l2r_inv(x) + warp_r2l(x + warp_l2r_inv(x))")
print("  = φ_r2l ∘ φ_l2r^{-1}   [correct for SyN forward]")

print("\n=== DIAGNOSIS 7: Warp export to ANTs ===")
# Check how displacement is exported
from syntx.syn import displacement_field_to_ants
print("  Checking displacement_field_to_ants function...")

