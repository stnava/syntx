"""Diagnose: internal warp is correct, but ANTs export is wrong."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'src')
import syntx
import ants
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch
import tempfile

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_aff = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx_aff['fwdtransforms'])

reg_pt = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                    reg_iterations=[50, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.2)

model = reg_pt['model']
warp_l2r = model.warp_l2r.data[0].numpy()  # (H, W, 2) in YX order

# The internal warp is in YX order: [..0]=Y displacement, [..1]=X displacement
# ANTs expects XY order: [..0]=X displacement, [..1]=Y displacement
# Current export: disp[..., ::-1] reverses YX to XY — this should be correct!

print("Internal warp shape:", warp_l2r.shape)
print("Internal warp[..., 0] range (Y disp):", warp_l2r[..., 0].min(), warp_l2r[..., 0].max())
print("Internal warp[..., 1] range (X disp):", warp_l2r[..., 1].min(), warp_l2r[..., 1].max())

# Read what was exported
fwd_file = reg_pt['fwdtransforms'][0]
fwd_ants = ants.image_read(fwd_file)
print(f"\nExported warp shape: {fwd_ants.numpy().shape}")
print(f"Exported warp[..., 0] range (X disp):", fwd_ants.numpy()[..., 0].min(), fwd_ants.numpy()[..., 0].max())
print(f"Exported warp[..., 1] range (Y disp):", fwd_ants.numpy()[..., 1].min(), fwd_ants.numpy()[..., 1].max())

# Test 1: Apply internal warp directly (bypass ANTs)
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_aff.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)

phi_fwd = X_phys + model.warp_l2r.data
coords = physical_to_normalized_torch(phi_fwd, fi.shape, fi.spacing, fi.origin, fi.direction)
J_warped_internal = F.grid_sample(J_t, coords, align_corners=True, padding_mode='border')
mi_internal = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_internal[0,0].numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))
print(f"\nMI (internal apply, no affine): {mi_internal:.4f}")

# Test 2: Try not reversing channels for ANTs export
fwd_file_noreverse = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
fwd_img_nr = ants.from_numpy(warp_l2r.copy(), has_components=True, 
                               spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
ants.image_write(fwd_img_nr, fwd_file_noreverse)
warped_nr = ants.apply_transforms(fi, mi_aff, [fwd_file_noreverse, reg_pt['fwdtransforms'][1]])
mi_nr = ants.image_mutual_information(fi, warped_nr)
print(f"MI (no channel reverse + affine): {mi_nr:.4f}")

# Test 3: Reversed channels (current behavior)
fwd_file_rev = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
warp_rev = warp_l2r[..., ::-1].copy()
fwd_img_rev = ants.from_numpy(warp_rev, has_components=True,
                                spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
ants.image_write(fwd_img_rev, fwd_file_rev)
warped_rev = ants.apply_transforms(fi, mi_aff, [fwd_file_rev, reg_pt['fwdtransforms'][1]])
mi_rev = ants.image_mutual_information(fi, warped_rev)
print(f"MI (reversed channels + affine): {mi_rev:.4f}")

# Test 4: No reverse, no affine
warped_nr_na = ants.apply_transforms(fi, mi_aff, [fwd_file_noreverse])
mi_nr_na = ants.image_mutual_information(fi, warped_nr_na)
print(f"MI (no reverse, no affine): {mi_nr_na:.4f}")

# Test 5: Reversed, no affine
warped_rev_na = ants.apply_transforms(fi, mi_aff, [fwd_file_rev])
mi_rev_na = ants.image_mutual_information(fi, warped_rev_na)
print(f"MI (reversed, no affine): {mi_rev_na:.4f}")

# Test 6: Negated
for name, arr in [("neg+rev", -warp_rev), ("neg+norev", -warp_l2r.copy())]:
    f = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
    img = ants.from_numpy(arr, has_components=True, spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
    ants.image_write(img, f)
    w = ants.apply_transforms(fi, mi_aff, [f])
    mi_val = ants.image_mutual_information(fi, w)
    print(f"MI ({name}, no affine): {mi_val:.4f}")

print(f"\nInitial MI: {ants.image_mutual_information(fi, mi_aff):.4f}")
