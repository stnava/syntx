"""Check gradient magnitude and step size per iteration vs expected."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'src')
import ants
from syntx.syn import (get_physical_grid_torch, physical_to_normalized_torch_cached,
                        mattes_mi_loss_nd, separable_gaussian_filter, get_boundary_mask,
                        prepare_mid_images_and_gradients_torch)

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

# Level 8 (32x32) — coarsest level
scale = 8
ds = tuple(max(1, d // scale) for d in fi.shape)
sp = tuple(s * (N-1)/(n-1) for s, N, n in zip(fi.spacing, fi.shape, ds))
I = F.interpolate(torch.tensor(fi.numpy()).float().unsqueeze(0).unsqueeze(0), size=ds, mode='bilinear', align_corners=True)
J = F.interpolate(torch.tensor(mi_aff.numpy()).float().unsqueeze(0).unsqueeze(0), size=ds, mode='bilinear', align_corners=True)

sp_rev = tuple(reversed(sp)); orig_rev = tuple(reversed(fi.origin)); dir_rev = np.asarray(fi.direction)[::-1,::-1].copy()
shape_t = torch.tensor(list(ds), dtype=torch.float32)
sp_t = torch.tensor(sp_rev, dtype=torch.float32)
orig_t = torch.tensor(orig_rev, dtype=torch.float32)
dir_t = torch.tensor(dir_rev, dtype=torch.float32)
sp_fixed_t = torch.tensor(list(reversed(sp)), dtype=torch.float32)
X = get_physical_grid_torch(ds, sp, fi.origin, fi.direction)
bm = get_boundary_mask(ds, 'cpu', torch.float32)

w = torch.zeros(1, *ds, 2, requires_grad=True)
wr = torch.zeros(1, *ds, 2, requires_grad=True)
wi = torch.zeros_like(w); wri = torch.zeros_like(wr)
M = torch.eye(2); t = torch.zeros(2)

# Single iteration analysis
I_mid, J_mid, _, _ = prepare_mid_images_and_gradients_torch(
    w, wr, wi, wri, I, J, X, shape_t, sp_t, orig_t, dir_t,
    shape_t, sp_t, orig_t, dir_t, sp, sp, M, t, None)
loss = mattes_mi_loss_nd(J_mid, I_mid)
loss.backward()

grad_l = w.grad
print(f"Level 8 ({ds[0]}x{ds[1]}) spacing={sp[0]:.2f}mm")
print(f"Raw grad max: {grad_l.abs().max():.6f}")
print(f"Raw grad mean: {grad_l.abs().mean():.6f}")
print(f"Raw grad norm max: {torch.sqrt(torch.sum(grad_l**2, dim=-1)).max():.6f}")

# After smoothing
grad_sm = separable_gaussian_filter(grad_l * bm, 3.0, spacing=sp)
print(f"\nSmoothed grad max: {grad_sm.abs().max():.6f}")
print(f"Smoothed grad norm max: {torch.sqrt(torch.sum(grad_sm**2, dim=-1)).max():.6f}")

# CFL step
grad_vox = grad_sm * sp_fixed_t
max_norm = torch.sqrt(torch.sum(grad_vox**2, dim=-1)).max() + 1e-8
lr = 0.25 / max_norm
delta = lr * grad_vox * sp_fixed_t
print(f"\ndelta max: {delta.abs().max():.4f}mm")
print(f"delta max in voxels: {(delta / sp_fixed_t).abs().max():.4f} voxels")
print(f"Spacing: {sp[0]:.2f}mm/voxel")
print(f"Max displacement per iter: {delta.abs().max():.4f}mm = {delta.abs().max() / sp[0]:.4f} voxels")

# What should it be? In ITK SyN, gradStep=0.25 means 0.25 voxels max displacement
# The actual max displacement should be 0.25 * spacing = 0.25 * 8.23 = 2.06mm
print(f"\nExpected max displacement (0.25 * spacing): {0.25 * sp[0]:.2f}mm")
print(f"Actual/Expected ratio: {delta.abs().max() / (0.25 * sp[0]):.4f}")
