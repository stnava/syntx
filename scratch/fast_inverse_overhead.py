"""Check if the inverse projection is eating warp magnitude."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'src')
import ants
from syntx.syn import (get_physical_grid_torch, physical_to_normalized_torch_cached,
                        physical_to_normalized_torch,
                        mattes_mi_loss_nd, separable_gaussian_filter, get_boundary_mask,
                        update_inverse_field_nd, prepare_mid_images_and_gradients_torch)

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

# Level 4
scale = 4
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
M = torch.eye(2); t = torch.zeros(2)

for name, do_proj in [("No projection", False), ("With projection (5 steps)", True)]:
    w = torch.zeros(1, *ds, 2, requires_grad=True)
    wr = torch.zeros(1, *ds, 2, requires_grad=True)
    wi = torch.zeros_like(w); wri = torch.zeros_like(wr)
    losses = []
    for epoch in range(10):
        if w.grad is not None: w.grad.zero_()
        if wr.grad is not None: wr.grad.zero_()
        I_mid, J_mid, _, _ = prepare_mid_images_and_gradients_torch(
            w, wr, wi, wri, I, J, X, shape_t, sp_t, orig_t, dir_t,
            shape_t, sp_t, orig_t, dir_t, sp, sp, M, t, None)
        loss = mattes_mi_loss_nd(J_mid, I_mid)
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            grad_l = separable_gaussian_filter(w.grad * bm, 3.0, spacing=sp)
            grad_r = separable_gaussian_filter(wr.grad * bm, 3.0, spacing=sp)
            gv_l = grad_l * sp_fixed_t; gv_r = grad_r * sp_fixed_t
            mn_l = torch.sqrt(torch.sum(gv_l**2, dim=-1)).max() + 1e-8
            mn_r = torch.sqrt(torch.sum(gv_r**2, dim=-1)).max() + 1e-8
            delta_l = (0.25 / mn_l) * gv_l * sp_fixed_t
            delta_r = (0.25 / mn_r) * gv_r * sp_fixed_t
            for warp, delta in [(w, delta_l), (wr, delta_r)]:
                cp = X - delta
                cn = physical_to_normalized_torch_cached(cp, shape_t, sp_t, orig_t, dir_t)
                ws = F.grid_sample(warp.movedim(-1,1), cn, padding_mode='border', align_corners=True).movedim(1,-1)
                warp.copy_(ws - delta)
            w.mul_(bm); wr.mul_(bm)
            if do_proj:
                wi = update_inverse_field_nd(w, wi.detach(), steps=5, spacing=sp, origin=fi.origin, direction=fi.direction)
                w.copy_(update_inverse_field_nd(wi, w.detach(), steps=5, spacing=sp, origin=fi.origin, direction=fi.direction))
                wri = update_inverse_field_nd(wr, wri.detach(), steps=5, spacing=sp, origin=fi.origin, direction=fi.direction)
                wr.copy_(update_inverse_field_nd(wri, wr.detach(), steps=5, spacing=sp, origin=fi.origin, direction=fi.direction))
    print(f"{name}: losses=[{losses[0]:.4f}→{losses[-1]:.4f}] warp_max={w.abs().max():.2f}mm")

# Also check: loss without inv warps for midpoint image generation
print("\n--- Checking: warp_l2r_inv used for I_mid generation ---")
print("In prepare_mid_images_and_gradients_torch:")
print("  phi_l2r_phys = X_phys + warp_l2r  (NOT warp_l2r_inv)")
print("  So warp_l2r_inv is NOT used for generating I_mid or J_mid")
print("  It's only used for the final composition: total_fwd = warp_r2l(warp_l2r_inv(x))")
print("  During the SyN loop, warp_l2r_inv only affects midpoint warping via the INVERSE FIELD,")
print("  but actually... let me check if it's passed as a parameter...")
