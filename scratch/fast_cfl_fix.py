"""Test CFL variants."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'src')
import ants
from syntx.syn import (get_physical_grid_torch, physical_to_normalized_torch_cached,
                        mattes_mi_loss_nd, separable_gaussian_filter, get_boundary_mask,
                        update_inverse_field_nd, prepare_mid_images_and_gradients_torch)

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

scale = 4
down_shape = tuple(max(1, d // scale) for d in fi.shape)
curr_spacing = tuple(sp * (N-1)/(n-1) for sp, N, n in zip(fi.spacing, fi.shape, down_shape))
I_full = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_full = torch.tensor(mi_aff.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
I_curr = F.interpolate(I_full, size=down_shape, mode='bilinear', align_corners=True)
J_curr = F.interpolate(J_full, size=down_shape, mode='bilinear', align_corners=True)

spacing_rev = tuple(reversed(curr_spacing))
origin_rev = tuple(reversed(fi.origin))
dir_rev = np.asarray(fi.direction)[::-1, ::-1].copy()
shape_t = torch.tensor(list(down_shape), dtype=torch.float32)
sp_t = torch.tensor(spacing_rev, dtype=torch.float32)
orig_t = torch.tensor(origin_rev, dtype=torch.float32)
dir_t = torch.tensor(dir_rev, dtype=torch.float32)
X_phys = get_physical_grid_torch(down_shape, curr_spacing, fi.origin, fi.direction)
b_mask = get_boundary_mask(down_shape, 'cpu', torch.float32)
sp_fixed_t = torch.tensor(list(reversed(curr_spacing)), dtype=torch.float32)

def run_loop(cfl_fn_name, n_iters=20):
    warp_l2r = torch.zeros(1, *down_shape, 2, requires_grad=True)
    warp_r2l = torch.zeros(1, *down_shape, 2, requires_grad=True)
    warp_l2r_inv = torch.zeros_like(warp_l2r)
    warp_r2l_inv = torch.zeros_like(warp_r2l)
    M_phys = torch.eye(2); t_phys = torch.zeros(2)
    losses = []
    for epoch in range(n_iters):
        if warp_l2r.grad is not None: warp_l2r.grad.zero_()
        if warp_r2l.grad is not None: warp_r2l.grad.zero_()
        I_mid, J_mid, _, _ = prepare_mid_images_and_gradients_torch(
            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr, X_phys,
            shape_t, sp_t, orig_t, dir_t, shape_t, sp_t, orig_t, dir_t,
            curr_spacing, curr_spacing, M_phys, t_phys, None)
        loss = mattes_mi_loss_nd(J_mid, I_mid)
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, 3.0, spacing=curr_spacing)
            grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, 3.0, spacing=curr_spacing)
            if cfl_fn_name == 'current':
                gv_l = grad_l * sp_fixed_t; gv_r = grad_r * sp_fixed_t
                mn_l = torch.sqrt(torch.sum(gv_l**2, dim=-1)).max() + 1e-8
                mn_r = torch.sqrt(torch.sum(gv_r**2, dim=-1)).max() + 1e-8
                delta_l = (0.25 / mn_l) * gv_l * sp_fixed_t
                delta_r = (0.25 / mn_r) * gv_r * sp_fixed_t
            else:  # ITK-style
                mn_l = torch.sqrt(torch.sum(grad_l**2, dim=-1)).max() + 1e-8
                mn_r = torch.sqrt(torch.sum(grad_r**2, dim=-1)).max() + 1e-8
                max_sp = max(curr_spacing)
                delta_l = (0.25 * max_sp / mn_l) * grad_l
                delta_r = (0.25 * max_sp / mn_r) * grad_r
            for warp, delta, warp_inv in [(warp_l2r, delta_l, warp_l2r_inv), (warp_r2l, delta_r, warp_r2l_inv)]:
                cp = X_phys - delta
                cn = physical_to_normalized_torch_cached(cp, shape_t, sp_t, orig_t, dir_t)
                ws = F.grid_sample(warp.movedim(-1,1), cn, padding_mode='border', align_corners=True).movedim(1,-1)
                warp.copy_(ws - delta)
            warp_l2r.mul_(b_mask); warp_r2l.mul_(b_mask)
    return losses, warp_l2r.abs().max().item(), (delta_l.abs().max().item(), delta_r.abs().max().item())

for name in ['current', 'itk']:
    losses, wmax, dmax = run_loop(name)
    print(f"{name:>8}: losses=[{losses[0]:.4f}→{losses[-1]:.4f}] warp_max={wmax:.2f}mm delta_max={dmax}")
