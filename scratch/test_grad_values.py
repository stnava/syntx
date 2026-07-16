import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

# Load to tensors
device = 'cpu'
dtype = torch.float32
I_curr = torch.tensor(fi.numpy(), dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
J_curr = torch.tensor(mi.numpy(), dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

# Normalize
I_curr = (I_curr - I_curr.mean()) / (I_curr.std() + 1e-8)
J_curr = (J_curr - J_curr.mean()) / (J_curr.std() + 1e-8)

spatial = I_curr.shape[2:]
dim = 2

warp_l2r = torch.zeros(1, *spatial, dim, device=device, dtype=dtype, requires_grad=True)
warp_r2l = torch.zeros(1, *spatial, dim, device=device, dtype=dtype, requires_grad=True)
warp_l2r_inv = torch.zeros_like(warp_l2r)
warp_r2l_inv = torch.zeros_like(warp_r2l)

from syntx.syn import get_physical_grid_torch, prepare_mid_images_and_gradients_torch, local_ncc_loss_nd

X_phys = get_physical_grid_torch(spatial, fi.spacing, fi.origin, fi.direction, device=device, dtype=dtype)
fixed_shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
fixed_spacing_t = torch.tensor(list(reversed(fi.spacing)), device=device, dtype=dtype)
fixed_origin_t = torch.tensor(list(reversed(fi.origin)), device=device, dtype=dtype)
fixed_direction_t = torch.tensor(np.eye(2), device=device, dtype=dtype)

moving_shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
moving_spacing_t = torch.tensor(list(reversed(mi.spacing)), device=device, dtype=dtype)
moving_origin_t = torch.tensor(list(reversed(mi.origin)), device=device, dtype=dtype)
moving_direction_t = torch.tensor(np.eye(2), device=device, dtype=dtype)

M_phys = torch.eye(2, device=device, dtype=dtype)
t_phys = torch.zeros(2, device=device, dtype=dtype)
initial_grid_level = None

# Compute Autograd
I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled = prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fi.spacing, mi.spacing,
    M_phys, t_phys, initial_grid_level
)

loss = local_ncc_loss_nd(J_mid, I_mid, window_size=5)
loss.backward()

grad_autograd = warp_l2r.grad.clone()

# Compute Analytical Components
I_mid_a, J_mid_a, grad_I_mid_sampled_a, grad_J_mid_sampled_a = prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fi.spacing, mi.spacing,
    M_phys, t_phys, initial_grid_level
)

I_mid_a.retain_grad()
loss_a = local_ncc_loss_nd(J_mid_a, I_mid_a, window_size=5)
loss_a.backward()

grad_analytical = (I_mid_a.grad.movedim(1, -1) * grad_I_mid_sampled_a).contiguous()

# Find a voxel where autograd is significant
mask = grad_autograd.abs().sum(dim=-1) > 1e-4
indices = torch.nonzero(mask)
print(f"Found {len(indices)} voxels with significant autograd gradient.")

# Print for top 5 voxels
for idx in indices[:5]:
    idx_tuple = tuple(idx.tolist())
    val_auto = grad_autograd[idx_tuple]
    val_ana = grad_analytical[idx_tuple].detach()
    val_img_grad = grad_I_mid_sampled_a[idx_tuple].detach()
    val_loss_grad = I_mid_a.grad.movedim(1, -1)[idx_tuple].detach()
    print(f"\nVoxel {idx_tuple[1:-1]}:")
    print(f"  Autograd gradient:   {val_auto.numpy()}")
    print(f"  Analytical gradient: {val_ana.numpy()}")
    print(f"  Image gradient (grad_I): {val_img_grad.numpy()}")
    print(f"  Loss gradient (dLoss/dI): {val_loss_grad.numpy()}")
