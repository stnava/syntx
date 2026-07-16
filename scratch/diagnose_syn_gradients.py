import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import jax
import jax.numpy as jnp

# Create synthetic 3D images
shape = (64, 64, 64)
fi_np = np.zeros(shape, dtype=np.float32)
mi_np = np.zeros(shape, dtype=np.float32)
x, y, z = np.ogrid[:64, :64, :64]
mask_fi = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
fi_np[mask_fi] = 1.0
mask_mi = (x - 35)**2 + (y - 30)**2 + (z - 33)**2 <= 15**2
mi_np[mask_mi] = 1.0

fi = ants.from_numpy(fi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
mi = ants.from_numpy(mi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

from syntx.syn import (
    get_physical_grid_torch,
    get_boundary_mask,
    prepare_mid_images_and_gradients_torch,
    local_ncc_loss_nd
)

# We will run one step of PyTorch and print values
print("--- PyTorch Step 1 ---")
fi_np_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_np_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

I_curr = torch.tensor(fi_np_norm, dtype=torch.float32)[None, None, ...]
J_curr = torch.tensor(mi_np_norm, dtype=torch.float32)[None, None, ...]

warp_l2r = torch.zeros(1, 64, 64, 64, 3, requires_grad=True)
warp_r2l = torch.zeros(1, 64, 64, 64, 3, requires_grad=True)
warp_l2r_inv = torch.zeros_like(warp_l2r)
warp_r2l_inv = torch.zeros_like(warp_r2l)

X_phys = get_physical_grid_torch((64, 64, 64), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
b_mask = get_boundary_mask((64, 64, 64), device='cpu', dtype=torch.float32)

fixed_shape_t = torch.tensor([64, 64, 64], dtype=torch.float32)
fixed_spacing_t = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
fixed_origin_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
fixed_direction_t = torch.tensor(np.eye(3), dtype=torch.float32)

M_phys = torch.tensor(np.eye(3), dtype=torch.float32)
t_phys = torch.tensor(np.zeros(3), dtype=torch.float32)

I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled = prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    M_phys, t_phys, None
)

loss = local_ncc_loss_nd(J_mid, I_mid, window_size=9)
I_mid.retain_grad()
J_mid.retain_grad()
loss.backward()

print("PyTorch loss:", loss.item())
print("PyTorch I_mid.grad max absolute:", I_mid.grad.abs().max().item())
print("PyTorch J_mid.grad max absolute:", J_mid.grad.abs().max().item())
print("PyTorch grad_I_mid_sampled max absolute:", grad_I_mid_sampled.abs().max().item())

# Now run JAX and print values
print("\n--- JAX Step 1 ---")
from syntx.syn_jax import prepare_mid_images_and_gradients_jax, local_ncc_loss_nd_jax

I_curr_jax = jnp.array(fi_np_norm)[None, None, ...]
J_curr_jax = jnp.array(mi_np_norm)[None, None, ...]

warp_l2r_jax = jnp.zeros((1, 64, 64, 64, 3))
warp_r2l_jax = jnp.zeros((1, 64, 64, 64, 3))
warp_l2r_inv_jax = jnp.zeros_like(warp_l2r_jax)
warp_r2l_inv_jax = jnp.zeros_like(warp_r2l_jax)

X_phys_jax = jnp.array(X_phys.numpy())

fixed_shape_t_jax = jnp.array([64, 64, 64], dtype=jnp.float32)
fixed_spacing_t_jax = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
fixed_origin_t_jax = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
fixed_direction_t_jax = jnp.array(np.eye(3), dtype=jnp.float32)

M_phys_jax = jnp.array(np.eye(3), dtype=jnp.float32)
t_phys_jax = jnp.array(np.zeros(3), dtype=jnp.float32)

I_mid_jax, J_mid_jax, grad_I_mid_sampled_jax, grad_J_mid_sampled_jax = prepare_mid_images_and_gradients_jax(
    warp_l2r_jax, warp_r2l_jax, warp_l2r_inv_jax, warp_r2l_inv_jax, I_curr_jax, J_curr_jax,
    X_phys_jax,
    fixed_shape_t_jax, fixed_spacing_t_jax, fixed_origin_t_jax, fixed_direction_t_jax,
    fixed_shape_t_jax, fixed_spacing_t_jax, fixed_origin_t_jax, fixed_direction_t_jax,
    (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    M_phys_jax, t_phys_jax, None
)

loss_fn_jax = lambda jm, im: local_ncc_loss_nd_jax(jm, im, window_size=9)
val_jax, (g_jm_jax, g_im_jax) = jax.value_and_grad(loss_fn_jax, argnums=(0, 1))(J_mid_jax, I_mid_jax)

print("JAX loss:", float(val_jax))
print("JAX g_im_jax (grad wrt I_mid) max absolute:", float(jnp.abs(g_im_jax).max()))
print("JAX g_jm_jax (grad wrt J_mid) max absolute:", float(jnp.abs(g_jm_jax).max()))
print("JAX grad_I_mid_sampled max absolute:", float(jnp.abs(grad_I_mid_sampled_jax).max()))
