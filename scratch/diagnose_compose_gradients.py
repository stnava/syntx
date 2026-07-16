import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn import compute_initial_grid, get_physical_grid_torch
from syntx.syn_jax import (
    prepare_mid_images_and_gradients_jax,
    local_ncc_loss_nd_jax,
    compose_grids_jax,
    physical_to_normalized_jax
)
import ants

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

fi_np_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_np_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

I_curr = jnp.array(fi_np_norm)[None, None, ...]
J_curr = jnp.array(mi_np_norm)[None, None, ...]

X_phys_torch = get_physical_grid_torch(shape, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
X_phys = jnp.array(X_phys_torch.numpy())

# Compute initial grid using ants (identity)
init_grid_np = compute_initial_grid(fi, mi, [])
# Transpose as done in registration
init_grid_np = init_grid_np.transpose(0, 3, 2, 1, 4)
init_grid = jnp.array(init_grid_np)

fixed_shape_t = jnp.array([64, 64, 64], dtype=jnp.float32)
fixed_spacing_t = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
fixed_origin_t = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
fixed_direction_t = jnp.array(np.eye(3), dtype=jnp.float32)

def evaluate_grad(use_init_grid):
    def loss_fn(w_l2r_inv):
        w_l2r = jnp.zeros((1, 64, 64, 64, 3))
        w_r2l = jnp.zeros((1, 64, 64, 64, 3))
        w_r2l_inv = jnp.zeros((1, 64, 64, 64, 3))
        
        # Pull fixed to mid using w_l2r_inv
        phi_l2r_phys = X_phys + w_l2r_inv
        y_norm = physical_to_normalized_jax(phi_l2r_phys, (64, 64, 64), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
        if use_init_grid:
            y_norm = compose_grids_jax(init_grid, y_norm)
        # Sample it
        from syntx.syn_jax import jax_grid_sample
        I_mid = jax_grid_sample(I_curr, y_norm, padding_mode='border')
        
        # Pull moving to mid (simplified)
        J_mid = J_curr
        
        return local_ncc_loss_nd_jax(J_mid, I_mid, window_size=9)
        
    w_l2r_inv_init = jnp.zeros((1, 64, 64, 64, 3))
    val, grad = jax.value_and_grad(loss_fn)(w_l2r_inv_init)
    return val, grad

val1, grad1 = evaluate_grad(use_init_grid=False)
val2, grad2 = evaluate_grad(use_init_grid=True)

print("Without init_grid | Loss:", float(val1), "| Grad max:", float(jnp.abs(grad1).max()))
print("With init_grid    | Loss:", float(val2), "| Grad max:", float(jnp.abs(grad2).max()))
