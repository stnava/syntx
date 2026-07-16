import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn_jax import local_ncc_loss_nd_jax, box_filter_jax
from syntx.syn import local_ncc_loss_nd

# Create identical 3D inputs using binary spheres
shape = (64, 64, 64)
fi_np = np.zeros(shape, dtype=np.float32)
mi_np = np.zeros(shape, dtype=np.float32)
x, y, z = np.ogrid[:64, :64, :64]
mask_fi = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
fi_np[mask_fi] = 1.0
mask_mi = (x - 35)**2 + (y - 30)**2 + (z - 33)**2 <= 15**2
mi_np[mask_mi] = 1.0

fi_np_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_np_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

x_np = fi_np_norm[None, None, ...]
y_np = mi_np_norm[None, None, ...]

# PyTorch
def local_ncc_loss_stable_py(I, J):
    def box_filter(x):
        return torch.nn.functional.avg_pool3d(x, kernel_size=9, stride=1, padding=4, count_include_pad=True)
    I_mean = box_filter(I)
    J_mean = box_filter(J)
    I_var = box_filter((I - I_mean)**2)
    J_var = box_filter((J - J_mean)**2)
    IJ_cov = box_filter((I - I_mean) * (J - J_mean))
    valid_mask = (I_var > 1e-8) & (J_var > 1e-8)
    safe_I_var = torch.clamp(I_var, min=1e-8)
    safe_J_var = torch.clamp(J_var, min=1e-8)
    cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-8)
    cc = torch.where(valid_mask, cc_raw, torch.zeros_like(cc_raw))
    return -torch.sum(cc * valid_mask.float()) / (torch.sum(valid_mask.float()) + 1e-8)

x_py = torch.tensor(x_np, dtype=torch.float32).requires_grad_(True)
y_py = torch.tensor(y_np, dtype=torch.float32)
loss_py = local_ncc_loss_stable_py(y_py, x_py)
loss_py.backward()
g_py = x_py.grad.numpy()

# JAX
def local_ncc_loss_stable_jax(I, J):
    I_mean = box_filter_jax(I, 9)
    J_mean = box_filter_jax(J, 9)
    I_var = box_filter_jax((I - I_mean)**2, 9)
    J_var = box_filter_jax((J - J_mean)**2, 9)
    IJ_cov = box_filter_jax((I - I_mean) * (J - J_mean), 9)
    valid_mask = (I_var > 1e-8) & (J_var > 1e-8)
    safe_I_var = jnp.maximum(I_var, 1e-8)
    safe_J_var = jnp.maximum(J_var, 1e-8)
    cc_raw = IJ_cov / (jnp.sqrt(safe_I_var * safe_J_var) + 1e-8)
    cc = jnp.where(valid_mask, cc_raw, 0.0)
    active_mask_float = valid_mask.astype(jnp.float32)
    return -jnp.sum(cc * active_mask_float) / (jnp.sum(active_mask_float) + 1e-8)

x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)

loss_fn = lambda x: local_ncc_loss_stable_jax(y_jax, x)
val_jax, g_jax = jax.value_and_grad(loss_fn)(x_jax)
g_jax = np.array(g_jax)

print("PyTorch Loss:", float(loss_py.item()))
print("JAX Loss:", float(val_jax))
print("PyTorch grad max absolute:", float(np.abs(g_py).max()))
print("JAX grad max absolute:", float(np.abs(g_jax).max()))
print("Gradients Max absolute difference:", float(np.abs(g_py - g_jax).max()))
