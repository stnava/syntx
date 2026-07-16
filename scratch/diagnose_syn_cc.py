import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn_jax import box_filter_jax
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

# PyTorch internal cc
def get_cc_py(I, J):
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
    return cc, valid_mask, I_var

# JAX internal cc
def get_cc_jax(I, J):
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
    return cc, valid_mask, I_var

cc_py, mask_py, I_var = get_cc_py(torch.tensor(y_np), torch.tensor(x_np))
cc_jax, mask_jax, I_var_jax = get_cc_jax(jnp.array(y_np), jnp.array(x_np))

cc_py_np = cc_py.numpy()
cc_jax_np = np.array(cc_jax)

print("cc_py max:", cc_py_np.max(), "min:", cc_py_np.min(), "mean:", cc_py_np.mean())
print("cc_jax max:", cc_jax_np.max(), "min:", cc_jax_np.min(), "mean:", cc_jax_np.mean())
print("cc max absolute difference:", np.abs(cc_py_np - cc_jax_np).max())
print("mask_py sum:", float(mask_py.sum()))
print("mask_jax sum:", float(mask_jax.sum()))
print("mask difference sum:", float(np.abs(mask_py.numpy().astype(float) - np.array(mask_jax).astype(float)).sum()))

# Find where PyTorch mask is True but JAX is False
diff_mask = mask_py.numpy() & (~np.array(mask_jax))
if diff_mask.any():
    print("Max PyTorch variance in diff region:", float(I_var.numpy()[diff_mask].max()))
    print("Min PyTorch variance in diff region:", float(I_var.numpy()[diff_mask].min()))
    print("Max JAX variance in diff region:", float(np.array(I_var_jax)[diff_mask].max()))
    print("Min JAX variance in diff region:", float(np.array(I_var_jax)[diff_mask].min()))


