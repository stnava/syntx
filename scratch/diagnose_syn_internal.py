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

fi_np_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_np_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

# PyTorch
I_py = torch.tensor(fi_np_norm, dtype=torch.float32)[None, None, ...]
J_py = torch.tensor(mi_np_norm, dtype=torch.float32)[None, None, ...]

I_mean = torch.nn.functional.avg_pool3d(I_py, kernel_size=9, stride=1, padding=4, count_include_pad=True)
J_mean = torch.nn.functional.avg_pool3d(J_py, kernel_size=9, stride=1, padding=4, count_include_pad=True)
I_var = torch.nn.functional.avg_pool3d(I_py**2, kernel_size=9, stride=1, padding=4, count_include_pad=True) - I_mean**2
J_var = torch.nn.functional.avg_pool3d(J_py**2, kernel_size=9, stride=1, padding=4, count_include_pad=True) - J_mean**2
valid_mask = (I_var > 1e-8) & (J_var > 1e-8)

print("PyTorch valid_mask sum:", float(valid_mask.sum()))

# JAX
from syntx.syn_jax import box_filter_jax
I_jax = jnp.array(fi_np_norm)[None, None, ...]
J_jax = jnp.array(mi_np_norm)[None, None, ...]

I_mean_jax = box_filter_jax(I_jax, 9)
J_mean_jax = box_filter_jax(J_jax, 9)
I_var_jax = box_filter_jax(I_jax**2, 9) - I_mean_jax**2
J_var_jax = box_filter_jax(J_jax**2, 9) - J_mean_jax**2
valid_mask_jax = (I_var_jax > 1e-8) & (J_var_jax > 1e-8)

print("JAX valid_mask sum:", float(valid_mask_jax.sum()))

# Compare
mean_diff = np.abs(I_mean.numpy() - np.array(I_mean_jax)).max()
var_diff = np.abs(I_var.numpy() - np.array(I_var_jax)).max()
print(f"Max difference in I_mean: {mean_diff}")
print(f"Max difference in I_var: {var_diff}")

