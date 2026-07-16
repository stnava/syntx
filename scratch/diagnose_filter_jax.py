import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn import separable_gaussian_filter
from syntx.syn_jax import separable_gaussian_filter_jax

# Create a test grid of shape (1, 64, 64, 64, 3)
grid_np = np.zeros((1, 64, 64, 64, 3), dtype=np.float32)
grid_np[0, 32, 32, 32, :] = 1.0

# Filter in PyTorch
grid_py = torch.tensor(grid_np, dtype=torch.float32)
res_py = separable_gaussian_filter(grid_py, 3.0, spacing=(1.0, 1.0, 1.0))

# Filter in JAX
grid_jax = jnp.array(grid_np)
res_jax = separable_gaussian_filter_jax(grid_jax, 3.0, spacing=(1.0, 1.0, 1.0))

print("PyTorch filter max:", float(res_py.max()))
print("JAX filter max:", float(res_jax.max()))
print("PyTorch filter sum:", float(res_py.sum()))
print("JAX filter sum:", float(res_jax.sum()))
