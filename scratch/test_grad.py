import torch
import jax
import jax.numpy as jnp
import numpy as np

from syntx.syn import _spatial_jacobian_nd
from syntx.syn_jax import _spatial_jacobian_nd_jax

# Define a 3D field: shape (1, 5, 5, 5, 1)
# Make it a simple linear field: f(z, y, x) = 2*z + 3*y + 4*x
z, y, x = np.ogrid[:5, :5, :5]
field_np = (2*z + 3*y + 4*x).astype(np.float32)[None, ..., None]

field_torch = torch.tensor(field_np)
field_jax = jnp.array(field_np)

spacing = (0.5, 1.0, 2.0) # dx, dy, dz

jac_torch = _spatial_jacobian_nd(field_torch, physical_spacing=spacing).squeeze(-2)
jac_jax = _spatial_jacobian_nd_jax(field_jax, physical_spacing=spacing).squeeze(-2)

print("Torch Jacobian shape:", jac_torch.shape)
print("Torch Jacobian at center:", jac_torch[0, 2, 2, 2].tolist())
print("JAX Jacobian shape:", jac_jax.shape)
print("JAX Jacobian at center:", jac_jax[0, 2, 2, 2].tolist())
