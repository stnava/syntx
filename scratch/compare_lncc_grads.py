import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn_jax import local_ncc_loss_nd_jax
from syntx.syn import local_ncc_loss_nd

# Create random inputs
np.random.seed(42)
I_np = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)
J_np = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)

# PyTorch
I_py = torch.tensor(I_np, requires_grad=True)
J_py = torch.tensor(J_np)
loss_py = local_ncc_loss_nd(J_py, I_py, window_size=9)
loss_py.backward()
grad_py = I_py.grad.numpy()

# JAX
I_jax = jnp.array(I_np)
J_jax = jnp.array(J_np)
loss_fn = lambda x: local_ncc_loss_nd_jax(J_jax, x, window_size=9)
val_jax, grad_jax = jax.value_and_grad(loss_fn)(I_jax)
grad_jax = np.array(grad_jax)

print("PyTorch Loss:", float(loss_py.item()))
print("JAX Loss:", float(val_jax))
print("Max absolute difference in loss:", abs(float(loss_py.item()) - float(val_jax)))
print("Max absolute difference in gradients:", np.abs(grad_py - grad_jax).max())
