import torch
import jax
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import local_ncc_loss_nd
from syntx.syn_jax import local_ncc_loss_nd_jax

# Generate random input images
np.random.seed(42)
I_np = np.random.rand(1, 1, 32, 32).astype(np.float32)
J_np = np.random.rand(1, 1, 32, 32).astype(np.float32)

# PyTorch
I_torch = torch.tensor(I_np, requires_grad=True)
J_torch = torch.tensor(J_np, requires_grad=True)
loss_py = local_ncc_loss_nd(I_torch, J_torch, window_size=9)
loss_py.backward()
grad_I_py = I_torch.grad.numpy()

# JAX
I_jax = jnp.array(I_np)
J_jax = jnp.array(J_np)

def jax_loss(I, J):
    return local_ncc_loss_nd_jax(I, J, window_size=9)

loss_jax = jax_loss(I_jax, J_jax)
grad_I_jax = jax.grad(jax_loss, argnums=0)(I_jax, J_jax)

print(f"PyTorch loss: {loss_py.item():.6f}")
print(f"JAX loss:     {float(loss_jax):.6f}")
print(f"Loss diff:    {abs(loss_py.item() - float(loss_jax)):.2e}")

print(f"PyTorch grad mean/std/max: {grad_I_py.mean():.6f} / {grad_I_py.std():.6f} / {grad_I_py.max():.6f}")
print(f"JAX grad mean/std/max:     {float(grad_I_jax.mean()):.6f} / {float(grad_I_jax.std()):.6f} / {float(grad_I_jax.max()):.6f}")
print(f"Grad diff:                 {np.abs(grad_I_py - np.array(grad_I_jax)).max():.2e}")
