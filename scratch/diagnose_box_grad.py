import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn_jax import box_filter_jax

# Shape (1, 1, 64, 64, 64)
shape = (1, 1, 64, 64, 64)
x_np = np.random.rand(*shape).astype(np.float32)

# PyTorch
x_py = torch.tensor(x_np, dtype=torch.float32).requires_grad_(True)
out_py = torch.nn.functional.avg_pool3d(x_py, kernel_size=9, stride=1, padding=4, count_include_pad=True)
loss_py = out_py.sum()
loss_py.backward()
g_py = x_py.grad.numpy()

# JAX
x_jax = jnp.array(x_np)
def loss_fn_jax(x):
    out = box_filter_jax(x, 9)
    return out.sum()
g_jax = jax.grad(loss_fn_jax)(x_jax)
g_jax = np.array(g_jax)

print("PyTorch grad sum:", float(g_py.sum()), "| JAX grad sum:", float(g_jax.sum()))
print("PyTorch grad max:", float(np.abs(g_py).max()), "| JAX grad max:", float(np.abs(g_jax).max()))
print("Max diff:", float(np.abs(g_py - g_jax).max()))
