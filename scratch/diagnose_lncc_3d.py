import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, 'src')
from syntx.syn_jax import box_filter_jax

shape = (64, 64, 64)
x_np = np.random.rand(*shape).astype(np.float32)
y_np = np.random.rand(*shape).astype(np.float32)

# --- PyTorch ---
x_py = torch.tensor(x_np, dtype=torch.float32)[None, None, ...].requires_grad_(True)
y_py = torch.tensor(y_np, dtype=torch.float32)[None, None, ...]

def box_filter_py(x):
    return torch.nn.functional.avg_pool3d(x, kernel_size=9, stride=1, padding=4, count_include_pad=True)

x_mean_py = box_filter_py(x_py)
y_mean_py = box_filter_py(y_py)
x_var_py = box_filter_py(x_py**2) - x_mean_py**2
y_var_py = box_filter_py(y_py**2) - y_mean_py**2
cov_py = box_filter_py(x_py * y_py) - x_mean_py * y_mean_py

safe_x_var_py = torch.clamp(x_var_py, min=1e-8)
safe_y_var_py = torch.clamp(y_var_py, min=1e-8)
cc_py = cov_py / (torch.sqrt(safe_x_var_py * safe_y_var_py) + 1e-8)

# Compute gradients of different stages
grad_mean_py = torch.autograd.grad(x_mean_py.sum(), x_py, retain_graph=True)[0].numpy()
grad_var_py = torch.autograd.grad(x_var_py.sum(), x_py, retain_graph=True)[0].numpy()
grad_cov_py = torch.autograd.grad(cov_py.sum(), x_py, retain_graph=True)[0].numpy()
grad_cc_py = torch.autograd.grad(cc_py.sum(), x_py, retain_graph=True)[0].numpy()

# --- JAX ---
x_jax = jnp.array(x_np)[None, None, ...]
y_jax = jnp.array(y_np)[None, None, ...]

def get_jax_grads(x):
    x_mean = box_filter_jax(x, 9)
    y_mean = box_filter_jax(y_jax, 9)
    x_var = box_filter_jax(x**2, 9) - x_mean**2
    y_var = box_filter_jax(y_jax**2, 9) - y_mean**2
    cov = box_filter_jax(x * y_jax, 9) - x_mean * y_mean
    
    safe_x_var = jnp.maximum(x_var, 1e-8)
    safe_y_var = jnp.maximum(y_var, 1e-8)
    cc = cov / (jnp.sqrt(safe_x_var * safe_y_var) + 1e-8)
    
    return x_mean.sum(), x_var.sum(), cov.sum(), cc.sum()

g_mean_jax = jax.grad(lambda x: get_jax_grads(x)[0])(x_jax)
g_var_jax = jax.grad(lambda x: get_jax_grads(x)[1])(x_jax)
g_cov_jax = jax.grad(lambda x: get_jax_grads(x)[2])(x_jax)
g_cc_jax = jax.grad(lambda x: get_jax_grads(x)[3])(x_jax)

print("--- Gradient Max Absolute Comparison ---")
print(f"Mean | PyTorch: {np.abs(grad_mean_py).max():.6f} | JAX: {np.abs(g_mean_jax).max():.6f}")
print(f"Var  | PyTorch: {np.abs(grad_var_py).max():.6f} | JAX: {np.abs(g_var_jax).max():.6f}")
print(f"Cov  | PyTorch: {np.abs(grad_cov_py).max():.6f} | JAX: {np.abs(g_cov_jax).max():.6f}")
print(f"CC   | PyTorch: {np.abs(grad_cc_py).max():.6f} | JAX: {np.abs(g_cc_jax).max():.6f}")
