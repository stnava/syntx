import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np

# PyTorch box filter
def box_filter_torch(x, window_size):
    dim = x.dim() - 2
    pad = window_size // 2
    if dim == 2:
        pool_fn = F.avg_pool2d
    elif dim == 3:
        pool_fn = F.avg_pool3d
    return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=True)

# JAX zero-padded box filter
def _conv1d_axis_zero(image, kernel, axis):
    ndim = image.ndim
    axes_order = [i for i in range(ndim) if i != axis] + [axis]
    image_trans = jnp.transpose(image, axes_order)
    orig_trans_shape = image_trans.shape
    N_d = orig_trans_shape[-1]
    image_flat = image_trans.reshape(-1, N_d)
    radius = len(kernel) // 2
    image_padded = jnp.pad(image_flat, ((0, 0), (radius, radius)), mode='constant', constant_values=0.0)
    
    def conv_row(row):
        return jnp.convolve(row, kernel, mode='valid')
    
    out_flat = jax.vmap(conv_row)(image_padded)
    out_trans = out_flat.reshape(orig_trans_shape)
    inv_axes_order = [0] * ndim
    for i, a in enumerate(axes_order):
        inv_axes_order[a] = i
    return jnp.transpose(out_trans, inv_axes_order)

def box_filter_jax_zero(x, window_size):
    ndim = x.ndim - 2
    kernel_1d = jnp.ones(window_size) / window_size
    out = x
    for i in range(ndim):
        out = _conv1d_axis_zero(out, kernel_1d, axis=i + 2)
    return out

# Test with a random 3D tensor: shape (1, 1, 8, 8, 8)
np.random.seed(1234)
x_np = np.random.rand(1, 1, 8, 8, 8).astype(np.float32)

x_torch = torch.tensor(x_np)
x_jax = jnp.array(x_np)

out_torch = box_filter_torch(x_torch, window_size=5).numpy()
out_jax = np.array(box_filter_jax_zero(x_jax, window_size=5))

max_diff = np.max(np.abs(out_torch - out_jax))
print("Max difference between PyTorch and JAX zero-padded box filter:", max_diff)
