import numpy as np
import torch
import jax
import jax.numpy as jnp
from syntx.syn import local_ncc_loss_nd
from syntx.syn_jax import local_ncc_loss_nd_jax

# Create tiny 2D images
shape = (8, 8)
fi_np = np.zeros(shape, dtype=np.float32)
mi_np = np.zeros(shape, dtype=np.float32)
fi_np[2:6, 2:6] = 1.0
mi_np[3:7, 1:5] = 1.0

fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)

# PyTorch
I_py = torch.tensor(fi_norm, dtype=torch.float32)[None, None, ...].requires_grad_(True)
J_py = torch.tensor(mi_norm, dtype=torch.float32)[None, None, ...].requires_grad_(True)
loss_py = local_ncc_loss_nd(J_py, I_py, window_size=3)
loss_py.backward()

# JAX
I_jax = jnp.array(fi_norm)[None, None, ...]
J_jax = jnp.array(mi_norm)[None, None, ...]
loss_fn_jax = lambda jm, im: local_ncc_loss_nd_jax(jm, im, window_size=3)
val_jax, (g_jm_jax, g_im_jax) = jax.value_and_grad(loss_fn_jax, argnums=(0, 1))(J_jax, I_jax)

# Retrieve internal valid_mask for diagnostics
# PyTorch
I_mean = torch.nn.functional.avg_pool2d(I_py, kernel_size=3, stride=1, padding=1, count_include_pad=True)
J_mean = torch.nn.functional.avg_pool2d(J_py, kernel_size=3, stride=1, padding=1, count_include_pad=True)
I_var = torch.nn.functional.avg_pool2d(I_py**2, kernel_size=3, stride=1, padding=1, count_include_pad=True) - I_mean**2
J_var = torch.nn.functional.avg_pool2d(J_py**2, kernel_size=3, stride=1, padding=1, count_include_pad=True) - J_mean**2
valid_mask = (I_var > 1e-8) & (J_var > 1e-8)

# JAX
from syntx.syn_jax import box_filter_jax
I_mean_jax = box_filter_jax(I_jax, 3)
J_mean_jax = box_filter_jax(J_jax, 3)
I_var_jax = box_filter_jax(I_jax**2, 3) - I_mean_jax**2
J_var_jax = box_filter_jax(J_jax**2, 3) - J_mean_jax**2
valid_mask_jax = (I_var_jax > 1e-8) & (J_var_jax > 1e-8)

print("PyTorch Loss:", loss_py.item())
print("JAX Loss:", float(val_jax))
print("PyTorch active voxels:", valid_mask.sum().item())
print("JAX active voxels:", int(valid_mask_jax.sum()))

print("\nJ_var PyTorch:")
print(J_var[0, 0].detach().numpy())
print("\nJ_var JAX:")
print(np.array(J_var_jax[0, 0]))




print("\nPyTorch I_py.grad:")
print(I_py.grad[0, 0].numpy())

print("\nJAX g_im_jax:")
print(np.array(g_im_jax[0, 0]))
