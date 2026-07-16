import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import jax
import jax.numpy as jnp
import syntx

# Create synthetic 3D images
shape = (64, 64, 64)
fi_np = np.zeros(shape, dtype=np.float32)
mi_np = np.zeros(shape, dtype=np.float32)
x, y, z = np.ogrid[:64, :64, :64]
mask_fi = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
fi_np[mask_fi] = 1.0
mask_mi = (x - 35)**2 + (y - 30)**2 + (z - 33)**2 <= 15**2
mi_np[mask_mi] = 1.0

fi = ants.from_numpy(fi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
mi = ants.from_numpy(mi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

import sys
syn_module = sys.modules['syntx.syn']

# Patch PyTorch filter to print max norm of filtered gradient
original_filter_py = syn_module.separable_gaussian_filter
def patched_filter_py(grid, sigma, spacing=None):
    res = original_filter_py(grid, sigma, spacing)
    if sigma == 3.0: # fluid_sigma
        spacing_rev = tuple(reversed(spacing))
        spacing_t = torch.tensor(spacing_rev, device=res.device, dtype=res.dtype)
        grad_voxel = res * spacing_t
        max_norm = torch.sqrt(torch.sum(grad_voxel**2, dim=-1)).max()
        print(f"[PyTorch] max_norm: {float(max_norm):.6f}")
    return res
syn_module.separable_gaussian_filter = patched_filter_py

# Patch JAX filter to print max norm of filtered gradient
original_syn_update_step_jax = syntx.syn_jax.syn_update_step_jax
def patched_syn_update_step_jax(*args, **kwargs):
    grad_l_raw = args[4]
    b_mask = args[7]
    spacing = args[13]
    fluid_sigma = args[16]
    
    # We will print the input components inside syn_update_step_jax
    # Since grad_l_raw is passed, we can check it.
    pass
    
    grad_l = syntx.syn_jax.separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=spacing)
    spacing_t = jnp.array(tuple(reversed(spacing)))
    grad_l_voxel = grad_l * spacing_t
    max_norm_l = jnp.sqrt(jnp.sum(grad_l_voxel**2, axis=-1)).max() + 1e-8
    print(f"[JAX] max_norm: {float(max_norm_l):.6f}")
    
    return original_syn_update_step_jax(*args, **kwargs)
syntx.syn_jax.syn_update_step_jax = patched_syn_update_step_jax

print("Running PyTorch...")
reg_syn_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    levels=[1], reg_iterations=[5], grad_step=0.1, flow_sigma=3.0,
    initial_transform=[]
)

print("\nRunning JAX...")
reg_syn_jax = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
    levels=[1], reg_iterations=[5], grad_step=0.1, flow_sigma=3.0,
    initial_transform=[]
)
