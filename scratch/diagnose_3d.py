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

fi = ants.from_numpy(fi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
mi = ants.from_numpy(mi_np, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

import sys
import importlib
syn_module = importlib.import_module('syntx.syn')

# Patch PyTorch to print epoch 0 intermediate values
original_fit_py = syn_module.SyNTo.fit
def patched_fit_py(self, *args, **kwargs):
    # We will hook into the fit loop or we can just print values during the run
    # Let's override SyNTo's prepare_mid_images_and_gradients_torch or let's just inspect it
    return original_fit_py(self, *args, **kwargs)

# Let's patch prepare_mid_images_and_gradients_torch instead
original_prep_py = syn_module.prepare_mid_images_and_gradients_torch
def patched_prep_py(*args, **kwargs):
    I_mid, J_mid, g_I, g_J = original_prep_py(*args, **kwargs)
    # Since this is called during the fit loop, we can print it
    print("[PyTorch] J_mid min/max:", float(J_mid.min()), float(J_mid.max()))
    print("[PyTorch] I_mid min/max:", float(I_mid.min()), float(I_mid.max()))
    print("[PyTorch] prepare_mid: g_I max:", float(g_I.abs().max()))
    return I_mid, J_mid, g_I, g_J
syn_module.prepare_mid_images_and_gradients_torch = patched_prep_py

# Let's patch local_ncc_loss_nd to print the gradient
original_loss_py = syn_module.local_ncc_loss_nd
def patched_loss_py(I, J, *args, **kwargs):
    res = original_loss_py(I, J, *args, **kwargs)
    # I is J_mid, J is I_mid in the fit loop call: fn(J_mid, I_mid)
    if I.requires_grad:
        I.register_hook(lambda grad: print("[PyTorch] J_mid.grad max:", float(grad.abs().max())))
    if J.requires_grad:
        J.register_hook(lambda grad: print("[PyTorch] I_mid.grad max:", float(grad.abs().max())))
    return res
syn_module.local_ncc_loss_nd = patched_loss_py

print("Running PyTorch...")
reg_syn_py = syn_module.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    levels=[1], reg_iterations=[1], grad_step=0.1, flow_sigma=3.0,
    syn_metric='lncc', lncc_radius=4, initial_transform=[]
)

# Patch JAX to print intermediates
import syntx.syn_jax as jax_module
original_prep_jax = jax_module.prepare_mid_images_and_gradients_jax
def patched_prep_jax(*args, **kwargs):
    I_mid, J_mid, g_I, g_J = original_prep_jax(*args, **kwargs)
    print("[JAX] prepare_mid: g_I max:", float(jnp.abs(g_I).max()))
    return I_mid, J_mid, g_I, g_J
jax_module.prepare_mid_images_and_gradients_jax = patched_prep_jax

print("\nRunning JAX...")
reg_syn_jax = syn_module.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
    levels=[1], reg_iterations=[1], grad_step=0.1, flow_sigma=3.0,
    syn_metric='lncc', lncc_radius=4, initial_transform=[]
)
