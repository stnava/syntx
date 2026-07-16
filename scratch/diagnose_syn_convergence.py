import sys, os
import syntx
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

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

print("Running PyTorch...")
reg_syn_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    levels=[1], reg_iterations=[5], grad_step=0.1, flow_sigma=3.0,
    initial_transform=[], verbose=True
)
print("PyTorch losses:", reg_syn_py['syn_losses'])

print("\nRunning JAX...")
reg_syn_jax = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
    levels=[1], reg_iterations=[5], grad_step=0.1, flow_sigma=3.0,
    initial_transform=[], verbose=True
)
print("JAX losses:", reg_syn_jax['syn_losses'])
