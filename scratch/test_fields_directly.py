import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
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

print("Running PyTorch...")
reg_syn_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    levels=[1], reg_iterations=[15], grad_step=0.1, flow_sigma=3.0,
    initial_transform=[]
)

print("\nRunning JAX...")
reg_syn_jax = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
    levels=[1], reg_iterations=[15], grad_step=0.1, flow_sigma=3.0,
    initial_transform=[]
)

py_field = reg_syn_py['model'].warp_l2r.data.cpu().numpy()
jax_field = np.array(reg_syn_jax['model'].warp_l2r)

# Both should have shape (1, 64, 64, 64, 3)
print(f"\nPyTorch field shape: {py_field.shape}")
print(f"JAX field shape: {jax_field.shape}")

max_diff = np.abs(py_field - jax_field).max()
mean_diff = np.abs(py_field - jax_field).mean()
print(f"Max field diff: {max_diff:.6f}")
print(f"Mean field diff: {mean_diff:.6f}")

print("\nPyTorch field min/max:", py_field.min(), py_field.max())
print("JAX field min/max:", jax_field.min(), jax_field.max())

# Clean up
for reg in [reg_syn_py, reg_syn_jax]:
    for path in reg['fwdtransforms'] + reg['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
