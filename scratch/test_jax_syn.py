import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Run PyTorch registration
rs_py = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                  reg_iterations=[10, 5, 0], affine_iterations=[100, 100, 0],
                  similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

wf_py = ants.apply_transforms(fi, mi, rs_py['fwdtransforms'])
wi_py = ants.apply_transforms(mi, fi, rs_py['invtransforms'], whichtoinvert=rs_py['whichtoinvert_inv'])

# Run JAX registration
rs_jax = syntx.syn(fi, mi, 'SyNTo', backend='jax',
                   reg_iterations=[10, 5, 0], affine_iterations=[100, 100, 0],
                   similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

wf_jax = ants.apply_transforms(fi, mi, rs_jax['fwdtransforms'])
wi_jax = ants.apply_transforms(mi, fi, rs_jax['invtransforms'], whichtoinvert=rs_jax['whichtoinvert_inv'])

print("PyTorch Results:")
print(f"  Forward MI: {ants.image_mutual_information(fi, wf_py):.4f}")
print(f"  Inverse MI: {ants.image_mutual_information(mi, wi_py):.4f}")

print("JAX Results:")
print(f"  Forward MI: {ants.image_mutual_information(fi, wf_jax):.4f}")
print(f"  Inverse MI: {ants.image_mutual_information(mi, wi_jax):.4f}")
