"""Check if inv warp file content is correct."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# SyN-only on pre-aligned images (bypassing affine issue)
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

rs = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[0,0,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

# SyN-only: check forward and inverse
wf = ants.apply_transforms(fi, mi_aff, rs['fwdtransforms'])
wi = ants.apply_transforms(mi_aff, fi, rs['invtransforms'])
print(f"SyN-only:")
print(f"  Forward: MI={ants.image_mutual_information(fi, wf):.4f}")
print(f"  Inverse: MI={ants.image_mutual_information(mi_aff, wi):.4f}")

# Check warp consistency
fwd_w = ants.image_read([f for f in rs['fwdtransforms'] if f.endswith('.nii.gz')][0])
inv_w = ants.image_read([f for f in rs['invtransforms'] if f.endswith('.nii.gz')][0])
print(f"\n  fwd warp: shape={fwd_w.numpy().shape}, max={np.abs(fwd_w.numpy()).max():.3f}")
print(f"  inv warp: shape={inv_w.numpy().shape}, max={np.abs(inv_w.numpy()).max():.3f}")

# Warp_l2r + warp_r2l should approximately cancel (sum near zero)
fwd_np = fwd_w.numpy()
inv_np = inv_w.numpy()
print(f"  sum(fwd+inv): max={np.abs(fwd_np + inv_np).max():.3f}")

# Now test: Full pipeline with affine
rs2 = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                reg_iterations=[10,5,0], affine_iterations=[100,100,0],
                similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

wf2 = ants.apply_transforms(fi, mi, rs2['fwdtransforms'])
wi2 = ants.apply_transforms(mi, fi, rs2['invtransforms'])
print(f"\nFull pipeline:")
print(f"  Forward: MI={ants.image_mutual_information(fi, wf2):.4f}")
print(f"  Inverse: MI={ants.image_mutual_information(mi, wi2):.4f}")
print(f"  fwdtransforms: {rs2['fwdtransforms']}")
print(f"  invtransforms: {rs2['invtransforms']}")

# Try: what if the affine export is transposed?
inv_aff = [f for f in rs2['invtransforms'] if f.endswith('.mat')][0]
tx_inv = ants.read_transform(inv_aff)
print(f"\n  inv_aff params: {tx_inv.parameters}")
print(f"  inv_aff fixed params: {tx_inv.fixed_parameters}")

# Read the fwd affine too
fwd_aff = [f for f in rs2['fwdtransforms'] if f.endswith('.mat')][0]
tx_fwd = ants.read_transform(fwd_aff)
print(f"  fwd_aff params: {tx_fwd.parameters}")

# ANTs reference
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
fwd_aff_a = [f for f in ra['fwdtransforms'] if f.endswith('.mat')][0]
tx_fwd_a = ants.read_transform(fwd_aff_a)
print(f"\n  ANTs fwd_aff params: {tx_fwd_a.parameters}")
