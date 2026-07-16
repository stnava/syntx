"""Check ANTs SyN inverse convention."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import ants, numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
print(f"fwdtransforms: {ra['fwdtransforms']}")
print(f"invtransforms: {ra['invtransforms']}")

# Check if the GenericAffine in invtransforms is the same file
fwd_aff = [f for f in ra['fwdtransforms'] if f.endswith('.mat')][0]
inv_aff = [f for f in ra['invtransforms'] if f.endswith('.mat')][0]
print(f"\nForward affine: {fwd_aff}")
print(f"Inverse affine: {inv_aff}")
print(f"Same file? {fwd_aff == inv_aff}")

# Check params
tx_fwd = ants.read_transform(fwd_aff)
tx_inv = ants.read_transform(inv_aff)
print(f"Fwd params: {tx_fwd.parameters}")
print(f"Inv params: {tx_inv.parameters}")
print(f"Same? {np.allclose(tx_fwd.parameters, tx_inv.parameters)}")

# The key question: when ANTs internally uses invtransforms, does it know to invert the affine?
# Test both ways
wi1 = ants.apply_transforms(mi, fi, ra['invtransforms'])
wi2 = ants.apply_transforms(mi, fi, ra['invtransforms'], whichtoinvert=[True, False])
print(f"\nInverse (no invert): MI={ants.image_mutual_information(mi, wi1):.4f}")
print(f"Inverse (invert aff): MI={ants.image_mutual_information(mi, wi2):.4f}")
