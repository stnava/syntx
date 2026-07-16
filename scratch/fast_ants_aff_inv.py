"""Check ANTs Affine inverse."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

ra = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
print(f"Forward: MI={ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
print(f"fwdtransforms: {ra['fwdtransforms']}")
print(f"invtransforms: {ra['invtransforms']}")

# Apply inverse
wi = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"Inverse: MI={ants.image_mutual_information(mi, wi):.4f}")

# Try inverting the forward instead
wi2 = ants.apply_transforms(mi, fi, ra['fwdtransforms'], whichtoinvert=[True])
print(f"Inverse (invert fwd): MI={ants.image_mutual_information(mi, wi2):.4f}")
