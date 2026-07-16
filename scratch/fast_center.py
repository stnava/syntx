"""Check ANTs transform center convention."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import numpy as np
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# ANTs affine
ra = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
tx = ants.read_transform(ra['fwdtransforms'][0])
print(f"ANTs params: {tx.parameters}")
print(f"ANTs fixed_params: {tx.fixed_parameters}")
print(f"ANTs transform_type: {tx.transform_type}")

# ITK AffineTransform:
# output = M * (input - center) + center + translation
# fixed_parameters = center
# parameters = [M.ravel(), translation]

# When center=(0,0), it simplifies to: output = M * input + translation
# But when center!=0: output = M * input + (I-M)*center + translation

# ANTs SyN uses center=(0,0) — so it IS just M * input + t
# But let me verify syntx also uses center=(0,0)
print(f"\nIs center zero? {np.allclose(tx.fixed_parameters, 0)}")

# Let me also check: syntx affine with same data
import syntx
rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[0,0,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False)
tx2 = ants.read_transform([f for f in rs['fwdtransforms'] if f.endswith('.mat')][0])
print(f"\nsyntx params: {tx2.parameters}")
print(f"syntx fixed_params: {tx2.fixed_parameters}")

# Compare affine quality
w1 = ants.apply_transforms(fi, mi, ra['fwdtransforms'])
w2 = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
print(f"\nANTs Affine MI: {ants.image_mutual_information(fi, w1):.4f}")
print(f"syntx Affine MI: {ants.image_mutual_information(fi, w2):.4f}")

# Cross-test: apply syntx affine inverse
tx_inv = ants.read_transform([f for f in rs['invtransforms'] if f.endswith('.mat')][0])
wi = ants.apply_transforms(mi, fi, [tx_inv.parameters], whichtoinvert=[False])
print(f"\nsyntx affine inverse MI: N/A (can't use params directly)")

# Apply using the file
wi2 = ants.apply_transforms(mi, fi, rs['invtransforms'])
print(f"syntx affine inverse (file): MI={ants.image_mutual_information(mi, wi2):.4f}")

# Apply ANTs inverse
wi3 = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"ANTs inverse (file): MI={ants.image_mutual_information(mi, wi3):.4f}")
