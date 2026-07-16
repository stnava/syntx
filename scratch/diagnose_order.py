"""Experimentally verify ANTs transform composition order."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Create a known affine (identity)
identity_affine = ants.new_ants_transform(dimension=2, transform_type='AffineTransform')
identity_affine.set_parameters(np.array([1, 0, 0, 1, 0, 0], dtype=float))
import tempfile
aff_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
ants.write_transform(identity_affine, aff_file)

# Create a known displacement: 5mm shift in X for ALL pixels
disp = np.zeros((*fi.shape, 2), dtype=np.float32)
disp[..., 0] = 5.0  # 5mm in X (first component in ANTs = x direction)
disp_img = ants.from_numpy(disp, has_components=True, spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
warp_file = tempfile.NamedTemporaryFile(suffix='_warp.nii.gz', delete=False).name
ants.image_write(disp_img, warp_file)

# Method 1: warp alone
warped_warp = ants.apply_transforms(fi, mi, [warp_file])
print("Warp alone (5mm X shift) MI:", ants.image_mutual_information(fi, warped_warp))

# Method 2: affine alone (identity)
warped_aff = ants.apply_transforms(fi, mi, [aff_file])
print("Identity affine alone MI:", ants.image_mutual_information(fi, warped_aff))

# Method 3: [warp, affine] - ANTs list order
warped_wa = ants.apply_transforms(fi, mi, [warp_file, aff_file])
print("[warp, identity_affine] MI:", ants.image_mutual_information(fi, warped_wa))

# Method 4: [affine, warp] - reversed order
warped_aw = ants.apply_transforms(fi, mi, [aff_file, warp_file])
print("[identity_affine, warp] MI:", ants.image_mutual_information(fi, warped_aw))

# All should be the same with identity affine
print("\nExpected: all methods with identity affine should give same MI as warp-alone")

print("\n=== Now test with a REAL affine ===")
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff_only = ants.apply_transforms(fi, mi, [tx['fwdtransforms'][0]])

# In ANTs standard registration with SyN:
# fwdtransforms = [warp, affine]
# ANTs applies RIGHT TO LEFT: first affine, then warp
# So: warped(x) = mi(warp(affine(x)))
# = mi(affine(x) + displacement(affine(x)))

# Let's verify: applying [warp, affine] should be same as 
# first applying affine, then warp
warped_chain = ants.apply_transforms(fi, mi, [warp_file, tx['fwdtransforms'][0]])
warped_step1 = ants.apply_transforms(fi, mi, [tx['fwdtransforms'][0]])
warped_step2 = ants.apply_transforms(fi, warped_step1, [warp_file])

mi_chain = ants.image_mutual_information(fi, warped_chain)
mi_step2 = ants.image_mutual_information(fi, warped_step2)
print(f"Chain [warp, affine] MI: {mi_chain:.4f}")
print(f"Step-by-step (affine then warp) MI: {mi_step2:.4f}")
print(f"These should be similar (single interpolation policy caveat)")

