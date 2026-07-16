import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'
fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')

fi_full = ants.image_read(fi_path)
mi_full = ants.image_read(mi_path)

mask = ants.get_mask(fi_full)
mask_dilated = ants.iMath(mask, "MD", 12)
fi = ants.crop_image(fi_full, mask_dilated)
mi = mi_full

print("Fixed image shape:", fi.shape)
print("Fixed image origin:", fi.origin)
print("Fixed image spacing:", fi.spacing)
print("Fixed image direction:\n", fi.direction)

print("\nMoving image shape:", mi.shape)
print("Moving image origin:", mi.origin)
print("Moving image spacing:", mi.spacing)
print("Moving image direction:\n", mi.direction)

# Run ANTs Affine
reg_ants = ants.registration(fi, mi, 'Affine')
tx = ants.read_transform(reg_ants['fwdtransforms'][0])
print("\nANTs Affine Matrix:\n", tx.parameters[:9].reshape(3,3))
print("ANTs Affine Translation:", tx.parameters[9:])

# Let's check: what is the physical coordinate of voxel (0,0,0) in fixed and moving?
print("\nFixed physical coordinate of (0,0,0):", fi.transform_index_to_physical([0, 0, 0]))
print("Moving physical coordinate of (0,0,0):", mi.transform_index_to_physical([0, 0, 0]))

# What is the physical coordinate of center voxel (fi.shape // 2) in fixed?
center_idx_fi = [s // 2 for s in fi.shape]
center_phys_fi = fi.transform_index_to_physical(center_idx_fi)
print(f"\nFixed center index {center_idx_fi} -> physical {center_phys_fi}")

# Warp the center_phys_fi using the ANTs transform
warped_phys_mi = tx.apply_to_point(center_phys_fi)
print("Warped physical coordinate in moving space:", warped_phys_mi)

# Convert warped_phys_mi to moving index coordinate
warped_idx_mi = mi.transform_physical_to_index(warped_phys_mi)
print("Warped index in moving space:", warped_idx_mi)
print("Moving center index:", [s // 2 for s in mi.shape])

# Clean up
for path in reg_ants['fwdtransforms']:
    if os.path.exists(path):
        os.remove(path)
