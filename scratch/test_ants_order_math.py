import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Create warp: dx=20, dy=30
warp_arr = np.zeros((*fi.shape, 2), dtype=np.float32)
warp_arr[..., 0] = 20
warp_arr[..., 1] = 30
fwd_img = ants.from_numpy(warp_arr, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, 'scratch/fwd_test.nii.gz')

# Create affine: tx=100, ty=200
# ANTs affine is 2D: [m11, m12, m21, m22, tx, ty]
ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2, parameters=(1,0,0,1,100,200)), 'scratch/aff_test.mat')

w = ants.apply_transforms(fi, mi, ['scratch/fwd_test.nii.gz', 'scratch/aff_test.mat'])
w_rev = ants.apply_transforms(fi, mi, ['scratch/aff_test.mat', 'scratch/fwd_test.nii.gz'])

print("FI com:", ants.get_center_of_mass(fi))
print("MI com:", ants.get_center_of_mass(mi))
print("W com [warp, affine]:", ants.get_center_of_mass(w))
print("W com [affine, warp]:", ants.get_center_of_mass(w_rev))
