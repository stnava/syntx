import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Create warp: dx=50, dy=60
warp_arr = np.zeros((*fi.shape, 2), dtype=np.float32)
warp_arr[..., 0] = 50.0
warp_arr[..., 1] = 60.0
fwd_img = ants.from_numpy(warp_arr, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, 'scratch/fwd_test3.nii.gz')

# Create affine: scaling by 2.0
ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2, parameters=(2.0,0,0,2.0,0,0)), 'scratch/aff_test2.mat')
ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2, parameters=(2.0,0,0,2.0,100,120)), 'scratch/aff_A.mat')
ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2, parameters=(2.0,0,0,2.0,50,60)), 'scratch/aff_B.mat')

w3 = ants.apply_transforms(fi, mi, ['scratch/fwd_test3.nii.gz', 'scratch/aff_test2.mat'])
w_A = ants.apply_transforms(fi, mi, ['scratch/aff_A.mat'])
w_B = ants.apply_transforms(fi, mi, ['scratch/aff_B.mat'])

print("W3 COM:", ants.get_center_of_mass(w3))
print("W_A COM:", ants.get_center_of_mass(w_A))
print("W_B COM:", ants.get_center_of_mass(w_B))
