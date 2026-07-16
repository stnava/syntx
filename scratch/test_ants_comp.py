import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Create dummy affine (scale by 2, translation by 0)
tx_aff = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
tx_aff.set_parameters(np.array([2,0,0,2,0,0]))
ants.write_transform(tx_aff, 'scratch/aff.mat')

# Create dummy warp (constant dx=5, dy=0)
warp_arr = np.zeros((*fi.shape, 2), dtype=np.float32)
warp_arr[..., 0] = 5
fwd_img = ants.from_numpy(warp_arr, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, 'scratch/warp.nii.gz')

# Apply [warp, aff]
w1 = ants.apply_transforms(fi, mi, ['scratch/warp.nii.gz', 'scratch/aff.mat'])

# Apply [aff, warp]
w2 = ants.apply_transforms(fi, mi, ['scratch/aff.mat', 'scratch/warp.nii.gz'])

com_fi = ants.get_center_of_mass(fi)
com_mi = ants.get_center_of_mass(mi)
com_w1 = ants.get_center_of_mass(w1)
com_w2 = ants.get_center_of_mass(w2)
print("FI com:", com_fi)
print("MI com:", com_mi)
print("W1 com:", com_w1)
print("W2 com:", com_w2)

