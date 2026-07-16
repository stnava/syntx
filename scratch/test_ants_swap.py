import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

warp_arr = np.zeros((*fi.shape, 2), dtype=np.float32)
# Set X displacement to 20, Y to 30
warp_arr[..., 0] = 20 # ITK component 0 is X
warp_arr[..., 1] = 30 # ITK component 1 is Y

fwd_img = ants.from_numpy(warp_arr, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, 'scratch/fwd_swap.nii.gz')

w = ants.apply_transforms(fi, mi, ['scratch/fwd_swap.nii.gz'])

com_fi = ants.get_center_of_mass(fi)
com_mi = ants.get_center_of_mass(mi)
com_w = ants.get_center_of_mass(w)
print("FI com:", com_fi)
print("MI com:", com_mi)
print("W com:", com_w)
