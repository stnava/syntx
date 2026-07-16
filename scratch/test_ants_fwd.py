import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Create dummy warp mapping Fixed to Moving
# If point in Fixed is x, y
# We want it to look up Moving at x + 20, y + 30
# If ANTs apply_transforms expects a displacement from Fixed to Moving:
warp_arr = np.zeros((*fi.shape, 2), dtype=np.float32)
warp_arr[..., 0] = 20
warp_arr[..., 1] = 30
fwd_img = ants.from_numpy(warp_arr, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True)
ants.image_write(fwd_img, 'scratch/fwd.nii.gz')

w = ants.apply_transforms(fi, mi, ['scratch/fwd.nii.gz'])

com_fi = ants.get_center_of_mass(fi)
com_mi = ants.get_center_of_mass(mi)
com_w = ants.get_center_of_mass(w)
print("FI com:", com_fi)
print("MI com:", com_mi)
print("W com:", com_w)
