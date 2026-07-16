import ants
import numpy as np

fixed = ants.image_read(ants.get_ants_data('r16'))
fixed_label = ants.threshold_image(fixed, 'Otsu', 3)

tx = ants.new_ants_transform(precision='float', dimension=2)
tx.set_parameters([1.0, 0.0, 0.0, 1.0, 5.0, -3.0])

import tempfile
import os
fd, temp_tx_path = tempfile.mkstemp(suffix='.mat')
os.close(fd)
try:
    ants.write_transform(tx, temp_tx_path)
    warped_ants = ants.apply_transforms(
        fixed=fixed, 
        moving=fixed_label, 
        transformlist=[temp_tx_path], 
        interpolator='nearestNeighbor'
    )
finally:
    if os.path.exists(temp_tx_path):
        os.remove(temp_tx_path)

# Let's inspect the physical coordinates of the first non-zero pixel
y_orig, x_orig = np.where(fixed_label.numpy() > 0)
orig_idx = (int(x_orig[0]), int(y_orig[0])) # (col, row)
orig_phys = fixed_label.to_physical_point(orig_idx)
print("Original index (col, row):", orig_idx, "Physical:", orig_phys)

y_warp, x_warp = np.where(warped_ants.numpy() > 0)
warp_idx = (int(x_warp[0]), int(y_warp[0]))
warp_phys = warped_ants.to_physical_point(warp_idx)
print("Warped index (col, row):", warp_idx, "Physical:", warp_phys)

# Check the shift in physical space
print("Physical shift (warp - orig):", np.array(warp_phys) - np.array(orig_phys))
