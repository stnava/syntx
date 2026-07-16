import ants
import numpy as np

fixed = ants.image_read(ants.get_ants_data('r16'))
fixed_label = ants.threshold_image(fixed, 'Otsu', 3)

tx = ants.new_ants_transform(precision='float', dimension=2)
tx.set_parameters([1.0, 0.0, 0.0, 1.0, 5.0, -3.0])

# Let's find first non-zero index in original label
orig_indices = np.argwhere(fixed_label.numpy() > 0)
idx_orig = tuple(int(x) for x in orig_indices[0]) # (y, x) in numpy
print("Numpy index orig:", idx_orig)

# ITK index in ANTs is (x, y)
itk_idx_orig = (idx_orig[1], idx_orig[0])
print("ITK index orig:", itk_idx_orig)

# Convert to physical point
pt_orig = ants.transform_index_to_physical_point(fixed, itk_idx_orig)
print("Physical point orig:", pt_orig)

# Map via transform
pt_warped = tx.apply_to_point(pt_orig)
print("Physical point warped via tx.apply_to_point:", pt_warped)

# Convert back to index in fixed image
itk_idx_warped = ants.transform_physical_point_to_index(fixed, pt_warped)
print("ITK index warped:", itk_idx_warped)
numpy_idx_warped = (itk_idx_warped[1], itk_idx_warped[0])
print("Numpy index warped:", numpy_idx_warped)

# Let's write the transform and run apply_transforms
import tempfile
import os
fd, tx_path = tempfile.mkstemp(suffix='.mat')
os.close(fd)
try:
    ants.write_transform(tx, tx_path)
    warped_ants = ants.apply_transforms(
        fixed=fixed, 
        moving=fixed_label, 
        transformlist=[tx_path], 
        interpolator='nearestNeighbor'
    )
finally:
    if os.path.exists(tx_path):
        os.remove(tx_path)

# Let's find where the warped label is non-zero in warped_ants
ants_indices = np.argwhere(warped_ants.numpy() > 0)
idx_ants = tuple(int(x) for x in ants_indices[0])
print("Actual non-zero numpy index in warped_ants:", idx_ants)
itk_idx_ants = (idx_ants[1], idx_ants[0])
print("Actual non-zero ITK index in warped_ants:", itk_idx_ants)
pt_ants = ants.transform_index_to_physical_point(fixed, itk_idx_ants)
print("Actual non-zero physical point in warped_ants:", pt_ants)
