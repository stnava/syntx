import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))

from syntx.syn import _physical_to_normalized_torch_yfirst

def physical_to_normalized_torch_corrected(phys_coords, target_shape, spacing, origin, direction):
    target_shape_rev = tuple(reversed(target_shape))
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape_rev, spacing_rev, origin_rev, direction_rev)

# Pick a physical point in fi_cropped
p_phys = np.array([22.5, 191.0, 30.5]) # XYZ physical point

# Use ANTs to convert physical point to index
idx_ants = ants.transform_physical_point_to_index(fi_cropped, list(p_phys))
print("ANTs Voxel Index (XYZ):", idx_ants)

p_phys_zyx = torch.tensor(p_phys[::-1].copy(), dtype=torch.float32).view(1, 1, 1, 1, 3)

y_norm = physical_to_normalized_torch_corrected(
    p_phys_zyx, fi_cropped.shape, fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)
print("Syntx Corrected Normalized Coordinates (XYZ):", y_norm.numpy().flatten())

shape_xyz = torch.tensor(fi_cropped.shape, dtype=torch.float32)
voxel_coords_syntx = (y_norm.view(-1) + 1.0) / 2.0 * (shape_xyz - 1)
print("Syntx Corrected Voxel Index (XYZ):", voxel_coords_syntx.numpy())
