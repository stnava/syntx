import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch

base_path = '/Users/stnava/data/mindboggle/volumes'
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")

from syntx.syn import _physical_to_normalized_torch_yfirst

def physical_to_normalized_torch_corrected(phys_coords, target_shape, spacing, origin, direction):
    target_shape_rev = tuple(reversed(target_shape))
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape_rev, spacing_rev, origin_rev, direction_rev)

# Generate 100 random points in mi FOV
shape = np.array(mi.shape)
np.random.seed(42)
random_indices = (np.random.rand(100, 3) * (shape - 1)).astype(int)

mismatches = 0
for idx in random_indices:
    p_phys = ants.transform_index_to_physical_point(mi, list(idx))
    
    # Syntx mapping
    p_phys_zyx = torch.tensor(p_phys[::-1].copy(), dtype=torch.float32).view(1, 1, 1, 1, 3)
    y_norm = physical_to_normalized_torch_corrected(
        p_phys_zyx, mi.shape, mi.spacing, mi.origin, mi.direction
    )
    shape_xyz = torch.tensor(mi.shape, dtype=torch.float32)
    idx_syntx = ((y_norm.view(-1) + 1.0) / 2.0 * (shape_xyz - 1)).numpy()
    
    diff = np.abs(idx - idx_syntx)
    if np.max(diff) > 1e-4:
        mismatches += 1
        print(f"Mismatch at index {idx}: ANTs index={idx}, Syntx index={idx_syntx}, diff={diff}")

print(f"Total mismatches for mi out of 100: {mismatches}")
