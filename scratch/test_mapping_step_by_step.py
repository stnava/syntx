import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))

p_phys = np.array([22.5, 191.0, 30.5]) # XYZ physical point

# ANTs index
idx_ants = ants.transform_physical_point_to_index(fi_cropped, list(p_phys))
print("ANTs Voxel Index (XYZ):", idx_ants)

# Syntx step-by-step
# 1. Reverse origin, spacing, direction to ZYX
origin = fi_cropped.origin
spacing = fi_cropped.spacing
direction = fi_cropped.direction

origin_rev = tuple(reversed(origin))
spacing_rev = tuple(reversed(spacing))
direction_rev = np.asarray(direction)[::-1, ::-1].copy()

p_phys_zyx = torch.tensor(p_phys[::-1].copy(), dtype=torch.float32)

print("\norigin_rev (ZYX):", origin_rev)
print("spacing_rev (ZYX):", spacing_rev)
print("direction_rev:\n", direction_rev)
print("p_phys_zyx (ZYX):", p_phys_zyx.numpy())

# 2. Compute diff
diff = p_phys_zyx - torch.tensor(origin_rev, dtype=torch.float32)
print("diff (ZYX):", diff.numpy())

# 3. Rotate
# In ANTs, index is v, physical is p.
# p = D @ (spacing * v) + origin
# So: p - origin = D @ (spacing * v)
# So: spacing * v = D.t() @ (p - origin)
# Let's check what D.t() @ (p - origin) is in XYZ space first!
p_diff_xyz = torch.tensor(p_phys - np.array(origin), dtype=torch.float32)
D_xyz = torch.tensor(direction, dtype=torch.float32)
rotated_xyz = D_xyz.t() @ p_diff_xyz
print("\nrotated_xyz (XYZ, D_xyz.t() @ diff):", rotated_xyz.numpy())
print("voxel_coords_xyz (rotated_xyz / spacing):", (rotated_xyz / torch.tensor(spacing, dtype=torch.float32)).numpy())

# Let's check what direction_rev does in ZYX space
D_zyx = torch.tensor(direction_rev, dtype=torch.float32)
rotated_zyx = diff @ D_zyx
print("\nrotated_zyx (ZYX, diff @ direction_rev):", rotated_zyx.numpy())
print("voxel_coords_zyx (rotated_zyx / spacing_rev):", (rotated_zyx / torch.tensor(spacing_rev, dtype=torch.float32)).numpy())
