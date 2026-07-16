import os
import sys
sys.path.insert(0, 'src')
import ants
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'

# Fixed Image: OASIS-TRT-20-1
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

# Moving Image: MMRR-21-1
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi = ants.image_read(fixed_img_path)
fl = ants.image_read(fixed_lbl_path)
mi = ants.image_read(moving_img_path)
ml = ants.image_read(moving_lbl_path)

fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)

# Compute H_x and H_y exactly as in grid_to_physical_affine (ZYX order)
Nx = np.array(list(reversed(fi_low.shape)), dtype=np.float32)
Ny = np.array(list(reversed(mi_low.shape)), dtype=np.float32)

Sx = np.array(fi_low.spacing)[::-1]
Sy = np.array(mi_low.spacing)[::-1]
Ox = np.array(fi_low.origin)[::-1]
Oy = np.array(mi_low.origin)[::-1]
Dx = np.array(fi_low.direction)[::-1, ::-1]
Dy = np.array(mi_low.direction)[::-1, ::-1]

dim = 3

Kx = np.diag((Nx - 1) / 2.0)
Cx = (Nx - 1) / 2.0
Vx = Dx @ np.diag(Sx) @ Kx
cx = Dx @ np.diag(Sx) @ Cx + Ox

H_x = np.eye(4)
H_x[:3, :3] = Vx
H_x[:3, 3] = cx

Ky = np.diag((Ny - 1) / 2.0)
Cy = (Ny - 1) / 2.0
Vy = Dy @ np.diag(Sy) @ Ky
cy = Dy @ np.diag(Sy) @ Cy + Oy

H_y = np.eye(4)
H_y[:3, :3] = Vy
H_y[:3, 3] = cy

# Set physical translation to best_t (in ZYX order!)
# best_t in ZYX:
best_t_zyx = cy - cx
T_phys = np.eye(4)
T_phys[:3, 3] = best_t_zyx

# Compute T_init in ZYX
T_init_zyx = np.linalg.inv(H_y) @ T_phys @ H_x

# Permute T_init_zyx back to XYZ grid order (which grid_to_physical_affine will permute back to ZYX)
# Since grid_to_physical_affine does:
# T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
# T_yx[:dim, dim] = T_grid[:dim, dim][perm]
# We must permute T_init_zyx from ZYX to XYZ!
perm = [2, 1, 0]
T_grid = np.eye(4)
T_grid[:3, :3] = T_init_zyx[:3, :3][perm][:, perm]
T_grid[:3, 3] = T_init_zyx[:3, 3][perm]

print(f"Computed T_grid:\n{T_grid}")

# Verify using grid_to_physical_affine
from syntx.syn import grid_to_physical_affine
M_phys, t_phys = grid_to_physical_affine(T_grid, fi_low, mi_low)
print(f"M_phys:\n{M_phys}")
print(f"t_phys: {t_phys}")

# Warp high-res labels fl and ml
affine_file = 'scratch/test_t_init_transform.mat'
tx = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
tx.set_fixed_parameters(np.zeros(dim))
ants.write_transform(tx, affine_file)

warped_labels = ants.apply_transforms(
    fixed=fl,
    moving=ml,
    transformlist=[affine_file],
    interpolator='nearestNeighbor'
)

overlap = ants.label_overlap_measures(fl, warped_labels)
overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
if 'TargetOverlap' in overlap_valid.columns:
    dice = overlap_valid['TargetOverlap'].mean()
else:
    dice = overlap_valid['MeanOverlap'].mean()
print(f"DICE score under physical identity (header/CoM alignment only): {dice:.4f}")

os.remove(affine_file)
