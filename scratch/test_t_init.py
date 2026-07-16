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

# Let's use low-res images to compute best_t
fi_low = ants.resample_image(fi, (2, 2, 2), use_voxels=False, interp_type=0)
mi_low = ants.resample_image(mi, (2, 2, 2), use_voxels=False, interp_type=0)

# Compute centers of mass in physical space
# For simplicity, let's use the foreground Center of Mass from fi and mi
# or let's use the FOV center translation as best_t.
# Since we know com_fixed_fov and com_moving_fov:
fixed_shape = np.array(fi_low.shape)
fixed_spacing = np.array(fi_low.spacing)
fixed_origin = np.array(fi_low.origin)
fixed_direction = np.array(fi_low.direction)

moving_shape = np.array(mi_low.shape)
moving_spacing = np.array(mi_low.spacing)
moving_origin = np.array(mi_low.origin)
moving_direction = np.array(mi_low.direction)

dim = 3

# Compute H_x for fixed image (z, y, x order for spacing/shape in Vy math, but let's reverse to match physical XYZ)
# H_x = [ [Dx @ diag(Sx) @ Kx,  com_fixed_fov], [0, 1] ]
# Let's do it in XYZ physical space:
Nx = fixed_shape[::-1]
Sx = fixed_spacing
Ox = fixed_origin
Dx = fixed_direction
Cx = (Nx - 1) / 2.0
Vx = Dx @ np.diag(Sx) @ np.diag(Cx) # Wait, Ky has (Ny-1)/2, so Kx has (Nx-1)/2. So diag(Cx) is exactly Kx!
cx = Dx @ (Sx * Cx) + Ox

H_x = np.eye(4)
H_x[:3, :3] = Vx
H_x[:3, 3] = cx

# Compute H_y for moving image
Ny = moving_shape[::-1]
Sy = moving_spacing
Oy = moving_origin
Dy = moving_direction
Cy = (Ny - 1) / 2.0
Vy = Dy @ np.diag(Sy) @ np.diag(Cy)
cy = Dy @ (Sy * Cy) + Oy

H_y = np.eye(4)
H_y[:3, :3] = Vy
H_y[:3, 3] = cy

# Set physical translation to best_t (using com_moving_fov - com_fixed_fov as a start)
best_t = cy - cx # t_fov
T_phys = np.eye(4)
T_phys[:3, 3] = best_t

# Compute T_init
T_init = np.linalg.inv(H_y) @ T_phys @ H_x
print(f"Computed T_init:\n{T_init}")

# Export physical transform using grid_to_physical_affine to verify it matches identity physical mapping
from syntx.syn import grid_to_physical_affine
M_phys, t_phys = grid_to_physical_affine(T_init, fi_low, mi_low)
print(f"M_phys:\n{M_phys}")
print(f"t_phys: {t_phys}")

# Warp high-res labels fl and ml using this physical transform
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
