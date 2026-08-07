import ants
import numpy as np
import syntx

data = syntx.benchmark_data('mbhard')
fi, mi = data['fixed'], data['moving']

reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[20, 0, 0])
fwd_tx = reg['fwdtransforms'][0]

# Reference ANTs C++ Jacobian
jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx, do_log=False)
ref_img = jac_ants.numpy()
mask = ants.get_mask(fi).numpy() > 0
ref_vals = ref_img[mask].ravel()

warp_img = ants.image_read(fwd_tx)
arr = warp_img.numpy()  # (Z, Y, X, 3) in ITK component order [dx, dy, dz]
sp = fi.spacing  # (sp_x, sp_y, sp_z)
direction = np.array(fi.direction)

print("Direction matrix:\n", direction)

# ANTs/ITK 3D convention:
# arr comp 0 = dx, comp 1 = dy, comp 2 = dz
# arr spatial axes: axis 0 = Z (sp[2]), axis 1 = Y (sp[1]), axis 2 = X (sp[0])
# So spatial derivative tensor in matrix index order (z, y, x):
# du_matrix[i, j] = d(u_comp_i)/d(axis_j)
# Mapping: comp 0 (dx) is along axis 2 (X), comp 1 (dy) along axis 1 (Y), comp 2 (dz) along axis 0 (Z).

sp_axis = [sp[2], sp[1], sp[0]]
comp_map = [2, 1, 0]  # row 0 = dz (axis 0), row 1 = dy (axis 1), row 2 = dx (axis 2)

# Compute raw voxel gradients
du = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        du[..., i, j] = np.gradient(arr[..., comp_map[i]], axis=j) / sp_axis[j]

# Direction matrix in matrix index order (Z, Y, X)
# Standard ANTs direction is (X, Y, Z). Matrix order is (Z, Y, X).
D_zyx = direction[::-1, ::-1]

# Transform Jacobian gradient tensor by direction matrix: J_phys = I + D @ du @ D.T
J = np.zeros_like(du)
for z in range(arr.shape[0]):
    pass  # We can do vectorized tensor multiplication

# Vectorized direction transformation:
# du shape: (Z, Y, X, 3, 3)
du_phys = np.einsum('ij,...jk,lk->...il', D_zyx, du, D_zyx)

J = np.eye(3)[None, None, None, :, :] + du_phys
det_phys = np.linalg.det(J)[mask].ravel()

r = np.corrcoef(ref_vals, det_phys)[0, 1]
diff = np.abs(ref_vals - det_phys)

print("================================================================================")
print("             PHYSICAL DIRECTION MATRIX TRANSFORMED 3D JACOBIAN                  ")
print("================================================================================")
print(f"Pearson Correlation r        : {r:.6f}")
print(f"Max Absolute Difference      : {diff.max():.6e}")
print(f"Mean Absolute Difference     : {diff.mean():.6e}")
print(f"95th %ile Absolute Diff      : {np.percentile(diff, 95):.6e}")
print(f"ANTs C++ Reference range     : [{ref_vals.min():.4f}, {ref_vals.max():.4f}]")
print(f"Physical Jacobian range      : [{det_phys.min():.4f}, {det_phys.max():.4f}]")
print("================================================================================")
