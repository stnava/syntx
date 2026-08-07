import numpy as np
import ants
import syntx

data = syntx.benchmark_data('mbhard')
fi, mi = data['fixed'], data['moving']

reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[20, 0, 0])
fwd_tx = reg['fwdtransforms'][0]

# Reference ANTs Jacobian
jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx, do_log=False)
ref_img = jac_ants.numpy()
mask = ants.get_mask(fi).numpy() > 0
ref_vals = ref_img[mask].ravel()

# Warp displacement field
warp_img = ants.image_read(fwd_tx)
arr = warp_img.numpy()  # (Z, Y, X, 3)
sp = fi.spacing  # (sp_x, sp_y, sp_z)

sp_x, sp_y, sp_z = sp[0], sp[1], sp[2]

# Test 1: Current code
comp_map_1 = [2, 1, 0]
sp_axis_1 = [sp_z, sp_y, sp_x]
J1 = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        deriv = np.gradient(arr[..., comp_map_1[i]], axis=j) / sp_axis_1[j]
        if i == j:
            J1[..., i, j] = 1.0 + deriv
        else:
            J1[..., i, j] = deriv
det1 = np.linalg.det(J1)[mask].ravel()
r1 = np.corrcoef(ref_vals, det1)[0, 1]

# Test 2: Standard ITK physical mapping: comp 0=dx (axis 2/sp_x), comp 1=dy (axis 1/sp_y), comp 2=dz (axis 0/sp_z)
comp_map_2 = [0, 1, 2]
axis_map_2 = [2, 1, 0]  # X is axis 2, Y is axis 1, Z is axis 0
sp_map_2 = [sp_x, sp_y, sp_z]
J2 = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        deriv = np.gradient(arr[..., comp_map_2[i]], axis=axis_map_2[j]) / sp_map_2[j]
        if i == j:
            J2[..., i, j] = 1.0 + deriv
        else:
            J2[..., i, j] = deriv
det2 = np.linalg.det(J2)[mask].ravel()
r2 = np.corrcoef(ref_vals, det2)[0, 1]

print(f"Test 1 (Current code)     : r = {r1:.6f}, range=[{det1.min():.3f}, {det1.max():.3f}]")
print(f"Test 2 (ITK Phys Mapping) : r = {r2:.6f}, range=[{det2.min():.3f}, {det2.max():.3f}]")
print(f"ANTs C++ Reference        : range=[{ref_vals.min():.3f}, {ref_vals.max():.3f}]")
