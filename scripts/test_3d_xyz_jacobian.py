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
arr = warp_img.numpy()  # ANTsPy numpy array shape: (X, Y, Z, 3) or (Z, Y, X, 3)
sp = fi.spacing          # (sp_x, sp_y, sp_z)

print("fi shape:", fi.shape)
print("warp_img shape:", warp_img.shape)
print("arr shape:", arr.shape)
print("sp:", sp)

# If arr shape is (X, Y, Z, 3):
# comp 0 = dx, comp 1 = dy, comp 2 = dz
# axis 0 = X (sp[0]), axis 1 = Y (sp[1]), axis 2 = Z (sp[2])
# Diagonal entries: J[i, i] = 1 + d(u_i)/d(axis_i)

# Option A: Array is indexed as (X, Y, Z, 3)
sp_XYZ = [sp[0], sp[1], sp[2]]
JA = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        deriv = np.gradient(arr[..., i], axis=j) / sp_XYZ[j]
        if i == j:
            JA[..., i, j] = 1.0 + deriv
        else:
            JA[..., i, j] = deriv
detA = np.linalg.det(JA)[mask].ravel()
rA = np.corrcoef(ref_vals, detA)[0, 1]

# Option B: Array is indexed as (Z, Y, X, 3)
sp_ZYX = [sp[2], sp[1], sp[0]]
JB = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        # comp 0 = dz, comp 1 = dy, comp 2 = dx
        deriv = np.gradient(arr[..., 2 - i], axis=j) / sp_ZYX[j]
        if i == j:
            JB[..., i, j] = 1.0 + deriv
        else:
            JB[..., i, j] = deriv
detB = np.linalg.det(JB)[mask].ravel()
rB = np.corrcoef(ref_vals, detB)[0, 1]

print("Option A (XYZ spatial indexing, comp 0=dx, axis 0=X):")
print(f"  Corr r = {rA:.8f}")
print(f"  Max abs diff = {np.max(np.abs(ref_vals - detA)):.6e}")
print(f"  Mean abs diff = {np.mean(np.abs(ref_vals - detA)):.6e}")

print("\nOption B (ZYX spatial indexing, comp 0=dz, axis 0=Z):")
print(f"  Corr r = {rB:.8f}")
print(f"  Max abs diff = {np.max(np.abs(ref_vals - detB)):.6e}")
print(f"  Mean abs diff = {np.mean(np.abs(ref_vals - detB)):.6e}")
