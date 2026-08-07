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
arr = warp_img.numpy()  # (192, 256, 256, 3) -> (X, Y, Z, 3) or (Z, Y, X, 3)
sp = fi.spacing          # (1.0, 1.0, 1.0)
direction = np.array(fi.direction)

print("direction diagonal:", np.diag(direction))

# Test sign combinations for Option A (comp 0=dx, comp 1=dy, comp 2=dz)
# Physical spacing per axis: sp[0], sp[1], sp[2]
sp_XYZ = [sp[0], sp[1], sp[0]]  # Or [sp[2], sp[1], sp[0]] depending on array order

for arr_order in ['XYZ', 'ZYX']:
    if arr_order == 'XYZ':
        # comp 0=dx (axis 0), comp 1=dy (axis 1), comp 2=dz (axis 2)
        c_map = [0, 1, 2]
        a_map = [0, 1, 2]
        s_map = [sp[0], sp[1], sp[2]]
    else:
        # comp 0=dx (axis 2), comp 1=dy (axis 1), comp 2=dz (axis 0)
        c_map = [0, 1, 2]
        a_map = [2, 1, 0]
        s_map = [sp[0], sp[1], sp[2]]

    for sign_x in [1.0, -1.0]:
        for sign_y in [1.0, -1.0]:
            for sign_z in [1.0, -1.0]:
                signs = [sign_x, sign_y, sign_z]
                J = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
                for i in range(3):
                    c = c_map[i]
                    a_i = a_map[i]
                    for j in range(3):
                        a_j = a_map[j]
                        sp_j = s_map[j]
                        deriv = np.gradient(arr[..., c], axis=a_j) / sp_j
                        if i == j:
                            J[..., i, j] = 1.0 + signs[i] * deriv
                        else:
                            J[..., i, j] = signs[i] * deriv
                det = np.linalg.det(J)[mask].ravel()
                r = np.corrcoef(ref_vals, det)[0, 1]
                if r > 0.95:
                    print(f"MATCH! order={arr_order}, signs=({sign_x}, {sign_y}, {sign_z}) -> Pearson r = {r:.6f}")
