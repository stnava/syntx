import ants
import numpy as np
import syntx

data = syntx.benchmark_data('mbhard')
fi, mi = data['fixed'], data['moving']

reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[20, 0, 0])
fwd_tx = reg['fwdtransforms'][0]

warp = ants.image_read(fwd_tx)
arr = warp.numpy()

print("fi shape:", fi.shape)
print("warp shape:", warp.shape)
print("arr shape:", arr.shape)
print("warp spacing:", warp.spacing)
print("warp origin:", warp.origin)
print("warp direction:\n", warp.direction)

# Let's compute finite differences along each spatial dimension and test all 6 component x axis permutations!
jac_ants = ants.create_jacobian_determinant_image(fi, fwd_tx, do_log=False)
ref = jac_ants.numpy().ravel()

for comp_perm in [(0,1,2), (2,1,0), (0,2,1), (1,0,2), (1,2,0), (2,0,1)]:
    for axis_perm in [(0,1,2), (2,1,0), (0,2,1), (1,0,2), (1,2,0), (2,0,1)]:
        for sign in [1.0, -1.0]:
            J = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
            for i in range(3):
                c = comp_perm[i]
                a = axis_perm[i]
                sp_val = warp.spacing[a]
                for j in range(3):
                    a_j = axis_perm[j]
                    sp_j = warp.spacing[a_j]
                    deriv = np.gradient(arr[..., c], axis=a_j) / sp_j
                    if i == j:
                        J[..., i, j] = 1.0 + sign * deriv
                    else:
                        J[..., i, j] = sign * deriv
            det_val = np.linalg.det(J).ravel()
            r = np.corrcoef(ref, det_val)[0, 1]
            if r > 0.90:
                print(f"MATCH FOUND! comp_perm={comp_perm}, axis_perm={axis_perm}, sign={sign} -> Pearson r = {r:.6f}")
