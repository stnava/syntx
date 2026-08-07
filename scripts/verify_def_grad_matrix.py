import ants
import numpy as np
import syntx
from syntx.spatial import jacobian_determinant

r16 = ants.image_read(ants.get_ants_data('r16'))
r64 = ants.image_read(ants.get_ants_data('r64'))

reg = syntx.syn(fixed=r16, moving=r64, reg_iterations=[20, 0])
fwd_tx = reg['fwdtransforms'][0]
warp = ants.image_read(fwd_tx)

dg_ants = ants.deformation_gradient(warp)
dg_np = dg_ants.numpy()
print("2D deformation_gradient array shape:", dg_np.shape)  # (256, 256, 4)

# Reshape (H, W, 4) into 2x2 matrix field (H, W, 2, 2)
F_2d = dg_np.reshape(*r16.shape, 2, 2)
det_F_ants = np.linalg.det(F_2d)

# Compare det(F_ants) against syntx.spatial.jacobian_determinant
jac_syntx = jacobian_determinant(warp, ref_image=r16)

diff = np.abs(det_F_ants - jac_syntx)
r = np.corrcoef(det_F_ants.ravel(), jac_syntx.ravel())[0, 1]

print("\n2D Parity:")
print(f"  ants.deformation_gradient det(F) vs syntx.spatial.jacobian_determinant:")
print(f"  Pearson correlation r : {r:.8f}")
print(f"  Max absolute diff     : {diff.max():.6e}")
print(f"  Mean absolute diff    : {diff.mean():.6e}")

# Test 3D (mbhard)
print("\n3D Parity (mbhard):")
data3d = syntx.benchmark_data('mbhard')
fi3d, mi3d = data3d['fixed'], data3d['moving']
reg3d = syntx.syn(fixed=fi3d, moving=mi3d, reg_iterations=[20, 0, 0])
fwd_tx_3d = reg3d['fwdtransforms'][0]
warp3d = ants.image_read(fwd_tx_3d)

dg_ants_3d = ants.deformation_gradient(warp3d)
dg_3d_np = dg_ants_3d.numpy()
print("3D deformation_gradient array shape:", dg_3d_np.shape)  # (D, H, W, 9)

F_3d = dg_3d_np.reshape(*fi3d.shape, 3, 3)
det_F_ants_3d = np.linalg.det(F_3d)

mask3d = ants.get_mask(fi3d).numpy() > 0
jac_syntx_3d = jacobian_determinant(warp3d, ref_image=fi3d)

diff3d = np.abs(det_F_ants_3d[mask3d] - jac_syntx_3d[mask3d])
r3d = np.corrcoef(det_F_ants_3d[mask3d].ravel(), jac_syntx_3d[mask3d].ravel())[0, 1]

print(f"  ants.deformation_gradient det(F) vs syntx.spatial.jacobian_determinant:")
print(f"  Pearson correlation r : {r3d:.8f}")
print(f"  Max absolute diff     : {diff3d.max():.6e}")
print(f"  Mean absolute diff    : {diff3d.mean():.6e}")
