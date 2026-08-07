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
jac_ants = ants.create_jacobian_determinant_image(r16, fwd_tx, do_log=False)

diff = np.abs(dg_ants.numpy() - jac_ants.numpy())
print("ants.deformation_gradient vs ants.create_jacobian_determinant_image:")
print(f"  Max absolute difference: {diff.max():.6e}")
print(f"  Pearson correlation r  : {np.corrcoef(dg_ants.numpy().ravel(), jac_ants.numpy().ravel())[0, 1]:.8f}")

# Now compare both against syntx.spatial.jacobian_determinant
jac_syntx = jacobian_determinant(warp, ref_image=r16)
diff_syntx = np.abs(dg_ants.numpy() - jac_syntx)
print("\nsyntx.spatial.jacobian_determinant vs ants.deformation_gradient:")
print(f"  Max absolute difference: {diff_syntx.max():.6e}")
print(f"  Pearson correlation r  : {np.corrcoef(dg_ants.numpy().ravel(), jac_syntx.ravel())[0, 1]:.8f}")
