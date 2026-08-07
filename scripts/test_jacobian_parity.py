import numpy as np
import ants
import syntx

# Load r16 and r64
r16 = ants.image_read(ants.get_ants_data('r16'))
r64 = ants.image_read(ants.get_ants_data('r64'))

reg = syntx.syn(fixed=r16, moving=r64, reg_iterations=[50, 20])
fwd_tx = reg['fwdtransforms'][0]

# ANTs reference
jac_ants = ants.create_jacobian_determinant_image(r16, fwd_tx)
jac_ref = jac_ants.numpy()

# Read displacement field
warp_img = ants.image_read(fwd_tx)
arr = warp_img.numpy()  # shape (H, W, 2)
sp = r16.spacing  # (sp_x, sp_y)

sp_x, sp_y = sp[0], sp[1]

# Variant A (Current code): comp 0 = dy (axis 0), comp 1 = dx (axis 1)
du0_d0_A = np.gradient(arr[..., 0], axis=0) / sp_y
du0_d1_A = np.gradient(arr[..., 0], axis=1) / sp_x
du1_d0_A = np.gradient(arr[..., 1], axis=0) / sp_y
du1_d1_A = np.gradient(arr[..., 1], axis=1) / sp_x
jac_A = (1.0 + du0_d0_A) * (1.0 + du1_d1_A) - du0_d1_A * du1_d0_A

# Variant B (ITK standard): comp 0 = dx (axis 1), comp 1 = dy (axis 0)
du_x_dx = np.gradient(arr[..., 0], axis=1) / sp_x
du_x_dy = np.gradient(arr[..., 0], axis=0) / sp_y
du_y_dx = np.gradient(arr[..., 1], axis=1) / sp_x
du_y_dy = np.gradient(arr[..., 1], axis=0) / sp_y
jac_B = (1.0 + du_x_dx) * (1.0 + du_y_dy) - du_x_dy * du_y_dx

print("Variant A (Current):")
print(f"  Corr r = {np.corrcoef(jac_ref.ravel(), jac_A.ravel())[0,1]:.8f}")
print(f"  Max abs diff = {np.max(np.abs(jac_ref - jac_A)):.6e}")
print(f"  Mean abs diff = {np.mean(np.abs(jac_ref - jac_A)):.6e}")

print("\nVariant B (ITK Standard comp 0=dx, comp 1=dy):")
print(f"  Corr r = {np.corrcoef(jac_ref.ravel(), jac_B.ravel())[0,1]:.8f}")
print(f"  Max abs diff = {np.max(np.abs(jac_ref - jac_B)):.6e}")
print(f"  Mean abs diff = {np.mean(np.abs(jac_ref - jac_B)):.6e}")
