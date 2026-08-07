import ants
import numpy as np
import syntx

print("Checking ANTsPy functions for deformation gradient / jacobian...")
has_def_grad = hasattr(ants, 'deformation_gradient')
print(f"  ants.deformation_gradient exists: {has_def_grad}")

if not has_def_grad:
    # List matching functions in ants
    matches = [fn for fn in dir(ants) if 'jacobian' in fn or 'gradient' in fn or 'deformation' in fn]
    print("  Matching ANTs functions:", matches)

r16 = ants.image_read(ants.get_ants_data('r16'))
r64 = ants.image_read(ants.get_ants_data('r64'))

reg = syntx.syn(fixed=r16, moving=r64, reg_iterations=[20, 0])
fwd_tx = reg['fwdtransforms'][0]

if has_def_grad:
    dg_ants = ants.deformation_gradient(r16, fwd_tx)
    print("  dg_ants shape:", dg_ants.shape if hasattr(dg_ants, 'shape') else type(dg_ants))

# Also test create_jacobian_determinant_image
jac_ants = ants.create_jacobian_determinant_image(r16, fwd_tx)
print("  create_jacobian_determinant_image min/max:", jac_ants.min(), jac_ants.max())
