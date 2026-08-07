import ants
import numpy as np
import syntx

# Test ANTsPy deformation_gradient on 2D and 3D
print("1. Testing 2D deformation gradient...")
r16 = ants.image_read(ants.get_ants_data('r16'))
r64 = ants.image_read(ants.get_ants_data('r64'))

reg2d = syntx.syn(fixed=r16, moving=r64, reg_iterations=[20, 0])
fwd_tx_2d = reg2d['fwdtransforms'][0]
warp_2d = ants.image_read(fwd_tx_2d)

try:
    dg_ants_2d = ants.deformation_gradient(warp_2d)
    print("  2D ants.deformation_gradient output:", type(dg_ants_2d))
    if isinstance(dg_ants_2d, ants.ANTsImage):
        print("  2D shape:", dg_ants_2d.shape, "min/max:", dg_ants_2d.min(), dg_ants_2d.max())
    elif isinstance(dg_ants_2d, list):
        print("  2D list length:", len(dg_ants_2d))
except Exception as e:
    print("  2D ants.deformation_gradient error:", e)

print("\n2. Testing 3D deformation gradient (mbhard)...")
data3d = syntx.benchmark_data('mbhard')
fi3d = data3d['fixed']
mi3d = data3d['moving']

reg3d = syntx.syn(fixed=fi3d, moving=mi3d, reg_iterations=[20, 0, 0])
fwd_tx_3d = reg3d['fwdtransforms'][0]
warp_3d = ants.image_read(fwd_tx_3d)

try:
    dg_ants_3d = ants.deformation_gradient(warp_3d)
    print("  3D ants.deformation_gradient output:", type(dg_ants_3d))
    if isinstance(dg_ants_3d, ants.ANTsImage):
        print("  3D shape:", dg_ants_3d.shape, "min/max:", dg_ants_3d.min(), dg_ants_3d.max())
    elif isinstance(dg_ants_3d, list):
        print("  3D list length:", len(dg_ants_3d))
except Exception as e:
    print("  3D ants.deformation_gradient error:", e)
