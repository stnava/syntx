"""Check ANTs displacement field convention. SyNOnly has no affine."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Affine first, then SyNOnly
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

reg_ants = ants.registration(fi, mi_affine, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')

print("=== ANTs SyNOnly result ===")
print(f"Forward transforms: {reg_ants['fwdtransforms']}")
print(f"Inverse transforms: {reg_ants['invtransforms']}")

# Read fwd displacement
fwd_ants = ants.image_read(reg_ants['fwdtransforms'][0])
d = fwd_ants.numpy()
print(f"\nFwd displacement shape: {d.shape}")
for c in range(d.shape[-1]):
    print(f"  Component {c}: [{d[...,c].min():.3f}, {d[...,c].max():.3f}]")

# Read inv displacement  
for tx in reg_ants['invtransforms']:
    if tx.endswith('.nii.gz'):
        inv_ants = ants.image_read(tx)
        d_inv = inv_ants.numpy()
        print(f"\nInv displacement shape: {d_inv.shape}")
        for c in range(d_inv.shape[-1]):
            print(f"  Component {c}: [{d_inv[...,c].min():.3f}, {d_inv[...,c].max():.3f}]")

mi_fwd = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
print(f"\nMI with warpedmovout: {mi_fwd:.4f}")
print(f"MI initial: {ants.image_mutual_information(fi, mi_affine):.4f}")

# The KEY question: what does "forward" mean in ANTs?
# In ANTs, the "forward" warp maps FIXED space -> MOVING space
# apply_transforms(fixed, moving, fwdtransforms) = moving(fwd(x_fixed))
# So warped(x) = moving(x + displacement(x))
# Displacement d(x) is the vector FROM fixed position TO moving position
