"""Debug: is the affine or the inverse broken?"""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# 1. Check affine-only quality
print("=== AFFINE ONLY ===")
ra = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
print(f"ANTs Affine MI: {ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[0, 0, 0], affine_iterations=[100, 100, 0],
               similarity_metric='mattes_mi', verbose=False)
ws = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
print(f"syntx Affine MI: {ants.image_mutual_information(fi, ws):.4f}")
print(f"syntx fwdtransforms: {rs['fwdtransforms']}")
print(f"syntx invtransforms: {rs['invtransforms']}")

# 2. SyN with ANTs affine
mi_aff = ants.apply_transforms(fi, mi, ra['fwdtransforms'])
print(f"\n=== SYN-ONLY (pre-aligned) ===")
rs2 = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                reg_iterations=[10, 5, 0], affine_iterations=[0, 0, 0],
                similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
ws2 = ants.apply_transforms(fi, mi_aff, rs2['fwdtransforms'])
print(f"syntx SyN-only MI: {ants.image_mutual_information(fi, ws2):.4f}")
print(f"syntx fwdtransforms: {rs2['fwdtransforms']}")

# Check inverse
wi2 = ants.apply_transforms(mi_aff, fi, rs2['invtransforms'], 
                             whichtoinvert=rs2.get('whichtoinvert_inv'))
print(f"syntx inverse MI: {ants.image_mutual_information(mi_aff, wi2):.4f}")

# Read warp files
import numpy as np
for tf in rs2['fwdtransforms']:
    if tf.endswith('.nii.gz'):
        w = ants.image_read(tf)
        print(f"  fwd warp: shape={w.numpy().shape} max={np.abs(w.numpy()).max():.3f}")
for tf in rs2['invtransforms']:
    if tf.endswith('.nii.gz'):
        w = ants.image_read(tf)
        print(f"  inv warp: shape={w.numpy().shape} max={np.abs(w.numpy()).max():.3f}")
