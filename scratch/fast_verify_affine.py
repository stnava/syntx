"""Verify affine fix."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Affine only
ra = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mia = ants.image_mutual_information(fi, ra['warpedmovout'])

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[0, 0, 0], affine_iterations=[100, 100, 0],
               similarity_metric='mattes_mi', verbose=False)
ws = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
mis = ants.image_mutual_information(fi, ws)

print(f"ANTs Affine MI: {mia:.4f}")
print(f"syntx Affine MI: {mis:.4f}")

# Full pipeline (affine + SyN)
ra2 = ants.registration(fi, mi, 'SyN', reg_iterations=[10, 5, 0, 0], syn_metric='mattes')
mia2 = ants.image_mutual_information(fi, ra2['warpedmovout'])

rs2 = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                reg_iterations=[10, 5, 0, 0], affine_iterations=[100, 100, 0, 0],
                similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
ws2 = ants.apply_transforms(fi, mi, rs2['fwdtransforms'])
mis2 = ants.image_mutual_information(fi, ws2)

print(f"\nANTs SyN MI: {mia2:.4f}")
print(f"syntx SyNTo MI: {mis2:.4f}")
print(f"Gap: {mia2 - mis2:.4f}")
