"""Test affine-only forward and inverse."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[0,0,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False)

wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
wi = ants.apply_transforms(mi, fi, rs['invtransforms'])
print(f"Affine-only forward: MI={ants.image_mutual_information(fi, wf):.4f}")
print(f"Affine-only inverse: MI={ants.image_mutual_information(mi, wi):.4f}")
print(f"fwd: {rs['fwdtransforms']}")
print(f"inv: {rs['invtransforms']}")

# ANTs reference
ra = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
wfa = ants.apply_transforms(fi, mi, ra['fwdtransforms'])
wia = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"\nANTs forward: MI={ants.image_mutual_information(fi, wfa):.4f}")
print(f"ANTs inverse: MI={ants.image_mutual_information(mi, wia):.4f}")
