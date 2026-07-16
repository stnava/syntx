"""Final inverse test after all fixes."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
wi = ants.apply_transforms(mi, fi, rs['invtransforms'])
print(f"Forward: MI={ants.image_mutual_information(fi, wf):.4f}")
print(f"Inverse: MI={ants.image_mutual_information(mi, wi):.4f}")

ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
print(f"ANTs fwd: MI={ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
wi_a = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"ANTs inv: MI={ants.image_mutual_information(mi, wi_a):.4f}")
