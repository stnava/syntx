"""Test inverse_steps impact on convergence."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])
mi0 = ants.image_mutual_information(fi, mi_aff)
print(f"Pre-aligned MI: {mi0:.4f}")

ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[10, 5, 0], syn_metric='mattes')
print(f"ANTs [10,5,0]: MI={ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")

for inv_steps in [1, 3, 5, 10, 20]:
    r = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                   reg_iterations=[10, 5, 0], affine_iterations=[0, 0, 0],
                   similarity_metric='mattes_mi', verbose=False, grad_step=0.25,
                   inverse_steps=inv_steps)
    w = ants.apply_transforms(fi, mi_aff, r['fwdtransforms'])
    mi_val = ants.image_mutual_information(fi, w)
    n = len(r['model'].syn_losses)
    wmax = r['model'].warp_l2r.data.abs().max().item()
    print(f"syntx inv_steps={inv_steps:<3}: MI={mi_val:.4f} ran={n} warp_max={wmax:.2f}mm")
