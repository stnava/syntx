"""Check flow_sigma and composition quality."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])
print(f"Pre-aligned MI: {ants.image_mutual_information(fi, mi_aff):.4f}")

ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[10, 5, 0], syn_metric='mattes')
print(f"ANTs [10,5,0]: MI={ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")

# ANTs default flow_sigma=3.0, total_sigma=0.0
# Try different flow_sigma values  
for fs in [0.0, 1.0, 2.0, 3.0, 5.0]:
    r = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                   reg_iterations=[10, 5, 0], affine_iterations=[0, 0, 0],
                   similarity_metric='mattes_mi', verbose=False, grad_step=0.25,
                   flow_sigma=fs)
    w = ants.apply_transforms(fi, mi_aff, r['fwdtransforms'])
    mi_val = ants.image_mutual_information(fi, w)
    n = len(r['model'].syn_losses)
    print(f"flow_sigma={fs}: MI={mi_val:.4f} ran={n}")

# Try ANTs with different syn_sigma
for ss in [0.0, 3.0]:
    ra2 = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[10, 5, 0], 
                             syn_metric='mattes', syn_sampling=32, 
                             flow_sigma=3.0, total_sigma=0.0)
    print(f"ANTs flow_sigma=3.0, total_sigma=0.0: MI={ants.image_mutual_information(fi, ra2['warpedmovout']):.4f}")
    break
