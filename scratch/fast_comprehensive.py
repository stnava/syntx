"""Comprehensive fast parity test after all fixes."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

print(f"{'='*75}")
print(f"FULL PIPELINE: syntx SyNTo vs ANTs SyN (from raw images)")
print(f"{'Config':<30} {'syntx':>8} {'ANTs':>8} {'gap':>7} {'s_t':>6} {'a_t':>6}")
print(f"{'='*75}")

configs = [
    ("Aff-only [100,100,0,0]", [0,0,0,0], [100,100,0,0]),
    ("Quick [10,5,0] [100,100,0]", [10,5,0], [100,100,0]),
    ("Med [50,20,0,0] [100,100,0,0]", [50,20,0,0], [100,100,0,0]),
]

for name, syn_its, aff_its in configs:
    t0=time.time()
    rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                    reg_iterations=syn_its, affine_iterations=aff_its,
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
    ts=time.time()-t0
    ws = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
    mis = ants.image_mutual_information(fi, ws)
    
    # ANTs with matching settings
    ants_type = 'SyN' if sum(syn_its) > 0 else 'Affine'
    if ants_type == 'SyN':
        t0=time.time()
        ra = ants.registration(fi, mi, ants_type, reg_iterations=syn_its, syn_metric='mattes')
        ta=time.time()-t0
    else:
        t0=time.time()
        ra = ants.registration(fi, mi, 'Affine', reg_iterations=aff_its[:3])
        ta=time.time()-t0
    mia = ants.image_mutual_information(fi, ra['warpedmovout'])
    
    print(f"{name:<30} {mis:>8.4f} {mia:>8.4f} {mia-mis:>7.4f} {ts:>5.1f}s {ta:>5.1f}s")

# Also test inverse quality
print(f"\n{'='*75}")
print(f"INVERSE QUALITY CHECK")
print(f"{'='*75}")
rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                reg_iterations=[10,5,0], affine_iterations=[100,100,0],
                similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
ws_fwd = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
ws_inv = ants.apply_transforms(mi, fi, rs['invtransforms'])
mi_fwd = ants.image_mutual_information(fi, ws_fwd)
mi_inv = ants.image_mutual_information(mi, ws_inv)
print(f"Forward MI:  {mi_fwd:.4f}")
print(f"Inverse MI:  {mi_inv:.4f}")

ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
ra_fwd = ants.image_mutual_information(fi, ra['warpedmovout'])
ra_inv_img = ants.apply_transforms(mi, fi, ra['invtransforms'])
ra_inv = ants.image_mutual_information(mi, ra_inv_img)
print(f"ANTs fwd MI: {ra_fwd:.4f}")
print(f"ANTs inv MI: {ra_inv:.4f}")
