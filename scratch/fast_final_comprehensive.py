"""Final comprehensive 2D parity test."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

print(f"{'='*75}")
print(f"FULL PIPELINE: syntx SyNTo vs ANTs SyN (2D)")
print(f"{'Config':<30} {'syntx':>8} {'ANTs':>8} {'gap':>7} {'s_t':>6} {'a_t':>6}")
print(f"{'='*75}")

for name, syn_its, aff_its in [
    ("Aff [100,100,0,0]", [0,0,0,0], [100,100,0,0]),
    ("Quick [10,5,0]", [10,5,0], [100,100,0]),
    ("Med [50,20,0,0]", [50,20,0,0], [100,100,0,0]),
]:
    t0=time.time()
    rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                    reg_iterations=syn_its, affine_iterations=aff_its,
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
    ts=time.time()-t0
    ws = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
    mis = ants.image_mutual_information(fi, ws)
    
    ants_type = 'SyN' if sum(syn_its) > 0 else 'Affine'
    t0=time.time()
    if ants_type == 'SyN':
        ra = ants.registration(fi, mi, ants_type, reg_iterations=syn_its, syn_metric='mattes')
    else:
        ra = ants.registration(fi, mi, 'Affine', reg_iterations=aff_its[:3])
    ta=time.time()-t0
    mia = ants.image_mutual_information(fi, ra['warpedmovout'])
    
    print(f"{name:<30} {mis:>8.4f} {mia:>8.4f} {mia-mis:>7.4f} {ts:>5.1f}s {ta:>5.1f}s")

# INVERSE QUALITY
print(f"\n{'='*75}")
print(f"INVERSE QUALITY CHECK")
print(f"{'='*75}")

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                reg_iterations=[10,5,0], affine_iterations=[100,100,0],
                similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
wi = ants.apply_transforms(mi, fi, rs['invtransforms'], whichtoinvert=rs['whichtoinvert_inv'])
print(f"syntx Forward MI:  {ants.image_mutual_information(fi, wf):.4f}")
print(f"syntx Inverse MI:  {ants.image_mutual_information(mi, wi):.4f}")
print(f"syntx warpedfixout MI: {ants.image_mutual_information(mi, rs['warpedfixout']):.4f}")

ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
print(f"ANTs Forward MI:   {ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
wa = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"ANTs Inverse MI:   {ants.image_mutual_information(mi, wa):.4f}")

# SYN-ONLY PARITY
print(f"\n{'='*75}")
print(f"SYN-ONLY PARITY (pre-aligned with ANTs Affine)")
print(f"{'='*75}")
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])
for iters in [[5,0,0], [10,5,0], [50,20,0]]:
    t0=time.time()
    rs = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                    reg_iterations=iters, affine_iterations=[0]*len(iters),
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
    ts=time.time()-t0
    ws = ants.apply_transforms(fi, mi_aff, rs['fwdtransforms'])
    mis = ants.image_mutual_information(fi, ws)
    t0=time.time()
    ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=iters, syn_metric='mattes')
    ta=time.time()-t0
    mia = ants.image_mutual_information(fi, ra['warpedmovout'])
    print(f"  {str(iters):<15} syntx={mis:.4f}  ANTs={mia:.4f}  gap={mia-mis:.4f}  t={ts:.1f}/{ta:.1f}s")
