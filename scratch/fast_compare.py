"""Side-by-side comparison: syntx vs ANTs with IDENTICAL settings, fast."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# ANTs affine pre-alignment
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])
mi0 = ants.image_mutual_information(fi, mi_aff)
print(f"Pre-aligned MI: {mi0:.4f}")

# Quick sweep: SyN-only at each resolution level separately
for iters in [[5,0,0], [10,0,0], [5,5,0], [10,5,0]]:
    # syntx
    t0 = time.time()
    r = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                   reg_iterations=iters, affine_iterations=[0]*len(iters),
                   similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
    t_s = time.time() - t0
    w = ants.apply_transforms(fi, mi_aff, r['fwdtransforms'])
    mi_s = ants.image_mutual_information(fi, w)
    
    # ANTs
    t0 = time.time()
    ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=iters, syn_metric='mattes')
    t_a = time.time() - t0
    mi_a = ants.image_mutual_information(fi, ra['warpedmovout'])
    
    n = len(r['model'].syn_losses)
    print(f"iters={str(iters):<12} syntx={mi_s:.4f} ({t_s:.1f}s, ran={n})  ANTs={mi_a:.4f} ({t_a:.1f}s)  gap={mi_a-mi_s:.4f}")

# Also try LNCC for syntx
print("\n--- With LNCC ---")
for iters in [[5,0,0], [10,5,0]]:
    r = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                   reg_iterations=iters, affine_iterations=[0]*len(iters),
                   similarity_metric='lncc', verbose=False, grad_step=0.25)
    w = ants.apply_transforms(fi, mi_aff, r['fwdtransforms'])
    mi_s = ants.image_mutual_information(fi, w)
    print(f"iters={str(iters):<12} syntx(lncc)={mi_s:.4f}  ran={len(r['model'].syn_losses)}")

# Try NCC
print("\n--- With NCC ---")
for iters in [[5,0,0], [10,5,0]]:
    r = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                   reg_iterations=iters, affine_iterations=[0]*len(iters),
                   similarity_metric='ncc', verbose=False, grad_step=0.25)
    w = ants.apply_transforms(fi, mi_aff, r['fwdtransforms'])
    mi_s = ants.image_mutual_information(fi, w)
    print(f"iters={str(iters):<12} syntx(ncc)={mi_s:.4f}  ran={len(r['model'].syn_losses)}")
