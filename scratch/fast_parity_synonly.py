"""Fast SyN-only parity: bypass affine, quick settings."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])
mi0 = ants.image_mutual_information(fi, mi_aff)
print(f"Pre-aligned MI: {mi0:.4f}")
print()

settings = [
    ([5, 0, 0], 0.25),
    ([10, 0, 0], 0.25),
    ([10, 5, 0], 0.25),
    ([20, 10, 0], 0.25),
    ([50, 20, 0], 0.25),
]

print(f"{'Iters':<18} {'syntx MI':>10} {'ANTs MI':>10} {'Gap':>8} {'syntx_t':>8} {'ants_t':>8}")
print("="*72)

for iters, gs in settings:
    t0 = time.time()
    rs = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                    reg_iterations=iters, affine_iterations=[0]*len(iters),
                    similarity_metric='mattes_mi', verbose=False, grad_step=gs)
    ts = time.time() - t0
    ws = ants.apply_transforms(fi, mi_aff, rs['fwdtransforms'])
    mis = ants.image_mutual_information(fi, ws)
    
    t0 = time.time()
    ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=iters, syn_metric='mattes')
    ta = time.time() - t0
    mia = ants.image_mutual_information(fi, ra['warpedmovout'])
    
    print(f"{str(iters):<18} {mis:>10.4f} {mia:>10.4f} {mia-mis:>8.4f} {ts:>7.1f}s {ta:>7.1f}s")
