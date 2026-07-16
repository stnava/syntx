"""Final 2D summary: SyN-only and full pipeline, gs=0.25, speed comparison."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Pre-align with ANTs affine
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

print(f"Unregistered MI: {ants.image_mutual_information(fi, mi):.4f}")
print(f"Pre-aligned MI:  {ants.image_mutual_information(fi, mi_aff):.4f}")

# === SYN-ONLY parity (using ANTs affine) ===
print(f"\n{'='*70}")
print(f"SYN-ONLY PARITY (pre-aligned with ANTs Affine)")
print(f"{'Iters':<16} {'syntx':>8} {'ANTs':>8} {'gap':>7} {'s_time':>7} {'a_time':>7}")
print(f"{'='*70}")

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
    
    print(f"{str(iters):<16} {mis:>8.4f} {mia:>8.4f} {mia-mis:>7.4f} {ts:>6.1f}s {ta:>6.1f}s")

# === FULL PIPELINE parity (syntx affine + SyN vs ANTs SyN) ===
print(f"\n{'='*70}")
print(f"FULL PIPELINE (syntx does its own affine + SyN)")
print(f"{'='*70}")

# With adjusted affine unlocking - test if unlocking all at level 1 helps
t0=time.time()
rs_full = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                     reg_iterations=[10, 5, 0, 0], affine_iterations=[100, 100, 0, 0],
                     similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
ts_full=time.time()-t0
ws_full = ants.apply_transforms(fi, mi, rs_full['fwdtransforms'])
mis_full = ants.image_mutual_information(fi, ws_full)

t0=time.time()
ra_full = ants.registration(fi, mi, 'SyN', reg_iterations=[10, 5, 0, 0], syn_metric='mattes')
ta_full=time.time()-t0
mia_full = ants.image_mutual_information(fi, ra_full['warpedmovout'])

print(f"syntx SyNTo: MI={mis_full:.4f} ({ts_full:.1f}s)")
print(f"ANTs SyN:    MI={mia_full:.4f} ({ta_full:.1f}s)")
print(f"Gap: {mia_full-mis_full:.4f}")
print(f"Speed ratio: {ts_full/ta_full:.1f}x slower")
