"""Quick 2D test: grad_step=0.25, pytorch only, with timing."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# ANTs reference (SyN full pipeline)
t0 = time.time()
reg_ants = ants.registration(fi, mi, 'SyN', reg_iterations=[100, 100, 100, 50], syn_metric='mattes')
t_ants = time.time() - t0
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])

# syntx with grad_step=0.25, mattes_mi 
t0 = time.time()
reg_025_mi = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                        reg_iterations=[100, 100, 100, 50],
                        similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
t_025_mi = time.time() - t0
w_025_mi = ants.apply_transforms(fi, mi, reg_025_mi['fwdtransforms'])
mi_025_mi = ants.image_mutual_information(fi, w_025_mi)

# syntx with grad_step=0.25, lncc
t0 = time.time()
reg_025_lncc = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                          reg_iterations=[100, 100, 100, 50],
                          similarity_metric='lncc', verbose=False, grad_step=0.25)
t_025_lncc = time.time() - t0
w_025_lncc = ants.apply_transforms(fi, mi, reg_025_lncc['fwdtransforms'])
mi_025_lncc = ants.image_mutual_information(fi, w_025_lncc)

# syntx with grad_step=0.75 (old default), mattes_mi
t0 = time.time()
reg_075 = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                     reg_iterations=[100, 100, 100, 50],
                     similarity_metric='mattes_mi', verbose=False, grad_step=0.75)
t_075 = time.time() - t0
w_075 = ants.apply_transforms(fi, mi, reg_075['fwdtransforms'])
mi_075 = ants.image_mutual_information(fi, w_075)

print("="*65)
print(f"{'Method':<35} {'MI':>8} {'Time':>7} {'Iters':>6}")
print("="*65)
print(f"{'ANTs SyN (mattes)':<35} {mi_ants:>8.4f} {t_ants:>6.1f}s {'—':>6}")
print(f"{'syntx gs=0.25 mattes_mi':<35} {mi_025_mi:>8.4f} {t_025_mi:>6.1f}s {len(reg_025_mi['model'].syn_losses):>6}")
print(f"{'syntx gs=0.25 lncc':<35} {mi_025_lncc:>8.4f} {t_025_lncc:>6.1f}s {len(reg_025_lncc['model'].syn_losses):>6}")
print(f"{'syntx gs=0.75 mattes_mi':<35} {mi_075:>8.4f} {t_075:>6.1f}s {len(reg_075['model'].syn_losses):>6}")
print("="*65)
