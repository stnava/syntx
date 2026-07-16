"""FAST 2D diagnostic: minimal iterations to see if deformable helps at each level."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# ANTs quick reference
t0 = time.time()
reg_ants = ants.registration(fi, mi, 'SyN', reg_iterations=[50, 20, 0, 0], syn_metric='mattes')
t_ants = time.time() - t0
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])

# syntx quick: just 5 deformable iters to see direction
t0 = time.time()
reg5 = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                  reg_iterations=[5, 0, 0], affine_iterations=[100, 100, 0],
                  similarity_metric='mattes_mi', verbose=True, grad_step=0.25)
t5 = time.time() - t0
w5 = ants.apply_transforms(fi, mi, reg5['fwdtransforms'])
mi5 = ants.image_mutual_information(fi, w5)

# syntx: 50+20 deformable iters (matching ANTs)
t0 = time.time()
reg_50_20 = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                       reg_iterations=[50, 20, 0, 0], affine_iterations=[100, 100, 0, 0],
                       similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
t_50_20 = time.time() - t0
w_50_20 = ants.apply_transforms(fi, mi, reg_50_20['fwdtransforms'])
mi_50_20 = ants.image_mutual_information(fi, w_50_20)

print("\n" + "="*70)
print(f"{'Method':<45} {'MI':>8} {'Time':>7} {'Iters':>6}")
print("="*70)
print(f"{'ANTs SyN [50,20,0,0]':<45} {mi_ants:>8.4f} {t_ants:>6.1f}s {'—':>6}")
print(f"{'syntx gs=0.25 [5,0,0]':<45} {mi5:>8.4f} {t5:>6.1f}s {len(reg5['model'].syn_losses):>6}")
print(f"{'syntx gs=0.25 [50,20,0,0]':<45} {mi_50_20:>8.4f} {t_50_20:>6.1f}s {len(reg_50_20['model'].syn_losses):>6}")
print("="*70)

# Show loss trajectory
losses = reg_50_20['model'].syn_losses
print(f"\nLoss trajectory (first 10): {[f'{l:.4f}' for l in losses[:10]]}")
print(f"Loss trajectory (last 10):  {[f'{l:.4f}' for l in losses[-10:]]}")
