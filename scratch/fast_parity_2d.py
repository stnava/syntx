"""Fast 2D parity: quick settings, pytorch only."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
mi0 = ants.image_mutual_information(fi, mi)
print(f"Unregistered MI: {mi0:.4f}")

# ANTs reference - quick
t0 = time.time()
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[50, 20, 0, 0], syn_metric='mattes')
ta = time.time() - t0
mia = ants.image_mutual_information(fi, ra['warpedmovout'])

# syntx - quick, same iterations
t0 = time.time()
rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[50, 20, 0, 0], affine_iterations=[100, 100, 0, 0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
ts = time.time() - t0
ws = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
mis = ants.image_mutual_information(fi, ws)

print(f"\n{'='*60}")
print(f"{'Method':<35} {'MI':>8} {'Time':>7}")
print(f"{'='*60}")
print(f"{'ANTs SyN [50,20,0,0]':<35} {mia:>8.4f} {ta:>6.1f}s")
print(f"{'syntx SyNTo [50,20,0,0]':<35} {mis:>8.4f} {ts:>6.1f}s")
print(f"{'='*60}")
print(f"Gap: {mia - mis:.4f}")

# Check inverse quality
wi = ants.apply_transforms(mi, fi, rs['invtransforms'], whichtoinvert=rs.get('whichtoinvert_inv'))
mi_inv = ants.image_mutual_information(mi, wi)
print(f"\nInverse warp MI: {mi_inv:.4f}")

# Check ANTs inverse
wi_ants = ants.apply_transforms(mi, fi, ra['invtransforms'], whichtoinvert=ra.get('whichtoinvert_inv'))
mi_inv_ants = ants.image_mutual_information(mi, wi_ants)
print(f"ANTs inverse MI: {mi_inv_ants:.4f}")
