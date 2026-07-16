"""Quick 3D sanity check to confirm fixes don't break 3D."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('mni'))
mi = ants.image_read(ants.get_ants_data('ch2'))
fi = ants.resample_image(fi, (4,4,4), use_voxels=False)
mi = ants.resample_image(mi, (4,4,4), use_voxels=False)
mi0 = ants.image_mutual_information(fi, mi)
print(f"Unregistered MI: {mi0:.4f}")

# Quick 3D: affine + SyN
t0 = time.time()
rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[5, 0, 0], affine_iterations=[50, 50, 0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
ts = time.time() - t0
wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
mis = ants.image_mutual_information(fi, wf)
wi = ants.apply_transforms(mi, fi, rs['invtransforms'], whichtoinvert=rs['whichtoinvert_inv'])
mis_inv = ants.image_mutual_information(mi, wi)
print(f"syntx 3D [5,0,0]: fwd MI={mis:.4f}  inv MI={mis_inv:.4f}  ({ts:.1f}s)")

# ANTs reference
t0 = time.time()
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[5, 0, 0], syn_metric='mattes')
ta = time.time() - t0
mia = ants.image_mutual_information(fi, ra['warpedmovout'])
wa = ants.apply_transforms(mi, fi, ra['invtransforms'])
mia_inv = ants.image_mutual_information(mi, wa)
print(f"ANTs  3D [5,0,0]: fwd MI={mia:.4f}  inv MI={mia_inv:.4f}  ({ta:.1f}s)")
print(f"Gap: {mia - mis:.4f}")

# Verify rapid similarity improvement
losses = rs.get('syn_losses', [])
if losses:
    print(f"\nSyN loss trajectory: [{losses[0]:.4f}→{losses[-1]:.4f}]")
    print(f"Improvement: {losses[0] - losses[-1]:.4f}")
