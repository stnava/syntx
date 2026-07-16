"""Test if warp_r2l needs negation or channel flip."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, tempfile
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

# Get the inverse warp and affine inv
inv_warp_file = [f for f in rs['invtransforms'] if f.endswith('.nii.gz')][0]
inv_aff_file = [f for f in rs['invtransforms'] if f.endswith('.mat')][0]
w = ants.image_read(inv_warp_file)
w_np = w.numpy()

# Test variants
variants = {
    'original': w_np.copy(),
    'negated': -w_np.copy(),
    'channels_reversed': w_np[..., ::-1].copy(),
    'negated+reversed': (-w_np)[..., ::-1].copy(),
}

for name, arr in variants.items():
    f = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    img = ants.from_numpy(arr, origin=w.origin, spacing=w.spacing, direction=w.direction, has_components=True)
    ants.image_write(img, f)
    wi = ants.apply_transforms(mi, fi, [f, inv_aff_file])
    mi_val = ants.image_mutual_information(mi, wi)
    print(f"{name:20s}: MI={mi_val:.4f}")

# Also try with the fwd warp as inverse (should work since warp_l2r ≈ -warp_r2l for small displacements)
fwd_warp_file = [f for f in rs['fwdtransforms'] if f.endswith('.nii.gz')][0]
wi_fwd = ants.apply_transforms(mi, fi, [fwd_warp_file, inv_aff_file])
print(f"\nUsing fwd warp as inv: MI={ants.image_mutual_information(mi, wi_fwd):.4f}")
wi_fwd_neg = ants.apply_transforms(mi, fi, [fwd_warp_file, inv_aff_file], whichtoinvert=[True, False])
print(f"Fwd warp inverted: MI={ants.image_mutual_information(mi, wi_fwd_neg):.4f}")
