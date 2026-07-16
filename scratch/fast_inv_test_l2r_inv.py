"""Test using warp_l2r_inv as the exported inverse warp."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, tempfile, torch
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

m = rs['model']
inv_aff_file = [f for f in rs['invtransforms'] if f.endswith('.mat')][0]

# Use warp_l2r_inv (computed inverse of forward warp)
warp_l2r_inv_np = m.warp_l2r_inv.data[0].numpy()
f = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
img = ants.from_numpy(warp_l2r_inv_np, origin=fi.origin, spacing=fi.spacing,
                       direction=fi.direction, has_components=True)
ants.image_write(img, f)
wi = ants.apply_transforms(mi, fi, [f, inv_aff_file])
print(f"warp_l2r_inv as inverse: MI={ants.image_mutual_information(mi, wi):.4f}")

# Original inverse
wi_orig = ants.apply_transforms(mi, fi, rs['invtransforms'])
print(f"Original inverse: MI={ants.image_mutual_information(mi, wi_orig):.4f}")

# Forward for reference
wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
print(f"Forward: MI={ants.image_mutual_information(fi, wf):.4f}")

# ANTs reference
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
print(f"\nANTs forward: MI={ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
wi_a = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"ANTs inverse: MI={ants.image_mutual_information(mi, wi_a):.4f}")
