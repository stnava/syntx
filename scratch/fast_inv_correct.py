"""Correct inverse: use whichtoinvert for the affine."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

inv_warp = [f for f in rs['invtransforms'] if f.endswith('.nii.gz')][0]
inv_aff = [f for f in rs['invtransforms'] if f.endswith('.mat')][0]
fwd_warp = [f for f in rs['fwdtransforms'] if f.endswith('.nii.gz')][0]
fwd_aff = [f for f in rs['fwdtransforms'] if f.endswith('.mat')][0]

# Forward check
wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
print(f"Forward: MI={ants.image_mutual_information(fi, wf):.4f}")

# Try all inverse combinations
combos = [
    ("inv_warp + inv_aff",       [inv_warp, inv_aff], [False, False]),
    ("inv_aff + inv_warp",       [inv_aff, inv_warp], [False, False]),
    ("inv_warp + fwd_aff(inv)",  [inv_warp, fwd_aff], [False, True]),
    ("fwd_aff(inv) + inv_warp",  [fwd_aff, inv_warp], [True, False]),
    ("inv_warp + fwd_aff",       [inv_warp, fwd_aff], [False, False]),
    ("fwd_aff + inv_warp",       [fwd_aff, inv_warp], [False, False]),
]

for name, tfs, wit in combos:
    try:
        wi = ants.apply_transforms(mi, fi, tfs, whichtoinvert=wit)
        mi_val = ants.image_mutual_information(mi, wi)
        print(f"{name:35s}: MI={mi_val:.4f}")
    except Exception as e:
        print(f"{name:35s}: ERROR: {e}")

# ANTs reference
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
print(f"\nANTs forward: MI={ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
wi_a = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"ANTs inverse: MI={ants.image_mutual_information(mi, wi_a):.4f}")
print(f"ANTs invtransforms: {ra['invtransforms']}")

# Check how ANTs structures its inverse
fwd_is_mat = [f.endswith('.mat') for f in ra['fwdtransforms']]
inv_is_mat = [f.endswith('.mat') for f in ra['invtransforms']]
print(f"ANTs fwd types: {['mat' if m else 'warp' for m in fwd_is_mat]}")
print(f"ANTs inv types: {['mat' if m else 'warp' for m in inv_is_mat]}")
