"""Debug: SyN-only inverse with pre-aligned images."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants, numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx['fwdtransforms'])

# SyN-only on pre-aligned
rs = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[0,0,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

# Check forward
wf = ants.apply_transforms(fi, mi_aff, rs['fwdtransforms'])
print(f"Forward MI: {ants.image_mutual_information(fi, wf):.4f}")

# Check inverse
wi = ants.apply_transforms(mi_aff, fi, rs['invtransforms'])
print(f"Inverse MI: {ants.image_mutual_information(mi_aff, wi):.4f}")

# Check what files we have
print(f"\nfwd: {rs['fwdtransforms']}")
print(f"inv: {rs['invtransforms']}")

# Check inv warp
for f in rs['invtransforms']:
    if f.endswith('.nii.gz'):
        w = ants.image_read(f)
        print(f"Inv warp shape: {w.numpy().shape} max: {np.abs(w.numpy()).max():.3f}")

# Also check model internals
m = rs['model']
print(f"\nwarp_l2r max: {m.warp_l2r.data.abs().max():.3f}")
print(f"warp_r2l max: {m.warp_r2l.data.abs().max():.3f}")
print(f"warp_l2r_inv max: {m.warp_l2r_inv.data.abs().max():.3f}")
print(f"warp_r2l_inv max: {m.warp_r2l_inv.data.abs().max():.3f}")

# ANTs reference
ra = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[10,5,0], syn_metric='mattes')
wi_a = ants.apply_transforms(mi_aff, fi, ra['invtransforms'])
print(f"\nANTs forward MI: {ants.image_mutual_information(fi, ra['warpedmovout']):.4f}")
print(f"ANTs inverse MI: {ants.image_mutual_information(mi_aff, wi_a):.4f}")
