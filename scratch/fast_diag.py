"""Fast diagnosis of multi-level issues."""
import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx
import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# 1. Compare affine quality
print("=== AFFINE COMPARISON ===")
reg_ants_aff = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 0])
mi_ants_aff = ants.image_mutual_information(fi, reg_ants_aff['warpedmovout'])

reg_syntx_aff = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                           reg_iterations=[0, 0, 0], affine_iterations=[100, 100, 0],
                           similarity_metric='mattes_mi', verbose=False)
warped_syntx = ants.apply_transforms(fi, mi, reg_syntx_aff['fwdtransforms'])
mi_syntx_aff = ants.image_mutual_information(fi, warped_syntx)
print(f"ANTs Affine MI: {mi_ants_aff:.4f}")
print(f"syntx Affine MI: {mi_syntx_aff:.4f}")

# 2. SyN-only with ANTs affine pre-alignment (bypass syntx affine)
mi_prealigned = ants.apply_transforms(fi, mi, reg_ants_aff['fwdtransforms'])
mi_pre = ants.image_mutual_information(fi, mi_prealigned)
print(f"\nPre-aligned MI: {mi_pre:.4f}")

# 3. Quick SyN-only with 5 iters at coarse level 
for gs in [0.1, 0.25, 0.5]:
    reg = syntx.syn(fi, mi_prealigned, 'SyNTo', backend='pytorch',
                     reg_iterations=[5, 0, 0], affine_iterations=[0, 0, 0],
                     similarity_metric='mattes_mi', verbose=False, grad_step=gs)
    warped = ants.apply_transforms(fi, mi_prealigned, reg['fwdtransforms'])
    mi_val = ants.image_mutual_information(fi, warped)
    losses = reg['model'].syn_losses
    print(f"  gs={gs}: MI={mi_val:.4f} losses=[{losses[0]:.4f}...{losses[-1]:.4f}] max_warp={reg['model'].warp_l2r.data.abs().max():.2f}mm")

# 4. ANTs SyN-only for comparison
reg_ants_syn = ants.registration(fi, mi_prealigned, 'SyNOnly', reg_iterations=[5, 0, 0], syn_metric='mattes')
mi_ants_syn = ants.image_mutual_information(fi, reg_ants_syn['warpedmovout'])
print(f"  ANTs SyNOnly 5 iters: MI={mi_ants_syn:.4f}")

# 5. Multi-level SyN-only
print("\n=== MULTI-LEVEL SYN-ONLY ===")
for iters in [[5,0,0], [20,0,0], [20,10,0], [50,20,0]]:
    reg = syntx.syn(fi, mi_prealigned, 'SyNTo', backend='pytorch',
                     reg_iterations=iters, affine_iterations=[0]*len(iters),
                     similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
    warped = ants.apply_transforms(fi, mi_prealigned, reg['fwdtransforms'])
    mi_val = ants.image_mutual_information(fi, warped)
    n = len(reg['model'].syn_losses)
    print(f"  iters={iters}: MI={mi_val:.4f} ran={n} max_warp={reg['model'].warp_l2r.data.abs().max():.2f}mm")

# ANTs multi-level reference
for iters in [[5,0,0], [20,0,0], [20,10,0], [50,20,0]]:
    reg = ants.registration(fi, mi_prealigned, 'SyNOnly', reg_iterations=iters, syn_metric='mattes')
    mi_val = ants.image_mutual_information(fi, reg['warpedmovout'])
    print(f"  ANTs iters={iters}: MI={mi_val:.4f}")
