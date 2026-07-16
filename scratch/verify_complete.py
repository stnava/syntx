"""Complete verification after all fixes."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_aff = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx_aff['fwdtransforms'])
mi_before = ants.image_mutual_information(fi, mi_aff)

# Test 1: Default settings
reg_pt = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                    reg_iterations=[50, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.2)
warped = ants.apply_transforms(fi, mi_aff, reg_pt['fwdtransforms'])
mi_pt = ants.image_mutual_information(fi, warped)

# ANTs reference
reg_ants = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[50, 0, 0], syn_metric='mattes')
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])

print("=== 2D SyN Parity Test (r16 vs r64, affine pre-aligned) ===")
print(f"Initial MI:     {mi_before:.4f}")
print(f"syntx SyN MI:   {mi_pt:.4f}  (ran {len(reg_pt['model'].syn_losses)} iters)")
print(f"ANTs SyNOnly MI: {mi_ants:.4f}")
print(f"syntx improvement: {mi_before - mi_pt:.4f}")
print(f"ANTs improvement:  {mi_before - mi_ants:.4f}")

# Test 2: More iterations
reg_pt2 = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                     reg_iterations=[100, 50, 0, 0], affine_iterations=[0, 0, 0, 0],
                     similarity_metric='mattes_mi', verbose=False, grad_step=0.2)
warped2 = ants.apply_transforms(fi, mi_aff, reg_pt2['fwdtransforms'])
mi_pt2 = ants.image_mutual_information(fi, warped2)
print(f"\nsyntx (100+50 iters) MI: {mi_pt2:.4f}  (ran {len(reg_pt2['model'].syn_losses)} iters)")

# Test 3: LNCC metric 
reg_pt_lncc = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                          reg_iterations=[50, 0, 0], affine_iterations=[0, 0, 0],
                          similarity_metric='lncc', verbose=False, grad_step=0.2)
warped_lncc = ants.apply_transforms(fi, mi_aff, reg_pt_lncc['fwdtransforms'])
mi_pt_lncc = ants.image_mutual_information(fi, warped_lncc)
print(f"\nsyntx (LNCC) MI: {mi_pt_lncc:.4f}  (ran {len(reg_pt_lncc['model'].syn_losses)} iters)")
