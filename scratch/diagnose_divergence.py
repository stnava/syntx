"""Check loss trajectory and warp growth to diagnose divergence."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

print("=== grad_step=0.75 (default, oscillates) ===")
reg_75 = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.75)
m = reg_75['model']
print(f"syn_losses: {[f'{x:.4f}' for x in m.syn_losses]}")
print(f"warp_l2r max: {m.warp_l2r.data.abs().max():.2f}mm")
warped = ants.apply_transforms(fi, mi_affine, reg_75['fwdtransforms'])
print(f"MI: {ants.image_mutual_information(fi, warped):.4f}")

print("\n=== grad_step=0.2 (ANTs default) ===")
reg_20 = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.2)
m = reg_20['model']
print(f"syn_losses: {[f'{x:.4f}' for x in m.syn_losses]}")
print(f"warp_l2r max: {m.warp_l2r.data.abs().max():.2f}mm")
warped = ants.apply_transforms(fi, mi_affine, reg_20['fwdtransforms'])
print(f"MI: {ants.image_mutual_information(fi, warped):.4f}")

print("\n=== grad_step=0.1 (stable) ===")
reg_10 = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.1)
m = reg_10['model']
print(f"syn_losses: {[f'{x:.4f}' for x in m.syn_losses]}")
print(f"warp_l2r max: {m.warp_l2r.data.abs().max():.2f}mm")
warped = ants.apply_transforms(fi, mi_affine, reg_10['fwdtransforms'])
print(f"MI: {ants.image_mutual_information(fi, warped):.4f}")

print("\n=== grad_step=0.1, 200 iterations ===")
reg_200 = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[200, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.1)
m = reg_200['model']
losses = m.syn_losses
print(f"syn_losses (first 10): {[f'{x:.4f}' for x in losses[:10]]}")
print(f"syn_losses (last 10):  {[f'{x:.4f}' for x in losses[-10:]]}")
print(f"Total iterations run: {len(losses)}")
print(f"warp_l2r max: {m.warp_l2r.data.abs().max():.2f}mm")
warped = ants.apply_transforms(fi, mi_affine, reg_200['fwdtransforms'])
print(f"MI: {ants.image_mutual_information(fi, warped):.4f}")
