"""Test each inverse piece separately."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants, numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

print("inv_transforms:", rs['invtransforms'])
aff_inv = rs['invtransforms'][0]  # .mat
warp_inv = rs['invtransforms'][1]  # .nii.gz

# Test 1: Affine inverse only (no deformable)
wi_aff = ants.apply_transforms(mi, fi, [aff_inv])
print(f"1. Affine inv only: MI={ants.image_mutual_information(mi, wi_aff):.4f}")

# Test 2: Deformable inverse only (no affine)
wi_warp = ants.apply_transforms(mi, fi, [warp_inv])
print(f"2. Warp inv only: MI={ants.image_mutual_information(mi, wi_warp):.4f}")

# Test 3: Both combined (current order: [aff_inv, warp_inv])
wi_both = ants.apply_transforms(mi, fi, [aff_inv, warp_inv])
print(f"3. [aff_inv, warp_inv]: MI={ants.image_mutual_information(mi, wi_both):.4f}")

# Test 4: Reversed order
wi_rev = ants.apply_transforms(mi, fi, [warp_inv, aff_inv])
print(f"4. [warp_inv, aff_inv]: MI={ants.image_mutual_information(mi, wi_rev):.4f}")

# For comparison: ANTs forward
wf = ants.apply_transforms(fi, mi, rs['fwdtransforms'])
print(f"\nForward MI: {ants.image_mutual_information(fi, wf):.4f}")

# ANTs reference inverse
ra = ants.registration(fi, mi, 'SyN', reg_iterations=[10,5,0], syn_metric='mattes')
wi_a = ants.apply_transforms(mi, fi, ra['invtransforms'])
print(f"ANTs inverse MI: {ants.image_mutual_information(mi, wi_a):.4f}")
print(f"ANTs invtransforms: {ra['invtransforms']}")
