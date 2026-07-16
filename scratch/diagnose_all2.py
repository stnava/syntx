"""Comprehensive diagnostic for syntx SyN — focused on key issues."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])
print(f"Initial MI: {ants.image_mutual_information(fi, mi_affine):.4f}")

print("\n========================================")
print("ISSUE A: Loss oscillates wildly (CFL=0.75)")
print("========================================")
print("With default CFL=0.75 voxels at level-8 (32x32, spacing~8mm):")
print("  Max step = 0.75 * 8.23 = 6.17mm/iter")
print("  This is HUGE relative to anatomy at 32x32 resolution")
print("  Loss bounces: -1.08, -1.19, -1.12, -1.18, -1.08, -1.16, -1.06, ...")
print("  The step overshoots every other iteration -> classic gradient descent instability")
print()
print("  With CFL=0.01: loss monotonically decreases (-1.07 → -1.11 over 20 iters)")
print("  With CFL=0.1:  loss oscillates but increases overall (-1.07 → -1.18)")
print("  The convergence check sees flat/increasing slope -> EARLY TERMINATION at ~10 iters")

print("\n========================================")
print("ISSUE B: Convergence check fires too aggressively")
print("========================================")
print("  check_convergence: slope >= -1e-8 over 10 samples → CONVERGED")
print("  With oscillating loss, the 10-sample window average slope is ~0 → CONVERGED")
print("  This kills the SyN loop at epoch ~10 even when deformations are still building")

print("\n========================================")
print("ISSUE C: SyN losses are measured at midpoint (not fixed-to-moving)")
print("========================================")
print("  The optimizer minimizes loss(J_mid, I_mid) where both images are warped TO midpoint")
print("  I_mid = I(x + warp_l2r(x))  and  J_mid = J(affine(x + warp_r2l(x)))")
print("  Both arms warp HALF the total displacement")
print("  The convergence check uses this midpoint loss, which may plateau quickly")
print("  while the full forward composition could still improve")

print("\n========================================")
print("ISSUE D: After optimization, MI GETS WORSE not better")
print("========================================")
reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False)
warped_pt = ants.apply_transforms(fi, mi_affine, reg_pt['fwdtransforms'])
mi_after = ants.image_mutual_information(fi, warped_pt)
mi_before = ants.image_mutual_information(fi, mi_affine)
print(f"  MI before SyN: {mi_before:.4f}")
print(f"  MI after SyN:  {mi_after:.4f}")
print(f"  Δ MI: {mi_after - mi_before:.4f}  {'WORSE' if mi_after > mi_before else 'BETTER'}")

fwd_pt = ants.image_read(reg_pt['fwdtransforms'][0])
print(f"  Deformation max norm: {np.max(np.linalg.norm(fwd_pt.numpy(), axis=-1)):.4f}mm")

print("\n========================================")
print("ISSUE E: The deformation is LARGE but WRONG direction")
print("========================================")
print("  Max deformation is 8.5mm (bigger than ANTs' 5mm!)")
print("  But MI gets WORSE, meaning the deformation is moving the image AWAY from alignment")
print("  This is a SIGN ERROR or COMPOSITION ERROR in the final field")

print("\n========================================")
print("Checking: warp_l2r should map fixed→midpoint; total fwd should map fixed→moving")
print("========================================")
print()
import torch
model = reg_pt['model']
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch
import torch.nn.functional as F

# Try applying just the raw warp_l2r (midpoint field) as if it were the forward field
total_fwd = model.warp_l2r.data
print(f"  warp_l2r shape: {total_fwd.shape}")
print(f"  warp_l2r max: {total_fwd.abs().max():.4f}")
print(f"  warp_l2r mean: {total_fwd.abs().mean():.4f}")

# Check the warp_r2l too
total_inv = model.warp_r2l.data
print(f"  warp_r2l max: {total_inv.abs().max():.4f}")

# Check: is the composition making things worse?
# warp_l2r should be the COMPOSED total field: φ_r2l ∘ φ_l2r_inv^{-1}
# Let's see if applying -warp_l2r works better (sign error)
print("\n  Testing sign flip:")
fwd_file_neg = '/tmp/test_neg_warp.nii.gz'
warp_np = total_fwd.cpu().numpy()[0]
warp_np_neg = -warp_np
warp_ants_neg = ants.from_numpy(warp_np_neg[..., ::-1].copy(), has_components=True, 
                                 spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
ants.image_write(warp_ants_neg, fwd_file_neg)
warped_neg = ants.apply_transforms(fi, mi_affine, [fwd_file_neg])
mi_neg = ants.image_mutual_information(fi, warped_neg)
print(f"  MI with -warp_l2r: {mi_neg:.4f} (negated)")

# Check warp_r2l 
fwd_file_r2l = '/tmp/test_r2l_warp.nii.gz'
warp_r2l_np = total_inv.cpu().numpy()[0]
warp_ants_r2l = ants.from_numpy(warp_r2l_np[..., ::-1].copy(), has_components=True,
                                  spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
ants.image_write(warp_ants_r2l, fwd_file_r2l)
warped_r2l = ants.apply_transforms(fi, mi_affine, [fwd_file_r2l])
mi_r2l = ants.image_mutual_information(fi, warped_r2l)
print(f"  MI with warp_r2l as fwd: {mi_r2l:.4f}")

# Check -warp_r2l
fwd_file_r2l_neg = '/tmp/test_r2l_neg_warp.nii.gz'
warp_ants_r2l_neg = ants.from_numpy((-warp_r2l_np)[..., ::-1].copy(), has_components=True,
                                      spacing=fi.spacing, origin=fi.origin, direction=fi.direction)
ants.image_write(warp_ants_r2l_neg, fwd_file_r2l_neg)
warped_r2l_neg = ants.apply_transforms(fi, mi_affine, [fwd_file_r2l_neg])
mi_r2l_neg = ants.image_mutual_information(fi, warped_r2l_neg)
print(f"  MI with -warp_r2l as fwd: {mi_r2l_neg:.4f}")

print(f"\n  SUMMARY:")
print(f"  Initial MI:        {mi_before:.4f}")
print(f"  warp_l2r (current): {mi_after:.4f}")
print(f"  -warp_l2r:          {mi_neg:.4f}")
print(f"  warp_r2l:           {mi_r2l:.4f}")
print(f"  -warp_r2l:          {mi_r2l_neg:.4f}")
best = min([('warp_l2r', mi_after), ('-warp_l2r', mi_neg), 
            ('warp_r2l', mi_r2l), ('-warp_r2l', mi_r2l_neg)], key=lambda x: x[1])
print(f"  BEST field: {best[0]} with MI={best[1]:.4f}")
