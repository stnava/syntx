"""Check ANTs displacement field convention vs syntx internal convention."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Get an ANTs SyN result to examine its displacement field convention
reg_ants = ants.registration(fi, mi, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')
fwd_ants = ants.image_read(reg_ants['fwdtransforms'][0])

print("=== ANTs displacement field convention ===")
print(f"  fwd field shape: {fwd_ants.numpy().shape}")
print(f"  fwd field spacing: {fwd_ants.spacing}")
print(f"  fwd field origin: {fwd_ants.origin}")
print(f"  fwd field direction: {fwd_ants.direction}")
print(f"  fwd field has_components: {fwd_ants.has_components}")
print(f"  fwd field components: {fwd_ants.components}")
print(f"  fwd field pixel type: {fwd_ants.pixeltype}")

# In ANTs, forward displacement maps fixed->moving.
# It's stored as a vector image with components in X,Y order.
# The displacement at pixel (i,j) tells where that fixed-space point maps to in moving space.
# When used with apply_transforms, ANTs interprets this as:
#   warped(x) = moving(x + disp(x))  
# So the displacement maps each fixed-space point to its corresponding moving-space location.

# Check: what does the ANTs fwd displacement look like at a known offset?
print(f"\n  fwd field value ranges per component:")
d = fwd_ants.numpy()
for c in range(d.shape[-1]):
    print(f"    Component {c}: [{d[...,c].min():.3f}, {d[...,c].max():.3f}]")

# Now check: does ANTs use negative or positive displacements?
# ANTs convention: displacement = target_position - source_position
# apply_transforms: warped_moving(x) = moving(x + displacement(x))
print(f"\n  ANTs MI with forward warp: {ants.image_mutual_information(fi, reg_ants['warpedmovout']):.4f}")

# Also check inverse
inv_ants = ants.image_read(reg_ants['invtransforms'][0])
d_inv = inv_ants.numpy()
print(f"\n  inv field value ranges per component:")
for c in range(d_inv.shape[-1]):
    print(f"    Component {c}: [{d_inv[...,c].min():.3f}, {d_inv[...,c].max():.3f}]")

print(f"\n=== Key question: sign convention ===")
print(f"  In syntx, warp_l2r is stored in YX order (PyTorch ZYX internal convention)")
print(f"  Export does warp[..., ::-1] to convert YX -> XY for ANTs")
print(f"  But does ANTs expect POSITIVE or NEGATIVE displacements?")
print(f"  ANTs forward disp: warped(x) = moving(x + d(x))")
print(f"  So d(x) = moving_position - fixed_position")
print(f"  This means the displacement should be POSITIVE when moving is to the right of fixed")
