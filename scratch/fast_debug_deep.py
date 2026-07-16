"""Deep debug: verify affine+warp composition."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

fwd_aff = [f for f in rs['fwdtransforms'] if f.endswith('.mat')][0]
fwd_warp = [f for f in rs['fwdtransforms'] if f.endswith('.nii.gz')][0]
inv_aff = [f for f in rs['invtransforms'] if f.endswith('.mat')][0]
inv_warp = [f for f in rs['invtransforms'] if f.endswith('.nii.gz')][0]

# Forward step-by-step:
# 1. Apply affine only
w1 = ants.apply_transforms(fi, mi, [fwd_aff])
mi1 = ants.image_mutual_information(fi, w1)
# 2. Apply warp + affine (full forward)
w2 = ants.apply_transforms(fi, mi, [fwd_warp, fwd_aff])
mi2 = ants.image_mutual_information(fi, w2)
print(f"Forward: affine-only={mi1:.4f}, warp+affine={mi2:.4f}")

# Inverse step-by-step:
# 1. Apply inv_aff only (should map mi→fi space)
w3 = ants.apply_transforms(mi, fi, [inv_aff])
mi3 = ants.image_mutual_information(mi, w3)
# 2. Apply inv_warp only
w4 = ants.apply_transforms(mi, fi, [inv_warp])
mi4 = ants.image_mutual_information(mi, w4)
# 3. Combined: [inv_aff, inv_warp] → warp first, then affine
w5 = ants.apply_transforms(mi, fi, [inv_aff, inv_warp])
mi5 = ants.image_mutual_information(mi, w5)
print(f"Inverse: inv_aff-only={mi3:.4f}, inv_warp-only={mi4:.4f}, combined={mi5:.4f}")

# The issue: what does the inverse warp map?
# warp_r2l was computed to map from fixed midpoint space → moving
# When exported, its displacement vectors are in YX physical coordinates
# ANTs expects displacement vectors to point FROM the reference space TO where we sample

# Let's check: the fwd warp maps in fixed space (warp_l2r)
# For apply_transforms(fi, mi, [fwd_warp, fwd_aff]):
# Step 1: affine maps fi_point → mi_point (in mi space)
# Step 2: warp adds displacement at fi_point to get final sampling location
# Wait, that's not right for displacement fields...

# In ANTs, a displacement field D applied at reference point x gives:
# sampling_point = x + D(x)
# For composite transforms: [warp, affine] applied to reference fi:
# For each fi point x: 
#   y = affine(x) = M @ x + t  (gets mi space point)
#   z = y + D(y)?  NO — the displacement field is in the reference frame!
#   Actually: z = x + D(x)?

# Let me check ANTs documentation...
# ANTs composites are applied RIGHT to LEFT
# For [warp, affine]: first affine(x) = M@x+t, then warp at that location

# But displacement field: z = y + D(y) where y = affine(x)
# No — displacement field in ANTs adds displacement at the mapped point

# Let me just test empirically
print(f"\nfwd_warp origin: {ants.image_read(fwd_warp).origin}")
print(f"inv_warp origin: {ants.image_read(inv_warp).origin}")
print(f"fi origin: {fi.origin}")
print(f"mi origin: {mi.origin}")

# The inv_warp was saved with moving.origin (=mi.origin=(0,0))
# But it represents displacements in the FIXED image space
# For the inverse, we need displacements defined in the MOVING image space
# This mismatch might be the issue!

# Let's try saving inv_warp with FIXED origin
import tempfile
inv_w = ants.image_read(inv_warp)
inv_w_fixed = ants.from_numpy(inv_w.numpy(), origin=fi.origin, spacing=fi.spacing,
                                direction=fi.direction, has_components=True)
f = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
ants.image_write(inv_w_fixed, f)
w6 = ants.apply_transforms(mi, fi, [inv_aff, f])
mi6 = ants.image_mutual_information(mi, w6)
print(f"\nInv warp with fi origin: MI={mi6:.4f}")
