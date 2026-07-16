"""Test model.apply_inverse vs export."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, 'src')
import syntx, ants
from syntx.syn import grid_to_physical_affine_torch, get_physical_grid_torch, physical_to_normalized_torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

m = rs['model']
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Model's own forward method
J_warped_model = m.apply_forward(I_t, J_t)
mi_model_fwd = ants.image_mutual_information(fi,
    ants.from_numpy(J_warped_model[0,0].detach().numpy(), origin=fi.origin, spacing=fi.spacing, direction=fi.direction))
print(f"Model forward: MI={mi_model_fwd:.4f}")

# Model's own inverse method (if it exists)
if hasattr(m, 'apply_inverse'):
    I_warped_model = m.apply_inverse(I_t, J_t)
    mi_model_inv = ants.image_mutual_information(mi,
        ants.from_numpy(I_warped_model[0,0].detach().numpy(), origin=mi.origin, spacing=mi.spacing, direction=mi.direction))
    print(f"Model inverse: MI={mi_model_inv:.4f}")

# Manual composition with affine
X = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)
T_grid = m.affine.get_matrix().detach()
M_phys, t_phys = grid_to_physical_affine_torch(
    T_grid, fi.shape, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction)

# Forward: x -> affine -> deformable
# phi_fwd(x) = x + warp_l2r(x), then composed with affine
# The forward warp maps: fixed → moving (through midpoint)
# Total forward: for fixed point x, get moving point y
#   y = M @ (x + warp_l2r(x)) + t  ... NO
# Actually in ANTs SyN:
#   fwd_transforms = [warp_file, affine_file]
#   ANTs applies: affine first (maps moving→fixed), then warp (maps further in fixed space)
#   So the exported fwd warp is the TOTAL deformable in fixed space, applied AFTER affine

# What does the export actually do?
# The model stores:
#   warp_l2r: total composed midpoint deformable field, mapping fixed→composed
#   The affine maps moving space→fixed space (pre-alignment)
# So fwd_transforms = [warp_l2r_file, affine_file]
# ANTs does: apply affine (moving→fixed), then warp (fixed→warped_fixed)

# For the inverse:
# We need to undo: first undo warp, then undo affine
# inv_transforms = [affine_inv_file, warp_r2l_file]
# ANTs applies: warp_r2l first (in fixed space, maps fixed→something), then affine_inv

# But warp_r2l was composed as: maps fixed→moving (through midpoint) 
# in the FIXED coordinate system. So it should be applied first.

# The issue is the ORDER of the inverse transforms.
# Current: [affine_inv, warp_inv] -- ANTs applies warp_inv FIRST, then affine_inv
# But we need: apply affine_inv FIRST (to get from moving to fixed), then warp_inv (in fixed space)
# Wait no -- we're going from FIXED to MOVING:
# Step 1: Apply warp_r2l in fixed space (maps fixed to something)  
# Step 2: Apply affine_inv (maps fixed space to moving space)

# So correct order should be: [affine_inv, warp_r2l] which means ANTs applies warp_r2l first, then affine_inv
# That IS the current order: [affine_inv_file, inv_file] → ANTs applies inv_file (right), then affine_inv (left)

# Hmm, but test showed [warp_inv, aff_inv] gives MI=-0.629 and [aff_inv, warp_inv] gives MI=-0.025
# So applying warp first then affine works better!
# That means: [warp_inv, aff_inv] → ANTs applies aff_inv (right) first, then warp_inv (left)
# This suggests: go from fixed→moving by first undoing affine (fixed→pre-affine-moving), then applying warp_r2l

# I think the issue is: warp_r2l is defined in FIXED space and maps to a halfway point, 
# not to moving space. It needs the affine to be undone FIRST.

print(f"\nANTs applies transforms RIGHT to LEFT")
print(f"Current inv order: [affine_inv, warp_inv] → applies warp_inv first, then affine_inv")
print(f"This means: go from fixed, apply warp_r2l, then apply affine_inv")
print(f"But warp_r2l is in fixed space, so we should: apply affine_inv first, THEN warp_r2l")
print(f"Correct order: [warp_inv, affine_inv] → applies affine_inv first, then warp_inv")
