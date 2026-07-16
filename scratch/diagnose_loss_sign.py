"""Check if loss direction is consistent: does the SyN loop minimize or maximize?"""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'src')
import ants
from syntx.syn import (get_physical_grid_torch, physical_to_normalized_torch, 
                        mattes_mi_loss_nd, local_ncc_loss_nd, SyNTo,
                        prepare_mid_images_and_gradients_torch, 
                        _physical_to_normalized_torch_yfirst)

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_affine.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Baseline MI
mi_0 = mattes_mi_loss_nd(J_t, I_t).item()
print(f"Baseline MI loss (J, I): {mi_0:.4f}")
print(f"Baseline MI loss (I, J): {mattes_mi_loss_nd(I_t, J_t).item():.4f}")

print("\n=== Checking gradient SIGN with a known translation ===")
# Apply a KNOWN small translation in physical space, check if MI improves
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)

# Create a small constant displacement (shift moving image by 2mm in X direction)
shift = torch.zeros(1, *fi.shape, 2, dtype=torch.float32)
shift[..., 1] = 2.0  # 2mm shift in X (component 1 in YX order)

phi_shifted = X_phys + shift
coords = physical_to_normalized_torch(phi_shifted, fi.shape, fi.spacing, fi.origin, fi.direction)
J_shifted = F.grid_sample(J_t, coords, align_corners=True, padding_mode='border')

mi_shifted = mattes_mi_loss_nd(J_shifted, I_t).item()
print(f"MI after +2mm X shift of moving: {mi_shifted:.4f}  (Δ={mi_shifted - mi_0:.4f})")

shift2 = torch.zeros(1, *fi.shape, 2, dtype=torch.float32)
shift2[..., 1] = -2.0
phi_shifted2 = X_phys + shift2
coords2 = physical_to_normalized_torch(phi_shifted2, fi.shape, fi.spacing, fi.origin, fi.direction)
J_shifted2 = F.grid_sample(J_t, coords2, align_corners=True, padding_mode='border')
mi_shifted2 = mattes_mi_loss_nd(J_shifted2, I_t).item()
print(f"MI after -2mm X shift of moving: {mi_shifted2:.4f}  (Δ={mi_shifted2 - mi_0:.4f})")

print("\n=== Checking what prepare_mid_images_and_gradients_torch does ===")
print("In the SyN loop:")
print("  I_mid = I(x + warp_l2r(x))    <- warps FIXED image using warp_l2r")
print("  J_mid = J(affine(x + warp_r2l(x)))  <- warps MOVING image using warp_r2l")
print("  loss = fn(J_mid, I_mid)  <- NOTE ORDER: J first, I second")
print()
print("  In local_ncc_loss_nd(I, J): returns -mean(cc)")
print("  In mattes_mi_loss_nd(I, J): returns -MI")
print()
print("  So loss(J_mid, I_mid) = -MI(J_mid, I_mid)")
print("  Minimizing loss = Maximizing MI ← CORRECT")
print()

print("=== Checking gradient direction w.r.t. warp_l2r ===")
# The gradient ∂loss/∂warp_l2r tells which direction to move the warp to INCREASE loss
# We want to DECREASE loss, so we move AGAINST the gradient
# In the update: coords_phys_l = X_phys - delta_l (subtracting the gradient-derived step)
# So warp_l2r is updated to push I_mid TOWARD J_mid at midpoint

# BUT: warp_l2r warps the FIXED image. Increasing warp_l2r pushes I(x + warp_l2r)
# further from identity. In the composition:
# warp_new = warp_old(x - delta) - delta
# This is correct for greedy SyN.

print("=== Checking the COMPOSITION of midpoint fields ===")
print("After optimization, lines 1580-1593 compose:")
print("  total_fwd = warp_l2r_inv + warp_r2l(X + warp_l2r_inv)")
print()
print("  warp_l2r: warps fixed toward midpoint (pushes I toward J)")
print("  warp_r2l: warps moving toward midpoint (pushes J toward I)")
print()
print("  The forward map should take a point in FIXED space and give")
print("  its corresponding position in MOVING space.")
print()
print("  SyN forward = φ_2 ∘ φ_1^{-1}")
print("  where φ_1 = Id + warp_l2r (fixed → midpoint)")
print("  and   φ_2 = Id + warp_r2l (moving → midpoint)")
print()
print("  forward(x) = φ_2(φ_1^{-1}(x))")
print("             = (φ_1^{-1}(x)) + warp_r2l(φ_1^{-1}(x))")
print()
print("  In the code (lines 1581-1585):")
print("    phi_l2r_phys = X + w_l2r_inv  ← this is φ_1^{-1}(x)")
print("    disp_r2l_sampled = warp_r2l(phi_l2r_phys)  ← warp_r2l at φ_1^{-1}")
print("    full_l2r_phys = phi_l2r_phys + disp_r2l_sampled")
print("    total_fwd_disp = full_l2r_phys - X")
print()
print("  So total_fwd_disp(x) = warp_l2r_inv(x) + warp_r2l(x + warp_l2r_inv(x))")
print("  This IS φ_2(φ_1^{-1}(x)) - x = the SyN forward displacement.")
print()
print("  BUT WAIT: warp_r2l maps the moving image's coordinates.")
print("  The affine maps from the midpoint space to moving space.")
print("  So applying the total_fwd_disp alone (without the affine) gives")
print("  coordinates in the midpoint/fixed-grid space, NOT in moving space!")
print("  The total forward should INCLUDE the affine to fully map to moving space.")
print()
print("  In the export (line 2183): fwd_transforms = [fwd_file, affine_file]")
print("  ANTs applies transforms right-to-left: first affine, then warp.")
print("  So ANTs computes: warped(x) = moving(affine(x + total_fwd_disp(x)))")
print("  Wait -- ANTs applies transforms in the order listed, from RIGHT to LEFT.")
print("  [fwd_file, affine_file] means: first apply affine, then apply fwd_file.")
print("  apply_transforms(fixed, moving, [warp, affine]):")
print("    1. Apply affine to fixed coords -> get intermediate coords")
print("    2. Apply warp at those coords -> get moving coords")
print("    3. Sample moving at those coords")
print("  NO WAIT. ANTs applies the LAST transform first.")
print("  So [fwd_file, affine_file] means: apply affine_file first, then fwd_file.")
print("  which means: first map through affine, then add the displacement.")
print("  This is: warped(x) = moving(total_fwd_disp(affine(x)) + affine(x))")
print("  WRONG! The total_fwd_disp should be applied FIRST, then affine.")

