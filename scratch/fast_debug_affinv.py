"""Debug: affine inverse parameters."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants, numpy as np
from syntx.syn import grid_to_physical_affine_torch
import torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[5,0,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

m = rs['model']
T_grid = m.affine.get_matrix().detach()
print(f"T_grid:\n{T_grid.numpy()}")

# Forward affine
M_phys_fwd, t_phys_fwd = grid_to_physical_affine_torch(
    T_grid, fi.shape, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction)
print(f"\nFwd M_phys:\n{M_phys_fwd.numpy()}")
print(f"Fwd t_phys: {t_phys_fwd.numpy()}")

# Inverse affine (how the code does it)
T_inv = torch.linalg.inv(T_grid)
M_phys_inv, t_phys_inv = grid_to_physical_affine_torch(
    T_inv, mi.shape, mi.spacing, mi.origin, mi.direction,
    fi.shape, fi.spacing, fi.origin, fi.direction)
print(f"\nInv M_phys:\n{M_phys_inv.numpy()}")
print(f"Inv t_phys: {t_phys_inv.numpy()}")

# Check: M_inv should be inverse of M_fwd
M_fwd_np = M_phys_fwd.numpy()
M_inv_np = M_phys_inv.numpy()
t_fwd_np = t_phys_fwd.numpy()
t_inv_np = t_phys_inv.numpy()
print(f"\nM_fwd @ M_inv:\n{M_fwd_np @ M_inv_np}")  # Should be identity
print(f"t_fwd + M_fwd @ t_inv: {t_fwd_np + M_fwd_np @ t_inv_np}")  # Should be zero

# Compare with direct numpy inverse
M_inv_direct = np.linalg.inv(M_fwd_np)
t_inv_direct = -M_inv_direct @ t_fwd_np
print(f"\nDirect inv M:\n{M_inv_direct}")
print(f"Direct inv t: {t_inv_direct}")

# Test: use direct inverse
import tempfile
aff_inv_direct = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
tx = ants.new_ants_transform(precision='float', dimension=2, transform_type='AffineTransform')
tx.set_parameters(np.concatenate([M_inv_direct.ravel(), t_inv_direct]).astype(np.float32))
tx.set_fixed_parameters(np.zeros(2))
ants.write_transform(tx, aff_inv_direct)

inv_transforms_fixed = [aff_inv_direct, rs['invtransforms'][1]]
wi = ants.apply_transforms(mi, fi, inv_transforms_fixed)
print(f"\nInverse with direct M_inv: MI={ants.image_mutual_information(mi, wi):.4f}")

# Original inverse
wi_orig = ants.apply_transforms(mi, fi, rs['invtransforms'])
print(f"Original inverse: MI={ants.image_mutual_information(mi, wi_orig):.4f}")
