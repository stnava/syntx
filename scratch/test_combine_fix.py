import ants
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

reg_iterations = [40, 20, 0]
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)

# Evaluate Affine alone
affine_file = reg_py['fwdtransforms'][1]
w_aff = ants.apply_transforms(fi, mi, [affine_file])
mi_aff = ants.image_mutual_information(fi, w_aff)
print(f"PyTorch Affine (from SyN) MI (Original): {mi_aff:.4f}")

# Read the affine file, swap it properly
aff_tx = ants.read_transform(affine_file)
params = aff_tx.parameters
print("Original params:", params)
M_yy, M_yx, M_xy, M_xx, t_y, t_x = params
fixed_params = aff_tx.fixed_parameters

# Create new transform with swapped params for ITK (X, Y)
# ITK expects: M_xx, M_xy, M_yx, M_yy, t_x, t_y
new_params = np.array([M_xx, M_xy, M_yx, M_yy, t_x, t_y])
new_tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2, parameters=new_params, fixed_parameters=fixed_params)
ants.write_transform(new_tx, 'scratch/fixed_affine.mat')

w_aff_fixed = ants.apply_transforms(fi, mi, ['scratch/fixed_affine.mat'])
mi_aff_fixed = ants.image_mutual_information(fi, w_aff_fixed)
print(f"PyTorch Affine (from SyN) MI (Fixed): {mi_aff_fixed:.4f}")

# Evaluate Deformable + Affine with original
fwd_file = reg_py['fwdtransforms'][0]
w_syn = ants.apply_transforms(fi, mi, [fwd_file, affine_file])
mi_syn = ants.image_mutual_information(fi, w_syn)
print(f"PyTorch SyN MI (Original): {mi_syn:.4f}")

# Evaluate Deformable + Affine with FIXED
w_syn_fixed = ants.apply_transforms(fi, mi, [fwd_file, 'scratch/fixed_affine.mat'])
mi_syn_fixed = ants.image_mutual_information(fi, w_syn_fixed)
print(f"PyTorch SyN MI (Fixed): {mi_syn_fixed:.4f}")

