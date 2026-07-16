import ants
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Run PyTorch SyN which optimizes BOTH Affine and Deformable
reg_iterations = [40, 20, 0]
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', reg_iterations=reg_iterations)

# Evaluate Affine alone
affine_file = reg_py['fwdtransforms'][1]
w_aff = ants.apply_transforms(fi, mi, [affine_file])
mi_aff = ants.image_mutual_information(fi, w_aff)
print(f"PyTorch Affine (from SyN) MI: {mi_aff:.4f}")

# Evaluate Deformable + Affine
fwd_file = reg_py['fwdtransforms'][0]
w_syn = ants.apply_transforms(fi, mi, [fwd_file, affine_file])
mi_syn = ants.image_mutual_information(fi, w_syn)
print(f"PyTorch SyN MI: {mi_syn:.4f}")

# What if we apply them separately? (Which is mathematically different but let's test)
w_sep = ants.apply_transforms(fi, mi, [affine_file])
w_sep = ants.apply_transforms(fi, w_sep, [fwd_file])
mi_sep = ants.image_mutual_information(fi, w_sep)
print(f"PyTorch Separate Application MI: {mi_sep:.4f}")
