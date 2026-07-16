import ants
import numpy as np
import sys
sys.path.insert(0, 'src')
import syntx

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

reg_ants_affine = ants.registration(fi, mi, 'Affine')
mi_ants_affine = ants.image_mutual_information(fi, reg_ants_affine['warpedmovout'])
print(f"ANTs Affine MI: {mi_ants_affine:.4f}")

reg_iterations = [40, 20, 0]
reg_py = syntx.syn(fi, mi, 'SyN', backend='pytorch', 
                   initial_transform=reg_ants_affine['fwdtransforms'],
                   reg_iterations=reg_iterations, affine_iterations=[0])

fwdtransforms = reg_py['fwdtransforms']
w_py = ants.apply_transforms(fi, mi, fwdtransforms)
mi_py = ants.image_mutual_information(fi, w_py)
print(f"PyTorch MI (with ANTs Affine init): {mi_py:.4f}")

reg_ants = ants.registration(fi, mi, 'SyN', syn_iterations=reg_iterations)
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
print(f"ANTs SyN MI: {mi_ants:.4f}")

