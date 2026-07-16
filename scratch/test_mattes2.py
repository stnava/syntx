import sys
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np

sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch', reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi', verbose=True, grad_step=0.1, fluid_sigma=np.sqrt(3.0))

print("Forward transforms:", reg_pt['fwdtransforms'])
warped_ants = ants.apply_transforms(fi, mi_affine, reg_pt['fwdtransforms'])
print("PyTorch SyN MI:", ants.image_mutual_information(fi, warped_ants))

fwd_pt = ants.image_read(reg_pt['fwdtransforms'][0])
print("PyTorch deformation max norm:", np.max(np.linalg.norm(fwd_pt.numpy(), axis=-1)))
