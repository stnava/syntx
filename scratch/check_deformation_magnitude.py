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

# ANTs SyN
reg_ants = ants.registration(fi, mi_affine, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')
fwd_ants = ants.image_read(reg_ants['fwdtransforms'][0])
print("ANTs deformation mean norm:", np.mean(np.linalg.norm(fwd_ants.numpy(), axis=-1)))
print("ANTs deformation max norm:", np.max(np.linalg.norm(fwd_ants.numpy(), axis=-1)))

# PyTorch SyN
reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch', reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi', verbose=False, grad_step=0.01)
fwd_pt = ants.image_read(reg_pt['fwdtransforms'][0])
print("PyTorch deformation mean norm:", np.mean(np.linalg.norm(fwd_pt.numpy(), axis=-1)))
print("PyTorch deformation max norm:", np.max(np.linalg.norm(fwd_pt.numpy(), axis=-1)))

