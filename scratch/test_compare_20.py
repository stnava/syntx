import sys
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'

sys.path.insert(0, 'src')
import syntx
import ants
import torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Initial Affine using ANTs
print("Running initial affine...")
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

# Run ANTs SyNOnly
print("Running ANTs SyNOnly...")
reg_ants = ants.registration(fi, mi_affine, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])
print("ANTs MI:", mi_ants)

# Run PyTorch SyN
print("Running PyTorch SyN...")
reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch', reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi')
mi_pt = ants.image_mutual_information(fi, reg_pt['warpedmovout'])
print("PyTorch MI:", mi_pt)

