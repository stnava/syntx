import sys
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np

sys.path.insert(0, 'src')
import syntx
import ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Initial Affine using ANTs
print("Running initial affine...")
tx_affine = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_affine = ants.apply_transforms(fi, mi, tx_affine['fwdtransforms'])

print("Running PyTorch SyN with LNCC...")
reg_pt = syntx.syn(fi, mi_affine, 'SyNTo', backend='pytorch', reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='lncc', verbose=True)

