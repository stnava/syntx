import sys
sys.path.insert(0, 'src')
import syntx
import ants
import torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = fi.clone()

reg = syntx.syn(fi, mi, 'SyNTo', backend='pytorch', reg_iterations=[0, 0, 0], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi', verbose=True)

warped = reg['warpedmovout']
mi_ants = ants.image_mutual_information(fi, warped)
print("PyTorch SyN identity MI:", mi_ants)
