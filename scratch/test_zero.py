import sys
sys.path.insert(0, 'src')
import syntx
import ants
import torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = fi.clone()

# Images are identical! SyN should NOT move anything!
reg = syntx.syn(fi, mi, 'SyNTo', backend='pytorch', reg_iterations=[40, 20, 10], affine_iterations=[0, 0, 0], similarity_metric='mattes_mi')

warped = reg['warpedmovout']
mi_ants = ants.image_mutual_information(fi, warped)
print("PyTorch SyN identity MI:", mi_ants)
print("Original MI:", ants.image_mutual_information(fi, mi))

# Let's see ANTs
reg_ants = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', reg_iterations=[40, 20, 10], aff_iterations=[0, 0, 0], aff_metric='mattes', syn_metric='mattes')
print("ANTs reference SyN identity MI:", ants.image_mutual_information(fi, reg_ants['warpedmovout']))
