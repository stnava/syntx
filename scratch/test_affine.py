import sys
sys.path.insert(0, 'src')
import syntx
import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('mni')).resample_image((64, 64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('mni')).resample_image((64, 64, 64), use_voxels=True)
tx = ants.create_ants_transform(transform_type='Euler3DTransform', translation=(20, 0, 0))
ants.write_transform(tx, 'scratch/tx.mat')
mi = ants.apply_transforms(fi, mi, transformlist=['scratch/tx.mat'])

# Affine only
reg = syntx.syn(fi, mi, 'SyNTo', backend='pytorch', reg_iterations=[0, 0, 0], affine_iterations=[40, 20, 10], similarity_metric='mattes_mi')

warped = reg['warpedmovout']
mi_ants = ants.image_mutual_information(fi, warped)
print("PyTorch Affine output MI:", mi_ants)

# ANTs affine only
reg_ants = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine', reg_iterations=[0, 0, 0], aff_iterations=[40, 20, 10], aff_smoothing_sigmas=[2, 1, 0], aff_shrink_factors=[4, 2, 1], aff_metric='mattes')
print("ANTs Affine output MI:", ants.image_mutual_information(fi, reg_ants['warpedmovout']))
