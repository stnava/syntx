import sys
import ants
import numpy as np

fi = ants.image_read(ants.get_ants_data('mni')).resample_image((64, 64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('mni')).resample_image((64, 64, 64), use_voxels=True)
tx = ants.create_ants_transform(transform_type='Euler3DTransform', translation=(20, 0, 0))
ants.write_transform(tx, 'scratch/tx.mat')
mi = ants.apply_transforms(fi, mi, transformlist=['scratch/tx.mat'])

reg = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', reg_iterations=[40, 20, 0], aff_iterations=[40, 20, 0], aff_smoothing_sigmas=[2, 1, 0], aff_shrink_factors=[4, 2, 1], syn_metric='mattes', aff_metric='mattes')
warped = reg['warpedmovout']

mi_ants = ants.image_mutual_information(fi, warped)
print("ANTs reference SyN MI of output:", mi_ants)
