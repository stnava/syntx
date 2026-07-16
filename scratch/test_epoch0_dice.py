import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import syntx

base_path = '/Users/stnava/data/mindboggle/volumes'
fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
fl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')
ml_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi_full = ants.image_read(fi_path)
mi_full = ants.image_read(mi_path)
fl_full = ants.image_read(fl_path)
ml_full = ants.image_read(ml_path)

mask = ants.get_mask(fi_full)
mask_dilated = ants.iMath(mask, "MD", 12)
fi = ants.crop_image(fi_full, mask_dilated)
fl = ants.crop_image(fl_full, mask_dilated)

mi = mi_full
ml = ml_full

print("Running Syntx SyN with 0 affine and 0 deformable iterations...")
reg = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[0], reg_iterations=[0]
)

# Compute Dice
warped_ml = ants.apply_transforms(fi, ml, reg['fwdtransforms'], interpolator='nearestNeighbor')
overlap = ants.label_overlap_measures(fl, warped_ml)
overlap = overlap[(overlap['Label'] != 0) & (overlap['Label'] != 'All')]
print("Epoch 0 3D DKT Dice:", overlap['TotalOrTargetOverlap'].mean())

# Clean up
for path in reg['fwdtransforms'] + reg['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
