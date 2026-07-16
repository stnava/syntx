import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
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

# 1. ANTs Affine Registration
print("Running ANTs Affine registration...")
reg_ants = ants.registration(fi, mi, 'Affine')
warped_ants_ml = ants.apply_transforms(fi, ml, reg_ants['fwdtransforms'], interpolator='nearestNeighbor')
overlap_ants = ants.label_overlap_measures(fl, warped_ants_ml)
overlap_ants = overlap_ants[(overlap_ants['Label'] != 0) & (overlap_ants['Label'] != 'All')]
print("ANTs Affine 3D DKT Dice:", overlap_ants['TotalOrTargetOverlap'].mean())

tx_ants = ants.read_transform(reg_ants['fwdtransforms'][0])
print("ANTs Affine Matrix:\n", tx_ants.parameters[:9].reshape(3,3))
print("ANTs Affine Translation:", tx_ants.parameters[9:])
print("ANTs Affine Fixed Parameters (center):", tx_ants.fixed_parameters)

# 2. Syntx Affine Registration (Epoch 0)
print("\nRunning Syntx Affine registration (0 epochs)...")
reg_syn = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='Affine', backend='pytorch',
    affine_iterations=[0]
)
tx_syn = ants.read_transform(reg_syn['fwdtransforms'][0])
print("Syntx Affine Matrix:\n", tx_syn.parameters[:9].reshape(3,3))
print("Syntx Affine Translation:", tx_syn.parameters[9:])
print("Syntx Affine Fixed Parameters (center):", tx_syn.fixed_parameters)

# Clean up
for path in reg_ants['fwdtransforms'] + reg_syn['fwdtransforms'] + reg_syn['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
