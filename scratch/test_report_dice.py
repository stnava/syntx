import os
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

print("Running PyTorch LNCC...")
reg_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[30, 20, 10], reg_iterations=[20, 10, 0],
    syn_metric='lncc', lncc_window_size=5
)

# Apply transforms to labels
warped_py_ml = ants.apply_transforms(fi, ml, reg_py['fwdtransforms'], interpolator='nearestNeighbor')

# Compute overlap
overlap = ants.label_overlap_measures(fl, warped_py_ml)
print("Overlap columns:", overlap.columns)
print("Overlap head:\n", overlap.head(10))
