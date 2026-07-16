import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F
import syntx
import tempfile

base_path = '/Users/stnava/data/mindboggle/volumes'
fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
fl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')
ml_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi_full = ants.image_read(fi_path)
mi_full = ants.image_read(mi_path)
fl_full = ants.image_read(fl_path)
ml_full = ants.image_read(ml_path)

# Crop
mask = ants.get_mask(fi_full)
mask_dilated = ants.iMath(mask, "MD", 12)
fi = ants.crop_image(fi_full, mask_dilated)
fl = ants.crop_image(fl_full, mask_dilated)

mi = mi_full
ml = ml_full

# Run Syntx SyN with very quick settings: 5 deformable iterations
print("Running Syntx SyN...")
reg = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[100, 50, 20], reg_iterations=[5, 0, 0],
    syn_metric='lncc', lncc_radius=2, sampling_percentage=0.2
)

# Compute Dice of original (unflipped components)
warped_ml_orig = ants.apply_transforms(fi, ml, reg['fwdtransforms'], interpolator='nearestNeighbor')
overlap = ants.label_overlap_measures(fl, warped_ml_orig)
overlap = overlap[(overlap['Label'] != 0) & (overlap['Label'] != 'All')]
print("Original (unflipped components) 3D DKT Dice:", overlap['TotalOrTargetOverlap'].mean())

# Flip components of the saved displacement field and check Dice
fwd_file_orig = reg['fwdtransforms'][0]
disp_img = ants.image_read(fwd_file_orig)
disp_np = disp_img.numpy()

# Flip the components from ZYX to XYZ!
disp_np_flipped = disp_np[..., ::-1].copy()

fwd_file_flipped = tempfile.NamedTemporaryFile(suffix='_fwd_flipped.nii.gz', delete=False).name
fwd_img_flipped = ants.from_numpy(
    disp_np_flipped, origin=fi.origin, spacing=fi.spacing, direction=fi.direction, has_components=True
)
ants.image_write(fwd_img_flipped, fwd_file_flipped)

flipped_transforms = [fwd_file_flipped] + reg['fwdtransforms'][1:]
warped_ml_flipped = ants.apply_transforms(fi, ml, flipped_transforms, interpolator='nearestNeighbor')
overlap_flipped = ants.label_overlap_measures(fl, warped_ml_flipped)
overlap_flipped = overlap_flipped[(overlap_flipped['Label'] != 0) & (overlap_flipped['Label'] != 'All')]
print("Flipped components 3D DKT Dice:", overlap_flipped['TotalOrTargetOverlap'].mean())

# Clean up
if os.path.exists(fwd_file_flipped):
    os.remove(fwd_file_flipped)
for path in reg['fwdtransforms'] + reg['invtransforms']:
    if os.path.exists(path):
        os.remove(path)
