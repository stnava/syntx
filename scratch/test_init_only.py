import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants
import numpy as np

# Load 3D Mindboggle images
base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_lbl = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")
mi_lbl = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/labels.DKT31.manual.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))
fi_lbl_cropped = ants.crop_image(fi_lbl, ants.iMath(ants.get_mask(fi), "MD", 12))

# Run Syntx registration with 0 epochs (only CoM Init runs!)
rs = syntx.syn(
    fixed=fi_cropped, moving=mi, type_of_transform='Affine', backend='pytorch',
    affine_iterations=[0], reg_iterations=[0]
)

wrp = ants.apply_transforms(
    fixed=fi_cropped, moving=mi_lbl, transformlist=rs['fwdtransforms'],
    interpolator='nearestNeighbor'
)
df = ants.label_overlap_measures(fi_lbl_cropped, wrp)
target_col = 'TargetOverlap' if 'TargetOverlap' in df.columns else 'TotalOrTargetOverlap'
print("Syntx Init-only DICE:", df[(df['Label'] != 0) & (df['Label'] != 'All')][target_col].mean())
