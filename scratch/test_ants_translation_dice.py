import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import ants
import numpy as np

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_lbl = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")
mi_lbl = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/labels.DKT31.manual.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))
fi_lbl_cropped = ants.crop_image(fi_lbl, ants.iMath(ants.get_mask(fi), "MD", 12))

reg = ants.registration(fixed=fi_cropped, moving=mi, type_of_transform='Translation')

wrp = ants.apply_transforms(
    fixed=fi_cropped, moving=mi_lbl, transformlist=reg['fwdtransforms'],
    interpolator='nearestNeighbor'
)

df = ants.label_overlap_measures(fi_lbl_cropped, wrp)
target_col = 'TargetOverlap' if 'TargetOverlap' in df.columns else 'TotalOrTargetOverlap'
df_valid = df[(df['Label'] != 0) & (df['Label'] != 'All')]
df_valid = df_valid[df_valid[target_col] < 1.0e10]
print("ANTs Translation DICE:", df_valid[target_col].mean())
