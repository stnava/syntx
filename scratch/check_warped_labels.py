import sys, os
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_lbl = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")
mi_lbl = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/labels.DKT31.manual.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))
fi_lbl_cropped = ants.crop_image(fi_lbl, ants.iMath(ants.get_mask(fi), "MD", 12))

rs = syntx.syn(
    fixed=fi_cropped, moving=mi, type_of_transform='Affine', backend='pytorch',
    affine_iterations=[0], reg_iterations=[0]
)

wrp = ants.apply_transforms(
    fixed=fi_cropped, moving=mi_lbl, transformlist=rs['fwdtransforms'],
    interpolator='nearestNeighbor'
)

print("Unique labels in fi_lbl_cropped:", np.unique(fi_lbl_cropped.numpy())[:10])
print("Unique labels in mi_lbl:", np.unique(mi_lbl.numpy())[:10])
print("Unique labels in warped mi_lbl:", np.unique(wrp.numpy())[:10])
print("Sum of warped mi_lbl:", wrp.numpy().sum())
