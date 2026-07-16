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

com_fi = np.array(ants.get_center_of_mass(fi_cropped))
com_mi = np.array(ants.get_center_of_mass(mi))
best_t = com_mi - com_fi

tx = ants.new_ants_transform(precision='float', dimension=3, transform_type='TranslationTransform')
tx.set_parameters(best_t)

tmp_path = '/tmp/tx_init.mat'
ants.write_transform(tx, tmp_path)

wrp = ants.apply_transforms(
    fixed=fi_cropped, moving=mi_lbl, transformlist=[tmp_path],
    interpolator='nearestNeighbor'
)

df = ants.label_overlap_measures(fi_lbl_cropped, wrp)
print(df.head(10))
print(df.tail(10))

if os.path.exists(tmp_path):
    os.remove(tmp_path)
