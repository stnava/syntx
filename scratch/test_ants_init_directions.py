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
print("best_t:", best_t)

# Run ANTs Translation registration
reg = ants.registration(fixed=fi_cropped, moving=mi, type_of_transform='Translation')
tx_ants = ants.read_transform(reg['fwdtransforms'][0])
print("ANTs Translation Parameters:", tx_ants.parameters)
print("ANTs Translation Fixed Parameters:", tx_ants.fixed_parameters)
