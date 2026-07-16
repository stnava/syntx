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

wrp_np = wrp.numpy().astype(np.int32)
fi_lbl_cropped_np = fi_lbl_cropped.numpy().astype(np.int32)

for val in np.unique(fi_lbl_cropped_np):
    if val == 0: continue
    fixed_mask = (fi_lbl_cropped_np == val)
    warped_mask = (wrp_np == val)
    intersection = (fixed_mask & warped_mask).sum()
    dice = 2.0 * intersection / (fixed_mask.sum() + warped_mask.sum()) if (fixed_mask.sum() + warped_mask.sum()) > 0 else 0.0
    if dice > 0.0:
        print(f"Label {val}: fixed_count={fixed_mask.sum()}, warped_count={warped_mask.sum()}, intersection={intersection}, dice={dice:.4f}")
