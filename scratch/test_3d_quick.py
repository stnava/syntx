import sys, os, time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants
import numpy as np

# Load 3D Mindboggle images
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

def compute_dkt_dice(fixed_labels, warped_moving_labels):
    overlap = ants.label_overlap_measures(fixed_labels, warped_moving_labels)
    overlap = overlap[(overlap['Label'] != 0) & (overlap['Label'] != 'All')]
    if 'TotalOrTargetOverlap' in overlap.columns:
        return float(overlap['TotalOrTargetOverlap'].mean())
    elif 'TargetOverlap' in overlap.columns:
        return float(overlap['TargetOverlap'].mean())
    elif 'MeanOverlap' in overlap.columns:
        return float(overlap['MeanOverlap'].mean())
    return 0.0

print("Running PyTorch 3D Quick...")
t0 = time.time()
rs_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='pytorch',
    affine_iterations=[50, 10, 0], reg_iterations=[5, 0, 0],
    syn_metric='lncc', lncc_window_size=5, sampling_percentage=0.2
)
print(f"  PyTorch done in {time.time() - t0:.1f}s")
warped_py_img = ants.apply_transforms(fi, mi, rs_py['fwdtransforms'])
warped_py_ml = ants.apply_transforms(fi, ml, rs_py['fwdtransforms'], interpolator='nearestNeighbor')
mi_py = ants.image_mutual_information(fi, warped_py_img)
dice_py = compute_dkt_dice(fl, warped_py_ml)
print(f"  PyTorch MI: {mi_py:.4f} | DKT Dice: {dice_py:.4f}")

print("Running JAX 3D Quick...")
t0 = time.time()
rs_jax = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyN', backend='jax',
    affine_iterations=[50, 10, 0], reg_iterations=[5, 0, 0],
    syn_metric='lncc', lncc_window_size=5, sampling_percentage=0.2
)
print(f"  JAX done in {time.time() - t0:.1f}s")
warped_jax_img = ants.apply_transforms(fi, mi, rs_jax['fwdtransforms'])
warped_jax_ml = ants.apply_transforms(fi, ml, rs_jax['fwdtransforms'], interpolator='nearestNeighbor')
mi_jax = ants.image_mutual_information(fi, warped_jax_img)
dice_jax = compute_dkt_dice(fl, warped_jax_ml)
print(f"  JAX MI: {mi_jax:.4f} | DKT Dice: {dice_jax:.4f}")
