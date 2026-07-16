import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import syntx

# Load 3D Mindboggle images
print("Loading 3D Mindboggle images...")
base_path = '/Users/stnava/data/mindboggle/volumes'
fi_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
mi_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
fl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')
ml_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi_full = ants.image_read(fi_path)
mi_full = ants.image_read(mi_path)
fl_full = ants.image_read(fl_path)
ml_full = ants.image_read(ml_path)

# Crop fixed image to bounding box
print("Cropping fixed image...")
mask = ants.get_mask(fi_full)
mask_dilated = ants.iMath(mask, "MD", 12)
fi_cropped = ants.crop_image(fi_full, mask_dilated)
fl_cropped = ants.crop_image(fl_full, mask_dilated)

# Downsample both fixed and moving images to low resolution (64x64x64) to run in seconds!
print("Downsampling images to 4.0mm spacing...")
fi = ants.resample_image(fi_cropped, (64, 64, 64), use_voxels=True, interp_type=0)
fl = ants.resample_image(fl_cropped, (64, 64, 64), use_voxels=True, interp_type=1)
mi = ants.resample_image(mi_full, (64, 64, 64), use_voxels=True, interp_type=0)
ml = ants.resample_image(ml_full, (64, 64, 64), use_voxels=True, interp_type=1)

# 1. ANTs Affine
print("\nRunning ANTs Affine registration...")
reg_ants_affine = ants.registration(fi, mi, 'Affine')
warped_ants_affine = reg_ants_affine['warpedmovout']
mi_affine_val = ants.image_mutual_information(fi, warped_ants_affine)
print(f"ANTs Affine | MI: {mi_affine_val:.6f}")

def compute_dkt_dice(fixed_labels, warped_moving_labels):
    overlap = ants.label_overlap_measures(fixed_labels, warped_moving_labels)
    overlap = overlap[(overlap['Label'] != 0) & (overlap['Label'] != 'All')]
    if 'MeanOverlap' in overlap.columns:
        return float(overlap['MeanOverlap'].mean())
    elif 'TotalOrTargetOverlap' in overlap.columns:
        return float(overlap['TotalOrTargetOverlap'].mean())
    return 0.0

# 2. ANTs SyN (SyNOnly, initialized with ANTs Affine)
print("\nRunning ANTs SyN (SyNOnly, 15 iterations)...")
reg_ants_syn = ants.registration(
    fixed=fi, moving=mi, type_of_transform='SyNOnly',
    initial_transform=reg_ants_affine['fwdtransforms'],
    reg_iterations=[15],
    syn_metric='cc', syn_sampling=4,
    flow_sigma=3.0, total_sigma=0.0,
    grad_step=0.1
)
mi_ants = ants.image_mutual_information(fi, reg_ants_syn['warpedmovout'])
warped_ants_ml = ants.apply_transforms(fi, ml, reg_ants_syn['fwdtransforms'], interpolator='nearestNeighbor')
dice_ants = compute_dkt_dice(fl, warped_ants_ml)
print(f"ANTs SyN | Dice: {dice_ants:.6f} | MI: {mi_ants:.6f}")

# 3. Syntx PyTorch SyN (SyNOnly, initialized with ANTs Affine)
print("\nRunning Syntx PyTorch SyN (SyNOnly, 15 iterations)...")
reg_syn_py = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='pytorch',
    initial_transform=reg_ants_affine['fwdtransforms'],
    levels=[1],
    reg_iterations=[15],
    grad_step=0.1, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='lncc', lncc_radius=4, inverse_steps=5
)
mi_py = ants.image_mutual_information(fi, reg_syn_py['warpedmovout'])
warped_py_ml = ants.apply_transforms(fi, ml, reg_syn_py['fwdtransforms'], interpolator='nearestNeighbor')
dice_py = compute_dkt_dice(fl, warped_py_ml)
print(f"Syntx PyTorch SyN | Dice: {dice_py:.6f} | MI: {mi_py:.6f}")

# 4. Syntx JAX SyN (SyNOnly, initialized with ANTs Affine)
print("\nRunning Syntx JAX SyN (SyNOnly, 15 iterations)...")
reg_syn_jax = syntx.syn(
    fixed=fi, moving=mi, type_of_transform='SyNOnly', backend='jax',
    initial_transform=reg_ants_affine['fwdtransforms'],
    levels=[1],
    reg_iterations=[15],
    grad_step=0.1, flow_sigma=3.0, elastic_sigma=0.0,
    syn_metric='lncc', lncc_radius=4, inverse_steps=5
)
mi_jax = ants.image_mutual_information(fi, reg_syn_jax['warpedmovout'])
warped_jax_ml = ants.apply_transforms(fi, ml, reg_syn_jax['fwdtransforms'], interpolator='nearestNeighbor')
dice_jax = compute_dkt_dice(fl, warped_jax_ml)
print(f"Syntx JAX SyN | Dice: {dice_jax:.6f} | MI: {mi_jax:.6f}")

# Clean up
for reg in [reg_ants_affine, reg_ants_syn, reg_syn_py, reg_syn_jax]:
    for path in reg['fwdtransforms'] + reg['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
