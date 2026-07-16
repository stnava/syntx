import ants
import os
import time
import numpy as np
import sys

sys.path.insert(0, 'src')
from syntx.syn import registration

def compute_mean_dice(fixed_labels, moving_labels, transforms):
    warped_labels = ants.apply_transforms(
        fixed=fixed_labels,
        moving=moving_labels,
        transformlist=transforms,
        interpolator='nearestNeighbor'
    )
    overlap = ants.label_overlap_measures(
        source_image=fixed_labels, 
        target_image=warped_labels
    )
    overlap_valid = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != '0') & (overlap['Label'] != 0)]
    if 'TargetOverlap' in overlap_valid.columns:
        dice = overlap_valid['TargetOverlap'].mean()
    else:
        dice = overlap_valid['MeanOverlap'].mean()
    return dice

base_path = '/Users/stnava/data/mindboggle/volumes'

fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fi = ants.image_read(fixed_img_path)
fl = ants.image_read(fixed_lbl_path)
mi = ants.image_read(moving_img_path)
ml = ants.image_read(moving_lbl_path)

fi_low = ants.resample_image(fi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
mi_low = ants.resample_image(mi, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)

# Run ANTs Affine first
print("--- Running ANTs Affine ---")
reg_aff = ants.registration(fixed=fi_low, moving=mi_low, type_of_transform='Affine')
aff_tx = reg_aff['fwdtransforms'][0]

print("\n--- Running Fast PyTorch SyN (1 level, 1 iteration) ---")
t0 = time.time()
reg_py = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    backend='pytorch', syn_metric='lncc', syn_sampling=4,
    initial_transform=aff_tx,
    levels=[4],
    reg_iterations=[1],
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=True
)
t1 = time.time()
print(f"PyTorch 1-level 1-iteration time: {t1 - t0:.4f}s")
print("\n--- Running Fast PyTorch SyN (1 level, 10 iterations) ---")
t0 = time.time()
reg_py = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    backend='pytorch', syn_metric='lncc', syn_sampling=4,
    initial_transform=aff_tx,
    levels=[4],
    reg_iterations=[10],
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=True
)
t1 = time.time()
print(f"PyTorch 1-level 10-iterations time: {t1 - t0:.4f}s")

print("\n--- Running Fast JAX SyN (1 level, 10 iterations) ---")
t0 = time.time()
reg_jax = registration(
    fixed=fi_low, moving=mi_low, type_of_transform='SyNOnly',
    backend='jax', syn_metric='lncc', syn_sampling=4,
    initial_transform=aff_tx,
    levels=[4],
    reg_iterations=[10],
    grad_step=0.1,
    flow_sigma=3.0,
    verbose=True
)
t1 = time.time()
print(f"JAX 1-level 10-iterations time: {t1 - t0:.4f}s")
