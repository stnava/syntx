import ants
import os
import time
import numpy as np
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

# Test sweeps in JAX
sweeps = [
    {'grad_step': 0.1, 'flow_sigma': 3.0, 'syn_sampling': 4, 'reg_iterations': [100, 50, 20]},
    {'grad_step': 0.25, 'flow_sigma': 3.0, 'syn_sampling': 4, 'reg_iterations': [100, 50, 20]},
    {'grad_step': 0.25, 'flow_sigma': 3.0, 'syn_sampling': 4, 'reg_iterations': [150, 100, 50]},
    {'grad_step': 0.25, 'flow_sigma': 4.0, 'syn_sampling': 4, 'reg_iterations': [100, 50, 20]},
    {'grad_step': 0.25, 'flow_sigma': 2.0, 'syn_sampling': 4, 'reg_iterations': [100, 50, 20]}
]

for idx, params in enumerate(sweeps):
    print(f"\n--- Sweep {idx + 1}: {params} ---")
    t0 = time.time()
    reg_jax = registration(
        fixed=fi_low, moving=mi_low, type_of_transform='SyN',
        backend='jax', syn_metric='lncc',
        syn_sampling=params['syn_sampling'],
        affine_iterations=[200, 100, 50],
        reg_iterations=params['reg_iterations'],
        grad_step=params['grad_step'],
        flow_sigma=params['flow_sigma'],
        verbose=False
    )
    t_jax = time.time() - t0
    dice_jax = compute_mean_dice(fl, ml, reg_jax['fwdtransforms'])
    print(f"Sweep {idx + 1} DICE: {dice_jax:.4f} ({t_jax:.2f}s)")
    
    for path in reg_jax['fwdtransforms'] + reg_jax['invtransforms']:
        if os.path.exists(path):
            os.remove(path)
