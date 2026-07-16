#!/usr/bin/env python
"""Quick verification of fixes: re-run baseline comparison after changes."""
import sys
import os
import numpy as np
import ants
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from syntx.syn import registration

fi = ants.resample_image(ants.image_read(ants.get_ants_data('mni')), (2, 2, 2), use_voxels=False, interp_type=0)
mi = ants.resample_image(ants.image_read(ants.get_ants_data('ch2')), (2, 2, 2), use_voxels=False, interp_type=0)

def compute_multilabel_dice(img1, img2, num_classes=3):
    t1 = ants.threshold_image(img1, 'Otsu', num_classes)
    t2 = ants.threshold_image(img2, 'Otsu', num_classes)
    arr1 = t1.numpy().astype(int)
    arr2 = t2.numpy().astype(int)
    dices = []
    for label in range(1, num_classes + 1):
        m1 = (arr1 == label).astype(np.float32)
        m2 = (arr2 == label).astype(np.float32)
        intersection = np.sum(m1 * m2)
        total = np.sum(m1) + np.sum(m2)
        if total > 0:
            dices.append(2.0 * intersection / total)
    return np.mean(dices) if dices else 0.0

# Baseline
baseline = compute_multilabel_dice(fi, mi)

# ANTs
ants_result = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine')
ants_dice = compute_multilabel_dice(fi, ants_result['warpedmovout'])

# syntx (with fixes - defaults)
t0 = time.time()
syntx_result = registration(
    fixed=fi, moving=mi,
    type_of_transform='Affine',
    backend='pytorch',
    syn_metric='mattes_mi'
)
t1 = time.time()
syntx_dice = compute_multilabel_dice(fi, syntx_result['warpedmovout'])

# syntx with affine_lr=2e-2
t2 = time.time()
syntx_result2 = registration(
    fixed=fi, moving=mi,
    type_of_transform='Affine',
    backend='pytorch',
    syn_metric='mattes_mi',
    affine_lr=2e-2
)
t3 = time.time()
syntx_dice2 = compute_multilabel_dice(fi, syntx_result2['warpedmovout'])

# syntx with more iterations and lr tuning
t4 = time.time()
syntx_result3 = registration(
    fixed=fi, moving=mi,
    type_of_transform='Affine',
    backend='pytorch',
    syn_metric='mattes_mi',
    affine_iterations=[300, 200, 100],
    affine_lr=2e-2
)
t5 = time.time()
syntx_dice3 = compute_multilabel_dice(fi, syntx_result3['warpedmovout'])

print("\n" + "=" * 65)
print("POST-FIX VERIFICATION")
print("=" * 65)
print(f"{'Method':<40} {'DICE':>10} {'Time':>8}")
print("-" * 60)
print(f"{'Baseline (no reg)':<40} {baseline:>10.6f} {'N/A':>8}")
print(f"{'ANTs Affine':<40} {ants_dice:>10.6f}")
print(f"{'syntx Affine (new defaults)':<40} {syntx_dice:>10.6f} {t1-t0:>7.1f}s")
print(f"{'syntx Affine (lr=2e-2)':<40} {syntx_dice2:>10.6f} {t3-t2:>7.1f}s")
print(f"{'syntx Affine (lr=2e-2, more iters)':<40} {syntx_dice3:>10.6f} {t5-t4:>7.1f}s")
print(f"\nGap (ANTs - syntx default): {ants_dice - syntx_dice:.6f}")
print(f"Gap (ANTs - syntx best):    {ants_dice - max(syntx_dice, syntx_dice2, syntx_dice3):.6f}")
