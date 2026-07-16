#!/usr/bin/env python
"""Quick verification that updated defaults achieve parity."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import ants
import numpy as np

def compute_tissue_overlap(fixed_img, warped_img, n_classes=3):
    fixed_seg = ants.threshold_image(fixed_img, 'Otsu', n_classes)
    warped_seg = ants.threshold_image(warped_img, 'Otsu', n_classes)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    if 'MeanOverlap' in overlap.columns:
        return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
    return 0.0

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

print("PARITY VERIFICATION (with updated defaults)")
print("=" * 60)

# ANTs baselines
for ttype in ['Affine', 'SyN', 'SyNCC']:
    reg = ants.registration(fi, mi, type_of_transform=ttype)
    dice = compute_tissue_overlap(fi, reg['warpedmovout'])
    print(f"ANTs {ttype:10s}: Dice={dice:.4f}")

# syntx with NEW defaults
from syntx.syn import registration
for ttype in ['Affine', 'SyN']:
    reg = registration(fi, mi, type_of_transform=ttype, backend='pytorch')
    dice = compute_tissue_overlap(fi, reg['warpedmovout'])
    n_syn = len(reg.get('syn_losses', []))
    n_aff = len(reg.get('affine_losses', []))
    print(f"syntx {ttype:10s}: Dice={dice:.4f} (aff_iters={n_aff}, syn_iters={n_syn})")

print("\n✓ If syntx SyN Dice ≥ ANTs SyNCC Dice, parity is achieved!")
