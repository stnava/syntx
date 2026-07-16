import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

fixed_seg = ants.threshold_image(fi, 'Otsu', 3)
reg_ants_affine = ants.registration(fi, mi, 'Affine')

def get_dice(warped_image):
    warped_seg = ants.threshold_image(warped_image, 'Otsu', 3)
    overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
    return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])

# We will temporarily patch separable_gaussian_filter calls in syn.py to NOT pass spacing
# This makes it smooth by a constant number of voxels (e.g. 3.0 voxels) at all levels, matching ANTs default!
import syntx.syn
from syntx.syn import SyNTo

# Let's test a run by patching separable_gaussian_filter inside SyNTo.fit
# Actually, we can just modify syn.py directly to test if voxel-based smoothing achieves the 0.84 Dice score!
# Let's see: in syn.py line 1510-1511:
# grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, self.fluid_sigma, spacing=curr_spacing_fixed)
# If we change spacing=curr_spacing_fixed to spacing=None:
# Let's do that in syn.py!
