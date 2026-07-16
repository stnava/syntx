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

# We will patch syntx.syn to use min-max scaling to [0, 1] instead of mean-std normalization!
import syntx.syn

original_syn = syntx.syn

def patched_syn(*args, **kwargs):
    # We will temporarily override the numpy normalization inside syn
    # In order to do that, we can temporarily patch fixed.numpy() and moving.numpy() to return scaled versions
    fixed = kwargs.get('fixed') if 'fixed' in kwargs else args[0]
    moving = kwargs.get('moving') if 'moving' in kwargs else args[1]
    
    # We want fi_norm = fi_np / fi_np.max()
    # But inside syn.py, it does:
    # fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    # If we want the result to be fi_np / fi_np.max(), we can pass a fake image whose mean and std are such that:
    # (x - mean) / std = x / max
    # This is: x / std - mean/std = x/max.
    # So we need mean = 0, and std = max!
    # So we can wrap the numpy() method of the image to return an array scaled by its std/max!
    # Or even simpler: we can just modify the code in src/syntx/syn.py directly to test!
    pass

# Let's just modify syn.py lines 1932-1933 directly to use [0, 1] scaling!
# Let's view the lines first to make sure we replace the correct target content.
