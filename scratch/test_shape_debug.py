import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))

fixed_image = torch.tensor(fi_cropped.numpy())[None, None]

from syntx.syn import get_physical_grid_torch

X_phys = get_physical_grid_torch(
    fixed_image.shape[2:], fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)

print("fixed_image shape:", fixed_image.shape)
print("X_phys shape:", X_phys.shape)
