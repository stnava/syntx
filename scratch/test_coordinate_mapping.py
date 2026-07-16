import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))

fixed_image = torch.tensor(fi_cropped.numpy())[None, None]

from syntx.syn import get_physical_grid_torch

X_phys = get_physical_grid_torch(
    fixed_image.shape[2:], fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)

# Pick a few points and compare
indices = [
    (0, 0, 0),
    (100, 80, 80),
    (205, 160, 159)
]

for idx in indices:
    p_ants = ants.transform_index_to_physical_point(fi_cropped, idx)
    # X_phys is in ZYX order, so we need to index it in ZYX order!
    # Let's print values at idx and idx reversed
    p_syntx_1 = X_phys[0, idx[0], idx[1], idx[2]].numpy()
    p_syntx_2 = X_phys[0, idx[2], idx[1], idx[0]].numpy()
    print(f"\nVoxel Index (XYZ): {idx}")
    print(f"ANTs Physical Point (XYZ): {p_ants}")
    print(f"Syntx X_phys at index {idx} (assuming ZYX): {p_syntx_1}")
    print(f"Syntx X_phys at reversed index {idx[::-1]} (assuming XYZ): {p_syntx_2}")
