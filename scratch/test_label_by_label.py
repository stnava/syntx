import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np
import torch
import torch.nn.functional as F

base_path = '/Users/stnava/data/mindboggle/volumes'
fi = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/t1weighted_brain.nii.gz")
fi_lbl = ants.image_read(f"{base_path}/OASIS-TRT-20_volumes/OASIS-TRT-20-1/labels.DKT31.manual.nii.gz")
mi = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/t1weighted_brain.nii.gz")
mi_lbl = ants.image_read(f"{base_path}/MMRR-21_volumes/MMRR-21-1/labels.DKT31.manual.nii.gz")

fi_cropped = ants.crop_image(fi, ants.iMath(ants.get_mask(fi), "MD", 12))
fi_lbl_cropped = ants.crop_image(fi_lbl, ants.iMath(ants.get_mask(fi), "MD", 12))

fixed_image = torch.tensor(fi_cropped.numpy())[None, None]
moving_lbl = torch.tensor(mi_lbl.numpy())[None, None].to(torch.float32)

from syntx.syn import get_physical_grid_torch, _physical_to_normalized_torch_yfirst

def physical_to_normalized_torch_corrected(phys_coords, target_shape, spacing, origin, direction):
    target_shape_rev = tuple(reversed(target_shape))
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape_rev, spacing_rev, origin_rev, direction_rev)

X_phys = get_physical_grid_torch(
    fixed_image.shape[2:], fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)

best_t = np.array([102.66337585, 123.56343842, 101.43486786])
best_t_tensor = torch.tensor(best_t, dtype=torch.float32)
best_t_zyx = torch.flip(best_t_tensor, dims=[0])

y_phys = X_phys + best_t_zyx
y_norm = physical_to_normalized_torch_corrected(
    y_phys, mi.shape, mi.spacing, mi.origin, mi.direction
)

moving_lbl_perm = moving_lbl.permute(0, 1, 4, 3, 2)
warped_lbl_torch = F.grid_sample(moving_lbl_perm, y_norm, mode='nearest', padding_mode='border', align_corners=True)

warped_lbl_np = np.round(warped_lbl_torch[0, 0].numpy()).astype(np.int32)
fi_lbl_cropped_np = fi_lbl_cropped.numpy().astype(np.int32)

# Print non-zero unique labels
print("Unique labels in warped:", np.unique(warped_lbl_np))
print("Unique labels in fixed:", np.unique(fi_lbl_cropped_np))

# Check overlap label-by-label
for val in np.unique(fi_lbl_cropped_np):
    if val == 0: continue
    fixed_mask = (fi_lbl_cropped_np == val)
    warped_mask = (warped_lbl_np == val)
    intersection = (fixed_mask & warped_mask).sum()
    dice = 2.0 * intersection / (fixed_mask.sum() + warped_mask.sum()) if (fixed_mask.sum() + warped_mask.sum()) > 0 else 0.0
    if dice > 0.0:
        print(f"Label {val}: fixed_count={fixed_mask.sum()}, warped_count={warped_mask.sum()}, intersection={intersection}, dice={dice:.4f}")
