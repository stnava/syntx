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

from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch

X_phys = get_physical_grid_torch(
    fixed_image.shape[2:], fi_cropped.spacing, fi_cropped.origin, fi_cropped.direction
)

com_fi = np.array(ants.get_center_of_mass(fi_cropped))
com_mi = np.array(ants.get_center_of_mass(mi))
best_t = com_mi - com_fi
best_t_zyx = torch.tensor(best_t[::-1].copy(), dtype=torch.float32)

y_phys = X_phys + best_t_zyx
y_norm = physical_to_normalized_torch(
    y_phys, mi.shape, mi.spacing, mi.origin, mi.direction
)

moving_lbl_perm = moving_lbl.permute(0, 1, 4, 3, 2)
warped_lbl_torch = F.grid_sample(moving_lbl_perm, y_norm, mode='nearest', padding_mode='border', align_corners=True)

# Convert to int32
warped_lbl_np = np.round(warped_lbl_torch[0, 0].numpy()).astype(np.int32)
warped_lbl_ants = fi_cropped.new_image_like(warped_lbl_np)

fi_lbl_cropped_np = fi_lbl_cropped.numpy().astype(np.int32)
fi_lbl_cropped_ants = fi_cropped.new_image_like(fi_lbl_cropped_np)

df = ants.label_overlap_measures(fi_lbl_cropped_ants, warped_lbl_ants)
target_col = 'TargetOverlap' if 'TargetOverlap' in df.columns else 'TotalOrTargetOverlap'

# Filter out All, 0, and large/inf values
df_valid = df[(df['Label'] != 0) & (df['Label'] != 'All')]
df_valid = df_valid[df_valid[target_col] < 1.0e10]

print("Valid labels count:", len(df_valid))
print("Physical Init-only DICE (non-inf):", df_valid[target_col].mean())
