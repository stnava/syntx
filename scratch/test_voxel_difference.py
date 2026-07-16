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

# 1. ANTs Warping
reg = ants.registration(fixed=fi_cropped, moving=mi, type_of_transform='Translation')
wrp_ants = ants.apply_transforms(
    fixed=fi_cropped, moving=mi_lbl, transformlist=reg['fwdtransforms'],
    interpolator='nearestNeighbor'
)
ants_np = wrp_ants.numpy().astype(np.int32)

# 2. PyTorch Warping
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

tx_ants = ants.read_transform(reg['fwdtransforms'][0])
t_ants = tx_ants.parameters[9:]
t_ants_tensor = torch.tensor(t_ants, dtype=torch.float32)
t_ants_zyx = torch.flip(t_ants_tensor, dims=[0])

y_phys = X_phys + t_ants_zyx
y_norm = physical_to_normalized_torch_corrected(
    y_phys, mi.shape, mi.spacing, mi.origin, mi.direction
)

moving_lbl_perm = moving_lbl.permute(0, 1, 4, 3, 2)
warped_plus = F.grid_sample(moving_lbl_perm, y_norm, mode='nearest', padding_mode='zeros', align_corners=True)
syntx_np = np.round(warped_plus[0, 0].numpy()).astype(np.int32)

# Find coordinates of non-zero elements in both
indices_ants = np.argwhere(ants_np > 0)
indices_syntx = np.argwhere(syntx_np > 0)

print("ANTs non-zero indices mean coordinate:", indices_ants.mean(axis=0))
print("Syntx non-zero indices mean coordinate:", indices_syntx.mean(axis=0))

print("\nANTs non-zero indices min coordinate:", indices_ants.min(axis=0))
print("Syntx non-zero indices min coordinate:", indices_syntx.min(axis=0))

print("\nANTs non-zero indices max coordinate:", indices_ants.max(axis=0))
print("Syntx non-zero indices max coordinate:", indices_syntx.max(axis=0))
