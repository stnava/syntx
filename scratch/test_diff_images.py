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

# 2. PyTorch Warping with corrected get_physical_grid and physical_to_normalized
fixed_image = torch.tensor(fi_cropped.numpy())[None, None]
moving_lbl = torch.tensor(mi_lbl.numpy())[None, None].to(torch.float32)

def _get_physical_grid_torch_yfirst_corrected(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    # Stack list(reversed(meshgrid)) to match ZYX physical coordinate order!
    idxs = torch.stack(list(reversed(meshgrid)), dim=-1)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)

def get_physical_grid_torch_corrected(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _get_physical_grid_torch_yfirst_corrected(shape, spacing_rev, origin_rev, direction_rev, device, dtype)

from syntx.syn import _physical_to_normalized_torch_yfirst

def physical_to_normalized_torch_corrected(phys_coords, target_shape, spacing, origin, direction):
    target_shape_rev = tuple(reversed(target_shape))
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape_rev, spacing_rev, origin_rev, direction_rev)

X_phys = get_physical_grid_torch_corrected(
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
warped_lbl_torch = F.grid_sample(moving_lbl_perm, y_norm, mode='nearest', padding_mode='zeros', align_corners=True)
warped_lbl_np = np.round(warped_lbl_torch[0, 0].numpy()).astype(np.int32)
warped_lbl_ants = fi_cropped.new_image_like(warped_lbl_np)

syntx_np = warped_lbl_ants.numpy().astype(np.int32)

print("\nANTs shape:", ants_np.shape)
print("Syntx shape:", syntx_np.shape)

print("ANTs non-zero voxels:", (ants_np > 0).sum())
print("Syntx non-zero voxels:", (syntx_np > 0).sum())

overlap = ((ants_np > 0) & (syntx_np > 0)).sum()
print("Overlap between ANTs and Syntx warped images:", overlap)

max_diff = np.abs(ants_np - syntx_np).max()
print("Max absolute difference:", max_diff)

if np.array_equal(ants_np, syntx_np):
    print("THEY ARE IDENTICAL!")
else:
    print("THEY ARE DIFFERENT!")
    # Print indices of some mismatches if any
    mismatches = np.argwhere(ants_np != syntx_np)
    print("Number of mismatched voxels:", len(mismatches))
    print("Sample mismatches (index, ANTs label, Syntx label):")
    for idx in mismatches[:10]:
        print(f"Index {idx}: ANTs={ants_np[idx[0], idx[1], idx[2]]}, Syntx={syntx_np[idx[0], idx[1], idx[2]]}")
