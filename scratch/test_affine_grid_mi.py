import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'src')
import ants

fi = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), use_voxels=True)
mi = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), use_voxels=True)

# Identity matrix
T_grid = torch.tensor([[[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]]])
grid = F.affine_grid(T_grid, (1, 1, 64, 64), align_corners=True)

J_curr = torch.tensor(mi.numpy()).unsqueeze(0).unsqueeze(0)
moving_warped = F.grid_sample(J_curr, grid, padding_mode='border', align_corners=True)
warped_img = fi.new_image_like(moving_warped[0, 0].numpy())
print("F.affine_grid MI:", ants.image_mutual_information(fi, warped_img))
print("ANTs Identity MI:", ants.image_mutual_information(fi, mi))

# Try swapping grid components
grid_swapped = grid.flip(-1)
moving_warped2 = F.grid_sample(J_curr, grid_swapped, padding_mode='border', align_corners=True)
warped_img2 = fi.new_image_like(moving_warped2[0, 0].numpy())
print("F.affine_grid flipped MI:", ants.image_mutual_information(fi, warped_img2))

