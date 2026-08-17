import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import ants
from syntx.benchmark.data import load_mindboggle_pair

pair = load_mindboggle_pair(67, "examples/pairs.csv")
fi = pair["fixed"]
mi = pair["moving"]
fl = pair["fixed_label"]
ml = pair["moving_label"]

reg_ants = ants.registration(fixed=fi, moving=mi, typeof_transform="AffineFast")
warped_ants = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg_ants["fwdtransforms"])

# Create fixed grid coordinates
z_idx = np.arange(fi.shape[2])
y_idx = np.arange(fi.shape[1])
x_idx = np.arange(fi.shape[0])
mz, my, mx = np.meshgrid(z_idx, y_idx, x_idx, indexing='ij')

vox_fixed = np.stack([mx, my, mz], axis=-1) # shape (160, 256, 256, 3)

# Physical coordinates
sp_f = np.array(fi.spacing)
orig_f = np.array(fi.origin)
dir_f = np.array(fi.direction)

phys_f = orig_f + (vox_fixed * sp_f) @ dir_f.T

# Map points using ANTs apply_transforms_to_points (subsample for speed)
sub = phys_f[::4, ::4, ::4].reshape(-1, 3)
df_sub = pd.DataFrame({"x": sub[:, 0], "y": sub[:, 1], "z": sub[:, 2]})
pts_fwd = ants.apply_transforms_to_points(dim=3, points=df_sub, transformlist=reg_ants["fwdtransforms"])
pts_inv = ants.apply_transforms_to_points(dim=3, points=df_sub, transformlist=reg_ants["invtransforms"])

print("Subsample points:", sub.shape)
print("Points transformed with fwdtransforms:\n", pts_fwd.head(2))
print("Points transformed with invtransforms:\n", pts_inv.head(2))

# Map to moving voxel indices
orig_m = np.array(mi.origin)
sp_m = np.array(mi.spacing)
dir_m = np.array(mi.direction)

vox_fwd = (np.array(pts_fwd) - orig_m) @ np.linalg.inv(dir_m).T / sp_m
vox_inv = (np.array(pts_inv) - orig_m) @ np.linalg.inv(dir_m).T / sp_m

print("Moving voxel index using fwdtransforms:\n", np.round(vox_fwd[:2], 2))
print("Moving voxel index using invtransforms:\n", np.round(vox_inv[:2], 2))

# Check intensity correlation
mi_np = mi.numpy()
w_ants_np = warped_ants.numpy()

# Compare sampled voxel values
sub_z = mz[::4, ::4, ::4].reshape(-1)
sub_y = my[::4, ::4, ::4].reshape(-1)
sub_x = mx[::4, ::4, ::4].reshape(-1)

vals_ants_warped = w_ants_np[sub_x, sub_y, sub_z]

# Sample from moving using vox_fwd vs vox_inv
# vox is in [X, Y, Z] of moving
def sample_nearest(img_np, vox_coords):
    vx = np.clip(np.round(vox_coords[:, 0]).astype(int), 0, img_np.shape[0] - 1)
    vy = np.clip(np.round(vox_coords[:, 1]).astype(int), 0, img_np.shape[1] - 1)
    vz = np.clip(np.round(vox_coords[:, 2]).astype(int), 0, img_np.shape[2] - 1)
    valid = (vox_coords[:, 0] >= 0) & (vox_coords[:, 0] < img_np.shape[0]) & \
            (vox_coords[:, 1] >= 0) & (vox_coords[:, 1] < img_np.shape[1]) & \
            (vox_coords[:, 2] >= 0) & (vox_coords[:, 2] < img_np.shape[2])
    out = img_np[vx, vy, vz]
    out[~valid] = 0
    return out

vals_fwd = sample_nearest(mi_np, vox_fwd)
vals_inv = sample_nearest(mi_np, vox_inv)

print("Correlation using fwdtransforms:", np.corrcoef(vals_ants_warped, vals_fwd)[0, 1])
print("Correlation using invtransforms:", np.corrcoef(vals_ants_warped, vals_inv)[0, 1])
