#!/usr/bin/env python
"""
Affine Coordinate Mapping and Optimizer Parity Diagnostic
=========================================================
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.core.losses import mattes_mi_loss_core
from syntx.robust_affine import _rodrigues_rotation_matrix_3d, compute_center_of_mass

def main():
    pair = load_mindboggle_pair(67, "examples/pairs.csv")
    fi = pair["fixed"]
    mi = pair["moving"]
    fl = pair["fixed_label"]
    ml = pair["moving_label"]
    
    print("=== Analyzing Affine Initializer on Pair 067 ===")
    
    # 1. Test ANTs AffineFast
    reg_ants = ants.registration(fixed=fi, moving=mi, typeof_transform="AffineFast")
    w_ants_l = ants.apply_transforms(fixed=fi, moving=ml, transformlist=reg_ants["fwdtransforms"], interpolator="nearestNeighbor")
    ov_ants = ants.label_overlap_measures(fl, w_ants_l)
    df_a = ov_ants[~ov_ants["Label"].astype(str).isin(["All", "0", "0.0"])]
    print(f"ANTs AffineFast Overlap: {np.mean(df_a['TotalOrTargetOverlap'].values):.4f}")
    
    # 2. Extract ANTs affine matrix & translation via point transforms
    pts = pd.DataFrame({
        "x": [0.0, 1.0, 0.0, 0.0],
        "y": [0.0, 0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 0.0, 1.0]
    })
    w_pts = ants.apply_transforms_to_points(dim=3, points=pts, transformlist=reg_ants["fwdtransforms"])
    
    p0 = np.array(w_pts.iloc[0])
    p1 = np.array(w_pts.iloc[1]) - p0
    p2 = np.array(w_pts.iloc[2]) - p0
    p3 = np.array(w_pts.iloc[3]) - p0
    
    # Forward map: y = x @ A_ants.T + t_ants
    A_ants = np.column_stack([p1, p2, p3])
    t_ants = p0
    print("\nExtracted ANTs Matrix A:\n", np.round(A_ants, 4))
    print("Extracted ANTs Trans t:\n", np.round(t_ants, 4))
    
    # 3. Test if PyTorch Grid Sample using A_ants and t_ants reproduces ANTs warped image
    device_obj = torch.device("cpu")
    fi_arr = torch.tensor(fi.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    mi_arr = torch.tensor(mi.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    
    sp_xyz = torch.tensor(fi.spacing, dtype=torch.float32, device=device_obj)
    orig_xyz = torch.tensor(fi.origin, dtype=torch.float32, device=device_obj)
    dir_xyz = torch.tensor(fi.direction, dtype=torch.float32, device=device_obj)
    
    mi_orig_xyz = torch.tensor(mi.origin, dtype=torch.float32, device=device_obj)
    mi_sp_xyz = torch.tensor(mi.spacing, dtype=torch.float32, device=device_obj)
    mi_dir_xyz = torch.tensor(mi.direction, dtype=torch.float32, device=device_obj)
    mi_shape_xyz = torch.tensor(mi.shape, dtype=torch.float32, device=device_obj)
    
    shape_zyx = fi_arr.shape[2:]
    grid_z = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
    grid_y = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
    grid_x = torch.linspace(0, shape_zyx[2] - 1, shape_zyx[2], device=device_obj)
    mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
    vox_coords_xyz = torch.stack([mesh_x, mesh_y, mesh_z], dim=-1).reshape(-1, 3)
    phys_coords_xyz = orig_xyz + (vox_coords_xyz * sp_xyz) @ dir_xyz.t()
    
    A_t = torch.tensor(A_ants, dtype=torch.float32, device=device_obj)
    t_t = torch.tensor(t_ants, dtype=torch.float32, device=device_obj)
    
    y_phys_xyz = phys_coords_xyz @ A_t.t() + t_t
    y_vox_xyz = (y_phys_xyz - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
    y_norm_xyz = 2.0 * (y_vox_xyz / (mi_shape_xyz - 1.0)) - 1.0
    
    sampling_grid = y_norm_xyz.reshape(1, *shape_zyx, 3)
    warped_pt = F.grid_sample(mi_arr, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    warped_ants = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg_ants["fwdtransforms"])
    warped_ants_arr = torch.tensor(warped_ants.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    
    diff = torch.abs(warped_pt - warped_ants_arr).mean().item()
    corr = np.corrcoef(warped_pt.flatten().numpy(), warped_ants_arr.flatten().numpy())[0, 1]
    print(f"\nPyTorch Grid Sample Parity with ANTs Apply Transforms:")
    print(f"  Mean Absolute Difference: {diff:.4f}")
    print(f"  Pearson Correlation:      {corr:.4f}")

if __name__ == "__main__":
    main()
