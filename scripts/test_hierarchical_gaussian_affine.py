#!/usr/bin/env python
"""
Hierarchical Gaussian Affine Solver with 2nd-98th Percentile Normalization
==========================================================================
"""

import os
import time
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.core.losses import mattes_mi_loss_core
from syntx.core.smoothing import separable_gaussian_filter
from syntx.robust_affine import _rodrigues_rotation_matrix_3d, compute_center_of_mass

def eval_overlap(fi, ml, fl, tx_path):
    w = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[tx_path], interpolator="nearestNeighbor")
    ov = ants.label_overlap_measures(fl, w)
    df = ov[~ov["Label"].astype(str).isin(["All", "0", "0.0"])]
    return float(np.mean(df["TotalOrTargetOverlap"].values))

def run_hierarchical_affine(pair_idx=67, device="cpu"):
    pair = load_mindboggle_pair(pair_idx, "examples/pairs.csv")
    fi = pair["fixed"]
    mi = pair["moving"]
    fl = pair["fixed_label"]
    ml = pair["moving_label"]
    
    t0 = time.time()
    device_obj = torch.device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 2nd-98th percentile intensity normalization with truncation
    fi_arr = torch.tensor(fi.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    mi_arr = torch.tensor(mi.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    
    f_pos = fi_arr[fi_arr > 0]
    m_pos = mi_arr[mi_arr > 0]
    f_p02, f_p98 = torch.quantile(f_pos, 0.02), torch.quantile(f_pos, 0.98)
    m_p02, m_p98 = torch.quantile(m_pos, 0.02), torch.quantile(m_pos, 0.98)
    fi_arr = torch.clamp((fi_arr - f_p02) / (f_p98 - f_p02 + 1e-6), 0.0, 1.0)
    mi_arr = torch.clamp((mi_arr - m_p02) / (m_p98 - m_p02 + 1e-6), 0.0, 1.0)
    
    # 2. Geometry
    com_f = compute_center_of_mass(fi, weighted=True)
    com_m = compute_center_of_mass(mi, weighted=True)
    t_init = np.array(com_m) - np.array(com_f)
    
    sp_xyz = torch.tensor(fi.spacing, dtype=torch.float32, device=device_obj)
    orig_xyz = torch.tensor(fi.origin, dtype=torch.float32, device=device_obj)
    dir_xyz = torch.tensor(fi.direction, dtype=torch.float32, device=device_obj)
    C_phys_xyz = torch.tensor(com_f, dtype=torch.float32, device=device_obj)
    
    mi_orig_xyz = torch.tensor(mi.origin, dtype=torch.float32, device=device_obj)
    mi_sp_xyz = torch.tensor(mi.spacing, dtype=torch.float32, device=device_obj)
    mi_dir_xyz = torch.tensor(mi.direction, dtype=torch.float32, device=device_obj)
    mi_shape_xyz = torch.tensor(mi.shape, dtype=torch.float32, device=device_obj)
    
    # 3. Gaussian Multi-Resolution Pyramid
    pyramid = [(4, 2.0), (2, 1.0), (1, 0.0)]
    fi_pyramid, mi_pyramid, coords_pyramid = {}, {}, {}
    for level, sigma in pyramid:
        if sigma > 0:
            fi_smooth = separable_gaussian_filter(fi_arr, sigma)
            mi_smooth = separable_gaussian_filter(mi_arr, sigma)
        else:
            fi_smooth, mi_smooth = fi_arr, mi_arr
            
        if level > 1:
            fi_lev = F.avg_pool3d(fi_smooth, kernel_size=level, stride=level)
            mi_lev = F.avg_pool3d(mi_smooth, kernel_size=level, stride=level)
        else:
            fi_lev, mi_lev = fi_smooth, mi_smooth
            
        shape_zyx = fi_lev.shape[2:]
        grid_z = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
        grid_y = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
        grid_x = torch.linspace(0, shape_zyx[2] - 1, shape_zyx[2], device=device_obj)
        mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
        vox_coords_xyz = torch.stack([mesh_x, mesh_y, mesh_z], dim=-1).reshape(-1, 3) * level
        phys_coords_xyz = orig_xyz + (vox_coords_xyz * sp_xyz) @ dir_xyz.t()
        
        fi_pyramid[level] = fi_lev
        mi_pyramid[level] = mi_lev
        coords_pyramid[level] = (phys_coords_xyz, shape_zyx)
        
    t_param = torch.tensor(t_init, dtype=torch.float32, device=device_obj, requires_grad=True)
    omega_param = torch.zeros(3, dtype=torch.float32, device=device_obj, requires_grad=True)
    scale_param = torch.zeros(3, dtype=torch.float32, device=device_obj, requires_grad=True)
    shear_param = torch.zeros(3, dtype=torch.float32, device=device_obj, requires_grad=True)
    
    # Staged Multi-Resolution Optimization: Translation -> Rigid -> Full Affine
    for level, sigma in pyramid:
        fi_lev = fi_pyramid[level]
        mi_lev = mi_pyramid[level]
        phys_coords_xyz, shape_zyx = coords_pyramid[level]
        
        if level == 4:
            stages = [
                ("Trans", 40, [t_param], [0.50]),
                ("Rigid", 40, [t_param, omega_param], [0.20, 0.02]),
                ("Affine", 40, [t_param, omega_param, scale_param, shear_param], [0.10, 0.01, 0.01, 0.01])
            ]
        elif level == 2:
            stages = [
                ("Rigid", 30, [t_param, omega_param], [0.10, 0.01]),
                ("Affine", 30, [t_param, omega_param, scale_param, shear_param], [0.05, 0.005, 0.005, 0.005])
            ]
        else:
            stages = [
                ("Affine", 25, [t_param, omega_param, scale_param, shear_param], [0.02, 0.002, 0.002, 0.002])
            ]
            
        for stage_name, iters, param_list, lr_list in stages:
            opt = torch.optim.Adam([{'params': [p], 'lr': lr} for p, lr in zip(param_list, lr_list)])
            for it in range(iters):
                opt.zero_grad()
                R = _rodrigues_rotation_matrix_3d(omega_param)
                S = torch.diag(torch.exp(torch.clamp(scale_param, -0.5, 0.5)))
                Sh = torch.eye(3, device=device_obj)
                Sh[0, 1] = shear_param[0]
                Sh[0, 2] = shear_param[1]
                Sh[1, 2] = shear_param[2]
                A = R @ S @ Sh
                
                t_eff = t_param + C_phys_xyz - A @ C_phys_xyz
                y_phys_xyz = phys_coords_xyz @ A.t() + t_eff
                y_vox_xyz = (y_phys_xyz - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
                y_norm_xyz = 2.0 * (y_vox_xyz / (mi_shape_xyz - 1.0)) - 1.0
                
                sampling_grid = y_norm_xyz.reshape(1, *shape_zyx, 3)
                warped = F.grid_sample(mi_lev, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                
                fg_mask = (fi_lev > 0.01) & (warped > 0.01)
                loss = mattes_mi_loss_core(warped.reshape(-1), fi_lev.reshape(-1), mask=fg_mask.reshape(-1), num_bins=32, min_val=0.0, max_val=1.0, sampling_percentage=0.30)
                loss.backward()
                opt.step()
                
                with torch.no_grad():
                    scale_param.clamp_(-0.5, 0.5)
                    shear_param.clamp_(-0.5, 0.5)
                    omega_param.clamp_(-np.pi/4, np.pi/4)
                    
    with torch.no_grad():
        R_fin = _rodrigues_rotation_matrix_3d(omega_param).cpu().numpy()
        S_fin = np.diag(np.exp(scale_param.cpu().numpy()))
        Sh_fin = np.eye(3)
        Sh_fin[0, 1] = shear_param[0].item()
        Sh_fin[0, 2] = shear_param[1].item()
        Sh_fin[1, 2] = shear_param[2].item()
        A_final = R_fin @ S_fin @ Sh_fin
        t_final = t_param.cpu().numpy()
        
    t_itk = t_final + com_f - A_final @ com_f
    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
    tx.set_parameters(np.concatenate([A_final.flatten(), t_itk]))
    tx.set_fixed_parameters(com_f)
    
    tmp = tempfile.mktemp(suffix='.mat')
    ants.write_transform(tx, tmp)
    ov = eval_overlap(fi, ml, fl, tmp)
    dt = time.time() - t0
    
    print(f"Hierarchical Gaussian Affine Overlap: {ov:.4f} ({dt:.1f}s)")
    print(f"Final Translation: {np.round(t_final, 2)}")
    print(f"Final Matrix Scale: {np.round(np.diag(S_fin), 3)}")
    return ov, tmp

if __name__ == "__main__":
    run_hierarchical_affine(67)
