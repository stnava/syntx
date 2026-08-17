#!/usr/bin/env python
"""
Testing Percentile Intensity Normalization on Pair 67 Affine
============================================================
"""

import sys
import os
import time
import math
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.core.losses import mattes_mi_loss_core, b_spline_3
from syntx.robust_affine import compute_center_of_mass, _rodrigues_rotation_matrix_3d

def eval_overlap(fi, mi_label, fl_label, tx_path):
    w = ants.apply_transforms(fixed=fi, moving=mi_label, transformlist=[tx_path], interpolator="nearestNeighbor")
    ov = ants.label_overlap_measures(fl_label, w)
    df = ov[~ov["Label"].astype(str).isin(["All", "0", "0.0"])]
    col = "TotalOrTargetOverlap" if "TotalOrTargetOverlap" in df.columns else "TargetOverlap"
    return float(np.mean(df[col].values))

def run_affine_with_norm(
    fixed, moving,
    norm_method="percentile_2_98",
    hierarchical=True,
    n_starts=1,
    pyramid_levels=[4, 2, 1],
    n_iters_per_level=[80, 60, 30],
    device="cpu"
):
    t0 = time.time()
    device_obj = torch.device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. COM
    com_f = compute_center_of_mass(fixed, weighted=True)
    com_m = compute_center_of_mass(moving, weighted=True)
    t_init = np.array(com_m) - np.array(com_f)
    
    # 2. Convert to PyTorch Tensors
    fi_arr = torch.tensor(fixed.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    mi_arr = torch.tensor(moving.numpy().transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    
    # Intensity Normalization Modes
    if norm_method == "min_max":
        fi_arr = (fi_arr - fi_arr.min()) / (fi_arr.max() - fi_arr.min() + 1e-6)
        mi_arr = (mi_arr - mi_arr.min()) / (mi_arr.max() - mi_arr.min() + 1e-6)
    elif norm_method == "percentile_2_98":
        # Foreground percentiles (positive voxels)
        f_pos = fi_arr[fi_arr > 0]
        m_pos = mi_arr[mi_arr > 0]
        f_p02, f_p98 = torch.quantile(f_pos, 0.02), torch.quantile(f_pos, 0.98)
        m_p02, m_p98 = torch.quantile(m_pos, 0.02), torch.quantile(m_pos, 0.98)
        fi_arr = torch.clamp((fi_arr - f_p02) / (f_p98 - f_p02 + 1e-6), 0.0, 1.0)
        mi_arr = torch.clamp((mi_arr - m_p02) / (m_p98 - m_p02 + 1e-6), 0.0, 1.0)
    elif norm_method == "percentile_1_99":
        f_pos = fi_arr[fi_arr > 0]
        m_pos = mi_arr[mi_arr > 0]
        f_p01, f_p99 = torch.quantile(f_pos, 0.01), torch.quantile(f_pos, 0.99)
        m_p01, m_p99 = torch.quantile(m_pos, 0.01), torch.quantile(m_pos, 0.99)
        fi_arr = torch.clamp((fi_arr - f_p01) / (f_p99 - f_p01 + 1e-6), 0.0, 1.0)
        mi_arr = torch.clamp((mi_arr - m_p01) / (m_p99 - m_p01 + 1e-6), 0.0, 1.0)
        
    sp_xyz = torch.tensor(fixed.spacing, dtype=torch.float32, device=device_obj)
    orig_xyz = torch.tensor(fixed.origin, dtype=torch.float32, device=device_obj)
    dir_xyz = torch.tensor(fixed.direction, dtype=torch.float32, device=device_obj)
    C_phys_xyz = torch.tensor(com_f, dtype=torch.float32, device=device_obj)
    
    mi_orig_xyz = torch.tensor(moving.origin, dtype=torch.float32, device=device_obj)
    mi_sp_xyz = torch.tensor(moving.spacing, dtype=torch.float32, device=device_obj)
    mi_dir_xyz = torch.tensor(moving.direction, dtype=torch.float32, device=device_obj)
    mi_shape_xyz = torch.tensor(moving.shape, dtype=torch.float32, device=device_obj)
    
    fi_pyramid, mi_pyramid, coords_pyramid = {}, {}, {}
    for level in pyramid_levels:
        if level > 1:
            fi_lev = F.avg_pool3d(fi_arr, kernel_size=level, stride=level)
            mi_lev = F.avg_pool3d(mi_arr, kernel_size=level, stride=level)
        else:
            fi_lev, mi_lev = fi_arr, mi_arr
            
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
    
    for lev_idx, level in enumerate(pyramid_levels):
        fi_lev = fi_pyramid[level]
        mi_lev = mi_pyramid[level]
        phys_coords_xyz, shape_zyx = coords_pyramid[level]
        total_iters = n_iters_per_level[lev_idx]
        
        if hierarchical:
            # Stage 1: Translation (30%)
            # Stage 2: Rigid (35%)
            # Stage 3: Affine (35%)
            iters_trans = int(total_iters * 0.30)
            iters_rigid = int(total_iters * 0.35)
            iters_affine = total_iters - iters_trans - iters_rigid
            stages = [
                ("Trans", iters_trans, [t_param], [0.08]),
                ("Rigid", iters_rigid, [t_param, omega_param], [0.04, 0.015]),
                ("Affine", iters_affine, [t_param, omega_param, scale_param, shear_param], [0.02, 0.008, 0.005, 0.005])
            ]
        else:
            stages = [("Affine", total_iters, [t_param, omega_param, scale_param, shear_param], [0.05, 0.01, 0.005, 0.005])]
            
        for stage_name, iters, param_list, lr_list in stages:
            if iters <= 0:
                continue
            optimizer = torch.optim.Adam([{'params': [p], 'lr': lr} for p, lr in zip(param_list, lr_list)])
            
            for it in range(iters):
                optimizer.zero_grad()
                
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
                
                # Foreground mask
                fg_mask = (fi_lev > 0.01) & (warped > 0.01)
                mask_flat = fg_mask.reshape(-1)
                
                loss = mattes_mi_loss_core(warped.reshape(-1), fi_lev.reshape(-1), mask=mask_flat, num_bins=32, min_val=0.0, max_val=1.0, sampling_percentage=0.30)
                loss.backward()
                optimizer.step()
                
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
    
    td = tempfile.mkdtemp(prefix="norm_aff_")
    tx_path = os.path.join(td, "affine.mat")
    ants.write_transform(tx, tx_path)
    dt = time.time() - t0
    return tx_path, dt

def main():
    pair = load_mindboggle_pair(67, "examples/pairs.csv")
    fi = pair["fixed"]
    mi = pair["moving"]
    fl = pair["fixed_label"]
    ml = pair["moving_label"]
    
    print(f"=== Testing Intensity Normalization on Pair 067 ({pair['fixed_id']} -> {pair['moving_id']}) ===")
    
    # 1. ANTs Baseline
    aff_ants = ants.registration(fixed=fi, moving=mi, typeof_transform="Affine")
    ov_ants = eval_overlap(fi, ml, fl, aff_ants["fwdtransforms"][0])
    print(f">> ANTs C++ Affine Baseline Overlap:       {ov_ants:.4f}")
    
    # 2. Min-Max (Original)
    tx_mm, dt_mm = run_affine_with_norm(fi, mi, norm_method="min_max")
    ov_mm = eval_overlap(fi, ml, fl, tx_mm)
    print(f">> Mode 'min_max' (Original):              {ov_mm:.4f} ({dt_mm:.1f}s)")
    
    # 3. 2nd-98th Percentile Normalization
    tx_p298, dt_p298 = run_affine_with_norm(fi, mi, norm_method="percentile_2_98")
    ov_p298 = eval_overlap(fi, ml, fl, tx_p298)
    print(f">> Mode 'percentile_2_98' (Hierarchical):  {ov_p298:.4f} ({dt_p298:.1f}s)")
    
    # 4. 1st-99th Percentile Normalization
    tx_p199, dt_p199 = run_affine_with_norm(fi, mi, norm_method="percentile_1_99")
    ov_p199 = eval_overlap(fi, ml, fl, tx_p199)
    print(f">> Mode 'percentile_1_99' (Hierarchical):  {ov_p199:.4f} ({dt_p199:.1f}s)")

if __name__ == "__main__":
    main()
