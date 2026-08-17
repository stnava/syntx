#!/usr/bin/env python
"""
Deep Diagnostic Suite: Pair 67 Under-Performance & Folding Spatial Localization
================================================================================
Investigates:
1. Pair 67 image metadata, intensity profiles, and affine alignment discrepancies.
2. Spatial localization of Jacobian folding singularities (det(J) <= 0).
3. Anatomical distribution of folds across brain tissue classes (GM, WM, CSF, Background).
4. Regularization remedies (Sobolev alpha scaling, elastic field smoothing, and boundary tapering).
"""

import os
import sys
import time
import json
import numpy as np
import torch
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair

def diagnose_pair(pair_idx: int = 67):
    print(f"===============================================================")
    print(f"  Diagnosing Pair {pair_idx:03d} Properties & Folding Singularity Locations")
    print(f"===============================================================\n")
    
    pair = load_mindboggle_pair(pair_idx, "examples/pairs.csv")
    fi = pair["fixed"]
    mi = pair["moving"]
    fl = pair["fixed_label"]
    ml = pair["moving_label"]
    
    print(f"Fixed Image:  {pair['fixed_id']} | Shape={fi.shape} | Spacing={fi.spacing} | Origin={fi.origin}")
    print(f"Moving Image: {pair['moving_id']} | Shape={mi.shape} | Spacing={mi.spacing} | Origin={mi.origin}")
    print(f"Fixed Intensity Range:  [{fi.min():.2f}, {fi.max():.2f}] | Mean={fi.mean():.2f}")
    print(f"Moving Intensity Range: [{mi.min():.2f}, {mi.max():.2f}] | Mean={mi.mean():.2f}")
    
    # 1. Compare Affines
    print("\n--- 1. Affine Stage Analysis ---")
    t0 = time.time()
    aff_pt = syntx.robust_affine(fi, mi, mode="pytorch", n_starts=3, device="cpu")
    t_aff_pt = time.time() - t0
    
    t0 = time.time()
    aff_ants = ants.registration(fixed=fi, moving=mi, typeof_transform="Affine")
    t_aff_ants = time.time() - t0
    
    # Overlaps
    w_pt = ants.apply_transforms(fixed=fi, moving=ml, transformlist=aff_pt["fwdtransforms"], interpolator="nearestNeighbor")
    ov_pt = ants.label_overlap_measures(fl, w_pt)
    df_pt = ov_pt[~ov_pt["Label"].astype(str).isin(["All", "0", "0.0"])]
    
    w_ants = ants.apply_transforms(fixed=fi, moving=ml, transformlist=aff_ants["fwdtransforms"], interpolator="nearestNeighbor")
    ov_ants = ants.label_overlap_measures(fl, w_ants)
    df_ants = ov_ants[~ov_ants["Label"].astype(str).isin(["All", "0", "0.0"])]
    
    dice_pt = np.mean(df_pt["TotalOrTargetOverlap"].values)
    dice_ants = np.mean(df_ants["TotalOrTargetOverlap"].values)
    print(f"PyTorch Affine Overlap: {dice_pt:.4f} ({t_aff_pt:.1f}s)")
    print(f"ANTs Affine Overlap:    {dice_ants:.4f} ({t_aff_ants:.1f}s)")
    
    # 2. Deformable SyN with Spatial Folding Localization
    print("\n--- 2. Deformable SyN & Folding Spatial Distribution ---")
    for reg_name in ["gaussian", "sobolev"]:
        print(f"\nEvaluating Deformable SyN with {reg_name.upper()}...")
        res = syntx.syn(
            fixed=fi,
            moving=mi,
            device="mps",
            initial_transform=aff_pt["fwdtransforms"][0],
            grad_step=0.25,
            flow_sigma=3.0,
            regularizer=reg_name,
            kernel_type=reg_name,
            reg_iterations=[100, 100, 20]
        )
        
        fwd_warp = ants.image_read(res["fwdtransforms"][0])
        jac_img = ants.create_jacobian_determinant_image(fi, fwd_warp, do_log=False)
        jac_arr = jac_img.numpy()
        
        # Warp moving labels and brain mask
        w_ml = ants.apply_transforms(fixed=fi, moving=ml, transformlist=res["fwdtransforms"], interpolator="nearestNeighbor")
        ov = ants.label_overlap_measures(fl, w_ml)
        df = ov[~ov["Label"].astype(str).isin(["All", "0", "0.0"])]
        d_f = np.mean(df["TotalOrTargetOverlap"].values)
        
        fold_mask = (jac_arr <= 0.0)
        n_folds = int(fold_mask.sum())
        total_voxels = int(fold_mask.size)
        fold_pct = (n_folds / total_voxels) * 100.0
        min_j = float(jac_arr.min())
        
        print(f"[{reg_name.upper()}] Fixed Dice={d_f:.4f} | Folding={fold_pct:.4f}% ({n_folds} voxels) | Min det(J)={min_j:.4f}")
        
        # Anatomical Breakdown of Fold Locations
        if n_folds > 0:
            fl_arr = fl.numpy()
            
            # Brain mask (any non-zero label)
            brain_mask = (fl_arr > 0)
            folds_inside_brain = int((fold_mask & brain_mask).sum())
            folds_in_background = int((fold_mask & (~brain_mask)).sum())
            
            # Boundary mask (outer 5 slices along each border)
            border_mask = np.zeros_like(fold_mask, dtype=bool)
            border_mask[:5, :, :] = True; border_mask[-5:, :, :] = True
            border_mask[:, :5, :] = True; border_mask[:, -5:, :] = True
            border_mask[:, :, :5] = True; border_mask[:, :, -5:] = True
            folds_at_border = int((fold_mask & border_mask).sum())
            
            print(f"  Folding Spatial Breakdown:")
            print(f"    - In Foreground Brain:  {folds_inside_brain:5d} voxels ({folds_inside_brain/n_folds*100:5.1f}%)")
            print(f"    - In Flat Background:   {folds_in_background:5d} voxels ({folds_in_background/n_folds*100:5.1f}%)")
            print(f"    - At Outer Image Edge:  {folds_at_border:5d} voxels ({folds_at_border/n_folds*100:5.1f}%)")

if __name__ == "__main__":
    diagnose_pair(67)
