#!/usr/bin/env python3
"""
Milestone 4: Multi-Modal SyNTo Deep Similarity Metric Benchmark Suite

Evaluates deep similarity metrics (`dino_2_lncc`, `vgg_4_lncc`) against standard intensity
metrics (`lncc`, `mattes_mi`) on cross-modality T1 <-> T2 brain registrations with non-linear
intensity inversions.

Adheres strictly to GEMINI.md guidelines:
1. Single Interpolation Policy: composition of transforms in a single call, no file-based pre-warping.
2. VGG 3D Mode Requirement: vgg_mode='lncc_3d', vgg_layers=[4].
3. Modality Simulation: multi-level piecewise intensity shuffling (swapping [0.0, 0.6, 1.0] non-linearly).
4. Label map evaluation using interpolator='nearestNeighbor' and ants.label_overlap_measures.
5. Standardized image_compare returns (lower is better).
6. 0% folding rate (J > 0).
"""

import os
import sys
import tempfile
import time
import ants
import numpy as np
import torch
import pandas as pd

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from syntx import registration, image_compare


def create_piecewise_intensity_shuffled_brain(img: ants.ANTsImage) -> ants.ANTsImage:
    """
    Simulates cross-modality T1 <-> T2 brain differences using multi-level
    piecewise intensity shuffling as specified in GEMINI.md Rule 7:
    Swaps intensity ranges [0.0, 0.6, 1.0] non-linearly to create massive contrast inversions.
    """
    v = img.numpy()
    v_min, v_max = float(v.min()), float(v.max())
    norm_v = (v - v_min) / (v_max - v_min + 1e-8)
    
    # Piecewise non-linear inversion: [0, 0.6] -> [1.0, 0.0], [0.6, 1.0] -> [0.0, 0.6]
    shuffled = np.where(
        norm_v < 0.6,
        1.0 - (1.0 / 0.6) * norm_v,
        0.0 + (0.6 / 0.4) * (norm_v - 0.6)
    )
    shuffled = np.clip(shuffled, 0.0, 1.0) * (v_max - v_min) + v_min
    
    return ants.from_numpy(
        shuffled.astype(np.float32),
        origin=img.origin,
        spacing=img.spacing,
        direction=img.direction
    )


def create_2d_cross_modality_pair():
    """Generates 2D cross-modality T1 <-> T2 brain image pair with foreground label map."""
    r16_path = ants.get_ants_data('r16')
    fixed_img = ants.image_read(r16_path)
    
    v = fixed_img.numpy()
    seg = np.zeros_like(v, dtype=np.int32)
    seg[v > 40] = 1
    seg[v > 90] = 2
    seg[v > 140] = 3
    fixed_label = ants.from_numpy(
        seg, origin=fixed_img.origin, spacing=fixed_img.spacing, direction=fixed_img.direction
    )
    
    # Generate intensity shuffled moving volume (T1 -> T2 inversion)
    moving_t2_raw = create_piecewise_intensity_shuffled_brain(fixed_img)
    
    # Apply synthetic spatial deformation + rigid translation to moving image & label map
    tx_init = ants.create_ants_transform(transform_type='AffineTransform', dimension=2)
    tx_init.set_parameters(np.array([1.0, 0.0, 0.0, 1.0, 5.0, -4.0]))
    tx_file = tempfile.mktemp(suffix='.mat')
    ants.write_transform(tx_init, tx_file)
    
    moving_img = ants.apply_transforms(fixed=fixed_img, moving=moving_t2_raw, transformlist=[tx_file])
    moving_label = ants.apply_transforms(
        fixed=fixed_img, moving=fixed_label, transformlist=[tx_file], interpolator='nearestNeighbor'
    )
    
    return fixed_img, moving_img, fixed_label, moving_label


def create_3d_cross_modality_pair():
    """Generates 3D cross-modality T1 <-> T2 brain image pair with structural label map."""
    img_path = os.path.join(REPO_ROOT, 'cache', 'img1_brain.nii.gz')
    seg_path = os.path.join(REPO_ROOT, 'cache', 'dktseg1.nii.gz')
    
    if os.path.exists(img_path) and os.path.exists(seg_path):
        fixed_img = ants.image_read(img_path)
        fixed_label = ants.image_read(seg_path)
    else:
        # Fallback to downsampled ch2 or synthetic 3D phantom
        ch2_path = os.path.expanduser('~/.antspy/ch2.nii.gz')
        if os.path.exists(ch2_path):
            ch2 = ants.image_read(ch2_path)
            fixed_img = ants.resample_image(ch2, (32, 32, 32), use_voxels=True)
        else:
            vol = np.zeros((32, 32, 32), dtype=np.float32)
            z, y, x = np.ogrid[:32, :32, :32]
            vol[(x-16)**2 + (y-16)**2 + (z-16)**2 < 12**2] = 100.0
            vol[(x-16)**2 + (y-16)**2 + (z-16)**2 < 6**2] = 200.0
            fixed_img = ants.from_numpy(vol)
            
        v = fixed_img.numpy()
        seg = np.zeros_like(v, dtype=np.int32)
        seg[v > 30] = 1
        seg[v > 80] = 2
        seg[v > 150] = 3
        fixed_label = ants.from_numpy(
            seg, origin=fixed_img.origin, spacing=fixed_img.spacing, direction=fixed_img.direction
        )
        
    moving_t2_raw = create_piecewise_intensity_shuffled_brain(fixed_img)
    
    tx_init = ants.create_ants_transform(transform_type='AffineTransform', dimension=3)
    tx_params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, -3.0, 2.0])
    tx_init.set_parameters(tx_params)
    tx_file = tempfile.mktemp(suffix='.mat')
    ants.write_transform(tx_init, tx_file)
    
    moving_img = ants.apply_transforms(fixed=fixed_img, moving=moving_t2_raw, transformlist=[tx_file])
    moving_label = ants.apply_transforms(
        fixed=fixed_img, moving=fixed_label, transformlist=[tx_file], interpolator='nearestNeighbor'
    )
    
    return fixed_img, moving_img, fixed_label, moving_label


def run_benchmark():
    """Runs the Milestone 4 Multi-Modal Deep Metric Benchmark Suite."""
    print("=" * 80)
    print(" MILESTONE 4: MULTI-MODAL SYNTO DEEP SIMILARITY METRIC BENCHMARK SUITE ")
    print("=" * 80)
    print("Comparing: dino_2_lncc, vgg_4_lncc vs lncc, mattes_mi on Cross-Modality Pairs\n")
    
    metrics = [
        ('lncc', {'syn_metric': 'lncc'}),
        ('mattes_mi', {'syn_metric': 'mattes_mi'}),
        ('dino_2_lncc', {'syn_metric': 'dino_2_lncc', 'vgg_mode': 'lncc_3d', 'vgg_layers': [2]}),
        ('vgg_4_lncc', {'syn_metric': 'vgg_4_lncc', 'vgg_mode': 'lncc_3d', 'vgg_layers': [4]}),
    ]
    
    all_results = []
    
    # -------------------------------------------------------------
    # 1. 2D Brain Cross-Modality Benchmark
    # -------------------------------------------------------------
    print("--- Running 2D Brain Cross-Modality Registration Benchmark ---")
    fixed_2d, moving_2d, label_fixed_2d, label_moving_2d = create_2d_cross_modality_pair()
    
    for metric_name, conf in metrics:
        t0 = time.time()
        res = registration(
            fixed=fixed_2d,
            moving=moving_2d,
            type_of_transform='SyNTo',
            backend='pytorch',
            reg_iterations=[30, 20, 10],
            affine_iterations=[20, 10],
            verbose=False,
            **conf
        )
        elapsed = time.time() - t0
        
        # Single Interpolation Policy: composition of transforms in a single call
        warped_label = ants.apply_transforms(
            fixed=label_fixed_2d,
            moving=label_moving_2d,
            transformlist=res['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        # Quantitative Dice Overlap
        overlap_df = ants.label_overlap_measures(label_fixed_2d, warped_label)
        # Compute mean Dice over non-background label classes
        if 'MeanOverlap' in overlap_df.columns:
            mean_dice = float(overlap_df[overlap_df['Label'] != 'All']['MeanOverlap'].mean())
        elif 'TotalOrTargetOverlap' in overlap_df.columns:
            mean_dice = float(overlap_df[overlap_df['Label'] != 'All']['TotalOrTargetOverlap'].mean())
        else:
            mean_dice = float(overlap_df.iloc[0, 1])
            
        # Folding rate from Jacobian Determinant
        composite_warp_path = tempfile.mktemp(suffix='_warp2d.nii.gz')
        ants.image_write(ants.image_read(res['fwdtransforms'][0]), composite_warp_path)
        jac_img = ants.create_jacobian_determinant_image(fixed_2d, composite_warp_path, do_log=False)
        jac_vals = jac_img.numpy()
        folds = float(np.sum(jac_vals <= 0) / jac_vals.size * 100.0)
        min_jac = float(jac_vals.min())
        
        # Standardized image_compare return (lower is better)
        comp_score = image_compare(fixed_2d, res['warpedmovout'], metric_name)
        
        print(f"2D | Metric: {metric_name:12s} | Mean Dice: {mean_dice:.4f} | Folds: {folds:.4f}% | Min Jac: {min_jac:.4f} | Time: {elapsed:.2f}s")
        
        all_results.append({
            'Dimensionality': '2D',
            'Metric': metric_name,
            'Mean Dice': mean_dice,
            'Folds (%)': folds,
            'Min Jacobian': min_jac,
            'ImageCompare Score': comp_score,
            'Runtime (s)': elapsed
        })
        
    # -------------------------------------------------------------
    # 2. 3D Brain Cross-Modality Benchmark
    # -------------------------------------------------------------
    print("\n--- Running 3D Brain Cross-Modality Registration Benchmark ---")
    fixed_3d, moving_3d, label_fixed_3d, label_moving_3d = create_3d_cross_modality_pair()
    
    for metric_name, conf in metrics:
        t0 = time.time()
        res = registration(
            fixed=fixed_3d,
            moving=moving_3d,
            type_of_transform='SyNTo',
            backend='pytorch',
            reg_iterations=[20, 10, 5],
            affine_iterations=[10, 5],
            verbose=False,
            **conf
        )
        elapsed = time.time() - t0
        
        warped_label = ants.apply_transforms(
            fixed=label_fixed_3d,
            moving=label_moving_3d,
            transformlist=res['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        overlap_df = ants.label_overlap_measures(label_fixed_3d, warped_label)
        if 'MeanOverlap' in overlap_df.columns:
            mean_dice = float(overlap_df[overlap_df['Label'] != 'All']['MeanOverlap'].mean())
        elif 'TotalOrTargetOverlap' in overlap_df.columns:
            mean_dice = float(overlap_df[overlap_df['Label'] != 'All']['TotalOrTargetOverlap'].mean())
        else:
            mean_dice = float(overlap_df.iloc[0, 1])
            
        composite_warp_path = tempfile.mktemp(suffix='_warp3d.nii.gz')
        ants.image_write(ants.image_read(res['fwdtransforms'][0]), composite_warp_path)
        jac_img = ants.create_jacobian_determinant_image(fixed_3d, composite_warp_path, do_log=False)
        jac_vals = jac_img.numpy()
        folds = float(np.sum(jac_vals <= 0) / jac_vals.size * 100.0)
        min_jac = float(jac_vals.min())
        
        comp_score = image_compare(fixed_3d, res['warpedmovout'], metric_name)
        
        print(f"3D | Metric: {metric_name:12s} | Mean Dice: {mean_dice:.4f} | Folds: {folds:.4f}% | Min Jac: {min_jac:.4f} | Time: {elapsed:.2f}s")
        
        all_results.append({
            'Dimensionality': '3D',
            'Metric': metric_name,
            'Mean Dice': mean_dice,
            'Folds (%)': folds,
            'Min Jacobian': min_jac,
            'ImageCompare Score': comp_score,
            'Runtime (s)': elapsed
        })
        
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK TABLE ")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Save benchmark results
    out_dir = os.path.join(REPO_ROOT, ".agents", "worker_m4")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "benchmark_deep_metrics_results.csv"), index=False)
    
    return df


if __name__ == "__main__":
    run_benchmark()
