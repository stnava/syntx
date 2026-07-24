import os
import sys
import tempfile
import pytest
import ants
import numpy as np
import torch

from syntx import registration, image_compare, FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor
from benchmarks.benchmark_deep_metrics import (
    create_piecewise_intensity_shuffled_brain,
    create_2d_cross_modality_pair,
    create_3d_cross_modality_pair,
    run_benchmark
)

def test_piecewise_intensity_shuffling():
    """Verify piecewise intensity shuffling creates valid contrast inversions without NaNs."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    shuffled = create_piecewise_intensity_shuffled_brain(r16)
    
    assert shuffled.shape == r16.shape
    v_orig = r16.numpy()
    v_shuf = shuffled.numpy()
    
    assert not np.isnan(v_shuf).any()
    assert not np.isinf(v_shuf).any()
    # High intensity values in original should be mapped to lower values in shuffled
    idx_high = v_orig > 200
    idx_low = (v_orig > 30) & (v_orig < 80)
    if idx_high.any() and idx_low.any():
        assert v_shuf[idx_high].mean() < v_orig[idx_high].mean()


def test_vgg_3d_mode_layer4_configuration():
    """Enforce GEMINI.md rule: VGG 3D Mode Requirement: vgg_mode='lncc_3d', vgg_layers=[4]."""
    vgg_ext = VGG19Extractor(feature_layers=[4])
    loss_fn = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d')
    
    x = torch.rand(1, 1, 16, 16, 16)
    y = torch.rand(1, 1, 16, 16, 16)
    val = loss_fn(x, y)
    
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0
    assert not torch.isnan(val).item()


def test_image_compare_standardized_returns():
    """Enforce GEMINI.md rule: image_compare metrics return standardized scores (lower is better)."""
    r16 = ants.image_read(ants.get_ants_data('r16'))
    # Identical images should yield 0.0 or lower score than mismatched images
    val_same_dino = image_compare(r16, r16, 'dino_2_lncc')
    val_same_vgg = image_compare(r16, r16, 'vgg_4_lncc')
    
    shuffled = create_piecewise_intensity_shuffled_brain(r16)
    val_diff_dino = image_compare(r16, shuffled, 'dino_2_lncc')
    val_diff_vgg = image_compare(r16, shuffled, 'vgg_4_lncc')
    
    assert val_same_dino < val_diff_dino
    assert val_same_vgg < val_diff_vgg


def test_milestone_4_deep_metrics_benchmark_execution():
    """
    Executes Milestone 4 benchmark suite and asserts:
    1. Deep metrics (dino_2_lncc, vgg_4_lncc) achieve higher Dice overlap on cross-modality
       inverted pairs than standard intensity LNCC.
    2. 0% folding rate (J > 0).
    """
    df = run_benchmark()
    assert not df.empty
    
    # Check 0% folding rate across all metrics
    assert (df['Folds (%)'] == 0.0).all()
    assert (df['Min Jacobian'] > 0.0).all()
    
    # Group by Dimensionality
    for dim in ['2D', '3D']:
        sub_df = df[df['Dimensionality'] == dim].set_index('Metric')
        dice_lncc = sub_df.loc['lncc', 'Mean Dice']
        dice_dino = sub_df.loc['dino_2_lncc', 'Mean Dice']
        dice_vgg = sub_df.loc['vgg_4_lncc', 'Mean Dice']
        
        print(f"[{dim}] LNCC Dice: {dice_lncc:.4f} | DINOv2 Dice: {dice_dino:.4f} | VGG4 Dice: {dice_vgg:.4f}")
        
        # Confirm deep metrics achieve higher Dice overlap than standard LNCC on modality-inverted pairs
        assert dice_dino > dice_lncc, f"DINOv2 ({dice_dino:.4f}) did not beat LNCC ({dice_lncc:.4f}) in {dim}"
        assert dice_vgg > dice_lncc, f"VGG Layer 4 ({dice_vgg:.4f}) did not beat LNCC ({dice_lncc:.4f}) in {dim}"
