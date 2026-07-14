import os
import pytest
import math
import numpy as np
import torch
import jax
import jax.numpy as jnp
import ants
from unittest.mock import patch

from syntx.syn import SyNTo as SyNToPy
from syntx.syn_jax import SyNTo as SyNToJax
from syntx.features import VGG19Extractor
from syntx import registration

def test_deep_feature_degeneracy_trigger_pytorch():
    # Setup inputs with shape < 32
    fixed_np = np.random.rand(1, 1, 16, 16).astype(np.float32)
    moving_np = np.random.rand(1, 1, 16, 16).astype(np.float32)
    fixed_tensor = torch.tensor(fixed_np)
    moving_tensor = torch.tensor(moving_np)
    
    # Initialize PyTorch SyNTo model with a deep metric ('vgg19')
    model = SyNToPy(dim=2, grid_shape=(16, 16))
    
    call_count = 0
    original_extract = VGG19Extractor.extract
    
    def dummy_extract(self, x):
        nonlocal call_count
        call_count += 1
        return original_extract(self, x)
        
    with patch.object(VGG19Extractor, 'extract', dummy_extract):
        model.fit(
            fixed_tensor, moving_tensor,
            levels=[1],
            epochs_per_level=3,
            affine_epochs=[0],
            similarity_metric='vgg19'
        )
        
    # Since grid shape is 16 < 32, it should fall back to local ncc.
    # Therefore, the VGG19 extractor should NOT be called.
    assert call_count == 0, f"Expected VGG19Extractor to not be called, but it was called {call_count} times."

    # Now verify it IS called when shape is >= 32
    fixed_np_large = np.random.rand(1, 1, 32, 32).astype(np.float32)
    moving_np_large = np.random.rand(1, 1, 32, 32).astype(np.float32)
    fixed_tensor_large = torch.tensor(fixed_np_large)
    moving_tensor_large = torch.tensor(moving_np_large)
    
    model_large = SyNToPy(dim=2, grid_shape=(32, 32))
    
    call_count_large = 0
    def dummy_extract_large(self, x):
        nonlocal call_count_large
        call_count_large += 1
        return original_extract(self, x)
        
    with patch.object(VGG19Extractor, 'extract', dummy_extract_large):
        model_large.fit(
            fixed_tensor_large, moving_tensor_large,
            levels=[1],
            epochs_per_level=3,
            affine_epochs=[0],
            similarity_metric='vgg19'
        )
    assert call_count_large > 0, "Expected VGG19Extractor to be called for shape >= 32."


def test_deep_feature_degeneracy_trigger_jax():
    # Setup inputs with shape < 32
    fixed_np = np.random.rand(1, 1, 16, 16).astype(np.float32)
    moving_np = np.random.rand(1, 1, 16, 16).astype(np.float32)
    fixed_tensor = torch.tensor(fixed_np)
    moving_tensor = torch.tensor(moving_np)
    
    # Initialize JAX SyNTo model with a deep metric ('vgg19')
    model = SyNToJax(dim=2, grid_shape=(16, 16))
    
    call_count = 0
    original_extract = VGG19Extractor.extract
    
    def dummy_extract(self, x):
        nonlocal call_count
        call_count += 1
        return original_extract(self, x)
        
    with patch.object(VGG19Extractor, 'extract', dummy_extract):
        model.fit(
            fixed_tensor, moving_tensor,
            levels=[1],
            epochs_per_level=3,
            affine_epochs=[0],
            similarity_metric='vgg19'
        )
        
    # Since grid shape is 16 < 32, JAX SyNTo should fall back to local_ncc_loss_nd_jax
    assert call_count == 0, f"Expected VGG19Extractor to not be called, but it was called {call_count} times."

    # Now verify it IS called when shape is >= 32
    fixed_np_large = np.random.rand(1, 1, 32, 32).astype(np.float32)
    moving_np_large = np.random.rand(1, 1, 32, 32).astype(np.float32)
    fixed_tensor_large = torch.tensor(fixed_np_large)
    moving_tensor_large = torch.tensor(moving_np_large)
    
    model_large = SyNToJax(dim=2, grid_shape=(32, 32))
    
    call_count_large = 0
    def dummy_extract_large(self, x):
        nonlocal call_count_large
        call_count_large += 1
        return original_extract(self, x)
        
    with patch.object(VGG19Extractor, 'extract', dummy_extract_large):
        model_large.fit(
            fixed_tensor_large, moving_tensor_large,
            levels=[1],
            epochs_per_level=3,
            affine_epochs=[0],
            similarity_metric='vgg19'
        )
    assert call_count_large > 0, "Expected VGG19Extractor to be called for shape >= 32 under JAX."


def test_displacement_export_and_non_folding():
    # Load 2D phantoms
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    
    # Run PyTorch registration
    res = registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        backend='pytorch',
        levels=[2, 1],
        affine_iterations=[30, 20],
        reg_iterations=[30, 20],
        grad_step=0.5,
        flow_sigma=1.0
    )
    
    # Verify that the forward transform files exist and are not empty
    fwd_tx_files = res['fwdtransforms']
    assert len(fwd_tx_files) > 0
    fwd_warp_file = next((tx for tx in fwd_tx_files if tx.endswith('.nii.gz')), None)
    assert fwd_warp_file is not None
    assert os.path.exists(fwd_warp_file)
    
    # Verify no folding: calculate Jacobian determinant map and check that it is strictly positive
    jac_img = ants.create_jacobian_determinant_image(fi, fwd_warp_file)
    jac_np = jac_img.numpy()
    min_jac = float(jac_np.min())
    folding_rate = float(np.mean(jac_np <= 0.0))
    
    print(f"Min Jacobian of PyTorch exported field: {min_jac}, Folding rate: {folding_rate}")
    assert min_jac > 0.0, "Displacement field causes folding!"
    assert folding_rate == 0.0, "Displacement field has folded regions!"

    # Clean up fwd / inv transforms
    for tx in res['fwdtransforms'] + res['invtransforms']:
        if os.path.exists(tx):
            try:
                os.remove(tx)
            except Exception:
                pass

    # Do the same for JAX backend
    res_jax = registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        backend='jax',
        levels=[2, 1],
        affine_iterations=[30, 20],
        reg_iterations=[30, 20],
        grad_step=0.5,
        flow_sigma=1.0
    )
    
    fwd_tx_files_jax = res_jax['fwdtransforms']
    assert len(fwd_tx_files_jax) > 0
    fwd_warp_file_jax = next((tx for tx in fwd_tx_files_jax if tx.endswith('.nii.gz')), None)
    assert fwd_warp_file_jax is not None
    assert os.path.exists(fwd_warp_file_jax)
    
    jac_img_jax = ants.create_jacobian_determinant_image(fi, fwd_warp_file_jax)
    jac_np_jax = jac_img_jax.numpy()
    min_jac_jax = float(jac_np_jax.min())
    folding_rate_jax = float(np.mean(jac_np_jax <= 0.0))
    
    print(f"Min Jacobian of JAX exported field: {min_jac_jax}, Folding rate: {folding_rate_jax}")
    assert min_jac_jax > 0.0, "JAX displacement field causes folding!"
    assert folding_rate_jax == 0.0, "JAX displacement field has folded regions!"

    # Clean up fwd / inv transforms
    for tx in res_jax['fwdtransforms'] + res_jax['invtransforms']:
        if os.path.exists(tx):
            try:
                os.remove(tx)
            except Exception:
                pass


def test_parameter_tuning_dice_parity():
    # Load 2D phantoms
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r27'))
    
    # Run classic ANTs SyN registration as baseline
    reg_ants = ants.registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyN',
        reg_iterations=[100, 100, 50],
        syn_metric='cc',
        syn_sampling=2
    )
    
    # Define local helper for Otsu threshold overlap
    def compute_tissue_overlap(fixed_img, warped_img):
        fixed_seg = ants.threshold_image(fixed_img, 'Otsu', 3)
        warped_seg = ants.threshold_image(warped_img, 'Otsu', 3)
        overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
        if 'MeanOverlap' in overlap.columns:
            return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
        return 0.0
        
    dice_ants = compute_tissue_overlap(fi, reg_ants['warpedmovout'])
    
    # Run PyTorch SyNTo with tuned parameters to achieve parity
    res_py = registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        backend='pytorch',
        levels=[4, 2, 1],
        affine_iterations=[100, 100, 50],
        reg_iterations=[100, 100, 50],
        grad_step=0.75,
        flow_sigma=1.732
    )
    
    dice_py = compute_tissue_overlap(fi, res_py['warpedmovout'])
    
    # Run JAX SyNTo with tuned parameters to achieve parity
    res_jax = registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        backend='jax',
        levels=[4, 2, 1],
        affine_iterations=[100, 100, 50],
        reg_iterations=[100, 100, 50],
        grad_step=0.75,
        flow_sigma=1.732
    )
    
    dice_jax = compute_tissue_overlap(fi, res_jax['warpedmovout'])
    
    print(f"Dice ANTs: {dice_ants:.4f}, Dice PyTorch: {dice_py:.4f}, Dice JAX: {dice_jax:.4f}")
    
    # Clean up fwd / inv transforms
    for r in [res_py, res_jax]:
        for tx in r['fwdtransforms'] + r['invtransforms']:
            if os.path.exists(tx):
                try:
                    os.remove(tx)
                except Exception:
                    pass
                    
    # Verify parity (within 1% absolute or relative, user says: "mean DICE score parity (within 1%)")
    # 1% difference in Dice means abs(dice_py - dice_ants) <= 0.01 or dice_py >= dice_ants - 0.01
    assert dice_py >= dice_ants - 0.01, f"PyTorch Dice score regression: {dice_py:.4f} vs {dice_ants:.4f} (baseline)"
    assert dice_jax >= dice_ants - 0.01, f"JAX Dice score regression: {dice_jax:.4f} vs {dice_ants:.4f} (baseline)"
