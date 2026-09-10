"""Adversarial stress testing of registration engines under centralized syntx.spatial.

Challenges:
1. Stress test syn, tvf, syngs, and robust_affine across 2D non-square and 3D non-cubic
   geometries with anisotropic spacing.
2. Verify forward/inverse deformation field exports produce valid ants.ANTsImage objects
   with matching reference metadata (origin, spacing, direction).
3. Verify affine transform exports produce valid .mat files reloadable by ants.read_transform
   and usable in ants.apply_transforms.
4. Verify image_to_tensor dimensional contracts across ANTsImage, NumPy ndarray, and PyTorch Tensor.
"""
import os
import tempfile
import pytest
import numpy as np
import torch
import ants

import syntx
from syntx.spatial import (
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    export_ants_displacement_field,
    export_ants_affine_transform,
    grid_to_physical_affine,
    image_to_tensor,
    tensor_to_image,
)


def create_synthetic_2d_pair(shape=(48, 64), spacing=(1.5, 0.8), origin=(12.3, -4.5), direction=None):
    """Creates a synthetic 2D image pair with an ellipse."""
    if direction is None:
        direction = np.eye(2)
    
    arr_f = np.zeros(shape, dtype=np.float32)
    arr_m = np.zeros(shape, dtype=np.float32)
    
    cx, cy = shape[0] // 2, shape[1] // 2
    rx, ry = shape[0] // 4, shape[1] // 4
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if ((i - cx) / rx) ** 2 + ((j - cy) / ry) ** 2 <= 1.0:
                arr_f[i, j] = 1.0
            if ((i - (cx + 2)) / (rx * 1.1)) ** 2 + ((j - (cy - 3)) / (ry * 0.9)) ** 2 <= 1.0:
                arr_m[i, j] = 1.0

    fixed = ants.from_numpy(arr_f, origin=origin, spacing=spacing, direction=direction)
    moving = ants.from_numpy(arr_m, origin=origin, spacing=spacing, direction=direction)
    return fixed, moving


def create_synthetic_3d_pair(shape=(20, 28, 24), spacing=(1.2, 0.9, 1.5), origin=(-10.0, 15.0, 25.0), direction=None):
    """Creates a synthetic 3D image pair with an ellipsoid."""
    if direction is None:
        direction = np.eye(3)
    
    arr_f = np.zeros(shape, dtype=np.float32)
    arr_m = np.zeros(shape, dtype=np.float32)
    
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rx, ry, rz = shape[0] // 4, shape[1] // 4, shape[2] // 4
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if ((i - cx) / rx) ** 2 + ((j - cy) / ry) ** 2 + ((k - cz) / rz) ** 2 <= 1.0:
                    arr_f[i, j, k] = 1.0
                if ((i - (cx + 1)) / (rx * 1.1)) ** 2 + ((j - (cy - 1)) / (ry * 0.9)) ** 2 + ((k - (cz + 1)) / rz) ** 2 <= 1.0:
                    arr_m[i, j, k] = 1.0

    fixed = ants.from_numpy(arr_f, origin=origin, spacing=spacing, direction=direction)
    moving = ants.from_numpy(arr_m, origin=origin, spacing=spacing, direction=direction)
    return fixed, moving


class TestSpatialImageToTensorContract:
    """Stress tests syntx.spatial.image_to_tensor 2D and 3D dimensionality contracts."""

    def test_image_to_tensor_2d_dimensions(self):
        """2D ANTsImage, NumPy ndarray, and Tensor must produce shape (1, 1, H, W) [4D]."""
        # ANTsImage
        img_ants = ants.from_numpy(np.zeros((30, 40), dtype=np.float32))
        t_ants = image_to_tensor(img_ants)
        assert t_ants.shape == (1, 1, 30, 40), f"ANTs 2D shape mismatch: {t_ants.shape}"

        # NumPy array
        arr_np = np.zeros((30, 40), dtype=np.float32)
        t_np = image_to_tensor(arr_np)
        assert t_np.shape == (1, 1, 30, 40), f"NumPy 2D shape mismatch: {t_np.shape}"

        # PyTorch Tensor
        t_in = torch.zeros((30, 40), dtype=torch.float32)
        t_out = image_to_tensor(t_in)
        assert t_out.shape == (1, 1, 30, 40), f"Tensor 2D shape mismatch: {t_out.shape}"

    def test_image_to_tensor_3d_dimensions(self):
        """3D ANTsImage, NumPy ndarray, and Tensor must produce shape (1, 1, D, H, W) [5D]."""
        # ANTsImage: passes (correctly unsqueezes twice)
        img_ants = ants.from_numpy(np.zeros((16, 20, 24), dtype=np.float32))
        t_ants = image_to_tensor(img_ants)
        assert t_ants.shape == (1, 1, 16, 20, 24), f"ANTs 3D shape mismatch: {t_ants.shape}"

        # NumPy array: produces 5D tensor (1, 1, D, H, W)
        arr_np = np.zeros((16, 20, 24), dtype=np.float32)
        t_np = image_to_tensor(arr_np)
        assert t_np.shape == (1, 1, 16, 20, 24), f"NumPy 3D shape mismatch: {t_np.shape} (expected 5D tensor)"

        # PyTorch Tensor: produces 5D tensor (1, 1, D, H, W)
        t_in = torch.zeros((16, 20, 24), dtype=torch.float32)
        t_out = image_to_tensor(t_in)
        assert t_out.shape == (1, 1, 16, 20, 24), f"Tensor 3D shape mismatch: {t_out.shape} (expected 5D tensor)"

    def test_image_to_tensor_batched_contract(self):
        """Batched 2D and 3D tensors must preserve batch dim and ensure channel dim=1."""
        # 2D already batched with channel: (2, 1, 30, 40)
        t_2d_batched = torch.zeros((2, 1, 30, 40), dtype=torch.float32)
        out_2d = image_to_tensor(t_2d_batched)
        assert out_2d.shape == (2, 1, 30, 40)

        # 3D already batched with channel: (2, 1, 16, 20, 24)
        t_3d_batched = torch.zeros((2, 1, 16, 20, 24), dtype=torch.float32)
        out_3d = image_to_tensor(t_3d_batched)
        assert out_3d.shape == (2, 1, 16, 20, 24)

        # 3D with 1 leading batch dimension without channel: (1, 16, 20, 24)
        t_3d_single_batch = torch.zeros((1, 16, 20, 24), dtype=torch.float32)
        out_3d_single = image_to_tensor(t_3d_single_batch)
        assert out_3d_single.shape == (1, 1, 16, 20, 24)

        # 3D with multi-batch without channel: (3, 16, 20, 24)
        t_3d_multi_batch = torch.zeros((3, 16, 20, 24), dtype=torch.float32)
        out_3d_multi = image_to_tensor(t_3d_multi_batch)
        assert out_3d_multi.shape == (3, 1, 16, 20, 24)

    def test_image_to_tensor_to_zyx_contract(self):
        """to_zyx=True must transpose spatial dimensions for ANTsImage, ndarray, and Tensor."""
        # 2D ANTsImage, ndarray, Tensor
        img_2d = ants.from_numpy(np.zeros((30, 40), dtype=np.float32))
        arr_2d = np.zeros((30, 40), dtype=np.float32)
        t_2d = torch.zeros((30, 40), dtype=torch.float32)
        assert image_to_tensor(img_2d, to_zyx=True).shape == (1, 1, 40, 30)
        assert image_to_tensor(arr_2d, to_zyx=True).shape == (1, 1, 40, 30)
        assert image_to_tensor(t_2d, to_zyx=True).shape == (1, 1, 40, 30)

        # 3D ANTsImage, ndarray, Tensor
        img_3d = ants.from_numpy(np.zeros((16, 20, 24), dtype=np.float32))
        arr_3d = np.zeros((16, 20, 24), dtype=np.float32)
        t_3d = torch.zeros((16, 20, 24), dtype=torch.float32)
        assert image_to_tensor(img_3d, to_zyx=True).shape == (1, 1, 24, 20, 16)
        assert image_to_tensor(arr_3d, to_zyx=True).shape == (1, 1, 24, 20, 16)
        assert image_to_tensor(t_3d, to_zyx=True).shape == (1, 1, 24, 20, 16)



class TestRobustAffineAnisotropic:
    """Stress tests robust_affine across 2D and 3D anisotropic geometries."""

    def test_robust_affine_2d_anisotropic_non_square(self):
        fixed, moving = create_synthetic_2d_pair(shape=(48, 64), spacing=(1.5, 0.8), origin=(12.3, -4.5))
        reg = syntx.robust_affine(
            fixed=fixed, moving=moving,
            type_of_transform='Affine',
            mode='pytorch',
            multi_start=False,
            verbose=False
        )
        assert 'fwdtransforms' in reg
        tx_path = reg['fwdtransforms'][0]
        assert os.path.exists(tx_path)
        
        # Test reloadability
        tx = ants.read_transform(tx_path)
        assert tx is not None
        
        # Test application with ants.apply_transforms
        warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
        assert warped.shape == fixed.shape
        assert np.allclose(warped.spacing, fixed.spacing)
        assert np.allclose(warped.origin, fixed.origin)
        assert np.allclose(warped.direction, fixed.direction)
        assert not np.isnan(warped.numpy()).any()

    def test_robust_affine_3d_anisotropic_non_cubic(self):
        fixed, moving = create_synthetic_3d_pair(shape=(20, 28, 24), spacing=(1.2, 0.9, 1.5), origin=(-10.0, 15.0, 25.0))
        reg = syntx.robust_affine(
            fixed=fixed, moving=moving,
            type_of_transform='Affine',
            mode='pytorch',
            multi_start=False,
            verbose=False
        )
        assert 'fwdtransforms' in reg
        tx_path = reg['fwdtransforms'][0]
        assert os.path.exists(tx_path)
        
        tx = ants.read_transform(tx_path)
        assert tx is not None
        
        warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
        assert warped.shape == fixed.shape
        assert np.allclose(warped.spacing, fixed.spacing)
        assert np.allclose(warped.origin, fixed.origin)
        assert np.allclose(warped.direction, fixed.direction)
        assert not np.isnan(warped.numpy()).any()


class TestSyNAnisotropic:
    """Stress tests syntx.syn across 2D and 3D anisotropic geometries."""

    def test_syn_2d_anisotropic_non_square(self):
        fixed, moving = create_synthetic_2d_pair(shape=(48, 64), spacing=(1.5, 0.8), origin=(12.3, -4.5))
        reg = syntx.syn(
            fixed=fixed, moving=moving,
            reg_iterations=[8, 4],
            affine_iterations=[4, 4],
            verbose=False
        )
        assert 'fwdtransforms' in reg
        assert 'invtransforms' in reg
        
        fwd_warp_path = reg['fwdtransforms'][0]
        affine_path = reg['fwdtransforms'][1]
        
        # Verify deformation field export
        fwd_warp = ants.image_read(fwd_warp_path)
        assert fwd_warp.components == 2
        assert fwd_warp.shape == fixed.shape
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert np.allclose(fwd_warp.direction, fixed.direction, atol=1e-5)
        assert not np.isnan(fwd_warp.numpy()).any()
        
        # Verify affine .mat reloadability
        tx = ants.read_transform(affine_path)
        assert tx is not None
        
        # Verify composite application
        warped_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert warped_fwd.shape == fixed.shape
        assert np.allclose(warped_fwd.spacing, fixed.spacing, atol=1e-5)
        assert not np.isnan(warped_fwd.numpy()).any()
        
        # Verify inverse application
        warped_inv = ants.apply_transforms(
            fixed=moving, moving=fixed,
            transformlist=reg['invtransforms'],
            whichtoinvert=reg['whichtoinvert_inv']
        )
        assert warped_inv.shape == moving.shape
        assert np.allclose(warped_inv.spacing, moving.spacing, atol=1e-5)
        assert not np.isnan(warped_inv.numpy()).any()

    def test_syn_3d_anisotropic_non_cubic(self):
        fixed, moving = create_synthetic_3d_pair(shape=(16, 24, 20), spacing=(1.2, 0.9, 1.5), origin=(-10.0, 15.0, 25.0))
        reg = syntx.syn(
            fixed=fixed, moving=moving,
            reg_iterations=[4, 2],
            affine_iterations=[2, 2],
            verbose=False
        )
        assert 'fwdtransforms' in reg
        fwd_warp_path = reg['fwdtransforms'][0]
        affine_path = reg['fwdtransforms'][1]
        
        fwd_warp = ants.image_read(fwd_warp_path)
        assert fwd_warp.components == 3
        assert fwd_warp.shape == fixed.shape
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert np.allclose(fwd_warp.direction, fixed.direction, atol=1e-5)
        assert not np.isnan(fwd_warp.numpy()).any()
        
        tx = ants.read_transform(affine_path)
        assert tx is not None
        
        warped_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert warped_fwd.shape == fixed.shape
        assert np.allclose(warped_fwd.spacing, fixed.spacing, atol=1e-5)
        assert not np.isnan(warped_fwd.numpy()).any()


class TestTVFAnisotropic:
    """Stress tests syntx.tvf across 2D and 3D anisotropic geometries."""

    def test_tvf_2d_anisotropic_non_square(self):
        fixed, moving = create_synthetic_2d_pair(shape=(36, 48), spacing=(1.4, 0.9), origin=(5.0, -10.0))
        reg = syntx.tvf(
            fixed=fixed, moving=moving,
            reg_iterations=[4, 2],
            affine_iterations=[2, 2],
            verbose=False
        )
        assert 'fwdtransforms' in reg
        fwd_warp_path = reg['fwdtransforms'][0]
        affine_path = reg['fwdtransforms'][1]
        
        fwd_warp = ants.image_read(fwd_warp_path)
        assert fwd_warp.components == 2
        assert fwd_warp.shape == fixed.shape
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert np.allclose(fwd_warp.direction, fixed.direction, atol=1e-5)
        assert not np.isnan(fwd_warp.numpy()).any()
        
        tx = ants.read_transform(affine_path)
        assert tx is not None
        
        warped_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert warped_fwd.shape == fixed.shape
        assert not np.isnan(warped_fwd.numpy()).any()

    def test_tvf_3d_anisotropic_non_cubic(self):
        fixed, moving = create_synthetic_3d_pair(shape=(16, 20, 18), spacing=(1.3, 1.0, 1.2), origin=(0.0, 5.0, 10.0))
        reg = syntx.tvf(
            fixed=fixed, moving=moving,
            reg_iterations=[3, 1],
            affine_iterations=[2, 1],
            verbose=False
        )
        assert 'fwdtransforms' in reg
        fwd_warp_path = reg['fwdtransforms'][0]
        affine_path = reg['fwdtransforms'][1]
        
        fwd_warp = ants.image_read(fwd_warp_path)
        assert fwd_warp.components == 3
        assert fwd_warp.shape == fixed.shape
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert not np.isnan(fwd_warp.numpy()).any()
        
        tx = ants.read_transform(affine_path)
        assert tx is not None
        
        warped_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert warped_fwd.shape == fixed.shape
        assert not np.isnan(warped_fwd.numpy()).any()


class TestSyNGSAnisotropic:
    """Stress tests syntx.syngs across 2D and 3D anisotropic geometries."""

    def test_syngs_2d_anisotropic_non_square(self):
        fixed, moving = create_synthetic_2d_pair(shape=(36, 48), spacing=(1.4, 0.9), origin=(5.0, -10.0))
        reg = syntx.syngs(
            fixed=fixed, moving=moving,
            reg_iterations=[4, 2],
            affine_iterations=[2, 2],
            verbose=False
        )
        assert 'fwdtransforms' in reg
        fwd_warp_path = reg['fwdtransforms'][0]
        affine_path = reg['fwdtransforms'][1]
        
        fwd_warp = ants.image_read(fwd_warp_path)
        assert fwd_warp.components == 2
        assert fwd_warp.shape == fixed.shape
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert not np.isnan(fwd_warp.numpy()).any()
        
        tx = ants.read_transform(affine_path)
        assert tx is not None
        
        warped_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert warped_fwd.shape == fixed.shape
        assert not np.isnan(warped_fwd.numpy()).any()

    def test_syngs_3d_anisotropic_non_cubic(self):
        """Stress test 3D syngs registration (reveals image_to_tensor 3D tensor dimensionality bug)."""
        fixed, moving = create_synthetic_3d_pair(shape=(16, 20, 18), spacing=(1.3, 1.0, 1.2), origin=(0.0, 5.0, 10.0))
        reg = syntx.syngs(
            fixed=fixed, moving=moving,
            reg_iterations=[3, 1],
            affine_iterations=[2, 1],
            verbose=False
        )
        assert 'fwdtransforms' in reg
        fwd_warp_path = reg['fwdtransforms'][0]
        affine_path = reg['fwdtransforms'][1]
        
        fwd_warp = ants.image_read(fwd_warp_path)
        assert fwd_warp.components == 3
        assert fwd_warp.shape == fixed.shape
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert not np.isnan(fwd_warp.numpy()).any()
        
        tx = ants.read_transform(affine_path)
        assert tx is not None
        
        warped_fwd = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert warped_fwd.shape == fixed.shape
        assert not np.isnan(warped_fwd.numpy()).any()


class TestObliqueDirectionParity:
    """Stress tests registration with oblique direction cosines."""

    def test_syn_2d_oblique_direction(self):
        theta = np.radians(30.0)
        c, s = np.cos(theta), np.sin(theta)
        direction = np.array([[c, -s], [s, c]])
        
        fixed, moving = create_synthetic_2d_pair(
            shape=(40, 48), spacing=(1.2, 0.8), origin=(10.0, 20.0), direction=direction
        )
        reg = syntx.syn(
            fixed=fixed, moving=moving,
            reg_iterations=[4, 2],
            affine_iterations=[2, 2],
            verbose=False
        )
        fwd_warp = ants.image_read(reg['fwdtransforms'][0])
        assert np.allclose(fwd_warp.direction, direction, atol=1e-5)
        assert np.allclose(fwd_warp.origin, fixed.origin, atol=1e-5)
        assert np.allclose(fwd_warp.spacing, fixed.spacing, atol=1e-5)
        
        warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])
        assert np.allclose(warped.direction, direction, atol=1e-5)
        assert not np.isnan(warped.numpy()).any()
