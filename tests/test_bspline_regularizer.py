"""Tests for ANTsTorch B-spline regularizer (BSplineSyN) across syntx.core, syn, syngs, and tvf."""

import pytest
import torch
import numpy as np
import ants

import syntx
from syntx.core.smoothing import (
    smooth_displacement_field_bspline,
    apply_bspline_fluid_operator,
    has_antstorch,
)


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is not available")
class TestBSplineRegularizerCore:
    def test_bspline_smoothing_unbatched_2d(self):
        field = torch.randn(24, 24, 2, dtype=torch.float32)
        smoothed = smooth_displacement_field_bspline(field, spacing=(1.0, 1.0), mesh_size=4)
        assert smoothed.shape == field.shape
        assert smoothed.dtype == field.dtype
        assert torch.isfinite(smoothed).all()

    def test_bspline_smoothing_batched_2d(self):
        field = torch.randn(3, 20, 20, 2, dtype=torch.float32)
        smoothed = smooth_displacement_field_bspline(field, spacing=(1.0, 1.0), mesh_size=4)
        assert smoothed.shape == field.shape
        assert smoothed.dtype == field.dtype
        assert torch.isfinite(smoothed).all()

    def test_bspline_smoothing_unbatched_3d(self):
        field = torch.randn(16, 16, 16, 3, dtype=torch.float32)
        smoothed = smooth_displacement_field_bspline(field, spacing=(1.0, 1.0, 1.0), mesh_size=3)
        assert smoothed.shape == field.shape
        assert smoothed.dtype == field.dtype
        assert torch.isfinite(smoothed).all()

    def test_bspline_smoothing_batched_3d(self):
        field = torch.randn(2, 12, 12, 12, 3, dtype=torch.float32)
        smoothed = smooth_displacement_field_bspline(field, spacing=(1.0, 1.0, 1.0), mesh_size=3)
        assert smoothed.shape == field.shape
        assert smoothed.dtype == field.dtype
        assert torch.isfinite(smoothed).all()

    def test_bspline_parameter_fallbacks(self):
        field = torch.randn(20, 20, 2, dtype=torch.float32)
        # Test explicit spline_distance
        s1 = smooth_displacement_field_bspline(field, spline_distance=10.0)
        assert s1.shape == field.shape

        # Test explicit tuple mesh_size
        s2 = smooth_displacement_field_bspline(field, mesh_size=(5, 5))
        assert s2.shape == field.shape

        # Test fallback from fluid_sigma
        s3 = smooth_displacement_field_bspline(field, fluid_sigma=2.5)
        assert s3.shape == field.shape

        # Test default fallback when no parameters provided
        s4 = smooth_displacement_field_bspline(field)
        assert s4.shape == field.shape

    def test_bspline_autograd_differentiability(self):
        field = torch.randn(16, 16, 2, dtype=torch.float32, requires_grad=True)
        smoothed = smooth_displacement_field_bspline(field, mesh_size=4, enforce_stationary_boundary=False)
        loss = (smoothed**2).sum()
        loss.backward()
        assert field.grad is not None
        assert field.grad.shape == field.shape
        assert field.grad.norm().item() > 0.0

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_bspline_mps_device(self):
        field_mps = torch.randn(20, 20, 2, device='mps', dtype=torch.float32)
        smoothed = smooth_displacement_field_bspline(field_mps, mesh_size=4)
        assert smoothed.device.type == 'mps'
        assert smoothed.shape == field_mps.shape


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is not available")
class TestBSplineRegistrationIntegration:
    @pytest.fixture
    def image_pair_2d(self):
        f_arr = np.pad(np.ones((16, 16), dtype=np.float32), 6)
        m_arr = np.pad(np.ones((16, 16), dtype=np.float32), ((8, 4), (4, 8)))
        fixed = ants.from_numpy(f_arr)
        moving = ants.from_numpy(m_arr)
        return fixed, moving

    def test_syn_regularizer_bspline_2d(self, image_pair_2d):
        fixed, moving = image_pair_2d
        reg = syntx.syn(
            fixed=fixed,
            moving=moving,
            reg_iterations=[2, 2],
            affine_iterations=[2],
            regularizer='bspline',
            mesh_size=4,
        )
        assert 'warpedmovout' in reg
        assert reg['warpedmovout'].shape == fixed.shape
        assert 'fwdtransforms' in reg

    def test_syn_type_of_transform_bsplinesyn_2d(self, image_pair_2d):
        fixed, moving = image_pair_2d
        reg = syntx.syn(
            fixed=fixed,
            moving=moving,
            reg_iterations=[2, 2],
            affine_iterations=[2],
            type_of_transform='BSplineSyN',
            spline_distance=15.0,
        )
        assert 'warpedmovout' in reg
        assert reg['warpedmovout'].shape == fixed.shape

    def test_syngs_regularizer_bspline_2d(self, image_pair_2d):
        fixed, moving = image_pair_2d
        reg = syntx.syngs(
            fixed=fixed,
            moving=moving,
            reg_iterations=[2, 2],
            affine_iterations=[2],
            regularizer='bspline',
            mesh_size=4,
        )
        assert 'warpedmovout' in reg
        assert reg['warpedmovout'].shape == fixed.shape

    def test_tvf_regularizer_bspline_2d(self, image_pair_2d):
        fixed, moving = image_pair_2d
        reg = syntx.tvf(
            fixed=fixed,
            moving=moving,
            reg_iterations=[2, 2],
            affine_iterations=0,
            regularizer='bspline',
            mesh_size=4,
        )
        assert 'warpedmovout' in reg
        assert reg['warpedmovout'].shape == fixed.shape
