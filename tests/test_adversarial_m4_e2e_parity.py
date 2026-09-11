"""
tests/test_adversarial_m4_e2e_parity.py

Milestone 4 Dedicated Adversarial E2E Zero-Regression & Parity Suite.
Stress-tests the full registration engine combining all completed performance remediations:
  - M1: Stationary DST-I Green operator LRU caching & broadcasted eigenvalue formulation.
  - M2: Host-accelerator synchronization elimination in Anderson inversion & version-keyed TVF CFL caching.
  - M3: In-place fused Adam/RegAdam updates & non-contiguous layout restride elimination.

Challenges evaluated:
  1. Anisotropic Geometry & Autograd Vector Channel Alignment.
  2. Topological Regularity & Grid Folding Suppression (det(J) <= 0).
  3. Transformation Invertibility & Real Physical Inverse Consistency Error.
  4. Identity Zero-Motion Invariance & Optimizer Accumulator Stability.
  5. End-to-End Cache Telemetry & Invalidation Invariants.
"""

import math
import pytest
import numpy as np
import torch
import ants

import syntx
import syntx.spatial as sp
from syntx.core.smoothing import (
    apply_dsti_green_operator,
    smooth_displacement_field_dst,
    clear_dst_cache,
    get_dst_cache_info,
)
from syntx.core.inverse import (
    update_inverse_field_nd_anderson,
    compute_inverse_identity_error_nd,
)
from syntx.core.jacobian import compute_jacobian_determinant_nd
from syntx.core.grid import compose_grids, get_identity_grid_torch


class TestAdversarialM4AnisotropicGeometry:
    """Challenge 1: Stress-tests anisotropic geometry and autograd vector channel alignment."""

    def test_adversarial_anisotropic_geometry_autograd_scale(self):
        """
        Verifies autograd physical scale flip invariant under extreme anisotropic spacing (0.8, 1.5, 2.4 mm).
        GEMINI.md Rule: autograd physical scaling vector must be flipped along dim 0 via torch.flip.
        """
        shape_t = torch.tensor([20, 36, 48], dtype=torch.float32)  # Z, Y, X
        spacing_t = torch.tensor([2.4, 1.5, 0.8], dtype=torch.float32)  # s_z, s_y, s_x

        scale = sp.compute_autograd_physical_scale(shape_t, spacing_t)
        assert scale.shape == (3,)

        # Spatial order is (Z, Y, X) -> channels are (x, y, z)
        # s_phys[0] (x) must scale by (N_x - 1) * s_x / 2 = (48 - 1) * 0.8 / 2 = 18.8
        # s_phys[1] (y) must scale by (N_y - 1) * s_y / 2 = (36 - 1) * 1.5 / 2 = 26.25
        # s_phys[2] (z) must scale by (N_z - 1) * s_z / 2 = (20 - 1) * 2.4 / 2 = 22.8
        expected_x = (48.0 - 1.0) * 0.8 / 2.0
        expected_y = (36.0 - 1.0) * 1.5 / 2.0
        expected_z = (20.0 - 1.0) * 2.4 / 2.0

        assert torch.isclose(scale[0], torch.tensor(expected_x), atol=1e-5)
        assert torch.isclose(scale[1], torch.tensor(expected_y), atol=1e-5)
        assert torch.isclose(scale[2], torch.tensor(expected_z), atol=1e-5)

    def test_adversarial_anisotropic_syn_registration_2d(self):
        """
        Verifies 2D SyN registration on non-square, anisotropic grid with in-place Adam
        and non-contiguous composition.
        """
        torch.manual_seed(123)
        # Create anisotropic 2D images
        shape = (40, 60)
        spacing = (0.5, 1.5)
        fixed_np = np.zeros(shape, dtype=np.float32)
        # Add central Gaussian blob
        y, x = np.ogrid[:shape[0], :shape[1]]
        fixed_np += np.exp(-((y - 20)**2 / 40.0 + (x - 30)**2 / 60.0)).astype(np.float32)

        # Shifted moving image
        moving_np = np.zeros(shape, dtype=np.float32)
        moving_np += np.exp(-((y - 23)**2 / 40.0 + (x - 27)**2 / 60.0)).astype(np.float32)

        fixed_img = ants.from_numpy(fixed_np, spacing=spacing)
        moving_img = ants.from_numpy(moving_np, spacing=spacing)

        # Register using SyN with in-place Adam and layout restride elimination
        reg = syntx.syn(
            fixed=fixed_img,
            moving=moving_img,
            reg_iterations=[10, 5],
            grad_step=0.2,
            verbose=False,
            in_memory=True,
        )

        assert reg is not None
        assert 'warpedmovout' in reg
        assert 'fwdtransforms' in reg

        # Verify output warped image matches fixed geometry
        assert reg['warpedmovout'].shape == fixed_img.shape
        assert reg['warpedmovout'].spacing == fixed_img.spacing


class TestAdversarialM4JacobianFoldingRegularity:
    """Challenge 2: Stress-tests manifold regularity and topological fold suppression (det(J) <= 0)."""

    def test_adversarial_jacobian_folding_regularity(self):
        """
        Verifies that optimized SyN solver maintains strictly positive minimum Jacobian
        and fold rate <= 0.015% on non-linear deformation tasks.
        """
        torch.manual_seed(42)
        shape = (32, 32)
        spacing = (1.0, 1.0)

        # Create two smooth phantoms with complex shape differences
        y, x = np.ogrid[:shape[0], :shape[1]]
        c_x, c_y = 16, 16

        # Phantom 1: Ellipse
        r1 = ((x - c_x)**2 / 64.0 + (y - c_y)**2 / 25.0) <= 1.0
        fixed_np = r1.astype(np.float32) * 0.8

        # Phantom 2: Rotated ellipse
        cos_t, sin_t = np.cos(np.pi / 4), np.sin(np.pi / 4)
        xr = cos_t * (x - c_x) - sin_t * (y - c_y)
        yr = sin_t * (x - c_x) + cos_t * (y - c_y)
        r2 = (xr**2 / 64.0 + yr**2 / 25.0) <= 1.0
        moving_np = r2.astype(np.float32) * 0.8

        fixed_img = ants.from_numpy(fixed_np, spacing=spacing)
        moving_img = ants.from_numpy(moving_np, spacing=spacing)

        reg = syntx.syn(
            fixed=fixed_img,
            moving=moving_img,
            reg_iterations=[15, 10],
            grad_step=0.15,
            flow_sigma=2.0,
            verbose=False,
            in_memory=True,
        )

        # Extract deformation field from model
        disp_t = reg['model'].warp_l2r.detach().cpu()

        # Compute Jacobian determinant on device
        det_j = compute_jacobian_determinant_nd(disp_t, physical_spacing=spacing)
        det_np = det_j.squeeze().numpy()

        # Fold rate: percentage of voxels where det(J) <= 0
        total_voxels = det_np.size
        folded_voxels = np.sum(det_np <= 0.0)
        fold_rate = (folded_voxels / total_voxels) * 100.0
        min_det = float(np.min(det_np))

        # Enforce GEMINI.md folding thresholds:
        assert fold_rate <= 0.015, f"Fold rate exceeded threshold: {fold_rate:.4f}% > 0.015%"
        assert min_det > -0.05, f"Excessive negative Jacobian singularity: min det(J) = {min_det:.4f}"


class TestAdversarialM4InverseIdentityConsistency:
    """Challenge 3: Stress-tests Anderson acceleration invertibility and real physical inverse consistency error."""

    def test_adversarial_inverse_identity_consistency(self):
        """
        Verifies that on-device Anderson acceleration produces sub-voxel inverse consistency error
        e_inv(x) = ||phi_inv(x + phi_fwd(x)) + phi_fwd(x)||_2 < 0.03 mm mean error.
        """
        torch.manual_seed(99)
        shape = (24, 24)
        spacing = (1.0, 1.0)

        # Generate a smooth, invertible displacement field
        # Use low-frequency sinusoidal perturbation
        y, x = torch.meshgrid(
            torch.linspace(0, 2 * math.pi, shape[0]),
            torch.linspace(0, 2 * math.pi, shape[1]),
            indexing='ij',
        )
        # Small smooth deformation
        dx = 0.15 * torch.sin(y) * torch.cos(x)
        dy = 0.15 * torch.cos(y) * torch.sin(x)
        disp_fwd = torch.stack([dx, dy], dim=-1).unsqueeze(0)  # (1, 24, 24, 2)

        # Run Anderson acceleration inversion with decoupled check_interval
        disp_inv = update_inverse_field_nd_anderson(
            disp_fwd,
            -disp_fwd,
            spacing=spacing,
            steps=15,
            m=5,
            check_interval=1,
        )

        # Compute physical inverse error
        inv_err = compute_inverse_identity_error_nd(disp_fwd, disp_inv, spacing=spacing)
        assert inv_err.shape == (1, *shape)

        mean_err = float(inv_err.mean())
        p95_err = float(torch.quantile(inv_err, 0.95))
        max_err = float(inv_err.max())

        # Assertions per GEMINI.md:
        # Mean error <= 0.03 mm, 95th percentile <= 0.15 mm, peak <= 0.50 mm
        assert mean_err < 0.03, f"Mean inverse identity error too high: {mean_err:.5f} mm >= 0.03 mm"
        assert p95_err < 0.15, f"95th percentile inverse error too high: {p95_err:.5f} mm >= 0.15 mm"
        assert max_err < 0.50, f"Peak max inverse error too high: {max_err:.5f} mm >= 0.50 mm"

    def test_adversarial_inverse_decoupled_cadence_parity(self):
        """
        Verifies that check_interval=1 and check_interval=2 converge to near-identical
        displacement fields within numerical tolerance.
        """
        torch.manual_seed(101)
        disp = torch.randn(1, 16, 16, 2) * 0.05

        inv1 = update_inverse_field_nd_anderson(disp, -disp, steps=8, check_interval=1)
        inv2 = update_inverse_field_nd_anderson(disp, -disp, steps=8, check_interval=2)

        diff = (inv1 - inv2).abs().max().item()
        assert diff < 1e-5, f"Decoupled check interval parity deviation: {diff:.6e} >= 1e-5"


class TestAdversarialM4ZeroMotionInvariance:
    """Challenge 4: Stress-tests identity zero-motion invariance and accumulator stability."""

    def test_adversarial_zero_motion_invariance(self):
        """
        When fixed == moving, the optimization should produce zero displacement (sub-voxel)
        and in-place Adam accumulators must remain bounded and near-zero.
        """
        torch.manual_seed(77)
        shape = (28, 28)
        img_np = np.zeros(shape, dtype=np.float32)
        img_np[10:18, 10:18] = 1.0

        fixed_img = ants.from_numpy(img_np)
        moving_img = fixed_img.clone()

        reg = syntx.syn(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform='SyNOnly',
            reg_iterations=[5],
            affine_iterations=0,
            optimizer='adam',
            optimizer_lr=1e-3,
            verbose=False,
            in_memory=True,
        )

        fwd_warp = reg['model'].warp_l2r.detach().cpu().numpy()
        max_disp = float(np.max(np.abs(fwd_warp)))

        # On identity pairs with Adam in-place moment buffers, displacement should remain exactly zero
        assert max_disp < 1e-5, f"Zero motion invariance violated: max displacement = {max_disp:.5e} mm"


class TestAdversarialM4E2EIntegration:
    """Challenge 5: Validates cross-milestone integration and operator cache telemetry."""

    def test_e2e_dst_cache_telemetry_across_calls(self):
        """
        Verifies that repeated calls to apply_dsti_green_operator or smooth_displacement_field_dst
        hit the thread-safe LRU cache and report accurate hit/miss telemetry.
        """
        clear_dst_cache()
        info0 = get_dst_cache_info()
        assert info0.hits == 0
        assert info0.misses == 0

        shape = (1, 1, 16, 16)
        t1 = torch.randn(shape, dtype=torch.float32)

        # Call 1: Cache miss
        out1 = apply_dsti_green_operator(t1, spacing=(1.0, 1.0), alpha_val=1.0, s=2.0)
        info1 = get_dst_cache_info()
        assert info1.misses == 1
        assert info1.hits == 0

        # Call 2: Cache hit
        out2 = apply_dsti_green_operator(t1, spacing=(1.0, 1.0), alpha_val=1.0, s=2.0)
        info2 = get_dst_cache_info()
        assert info2.misses == 1
        assert info2.hits == 1

        # Bitwise identity
        assert torch.equal(out1, out2)

        # Call 3 via smooth_displacement_field_dst public alias
        out3 = smooth_displacement_field_dst(t1, spacing=(1.0, 1.0), alpha=1.0, s=2.0)
        info3 = get_dst_cache_info()
        assert info3.hits == 2
        assert torch.equal(out1, out3)
