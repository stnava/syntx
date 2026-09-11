"""
tests/test_challenger_m4_stress.py

Dedicated Adversarial Challenger Verification Suite for Milestone 4 (Registration & Parity Stress).
Author: challenger_perf4_m4_1
Archetype: Empirical Challenger (critic, specialist)

Strict Verification Standards:
1. GEMINI.md Autograd Vector Channel Alignment & Physical Scaling:
   s_phys = flip((N - 1) * s / 2, dim=0)
2. Manifold Regularity & Topological Folding Bounds:
   min det(J) > 0, fold rate <= 0.015%
3. Real Physical Sub-Voxel Inverse Consistency Error:
   mean e_inv < 0.03 mm, p95 < 0.15 mm, max < 0.50 mm
4. Optimizer & Numerical Robustness:
   In-place Adam/RegAdam memory address invariance and non-contiguous gradient resilience
5. TVF & Inversion Decoupled Invariance:
   Version-keyed CFL cache invalidation and check_interval cadence consistency
"""

import math
import threading
import numpy as np
import pytest
import torch
import torch.nn.functional as F
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
from syntx.tvf import TVFModel


# ==============================================================================
# Challenge 1: Autograd Vector Channel Alignment & Extreme Anisotropy
# ==============================================================================

class TestChallengerAutogradPhysicalScaling:
    """Stress-tests GEMINI.md physical scaling flip rule and autograd coordinate alignment."""

    def test_autograd_scale_flip_anisotropic_2d_and_3d(self):
        """
        Verify that compute_autograd_physical_scale strictly flips the spatial dimension order
        (Z, Y, X) -> channels (x, y, z) across extreme anisotropic geometries.
        """
        # 2D Case: shape=(N_y, N_x)=(15, 65), spacing=(s_y, s_x)=(3.2, 0.4)
        shape_2d = torch.tensor([15.0, 65.0], dtype=torch.float32)
        spacing_2d = torch.tensor([3.2, 0.4], dtype=torch.float32)
        scale_2d = sp.compute_autograd_physical_scale(shape_2d, spacing_2d)

        assert scale_2d.shape == (2,)
        # Channel 0 (x): (65 - 1) * 0.4 / 2 = 12.8
        # Channel 1 (y): (15 - 1) * 3.2 / 2 = 22.4
        expected_x_2d = (65.0 - 1.0) * 0.4 / 2.0
        expected_y_2d = (15.0 - 1.0) * 3.2 / 2.0
        assert torch.isclose(scale_2d[0], torch.tensor(expected_x_2d), atol=1e-5)
        assert torch.isclose(scale_2d[1], torch.tensor(expected_y_2d), atol=1e-5)

        # 3D Case: shape=(N_z, N_y, N_x)=(16, 32, 64), spacing=(s_z, s_y, s_x)=(4.0, 1.5, 0.5)
        shape_3d = torch.tensor([16.0, 32.0, 64.0], dtype=torch.float32)
        spacing_3d = torch.tensor([4.0, 1.5, 0.5], dtype=torch.float32)
        scale_3d = sp.compute_autograd_physical_scale(shape_3d, spacing_3d)

        assert scale_3d.shape == (3,)
        # Channel 0 (x): (64 - 1) * 0.5 / 2 = 15.75
        # Channel 1 (y): (32 - 1) * 1.5 / 2 = 23.25
        # Channel 2 (z): (16 - 1) * 4.0 / 2 = 30.0
        expected_x_3d = (64.0 - 1.0) * 0.5 / 2.0
        expected_y_3d = (32.0 - 1.0) * 1.5 / 2.0
        expected_z_3d = (16.0 - 1.0) * 4.0 / 2.0
        assert torch.isclose(scale_3d[0], torch.tensor(expected_x_3d), atol=1e-5)
        assert torch.isclose(scale_3d[1], torch.tensor(expected_y_3d), atol=1e-5)
        assert torch.isclose(scale_3d[2], torch.tensor(expected_z_3d), atol=1e-5)

    def test_autograd_gradient_backprop_channel_cross_axis_invariance(self):
        """
        Verify that backpropagating a loss with respect to displacement field coordinates
        scales displacements with the correct physical axis, and does not cross-scale.
        """
        shape = (10, 20, 30)  # Z, Y, X
        spacing = (5.0, 2.0, 0.5)  # s_z, s_y, s_x

        shape_t = torch.tensor(shape, dtype=torch.float32)
        spacing_t = torch.tensor(spacing, dtype=torch.float32)
        scale = sp.compute_autograd_physical_scale(shape_t, spacing_t)

        # Displacement field in physical units mm (channel order x, y, z)
        disp = torch.zeros(1, *shape, 3, requires_grad=True)

        # Normalized coordinates in [-1, 1]
        # dx_norm = dx_phys / scale[0], dy_norm = dy_phys / scale[1], dz_norm = dz_phys / scale[2]
        disp_norm = disp / scale

        # Loss sensitive to normalized grid shifts
        loss = (disp_norm[..., 0] * 1.0 + disp_norm[..., 1] * 2.0 + disp_norm[..., 2] * 3.0).sum()
        loss.backward()

        # dLoss / d(disp_phys) should equal factor / scale
        grad_x = disp.grad[0, 0, 0, 0, 0].item()
        grad_y = disp.grad[0, 0, 0, 0, 1].item()
        grad_z = disp.grad[0, 0, 0, 0, 2].item()

        expected_grad_x = 1.0 / scale[0].item()
        expected_grad_y = 2.0 / scale[1].item()
        expected_grad_z = 3.0 / scale[2].item()

        assert abs(grad_x - expected_grad_x) < 1e-6
        assert abs(grad_y - expected_grad_y) < 1e-6
        assert abs(grad_z - expected_grad_z) < 1e-6


# ==============================================================================
# Challenge 2: Topological Regularity & Grid Folding Bounds (det(J) <= 0)
# ==============================================================================

class TestChallengerDeformationRegularity:
    """Stress-tests deformation regularity under high-shear and complex non-linear registration."""

    def test_syn_3d_anisotropic_deformation_regularity(self):
        """
        Challenge: 3D SyN registration on an anisotropic phantom (Z=16, Y=24, X=32)
        with spacing=(2.5, 1.2, 0.8). Verify min det(J) > 0 and fold_rate <= 0.015%.
        """
        torch.manual_seed(314)
        shape = (16, 24, 32)
        spacing = (2.5, 1.2, 0.8)

        # Generate smooth 3D ellipsoidal phantom
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        c_z, c_y, c_x = 8, 12, 16

        # Fixed sphere
        r_fix = ((z - c_z)**2 / 16.0 + (y - c_y)**2 / 36.0 + (x - c_x)**2 / 64.0) <= 1.0
        fixed_np = r_fix.astype(np.float32) * 0.9

        # Moving shifted & sheared ellipsoid
        r_mov = ((z - c_z - 1)**2 / 16.0 + (y - c_y + 2)**2 / 40.0 + (x - c_x - 2)**2 / 55.0) <= 1.0
        moving_np = r_mov.astype(np.float32) * 0.9

        fixed_img = ants.from_numpy(fixed_np, spacing=spacing)
        moving_img = ants.from_numpy(moving_np, spacing=spacing)

        reg = syntx.syn(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform='SyNOnly',
            reg_iterations=[10, 5],
            grad_step=0.2,
            flow_sigma=2.0,
            verbose=False,
            in_memory=True,
        )

        warp_disp = reg['model'].warp_l2r.detach().cpu()
        det_j = compute_jacobian_determinant_nd(warp_disp, physical_spacing=spacing)
        det_np = det_j.squeeze().numpy()

        fold_rate = (np.sum(det_np <= 0.0) / det_np.size) * 100.0
        min_det = float(np.min(det_np))

        # Check strict GEMINI.md compliance:
        assert fold_rate <= 0.015, f"Fold rate {fold_rate:.5f}% exceeds 0.015% limit"
        assert min_det > 0.0, f"Non-positive interior minimum Jacobian: min det(J) = {min_det:.5f}"

    def test_syn_2d_high_shear_regularity(self):
        """
        Challenge: 2D SyN with sharp shear and gradient forces under Eulerian formulation.
        Verify min det(J) > 0 and zero interior grid folding.
        """
        torch.manual_seed(271)
        shape = (36, 36)
        spacing = (1.0, 1.0)
        y, x = np.ogrid[:shape[0], :shape[1]]

        # Star / Cross pattern
        fixed_np = np.zeros(shape, dtype=np.float32)
        fixed_np[12:25, 16:21] = 1.0
        fixed_np[16:21, 12:25] = 1.0

        moving_np = np.zeros(shape, dtype=np.float32)
        # Shifted diagonal
        moving_np[14:27, 14:19] = 1.0
        moving_np[18:23, 10:23] = 1.0

        fixed_img = ants.from_numpy(fixed_np, spacing=spacing)
        moving_img = ants.from_numpy(moving_np, spacing=spacing)

        reg = syntx.syn(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform='SyNOnly',
            reg_iterations=[12, 6],
            grad_step=0.2,
            flow_sigma=2.5,
            verbose=False,
            in_memory=True,
        )

        warp_disp = reg['model'].warp_l2r.detach().cpu()
        det_j = compute_jacobian_determinant_nd(warp_disp, physical_spacing=spacing)
        det_np = det_j.squeeze().numpy()

        fold_rate = (np.sum(det_np <= 0.0) / det_np.size) * 100.0
        min_det = float(np.min(det_np))

        assert fold_rate <= 0.015, f"Fold rate {fold_rate:.5f}% exceeds 0.015%"
        assert min_det > 0.0, f"Minimum det(J) = {min_det:.5f} <= 0"


# ==============================================================================
# Challenge 3: Real Physical Inverse Identity Error (< 0.03 mm)
# ==============================================================================

class TestChallengerInverseIdentityConsistency:
    """Stress-tests Anderson acceleration invertibility and sub-voxel inverse consistency error."""

    def test_anderson_inverse_subvoxel_consistency_3d_anisotropic(self):
        """
        Challenge: 3D anisotropic deformation field inverted via Anderson acceleration.
        Verify:
        - Mean inverse consistency error < 0.03 mm
        - 95th percentile error < 0.15 mm
        - Peak max error < 0.50 mm
        """
        torch.manual_seed(555)
        shape = (16, 20, 24)
        spacing = (2.0, 1.2, 0.8)

        # Construct smooth forward displacement field
        z, y, x = torch.meshgrid(
            torch.linspace(0, 2 * math.pi, shape[0]),
            torch.linspace(0, 2 * math.pi, shape[1]),
            torch.linspace(0, 2 * math.pi, shape[2]),
            indexing='ij',
        )
        dx = 0.12 * torch.sin(z) * torch.cos(y) * torch.sin(x)
        dy = 0.12 * torch.cos(z) * torch.sin(y) * torch.cos(x)
        dz = 0.12 * torch.sin(z) * torch.sin(y) * torch.cos(x)
        disp_fwd = torch.stack([dx, dy, dz], dim=-1).unsqueeze(0)  # (1, 16, 20, 24, 3)

        # Invert with Anderson acceleration
        disp_inv = update_inverse_field_nd_anderson(
            disp_fwd,
            -disp_fwd,
            spacing=spacing,
            steps=15,
            m=5,
            check_interval=2,
        )

        inv_err = compute_inverse_identity_error_nd(disp_fwd, disp_inv, spacing=spacing)
        assert inv_err.shape == (1, *shape)

        mean_err = float(inv_err.mean())
        p95_err = float(torch.quantile(inv_err, 0.95))
        max_err = float(inv_err.max())

        # Assertions per GEMINI.md:
        assert mean_err < 0.03, f"Mean inverse error {mean_err:.5f} mm >= 0.03 mm"
        assert p95_err < 0.15, f"95th percentile inverse error {p95_err:.5f} mm >= 0.15 mm"
        assert max_err < 0.50, f"Max inverse error {max_err:.5f} mm >= 0.50 mm"

    def test_anderson_vs_picard_convergence_supremacy(self):
        """
        Verify that Anderson acceleration achieves lower residual error than Picard (m=0)
        within 10 iterations on identical displacement inputs.
        """
        torch.manual_seed(888)
        shape = (20, 20)
        disp = torch.randn(1, *shape, 2) * 0.08

        # Anderson (m=5)
        inv_anderson = update_inverse_field_nd_anderson(disp, -disp, steps=10, m=5, check_interval=1)
        err_anderson = float(compute_inverse_identity_error_nd(disp, inv_anderson).mean())

        # Picard (m=0)
        inv_picard = update_inverse_field_nd_anderson(disp, -disp, steps=10, m=0, check_interval=1)
        err_picard = float(compute_inverse_identity_error_nd(disp, inv_picard).mean())

        assert err_anderson <= err_picard + 1e-6, (
            f"Anderson error ({err_anderson:.6f}) was expected to be <= Picard ({err_picard:.6f})"
        )


# ==============================================================================
# Challenge 4: In-Place Adam / RegAdam Accumulator Numerical Stability
# ==============================================================================

class TestChallengerInPlaceAdamRobustness:
    """Stress-tests in-place Adam updates against non-contiguous tensors, memory reallocation, and extreme scales."""

    def test_adam_non_contiguous_gradient_resilience(self):
        """
        Verify that in-place Adam moment updates (.mul_().add_(), .addcmul_())
        operate accurately when fed non-contiguous gradients (e.g. permuted or sliced).
        """
        shape = (1, 10, 12, 14, 3)
        torch.manual_seed(444)

        m_buffer = torch.zeros(shape, dtype=torch.float32)
        v_buffer = torch.zeros(shape, dtype=torch.float32)

        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        # Create non-contiguous gradient via transpose
        grad_base = torch.randn(1, 14, 12, 10, 3, dtype=torch.float32)
        grad_non_contig = grad_base.permute(0, 3, 2, 1, 4)  # shape is (1, 10, 12, 14, 3), non-contiguous
        assert not grad_non_contig.is_contiguous()

        addr_m_before = m_buffer.data_ptr()
        addr_v_before = v_buffer.data_ptr()

        # Step 1 update
        t = 1
        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t
        b2_sqrt = math.sqrt(b2)
        step_size = b2_sqrt / b1
        eps_scaled = eps * b2_sqrt

        m_buffer.mul_(beta1).add_(grad_non_contig, alpha=1.0 - beta1)
        v_buffer.mul_(beta2).addcmul_(grad_non_contig, grad_non_contig, value=1.0 - beta2)
        u_opt = (m_buffer * step_size).div_(v_buffer.sqrt().add_(eps_scaled))

        # Check address invariance (strictly in-place)
        assert m_buffer.data_ptr() == addr_m_before
        assert v_buffer.data_ptr() == addr_v_before

        # Compare with contiguous reference
        grad_contig = grad_non_contig.contiguous()
        m_ref = (1.0 - beta1) * grad_contig
        v_ref = (1.0 - beta2) * (grad_contig ** 2)
        u_ref = (m_ref / b1) / (torch.sqrt(v_ref / b2) + eps)

        diff = (u_opt - u_ref).abs().max().item()
        assert diff < 1e-6, f"Non-contiguous gradient step deviated from reference: {diff:.6e}"

    def test_adam_long_horizon_gradient_stability(self):
        """
        Verify that 100 consecutive in-place Adam updates remain bounded without accumulation error.
        """
        shape = (1, 8, 8, 8, 3)
        torch.manual_seed(777)

        m = torch.zeros(shape)
        v = torch.zeros(shape)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for t in range(1, 101):
            grad = torch.randn(shape) * 0.1
            b1 = 1.0 - beta1 ** t
            b2 = 1.0 - beta2 ** t
            b2_sqrt = math.sqrt(b2)
            step_size = b2_sqrt / b1
            eps_scaled = eps * b2_sqrt

            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            u = (m * step_size).div_(v.sqrt().add_(eps_scaled))

            assert not torch.isnan(u).any(), f"NaN detected at step {t}"
            assert not torch.isinf(u).any(), f"Inf detected at step {t}"
            assert u.abs().max() < 100.0, f"Exploding update at step {t}: {u.abs().max()}"


# ==============================================================================
# Challenge 5: TVF Version Invalidation & Geodesic Integration Invariance
# ==============================================================================

class TestChallengerTVFIntegrationRobustness:
    """Stress-tests TVF version tracking, cache invalidation, and integration stability."""

    def test_tvf_velocity_invalidation_and_reintegration_consistency(self):
        """
        Verify that in-place updates to TVF velocity properly invalidate _v_max_cache
        and that forward and backward integration are smooth and stable.
        """
        model = TVFModel(dim=2, image_shape=(24, 24), velocity_shape=(12, 12), n_time_steps=3)
        target_shape = (24, 24)

        # Pre-populate cache
        v_max_0 = model._get_v_max_voxel(model.velocity, target_shape)
        v0_ver = model.velocity._version

        # Update velocity in-place using modernized pattern
        with torch.no_grad():
            model.velocity.add_(torch.randn_like(model.velocity) * 0.2)

        v1_ver = model.velocity._version
        assert v1_ver > v0_ver, "Velocity version did not increment on in-place update"

        # Key with old version must be stale
        key_new = (id(model.velocity), v1_ver, target_shape)
        assert key_new not in model._v_max_cache

        # Re-evaluating populates new key
        v_max_1 = model._get_v_max_voxel(model.velocity, target_shape)
        assert key_new in model._v_max_cache
        assert v_max_1 > 0.0

        # Integrate forward and backward
        disp_fwd = model.integrate(0.0, 1.0)
        disp_inv = model.integrate(1.0, 0.0)

        assert disp_fwd.shape == (1, 24, 24, 2)
        assert disp_inv.shape == (1, 24, 24, 2)
        assert not torch.isnan(disp_fwd).any()
        assert not torch.isnan(disp_inv).any()


# ==============================================================================
# Challenge 6: DST-I Green Operator Concurrency & Cache Capacity
# ==============================================================================

class TestChallengerDSTICacheRobustness:
    """Stress-tests DST-I Green operator LRU caching under concurrent multi-threaded requests."""

    def test_dsti_cache_multi_threaded_storm(self):
        """
        Concurrently execute 32 threads calling apply_dsti_green_operator with both matching
        and varying geometries to stress-test the thread-safe LRU lock.
        """
        clear_dst_cache()
        errors = []

        def worker(thread_id):
            try:
                # Half threads use common geometry, half use thread-specific geometry
                if thread_id % 2 == 0:
                    shape = (1, 1, 16, 16)
                    spacing = (1.0, 1.0)
                else:
                    sz = 12 + (thread_id % 5)
                    shape = (1, 1, sz, sz)
                    spacing = (1.0 + thread_id * 0.1, 1.0)

                t = torch.randn(shape, dtype=torch.float32)
                out1 = apply_dsti_green_operator(t, spacing=spacing, alpha_val=1.0, s=2.0)
                out2 = apply_dsti_green_operator(t, spacing=spacing, alpha_val=1.0, s=2.0)

                if not torch.equal(out1, out2):
                    errors.append(f"Thread {thread_id}: repeated call outputs differed!")
            except Exception as e:
                errors.append(f"Thread {thread_id} raised: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0, f"Multi-threaded DST-I errors: {errors}"
        info = get_dst_cache_info()
        assert info.hits > 0, f"Expected cache hits during multi-threaded run, got {info.hits}"
