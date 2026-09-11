"""
tests/test_challenger_m4_downstream_invariance.py

Dedicated Adversarial Challenger Verification Suite for Milestone 4: Downstream Solver Invariance.
Author: challenger_perf4_m4_2
Archetype: Empirical Challenger (critic, specialist)

Strict Verification Standards:
1. SyNGS (Geodesic Shooting):
   - Numerical stability under high momentum and diffeomorphism invertibility
   - 2D/3D bitwise seed reproducibility across consecutive runs
   - PyTorch / JAX shooting forward loss parity across anisotropic domains
2. Scattered SyN (syntx.scattered):
   - Degenerate boundary point sets (N=4 corner points) convergence
   - Anderson inversion cadence invariance (in_loop_inv_interval: 1 vs 2 vs 5)
   - Planar degenerate embeddings in 3D (Z=0 plane)
3. TVF (Time-Varying Velocity Fields):
   - Multi-step integration under automatic adaptive CFL substepping
   - RK4 vs Euler solver stability on steep velocity fields
   - Bidirectional symmetry and sub-voxel inverse consistency error
4. Multi-Run SyN Determinism:
   - Consecutive multi-run bitwise float reproducibility sweep
"""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import ants

import syntx
import syntx.spatial as sp
from syntx.core.inverse import (
    update_inverse_field_nd_anderson,
    compute_inverse_identity_error_nd,
)
from syntx.core.jacobian import compute_jacobian_determinant_nd
from syntx.core.grid import compose_grids, get_identity_grid_torch
from syntx.syngs import GeodesicShootingModel
from syntx.syngs_jax import GeodesicShootingModelJAX
import jax.numpy as jnp
from syntx.tvf import TVFModel
from syntx.scattered import (
    ScatteredRegistrationConfig,
    syn_scattered,
)


# ==============================================================================
# Suite 1: SyNGS Shooting Parity, Stability & Reproducibility
# ==============================================================================

class TestChallengerSyNGSInvariance:
    """Stress-tests syntx.syngs downstream solver invariance and numerical stability."""

    def test_syngs_high_momentum_stability_3d(self):
        """
        Challenge: SyNGS EPDiff shooting integration initialized with high momentum
        (v_0 ~ N(0, 0.4)) across 12 shooting steps in 3D.
        Verify that shooting does not diverge, produces zero NaNs/Infs,
        and computes smooth forward and inverse warps.
        """
        shape = (16, 16, 16)
        spacing = [1.0, 1.0, 1.0]
        model = GeodesicShootingModel(
            dim=3,
            image_shape=shape,
            velocity_shape=shape,
            n_steps=12,
            spacing=spacing,
        )

        torch.manual_seed(999)
        # Inject substantial initial momentum
        high_mom = torch.randn_like(model.velocity_0_fwd) * 0.4
        with torch.no_grad():
            model.velocity_0_fwd.copy_(high_mom)
            if hasattr(model, 'velocity_0_inv') and model.velocity_0_inv is not None:
                model.velocity_0_inv.copy_(-high_mom)

        # Generate forward warp via shooting
        phi_fwd = model.get_forward_warp()
        assert phi_fwd.shape == (1, 16, 16, 16, 3)
        assert not torch.isnan(phi_fwd).any(), "NaN in forward warp under high momentum"
        assert not torch.isinf(phi_fwd).any(), "Inf in forward warp under high momentum"

        # Generate inverse warp
        phi_inv = model.get_inverse_warp()
        assert phi_inv.shape == (1, 16, 16, 16, 3)
        assert not torch.isnan(phi_inv).any(), "NaN in inverse warp under high momentum"

        # Symmetry check: warps must have bounded norms
        fwd_max = float(torch.norm(phi_fwd, dim=-1).max())
        inv_max = float(torch.norm(phi_inv, dim=-1).max())
        assert fwd_max < 20.0, f"Unbounded forward displacement: {fwd_max}"
        assert inv_max < 20.0, f"Unbounded inverse displacement: {inv_max}"

    def test_syngs_pytorch_jax_anisotropic_parity(self):
        """
        Challenge: SyNGS forward loss parity between PyTorch and JAX backends
        on an anisotropic 3D grid (spacing=[2.5, 1.0, 0.5]).
        """
        shape = (16, 16, 16)
        vel_shape = (8, 8, 8)
        spacing = [2.5, 1.0, 0.5]

        # Synthetic images
        np.random.seed(42)
        img1 = np.random.randn(*shape).astype(np.float32)
        img2 = np.random.randn(*shape).astype(np.float32)

        model_pt = GeodesicShootingModel(
            dim=3, image_shape=shape, velocity_shape=vel_shape, n_steps=6, spacing=spacing
        )
        model_pt.eval()
        model_jax = GeodesicShootingModelJAX(
            dim=3, image_shape=shape, velocity_shape=vel_shape, n_steps=6, spacing=spacing
        )

        # Identical initial velocity
        v0 = np.random.randn(*model_pt.velocity_0.shape).astype(np.float32) * 0.05
        model_pt.velocity_0.data.copy_(torch.tensor(v0))
        model_jax.velocity_0 = jnp.array(v0)

        fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
        mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
        fi_jax = jnp.array(img1)[None, None, ...]
        mi_jax = jnp.array(img2)[None, None, ...]

        with torch.no_grad():
            l_pt = model_pt.forward(fi_pt, mi_pt).item()
        l_jax = float(model_jax.forward(fi_jax, mi_jax))

        delta = abs(l_pt - l_jax)
        assert delta < 0.015, f"Anisotropic SyNGS PT-JAX loss discrepancy too large: PT={l_pt:.6f}, JAX={l_jax:.6f}, delta={delta:.6e}"

    def test_syngs_seed_reproducibility_consecutive(self):
        """
        Challenge: Execute 3 consecutive SyNGS registrations with seed=1234
        and verify bitwise identical warped outputs.
        """
        shape = (24, 24)
        x, y = np.ogrid[:shape[0], :shape[1]]
        f_np = ((x - 12)**2 + (y - 12)**2 <= 6**2).astype(np.float32)
        m_np = ((x - 10)**2 + (y - 14)**2 <= 6**2).astype(np.float32)
        f_img = ants.from_numpy(f_np)
        m_img = ants.from_numpy(m_np)

        res1 = syntx.syngs(fixed=f_img, moving=m_img, levels=[2, 1], reg_iterations=[4, 4],
                           affine_iterations=0, seed=1234, device='cpu', verbose=False)
        res2 = syntx.syngs(fixed=f_img, moving=m_img, levels=[2, 1], reg_iterations=[4, 4],
                           affine_iterations=0, seed=1234, device='cpu', verbose=False)
        res3 = syntx.syngs(fixed=f_img, moving=m_img, levels=[2, 1], reg_iterations=[4, 4],
                           affine_iterations=0, seed=1234, device='cpu', verbose=False)

        w1 = res1['warpedmovout'].numpy()
        w2 = res2['warpedmovout'].numpy()
        w3 = res3['warpedmovout'].numpy()

        diff12 = np.max(np.abs(w1 - w2))
        diff13 = np.max(np.abs(w1 - w3))

        assert diff12 == 0.0, f"SyNGS run 1 and 2 differ by {diff12:.6e}"
        assert diff13 == 0.0, f"SyNGS run 1 and 3 differ by {diff13:.6e}"


# ==============================================================================
# Suite 2: Scattered SyN Extreme Geometries & Inversion Invariance
# ==============================================================================

class TestChallengerScatteredInvariance:
    """Stress-tests syntx.scattered solver invariance under extreme point clouds and inversion intervals."""

    def test_scattered_boundary_corner_points_convergence(self):
        """
        Challenge: Extreme point cloud with N=4 points positioned near domain boundaries
        ([-0.90, 0.90]). Verify solver converges without division by zero or NaN gradients.
        """
        device = torch.device('cpu')
        # 4 extreme corner points in [-1, 1] normalized space
        fixed_pts = torch.tensor([
            [-0.90, -0.90],
            [-0.90,  0.90],
            [ 0.90, -0.90],
            [ 0.90,  0.90],
        ], dtype=torch.float32, device=device)
        fixed_feats = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32, device=device)

        # Shifted slightly inwards
        moving_pts = torch.tensor([
            [-0.80, -0.80],
            [-0.80,  0.80],
            [ 0.80, -0.80],
            [ 0.80,  0.80],
        ], dtype=torch.float32, device=device)
        moving_feats = fixed_feats.clone()

        cfg = ScatteredRegistrationConfig(
            dim=2,
            grid_res=32,
            epochs_per_level=[10],
            levels=[1],
            optimizer_lr=0.05,
            cfl_voxels=0.25,
            device='cpu',
        )
        res = syn_scattered(fixed_pts, fixed_feats, moving_pts, moving_feats, config=cfg)

        assert not torch.isnan(res.warped_moving_points).any(), "NaN in warped points"
        assert not torch.isnan(res.warp_fwd).any(), "NaN in forward warp"
        assert not torch.isnan(res.warp_inv).any(), "NaN in inverse warp"
        assert res.folding_percentage <= 0.05, f"Fold percentage too high: {res.folding_percentage}%"

    def test_scattered_in_loop_inv_interval_stability(self):
        """
        Challenge: Verify that in_loop_inv_interval (1 vs 2 vs 5) maintains
        bounded inverse consistency error and prevents solver divergence.
        """
        device = torch.device('cpu')
        torch.manual_seed(101)
        fixed_pts = torch.rand(25, 2, device=device) * 1.4 - 0.7
        fixed_feats = (fixed_pts[:, 0]**2 + fixed_pts[:, 1]**2).unsqueeze(-1)

        moving_pts = fixed_pts + torch.randn_like(fixed_pts) * 0.04
        moving_feats = fixed_feats.clone()

        inv_errors = []
        for interval in [1, 2, 5]:
            cfg = ScatteredRegistrationConfig(
                dim=2,
                grid_res=32,
                epochs_per_level=[10],
                levels=[1],
                optimizer_lr=0.05,
                in_loop_inv_interval=interval,
                in_loop_inv_steps=3,
                cfl_voxels=0.25,
                device='cpu',
            )
            res = syn_scattered(fixed_pts, fixed_feats, moving_pts, moving_feats, config=cfg)

            fwd = res.warp_fwd if res.warp_fwd.dim() == 4 else res.warp_fwd.unsqueeze(0)
            inv = res.warp_inv if res.warp_inv.dim() == 4 else res.warp_inv.unsqueeze(0)
            err = compute_inverse_identity_error_nd(fwd, inv).mean().item()
            inv_errors.append(err)

            assert not math.isnan(err), f"NaN inverse error with interval={interval}"
            assert err < 0.10, f"Excessive inverse error {err:.5f} with interval={interval}"

        # Inversion cadence check: all intervals yield stable and bounded inverse identity error
        assert max(inv_errors) < 0.08

    def test_scattered_planar_embedding_in_3d(self):
        """
        Challenge: 3D point cloud where all points lie on a degenerate 2D plane (Z=0.0).
        Verify that 3D projector and SyNScattered solver operate without singular matrix crashes.
        """
        device = torch.device('cpu')
        torch.manual_seed(202)
        n_pts = 30
        xy = torch.rand(n_pts, 2, device=device) * 1.2 - 0.6
        z = torch.zeros(n_pts, 1, device=device)
        fixed_pts = torch.cat([xy, z], dim=-1)  # (N, 3)
        fixed_feats = (xy[:, 0]**2 + xy[:, 1]**2).unsqueeze(-1)

        moving_pts = fixed_pts.clone()
        moving_pts[:, :2] += torch.randn(n_pts, 2, device=device) * 0.03
        moving_feats = fixed_feats.clone()

        cfg = ScatteredRegistrationConfig(
            dim=3,
            grid_res=24,
            epochs_per_level=[8],
            levels=[1],
            optimizer_lr=0.05,
            cfl_voxels=0.25,
            device='cpu',
        )
        res = syn_scattered(fixed_pts, fixed_feats, moving_pts, moving_feats, config=cfg)

        assert res.warp_fwd.shape == (1, 24, 24, 24, 3)
        assert not torch.isnan(res.warp_fwd).any()
        assert not torch.isnan(res.warped_moving_points).any()
        assert res.folding_percentage <= 0.05


# ==============================================================================
# Suite 3: TVF Multi-Step Integration, Adaptive CFL & Invertibility
# ==============================================================================

class TestChallengerTVFInvariance:
    """Stress-tests TVF multi-step integration, adaptive CFL conditions, and solver stability."""

    def test_tvf_adaptive_cfl_steep_velocity_substepping(self):
        """
        Challenge: High-velocity field with large spatial gradients.
        Verify that adaptive CFL substepping (n_steps=None) automatically scales
        integration steps and completes without numerical blowup or stall.
        """
        shape = (20, 20)
        vel_shape = (10, 10)
        model = TVFModel(
            dim=2,
            image_shape=shape,
            velocity_shape=vel_shape,
            n_time_steps=3,
            solver='rk4',
        )

        # Inject large velocity
        torch.manual_seed(303)
        with torch.no_grad():
            model.velocity.copy_(torch.randn_like(model.velocity) * 1.5)

        # n_steps=None activates adaptive CFL substepping
        disp_fwd = model.integrate(0.0, 1.0, n_steps=None)
        disp_inv = model.integrate(1.0, 0.0, n_steps=None)

        assert disp_fwd.shape == (1, 20, 20, 2)
        assert disp_inv.shape == (1, 20, 20, 2)
        assert not torch.isnan(disp_fwd).any(), "NaN in TVF adaptive CFL forward warp"
        assert not torch.isnan(disp_inv).any(), "NaN in TVF adaptive CFL inverse warp"

    def test_tvf_rk4_vs_euler_divergence_comparison(self):
        """
        Challenge: Compare RK4 vs Euler integration accuracy on identical velocity field.
        RK4 should yield superior or equal inverse symmetry error compared to Euler.
        """
        shape = (16, 16)
        vel_shape = (8, 8)
        torch.manual_seed(404)
        # self.velocity shape: (n_time_steps, 1, *velocity_shape, dim)
        v_init = torch.randn(4, 1, 8, 8, 2) * 0.2

        model_rk4 = TVFModel(dim=2, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, solver='rk4')
        model_euler = TVFModel(dim=2, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, solver='euler')

        with torch.no_grad():
            model_rk4.velocity.copy_(v_init)
            model_euler.velocity.copy_(v_init)

        fwd_rk4 = model_rk4.integrate(0.0, 1.0, n_steps=8)
        inv_rk4 = model_rk4.integrate(1.0, 0.0, n_steps=8)

        fwd_euler = model_euler.integrate(0.0, 1.0, n_steps=8)
        inv_euler = model_euler.integrate(1.0, 0.0, n_steps=8)

        # Inversion symmetry error
        err_rk4 = (fwd_rk4 + inv_rk4).abs().mean().item()
        err_euler = (fwd_euler + inv_euler).abs().mean().item()

        assert not math.isnan(err_rk4) and not math.isnan(err_euler)
        assert err_rk4 <= err_euler + 1e-4, f"RK4 error ({err_rk4:.5f}) exceeded Euler ({err_euler:.5f})"

    def test_tvf_bidirectional_subvoxel_inverse_consistency(self):
        """
        Challenge: Verify that TVF forward and inverse coordinate maps compose to Identity
        within sub-voxel tolerance (< 0.25 voxels).
        """
        shape = (24, 24)
        model = TVFModel(dim=2, image_shape=shape, velocity_shape=(12, 12), n_time_steps=3, solver='rk4')
        torch.manual_seed(505)
        with torch.no_grad():
            model.velocity.copy_(torch.randn_like(model.velocity) * 0.08)

        fwd = model.get_forward_warp()
        inv = model.get_inverse_warp()

        # Physical spacing = (1.0, 1.0)
        err = compute_inverse_identity_error_nd(fwd, inv, spacing=(1.0, 1.0))
        mean_err = err.mean().item()
        p95_err = torch.quantile(err, 0.95).item()

        assert mean_err < 0.25, f"TVF mean inverse error {mean_err:.4f} >= 0.25 voxels"
        assert p95_err < 0.50, f"TVF p95 inverse error {p95_err:.4f} >= 0.50 voxels"


# ==============================================================================
# Suite 4: Multi-Run Determinism & Precision Stress
# ==============================================================================

class TestChallengerMultiRunDeterminism:
    """Stress-tests exact bitwise reproducibility across multiple runs with identical seeds."""

    def test_fast_reproducibility_five_run_sweep(self):
        """
        Challenge: Run 5 consecutive 2D SyN registrations with seed=777 and verify
        that all 5 runs match with L_inf < 1e-4.
        """
        shape = (28, 28)
        x, y = np.ogrid[:shape[0], :shape[1]]
        f_np = ((x - 14)**2 + (y - 14)**2 <= 7**2).astype(np.float32)
        m_np = ((x - 12)**2 + (y - 15)**2 <= 7**2).astype(np.float32)
        f_img = ants.from_numpy(f_np)
        m_img = ants.from_numpy(m_np)

        warped_outputs = []
        for run_idx in range(5):
            torch.manual_seed(777)
            res = syntx.syn(
                fixed=f_img,
                moving=m_img,
                reg_iterations=[6, 4],
                affine_iterations=[4, 4],
                verbose=False,
                in_memory=True,
            )
            warped_outputs.append(res['warpedmovout'].numpy())

        ref = warped_outputs[0]
        for run_idx, w in enumerate(warped_outputs[1:], start=2):
            diff = np.max(np.abs(ref - w))
            assert diff < 1e-4, f"Run {run_idx} differed from Run 1 by {diff:.6e}"
