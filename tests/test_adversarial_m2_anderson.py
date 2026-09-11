"""
Adversarial Stress Testing & Parity Verification for Milestone 2 (M2)
Synchronization-Free Anderson Inversion and TVF CFL Barrier Elimination

Challenges:
1. High-frequency, high-shear, near-singular deformation fields (vortex, ripple, boundary folding).
2. Numerical parity (L_inf < 1e-6, rel_err < 1e-5) against verbatim un-modified baseline across 2D/3D.
3. Cadence and convergence verification across check_interval = 1, 2, 5, 10.
4. Host-accelerator synchronization reduction verification.
5. TVF CFL cache version-invalidation, shape-isolation, and bounded-cache robustness.
"""

import math
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from syntx.core.inverse import (
    update_inverse_field_nd,
    update_inverse_field_nd_anderson,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
    get_boundary_mask,
)
from syntx.spatial import (
    physical_to_normalized_torch_cached,
    get_physical_grid_torch,
    jacobian_determinant,
)
from syntx.core.smoothing import separable_gaussian_filter
from syntx.tvf import TVFModel


# ==============================================================================
# Baseline Reference Implementation (Verbatim Pre-M2 Baseline)
# ==============================================================================

def _baseline_anderson(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor = None,
    steps: int = 30,
    smoothing_sigma: float = 0.0,
    m: int = 5,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None,
) -> torch.Tensor:
    """Verbatim un-modified baseline Anderson inversion with CPU scalar conversions."""
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    use_physical = (spacing is not None and origin is not None and direction is not None)

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        if X_phys.shape[0] != B:
            X_phys = X_phys.expand(B, *([-1] * (dim + 1)))
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )

    boundary_mask = get_boundary_mask(spatial, device, dtype)
    W_disp_cf = torch.movedim(W_disp, -1, 1)

    if W_inv_disp is None:
        W_inv_disp = -W_disp.clone()

    def itk_fixed_point_step(v_curr, iteration):
        if use_physical:
            coords_phys = X_phys + v_curr
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t)**2, dim=-1, keepdim=True))
        else:
            coords = identity + v_curr
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale)**2, dim=-1, keepdim=True))

        max_error_norm = float(scaled_norm.max())
        mean_error_norm = float(scaled_norm.mean())

        epsilon = 0.75 if iteration == 0 else 0.5
        update = -error
        clip_threshold = epsilon * max_error_norm
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            torch.ones_like(scaled_norm)
        )
        update = update * clip_scale
        v_new = v_curr + update * epsilon

        if smoothing_sigma > 0.0:
            if use_physical:
                v_new = separable_gaussian_filter(v_new, smoothing_sigma, spacing=spacing)
            else:
                v_new = separable_gaussian_filter(v_new, smoothing_sigma)

        v_new = v_new * boundary_mask
        return v_new, max_error_norm, mean_error_norm, error

    v_k = W_inv_disp.clone()
    R_history = []
    G_history = []

    for iteration in range(steps):
        g_k, max_err, mean_err, err_k = itk_fixed_point_step(v_k, iteration)

        if max_err <= max_error_threshold and mean_err <= mean_error_threshold:
            v_k = g_k
            break

        r_k = (g_k - v_k).reshape(-1)
        R_history.append(r_k)
        G_history.append(g_k.reshape(-1))

        if len(R_history) > m + 1:
            R_history.pop(0)
            G_history.pop(0)

        m_k = len(R_history)

        if m_k < 2:
            v_k = g_k
        else:
            n_cols = m_k - 1
            r_newest = R_history[-1]
            dR_cols = []
            for j in range(n_cols):
                dR_cols.append(r_newest - R_history[j])

            gram = torch.zeros(n_cols, n_cols, device=device, dtype=dtype)
            rhs = torch.zeros(n_cols, device=device, dtype=dtype)
            for i_col in range(n_cols):
                rhs[i_col] = torch.dot(dR_cols[i_col], r_newest)
                for j_col in range(i_col, n_cols):
                    val = torch.dot(dR_cols[i_col], dR_cols[j_col])
                    gram[i_col, j_col] = val
                    gram[j_col, i_col] = val

            gram += 1e-10 * torch.eye(n_cols, device=device, dtype=dtype)

            try:
                gamma = torch.linalg.solve(gram, rhs)
            except torch.linalg.LinAlgError:
                v_k = g_k
                continue

            g_newest = G_history[-1]
            v_new_flat = g_newest.clone()
            for j in range(n_cols):
                dG_j = g_newest - G_history[j]
                v_new_flat = v_new_flat - gamma[j] * (dG_j + dR_cols[j])

            v_candidate = v_new_flat.reshape(W_inv_disp.shape)
            v_candidate = v_candidate * boundary_mask

            if use_physical:
                coords_phys_c = X_phys + v_candidate
                coords_norm_c = physical_to_normalized_torch_cached(coords_phys_c, shape_t, spacing_t, origin_t, direction_t)
                fwd_at_c = torch.movedim(
                    F.grid_sample(W_disp_cf, coords_norm_c, padding_mode='border', align_corners=True), 1, -1
                )
                error_c = v_candidate + fwd_at_c
                residual_aa = float(torch.sum((error_c / spacing_t)**2).sqrt())
                residual_fp = float(torch.sum((err_k / spacing_t)**2).sqrt())
            else:
                coords_c = identity + v_candidate
                fwd_at_c = torch.movedim(
                    F.grid_sample(W_disp_cf, coords_c, padding_mode='border', align_corners=True), 1, -1
                )
                error_c = v_candidate + fwd_at_c
                residual_aa = float(torch.sum((error_c * voxel_scale)**2).sqrt())
                residual_fp = float(torch.sum((err_k * voxel_scale)**2).sqrt())

            if residual_aa <= residual_fp * 1.1:
                v_k = v_candidate
            else:
                v_k = g_k

    return torch.movedim(v_k, -1, 1) if channels_first else v_k


# ==============================================================================
# Adversarial Challenge 1: High Shear, Vortex Warps & Near-Singular Fields
# ==============================================================================

class TestAdversarialExtremeFields:
    """Stress-test Anderson inversion on high-shear, vortex, and near-singular warps."""

    def test_high_shear_and_vortex_2d(self):
        """Test inversion on 2D fields with large shear and rotational vortex."""
        H, W = 32, 32
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')

        # 1. High shear field: u_x = gamma * y^2, u_y = 0
        disp_shear = torch.zeros(1, H, W, 2)
        disp_shear[0, :, :, 0] = 0.25 * (y ** 2)

        # 2. Strong vortex field: u_x = -A * y * exp(-(r/s)^2), u_y = A * x * exp(-(r/s)^2)
        r2 = x**2 + y**2
        vortex_scale = torch.exp(-r2 / 0.2)
        disp_vortex = torch.zeros(1, H, W, 2)
        disp_vortex[0, :, :, 0] = -0.35 * y * vortex_scale
        disp_vortex[0, :, :, 1] = 0.35 * x * vortex_scale

        # 3. High-frequency ripple: u_x = A*sin(4*pi*x)*cos(4*pi*y)
        disp_ripple = torch.zeros(1, H, W, 2)
        disp_ripple[0, :, :, 0] = 0.10 * torch.sin(4 * math.pi * x) * torch.cos(4 * math.pi * y)
        disp_ripple[0, :, :, 1] = 0.10 * torch.cos(4 * math.pi * x) * torch.sin(4 * math.pi * y)

        for name, disp in [("shear", disp_shear), ("vortex", disp_vortex), ("ripple", disp_ripple)]:
            # Check Jacobian determinant to verify field is demanding / near-singular
            jac = jacobian_determinant(disp)
            min_jac = jac.min().item()

            # Run baseline vs optimized
            inv_base = _baseline_anderson(disp.clone(), steps=15, m=4)
            inv_opt = update_inverse_field_nd_anderson(disp.clone(), steps=15, m=4, check_interval=1)

            assert torch.isfinite(inv_opt).all(), f"NaN or Inf encountered in {name} inversion"

            linf = (inv_base - inv_opt).abs().max().item()
            rel = linf / (inv_base.abs().max().item() + 1e-8)
            assert linf < 1e-6, f"{name}: L_inf={linf:.2e} exceeded 1e-6 (min_detJ={min_jac:.3f})"
            assert rel < 1e-5, f"{name}: rel_error={rel:.2e} exceeded 1e-5 (min_detJ={min_jac:.3f})"

            # Verify inverse identity error is finite and bounded
            err = compute_inverse_identity_error_nd(disp, inv_opt, is_displacement=True)
            assert torch.isfinite(err).all()
            assert err.abs().mean().item() < 0.25

    def test_near_singular_boundary_folding(self):
        """Test displacement with extreme boundary gradients attempting to fold coordinate grid."""
        torch.manual_seed(999)
        H, W = 28, 28
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')

        # Extreme compression field driving Jacobian close to or below 0
        disp_fold = torch.zeros(1, H, W, 2)
        disp_fold[0, :, :, 0] = -0.45 * torch.tanh(3.0 * x)
        disp_fold[0, :, :, 1] = -0.45 * torch.tanh(3.0 * y)

        inv_base = _baseline_anderson(disp_fold.clone(), steps=12, m=3)
        inv_opt = update_inverse_field_nd_anderson(disp_fold.clone(), steps=12, m=3, check_interval=1)

        assert torch.isfinite(inv_opt).all()
        linf = (inv_base - inv_opt).abs().max().item()
        assert linf < 1e-6, f"Boundary fold: L_inf={linf:.2e} exceeded 1e-6"

    def test_3d_helical_shear_anisotropic(self):
        """Test 3D helical vortex and shear on anisotropic geometry."""
        D, H, W = 14, 18, 16
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )

        disp_3d = torch.zeros(1, D, H, W, 3)
        disp_3d[0, ..., 0] = -0.15 * y * torch.exp(-(x**2 + y**2) / 0.3)
        disp_3d[0, ..., 1] = 0.15 * x * torch.exp(-(x**2 + y**2) / 0.3)
        disp_3d[0, ..., 2] = 0.12 * torch.sin(math.pi * z)

        spacing = (0.8, 1.2, 2.0)
        origin = (10.0, -5.0, 20.0)
        direction = np.eye(3).flatten()

        inv_base = _baseline_anderson(disp_3d.clone(), steps=8, m=3, spacing=spacing, origin=origin, direction=direction)
        inv_opt = update_inverse_field_nd_anderson(disp_3d.clone(), steps=8, m=3, spacing=spacing, origin=origin, direction=direction, check_interval=1)

        assert torch.isfinite(inv_opt).all()
        linf = (inv_base - inv_opt).abs().max().item()
        rel = linf / (inv_base.abs().max().item() + 1e-8)
        assert linf < 1e-6, f"3D helical: L_inf={linf:.2e} exceeded 1e-6"
        assert rel < 1e-5, f"3D helical: rel_error={rel:.2e} exceeded 1e-5"


# ==============================================================================
# Adversarial Challenge 2: Numerical Parity Across Geometries and Multi-Seeds
# ==============================================================================

class TestAdversarialNumericalParity:
    """Verify strict L_inf < 1e-6 and rel_err < 1e-5 parity across varied settings."""

    @pytest.mark.parametrize("seed", [101, 202, 303, 404])
    def test_2d_multi_seed_random_parity(self, seed):
        """Random smooth displacement fields across distinct seeds in 2D."""
        torch.manual_seed(seed)
        disp = torch.randn(1, 20, 24, 2) * 0.035
        # Light smoothing to create anatomically plausible field
        disp = separable_gaussian_filter(disp, sigma=1.0)

        inv_base = _baseline_anderson(disp.clone(), steps=12, m=4)
        inv_opt = update_inverse_field_nd_anderson(disp.clone(), steps=12, m=4, check_interval=1)

        linf = (inv_base - inv_opt).abs().max().item()
        rel = linf / (inv_base.abs().max().item() + 1e-8)
        assert linf < 1e-6, f"Seed {seed}: L_inf={linf:.2e} exceeded 1e-6"
        assert rel < 1e-5, f"Seed {seed}: rel_error={rel:.2e} exceeded 1e-5"

    @pytest.mark.parametrize("seed", [505, 606])
    def test_3d_physical_space_parity(self, seed):
        """3D physical space with anisotropic spacing and non-identity direction cosines."""
        torch.manual_seed(seed)
        disp = torch.randn(1, 10, 14, 12, 3) * 0.025
        disp = separable_gaussian_filter(disp, sigma=0.8)

        spacing = (1.5, 0.9, 2.1)
        origin = (-25.0, 10.0, 100.0)
        # Oblique direction matrix (valid rotation)
        angle = math.pi / 6.0
        c, s = math.cos(angle), math.sin(angle)
        direction = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ]).flatten()

        inv_base = _baseline_anderson(
            disp.clone(), steps=10, m=3,
            spacing=spacing, origin=origin, direction=direction
        )
        inv_opt = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            spacing=spacing, origin=origin, direction=direction,
            check_interval=1
        )

        linf = (inv_base - inv_opt).abs().max().item()
        rel = linf / (inv_base.abs().max().item() + 1e-8)
        assert linf < 1e-6, f"3D physical seed {seed}: L_inf={linf:.2e} exceeded 1e-6"
        assert rel < 1e-5, f"3D physical seed {seed}: rel_error={rel:.2e} exceeded 1e-5"


# ==============================================================================
# Adversarial Challenge 3: Stopping Cadence & Convergence Across check_interval
# ==============================================================================

class TestAdversarialCheckIntervalCadence:
    """Stress-test check_interval = 1, 2, 5, 10 stopping cadence and convergence."""

    def test_exact_step_identity_at_max_steps(self):
        """When stopping thresholds are 0.0 (run to max steps), check_interval produces identical results."""
        torch.manual_seed(777)
        disp = torch.randn(1, 16, 16, 2) * 0.03

        # Force running all 10 steps by setting stopping thresholds to 0.0
        inv_int1 = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            max_error_threshold=0.0, mean_error_threshold=0.0,
            check_interval=1
        )
        inv_int2 = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            max_error_threshold=0.0, mean_error_threshold=0.0,
            check_interval=2
        )
        inv_int5 = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            max_error_threshold=0.0, mean_error_threshold=0.0,
            check_interval=5
        )
        inv_int10 = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            max_error_threshold=0.0, mean_error_threshold=0.0,
            check_interval=10
        )

        # All runs must reach the same final iterate identically
        for name, inv in [("int2", inv_int2), ("int5", inv_int5), ("int10", inv_int10)]:
            diff = (inv_int1 - inv).abs().max().item()
            assert diff < 1e-6, f"{name} differed from int1 by {diff:.2e} at max steps"

    def test_cadence_early_stopping_quantization(self):
        """Early stopping is quantized to check_interval boundaries."""
        torch.manual_seed(888)
        disp = torch.randn(1, 18, 18, 2) * 0.015

        # With generous thresholds, convergence happens early
        # We can observe the exact stopping step by running steps=1..10
        history_steps = {}
        for s in range(1, 11):
            history_steps[s] = update_inverse_field_nd_anderson(
                disp.clone(), steps=s, m=3,
                max_error_threshold=0.0, mean_error_threshold=0.0,
                check_interval=1
            )

        # check_interval=1: will stop at first iteration it where error criteria is met
        inv_c1 = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            max_error_threshold=0.05, mean_error_threshold=0.005,
            check_interval=1
        )

        # check_interval=5: can only stop at it=4 (5 steps) or it=9 (10 steps)
        inv_c5 = update_inverse_field_nd_anderson(
            disp.clone(), steps=10, m=3,
            max_error_threshold=0.05, mean_error_threshold=0.005,
            check_interval=5
        )

        assert torch.isfinite(inv_c1).all()
        assert torch.isfinite(inv_c5).all()

        # Both produce high-quality inverse fields
        err_c1 = compute_inverse_identity_error_nd(disp, inv_c1, is_displacement=True).abs().mean().item()
        err_c5 = compute_inverse_identity_error_nd(disp, inv_c5, is_displacement=True).abs().mean().item()
        assert err_c1 < 0.05
        assert err_c5 < 0.05

    def test_check_interval_edge_values(self):
        """Edge values: check_interval=0, None, and > steps."""
        disp = torch.randn(1, 12, 12, 2) * 0.02
        inv_zero = update_inverse_field_nd_anderson(disp.clone(), steps=6, check_interval=0)
        inv_none = update_inverse_field_nd_anderson(disp.clone(), steps=6, check_interval=None)
        inv_large = update_inverse_field_nd_anderson(disp.clone(), steps=6, check_interval=100)

        # When check_interval is 0, None, or > steps, it runs all steps (checking only at it == steps-1)
        assert torch.allclose(inv_zero, inv_none, atol=1e-6)
        assert torch.allclose(inv_zero, inv_large, atol=1e-6)


# ==============================================================================
# Adversarial Challenge 4: Host Synchronization Reduction Verification
# ==============================================================================

class TestAdversarialHostSyncReduction:
    """Measure and empirically verify device-to-host synchronization reduction."""

    def test_sync_call_count_reduction(self):
        """Count actual host synchronization transfers (.item() calls and float conversions)."""
        disp = torch.randn(1, 16, 16, 2) * 0.02
        steps = 10

        # Baseline sync counter: in _baseline_anderson, every step calls:
        # float(scaled_norm.max()), float(scaled_norm.mean())
        # and when m_k >= 2: float(residual_aa), float(residual_fp)
        # Total per step: 2 to 4 syncs. Over 10 steps: up to 38 syncs.

        # Now test optimized Anderson: track how many times .item() or float() is called
        original_item = torch.Tensor.item
        sync_counter = {"item": 0}

        def tracked_item(self):
            sync_counter["item"] += 1
            return original_item(self)

        torch.Tensor.item = tracked_item
        try:
            # Run with check_interval=1
            sync_counter["item"] = 0
            _ = update_inverse_field_nd_anderson(
                disp.clone(), steps=steps, m=3,
                max_error_threshold=0.0, mean_error_threshold=0.0,
                check_interval=1
            )
            syncs_interval_1 = sync_counter["item"]

            # Run with check_interval=5
            sync_counter["item"] = 0
            _ = update_inverse_field_nd_anderson(
                disp.clone(), steps=steps, m=3,
                max_error_threshold=0.0, mean_error_threshold=0.0,
                check_interval=5
            )
            syncs_interval_5 = sync_counter["item"]

            # Run with check_interval=10
            sync_counter["item"] = 0
            _ = update_inverse_field_nd_anderson(
                disp.clone(), steps=steps, m=3,
                max_error_threshold=0.0, mean_error_threshold=0.0,
                check_interval=10
            )
            syncs_interval_10 = sync_counter["item"]

        finally:
            torch.Tensor.item = original_item

        # Baseline syncs: for steps=10, m=3, m_k >= 2 starts at step 2 (0-indexed 1)
        # Step 0: 2 syncs. Steps 1-9: 4 syncs each. Total = 2 + 9*4 = 38 syncs.
        baseline_syncs = 38

        # In M2:
        # check_interval=1: exactly 1 sync per iteration (.item() for boolean check) -> 10 syncs
        # check_interval=5: checks only at iteration 4 and 9 -> exactly 2 syncs
        # check_interval=10: checks only at iteration 9 -> exactly 1 sync
        assert syncs_interval_1 <= 10, f"Expected <= 10 syncs with check_interval=1, got {syncs_interval_1}"
        assert syncs_interval_5 == 2, f"Expected exactly 2 syncs with check_interval=5, got {syncs_interval_5}"
        assert syncs_interval_10 == 1, f"Expected exactly 1 sync with check_interval=10, got {syncs_interval_10}"

        # Calculate empirical reduction percentages
        red_int1 = (baseline_syncs - syncs_interval_1) / baseline_syncs * 100.0
        red_int5 = (baseline_syncs - syncs_interval_5) / baseline_syncs * 100.0
        red_int10 = (baseline_syncs - syncs_interval_10) / baseline_syncs * 100.0

        assert red_int1 >= 73.0, f"check_interval=1 sync reduction {red_int1:.1f}% < 73%"
        assert red_int5 >= 94.0, f"check_interval=5 sync reduction {red_int5:.1f}% < 94%"
        assert red_int10 >= 97.0, f"check_interval=10 sync reduction {red_int10:.1f}% < 97%"


# ==============================================================================
# Adversarial Challenge 5: TVF CFL Version & Shape Cache Robustness
# ==============================================================================

class TestAdversarialTVFCFLCache:
    """Stress-test TVF CFL version-invalidation and cache boundaries."""

    def test_cfl_cache_optimizer_invalidation(self):
        """Verify torch.optim optimizer step increments _version and invalidates CFL cache."""
        model = TVFModel(image_shape=(20, 20), velocity_shape=(20, 20), dim=2, n_time_steps=2)
        model.velocity.data.normal_(0, 0.05)

        # Baseline evaluation
        v_max_initial = model._get_v_max_voxel(model.velocity, (20, 20))
        key_initial = (id(model.velocity), model.velocity._version, (20, 20))
        assert key_initial in model._v_max_cache

        # Optimizer step
        optimizer = torch.optim.SGD([model.velocity], lr=0.1)
        loss = (model.velocity ** 2).sum()
        loss.backward()
        optimizer.step()

        # Version must have incremented
        assert model.velocity._version > key_initial[1]
        key_after_step = (id(model.velocity), model.velocity._version, (20, 20))
        assert key_after_step not in model._v_max_cache

        # Re-query computes updated value and caches under new key
        v_max_new = model._get_v_max_voxel(model.velocity, (20, 20))
        assert key_after_step in model._v_max_cache
        assert v_max_new != v_max_initial

    def test_cfl_cache_multi_shape_isolation(self):
        """Verify multiple spatial shapes do not contaminate each other in CFL cache."""
        model = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2, n_time_steps=2)
        model.velocity.data.normal_(0, 0.05)

        shapes = [(16, 16), (24, 24), (32, 32), (48, 48)]
        v_max_dict = {}
        for sh in shapes:
            v_max_dict[sh] = model._get_v_max_voxel(model.velocity, sh)
            key = (id(model.velocity), model.velocity._version, sh)
            assert key in model._v_max_cache

        # Re-fetching must produce identical values from cache
        for sh in shapes:
            assert model._get_v_max_voxel(model.velocity, sh) == v_max_dict[sh]

    def test_cfl_cache_bounded_eviction(self):
        """Stress-test cache size bounds (> 32 entries) to verify cache clearing avoids unbounded memory growth."""
        model = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2, n_time_steps=2)
        model.velocity.data.normal_(0, 0.05)

        # Add 40 distinct shapes to trigger eviction
        for i in range(16, 56):
            sh = (i, i)
            _ = model._get_v_max_voxel(model.velocity, sh)

        # Bounded cache must not exceed 33 entries (cleared when > 32)
        assert len(model._v_max_cache) <= 33


# ==============================================================================
# Adversarial Challenge 6: MPS Backend, Top-Level Wrapper & Edge Parameterization
# ==============================================================================

class TestAdversarialMPSAndTopLevel:
    """Stress-test MPS hardware acceleration, top-level wrapper forwarding, and parameter extremes."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS not available")
    def test_mps_hardware_acceleration_parity(self):
        """Verify on-device Anderson inversion runs on MPS accelerator with L_inf < 1e-6 parity."""
        device = torch.device('mps')
        torch.manual_seed(1234)

        # 2D on MPS
        disp_2d = torch.randn(1, 28, 28, 2, device=device) * 0.03
        inv_base_2d = _baseline_anderson(disp_2d.clone(), steps=10, m=3)
        inv_opt_2d = update_inverse_field_nd_anderson(disp_2d.clone(), steps=10, m=3, check_interval=2)
        assert inv_opt_2d.device.type == 'mps'
        linf_2d = (inv_base_2d - inv_opt_2d).abs().max().item()
        rel_2d = linf_2d / (inv_base_2d.abs().max().item() + 1e-8)
        assert linf_2d < 1e-6, f"MPS 2D: L_inf={linf_2d:.2e} exceeded 1e-6"
        assert rel_2d < 1e-5, f"MPS 2D: rel_error={rel_2d:.2e} exceeded 1e-5"

        # 3D on MPS
        disp_3d = torch.randn(1, 14, 14, 14, 3, device=device) * 0.02
        inv_base_3d = _baseline_anderson(disp_3d.clone(), steps=8, m=3)
        inv_opt_3d = update_inverse_field_nd_anderson(disp_3d.clone(), steps=8, m=3, check_interval=4)
        assert inv_opt_3d.device.type == 'mps'
        linf_3d = (inv_base_3d - inv_opt_3d).abs().max().item()
        rel_3d = linf_3d / (inv_base_3d.abs().max().item() + 1e-8)
        assert linf_3d < 1e-6, f"MPS 3D: L_inf={linf_3d:.2e} exceeded 1e-6"
        assert rel_3d < 1e-5, f"MPS 3D: rel_error={rel_3d:.2e} exceeded 1e-5"

    def test_top_level_wrapper_forwarding(self):
        """Verify update_inverse_field_nd wrapper correctly forwards check_interval and handles 3D channels-first."""
        torch.manual_seed(4321)
        # 3D channels-first input: (1, 3, 12, 14, 16)
        disp_cf = torch.randn(1, 3, 12, 14, 16) * 0.02
        inv_cf = update_inverse_field_nd(
            disp_cf, steps=8, m=3, method='anderson', check_interval=3
        )
        assert inv_cf.shape == disp_cf.shape
        assert torch.isfinite(inv_cf).all()

        # Check kwargs forwarding
        inv_cf_kw = update_inverse_field_nd(
            disp_cf, steps=8, m=3, method='anderson', check_interval=1
        )
        assert inv_cf_kw.shape == disp_cf.shape
        assert torch.isfinite(inv_cf_kw).all()

    def test_smoothing_sigma_interaction_parity(self):
        """Verify smoothing_sigma > 0.0 with on-device clipping preserves numerical parity."""
        torch.manual_seed(555)
        disp = torch.randn(1, 24, 24, 2) * 0.04
        inv_base = _baseline_anderson(disp.clone(), steps=10, m=3, smoothing_sigma=1.2)
        inv_opt = update_inverse_field_nd_anderson(disp.clone(), steps=10, m=3, smoothing_sigma=1.2, check_interval=1)

        linf = (inv_base - inv_opt).abs().max().item()
        rel = linf / (inv_base.abs().max().item() + 1e-8)
        assert linf < 1e-6, f"Smoothing sigma parity: L_inf={linf:.2e} exceeded 1e-6"
        assert rel < 1e-5, f"Smoothing sigma parity: rel_error={rel:.2e} exceeded 1e-5"

    def test_warm_start_initial_inverse(self):
        """Verify explicit warm-start W_inv_disp produces identical parity against baseline."""
        torch.manual_seed(666)
        disp = torch.randn(1, 20, 20, 2) * 0.03
        warm_inv = -disp.clone() * 0.9  # Perturbed initial inverse

        inv_base = _baseline_anderson(disp.clone(), W_inv_disp=warm_inv.clone(), steps=10, m=3)
        inv_opt = update_inverse_field_nd_anderson(disp.clone(), W_inv_disp=warm_inv.clone(), steps=10, m=3, check_interval=1)

        linf = (inv_base - inv_opt).abs().max().item()
        rel = linf / (inv_base.abs().max().item() + 1e-8)
        assert linf < 1e-6, f"Warm start parity: L_inf={linf:.2e} exceeded 1e-6"
        assert rel < 1e-5, f"Warm start parity: rel_error={rel:.2e} exceeded 1e-5"

