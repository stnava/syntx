import pytest
import torch
import numpy as np

from syntx.core.inverse import (
    update_inverse_field_nd,
    update_inverse_field_nd_anderson,
    update_inverse_field_nd_hybrid_lm,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
    integrate_time_varying_velocity_field,
)


def test_inverse_identity_error_zero_displacement():
    disp = torch.zeros(1, 16, 16, 2)
    err = compute_inverse_identity_error_nd(disp, disp, is_displacement=True)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-6)

    err_dict = calculate_inverse_identity_error(disp, disp, spacing=[1.0, 1.0], origin=[0.0, 0.0], direction=np.eye(2))
    assert np.isclose(err_dict['max_error'], 0.0, atol=1e-5)
    assert np.isclose(err_dict['mean_error'], 0.0, atol=1e-5)


def test_update_inverse_field_small_displacement():
    # Small synthetic translation
    disp = torch.zeros(1, 16, 16, 2)
    disp[..., 0] = 0.05
    disp[..., 1] = -0.05

    inv_anderson = update_inverse_field_nd(disp, method='anderson', steps=15)
    assert inv_anderson.shape == disp.shape

    # Inverse should be approximately negative displacement
    assert torch.allclose(inv_anderson[0, 4:12, 4:12, 0], -disp[0, 4:12, 4:12, 0], atol=1e-2)
    assert torch.allclose(inv_anderson[0, 4:12, 4:12, 1], -disp[0, 4:12, 4:12, 1], atol=1e-2)


def test_integrate_time_varying_velocity_field_zero():
    v = [torch.zeros(1, 16, 16, 2) for _ in range(4)]
    phi = integrate_time_varying_velocity_field(v, dt=0.25, mode='forward')
    assert torch.allclose(phi, torch.zeros_like(v[0]), atol=1e-6)


def test_anderson_on_device_parity_2d_and_3d():
    """Verify on-device Anderson inversion produces finite fields and low inverse error."""
    # 2D normalized space
    torch.manual_seed(42)
    disp_2d = torch.randn(1, 24, 24, 2) * 0.03
    inv_2d = update_inverse_field_nd_anderson(disp_2d, None, steps=10, m=3, check_interval=1)
    assert inv_2d.shape == disp_2d.shape
    assert torch.isfinite(inv_2d).all()

    # 2D physical space
    inv_2d_phys = update_inverse_field_nd_anderson(
        disp_2d, None, steps=10, m=3,
        spacing=(1.5, 1.5), origin=(0.0, 0.0), direction=np.eye(2).flatten(),
        check_interval=1
    )
    assert inv_2d_phys.shape == disp_2d.shape
    assert torch.isfinite(inv_2d_phys).all()

    # 3D normalized space
    disp_3d = torch.randn(1, 12, 12, 12, 3) * 0.02
    inv_3d = update_inverse_field_nd_anderson(disp_3d, None, steps=8, m=3, check_interval=1)
    assert inv_3d.shape == disp_3d.shape
    assert torch.isfinite(inv_3d).all()

    # 3D physical space
    inv_3d_phys = update_inverse_field_nd_anderson(
        disp_3d, None, steps=8, m=3,
        spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=np.eye(3).flatten(),
        check_interval=1
    )
    assert inv_3d_phys.shape == disp_3d.shape
    assert torch.isfinite(inv_3d_phys).all()

    # Inverse identity error verification
    err_2d = compute_inverse_identity_error_nd(disp_2d, inv_2d, is_displacement=True)
    assert err_2d.abs().mean().item() < 0.05


def test_anderson_check_interval_behavior():
    """Verify check_interval parameter controls stopping check cadence."""
    torch.manual_seed(123)
    disp = torch.randn(1, 16, 16, 2) * 0.02

    # check_interval=1 vs check_interval=3 vs check_interval=10
    inv_1 = update_inverse_field_nd_anderson(disp, None, steps=10, m=3, check_interval=1)
    inv_3 = update_inverse_field_nd_anderson(disp, None, steps=10, m=3, check_interval=3)
    inv_10 = update_inverse_field_nd_anderson(disp, None, steps=10, m=3, check_interval=10)
    inv_none = update_inverse_field_nd_anderson(disp, None, steps=10, m=3, check_interval=None)
    inv_zero = update_inverse_field_nd_anderson(disp, None, steps=10, m=3, check_interval=0)

    for inv in [inv_1, inv_3, inv_10, inv_none, inv_zero]:
        assert inv.shape == disp.shape
        assert torch.isfinite(inv).all()

    # When thresholds are very high, stopping triggers on the first check iteration
    inv_stop_step1 = update_inverse_field_nd_anderson(
        disp, None, steps=10, m=3,
        max_error_threshold=100.0, mean_error_threshold=100.0,
        check_interval=1
    )
    inv_stop_step3 = update_inverse_field_nd_anderson(
        disp, None, steps=10, m=3,
        max_error_threshold=100.0, mean_error_threshold=100.0,
        check_interval=3
    )
    # check_interval=1 stops after 1 step; check_interval=3 stops after 3 steps
    assert not torch.allclose(inv_stop_step1, inv_stop_step3, atol=1e-5)


def test_anderson_channels_first():
    """Verify channels-first tensor input is properly preserved on output."""
    torch.manual_seed(42)
    disp_cf = torch.randn(1, 2, 16, 16) * 0.02
    inv_cf = update_inverse_field_nd_anderson(disp_cf, steps=8, m=2, check_interval=1)
    assert inv_cf.shape == disp_cf.shape
    assert torch.isfinite(inv_cf).all()


def test_tvf_cfl_version_and_shape_cache():
    """Verify TVFModel._get_v_max_voxel caches results and invalidates properly."""
    from syntx.tvf import TVFModel

    model = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2, n_time_steps=2)
    model.velocity.data.normal_(0, 0.05)

    # Initial computation
    v_max_1 = model._get_v_max_voxel(model.velocity, (16, 16))
    cache_key_1 = (id(model.velocity), model.velocity._version, (16, 16))
    assert hasattr(model, '_v_max_cache')
    assert cache_key_1 in model._v_max_cache
    assert model._v_max_cache[cache_key_1] == v_max_1

    # Cache hit (must return identical float without recomputing)
    v_max_hit = model._get_v_max_voxel(model.velocity, (16, 16))
    assert v_max_hit == v_max_1

    # Different shape triggers a new cache entry
    v_max_shape2 = model._get_v_max_voxel(model.velocity, (32, 32))
    cache_key_2 = (id(model.velocity), model.velocity._version, (32, 32))
    assert cache_key_2 in model._v_max_cache

    # In-place parameter update increments version and invalidates old cache entry
    v_old_version = model.velocity._version
    with torch.no_grad():
        model.velocity.add_(0.02)
    assert model.velocity._version > v_old_version

    v_max_updated = model._get_v_max_voxel(model.velocity, (16, 16))
    cache_key_updated = (id(model.velocity), model.velocity._version, (16, 16))
    assert cache_key_updated in model._v_max_cache
    assert cache_key_updated != cache_key_1

    # Test integration runs cleanly with CFL cache
    disp = model.integrate(0.0, 1.0)
    assert disp.shape == (1, 16, 16, 2)
    assert torch.isfinite(disp).all()


def test_anderson_baseline_numerical_parity():
    """Verify bitwise or near-bitwise parity (L_inf < 1e-6, relative error < 1e-5) against baseline."""
    import torch.nn.functional as F
    from syntx.core.inverse import get_boundary_mask

    def baseline_anderson(
        W_disp: torch.Tensor,
        W_inv_disp: torch.Tensor = None,
        steps: int = 30,
        m: int = 5,
        max_error_threshold: float = 0.1,
        mean_error_threshold: float = 0.001,
    ) -> torch.Tensor:
        B = W_disp.shape[0]
        dim = W_disp.shape[-1]
        spatial = W_disp.shape[1:-1]
        device = W_disp.device
        dtype = W_disp.dtype

        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )
        W_disp_cf = torch.movedim(W_disp, -1, 1)

        if W_inv_disp is None:
            W_inv_disp = -W_disp.clone()

        def itk_fixed_point_step(v_curr, iteration):
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

        return v_k

    for seed in [42, 99]:
        torch.manual_seed(seed)
        disp = torch.randn(1, 20, 20, 2) * 0.04
        v_base = baseline_anderson(disp.clone(), None, steps=10, m=3)
        v_opt = update_inverse_field_nd_anderson(disp.clone(), None, steps=10, m=3, check_interval=1)
        linf = (v_base - v_opt).abs().max().item()
        rel = linf / (v_base.abs().max().item() + 1e-8)
        assert linf < 1e-6, f"L_inf={linf} exceeded 1e-6"
        assert rel < 1e-5, f"rel_error={rel} exceeded 1e-5"


