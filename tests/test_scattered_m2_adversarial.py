"""Adversarial stress tests and empirical verification for M2: Adam & Anderson in SyNScattered."""

import math
import time
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from syntx.scattered.solver import SyNScattered, ScatteredRegistrationConfig
import syntx.scattered.solver as solver_mod
from syntx.core.inverse import compute_inverse_identity_error_nd


def test_in_loop_inv_interval_zero_degradation():
    """Verify that forward loss trajectory has strictly zero degradation (delta = 0.0)

    across intervals [1, 2, 5, 10] while execution throughput scales.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    N = 150
    pts_f = torch.randn(N, 2) * 0.4
    feat_f = torch.randn(N, 2)
    pts_m = pts_f + 0.03 * torch.sin(pts_f * 3.14159)
    feat_m = feat_f.clone()

    intervals = [1, 2, 5, 10]
    results = {}
    timings = {}

    for inv_int in intervals:
        model = SyNScattered(
            grid_res=32,
            iterations=20,
            optimizer_type='adam',
            in_loop_inv_steps=5,
            in_loop_inv_interval=inv_int,
        )
        t0 = time.perf_counter()
        res = model.fit(pts_f, feat_f, pts_m, feat_m)
        t1 = time.perf_counter()
        results[inv_int] = res
        timings[inv_int] = t1 - t0

    base_loss = results[1].loss_history
    for inv_int in [2, 5, 10]:
        curr_loss = results[inv_int].loss_history
        assert len(base_loss) == len(curr_loss)
        max_delta = max(abs(a - b) for a, b in zip(base_loss, curr_loss))
        assert max_delta == 0.0, f"Forward loss degraded for interval {inv_int}: max delta {max_delta}"

    # Verify throughput improvement (interval 10 should be faster than interval 1)
    assert timings[10] < timings[1], f"Expected interval 10 ({timings[10]:.3f}s) to be faster than interval 1 ({timings[1]:.3f}s)"


def test_in_loop_inv_interval_multires_pyramid():
    """Verify zero degradation in multi-resolution pyramid registration."""
    torch.manual_seed(42)
    N = 100
    pts_f = torch.randn(N, 2) * 0.4
    feat_f = torch.randn(N, 1)
    pts_m = pts_f + 0.02
    feat_m = feat_f.clone()

    res1 = SyNScattered(
        grid_res=32,
        levels=[2, 1],
        epochs_per_level=[10, 10],
        optimizer_type='adam',
        in_loop_inv_steps=5,
        in_loop_inv_interval=1,
    ).fit(pts_f, feat_f, pts_m, feat_m)

    res5 = SyNScattered(
        grid_res=32,
        levels=[2, 1],
        epochs_per_level=[10, 10],
        optimizer_type='adam',
        in_loop_inv_steps=5,
        in_loop_inv_interval=5,
    ).fit(pts_f, feat_f, pts_m, feat_m)

    max_delta = max(abs(a - b) for a, b in zip(res1.loss_history, res5.loss_history))
    assert max_delta == 0.0, f"Multi-resolution loss degraded: max delta {max_delta}"


def test_final_epoch_inverse_update_guarantee():
    """Verify that in-loop inverse is guaranteed to execute on the final epoch

    even when (epoch + 1) % interval != 0.
    """
    orig_update = solver_mod.update_inverse_field_nd_anderson
    calls = []

    def logging_update(w, w_inv, **kwargs):
        calls.append(w.shape)
        return orig_update(w, w_inv, **kwargs)

    solver_mod.update_inverse_field_nd_anderson = logging_update

    try:
        torch.manual_seed(42)
        pts_f = torch.randn(50, 2) * 0.4
        feat_f = torch.randn(50, 1)
        pts_m = pts_f + 0.02
        feat_m = feat_f.clone()

        # 7 epochs, interval 5:
        # epoch 4: (4+1)%5 == 0 -> triggers (2 calls: l2r, r2l)
        # epoch 6: (6+1)%5 != 0, but epoch == 7 - 1 -> triggers (2 calls: l2r, r2l)
        model = SyNScattered(
            grid_res=32,
            iterations=7,
            optimizer_type='adam',
            in_loop_inv_steps=5,
            in_loop_inv_interval=5,
            inverse_steps=0, # isolate in-loop
        )
        model.fit(pts_f, feat_f, pts_m, feat_m)
        assert len(calls) == 4, f"Expected exactly 4 calls (2 epochs x 2 warps), got {len(calls)}"
    finally:
        solver_mod.update_inverse_field_nd_anderson = orig_update


def test_final_epoch_inverse_update_multires_pyramid():
    """Verify final epoch inverse update across each multi-resolution pyramid level."""
    orig_update = solver_mod.update_inverse_field_nd_anderson
    calls_by_res = []

    def logging_update(w, w_inv, **kwargs):
        calls_by_res.append(w.shape[1])
        return orig_update(w, w_inv, **kwargs)

    solver_mod.update_inverse_field_nd_anderson = logging_update

    try:
        torch.manual_seed(42)
        pts = torch.randn(50, 2) * 0.4
        feat = torch.randn(50, 1)

        # 3 levels: [4, 2, 1] downsampling, epochs [7, 8, 9]
        model = SyNScattered(
            grid_res=32,
            levels=[4, 2, 1],
            epochs_per_level=[7, 8, 9],
            optimizer_type='adam',
            in_loop_inv_steps=5,
            in_loop_inv_interval=5,
            inverse_steps=0,
        )
        model.fit(pts, feat, pts + 0.01, feat)

        assert calls_by_res.count(8) == 4   # level 0 (size 8): epoch 4 and 6
        assert calls_by_res.count(16) == 4  # level 1 (size 16): epoch 4 and 7
        assert calls_by_res.count(32) == 4  # level 2 (size 32): epoch 4 and 8
    finally:
        solver_mod.update_inverse_field_nd_anderson = orig_update


def test_inverse_field_accuracy_across_intervals():
    """Verify final inverse field accuracy across intervals with and without post refinement."""
    torch.manual_seed(42)
    N = 100
    pts_f = torch.randn(N, 2) * 0.4
    feat_f = torch.randn(N, 2)
    pts_m = pts_f + 0.03 * torch.sin(pts_f * 3.14159)
    feat_m = feat_f.clone()

    for inv_int in [1, 2, 5, 10]:
        # Default inverse_steps=20
        res = SyNScattered(
            grid_res=32,
            iterations=15,
            optimizer_type='adam',
            in_loop_inv_steps=5,
            in_loop_inv_interval=inv_int,
            inverse_steps=20,
        ).fit(pts_f, feat_f, pts_m, feat_m)
        assert res.inverse_identity_errors['mean_error'] < 0.005
        assert res.inverse_identity_errors['max_error'] < 0.05

        # Pure in-loop (inverse_steps=0)
        res_inloop = SyNScattered(
            grid_res=32,
            iterations=15,
            optimizer_type='adam',
            in_loop_inv_steps=5,
            in_loop_inv_interval=inv_int,
            inverse_steps=0,
        ).fit(pts_f, feat_f, pts_m, feat_m)
        assert res_inloop.inverse_identity_errors['mean_error'] < 5e-3


def test_zero_points_and_empty_inputs_validation():
    """Verify proper validation errors for empty / None inputs."""
    model = SyNScattered(grid_res=16, iterations=2)
    with pytest.raises(ValueError, match="Fixed points tensor cannot be empty"):
        model.fit(torch.empty((0, 2)), torch.empty((0, 1)), torch.randn(10, 2), torch.randn(10, 1))

    with pytest.raises(ValueError, match="Moving points tensor cannot be empty"):
        model.fit(torch.randn(10, 2), torch.randn(10, 1), torch.empty((0, 2)), torch.empty((0, 1)))

    with pytest.raises(ValueError, match="Fixed points tensor cannot be empty"):
        model.fit(np.empty((0, 2)), np.empty((0, 1)), np.random.randn(10, 2), np.random.randn(10, 1))

    with pytest.raises(ValueError, match="Must provide either fixed_points or fixed_grid"):
        model.fit()


def test_small_point_clouds_2d_and_3d():
    """Stress test minimal point clouds (N=1, 2, 3) in 2D and 3D."""
    for d in [2, 3]:
        for N in [1, 2, 3]:
            torch.manual_seed(42)
            pts_f = torch.randn(N, d) * 0.2
            feat_f = torch.ones(N, 1)
            pts_m = pts_f + 0.05
            feat_m = feat_f.clone()

            model = SyNScattered(
                dim=d,
                grid_res=16,
                iterations=5,
                optimizer_type='adam',
                in_loop_inv_steps=3,
                in_loop_inv_interval=2,
            )
            res = model.fit(pts_f, feat_f, pts_m, feat_m)
            assert not torch.isnan(res.disp_fwd).any()
            assert not torch.isnan(res.disp_inv).any()
            assert len(res.loss_history) == 5


def test_extreme_learning_rates():
    """Stress test extreme learning rates (zero, tiny, astronomical with CFL clamping)."""
    torch.manual_seed(42)
    pts_f = torch.randn(30, 2) * 0.3
    feat_f = torch.ones(30, 1)
    pts_m = pts_f + 0.03
    feat_m = feat_f.clone()

    # Zero LR
    res_zero = SyNScattered(grid_res=16, iterations=5, optimizer_type='adam', optimizer_lr=0.0).fit(pts_f, feat_f, pts_m, feat_m)
    assert abs(res_zero.loss_history[-1] - res_zero.loss_history[0]) < 1e-5
    assert torch.allclose(res_zero.disp_fwd, torch.zeros_like(res_zero.disp_fwd))

    # Microscopic LR
    res_tiny = SyNScattered(grid_res=16, iterations=5, optimizer_type='adam', optimizer_lr=1e-15).fit(pts_f, feat_f, pts_m, feat_m)
    assert not torch.isnan(res_tiny.disp_fwd).any()

    # Astronomical LR clamped by CFL bounding
    res_huge = SyNScattered(grid_res=16, iterations=5, optimizer_type='adam', optimizer_lr=1e6).fit(pts_f, feat_f, pts_m, feat_m)
    assert not torch.isnan(res_huge.disp_fwd).any()
    assert not torch.isinf(res_huge.disp_fwd).any()
    assert res_huge.disp_fwd.abs().max().item() < 2.0


def test_boundary_configurations():
    """Verify extreme/boundary configurations for in_loop_inv_interval and iterations."""
    torch.manual_seed(42)
    pts_f = torch.randn(20, 2) * 0.3
    feat_f = torch.ones(20, 1)
    pts_m = pts_f + 0.02
    feat_m = feat_f.clone()

    # in_loop_inv_interval <= 0 clamped to 1
    m0 = SyNScattered(grid_res=16, iterations=3, in_loop_inv_interval=0)
    res0 = m0.fit(pts_f, feat_f, pts_m, feat_m)
    assert not torch.isnan(res0.disp_fwd).any()

    # in_loop_inv_interval > iterations
    m100 = SyNScattered(grid_res=16, iterations=3, in_loop_inv_interval=100)
    res100 = m100.fit(pts_f, feat_f, pts_m, feat_m)
    assert not torch.isnan(res100.disp_fwd).any()

    # iterations = 0
    m_zero = SyNScattered(grid_res=16, iterations=0)
    res_zero = m_zero.fit(pts_f, feat_f, pts_m, feat_m)
    assert res_zero.disp_fwd.abs().max().item() == 0.0


def test_adam_factored_inplace_parity():
    """Verify mathematical parity between original and optimized factored Adam."""
    shape = (1, 32, 32, 2)
    m_orig = torch.zeros(shape)
    v_orig = torch.zeros(shape)
    m_opt = torch.zeros(shape)
    v_opt = torch.zeros(shape)

    lr, beta1, beta2, eps = 0.05, 0.9, 0.999, 1e-8
    torch.manual_seed(1234)

    for step in range(1, 101):
        grad = torch.randn(shape) * 5.0
        # Original
        m_orig = beta1 * m_orig + (1.0 - beta1) * grad
        v_orig = beta2 * v_orig + (1.0 - beta2) * (grad ** 2)
        m_hat = m_orig / (1.0 - beta1 ** step)
        v_hat = v_orig / (1.0 - beta2 ** step)
        delta_orig = lr * m_hat / (torch.sqrt(v_hat) + eps)

        # Optimized
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        step_size = (lr * bias_correction2_sqrt) / bias_correction1
        eps_scaled = eps * bias_correction2_sqrt

        m_opt.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v_opt.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        delta_opt = (m_opt * step_size).div_(v_opt.sqrt().add_(eps_scaled))

        diff = (delta_orig - delta_opt).abs().max().item()
        assert diff < 1e-7, f"Adam divergence at step {step}: {diff}"


def test_lagrangian_composition_bitwise_parity_and_contiguity():
    """Verify exact bitwise identity and contiguous storage for Lagrangian composition."""
    shape = (1, 32, 32, 2)
    grids = [torch.linspace(-1, 1, s) for s in (32, 32)]
    mesh = torch.meshgrid(*grids, indexing='ij')
    identity = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)

    warp_old = torch.zeros(shape)
    warp_new = torch.zeros(shape)
    torch.manual_seed(777)

    for step in range(30):
        delta = torch.randn(shape) * 0.02
        delta_old_pb = F.grid_sample(
            delta.movedim(-1, 1).contiguous(),
            (identity + warp_old).contiguous(),
            padding_mode='border',
            align_corners=True
        ).movedim(1, -1).contiguous()
        warp_old.sub_(delta_old_pb)

        delta_new_pb = F.grid_sample(
            delta.movedim(-1, 1),
            identity + warp_new,
            padding_mode='border',
            align_corners=True
        ).movedim(1, -1)
        warp_new.sub_(delta_new_pb)

        assert (warp_old - warp_new).abs().max().item() == 0.0
        assert warp_new.is_contiguous()
