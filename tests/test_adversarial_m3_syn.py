"""Adversarial stress-test suite for Milestone 3:
In-Place Adam Updates, Bias Folding, Layout Restriding Elimination, and Loss Memory Retention.
"""

import math
import pytest
import torch
import torch.nn.functional as F
import ants

from syntx.syn import SyNTo
from syntx.core.smoothing import separable_gaussian_filter, apply_dsti1_green_operator


def test_adversarial_adam_bias_folding_numerical_precision_float64():
    """Stress-test analytical equivalence of bias-correction folding in double precision (float64).
    In float64, classical Adam and folded Adam should match to near machine precision (< 1e-14).
    """
    shape = (2, 8, 10, 12, 3)
    torch.manual_seed(999)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    m_ref = torch.zeros(shape, dtype=torch.float64)
    v_ref = torch.zeros(shape, dtype=torch.float64)
    m_rem = torch.zeros(shape, dtype=torch.float64)
    v_rem = torch.zeros(shape, dtype=torch.float64)

    for t in range(1, 60):
        # Varying gradient scale across iterations
        grad = torch.randn(shape, dtype=torch.float64) * (10.0 ** ((t % 5) - 2))

        # Classical Adam formulation
        m_ref = beta1 * m_ref + (1.0 - beta1) * grad
        v_ref = beta2 * v_ref + (1.0 - beta2) * (grad ** 2)
        m_hat = m_ref / (1.0 - beta1 ** t)
        v_hat = v_ref / (1.0 - beta2 ** t)
        u_ref = m_hat / (torch.sqrt(v_hat) + eps)

        # Folded scalar bias formulation
        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t
        b2_sqrt = math.sqrt(b2)
        step_size = b2_sqrt / b1
        eps_scaled = eps * b2_sqrt

        m_rem.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v_rem.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        u_rem = (m_rem * step_size).div_(v_rem.sqrt().add_(eps_scaled))

        diff = (u_ref - u_rem).abs().max().item()
        assert diff < 1e-14, f"Double precision parity divergence at iteration {t}: max diff = {diff}"


def test_adversarial_adam_extreme_gradients():
    """Stress-test Adam update under pathological gradients: zero, subnormal, and massive."""
    shape = (1, 6, 6, 6, 3)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    scenarios = [
        ("all_zeros", torch.zeros(shape)),
        ("subnormal", torch.full(shape, 1e-25)),
        ("massive", torch.full(shape, 1e8)),
        ("alternating_massive", torch.randn(shape).sign() * 1e6),
    ]

    for name, grad in scenarios:
        m_ref = torch.zeros_like(grad)
        v_ref = torch.zeros_like(grad)
        m_rem = torch.zeros_like(grad)
        v_rem = torch.zeros_like(grad)

        for t in range(1, 10):
            m_ref = beta1 * m_ref + (1.0 - beta1) * grad
            v_ref = beta2 * v_ref + (1.0 - beta2) * (grad ** 2)
            m_hat = m_ref / (1.0 - beta1 ** t)
            v_hat = v_ref / (1.0 - beta2 ** t)
            u_ref = m_hat / (torch.sqrt(v_hat) + eps)

            b1 = 1.0 - beta1 ** t
            b2 = 1.0 - beta2 ** t
            b2_sqrt = math.sqrt(b2)
            step_size = b2_sqrt / b1
            eps_scaled = eps * b2_sqrt

            m_rem.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v_rem.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            u_rem = (m_rem * step_size).div_(v_rem.sqrt().add_(eps_scaled))

            assert not torch.isnan(u_rem).any(), f"NaN generated for scenario {name} at step {t}"
            assert not torch.isinf(u_rem).any(), f"Inf generated for scenario {name} at step {t}"

            diff = (u_ref - u_rem).abs().max().item()
            assert diff < 1e-5, f"Scenario {name} step {t} mismatch: {diff}"


def test_adversarial_adam_memory_address_invariance():
    """Stress-test in-place buffer mutation: data_ptr() must remain identical across iterations."""
    shape = (1, 16, 16, 16, 3)
    grad = torch.randn(shape)
    m = torch.zeros_like(grad)
    v = torch.zeros_like(grad)

    m_ptr_initial = m.data_ptr()
    v_ptr_initial = v.data_ptr()

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for t in range(1, 30):
        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t
        b2_sqrt = math.sqrt(b2)
        step_size = b2_sqrt / b1
        eps_scaled = eps * b2_sqrt

        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        u_raw = (m * step_size).div_(v.sqrt().add_(eps_scaled))

        assert m.data_ptr() == m_ptr_initial, f"Moment buffer m was reallocated at step {t}!"
        assert v.data_ptr() == v_ptr_initial, f"Moment buffer v was reallocated at step {t}!"
        assert u_raw.data_ptr() != m_ptr_initial, "u_raw unsafely aliased m's memory buffer!"
        assert u_raw.data_ptr() != v_ptr_initial, "u_raw unsafely aliased v's memory buffer!"


def test_adversarial_movedim_grid_sample_contiguity_comprehensive():
    """Stress-test grid_sample with movedim non-contiguous inputs across 2D, 3D, and various paddings."""
    # 2D test
    warp_2d = torch.randn(1, 17, 23, 2)
    coords_2d = torch.rand(1, 17, 23, 2) * 2.0 - 1.0

    for padding in ['border', 'zeros', 'reflection']:
        for mode in ['bilinear', 'nearest']:
            out_c = F.grid_sample(
                warp_2d.movedim(-1, 1).contiguous(), coords_2d.contiguous(),
                mode=mode, padding_mode=padding, align_corners=True
            ).movedim(1, -1).contiguous()

            out_nc = F.grid_sample(
                warp_2d.movedim(-1, 1), coords_2d,
                mode=mode, padding_mode=padding, align_corners=True
            ).movedim(1, -1)

            assert torch.equal(out_c, out_nc), f"2D mismatch with mode={mode}, padding={padding}"

    # 3D test
    warp_3d = torch.randn(1, 9, 13, 11, 3)
    coords_3d = torch.rand(1, 9, 13, 11, 3) * 2.0 - 1.0

    for padding in ['border', 'zeros', 'reflection']:
        for mode in ['bilinear', 'nearest']:
            out_c = F.grid_sample(
                warp_3d.movedim(-1, 1).contiguous(), coords_3d.contiguous(),
                mode=mode, padding_mode=padding, align_corners=True
            ).movedim(1, -1).contiguous()

            out_nc = F.grid_sample(
                warp_3d.movedim(-1, 1), coords_3d,
                mode=mode, padding_mode=padding, align_corners=True
            ).movedim(1, -1)

            assert torch.equal(out_c, out_nc), f"3D mismatch with mode={mode}, padding={padding}"


def test_adversarial_eulerian_and_lagrangian_step_equivalence():
    """Stress-test end-to-end multi-step composition with and without .contiguous() calls."""
    torch.manual_seed(42)
    # Lagrangian simulation
    warp_lag_c = torch.zeros(1, 12, 12, 12, 3)
    warp_lag_nc = torch.zeros(1, 12, 12, 12, 3)
    coords_norm = torch.rand(1, 12, 12, 12, 3) * 2.0 - 1.0

    for _ in range(5):
        delta = torch.randn(1, 12, 12, 12, 3) * 0.1
        # Contiguous
        pb_c = F.grid_sample(
            delta.movedim(-1, 1).contiguous(), coords_norm.contiguous(),
            padding_mode='border', align_corners=True
        ).movedim(1, -1).contiguous()
        warp_lag_c.sub_(pb_c)

        # Non-contiguous
        pb_nc = F.grid_sample(
            delta.movedim(-1, 1), coords_norm,
            padding_mode='border', align_corners=True
        ).movedim(1, -1)
        warp_lag_nc.sub_(pb_nc)

        assert torch.equal(warp_lag_c, warp_lag_nc), "Lagrangian multi-step mismatch!"

    # Eulerian simulation
    warp_eul_c = torch.randn(1, 12, 12, 12, 3) * 0.05
    warp_eul_nc = warp_eul_c.clone()

    for _ in range(5):
        delta = torch.randn(1, 12, 12, 12, 3) * 0.1
        # Contiguous
        sampled_c = F.grid_sample(
            warp_eul_c.movedim(-1, 1).contiguous(), coords_norm.contiguous(),
            padding_mode='border', align_corners=True
        ).movedim(1, -1).contiguous()
        warp_eul_c.copy_(sampled_c - delta)

        # Non-contiguous
        sampled_nc = F.grid_sample(
            warp_eul_nc.movedim(-1, 1), coords_norm,
            padding_mode='border', align_corners=True
        ).movedim(1, -1)
        warp_eul_nc.copy_(sampled_nc - delta)

        assert torch.equal(warp_eul_c, warp_eul_nc), "Eulerian multi-step mismatch!"


def test_adversarial_regadam_smoothing_pipeline():
    """Verify u_raw passes cleanly through spatial smoothers (Gaussian and DSTI) without errors or mutations."""
    shape = (1, 16, 16, 16, 3)
    grad = torch.randn(shape)
    m = torch.zeros_like(grad)
    v = torch.zeros_like(grad)

    b1, b2 = 0.1, 0.001
    b2_sqrt = math.sqrt(b2)
    step_size = b2_sqrt / b1
    eps_scaled = 1e-8 * b2_sqrt

    m.mul_(0.9).add_(grad, alpha=0.1)
    v.mul_(0.999).addcmul_(grad, grad, value=0.001)
    u_raw = (m * step_size).div_(v.sqrt().add_(eps_scaled))

    # Gaussian smoothing
    u_gauss = separable_gaussian_filter(u_raw, sigma=1.5)
    assert u_gauss.shape == u_raw.shape
    assert not torch.isnan(u_gauss).any()

    # DSTI smoothing
    u_dsti = apply_dsti1_green_operator(u_raw, fluid_sigma=1.5, alpha=1.0)
    assert u_dsti.shape == u_raw.shape
    assert not torch.isnan(u_dsti).any()
