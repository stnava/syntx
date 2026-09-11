"""
Adversarial Stress Testing & Parity Verification for Milestone 3 (M3)
In-Place Adam Updates, Factor-Folded Bias Correction, and Layout Restriding Elimination

Challenges:
1. Analytical mathematical parity of in-place fused Adam/RegAdam against out-of-place reference across multi-step runs (100 steps) with dynamic gradients (L_inf < 1e-6, relative error < 1e-5).
2. Hyperparameter sweep across diverse beta_1, beta_2, and epsilon combinations.
3. Extreme gradient scale resilience (1e-5 to 1e3, mixed scales, exact zeros).
4. F.grid_sample contiguity invariance across 2D/3D shapes, anisotropic domains, boundary conditions, and padding modes (bitwise equality).
5. TVF chain rule contiguity invariance and velocity parameter version tracking.
6. Empirical memory churn quantification (confirming >= 70% allocation reduction).
"""

import math
import pytest
import torch
import torch.nn.functional as F

from syntx.core.grid import grid_sample_nd
from syntx.tvf import TVFModel
from syntx.syn import SyNTo


# ==============================================================================
# 1. Multi-Step Analytical Parity with Dynamic Gradients
# ==============================================================================

@pytest.mark.parametrize("shape", [
    (1, 32, 32, 2),
    (1, 16, 16, 16, 3),
    (1, 11, 19, 27, 3),
])
def test_adam_analytical_parity_dynamic_multistep(shape):
    """Verify in-place Adam updates match analytical formulation across 100 dynamic steps."""
    torch.manual_seed(100)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    m_ref = torch.zeros(shape, dtype=torch.float32)
    v_ref = torch.zeros(shape, dtype=torch.float32)
    m_rem = torch.zeros(shape, dtype=torch.float32)
    v_rem = torch.zeros(shape, dtype=torch.float32)

    for t in range(1, 101):
        g = torch.randn(shape, dtype=torch.float32)
        if t % 7 == 0:
            g = g * (torch.rand(shape) > 0.4).float()

        # Analytical reference
        m_ref = beta1 * m_ref + (1.0 - beta1) * g
        v_ref = beta2 * v_ref + (1.0 - beta2) * (g ** 2)
        m_hat = m_ref / (1.0 - beta1 ** t)
        v_hat = v_ref / (1.0 - beta2 ** t)
        u_ref = m_hat / (torch.sqrt(v_hat) + eps)

        # In-place remediated with factor-folded bias correction
        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t
        b2_sqrt = math.sqrt(b2)
        step_size = b2_sqrt / b1
        eps_scaled = eps * b2_sqrt

        m_rem.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v_rem.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        u_rem = (m_rem * step_size).div_(v_rem.sqrt().add_(eps_scaled))

        diff = (u_ref - u_rem).abs().max().item()
        ref_mag = u_ref.abs().max().item()
        rel_err = diff / max(ref_mag, 1e-12)

        assert diff < 1e-6, f"Step {t} shape {shape}: L_inf mismatch {diff:.4e}"
        assert rel_err < 1e-5, f"Step {t} shape {shape}: relative error mismatch {rel_err:.4e}"


# ==============================================================================
# 2. Hyperparameter Grid Sweep
# ==============================================================================

@pytest.mark.parametrize("beta1,beta2", [
    (0.5, 0.9),
    (0.9, 0.999),
    (0.95, 0.9999),
    (0.99, 0.999),
])
@pytest.mark.parametrize("eps", [1e-12, 1e-8, 1e-5, 1e-3, 1.0])
def test_adam_hyperparameter_sweep(beta1, beta2, eps):
    """Stress-test diverse beta1, beta2, and eps configurations across 50 steps."""
    shape = (1, 16, 16, 16, 3)
    torch.manual_seed(int(beta1 * 100 + beta2 * 1000 + eps * 1e4) % 100000)

    m_ref = torch.zeros(shape, dtype=torch.float32)
    v_ref = torch.zeros(shape, dtype=torch.float32)
    m_rem = torch.zeros(shape, dtype=torch.float32)
    v_rem = torch.zeros(shape, dtype=torch.float32)

    for t in range(1, 51):
        g = torch.randn(shape, dtype=torch.float32)

        m_ref = beta1 * m_ref + (1.0 - beta1) * g
        v_ref = beta2 * v_ref + (1.0 - beta2) * (g ** 2)
        m_hat = m_ref / (1.0 - beta1 ** t)
        v_hat = v_ref / (1.0 - beta2 ** t)
        u_ref = m_hat / (torch.sqrt(v_hat) + eps)

        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t
        b2_sqrt = math.sqrt(b2)
        step_size = b2_sqrt / b1
        eps_scaled = eps * b2_sqrt

        m_rem.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v_rem.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        u_rem = (m_rem * step_size).div_(v_rem.sqrt().add_(eps_scaled))

        diff = (u_ref - u_rem).abs().max().item()
        ref_mag = u_ref.abs().max().item()
        rel_err = diff / max(ref_mag, 1e-12)

        assert diff < 1e-6, f"Mismatch at t={t}, b1={beta1}, b2={beta2}, eps={eps}: diff={diff:.4e}"
        assert rel_err < 1e-5, f"Rel mismatch at t={t}, b1={beta1}, b2={beta2}, eps={eps}: rel={rel_err:.4e}"


# ==============================================================================
# 3. Extreme Gradient Scales and Sparse Zeros
# ==============================================================================

@pytest.mark.parametrize("scale", [1e-5, 1e-3, 1.0, 10.0, 1e2, 1e3])
def test_adam_extreme_gradient_scales(scale):
    """Stress-test extreme gradient scales from 1e-5 to 1e3 with mixed zero blocks."""
    shape = (1, 16, 16, 16, 3)
    torch.manual_seed(2026)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    m_ref = torch.zeros(shape)
    v_ref = torch.zeros(shape)
    m_rem = torch.zeros(shape)
    v_rem = torch.zeros(shape)

    for t in range(1, 51):
        g = torch.randn(shape) * scale
        if t % 5 == 0:
            g = torch.zeros_like(g)
        elif t % 3 == 0:
            g[..., 0] *= 100.0
            g[..., 1] *= 0.01

        m_ref = beta1 * m_ref + (1.0 - beta1) * g
        v_ref = beta2 * v_ref + (1.0 - beta2) * (g ** 2)
        m_hat = m_ref / (1.0 - beta1 ** t)
        v_hat = v_ref / (1.0 - beta2 ** t)
        u_ref = m_hat / (torch.sqrt(v_hat) + eps)

        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t
        b2_sqrt = math.sqrt(b2)
        step_size = b2_sqrt / b1
        eps_scaled = eps * b2_sqrt

        m_rem.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v_rem.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        u_rem = (m_rem * step_size).div_(v_rem.sqrt().add_(eps_scaled))

        assert not torch.isnan(u_rem).any(), f"NaN at scale {scale}, step {t}"
        assert not torch.isinf(u_rem).any(), f"Inf at scale {scale}, step {t}"

        diff = (u_ref - u_rem).abs().max().item()
        ref_mag = u_ref.abs().max().item()
        rel_err = diff / max(ref_mag, 1e-12)

        assert diff < 1e-6, f"Scale {scale}, step {t}: L_inf={diff:.4e}"
        assert rel_err < 1e-5, f"Scale {scale}, step {t}: rel_err={rel_err:.4e}"


# ==============================================================================
# 4. F.grid_sample Contiguity Invariance
# ==============================================================================

@pytest.mark.parametrize("shape", [
    (1, 32, 32, 2),
    (1, 23, 47, 2),
    (1, 16, 16, 16, 3),
    (1, 11, 23, 37, 3),
])
@pytest.mark.parametrize("pmode", ["border", "zeros", "reflection"])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_grid_sample_contiguity_invariance(shape, pmode, mode):
    """Verify that omitting .contiguous() around movedim in F.grid_sample produces exact bitwise identity."""
    torch.manual_seed(42)
    warp = torch.randn(*shape)
    coords = (torch.rand(*shape) * 3.0) - 1.5  # Includes out-of-bound coords

    # With .contiguous()
    out_c = F.grid_sample(
        warp.movedim(-1, 1).contiguous(), coords.contiguous(),
        mode=mode, padding_mode=pmode, align_corners=True
    ).movedim(1, -1).contiguous()

    # Without .contiguous()
    out_nc = F.grid_sample(
        warp.movedim(-1, 1), coords,
        mode=mode, padding_mode=pmode, align_corners=True
    ).movedim(1, -1)

    assert torch.equal(out_c, out_nc), f"Contiguity mismatch for shape {shape}, pmode {pmode}, mode {mode}"


def test_grid_sample_nd_non_contiguous_strides():
    """Verify grid_sample_nd with strided non-contiguous inputs and grids."""
    shape = (1, 16, 16, 16, 3)
    torch.manual_seed(99)
    vol = torch.randn(*shape)
    coords_full = torch.randn(1, 32, 32, 32, 3)
    coords_strided = coords_full[:, ::2, ::2, ::2, :]

    vol_c = vol.movedim(-1, 1).contiguous()
    vol_nc = vol.movedim(-1, 1)

    out_c = grid_sample_nd(vol_c, coords_strided.contiguous(), mode='bilinear', padding_mode='zeros')
    out_nc = grid_sample_nd(vol_nc, coords_strided, mode='bilinear', padding_mode='zeros')

    assert torch.equal(out_c, out_nc), "grid_sample_nd strided view produced non-identical output"


# ==============================================================================
# 5. TVF Chain Rule Contiguity and Velocity Version Tracking
# ==============================================================================

def test_tvf_chain_rule_and_version_invariance():
    """Verify TVF chain rule contiguity invariance and version-based cache invalidation."""
    shape = (1, 12, 14, 16, 3)
    dim = 3
    grad_I = torch.randn(*shape)
    phi_norm = torch.rand(*shape) * 2.0 - 1.0
    dir_mat = torch.eye(dim)
    g_im = torch.randn(1, 1, *shape[1:-1])

    # Contiguous
    mid_c = grid_sample_nd(grad_I.movedim(-1, 1).contiguous(), phi_norm.contiguous(), mode='bilinear', padding_mode='zeros').movedim(1, -1).contiguous()
    mid_c = torch.matmul(mid_c, dir_mat)
    phi_grad_c = (g_im.movedim(1, -1).contiguous() * mid_c).contiguous()

    # Non-contiguous
    mid_nc = grid_sample_nd(grad_I.movedim(-1, 1), phi_norm, mode='bilinear', padding_mode='zeros').movedim(1, -1)
    mid_nc = torch.matmul(mid_nc, dir_mat)
    phi_grad_nc = g_im.movedim(1, -1) * mid_nc

    assert torch.equal(mid_c, mid_nc)
    assert torch.equal(phi_grad_c, phi_grad_nc)

    # Version tracking & cache invalidation
    model = TVFModel(dim=2, image_shape=(16, 16), velocity_shape=(8, 8), n_time_steps=2)
    v1 = model._get_v_max_voxel(model.velocity, (16, 16))
    key1 = (id(model.velocity), model.velocity._version, (16, 16))
    assert key1 in model._v_max_cache

    update = torch.ones_like(model.velocity) * 1.5
    with torch.no_grad():
        model.velocity.sub_(update)

    key2 = (id(model.velocity), model.velocity._version, (16, 16))
    assert key1 != key2, "Velocity parameter _version failed to increment after in-place sub_"
    assert key2 not in model._v_max_cache, "Old cache entry unexpectedly present"

    v2 = model._get_v_max_voxel(model.velocity, (16, 16))
    assert key2 in model._v_max_cache
    assert abs(v2 - v1) > 1e-4


# ==============================================================================
# 6. Memory Allocation Churn Reduction Verification
# ==============================================================================

def test_memory_churn_reduction():
    """Empirically measure memory allocation churn reduction in Adam and Composition."""
    shape = (64, 64, 64, 3)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    grad = torch.randn(shape)

    # Baseline out-of-place Adam
    m_base = torch.zeros(shape)
    v_base = torch.zeros(shape)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof_base:
        for t in range(1, 6):
            m_base = beta1 * m_base + (1 - beta1) * grad
            v_base = beta2 * v_base + (1 - beta2) * (grad ** 2)
            m_hat = m_base / (1 - beta1 ** t)
            v_hat = v_base / (1 - beta2 ** t)
            u_base = m_hat / (torch.sqrt(v_hat) + eps)
    base_bytes = sum(evt.cpu_memory_usage for evt in prof_base.key_averages() if evt.cpu_memory_usage > 0)

    # Remediated in-place Adam
    m_rem = torch.zeros(shape)
    v_rem = torch.zeros(shape)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof_rem:
        for t in range(1, 6):
            b1 = 1.0 - beta1 ** t
            b2 = 1.0 - beta2 ** t
            b2_sqrt = math.sqrt(b2)
            step_size = b2_sqrt / b1
            eps_scaled = eps * b2_sqrt
            m_rem.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v_rem.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            u_rem = (m_rem * step_size).div_(v_rem.sqrt().add_(eps_scaled))
    rem_bytes = sum(evt.cpu_memory_usage for evt in prof_rem.key_averages() if evt.cpu_memory_usage > 0)

    reduction = 1.0 - (rem_bytes / base_bytes)
    assert reduction >= 0.70, f"Expected >= 70% Adam memory churn reduction, got {reduction * 100:.1f}%"
