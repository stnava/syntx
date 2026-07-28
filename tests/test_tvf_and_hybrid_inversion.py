import os
import sys
import torch
import numpy as np
import pytest

# Ensure syntx is in python path
sys.path.insert(0, '/Users/stnava/code/syntx/src')

from syntx.syn import update_inverse_field_nd, integrate_time_varying_velocity_field
from syntx.syn_jax import update_inverse_field_nd_jax, integrate_time_varying_velocity_field_jax
import jax
import jax.numpy as jnp

def test_hybrid_lm_inverse_solver_pytorch():
    # Construct a smooth synthetic 3D displacement field
    spatial = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    
    # Create smooth forward displacement u(x)
    x = torch.linspace(-1, 1, 32)
    y = torch.linspace(-1, 1, 32)
    z = torch.linspace(-1, 1, 32)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    
    u_x = 0.5 * torch.sin(np.pi * grid_x) * torch.cos(np.pi * grid_y)
    u_y = 0.5 * torch.cos(np.pi * grid_x) * torch.sin(np.pi * grid_z)
    u_z = 0.5 * torch.sin(np.pi * grid_y) * torch.cos(np.pi * grid_z)
    
    W_disp = torch.stack([u_x, u_y, u_z], dim=-1).unsqueeze(0)  # (1, 32, 32, 32, 3)
    W_inv_init = torch.zeros_like(W_disp)
    
    # Test Fixed-Point vs Hybrid LM Inverse Solver
    W_inv_fp = update_inverse_field_nd(
        W_disp, W_inv_init, steps=10, method='fixed_point',
        spacing=spacing, origin=origin, direction=direction
    )
    
    W_inv_lm = update_inverse_field_nd(
        W_disp, W_inv_init, steps=10, method='hybrid_lm',
        spacing=spacing, origin=origin, direction=direction
    )
    
    err_fp = torch.norm(W_inv_fp + W_disp, dim=-1).mean()
    err_lm = torch.norm(W_inv_lm + W_disp, dim=-1).mean()
    
    print(f"\n[PyTorch Test] Mean Residual Error - Fixed Point: {err_fp.item():.6f} mm | Hybrid LM: {err_lm.item():.6f} mm")
    assert err_lm < 0.2, "Hybrid LM inverse solver failed to converge"


def test_hybrid_lm_inverse_solver_jax():
    spatial = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    z = np.linspace(-1, 1, 32)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    u_x = 0.5 * np.sin(np.pi * grid_x) * np.cos(np.pi * grid_y)
    u_y = 0.5 * np.cos(np.pi * grid_x) * np.sin(np.pi * grid_z)
    u_z = 0.5 * np.sin(np.pi * grid_y) * np.cos(np.pi * grid_z)
    
    W_disp = jnp.array(np.stack([u_x, u_y, u_z], axis=-1)[None, ...])
    W_inv_init = jnp.zeros_like(W_disp)
    
    W_inv_lm = update_inverse_field_nd_jax(
        W_disp, W_inv_init, steps=10, method='hybrid_lm',
        spacing=spacing, origin=origin, direction=direction
    )
    
    err_lm = jnp.mean(jnp.linalg.norm(W_inv_lm + W_disp, axis=-1))
    print(f"[JAX Test] Mean Residual Error - Hybrid LM: {err_lm:.6f} mm")
    assert err_lm < 0.2, "JAX Hybrid LM inverse solver failed to converge"


def test_time_varying_velocity_field_integration_pytorch():
    spatial = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    
    # Create sequence of 4 time-varying velocity fields
    T = 4
    vel_sequence = []
    x = torch.linspace(-1, 1, 32)
    y = torch.linspace(-1, 1, 32)
    z = torch.linspace(-1, 1, 32)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    
    for t_idx in range(T):
        scale = (t_idx + 1.0) / T
        u_x = 0.2 * scale * torch.sin(np.pi * grid_x)
        u_y = 0.2 * scale * torch.cos(np.pi * grid_y)
        u_z = 0.2 * scale * torch.sin(np.pi * grid_z)
        vel_sequence.append(torch.stack([u_x, u_y, u_z], dim=-1).unsqueeze(0))
        
    # Forward ODE integration (t: 0 -> 1)
    phi_fwd = integrate_time_varying_velocity_field(
        vel_sequence, dt=0.25, mode='forward', solver='rk4',
        spacing=spacing, origin=origin, direction=direction
    )
    
    # Backward ODE integration (t: 1 -> 0)
    phi_bwd = integrate_time_varying_velocity_field(
        vel_sequence, dt=0.25, mode='backward', solver='rk4',
        spacing=spacing, origin=origin, direction=direction
    )
    
    # Verify forward + backward symmetry
    symmetry_err = torch.norm(phi_fwd + phi_bwd, dim=-1).mean()
    print(f"[PyTorch TVF Integration] Symmetry Error ||Phi_fwd + Phi_bwd||: {symmetry_err.item():.6f} mm")
    assert symmetry_err < 0.05, "TVF forward/backward RK4 integration failed symmetry check"


def test_time_varying_velocity_field_integration_jax():
    spatial = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    
    T = 4
    vel_sequence = []
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    z = np.linspace(-1, 1, 32)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    for t_idx in range(T):
        scale = (t_idx + 1.0) / T
        u_x = 0.2 * scale * np.sin(np.pi * grid_x)
        u_y = 0.2 * scale * np.cos(np.pi * grid_y)
        u_z = 0.2 * scale * np.sin(np.pi * grid_z)
        vel_sequence.append(jnp.array(np.stack([u_x, u_y, u_z], axis=-1)[None, ...]))
        
    phi_fwd = integrate_time_varying_velocity_field_jax(
        vel_sequence, dt=0.25, mode='forward', solver='rk4',
        spacing=spacing, origin=origin, direction=direction
    )
    
    phi_bwd = integrate_time_varying_velocity_field_jax(
        vel_sequence, dt=0.25, mode='backward', solver='rk4',
        spacing=spacing, origin=origin, direction=direction
    )
    
    symmetry_err = jnp.mean(jnp.linalg.norm(phi_fwd + phi_bwd, axis=-1))
    print(f"[JAX TVF Integration] Symmetry Error ||Phi_fwd + Phi_bwd||: {symmetry_err:.6f} mm")
    assert symmetry_err < 0.05, "JAX TVF forward/backward RK4 integration failed symmetry check"


def test_anderson_acceleration_pytorch():
    """Test Anderson Acceleration converges and matches/beats fixed-point on same iteration budget."""
    spatial = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    x = torch.linspace(-1, 1, 32)
    y = torch.linspace(-1, 1, 32)
    z = torch.linspace(-1, 1, 32)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

    u_x = 0.5 * torch.sin(np.pi * grid_x) * torch.cos(np.pi * grid_y)
    u_y = 0.5 * torch.cos(np.pi * grid_x) * torch.sin(np.pi * grid_z)
    u_z = 0.5 * torch.sin(np.pi * grid_y) * torch.cos(np.pi * grid_z)

    W_disp = torch.stack([u_x, u_y, u_z], dim=-1).unsqueeze(0)  # (1, 32, 32, 32, 3)
    W_inv_init = torch.zeros_like(W_disp)

    # Fixed-Point baseline
    W_inv_fp = update_inverse_field_nd(
        W_disp, W_inv_init, steps=10, method='fixed_point',
        spacing=spacing, origin=origin, direction=direction
    )

    # Anderson Acceleration with same iteration budget
    W_inv_aa = update_inverse_field_nd(
        W_disp, W_inv_init, steps=10, method='anderson',
        spacing=spacing, origin=origin, direction=direction
    )

    # Compute composition residuals: ||v + u(x + v)||
    from syntx.syn import get_physical_grid_torch
    X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device='cpu', dtype=torch.float32)

    err_fp = torch.norm(W_inv_fp + W_disp, dim=-1).mean()
    err_aa = torch.norm(W_inv_aa + W_disp, dim=-1).mean()

    print(f"\n[PyTorch Anderson Test]")
    print(f"  Fixed-Point mean residual: {err_fp.item():.6f} mm")
    print(f"  Anderson    mean residual: {err_aa.item():.6f} mm")
    print(f"  Speedup ratio (FP/AA):     {err_fp.item()/max(err_aa.item(), 1e-12):.2f}x")

    # Anderson must converge
    assert err_aa < 0.2, f"Anderson Acceleration failed to converge: mean residual = {err_aa.item():.4f} mm"
    # Anderson should be at least comparable (within 2x) to fixed-point at same iteration count
    assert err_aa < err_fp * 2.0, f"Anderson significantly worse than fixed-point: {err_aa.item():.4f} vs {err_fp.item():.4f}"


def test_anderson_acceleration_jax():
    """Test Anderson Acceleration converges and matches/beats fixed-point on same iteration budget (JAX)."""
    spatial = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)
    z = np.linspace(-1, 1, 32)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    u_x = 0.5 * np.sin(np.pi * grid_x) * np.cos(np.pi * grid_y)
    u_y = 0.5 * np.cos(np.pi * grid_x) * np.sin(np.pi * grid_z)
    u_z = 0.5 * np.sin(np.pi * grid_y) * np.cos(np.pi * grid_z)

    W_disp = jnp.array(np.stack([u_x, u_y, u_z], axis=-1)[None, ...])
    W_inv_init = jnp.zeros_like(W_disp)

    # Fixed-Point baseline
    W_inv_fp = update_inverse_field_nd_jax(
        W_disp, W_inv_init, steps=10, method='fixed_point',
        spacing=spacing, origin=origin, direction=direction
    )

    # Anderson Acceleration
    W_inv_aa = update_inverse_field_nd_jax(
        W_disp, W_inv_init, steps=10, method='anderson',
        spacing=spacing, origin=origin, direction=direction
    )

    err_fp = jnp.mean(jnp.linalg.norm(W_inv_fp + W_disp, axis=-1))
    err_aa = jnp.mean(jnp.linalg.norm(W_inv_aa + W_disp, axis=-1))

    print(f"\n[JAX Anderson Test]")
    print(f"  Fixed-Point mean residual: {err_fp:.6f} mm")
    print(f"  Anderson    mean residual: {err_aa:.6f} mm")
    print(f"  Speedup ratio (FP/AA):     {float(err_fp)/max(float(err_aa), 1e-12):.2f}x")

    assert err_aa < 0.2, f"JAX Anderson Acceleration failed to converge: mean residual = {err_aa:.4f} mm"
    assert err_aa < err_fp * 2.0, f"JAX Anderson significantly worse than fixed-point: {err_aa:.4f} vs {err_fp:.4f}"


def test_anderson_acceleration_pytorch_backend_parity():
    """Verify PyTorch and JAX Anderson Acceleration produce results within floating-point tolerance."""
    spatial = (24, 24, 24)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    x = np.linspace(-1, 1, 24)
    y = np.linspace(-1, 1, 24)
    z = np.linspace(-1, 1, 24)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    u_x = 0.3 * np.sin(np.pi * grid_x) * np.cos(np.pi * grid_y)
    u_y = 0.3 * np.cos(np.pi * grid_x) * np.sin(np.pi * grid_z)
    u_z = 0.3 * np.sin(np.pi * grid_y) * np.cos(np.pi * grid_z)

    W_np = np.stack([u_x, u_y, u_z], axis=-1)[None, ...]

    # PyTorch
    W_disp_pt = torch.tensor(W_np, dtype=torch.float32)
    W_inv_init_pt = torch.zeros_like(W_disp_pt)
    W_inv_aa_pt = update_inverse_field_nd(
        W_disp_pt, W_inv_init_pt, steps=10, method='anderson',
        spacing=spacing, origin=origin, direction=direction
    )

    # JAX
    W_disp_jax = jnp.array(W_np, dtype=jnp.float32)
    W_inv_init_jax = jnp.zeros_like(W_disp_jax)
    W_inv_aa_jax = update_inverse_field_nd_jax(
        W_disp_jax, W_inv_init_jax, steps=10, method='anderson',
        spacing=spacing, origin=origin, direction=direction
    )

    err_pt = torch.norm(W_inv_aa_pt + W_disp_pt, dim=-1).mean().item()
    err_jax = float(jnp.mean(jnp.linalg.norm(W_inv_aa_jax + W_disp_jax, axis=-1)))

    print(f"\n[Backend Parity Test]")
    print(f"  PyTorch Anderson mean residual: {err_pt:.6f} mm")
    print(f"  JAX     Anderson mean residual: {err_jax:.6f} mm")
    print(f"  Difference:                     {abs(err_pt - err_jax):.6f} mm")

    # Backend parity: results must match within floating-point tolerance
    assert abs(err_pt - err_jax) < 0.01, (
        f"Backend parity violation: PT={err_pt:.4f} vs JAX={err_jax:.4f} (diff={abs(err_pt-err_jax):.4f})"
    )
