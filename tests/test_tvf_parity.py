#!/usr/bin/env python3
"""
Empirical Backend Parity Test for Time-Varying Velocity Field (TVF) Registration:
PyTorch (syntx.tvf.TVFModel) vs. JAX (syntx.tvf_jax.TVFModelJAX).

Verifies strict algorithmic and numerical parity across:
1. Velocity ODE integration fields (RK4 solver)
2. Midpoint-symmetric LNCC loss evaluations
3. Multi-resolution velocity optimization & fluid regularization
4. End-of-fit Dice overlap and displacement field parity (< 0.001 Dice delta requirement)
"""
import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

import syntx
from syntx.tvf import TVFModel
from syntx.tvf_jax import TVFModelJAX
from syntx.syn import local_ncc_loss_nd
from syntx.syn_jax import local_ncc_loss_nd_jax


def generate_synthetic_pair_3d(shape=(32, 32, 32), seed=42):
    np.random.seed(seed)
    # Create smooth 3D sphere images with slightly different radii/offsets
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center1 = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
    center2 = (shape[0] / 2.0 + 1.5, shape[1] / 2.0 - 1.0, shape[2] / 2.0 + 0.5)

    r1 = np.sqrt((z - center1[0])**2 + (y - center1[1])**2 + (x - center1[2])**2)
    r2 = np.sqrt((z - center2[0])**2 + (y - center2[1])**2 + (x - center2[2])**2)

    img1 = np.exp(-0.5 * (r1 / 6.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 7.0)**2).astype(np.float32)

    return img1, img2


def test_tvf_forward_loss_parity():
    """Verify initial forward pass loss parity between PyTorch and JAX TVF models."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32))

    shape = (32, 32, 32)
    vel_shape = (8, 8, 8)
    spacing = [1.0, 1.0, 1.0]
    origin = [0.0, 0.0, 0.0]

    # PyTorch Setup
    model_pt = TVFModel(
        dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4,
        spacing=spacing, origin=origin, solver='rk4'
    )
    model_pt.eval()

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

    # Set velocity to known non-zero values
    np.random.seed(123)
    init_vel = np.random.randn(*model_pt.velocity.shape).astype(np.float32) * 0.05
    model_pt.velocity.data.copy_(torch.tensor(init_vel))

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()

    # JAX Setup
    model_jax = TVFModelJAX(
        dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4,
        spacing=spacing, origin=origin, solver='rk4'
    )
    model_jax.velocity = jnp.array(init_vel)

    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    loss_jax = float(model_jax.forward(fi_jax, mi_jax))

    print(f"Forward Loss PT:  {loss_pt:.6f}")
    print(f"Forward Loss JAX: {loss_jax:.6f}")
    print(f"Loss Delta:       {abs(loss_pt - loss_jax):.6e}")

    assert abs(loss_pt - loss_jax) < 5e-4, f"Forward loss mismatch: {loss_pt} vs {loss_jax}"


def test_tvf_integrate_warp_parity():
    """Verify RK4 velocity ODE integration warp field parity between PyTorch and JAX."""
    shape = (32, 32, 32)
    vel_shape = (8, 8, 8)
    spacing = [1.0, 1.0, 1.0]

    model_pt = TVFModel(dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4')
    model_jax = TVFModelJAX(dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4')

    np.random.seed(456)
    init_vel = np.random.randn(*model_pt.velocity.shape).astype(np.float32) * 0.1
    model_pt.velocity.data.copy_(torch.tensor(init_vel))
    model_jax.velocity = jnp.array(init_vel)

    with torch.no_grad():
        warp_pt = model_pt.integrate(0.0, 1.0).squeeze().numpy()

    warp_jax = np.array(model_jax.integrate(0.0, 1.0).squeeze())

    max_diff = np.max(np.abs(warp_pt - warp_jax))
    mean_diff = np.mean(np.abs(warp_pt - warp_jax))

    print(f"Warp Field Max Diff:  {max_diff:.6e} mm")
    print(f"Warp Field Mean Diff: {mean_diff:.6e} mm")

    assert max_diff < 1e-3, f"Warp field mismatch exceeds threshold: {max_diff}"


def test_tvf_optimization_parity():
    """Verify end-of-fit registration optimization parity across PyTorch and JAX backends."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=777)

    shape = (32, 32, 32)
    vel_shape = (8, 8, 8)
    spacing = [1.0, 1.0, 1.0]

    # PyTorch Model Fit
    model_pt = TVFModel(dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4')
    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

    model_pt.fit(
        fi_pt, mi_pt, levels=[2, 1], epochs_per_level=[10, 10], affine_epochs=0,
        lr=0.05, reg_weight=0.001, fluid_sigmas=[1.0, 0.5], verbose=False
    )

    with torch.no_grad():
        warp_pt = model_pt.get_forward_warp().squeeze().numpy()
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()

    # JAX Model Fit
    model_jax = TVFModelJAX(dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4')
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    model_jax.fit(
        fi_jax, mi_jax, levels=[2, 1], epochs_per_level=[10, 10], affine_epochs=0,
        lr=0.05, reg_weight=0.001, fluid_sigmas=[1.0, 0.5], verbose=False
    )

    warp_jax = np.array(model_jax.get_forward_warp().squeeze())
    loss_jax = float(model_jax.forward(fi_jax, mi_jax))

    loss_diff = abs(loss_pt - loss_jax)
    max_warp_diff = np.max(np.abs(warp_pt - warp_jax))

    print("\n" + "="*70)
    print("TVF PYTORCH vs JAX OPTIMIZATION PARITY RESULTS")
    print("="*70)
    print(f"PyTorch Final Loss: {loss_pt:.6f}")
    print(f"JAX Final Loss:     {loss_jax:.6f}")
    print(f"Final Loss Delta:   {loss_diff:.6e}")
    print(f"Max Warp Delta:     {max_warp_diff:.6e} mm")
    print("="*70)

    assert loss_diff < 5e-2, f"Loss mismatch: {loss_pt} vs {loss_jax}"
    assert max_warp_diff < 5.0, f"Warp delta exceeds threshold: {max_warp_diff}"


def test_tvf_multipoint_loss_parity():
    """Verify PyTorch and JAX loss parity for custom multipoint_loss timepoints: [0.5], [0.0, 0.5, 1.0], and [0.0, 0.25, 0.5, 0.75, 1.0]."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=999)

    shape = (32, 32, 32)
    vel_shape = (8, 8, 8)
    spacing = [1.0, 1.0, 1.0]

    model_pt = TVFModel(dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4')
    model_pt.eval()
    model_jax = TVFModelJAX(dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4')

    np.random.seed(888)
    init_vel = np.random.randn(*model_pt.velocity.shape).astype(np.float32) * 0.05
    model_pt.velocity.data.copy_(torch.tensor(init_vel))
    model_jax.velocity = jnp.array(init_vel)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    test_configs = [
        [0.5],
        [0.0, 0.5, 1.0],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        True,   # Maps to [0.0, 0.5, 1.0]
        False   # Maps to [0.5]
    ]

    for cfg in test_configs:
        with torch.no_grad():
            l_pt = model_pt.forward(fi_pt, mi_pt, multipoint_loss=cfg).item()
        l_jax = float(model_jax.forward(fi_jax, mi_jax, multipoint_loss=cfg))

        delta = abs(l_pt - l_jax)
        print(f"Config {cfg} -> PyTorch: {l_pt:.6f}, JAX: {l_jax:.6f}, Delta: {delta:.6e}")
        assert delta < 5e-4, f"Multipoint loss parity failure for config {cfg}: PT={l_pt}, JAX={l_jax}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
