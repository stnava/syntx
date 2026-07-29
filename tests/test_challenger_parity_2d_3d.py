#!/usr/bin/env python3
"""
Empirical Challenger Verification Suite:
Testing PyTorch vs JAX TVF and Geodesic Shooting backend parity on both 2D and 3D shapes.

Criteria:
1. PyTorch vs JAX forward loss delta <= 0.001
2. PyTorch vs JAX output displacement warp field MSE <= 0.001
"""
import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

import syntx
from syntx.shooting import GeodesicShootingModel, epdiff_advection_nd
from syntx.shooting_jax import GeodesicShootingModelJAX, epdiff_advection_nd_jax
from syntx.tvf import TVFModel
from syntx.tvf_jax import TVFModelJAX


def generate_synthetic_pair_2d(shape=(64, 64), seed=42):
    np.random.seed(seed)
    y, x = np.ogrid[:shape[0], :shape[1]]
    center1 = (shape[0] / 2.0, shape[1] / 2.0)
    center2 = (shape[0] / 2.0 + 2.0, shape[1] / 2.0 - 1.5)

    r1 = np.sqrt((y - center1[0])**2 + (x - center1[1])**2)
    r2 = np.sqrt((y - center2[0])**2 + (x - center2[1])**2)

    img1 = np.exp(-0.5 * (r1 / 10.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 12.0)**2).astype(np.float32)
    return img1, img2


def generate_synthetic_pair_3d(shape=(32, 32, 32), seed=42):
    np.random.seed(seed)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center1 = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
    center2 = (shape[0] / 2.0 + 1.5, shape[1] / 2.0 - 1.0, shape[2] / 2.0 + 0.5)

    r1 = np.sqrt((z - center1[0])**2 + (y - center1[1])**2 + (x - center1[2])**2)
    r2 = np.sqrt((z - center2[0])**2 + (y - center2[1])**2 + (x - center2[2])**2)

    img1 = np.exp(-0.5 * (r1 / 6.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 7.0)**2).astype(np.float32)
    return img1, img2


# ==============================================================================
# GEODESIC SHOOTING PARITY (2D & 3D)
# ==============================================================================

def test_shooting_2d_parity():
    """Verify PyTorch vs JAX Geodesic Shooting parity on 2D inputs."""
    img1, img2 = generate_synthetic_pair_2d(shape=(64, 64), seed=101)
    shape = (64, 64)
    spacing = [1.0, 1.0]

    model_pt = GeodesicShootingModel(
        dim=2, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0
    )
    model_pt.eval()

    model_jax = GeodesicShootingModelJAX(
        dim=2, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0
    )

    np.random.seed(202)
    p0_init = np.random.randn(*model_pt.p0.shape).astype(np.float32) * 0.05
    model_pt.p0.data.copy_(torch.tensor(p0_init))
    model_jax.p0 = jnp.array(p0_init)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()
        warp_pt = model_pt.get_forward_warp().squeeze().numpy()

    loss_jax = float(model_jax.forward(fi_jax, mi_jax))
    warp_jax = np.array(model_jax.get_forward_warp().squeeze())

    loss_delta = abs(loss_pt - loss_jax)
    warp_mse = float(np.mean((warp_pt - warp_jax) ** 2))
    warp_max_diff = float(np.max(np.abs(warp_pt - warp_jax)))

    print(f"\n[2D Shooting Parity]")
    print(f"Loss PT:  {loss_pt:.6f}, Loss JAX: {loss_jax:.6f}, Loss Delta: {loss_delta:.6e}")
    print(f"Warp Field MSE: {warp_mse:.6e}, Max Diff: {warp_max_diff:.6e} mm")

    assert loss_delta <= 0.001, f"2D Shooting loss delta exceeds 0.001: {loss_delta}"
    assert warp_mse <= 0.001, f"2D Shooting warp MSE exceeds 0.001: {warp_mse}"


def test_shooting_3d_parity():
    """Verify PyTorch vs JAX Geodesic Shooting parity on 3D inputs."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=303)
    shape = (32, 32, 32)
    spacing = [1.0, 1.0, 1.0]

    model_pt = GeodesicShootingModel(
        dim=3, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0
    )
    model_pt.eval()

    model_jax = GeodesicShootingModelJAX(
        dim=3, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0
    )

    np.random.seed(404)
    p0_init = np.random.randn(*model_pt.p0.shape).astype(np.float32) * 0.05
    model_pt.p0.data.copy_(torch.tensor(p0_init))
    model_jax.p0 = jnp.array(p0_init)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()
        warp_pt = model_pt.get_forward_warp().squeeze().numpy()

    loss_jax = float(model_jax.forward(fi_jax, mi_jax))
    warp_jax = np.array(model_jax.get_forward_warp().squeeze())

    loss_delta = abs(loss_pt - loss_jax)
    warp_mse = float(np.mean((warp_pt - warp_jax) ** 2))
    warp_max_diff = float(np.max(np.abs(warp_pt - warp_jax)))

    print(f"\n[3D Shooting Parity]")
    print(f"Loss PT:  {loss_pt:.6f}, Loss JAX: {loss_jax:.6f}, Loss Delta: {loss_delta:.6e}")
    print(f"Warp Field MSE: {warp_mse:.6e}, Max Diff: {warp_max_diff:.6e} mm")

    assert loss_delta <= 0.001, f"3D Shooting loss delta exceeds 0.001: {loss_delta}"
    assert warp_mse <= 0.001, f"3D Shooting warp MSE exceeds 0.001: {warp_mse}"


# ==============================================================================
# TVF MODEL PARITY (2D & 3D)
# ==============================================================================

def test_tvf_2d_parity():
    """Verify PyTorch vs JAX TVF Model parity on 2D inputs."""
    img1, img2 = generate_synthetic_pair_2d(shape=(64, 64), seed=505)
    shape = (64, 64)
    vel_shape = (16, 16)
    spacing = [1.0, 1.0]

    model_pt = TVFModel(
        dim=2, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4'
    )
    model_pt.eval()

    model_jax = TVFModelJAX(
        dim=2, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4'
    )

    np.random.seed(606)
    init_vel = np.random.randn(*model_pt.velocity.shape).astype(np.float32) * 0.05
    model_pt.velocity.data.copy_(torch.tensor(init_vel))
    model_jax.velocity = jnp.array(init_vel)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()
        warp_pt = model_pt.integrate(0.0, 1.0).squeeze().numpy()

    loss_jax = float(model_jax.forward(fi_jax, mi_jax))
    warp_jax = np.array(model_jax.integrate(0.0, 1.0).squeeze())

    loss_delta = abs(loss_pt - loss_jax)
    warp_mse = float(np.mean((warp_pt - warp_jax) ** 2))
    warp_max_diff = float(np.max(np.abs(warp_pt - warp_jax)))

    print(f"\n[2D TVF Parity]")
    print(f"Loss PT:  {loss_pt:.6f}, Loss JAX: {loss_jax:.6f}, Loss Delta: {loss_delta:.6e}")
    print(f"Warp Field MSE: {warp_mse:.6e}, Max Diff: {warp_max_diff:.6e} mm")

    assert loss_delta <= 0.001, f"2D TVF loss delta exceeds 0.001: {loss_delta}"
    assert warp_mse <= 0.001, f"2D TVF warp MSE exceeds 0.001: {warp_mse}"


def test_tvf_3d_parity():
    """Verify PyTorch vs JAX TVF Model parity on 3D inputs."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=707)
    shape = (32, 32, 32)
    vel_shape = (8, 8, 8)
    spacing = [1.0, 1.0, 1.0]

    model_pt = TVFModel(
        dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4'
    )
    model_pt.eval()

    model_jax = TVFModelJAX(
        dim=3, image_shape=shape, velocity_shape=vel_shape, n_time_steps=4, spacing=spacing, solver='rk4'
    )

    np.random.seed(808)
    init_vel = np.random.randn(*model_pt.velocity.shape).astype(np.float32) * 0.05
    model_pt.velocity.data.copy_(torch.tensor(init_vel))
    model_jax.velocity = jnp.array(init_vel)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()
        warp_pt = model_pt.integrate(0.0, 1.0).squeeze().numpy()

    loss_jax = float(model_jax.forward(fi_jax, mi_jax))
    warp_jax = np.array(model_jax.integrate(0.0, 1.0).squeeze())

    loss_delta = abs(loss_pt - loss_jax)
    warp_mse = float(np.mean((warp_pt - warp_jax) ** 2))
    warp_max_diff = float(np.max(np.abs(warp_pt - warp_jax)))

    print(f"\n[3D TVF Parity]")
    print(f"Loss PT:  {loss_pt:.6f}, Loss JAX: {loss_jax:.6f}, Loss Delta: {loss_delta:.6e}")
    print(f"Warp Field MSE: {warp_mse:.6e}, Max Diff: {warp_max_diff:.6e} mm")

    assert loss_delta <= 0.001, f"3D TVF loss delta exceeds 0.001: {loss_delta}"
    assert warp_mse <= 0.001, f"3D TVF warp MSE exceeds 0.001: {warp_mse}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
