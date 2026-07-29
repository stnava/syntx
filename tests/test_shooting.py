#!/usr/bin/env python3
"""
Unit and Parity Tests for Geodesic Shooting Models:
PyTorch (syntx.shooting.GeodesicShootingModel) vs. JAX (syntx.shooting_jax.GeodesicShootingModelJAX).
"""
import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

import syntx
from syntx.shooting import GeodesicShootingModel, epdiff_advection_nd
from syntx.shooting_jax import GeodesicShootingModelJAX, epdiff_advection_nd_jax


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


def test_epdiff_advection_parity():
    """Verify EPDiff coadjoint advection operator parity between PyTorch and JAX."""
    np.random.seed(123)
    p_np = np.random.randn(1, 16, 16, 16, 3).astype(np.float32) * 0.1
    v_np = np.random.randn(1, 16, 16, 16, 3).astype(np.float32) * 0.1

    p_pt = torch.tensor(p_np)
    v_pt = torch.tensor(v_np)
    ad_pt = epdiff_advection_nd(p_pt, v_pt).numpy()

    p_jax = jnp.array(p_np)
    v_jax = jnp.array(v_np)
    ad_jax = np.array(epdiff_advection_nd_jax(p_jax, v_jax))

    mse = float(np.mean((ad_pt - ad_jax) ** 2))
    max_diff = float(np.max(np.abs(ad_pt - ad_jax)))

    print(f"EPDiff Advection MSE: {mse:.6e}, Max Diff: {max_diff:.6e}")
    assert mse <= 0.001, f"EPDiff advection MSE discrepancy exceeds threshold: {mse}"
    assert max_diff <= 0.001, f"EPDiff advection max diff exceeds threshold: {max_diff}"


def test_shooting_forward_loss_parity():
    """Verify initial forward pass loss parity between PyTorch and JAX shooting models."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=42)
    shape = (32, 32, 32)
    spacing = [1.0, 1.0, 1.0]

    model_pt = GeodesicShootingModel(
        dim=3, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0
    )
    model_pt.eval()

    model_jax = GeodesicShootingModelJAX(
        dim=3, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0
    )

    np.random.seed(777)
    p0_init = np.random.randn(*model_pt.p0.shape).astype(np.float32) * 0.05
    model_pt.p0.data.copy_(torch.tensor(p0_init))
    model_jax.p0 = jnp.array(p0_init)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()
    loss_jax = float(model_jax.forward(fi_jax, mi_jax))

    loss_delta = abs(loss_pt - loss_jax)
    print(f"Shooting Forward Loss PT:  {loss_pt:.6f}")
    print(f"Shooting Forward Loss JAX: {loss_jax:.6f}")
    print(f"Loss Delta:                {loss_delta:.6e}")

    assert loss_delta <= 0.001, f"Shooting forward loss mismatch: {loss_pt} vs {loss_jax} (delta={loss_delta})"


def test_shooting_warp_parity():
    """Verify geodesic shooting forward displacement warp field parity between PyTorch and JAX."""
    shape = (32, 32, 32)
    spacing = [1.0, 1.0, 1.0]

    model_pt = GeodesicShootingModel(dim=3, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0)
    model_jax = GeodesicShootingModelJAX(dim=3, image_shape=shape, n_time_steps=6, spacing=spacing, fluid_sigma=1.0)

    np.random.seed(999)
    p0_init = np.random.randn(*model_pt.p0.shape).astype(np.float32) * 0.05
    model_pt.p0.data.copy_(torch.tensor(p0_init))
    model_jax.p0 = jnp.array(p0_init)

    with torch.no_grad():
        warp_pt = model_pt.get_forward_warp().squeeze().numpy()
    warp_jax = np.array(model_jax.get_forward_warp().squeeze())

    mse_discrepancy = float(np.mean((warp_pt - warp_jax) ** 2))
    max_diff = float(np.max(np.abs(warp_pt - warp_jax)))

    print(f"Shooting Warp MSE Discrepancy: {mse_discrepancy:.6e}")
    print(f"Shooting Warp Max Diff:        {max_diff:.6e} mm")

    assert mse_discrepancy <= 0.001, f"Shooting warp field MSE discrepancy exceeds 0.001: {mse_discrepancy}"
    assert max_diff <= 0.001, f"Shooting warp field max diff exceeds 0.001: {max_diff}"


def test_shooting_model_fit_parity():
    """Verify multi-resolution optimization parity for PyTorch and JAX GeodesicShootingModel."""
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=555)
    shape = (32, 32, 32)

    model_pt = GeodesicShootingModel(dim=3, image_shape=shape, n_time_steps=4, fluid_sigma=1.0)
    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

    model_pt.fit(
        fi_pt, mi_pt, levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0, lr=0.01, verbose=False
    )

    model_jax = GeodesicShootingModelJAX(dim=3, image_shape=shape, n_time_steps=4, fluid_sigma=1.0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    model_jax.fit(
        fi_jax, mi_jax, levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0, lr=0.01, verbose=False
    )

    with torch.no_grad():
        loss_pt = model_pt.forward(fi_pt, mi_pt).item()
        warp_pt = model_pt.get_forward_warp().squeeze().numpy()
    loss_jax = float(model_jax.forward(fi_jax, mi_jax))
    warp_jax = np.array(model_jax.get_forward_warp().squeeze())

    loss_delta = abs(loss_pt - loss_jax)
    mse_discrepancy = float(np.mean((warp_pt - warp_jax) ** 2))

    print(f"Shooting Fit Loss PT:   {loss_pt:.6f}")
    print(f"Shooting Fit Loss JAX:  {loss_jax:.6f}")
    print(f"Shooting Fit Loss Delta: {loss_delta:.6e}")
    print(f"Shooting Fit Warp MSE:  {mse_discrepancy:.6e}")

    assert loss_delta <= 0.05, f"Shooting fit loss delta exceeds threshold: {loss_delta}"
    assert mse_discrepancy <= 0.01, f"Shooting fit warp MSE discrepancy exceeds threshold: {mse_discrepancy}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
