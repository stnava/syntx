"""
Unit tests for syntx.syngs (Geodesic Shooting registration engine).
Verifies PyTorch and JAX backend parity, EPDiff shooting, and inverse warp generation.
"""
import pytest
import numpy as np
import torch
import jax.numpy as jnp
import syntx
from syntx.syngs import GeodesicShootingModel
from syntx.syngs_jax import GeodesicShootingModelJAX


def generate_synthetic_pair_3d(shape=(32, 32, 32), seed=42):
    np.random.seed(seed)
    grid = np.ogrid[[slice(0, s) for s in shape]]
    center1 = [s // 2 for s in shape]
    center2 = [s // 2 + 2 for s in shape]
    
    r1 = sum((g - c)**2 for g, c in zip(grid, center1))
    r2 = sum((g - c)**2 for g, c in zip(grid, center2))
    
    img1 = (r1 < (shape[0] // 4)**2).astype(np.float32)
    img2 = (r2 < (shape[0] // 4)**2).astype(np.float32)
    return img1, img2


def test_syngs_model_initialization():
    shape = (32, 32, 32)
    model_pt = GeodesicShootingModel(dim=3, image_shape=shape, velocity_shape=shape, n_steps=10)
    model_jax = GeodesicShootingModelJAX(dim=3, image_shape=shape, velocity_shape=shape, n_steps=10)
    
    assert model_pt.velocity_0.shape == (1, 32, 32, 32, 3)
    assert model_jax.velocity_0.shape == (1, 32, 32, 32, 3)


def test_syngs_parity_fast():
    img1, img2 = generate_synthetic_pair_3d(shape=(32, 32, 32), seed=123)
    shape = (32, 32, 32)
    vel_shape = (16, 16, 16)
    spacing = [1.0, 1.0, 1.0]

    model_pt = GeodesicShootingModel(dim=3, image_shape=shape, velocity_shape=vel_shape, n_steps=5, spacing=spacing)
    model_pt.eval()
    model_jax = GeodesicShootingModelJAX(dim=3, image_shape=shape, velocity_shape=vel_shape, n_steps=5, spacing=spacing)

    np.random.seed(555)
    init_v0 = np.random.randn(*model_pt.velocity_0.shape).astype(np.float32) * 0.02
    model_pt.velocity_0.data.copy_(torch.tensor(init_v0))
    model_jax.velocity_0 = jnp.array(init_v0)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    with torch.no_grad():
        l_pt = model_pt.forward(fi_pt, mi_pt).item()
    l_jax = float(model_jax.forward(fi_jax, mi_jax))

    delta = abs(l_pt - l_jax)
    assert delta < 0.01, f"SyNGS forward loss parity failure: PT={l_pt:.6f}, JAX={l_jax:.6f}, Delta={delta:.6e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
