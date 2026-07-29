#!/usr/bin/env python3
"""
Adversarial Stress Test for Geodesic Shooting Models at Zero ($p_0 = 0$) and Near-Zero Initialization.
Tests both PyTorch (syntx.shooting.GeodesicShootingModel) and JAX (syntx.shooting_jax.GeodesicShootingModelJAX).
"""
import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

import syntx
from syntx.shooting import GeodesicShootingModel, epdiff_advection_nd
from syntx.shooting_jax import GeodesicShootingModelJAX, epdiff_advection_nd_jax
from syntx.tvf import normalize_tensor

def create_test_images_2d(shape=(32, 32)):
    np.random.seed(42)
    y, x = np.ogrid[:shape[0], :shape[1]]
    c1 = (shape[0] / 2.0, shape[1] / 2.0)
    c2 = (shape[0] / 2.0 + 2.0, shape[1] / 2.0 - 1.5)
    r1 = np.sqrt((y - c1[0])**2 + (x - c1[1])**2)
    r2 = np.sqrt((y - c2[0])**2 + (x - c2[1])**2)
    img1 = np.exp(-0.5 * (r1 / 5.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 6.0)**2).astype(np.float32)
    return img1, img2

def create_test_images_3d(shape=(24, 24, 24)):
    np.random.seed(42)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c1 = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
    c2 = (shape[0] / 2.0 + 1.5, shape[1] / 2.0 - 1.0, shape[2] / 2.0 + 0.5)
    r1 = np.sqrt((z - c1[0])**2 + (y - c1[1])**2 + (x - c1[2])**2)
    r2 = np.sqrt((z - c2[0])**2 + (y - c2[1])**2 + (x - c2[2])**2)
    img1 = np.exp(-0.5 * (r1 / 5.0)**2).astype(np.float32)
    img2 = np.exp(-0.5 * (r2 / 6.0)**2).astype(np.float32)
    return img1, img2


@pytest.mark.parametrize("dim", [2, 3])
def test_pytorch_p0_zero_gradient_flow(dim):
    """Verify PyTorch GeodesicShootingModel gradient flow at p0 = 0 and near-zero values."""
    if dim == 2:
        img1, img2 = create_test_images_2d((32, 32))
        shape = (32, 32)
    else:
        img1, img2 = create_test_images_3d((24, 24, 24))
        shape = (24, 24, 24)

    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

    model = GeodesicShootingModel(dim=dim, image_shape=shape, n_time_steps=4, fluid_sigma=1.0)

    # 1. Test exact p0 = 0
    p0_zero = torch.zeros(1, *shape, dim, requires_grad=True)
    loss_zero = model.forward(fi_pt, mi_pt, p0=p0_zero)
    assert not torch.isnan(loss_zero) and not torch.isinf(loss_zero), "Loss at p0=0 is NaN or Inf"

    loss_zero.backward()
    grad_zero = p0_zero.grad

    assert grad_zero is not None, "Grad at p0=0 is None"
    assert not torch.isnan(grad_zero).any(), "Grad at p0=0 contains NaNs"
    assert not torch.isinf(grad_zero).any(), "Grad at p0=0 contains Infs"

    grad_norm_zero = torch.linalg.vector_norm(grad_zero).item()
    print(f"[{dim}D PyTorch] p0=0 Loss: {loss_zero.item():.6f}, Grad norm: {grad_norm_zero:.6e}")
    assert grad_norm_zero > 1e-7, f"Grad at p0=0 is zero (norm={grad_norm_zero:.6e})"

    # 2. Test near-zero scale sweep (1e-12 to 1e-2) & gradient continuity
    scales = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2]
    prev_grad = grad_zero.clone()

    for scale in scales:
        np.random.seed(123)
        noise = np.random.randn(1, *shape, dim).astype(np.float32)
        p0_scale = torch.tensor(scale * noise, requires_grad=True)

        loss_scale = model.forward(fi_pt, mi_pt, p0=p0_scale)
        assert not torch.isnan(loss_scale) and not torch.isinf(loss_scale), f"Loss at scale {scale} is NaN/Inf"

        loss_scale.backward()
        grad_scale = p0_scale.grad

        assert grad_scale is not None, f"Grad at scale {scale} is None"
        assert not torch.isnan(grad_scale).any(), f"Grad at scale {scale} contains NaNs"
        assert not torch.isinf(grad_scale).any(), f"Grad at scale {scale} contains Infs"

        grad_norm = torch.linalg.vector_norm(grad_scale).item()
        diff_from_zero = torch.linalg.vector_norm(grad_scale - grad_zero).item()
        print(f"[{dim}D PyTorch] scale={scale:.1e} Loss: {loss_scale.item():.6f}, Grad norm: {grad_norm:.6e}, Diff from p0=0 grad: {diff_from_zero:.6e}")

        # Continuity check: as scale -> 0, diff_from_zero -> 0
        if scale <= 1e-6:
            assert diff_from_zero < 1e-2, f"Gradient discontinuity detected at scale {scale}: diff={diff_from_zero}"


@pytest.mark.parametrize("dim", [2, 3])
def test_jax_p0_zero_gradient_flow(dim):
    """Verify JAX GeodesicShootingModelJAX gradient flow at p0 = 0 and near-zero values."""
    if dim == 2:
        img1, img2 = create_test_images_2d((32, 32))
        shape = (32, 32)
    else:
        img1, img2 = create_test_images_3d((24, 24, 24))
        shape = (24, 24, 24)

    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    model = GeodesicShootingModelJAX(dim=dim, image_shape=shape, n_time_steps=4, fluid_sigma=1.0)

    def loss_fn(p0_param):
        return model.forward(fi_jax, mi_jax, p0=p0_param)

    val_and_grad_fn = jax.value_and_grad(loss_fn)

    # 1. Test exact p0 = 0
    p0_zero = jnp.zeros((1, *shape, dim), dtype=jnp.float32)
    loss_zero, grad_zero = val_and_grad_fn(p0_zero)

    assert not jnp.isnan(loss_zero) and not jnp.isinf(loss_zero), "Loss at p0=0 is NaN or Inf"
    assert not jnp.isnan(grad_zero).any(), "Grad at p0=0 contains NaNs"
    assert not jnp.isnan(grad_zero).any(), "Grad at p0=0 contains Infs"

    grad_norm_zero = float(jnp.sqrt(jnp.sum(grad_zero ** 2)))
    print(f"[{dim}D JAX] p0=0 Loss: {float(loss_zero):.6f}, Grad norm: {grad_norm_zero:.6e}")
    assert grad_norm_zero > 1e-7, f"Grad at p0=0 is zero (norm={grad_norm_zero:.6e})"

    # 2. Test near-zero scale sweep (1e-12 to 1e-2) & gradient continuity
    scales = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2]

    for scale in scales:
        np.random.seed(123)
        noise = np.random.randn(1, *shape, dim).astype(np.float32)
        p0_scale = jnp.array(scale * noise)

        loss_scale, grad_scale = val_and_grad_fn(p0_scale)
        assert not jnp.isnan(loss_scale) and not jnp.isinf(loss_scale), f"JAX loss at scale {scale} is NaN/Inf"
        assert not jnp.isnan(grad_scale).any(), f"JAX grad at scale {scale} contains NaNs"
        assert not jnp.isinf(grad_scale).any(), f"JAX grad at scale {scale} contains Infs"

        grad_norm = float(jnp.sqrt(jnp.sum(grad_scale ** 2)))
        diff_from_zero = float(jnp.sqrt(jnp.sum((grad_scale - grad_zero) ** 2)))
        print(f"[{dim}D JAX] scale={scale:.1e} Loss: {float(loss_scale):.6f}, Grad norm: {grad_norm:.6e}, Diff from p0=0 grad: {diff_from_zero:.6e}")

        if scale <= 1e-6:
            assert diff_from_zero < 1e-2, f"JAX gradient discontinuity detected at scale {scale}: diff={diff_from_zero}"


def test_multiresolution_fit_from_p0_zero():
    """Verify multi-resolution fit starting from p0=0 produces smooth optimization across backends."""
    img1, img2 = create_test_images_3d((24, 24, 24))
    shape = (24, 24, 24)

    # PyTorch
    model_pt = GeodesicShootingModel(dim=3, image_shape=shape, n_time_steps=4, fluid_sigma=1.0)
    fi_pt = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
    mi_pt = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

    # Confirm initial p0 is zero
    assert torch.all(model_pt.p0 == 0.0)

    model_pt.fit(fi_pt, mi_pt, levels=[4, 2, 1], epochs_per_level=[5, 5, 5], lr=0.05, verbose=False)

    loss_pt_after = model_pt.forward(fi_pt, mi_pt).item()
    p0_norm_pt = torch.linalg.vector_norm(model_pt.p0).item()
    assert not np.isnan(loss_pt_after), "PyTorch post-fit loss is NaN"
    assert p0_norm_pt > 1e-4, "PyTorch p0 did not update away from zero during fit"

    # JAX
    model_jax = GeodesicShootingModelJAX(dim=3, image_shape=shape, n_time_steps=4, fluid_sigma=1.0)
    fi_jax = jnp.array(img1)[None, None, ...]
    mi_jax = jnp.array(img2)[None, None, ...]

    assert float(jnp.sum(jnp.abs(model_jax.p0))) == 0.0

    model_jax.fit(fi_jax, mi_jax, levels=[4, 2, 1], epochs_per_level=[5, 5, 5], lr=0.05, verbose=False)

    loss_jax_after = float(model_jax.forward(fi_jax, mi_jax))
    p0_norm_jax = float(jnp.sqrt(jnp.sum(model_jax.p0 ** 2)))
    assert not np.isnan(loss_jax_after), "JAX post-fit loss is NaN"
    assert p0_norm_jax > 1e-4, "JAX p0 did not update away from zero during fit"

    print(f"Multi-res Fit Loss PT:  {loss_pt_after:.6f}, p0 norm: {p0_norm_pt:.6f}")
    print(f"Multi-res Fit Loss JAX: {loss_jax_after:.6f}, p0 norm: {p0_norm_jax:.6f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
