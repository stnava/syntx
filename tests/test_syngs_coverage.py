"""
Unit tests targeting high code coverage for syntx.syngs and syntx.syngs_jax.
"""

import os
import numpy as np
import torch
import pytest
import ants

from syntx import syngs
from syntx.syngs import GeodesicShootingModel, LARS
try:
    from syntx.syngs_jax import GeodesicShootingModelJAX
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def test_syngs_lars_optimizer():
    params = [torch.nn.Parameter(torch.ones(2, 2))]
    opt = LARS(params, lr=0.5, trust_coefficient=0.1)
    params[0].grad = torch.ones(2, 2) * 2.0
    opt.step()
    assert not torch.allclose(params[0], torch.ones(2, 2))


def test_geodesic_shooting_model_2d():
    device = 'cpu'
    model = GeodesicShootingModel(
        dim=2,
        image_shape=(16, 16),
        velocity_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0],
        fluid_sigma=1.0,
        elastic_sigma=0.0,
        n_steps=3
    ).to(device)

    fixed = torch.randn(1, 1, 16, 16, device=device)
    moving = torch.randn(1, 1, 16, 16, device=device)

    # Test forward loss
    loss = model.forward(fixed, moving)
    assert isinstance(loss, torch.Tensor)

    # Test fit with fast parameters
    model.fit(
        fixed, moving,
        epochs_per_level=[2],
        levels=[1],
        affine_epochs=0,
        verbose=False
    )

    # Test warp integration & exports
    fwd_warp = model.get_forward_warp()
    inv_warp = model.get_inverse_warp()
    assert fwd_warp.shape == (1, 16, 16, 2)
    assert inv_warp.shape == (1, 16, 16, 2)


def test_syngs_registration_high_level_2d():
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    reg = syngs(
        fixed=fi,
        moving=mi,
        reg_iterations=[2],
        affine_iterations=[0],
        backend='pytorch',
        verbose=False
    )

    assert 'warpedmovout' in reg
    assert 'fwdtransforms' in reg
    assert 'invtransforms' in reg
    assert 'fwd_momentum' in reg
    assert 'inv_momentum' in reg
    assert 'fwd_deformation' in reg
    assert 'inv_deformation' in reg
    assert os.path.exists(reg['fwdtransforms'][0])
    assert os.path.exists(reg['fwd_momentum_file'])
    assert os.path.exists(reg['inv_momentum_file'])


def test_integrate_momentum_reconstruction():
    from syntx import integrate_momentum, shoot_geodesic, momentum_to_deformation
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    reg = syngs(fi, mi, reg_iterations=[4, 2], affine_iterations=[0], verbose=False)
    fwd_mom = reg['fwd_momentum']
    fwd_def = reg['fwd_deformation']

    # Test exact reconstruction
    reconstructed_def = integrate_momentum(fwd_mom, reference_image=fi)
    diff = np.abs(reconstructed_def.numpy() - fwd_def.numpy()).max()
    assert diff < 1e-4

    # Test aliases
    def_shoot = shoot_geodesic(fwd_mom, fi)
    def_mom2def = momentum_to_deformation(fwd_mom, fi)
    assert np.allclose(def_shoot.numpy(), reconstructed_def.numpy())
    assert np.allclose(def_mom2def.numpy(), reconstructed_def.numpy())

    # Test trajectory return
    traj = integrate_momentum(fwd_mom, reference_image=fi, n_steps=4, return_trajectory=True)
    assert len(traj) == 5
    for t_img in traj:
        assert isinstance(t_img, ants.ANTsImage)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_geodesic_shooting_model_jax_2d():
    import jax.numpy as jnp

    model_jax = GeodesicShootingModelJAX(
        dim=2,
        image_shape=(16, 16),
        velocity_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0],
        fluid_sigma=1.0,
        elastic_sigma=0.0,
        n_steps=3
    )

    I_jax = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)
    J_jax = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)

    model_jax.fit(
        I_jax, J_jax,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=0,
        verbose=False
    )

    fwd_warp = model_jax.get_forward_warp()
    inv_warp = model_jax.get_inverse_warp()
    assert fwd_warp is not None
    assert inv_warp is not None
