import pytest
import torch
import numpy as np
import jax
import jax.numpy as jnp

from syntx.tvf import TVFModel
from syntx.tvf_jax import TVFModelJAX
from syntx.shooting import GeodesicShootingModel
from syntx.shooting_jax import GeodesicShootingModelJAX


def create_synthetic_pair_2d(shape=(32, 32)):
    """Creates a synthetic 2D circle/square image pair for testing."""
    H, W = shape
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    fixed = np.zeros((H, W), dtype=np.float32)
    radius = min(H, W) // 4
    cy, cx = H // 2, W // 2
    mask_fix = ((grid_y - cy) ** 2 + (grid_x - cx) ** 2) <= (radius ** 2)
    fixed[mask_fix] = 1.0
    
    moving = np.zeros((H, W), dtype=np.float32)
    cy_m, cx_m = cy + 2, cx + 2
    mask_mov = ((grid_y - cy_m) ** 2 + (grid_x - cx_m) ** 2) <= (radius ** 2)
    moving[mask_mov] = 1.0
    
    return fixed, moving


def test_zero_reg_weight_fixed_pyramid_3level():
    """Verify models support reg_weight=0.0 with 3-level multi-res pyramid [4, 2, 1]."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    # 1. PyTorch TVF
    model_tvf = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=2)
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    loss_init = model_tvf.forward(fixed_t, moving_t).item()
    model_tvf.fit(
        fixed_t, moving_t,
        levels=[4, 2, 1],
        epochs_per_level=[5, 5, 5],
        affine_epochs=0,
        reg_weight=0.0,
        lr=0.5
    )
    loss_final = model_tvf.forward(fixed_t, moving_t).item()
    assert loss_final <= loss_init + 1e-4
    
    # 2. JAX TVF
    model_tvf_jax = TVFModelJAX(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=2)
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    loss_init_j = float(model_tvf_jax.forward(fixed_j, moving_j))
    model_tvf_jax.fit(
        fixed_j, moving_j,
        levels=[4, 2, 1],
        epochs_per_level=[5, 5, 5],
        affine_epochs=0,
        reg_weight=0.0,
        lr=0.5
    )
    loss_final_j = float(model_tvf_jax.forward(fixed_j, moving_j))
    assert loss_final_j <= loss_init_j + 1e-4
    
    # 3. PyTorch Shooting
    model_shoot = GeodesicShootingModel(dim=2, image_shape=(32, 32), n_time_steps=4)
    model_shoot.fit(
        fixed_t, moving_t,
        levels=[4, 2, 1],
        epochs_per_level=[5, 5, 5],
        affine_epochs=0,
        reg_weight=0.0,
        lr=0.5
    )
    loss_shoot = model_shoot.forward(fixed_t, moving_t, reg_weight=0.0).item()
    assert not np.isnan(loss_shoot)
    
    # 4. JAX Shooting
    model_shoot_jax = GeodesicShootingModelJAX(dim=2, image_shape=(32, 32), n_time_steps=4)
    model_shoot_jax.fit(
        fixed_j, moving_j,
        levels=[4, 2, 1],
        epochs_per_level=[5, 5, 5],
        affine_epochs=0,
        reg_weight=0.0,
        lr=0.5
    )
    loss_shoot_j = float(model_shoot_jax.forward(fixed_j, moving_j, reg_weight=0.0))
    assert not np.isnan(loss_shoot_j)


@pytest.mark.parametrize("optimizer_name", ["adam", "lars", "sgd"])
def test_optimizer_sweeps_tvf(optimizer_name):
    """Verify Adam, LARS, and SGD optimizers for TVFModel (PyTorch) and TVFModelJAX (JAX)."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    # PyTorch TVFModel
    model_pt = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=2)
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    model_pt.fit(
        fixed_t, moving_t,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        optimizer=optimizer_name,
        lr=0.2,
        reg_weight=0.0
    )
    loss_pt = model_pt.forward(fixed_t, moving_t).item()
    assert not np.isnan(loss_pt)
    
    # JAX TVFModelJAX
    model_jax = TVFModelJAX(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=2)
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    model_jax.fit(
        fixed_j, moving_j,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        optimizer=optimizer_name,
        lr=0.2,
        reg_weight=0.0
    )
    loss_jax = float(model_jax.forward(fixed_j, moving_j))
    assert not np.isnan(loss_jax)


@pytest.mark.parametrize("optimizer_name", ["adam", "lars", "sgd"])
def test_optimizer_sweeps_shooting(optimizer_name):
    """Verify Adam, LARS, and SGD optimizers for GeodesicShootingModel (PyTorch) and GeodesicShootingModelJAX (JAX)."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    # PyTorch GeodesicShootingModel
    model_pt = GeodesicShootingModel(dim=2, image_shape=(32, 32), n_time_steps=4)
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    model_pt.fit(
        fixed_t, moving_t,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        optimizer=optimizer_name,
        lr=0.2,
        reg_weight=0.0
    )
    loss_pt = model_pt.forward(fixed_t, moving_t).item()
    assert not np.isnan(loss_pt)
    
    # JAX GeodesicShootingModelJAX
    model_jax = GeodesicShootingModelJAX(dim=2, image_shape=(32, 32), n_time_steps=4)
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    model_jax.fit(
        fixed_j, moving_j,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        optimizer=optimizer_name,
        lr=0.2,
        reg_weight=0.0
    )
    loss_jax = float(model_jax.forward(fixed_j, moving_j))
    assert not np.isnan(loss_jax)


@pytest.mark.parametrize("sigma_mode", ["voxel", "physical"])
def test_voxel_vs_physical_sigma_space(sigma_mode):
    """Verify voxel vs physical sigma space evaluation with flow_sigma/total_sigma aliases."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    spacing = [1.0, 2.0]  # anisotropic spacing
    
    # 1. PyTorch TVFModel
    model_pt = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), spacing=spacing)
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    model_pt.fit(
        fixed_t, moving_t,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        flow_sigma=3.0,     # tests alias for fluid_sigma (var=3.0 => std=sqrt(3.0))
        total_sigma=1.0,    # tests alias for elastic_sigma (var=1.0 => std=1.0)
        sigma_mode=sigma_mode,
        reg_weight=0.0
    )
    warp_pt = model_pt.get_forward_warp()
    warp_pt_spatial = warp_pt.squeeze(0) if warp_pt.ndim == 4 else warp_pt
    assert warp_pt_spatial.shape == (32, 32, 2)
    assert not torch.isnan(warp_pt).any()
    
    # 2. JAX TVFModelJAX
    model_jax = TVFModelJAX(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), spacing=spacing)
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    model_jax.fit(
        fixed_j, moving_j,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        flow_sigma=3.0,
        total_sigma=1.0,
        sigma_mode=sigma_mode,
        reg_weight=0.0
    )
    warp_jax = model_jax.get_forward_warp()
    warp_jax_spatial = warp_jax.squeeze(0) if warp_jax.ndim == 4 else warp_jax
    assert warp_jax_spatial.shape == (32, 32, 2)
    assert not jnp.isnan(warp_jax).any()
