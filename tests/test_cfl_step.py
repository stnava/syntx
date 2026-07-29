import pytest
import torch
import numpy as np
import jax
import jax.numpy as jnp

from syntx.tvf import TVFModel
from syntx.tvf_jax import TVFModelJAX
from syntx.shooting import GeodesicShootingModel
from syntx.shooting_jax import GeodesicShootingModelJAX
from syntx.syn import local_ncc_loss_nd
from syntx.syn_jax import local_ncc_loss_nd_jax


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


def create_synthetic_pair_3d(shape=(16, 16, 16)):
    """Creates a synthetic 3D sphere pair for testing."""
    D, H, W = shape
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    
    fixed = np.zeros((D, H, W), dtype=np.float32)
    radius = min(D, H, W) // 4
    cz, cy, cx = D // 2, H // 2, W // 2
    mask_fix = ((grid_z - cz) ** 2 + (grid_y - cy) ** 2 + (grid_x - cx) ** 2) <= (radius ** 2)
    fixed[mask_fix] = 1.0
    
    moving = np.zeros((D, H, W), dtype=np.float32)
    cz_m, cy_m, cx_m = cz + 1, cy + 1, cx + 1
    mask_mov = ((grid_z - cz_m) ** 2 + (grid_y - cy_m) ** 2 + (grid_x - cx_m) ** 2) <= (radius ** 2)
    moving[mask_mov] = 1.0
    
    return fixed, moving


def test_tvf_cfl_step_2d():
    """Verify CFL step optimizer in PyTorch TVFModel 2D."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    model = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=2, spacing=[1.0, 2.0])
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    initial_loss = model.forward(fixed_t, moving_t).item()
    
    model.fit(
        fixed_t, moving_t,
        levels=[2, 1],
        epochs_per_level=[10, 10],
        affine_epochs=0,
        optimizer='cfl',
        cfl_voxels=0.2,
        reg_weight=0.0
    )
    
    final_loss = model.forward(fixed_t, moving_t).item()
    assert final_loss <= initial_loss + 1e-4, f"CFL optimization failed to improve loss: {initial_loss} -> {final_loss}"


def test_tvf_jax_cfl_step_2d():
    """Verify CFL step optimizer in JAX TVFModelJAX 2D."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    model = TVFModelJAX(dim=2, image_shape=(32, 32), velocity_shape=(16, 16), n_time_steps=2, spacing=[1.0, 2.0])
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    initial_loss = float(model.forward(fixed_j, moving_j))
    
    model.fit(
        fixed_j, moving_j,
        levels=[2, 1],
        epochs_per_level=[10, 10],
        affine_epochs=0,
        optimizer='cfl',
        cfl_voxels=0.2,
        reg_weight=0.0
    )
    
    final_loss = float(model.forward(fixed_j, moving_j))
    assert final_loss <= initial_loss + 1e-4, f"JAX CFL optimization failed to improve loss: {initial_loss} -> {final_loss}"


def test_shooting_cfl_step_2d():
    """Verify CFL step optimizer in PyTorch GeodesicShootingModel 2D."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    model = GeodesicShootingModel(dim=2, image_shape=(32, 32), n_time_steps=4, spacing=[1.5, 0.5])
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    initial_loss = model.forward(fixed_t, moving_t).item()
    
    model.fit(
        fixed_t, moving_t,
        levels=[2, 1],
        epochs_per_level=[10, 10],
        affine_epochs=0,
        optimizer='cfl',
        cfl_voxels=0.2,
        reg_weight=0.0
    )
    
    final_loss = model.forward(fixed_t, moving_t).item()
    assert final_loss <= initial_loss + 1e-4, f"GeodesicShooting CFL optimization failed: {initial_loss} -> {final_loss}"


def test_shooting_jax_cfl_step_2d():
    """Verify CFL step optimizer in JAX GeodesicShootingModelJAX 2D."""
    fixed, moving = create_synthetic_pair_2d((32, 32))
    
    model = GeodesicShootingModelJAX(dim=2, image_shape=(32, 32), n_time_steps=4, spacing=[1.5, 0.5])
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    initial_loss = float(model.forward(fixed_j, moving_j))
    
    model.fit(
        fixed_j, moving_j,
        levels=[2, 1],
        epochs_per_level=[10, 10],
        affine_epochs=0,
        optimizer='cfl',
        cfl_voxels=0.2,
        reg_weight=0.0
    )
    
    final_loss = float(model.forward(fixed_j, moving_j))
    assert final_loss <= initial_loss + 1e-4, f"JAX GeodesicShooting CFL optimization failed: {initial_loss} -> {final_loss}"


def test_cfl_step_anisotropic_spacing_3d():
    """Verify CFL step with anisotropic spacing in 3D across PyTorch and JAX backends."""
    fixed, moving = create_synthetic_pair_3d((16, 16, 16))
    spacing = [1.0, 2.0, 0.5]  # Anisotropic physical spacing (dz, dy, dx)
    
    # 1. PyTorch TVFModel
    torch_model = TVFModel(dim=3, image_shape=(16, 16, 16), velocity_shape=(8, 8, 8), n_time_steps=2, spacing=spacing)
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0)
    
    torch_model.fit(
        fixed_t, moving_t,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        optimizer='cfl',
        cfl_voxels=0.1,
        reg_weight=0.0
    )
    warp_torch = torch_model.get_forward_warp()
    warp_torch_spatial = warp_torch.squeeze(0) if warp_torch.ndim == 5 else warp_torch
    assert warp_torch_spatial.shape == (16, 16, 16, 3)
    assert not torch.isnan(warp_torch).any()
    
    # 2. JAX TVFModelJAX
    jax_model = TVFModelJAX(dim=3, image_shape=(16, 16, 16), velocity_shape=(8, 8, 8), n_time_steps=2, spacing=spacing)
    fixed_j = jnp.array(fixed)[None, None, ...]
    moving_j = jnp.array(moving)[None, None, ...]
    
    jax_model.fit(
        fixed_j, moving_j,
        levels=[2, 1],
        epochs_per_level=[5, 5],
        affine_epochs=0,
        optimizer='cfl',
        cfl_voxels=0.1,
        reg_weight=0.0
    )
    warp_jax = jax_model.get_forward_warp()
    warp_jax_spatial = warp_jax.squeeze(0) if warp_jax.ndim == 5 else warp_jax
    assert warp_jax_spatial.shape == (16, 16, 16, 3)
    assert not jnp.isnan(warp_jax).any()
