import pytest
import torch
import numpy as np

from syntx.core.grid import (
    get_physical_grid_torch,
    get_physical_to_normalized_affine,
    physical_to_normalized_fast,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    compose_grids,
    grid_sample_nd,
    grid_sample_bspline_torch,
)


def test_grid_physical_and_normalized_parity_2d():
    shape = (16, 16)
    spacing = (1.5, 1.5)
    origin = (10.0, 20.0)
    direction = [[1.0, 0.0], [0.0, 1.0]]

    grid_phys = get_physical_grid_torch(shape, spacing, origin, direction)
    assert grid_phys.shape == (1, 16, 16, 2)

    # Convert back to normalized coordinates [-1, 1]
    grid_norm = physical_to_normalized_torch(grid_phys, shape, spacing, origin, direction)
    assert grid_norm.shape == (1, 16, 16, 2)
    assert torch.allclose(grid_norm[0, 0, 0], torch.tensor([-1.0, -1.0]), atol=1e-5)
    assert torch.allclose(grid_norm[0, -1, -1], torch.tensor([1.0, 1.0]), atol=1e-5)

    # Fast affine precomputation parity
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_rev = np.asarray(direction)[::-1, ::-1].copy()
    shape_t = torch.tensor(list(shape), dtype=torch.float32)
    spacing_t = torch.tensor(spacing_rev, dtype=torch.float32)
    origin_t = torch.tensor(origin_rev, dtype=torch.float32)
    dir_t = torch.tensor(dir_rev, dtype=torch.float32)

    M, b = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, dir_t)
    grid_norm_fast = physical_to_normalized_fast(grid_phys, M, b)
    grid_norm_cached = physical_to_normalized_torch_cached(grid_phys, shape_t, spacing_t, origin_t, dir_t)
    assert torch.allclose(grid_norm_fast, grid_norm_cached, atol=1e-6)
    assert torch.allclose(grid_norm_fast, grid_norm, atol=1e-5)


def test_grid_physical_and_normalized_parity_3d():
    shape = (8, 10, 12)
    spacing = (1.0, 1.2, 0.8)
    origin = (5.0, -10.0, 15.0)
    # Non-trivial rotated direction matrix
    angle = 0.3
    c, s = np.cos(angle), np.sin(angle)
    direction = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    grid_phys = get_physical_grid_torch(shape, spacing, origin, direction)
    assert grid_phys.shape == (1, 8, 10, 12, 3)

    grid_norm = physical_to_normalized_torch(grid_phys, shape, spacing, origin, direction)
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_rev = np.asarray(direction)[::-1, ::-1].copy()
    shape_t = torch.tensor(list(shape), dtype=torch.float32)
    spacing_t = torch.tensor(spacing_rev, dtype=torch.float32)
    origin_t = torch.tensor(origin_rev, dtype=torch.float32)
    dir_t = torch.tensor(dir_rev, dtype=torch.float32)

    M, b = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, dir_t)
    grid_norm_fast = physical_to_normalized_fast(grid_phys, M, b)
    grid_norm_cached = physical_to_normalized_torch_cached(grid_phys, shape_t, spacing_t, origin_t, dir_t)
    assert torch.allclose(grid_norm_fast, grid_norm_cached, atol=1e-6)
    assert torch.allclose(grid_norm_fast, grid_norm, atol=1e-5)


def test_compose_grids_identity():
    # Identity grid in [-1, 1]
    y = torch.linspace(-1, 1, 16)
    x = torch.linspace(-1, 1, 16)
    mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
    id_grid = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0)  # (1, 16, 16, 2)

    composed = compose_grids(id_grid, id_grid)
    assert torch.allclose(composed, id_grid, atol=1e-5)


def test_grid_sample_nd_modes():
    img = torch.randn(1, 1, 16, 16)
    y = torch.linspace(-1, 1, 16)
    x = torch.linspace(-1, 1, 16)
    mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
    id_grid = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0)

    sampled_lin = grid_sample_nd(img, id_grid, mode='bilinear')
    assert torch.allclose(sampled_lin, img, atol=1e-5)

    sampled_nn = grid_sample_nd(img, id_grid, interpolator='nearestNeighbor')
    assert torch.allclose(sampled_nn, img, atol=1e-5)

    sampled_bspline = grid_sample_bspline_torch(img, id_grid)
    assert sampled_bspline.shape == img.shape
