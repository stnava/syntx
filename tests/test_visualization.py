"""
Unit Tests for Syntx Visualization Subpackage (syntx.viz).
"""

import os
import pytest
import numpy as np
import torch
import ants
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

import syntx
from syntx.viz import (
    extract_2d_slice,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
    render_input_pair_figure,
)


@pytest.fixture
def sample_2d_images():
    """Generates 2D synthetic images (ANTsImage, Tensor, NumPy)."""
    grid = np.zeros((64, 64), dtype=np.float32)
    grid[16:48, 16:48] = 1.0

    fi_ants = ants.from_numpy(grid)

    mov_grid = np.zeros((64, 64), dtype=np.float32)
    mov_grid[20:52, 20:52] = 1.0
    mi_ants = ants.from_numpy(mov_grid)

    fi_torch = torch.from_numpy(grid)
    mi_torch = torch.from_numpy(mov_grid)

    warp_2d = np.random.randn(64, 64, 2).astype(np.float32) * 0.5

    return {
        "fi_ants": fi_ants,
        "mi_ants": mi_ants,
        "fi_np": grid,
        "mi_np": mov_grid,
        "fi_torch": fi_torch,
        "mi_torch": mi_torch,
        "warp_2d": warp_2d
    }


@pytest.fixture
def sample_3d_images():
    """Generates 3D synthetic volumes (ANTsImage, NumPy, Tensor)."""
    grid = np.zeros((32, 32, 32), dtype=np.float32)
    grid[8:24, 8:24, 8:24] = 1.0

    fi_ants = ants.from_numpy(grid)

    mov_grid = np.zeros((32, 32, 32), dtype=np.float32)
    mov_grid[10:26, 10:26, 10:26] = 1.0
    mi_ants = ants.from_numpy(mov_grid)

    warp_3d = np.random.randn(32, 32, 32, 3).astype(np.float32) * 0.5

    return {
        "fi_ants": fi_ants,
        "mi_ants": mi_ants,
        "fi_np": grid,
        "mi_np": mov_grid,
        "warp_3d": warp_3d
    }


def test_extract_2d_slice_orientation_2d(sample_2d_images):
    """Verifies 2D slice extraction respects ANTs 2D transpose (.T) orientation convention."""
    arr_2d = sample_2d_images["fi_np"]
    slice_out = extract_2d_slice(arr_2d)
    assert slice_out.shape == (arr_2d.shape[1], arr_2d.shape[0])
    np.testing.assert_allclose(slice_out, arr_2d.T)


def test_extract_2d_slice_orientation_3d(sample_3d_images):
    """Verifies 3D volume slice extraction along Axial, Coronal, Sagittal axes."""
    vol_3d = sample_3d_images["fi_ants"]

    sl_ax = extract_2d_slice(vol_3d, slice_axis=2)  # Axial
    sl_co = extract_2d_slice(vol_3d, slice_axis=1)  # Coronal
    sl_sa = extract_2d_slice(vol_3d, slice_axis=0)  # Sagittal

    assert sl_ax.ndim == 2
    assert sl_co.ndim == 2
    assert sl_sa.ndim == 2


def test_render_input_pair_figure_2d(sample_2d_images, tmp_path):
    """Tests render_input_pair_figure on 2D inputs."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]
    out_png = str(tmp_path / "fig1_input_pair_2d.png")

    fig = render_input_pair_figure(fi, mi, output_path=out_png, title="2D Input Pair")
    assert os.path.exists(out_png)
    plt.close(fig)


def test_render_input_pair_figure_3d(sample_3d_images, tmp_path):
    """Tests render_input_pair_figure on 3D inputs."""
    fi = sample_3d_images["fi_ants"]
    mi = sample_3d_images["mi_ants"]
    out_png = str(tmp_path / "fig1_input_pair_3d.png")

    fig = render_input_pair_figure(fi, mi, output_path=out_png, title="3D Orthographic Input Pair")
    assert os.path.exists(out_png)
    plt.close(fig)


def test_render_standard_4panel_2d(sample_2d_images, tmp_path):
    """Tests render_standard_4panel on 2D inputs."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]
    warp = sample_2d_images["warp_2d"]
    detJ = np.ones((64, 64), dtype=np.float32)
    inv_err = np.zeros((64, 64), dtype=np.float32)
    out_png = str(tmp_path / "fig2_4panel_2d.png")

    fig = render_standard_4panel(fi, mi, warp=warp, detJ=detJ, inv_err_map=inv_err, filename=out_png)
    assert os.path.exists(out_png)
    plt.close(fig)


def test_render_standard_4panel_3d(sample_3d_images, tmp_path):
    """Tests render_standard_4panel on 3D inputs."""
    fi = sample_3d_images["fi_ants"]
    mi = sample_3d_images["mi_ants"]
    warp = sample_3d_images["warp_3d"]
    detJ = np.ones((32, 32, 32), dtype=np.float32)
    inv_err = np.zeros((32, 32, 32), dtype=np.float32)
    out_png = str(tmp_path / "fig2_4panel_3d.png")

    fig = render_standard_4panel(fi, mi, warp=warp, detJ=detJ, inv_err_map=inv_err, filename=out_png)
    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_edge_overlay(sample_2d_images, tmp_path):
    """Tests plot_edge_overlay utility."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]
    out_png = str(tmp_path / "edge_overlay.png")

    fig = plot_edge_overlay(fi, mi, filename=out_png)
    assert os.path.exists(out_png)
    plt.close(fig)


def test_top_level_viz_exports():
    """Verifies syntx exports render_input_pair_figure, render_standard_4panel, extract_2d_slice."""
    assert hasattr(syntx, "render_input_pair_figure")
    assert hasattr(syntx, "render_standard_4panel")
    assert hasattr(syntx, "extract_2d_slice")
    assert hasattr(syntx, "plot_deformation_grid")
    assert hasattr(syntx, "plot_edge_overlay")
