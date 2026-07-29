"""
Unit Tests for Syntx Visualization Module (syntx.plot_comparison).
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
from syntx.visualization import (
    extract_2d_slice,
    plot_comparison,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
    plot_structural_comparison
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

    # 2D scalar slice must be transposed to match ANTs native 2D plot
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


def test_plot_comparison_2d_side_by_side(sample_2d_images, tmp_path):
    """Tests 2D plot_comparison in side_by_side mode."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]

    out_png = str(tmp_path / "plot_2d_side_by_side.png")

    fig = plot_comparison(
        images=[fi, mi],
        mode="side_by_side",
        titles=["Target (Fixed)", "Source (Moving)"],
        main_title="2D Side-by-Side Registration Test",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_2d_difference(sample_2d_images, tmp_path):
    """Tests 2D plot_comparison in difference mode."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]

    out_png = str(tmp_path / "plot_2d_diff.png")

    fig = plot_comparison(
        images=[fi, mi],
        mode="difference",
        main_title="2D Difference Map",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_2d_edge_overlay(sample_2d_images, tmp_path):
    """Tests 2D plot_comparison in edge_overlay mode."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]

    out_png = str(tmp_path / "plot_2d_edges.png")

    fig = plot_comparison(
        images=[fi, mi],
        mode="edge_overlay",
        main_title="2D Canny Edge Overlap",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_2d_deformed_grid(sample_2d_images, tmp_path):
    """Tests 2D plot_comparison in deformed_grid mode."""
    warp = sample_2d_images["warp_2d"]
    fi = sample_2d_images["fi_np"]

    out_png = str(tmp_path / "plot_2d_grid.png")

    fig = plot_comparison(
        images=[warp, fi],
        mode="deformed_grid",
        main_title="2D Mesh Grid",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_2d_jacobian(sample_2d_images, tmp_path):
    """Tests 2D plot_comparison in jacobian mode."""
    warp = sample_2d_images["warp_2d"]

    out_png = str(tmp_path / "plot_2d_jac.png")

    fig = plot_comparison(
        images=[warp],
        mode="jacobian",
        main_title="2D Jacobian Determinant Map",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_3d_orthogonal(sample_3d_images, tmp_path):
    """Tests 3D plot_comparison in orthogonal mode."""
    fi = sample_3d_images["fi_ants"]
    mi = sample_3d_images["mi_ants"]

    out_png = str(tmp_path / "plot_3d_ortho.png")

    fig = plot_comparison(
        images={"Fixed": fi, "Moving": mi},
        mode="orthogonal",
        main_title="3D Orthogonal Triplanar Comparison",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_dict_input(sample_2d_images, tmp_path):
    """Tests plot_comparison with dictionary of labeled images."""
    dict_imgs = {
        "Target (Fixed)": sample_2d_images["fi_ants"],
        "Source (Moving)": sample_2d_images["mi_ants"],
        "PyTorch Model": sample_2d_images["mi_np"]
    }

    out_png = str(tmp_path / "plot_dict.png")

    fig = plot_comparison(
        images=dict_imgs,
        mode="side_by_side",
        main_title="Dict Input Benchmark Comparison",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_plot_comparison_light_theme(sample_2d_images, tmp_path):
    """Tests plot_comparison in light theme mode."""
    fi = sample_2d_images["fi_ants"]
    mi = sample_2d_images["mi_ants"]

    out_png = str(tmp_path / "plot_light.png")

    fig = plot_comparison(
        images=[fi, mi],
        theme="light",
        cbar=True,
        main_title="Light Theme Test",
        filename=out_png
    )

    assert os.path.exists(out_png)
    plt.close(fig)


def test_top_level_syntx_exports():
    """Verifies syntx exports plot_comparison, plot_structural_comparison, plot_edge_overlay, render_standard_4panel."""
    assert hasattr(syntx, "plot_comparison")
    assert hasattr(syntx, "extract_2d_slice")
    assert hasattr(syntx, "plot_structural_comparison")
    assert hasattr(syntx, "plot_edge_overlay")
    assert hasattr(syntx, "plot_deformation_grid")
    assert hasattr(syntx, "render_standard_4panel")
