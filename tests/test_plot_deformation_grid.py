import torch
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import syntx

def test_plot_deformation_grid_numpy_2d():
    # 2D dummy grid
    warp_2d = np.zeros((32, 32, 2), dtype=np.float32)
    warp_2d[..., 0] = np.sin(np.linspace(0, 3, 32))[:, None] * 2.0
    fig = syntx.plot_deformation_grid(warp_2d, grid_spacing=4, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_deformation_grid_pytorch_3d():
    # 3D dummy tensor
    warp_3d = torch.zeros((1, 3, 20, 20, 20), dtype=torch.float32)
    fig = syntx.plot_deformation_grid(warp_3d, slice_axis=2, slice_idx=10, grid_spacing=4, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_deformation_grid_with_background():
    warp_2d = np.zeros((32, 32, 2), dtype=np.float32)
    bg_2d = np.random.randn(32, 32).astype(np.float32)
    fig = syntx.plot_deformation_grid(warp_2d, fixed=bg_2d, grid_spacing=4, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
