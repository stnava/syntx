"""
Unit tests targeting high code coverage for syntx.viz subpackage.
"""

import os
import numpy as np
import pytest
import ants
import matplotlib.pyplot as plt

from syntx.viz.core import AnatomicalVisualizer, AnatomicalSlice
from syntx.viz.colormaps import build_dkt_label_palette, get_dkt_colormap, get_dkt_label_color_dict
from syntx.viz.figures import (
    render_input_pair_figure,
    render_standard_4panel,
    render_label_alignment_figure,
    plot_deformation_grid,
    plot_edge_overlay,
    plot_time_varying_velocity_grid,
    plot_correspondence_vectors,
    plot_vector_field,
    plot_deformation_tensor_rgb
)
from syntx.viz.stats import plot_label_overlap_stats, plot_jacobian_distribution
from syntx.viz.gallery import create_visualization_gallery, fig_to_base64_png
from syntx.viz.reports import create_registration_report, build_engine_provenance


def test_colormaps_extended():
    color_map, lut = build_dkt_label_palette([1, 2, 3, "Left-Cerebral-Cortex", "Right-Cerebral-Cortex"])
    assert isinstance(color_map, dict)
    assert lut.ndim == 2

    cmap = get_dkt_colormap(max_label=100)
    assert cmap is not None

    cdict = get_dkt_label_color_dict([10, 20, "Brain-Stem"])
    assert isinstance(cdict, dict)


def test_core_visualizer_extended():
    arr = np.random.randn(20, 20, 20).astype(np.float32)
    img_ants = ants.from_numpy(arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

    img_proc, arr_out, sp = AnatomicalVisualizer.prepare_image(img_ants, reorient=True)
    assert arr_out.shape == (20, 20, 20)

    slice_obj = AnatomicalVisualizer.extract_slice(img_ants, plane="axial", slice_idx=10)
    assert isinstance(slice_obj, AnatomicalSlice)
    assert slice_obj.data.ndim == 2

    # Test Coronal and Sagittal planes
    slice_co = AnatomicalVisualizer.extract_slice(img_ants, plane="coronal", slice_idx=10)
    slice_sa = AnatomicalVisualizer.extract_slice(img_ants, plane="sagittal", slice_idx=10)
    assert slice_co.data.ndim == 2
    assert slice_sa.data.ndim == 2


def test_figures_extended(tmp_path):
    fi_arr = np.pad(np.ones((20, 20)), 10).astype(np.float32)
    mi_arr = np.pad(np.ones((20, 20)), 10).astype(np.float32)

    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    warp_2d = np.zeros((40, 40, 2), dtype=np.float32)
    detJ_2d = np.ones((40, 40), dtype=np.float32)
    inv_err_2d = np.zeros((40, 40), dtype=np.float32)

    fig_grid = plot_deformation_grid(warp_2d, fixed=fi, show=False)
    assert fig_grid is not None
    plt.close(fig_grid)

    fig_edge = plot_edge_overlay(fi, mi, show=False)
    assert fig_edge is not None
    plt.close(fig_edge)

    fig_corr = plot_correspondence_vectors(warp_2d, fixed=fi, show=False)
    assert fig_corr is not None
    plt.close(fig_corr)

    fig_vec = plot_vector_field(warp_2d, fixed=fi, show=False)
    assert fig_vec is not None
    plt.close(fig_vec)

    fi3d = ants.from_numpy(np.ones((20, 20, 20), dtype=np.float32), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    warp_3d = np.zeros((20, 20, 20, 3), dtype=np.float32)
    fig_tensor = plot_deformation_tensor_rgb(warp_3d, fixed=fi3d, show_figure=False)
    assert fig_tensor is not None
    plt.close(fig_tensor)

    # Test label alignment figure
    fl3d_arr = (np.ones((20, 20, 20), dtype=np.float32) > 0).astype(np.uint32)
    wl3d_arr = (np.ones((20, 20, 20), dtype=np.float32) > 0).astype(np.uint32)
    fl3d = ants.from_numpy(fl3d_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    wl3d = ants.from_numpy(wl3d_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

    fig_label = render_label_alignment_figure(fl3d, wl3d, fixed_image=fi3d, show_figure=False)
    assert fig_label is not None
    plt.close(fig_label)


def test_stats_and_gallery_extended(tmp_path):
    detJ = np.ones((20, 20), dtype=np.float32)
    detJ[5, 5] = -0.1  # simulate 1 folding voxel

    fig_jac = plot_jacobian_distribution(detJ, show_figure=False)
    assert fig_jac is not None
    plt.close(fig_jac)

    fi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    warp_2d = np.zeros((20, 20, 2), dtype=np.float32)
    detJ_2d = np.ones((20, 20), dtype=np.float32)
    inv_err_2d = np.zeros((20, 20), dtype=np.float32)

    gallery_html = os.path.join(tmp_path, "gallery.html")
    out_gallery = create_visualization_gallery(
        fixed=fi,
        moving=mi,
        warped=mi,
        warp=warp_2d,
        detJ=detJ_2d,
        inv_err_map=inv_err_2d,
        output_path=gallery_html
    )
    assert os.path.exists(out_gallery)
