"""
Unit Tests for syntx.viz Sub-Package (Visualization & Reporting Infrastructure).
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import ants

import syntx
import syntx.viz as viz


@pytest.fixture
def temp_viz_dir():
    temp_dir = tempfile.mkdtemp(prefix="syntx_viz_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_viz_subpackage_exports():
    assert hasattr(viz, "render_input_pair_figure")
    assert hasattr(viz, "render_standard_4panel")
    assert hasattr(viz, "render_label_alignment_figure")
    assert hasattr(viz, "plot_label_overlap_stats")
    assert hasattr(viz, "plot_jacobian_distribution")
    assert hasattr(viz, "create_registration_report")
    assert hasattr(viz, "create_visualization_gallery")
    assert hasattr(viz, "build_engine_provenance")
    assert hasattr(viz, "plot_deformation_grid")
    assert hasattr(viz, "plot_edge_overlay")
    assert hasattr(viz, "extract_2d_slice")


def test_render_input_pair_figure_direct_import(temp_viz_dir):
    fi = np.pad(np.ones((20, 20), dtype=np.float32), 5)
    mi = np.pad(np.ones((20, 20), dtype=np.float32), 5) * 0.7

    out_png = os.path.join(temp_viz_dir, "test_pair.png")
    fig = viz.render_input_pair_figure(fi, mi, output_path=out_png, title="Direct Viz Import")

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
    assert fig is not None


def test_build_engine_provenance():
    prov = viz.build_engine_provenance(
        algorithm="syntx.tvf",
        backend="pytorch",
        device="mps",
        fit_time=8.5,
        reg_iterations=[100, 50]
    )

    assert prov["algorithm"] == "syntx.tvf"
    assert prov["backend"] == "pytorch"
    assert prov["device"] == "mps"
    assert prov["fit_time"] == 8.5
    assert prov["syntx_version"] == "1.1.8"


def test_render_input_pair_figure_anisotropic(temp_viz_dir):
    arr_f = np.zeros((24, 24, 16), dtype=np.float32)
    arr_f[6:18, 6:18, 4:12] = 1.0
    arr_m = np.zeros((24, 24, 16), dtype=np.float32)
    arr_m[8:20, 8:20, 4:12] = 0.8

    fi_aniso = ants.from_numpy(arr_f, spacing=(1.0, 1.0, 3.0))
    mi_aniso = ants.from_numpy(arr_m, spacing=(1.0, 1.0, 3.0))

    out_png = os.path.join(temp_viz_dir, "anisotropic_pair.png")
    fig = viz.render_input_pair_figure(fi_aniso, mi_aniso, output_path=out_png, title="Anisotropic Spacing (1x1x3mm)")

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1500
    assert fig is not None


def test_render_label_alignment_figure(temp_viz_dir):
    lbl_f = np.zeros((24, 24, 24), dtype=np.int32)
    lbl_f[6:18, 6:18, 6:18] = 10
    lbl_w = np.zeros((24, 24, 24), dtype=np.int32)
    lbl_w[8:20, 8:20, 6:18] = 10

    out_png = os.path.join(temp_viz_dir, "label_alignment.png")
    fig = viz.render_label_alignment_figure(lbl_f, lbl_w, output_path=out_png, title="Mindboggle DKT Alignment")

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1500
    assert fig is not None


def test_plot_label_overlap_stats(temp_viz_dir):
    dice_data = {
        "fixed_dice": [0.65, 0.70, 0.68, 0.72],
        "moving_dice": [0.64, 0.69, 0.67, 0.71],
        "sym_dice": [0.645, 0.695, 0.675, 0.715],
        "per_region": {"Superior frontal": 0.72, "Precentral": 0.68, "Postcentral": 0.65}
    }

    out_png = os.path.join(temp_viz_dir, "label_stats.png")
    fig = viz.plot_label_overlap_stats(dice_data, output_path=out_png)

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1500
    assert fig is not None


def test_plot_jacobian_distribution(temp_viz_dir):
    detJ_arr = np.random.normal(loc=1.0, scale=0.15, size=(20, 20, 20))

    out_png = os.path.join(temp_viz_dir, "jacobian_dist.png")
    fig = viz.plot_jacobian_distribution(detJ_arr, output_path=out_png)

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1500
    assert fig is not None


def test_create_visualization_gallery(temp_viz_dir):
    fi = ants.from_numpy(np.pad(np.ones((20, 20, 20), dtype=np.float32), 4))
    mi = ants.from_numpy(np.pad(np.ones((20, 20, 20), dtype=np.float32), 4) * 0.8)

    out_html = os.path.join(temp_viz_dir, "gallery.html")
    res_path = viz.create_visualization_gallery(fi, mi, output_path=out_html)

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 5000


