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
    assert hasattr(viz, "create_registration_report")
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

