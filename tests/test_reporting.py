"""
Unit Tests for syntx.reporting (Standard Registration Reporting Infrastructure).
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import ants

import syntx
from syntx.reporting import create_registration_report, _parse_image_metadata, _compute_jacobian_stats


@pytest.fixture
def temp_report_dir():
    temp_dir = tempfile.mkdtemp(prefix="syntx_report_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_parse_image_metadata():
    fi_arr = np.zeros((30, 30, 30), dtype=np.float32)
    meta_arr = _parse_image_metadata(fi_arr, "TestArray")
    assert meta_arr["name"] == "TestArray"
    assert meta_arr["shape"] == "(30, 30, 30)"
    assert not meta_arr["is_ants"]

    fi_ants = ants.from_numpy(fi_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    meta_ants = _parse_image_metadata(fi_ants, "TestANTS")
    assert meta_ants["name"] == "TestANTS"
    assert meta_ants["is_ants"]
    assert "1.000 × 1.000 × 1.000 mm" in meta_ants["spacing"]


def test_compute_jacobian_stats_3d():
    warp_np = np.zeros((20, 20, 20, 3), dtype=np.float32)
    detJ, stats = _compute_jacobian_stats(warp_np)

    assert detJ.shape == (20, 20, 20)
    assert pytest.approx(stats["min"], 0.01) == 1.0
    assert pytest.approx(stats["max"], 0.01) == 1.0
    assert stats["folding_pct"] == 0.0


def test_create_registration_report_2d(temp_report_dir):
    fi = np.pad(np.ones((20, 20)), 10)
    mi = np.pad(np.ones((20, 20)), 10)
    warped = mi.copy()
    warp = np.zeros((40, 40, 2), dtype=np.float32)

    output_html = os.path.join(temp_report_dir, "report_2d.html")
    provenance = {
        "algorithm": "syntx.syn",
        "backend": "pytorch",
        "device": "cpu",
        "fit_time": 1.25,
        "iterations": [100, 50],
    }

    report = create_registration_report(
        fixed=fi,
        moving=mi,
        warped=warped,
        warp=warp,
        output_html=output_html,
        fixed_name="Fixed2D",
        moving_name="Moving2D",
        provenance=provenance,
        title="2D Registration Unit Test Report"
    )

    assert os.path.exists(output_html)
    assert os.path.getsize(output_html) > 2000
    assert os.path.exists(report["fig_path"])
    assert report["provenance"]["backend"] == "pytorch"
    assert report["provenance"]["fit_time"] == 1.25


def test_create_registration_report_3d_ants(temp_report_dir):
    fi_arr = np.zeros((30, 30, 30), dtype=np.float32)
    fi_arr[10:20, 10:20, 10:20] = 1.0
    mi_arr = np.zeros((30, 30, 30), dtype=np.float32)
    mi_arr[12:22, 12:22, 10:20] = 1.0

    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    warped = mi

    fl_arr = (fi_arr > 0).astype(np.uint32)
    wl_arr = (mi_arr > 0).astype(np.uint32)
    fl = ants.from_numpy(fl_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    wl = ants.from_numpy(wl_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

    warp = np.zeros((30, 30, 30, 3), dtype=np.float32)

    output_html = os.path.join(temp_report_dir, "report_3d.html")
    provenance = {
        "algorithm": "syntx.tvf",
        "backend": "jax",
        "device": "mps",
        "fit_time": 12.3,
        "solver": "rk4",
        "fluid_sigma": 1.5,
        "learning_rate": 0.30,
    }

    report = create_registration_report(
        fixed=fi,
        moving=mi,
        warped=warped,
        warp=warp,
        fixed_label=fl,
        warped_label=wl,
        output_html=output_html,
        fixed_name="FixedSubject3D",
        moving_name="MovingSubject3D",
        provenance=provenance,
        title="3D ANTs Image Registration Verification Report"
    )

    assert os.path.exists(output_html)
    assert os.path.getsize(output_html) > 3000
    assert os.path.exists(report["fig_path"])
    assert report["dice"] is not None
    assert report["dice"] > 0.0
    assert report["jacobian"]["folding_pct"] == 0.0


def test_engine_provided_provenance(temp_report_dir):
    fi = ants.from_numpy(np.pad(np.ones((20, 20), dtype=np.float32), 5))
    mi = ants.from_numpy(np.pad(np.ones((20, 20), dtype=np.float32), 5))

    reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[5, 2], backend='pytorch')

    assert "provenance" in reg
    assert reg["provenance"]["algorithm"] == "syntx.syn"
    assert reg["provenance"]["backend"] == "pytorch"

    output_html = os.path.join(temp_report_dir, "auto_engine_report.html")
    report = syntx.create_registration_report(
        fixed=fi,
        moving=mi,
        reg=reg,
        output_html=output_html,
        title="Auto Provenance Test"
    )

    assert os.path.exists(output_html)
    assert report["provenance"]["algorithm"] == "syntx.syn"
    assert report["provenance"]["backend"] == "pytorch"


def test_render_input_pair_figure_2d(temp_report_dir):
    fi = np.pad(np.ones((20, 20), dtype=np.float32), 5)
    mi = np.pad(np.ones((20, 20), dtype=np.float32), 5) * 0.5

    out_png = os.path.join(temp_report_dir, "input_pair_2d.png")
    fig = syntx.render_input_pair_figure(fi, mi, output_path=out_png, title="2D Input Pair")

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
    assert fig is not None


def test_render_input_pair_figure_3d(temp_report_dir):
    fi_arr = np.zeros((24, 24, 24), dtype=np.float32)
    fi_arr[6:18, 6:18, 6:18] = 1.0
    mi_arr = np.zeros((24, 24, 24), dtype=np.float32)
    mi_arr[8:20, 8:20, 8:20] = 0.8

    fi = ants.from_numpy(fi_arr)
    mi = ants.from_numpy(mi_arr)

    out_png = os.path.join(temp_report_dir, "input_pair_3d.png")
    fig = syntx.render_input_pair_figure(fi, mi, output_path=out_png, title="3D Input Pair (Top/Bottom)")

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 2000
    assert fig is not None

