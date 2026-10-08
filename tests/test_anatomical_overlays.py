"""
Unit Tests for syntx.viz.anatomical_overlays (Anatomical Overlays & Flow Visualization Hierarchy).
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import ants
import matplotlib.pyplot as plt

import syntx.viz as viz
from syntx.viz.anatomical_overlays import (
    visualize_segmentation_on_anatomy,
    visualize_segmentation_pair_on_anatomy,
    visualize_flow_on_anatomy,
    visualize_flow_differential_on_anatomy,
    render_segmentation_alignment_series,
    render_gainer_anatomical_dissection,
    AnatomicalOverlayVisualizer,
)


@pytest.fixture
def dummy_data():
    shape = (32, 32, 32)
    fi_arr = np.zeros(shape, dtype=np.float32)
    fi_arr[8:24, 8:24, 8:24] = 1.0

    seg1_arr = np.zeros(shape, dtype=np.int32)
    seg1_arr[10:20, 10:20, 10:20] = 1  # Structure 1

    seg2_arr = np.zeros(shape, dtype=np.int32)
    seg2_arr[12:22, 12:22, 12:22] = 1  # Shifted structure

    fi = ants.from_numpy(fi_arr, spacing=(1.0, 1.0, 1.0))
    seg1 = ants.from_numpy(seg1_arr, spacing=(1.0, 1.0, 1.0))
    seg2 = ants.from_numpy(seg2_arr, spacing=(1.0, 1.0, 1.0))

    # Vector displacement fields: shape (32, 32, 32, 3)
    warp1_arr = np.zeros((*shape, 3), dtype=np.float32)
    warp1_arr[10:20, 10:20, 10:20, 0] = 2.0  # +2mm displacement along x

    warp2_arr = np.zeros((*shape, 3), dtype=np.float32)
    warp2_arr[10:20, 10:20, 10:20, 0] = 4.0  # +4mm displacement along x
    warp2_arr[10:20, 10:20, 10:20, 1] = 1.5

    warp1 = ants.from_numpy(warp1_arr, spacing=(1.0, 1.0, 1.0), has_components=True)
    warp2 = ants.from_numpy(warp2_arr, spacing=(1.0, 1.0, 1.0), has_components=True)

    return {
        "fi": fi, "seg1": seg1, "seg2": seg2,
        "warp1": warp1, "warp2": warp2
    }


def test_overlay_exports():
    assert hasattr(viz, "visualize_segmentation_on_anatomy")
    assert hasattr(viz, "visualize_segmentation_pair_on_anatomy")
    assert hasattr(viz, "visualize_flow_on_anatomy")
    assert hasattr(viz, "visualize_flow_differential_on_anatomy")
    assert hasattr(viz, "render_segmentation_alignment_series")
    assert hasattr(viz, "render_gainer_anatomical_dissection")
    assert hasattr(viz, "AnatomicalOverlayVisualizer")


def test_visualize_segmentation_on_anatomy(dummy_data):
    fig, ax = plt.subplots()
    res_ax = visualize_segmentation_on_anatomy(
        anatomy=dummy_data["fi"],
        segmentation=dummy_data["seg1"],
        slice_axis=2,
        slice_idx=16,
        target_contour=dummy_data["seg2"],
        badge_text="Dice: 0.75",
        title="Test Overlay",
        ax=ax
    )
    assert res_ax is ax
    plt.close(fig)


def test_visualize_segmentation_pair_on_anatomy(dummy_data):
    fig, ax = plt.subplots()
    res_ax = visualize_segmentation_pair_on_anatomy(
        anatomy=dummy_data["fi"],
        target_seg=dummy_data["seg1"],
        moving_seg=dummy_data["seg2"],
        slice_axis=2,
        slice_idx=16,
        mode="confusion",
        ax=ax
    )
    assert res_ax is ax
    plt.close(fig)


def test_visualize_flow_on_anatomy(dummy_data):
    # Test quiver_magnitude
    fig, ax = plt.subplots()
    res_ax, stats = visualize_flow_on_anatomy(
        anatomy=dummy_data["fi"],
        flow=dummy_data["warp1"],
        slice_axis=2,
        slice_idx=16,
        mode="quiver_magnitude",
        ax=ax
    )
    assert res_ax is ax
    assert "max_displacement_mm" in stats
    plt.close(fig)

    # Test mesh
    fig, ax = plt.subplots()
    res_ax, _ = visualize_flow_on_anatomy(
        anatomy=dummy_data["fi"],
        flow=dummy_data["warp1"],
        slice_axis=2,
        slice_idx=16,
        mode="mesh",
        ax=ax
    )
    assert res_ax is ax
    plt.close(fig)


def test_visualize_flow_differential_on_anatomy(dummy_data):
    fig, ax = plt.subplots()
    res_ax, stats = visualize_flow_differential_on_anatomy(
        anatomy=dummy_data["fi"],
        flow1=dummy_data["warp1"],
        flow2=dummy_data["warp2"],
        slice_axis=2,
        slice_idx=16,
        ax=ax
    )
    assert res_ax is ax
    assert "max_diff_mm" in stats
    assert stats["max_diff_mm"] > 0
    plt.close(fig)


def test_render_gainer_anatomical_dissection(dummy_data, tmp_path):
    models_config = {
        "affine": {"image": dummy_data["seg2"], "badge": "Affine 0.30", "color": "#94a3b8", "edge": "#475569"},
        "sobolev": {"image": dummy_data["seg2"], "badge": "Sobolev 0.55", "color": "#0284c7", "edge": "#0369a1"},
        "continuum": {"image": dummy_data["seg1"], "badge": "Hyper 0.65", "color": "#059669", "edge": "#047857"},
    }
    out_file = str(tmp_path / "test_dissection.png")
    fig = render_gainer_anatomical_dissection(
        anatomy=dummy_data["fi"],
        target_label=dummy_data["seg1"],
        models_config=models_config,
        flow_sobolev=dummy_data["warp1"],
        flow_continuum=dummy_data["warp2"],
        slice_axis=2,
        slice_idx=16,
        structure_name="Test Structure",
        gain_pct=10.0,
        output_filename=out_file
    )
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000
    plt.close("all")


def test_anatomical_overlay_visualizer_class(dummy_data):
    viz_obj = AnatomicalOverlayVisualizer(dummy_data["fi"], slice_axis=2, slice_idx=16)
    viz_obj.set_roi(4, 28, 4, 28)

    fig, ax = plt.subplots()
    viz_obj.plot_segmentation(ax, dummy_data["seg1"], badge_text="Class Test")
    plt.close(fig)

    fig, ax = plt.subplots()
    viz_obj.plot_flow(ax, dummy_data["warp1"], mode="quiver")
    plt.close(fig)
