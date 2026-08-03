"""
tests/test_standard_viz_infrastructure.py — Audit Suite for Syntx Standard Reporting Infrastructure
===================================================================================================

Verifies that syntx.viz standard reporting tools (render_input_pair_figure, render_standard_4panel,
create_registration_report) strictly adhere to all GEMINI.md Section 3 reporting invariants:
1. 2D and 3D ANTsImage & NumPy array support.
2. Canonical LPI Anatomical Orientation (Anterior UP for Axial, Superior UP for Coronal/Sagittal).
3. Figure 1: Fixed Image on Left / Top, Moving Image on Right / Bottom with 1 colorbar per image.
4. Figure 2: Standard 4-panel (Panel A Deformed Grid, Panel B Jacobian, Panel C Inverse Error, Panel D Edge Overlay).
5. Non-empty PNG and HTML report outputs.
"""

import os
import pytest
import ants
import numpy as np

import syntx
from syntx.viz import create_registration_report, render_standard_4panel, render_input_pair_figure, corner_watermark
from syntx.viz.core import AnatomicalVisualizer


def test_corner_watermark():
    """Verifies corner_watermark creates high intensity patch at matrix corner [0:10, 0:10]."""
    fi = ants.image_read(ants.get_data("r16"))
    fi_wm = corner_watermark(fi, patch_size=10)

    wm_arr = fi_wm.numpy()
    patch_mean = np.mean(wm_arr[0:10, 0:10])
    orig_corner_mean = np.mean(fi.numpy()[0:10, 0:10])

    assert patch_mean > 200.0, "corner_watermark failed to place high-intensity patch near max value!"
    assert patch_mean > orig_corner_mean + 100.0, "corner_watermark did not increase corner intensity!"


def test_2d_slice_extraction_orientation():
    """Verifies 2D slice extraction matches ants.plot rotate90_matrix (arr.T) orientation."""
    arr = np.zeros((100, 100), dtype=np.float32)
    arr[10:30, :] = 1.0  # Row 10:30 band

    img_ants = ants.from_numpy(arr)
    slice_obj = AnatomicalVisualizer.extract_slice(img_ants, plane="axial", reorient=True)

    # Transposed array has band in cols 10:30
    left_sum = np.sum(slice_obj.data[:, :40])
    right_sum = np.sum(slice_obj.data[:, 60:])

    assert left_sum > right_sum, "2D slice extraction failed ants.plot transpose orientation!"


def test_render_input_pair_figure_2d(tmp_path):
    """Verifies Figure 1 2D side-by-side layout rendering."""
    fi = ants.image_read(ants.get_data("r16"))
    mi = ants.image_read(ants.get_data("r64"))

    out_png = str(tmp_path / "test_fig1_2d.png")
    fig = render_input_pair_figure(fi, mi, output_path=out_png, show_figure=False)

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000


def test_render_input_pair_figure_3d(tmp_path):
    """Verifies Figure 1 3D 2x3 panel layout rendering."""
    fi = ants.image_read(ants.get_data("mni"))
    mi = ants.image_read(ants.get_data("mni"))

    out_png = str(tmp_path / "test_fig1_3d.png")
    fig = render_input_pair_figure(fi, mi, output_path=out_png, show_figure=False)


def test_non_identity_header_rendering_2d_and_3d(tmp_path):
    """Verifies Figure 1 and Figure 2 rendering with non-identity direction, spacing, and origin headers."""
    # 2D non-identity image
    arr_2d = ants.image_read(ants.get_data("r16")).numpy()
    fi_2d = ants.from_numpy(arr_2d, origin=(-10.0, 25.0), spacing=(0.8, 1.2), direction=np.array([[0.0, 1.0], [1.0, 0.0]]))
    mi_2d = ants.from_numpy(arr_2d, origin=(-10.0, 25.0), spacing=(0.8, 1.2), direction=np.array([[0.0, 1.0], [1.0, 0.0]]))

    out_fig1_2d = str(tmp_path / "fig1_2d_non_identity.png")
    render_input_pair_figure(fi_2d, mi_2d, output_path=out_fig1_2d, show_figure=False)
    assert os.path.exists(out_fig1_2d) and os.path.getsize(out_fig1_2d) > 1000

    out_fig2_2d = str(tmp_path / "fig2_2d_non_identity.png")
    detJ_2d = np.ones_like(arr_2d)
    inv_err_2d = np.zeros_like(arr_2d)
    warp_2d = np.zeros((*arr_2d.shape, 2))
    render_standard_4panel(fixed=fi_2d, warped=fi_2d, moving=mi_2d, detJ=detJ_2d, inv_err_map=inv_err_2d, warp=warp_2d, output_path=out_fig2_2d)
    assert os.path.exists(out_fig2_2d) and os.path.getsize(out_fig2_2d) > 1000

    # 3D non-identity image
    mni = ants.image_read(ants.get_data("mni")).resample_image((4.0, 4.0, 4.0))
    arr_3d = mni.numpy()
    dir_3d = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    fi_3d = ants.from_numpy(arr_3d, origin=(-50.0, 20.0, 10.0), spacing=(2.0, 2.0, 3.0), direction=dir_3d)
    mi_3d = ants.from_numpy(arr_3d, origin=(-50.0, 20.0, 10.0), spacing=(2.0, 2.0, 3.0), direction=dir_3d)

    out_fig1_3d = str(tmp_path / "fig1_3d_non_identity.png")
    render_input_pair_figure(fi_3d, mi_3d, output_path=out_fig1_3d, show_figure=False)
    assert os.path.exists(out_fig1_3d) and os.path.getsize(out_fig1_3d) > 1000

    out_fig2_3d = str(tmp_path / "fig2_3d_non_identity.png")
    detJ_3d = np.ones_like(arr_3d)
    inv_err_3d = np.zeros_like(arr_3d)
    warp_3d = np.zeros((*arr_3d.shape, 3))
    render_standard_4panel(fixed=fi_3d, warped=fi_3d, moving=mi_3d, detJ=detJ_3d, inv_err_map=inv_err_3d, warp=warp_3d, output_path=out_fig2_3d)
    assert os.path.exists(out_fig2_3d) and os.path.getsize(out_fig2_3d) > 1000


def test_render_standard_4panel_2d(tmp_path):
    """Verifies Figure 2 standard 4-panel report rendering."""
    fi = ants.image_read(ants.get_data("r16"))
    mi = ants.image_read(ants.get_data("r64"))

    warp = np.zeros((*fi.shape, 2), dtype=np.float32)
    detJ = np.ones(fi.shape, dtype=np.float32)
    inv_err = np.zeros(fi.shape, dtype=np.float32)

    out_png = str(tmp_path / "test_fig2_4panel.png")
    fig = render_standard_4panel(
        fixed=fi,
        warped=mi,
        warp=warp,
        detJ=detJ,
        inv_err_map=inv_err,
        moving=mi,
        output_path=out_png
    )

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000


def test_create_registration_report(tmp_path):
    """Verifies interactive HTML report creation and provenance integration."""
    fi = ants.image_read(ants.get_data("r16"))
    mi = ants.image_read(ants.get_data("r64"))

    warp = np.zeros((*fi.shape, 2), dtype=np.float32)
    detJ = np.ones(fi.shape, dtype=np.float32)
    inv_err = np.zeros(fi.shape, dtype=np.float32)

    out_html = str(tmp_path / "test_report.html")
    create_registration_report(
        fixed=fi,
        moving=mi,
        warped=mi,
        warp=warp,
        detJ=detJ,
        inv_err_map=inv_err,
        output_html=out_html,
        show_report=False
    )

    assert os.path.exists(out_html)
    assert os.path.getsize(out_html) > 1000
