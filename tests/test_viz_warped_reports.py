"""CPU unit tests for physical-space grid alignment and warped-image reporting in syntx.viz."""

import os

import ants
import matplotlib.pyplot as plt
import numpy as np
import pytest

import syntx.viz as viz
from syntx.viz import (
    create_registration_report,
    create_visualization_gallery,
    plot_edge_overlay,
    render_checkerboard_figure,
    render_standard_4panel,
    render_warped_comparison_figure,
)


def _oblique_rpi_direction(angle_deg: float = 12.0) -> np.ndarray:
    """Return a valid 3x3 orthonormal oblique RPI direction cosine matrix."""
    theta = np.deg2rad(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rpi = np.diag([1.0, -1.0, 1.0])
    return rot_z @ rpi


def _physical_gaussian_blob(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: np.ndarray,
    center_lps: tuple[float, ...],
    sigma_mm: float = 5.0,
) -> ants.ANTsImage:
    """Evaluate an analytical physical-LPS Gaussian phantom on an arbitrary ANTsImage grid."""
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).astype(np.float64)
    m_i2p = np.asarray(direction, dtype=np.float64) @ np.diag(np.asarray(spacing, dtype=np.float64))
    coords_lps = idx @ m_i2p.T + np.asarray(origin, dtype=np.float64)
    diff = coords_lps - np.asarray(center_lps, dtype=np.float64)
    sq_dist = np.sum(diff * diff, axis=-1)
    arr = (100.0 * np.exp(-0.5 * sq_dist / (sigma_mm ** 2))).astype(np.float32)
    arr[arr < 1.0] = 0.0
    return ants.from_numpy(arr, origin=origin, spacing=spacing, direction=direction)


def test_create_registration_report_asymmetric_oblique_resamples_warped_and_embeds_figures(tmp_path):
    """(a) Asymmetric (18, 24, 30) oblique/RPI ANTsImage with mismatched warped origin/spacing."""
    dir_fix = _oblique_rpi_direction(14.0)
    center_lps = (4.0, -6.0, 18.0)

    fixed = _physical_gaussian_blob(
        shape=(18, 24, 30),
        spacing=(1.2, 1.0, 1.5),
        origin=(-6.0, 14.0, -4.0),
        direction=dir_fix,
        center_lps=center_lps,
        sigma_mm=5.5,
    )
    moving = _physical_gaussian_blob(
        shape=(20, 22, 26),
        spacing=(1.1, 1.3, 1.6),
        origin=(-8.0, 16.0, -2.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
        center_lps=(7.5, -2.0, 21.0),
        sigma_mm=5.5,
    )
    # Same physical-LPS anatomy as fixed, but stored on a mismatched grid/origin/spacing/direction
    warped_mismatched = _physical_gaussian_blob(
        shape=(22, 26, 24),
        spacing=(1.0, 1.1, 1.8),
        origin=(-5.0, -20.0, -3.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=center_lps,
        sigma_mm=5.5,
    )

    out_html = tmp_path / "asymmetric_report.html"
    rep = create_registration_report(
        fixed=fixed,
        moving=moving,
        warped=warped_mismatched,
        output_html=str(out_html),
        show_report=False,
    )

    assert os.path.exists(rep["html_path"])
    assert os.path.exists(rep["fig_path"]) and os.path.getsize(rep["fig_path"]) > 1000
    assert rep["fig_warped_path"] is not None and os.path.exists(rep["fig_warped_path"])
    assert os.path.getsize(rep["fig_warped_path"]) > 1000
    assert rep["fig_checkerboard_path"] is not None and os.path.exists(rep["fig_checkerboard_path"])
    assert os.path.getsize(rep["fig_checkerboard_path"]) > 1000

    # Physical-space alignment before image_compare yields high similarity despite grid mismatch
    assert rep["metrics"]["NCC"] > 0.95
    assert rep["metrics"]["SSIM"] > 0.90

    # Missing warp / inv_err_map stay NaN (not synthetic 0% folding or 0 mm error)
    assert np.isnan(rep["jacobian"]["folding_pct"])
    assert np.isnan(rep["inverse_error"]["max"])

    html_txt = out_html.read_text(encoding="utf-8")
    assert os.path.basename(rep["fig_warped_path"]) in html_txt
    assert os.path.basename(rep["fig_checkerboard_path"]) in html_txt

    # When warped is omitted entirely, fig_warped_path and fig_checkerboard_path are None
    out_no_warped = tmp_path / "no_warped_report.html"
    rep_none = create_registration_report(
        fixed=fixed,
        moving=moving,
        output_html=str(out_no_warped),
        show_report=False,
    )
    assert rep_none["fig_warped_path"] is None
    assert rep_none["fig_checkerboard_path"] is None


def test_render_checkerboard_figure_physical_grid_validation_and_resampling(tmp_path):
    """(b) Reject shifted origin / rotated direction when resample_to_target=False; resample when True."""
    center_lps = (6.0, 8.0, 10.0)
    img_a = _physical_gaussian_blob(
        shape=(20, 22, 18),
        spacing=(1.0, 1.0, 1.2),
        origin=(-4.0, -3.0, -2.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=center_lps,
    )
    img_shifted_origin = _physical_gaussian_blob(
        shape=(20, 22, 18),
        spacing=(1.0, 1.0, 1.2),
        origin=(5.0, -3.0, -2.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=center_lps,
    )
    img_rotated_dir = _physical_gaussian_blob(
        shape=(20, 22, 18),
        spacing=(1.0, 1.0, 1.2),
        origin=(-4.0, 19.0, -2.0),
        direction=_oblique_rpi_direction(18.0),
        center_lps=center_lps,
    )

    with pytest.raises(ValueError, match="origin"):
        render_checkerboard_figure(img_a, img_shifted_origin, title="Origin Mismatch")

    with pytest.raises(ValueError, match="direction"):
        render_checkerboard_figure(img_a, img_rotated_dir, title="Direction Mismatch")

    out_resampled = tmp_path / "checker_resampled.png"
    ret = render_checkerboard_figure(
        img_a,
        img_rotated_dir,
        title="Physically Resampled Checkerboard",
        save_path=str(out_resampled),
        theme="light",
        resample_to_target=True,
    )
    assert ret == str(out_resampled)
    assert out_resampled.exists() and out_resampled.stat().st_size > 1000

    # Also verify 2-D default view handling
    img_2d_a = ants.from_numpy(np.ones((24, 28), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.2))
    img_2d_b = ants.from_numpy(np.ones((20, 22), dtype=np.float32), origin=(1.0, -1.0), spacing=(1.2, 1.4))
    fig_2d = render_checkerboard_figure(img_2d_a, img_2d_b, theme="light", resample_to_target=True)
    assert len(fig_2d.axes) == 1
    plt.close(fig_2d)


def test_render_warped_comparison_figure_2d_and_3d_themes(tmp_path):
    """(c) render_warped_comparison_figure in 2-D and 3-D across light & dark themes."""
    assert hasattr(viz, "render_warped_comparison_figure")

    # 2-D mismatched physical grids (light & dark, with and without moving)
    y2, x2 = np.mgrid[0:28, 0:32]
    arr2_f = np.exp(-((x2 - 16.0) ** 2 + (y2 - 14.0) ** 2) / 36.0).astype(np.float32)
    arr2_f[arr2_f < 0.05] = 0.0
    fi_2d = ants.from_numpy(arr2_f, origin=(-5.0, 3.0), spacing=(1.0, 1.5))
    wa_2d = ants.from_numpy(arr2_f, origin=(-4.0, 2.0), spacing=(1.1, 1.4))
    mi_2d = ants.from_numpy(np.roll(arr2_f, 3, axis=0), origin=(0.0, 0.0), spacing=(1.0, 2.0))

    out_2d_light = tmp_path / "warped_2d_light.png"
    fig_2d_l = render_warped_comparison_figure(
        fi_2d, wa_2d, theme="light", output_path=str(out_2d_light)
    )
    assert out_2d_light.exists() and out_2d_light.stat().st_size > 1000
    img_axes_2d = [ax for ax in fig_2d_l.axes if ax.images]
    assert len(img_axes_2d) == 2
    assert img_axes_2d[0].images[0].get_array().shape == img_axes_2d[1].images[0].get_array().shape
    plt.close(fig_2d_l)

    out_2d_dark = tmp_path / "warped_2d_dark_moving.png"
    fig_2d_d = render_warped_comparison_figure(
        fi_2d, wa_2d, moving=mi_2d, theme="dark", output_path=str(out_2d_dark)
    )
    assert out_2d_dark.exists() and out_2d_dark.stat().st_size > 1000
    assert len([ax for ax in fig_2d_d.axes if ax.images]) == 3
    plt.close(fig_2d_d)

    # 3-D oblique/mismatched physical grids (light & dark)
    fi_3d = _physical_gaussian_blob(
        shape=(18, 22, 24),
        spacing=(1.0, 1.2, 1.5),
        origin=(-5.0, 12.0, -3.0),
        direction=_oblique_rpi_direction(10.0),
        center_lps=(3.0, -1.0, 14.0),
    )
    wa_3d = _physical_gaussian_blob(
        shape=(20, 24, 22),
        spacing=(1.1, 1.0, 1.6),
        origin=(-6.0, -14.0, -2.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(3.0, -1.0, 14.0),
    )
    mi_3d = _physical_gaussian_blob(
        shape=(16, 20, 18),
        spacing=(1.3, 1.3, 2.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(8.0, 8.0, 16.0),
    )

    for theme_name in ("light", "dark"):
        out_3d = tmp_path / f"warped_3d_{theme_name}.png"
        fig_3d = render_warped_comparison_figure(
            fi_3d, wa_3d, moving=mi_3d, theme=theme_name, output_path=str(out_3d)
        )
        assert out_3d.exists() and out_3d.stat().st_size > 1000
        img_axes_3d = [ax for ax in fig_3d.axes if ax.images]
        assert len(img_axes_3d) == 9  # 3 rows (Fixed, Moving, Warped) x 3 planes
        # Fixed (row 0: axes 0..2) and Warped (row 2: axes 6..8) share identical cropped shapes per plane
        for col in range(3):
            assert (
                img_axes_3d[col].images[0].get_array().shape
                == img_axes_3d[6 + col].images[0].get_array().shape
            )
        plt.close(fig_3d)


def test_render_standard_4panel_show_header_moving_before_and_warped_after(tmp_path):
    """(d) render_standard_4panel(show_header=True) renders Fixed, Moving (Before), and Warped (After)."""
    fi = _physical_gaussian_blob(
        shape=(18, 22, 24),
        spacing=(1.0, 1.0, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(9.0, 11.0, 22.0),
    )
    # Moving has distinct anisotropy & center so Moving (Before) differs from Warped (After)
    mi = _physical_gaussian_blob(
        shape=(16, 20, 14),
        spacing=(1.0, 2.5, 2.0),
        origin=(-2.0, -4.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(6.0, 18.0, 14.0),
    )
    warped = _physical_gaussian_blob(
        shape=(20, 24, 22),
        spacing=(0.9, 0.9, 1.6),
        origin=(0.5, 0.5, -1.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(9.0, 11.0, 22.0),
    )
    warp = ants.from_numpy(
        np.zeros((*fi.shape, 3), dtype=np.float32),
        origin=fi.origin,
        spacing=fi.spacing,
        direction=fi.direction,
        has_components=True,
    )
    detJ = fi.new_image_like(np.ones(fi.shape, dtype=np.float32))
    inv_err = fi.new_image_like(np.zeros(fi.shape, dtype=np.float32))

    out_with_mov = tmp_path / "4panel_hdr_moving.png"
    fig_mov = render_standard_4panel(
        fixed=fi,
        warped=warped,
        warp=warp,
        detJ=detJ,
        inv_err_map=inv_err,
        moving=mi,
        show_header=True,
        theme="light",
        output_path=str(out_with_mov),
    )
    titles_mov = [ax.get_title() for ax in fig_mov.axes if ax.get_title()]
    assert "Fixed Target Image" in titles_mov
    assert "Moving Image (Before)" in titles_mov
    assert "Warped Moving Image (After)" in titles_mov
    assert out_with_mov.exists() and out_with_mov.stat().st_size > 1000
    plt.close(fig_mov)

    out_no_mov = tmp_path / "4panel_hdr_no_moving.png"
    fig_no_mov = render_standard_4panel(
        fixed=fi,
        warped=warped,
        warp=warp,
        detJ=detJ,
        inv_err_map=inv_err,
        moving=None,
        show_header=True,
        theme="light",
        output_path=str(out_no_mov),
    )
    titles_no_mov = [ax.get_title() for ax in fig_no_mov.axes if ax.get_title()]
    assert "Fixed Target Image" in titles_no_mov
    assert "Warped Moving Image (Aligned to Fixed)" in titles_no_mov
    plt.close(fig_no_mov)

    # Also verify plot_edge_overlay aligns mismatched ANTsImage warped at fixed's resolved slice
    fig_edge = plot_edge_overlay(fi, warped, theme="light")
    assert fig_edge is not None
    plt.close(fig_edge)


def test_create_visualization_gallery_embeds_warped_without_warp_or_detj(tmp_path):
    """(e) create_visualization_gallery(fixed, moving, warped=warped) displays warped & checkerboard."""
    fi = _physical_gaussian_blob(
        shape=(16, 18, 20),
        spacing=(1.0, 1.0, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(8.0, 9.0, 15.0),
    )
    mi = _physical_gaussian_blob(
        shape=(16, 18, 20),
        spacing=(1.0, 1.0, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(10.0, 7.0, 15.0),
    )
    warped = _physical_gaussian_blob(
        shape=(18, 20, 18),
        spacing=(0.9, 0.9, 1.6),
        origin=(0.5, -0.5, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(8.0, 9.0, 15.0),
    )

    out_html = tmp_path / "gallery_warped_only.html"
    res_path = create_visualization_gallery(
        fixed=fi,
        moving=mi,
        warped=warped,
        output_path=str(out_html),
    )
    assert os.path.exists(res_path)
    html_txt = out_html.read_text(encoding="utf-8")
    assert "Warped Moving-to-Fixed Comparison" in html_txt
    assert "Fixed Warped Checkerboard" in html_txt
    # 2 input pair PNGs + 1 warped comparison PNG + 1 checkerboard PNG = 4 embedded base64 PNGs
    assert html_txt.count("data:image/png;base64,") == 4


def test_create_registration_report_interactive_light_dark_theme_toggle(tmp_path):
    """(f) create_registration_report embeds interactive Light/Dark toggle and renders both PNG pairs."""
    fi = _physical_gaussian_blob(
        shape=(16, 18, 20),
        spacing=(1.0, 1.0, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(8.0, 9.0, 15.0),
    )
    mi = _physical_gaussian_blob(
        shape=(16, 18, 20),
        spacing=(1.0, 1.0, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        center_lps=(9.5, 8.0, 15.0),
    )

    out_html = tmp_path / "theme_toggle_report.html"
    rep = create_registration_report(
        fixed=fi,
        moving=mi,
        warped=mi,
        output_html=str(out_html),
        theme="light",
        show_report=False,
    )
    html_txt = out_html.read_text(encoding="utf-8")
    assert 'id="theme-toggle"' in html_txt
    assert 'data-theme="light"' in html_txt
    assert "data-src-light=" in html_txt
    assert "data-src-dark=" in html_txt

    # Verify both _light.png and _dark.png exist on disk for every generated figure
    assert rep["fig_path"].endswith("_light.png")
    dark_counterpart = rep["fig_path"].replace("_light.png", "_dark.png")
    assert os.path.exists(rep["fig_path"]) and os.path.getsize(rep["fig_path"]) > 1000
    assert os.path.exists(dark_counterpart) and os.path.getsize(dark_counterpart) > 1000

