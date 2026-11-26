"""Unit tests for unified syntx.viz.render_registration_qc (6-panel default,
4/8-panel + custom panels) and content-span striatal axial slice selection with
physical LPS mapping.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import ants

import syntx
import syntx.viz as viz
from syntx.core.inverse import _LazyKeyDict
from syntx.spatial import vox_to_physical
from syntx.viz import (
    STRIATAL_AXIAL_CONTENT_FRACTION,
    LpsSliceResolution,
    foreground_content_span,
    select_content_slice_index,
    resolve_lps_slice_pair,
    render_registration_qc,
)


def _oblique_direction(angle_deg: float = 15.0) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    return np.diag([-1.0, -1.0, 1.0]) @ rot_x


def _physical_gaussian_blob(
    shape=(28, 32, 36),
    spacing=(1.0, 1.0, 1.5),
    origin=(0.0, 0.0, 0.0),
    direction=None,
    center_lps=(14.0, 16.0, 27.0),
    sigma_mm=4.5,
) -> ants.ANTsImage:
    if direction is None:
        direction = np.diag([-1.0, -1.0, 1.0])
    img = ants.from_numpy(
        np.zeros(shape, dtype=np.float32),
        origin=tuple(float(x) for x in origin),
        spacing=tuple(float(x) for x in spacing),
        direction=np.asarray(direction, dtype=np.float64),
    )
    ijk = np.argwhere(np.ones(shape, dtype=bool))
    pts = vox_to_physical(img, ijk)
    c = np.asarray(center_lps, dtype=np.float32)
    dist2 = np.sum((pts - c[None, :]) ** 2, axis=-1)
    vals = (100.0 * np.exp(-0.5 * dist2 / (sigma_mm ** 2))).astype(np.float32).reshape(shape)
    return img.new_image_like(vals)


def _make_striatal_phantom(shape=(24, 24, 120), z_start=20, z_end=100, bg_val=0.0) -> np.ndarray:
    """Create 3D phantom with brain span [z_start, z_end) and distinct striatal vs cortical rings."""
    arr = np.full(shape, bg_val, dtype=np.float32)
    nx, ny, _ = shape
    cx, cy = nx // 2, ny // 2
    yy, xx = np.ogrid[:nx, :ny]
    brain_r2 = (xx - cy) ** 2 + (yy - cx) ** 2 <= 8 ** 2
    for z in range(z_start, z_end):
        arr[:, :, z][brain_r2] = bg_val + 50.0 if bg_val < 0 else 50.0
    span_len = (z_end - 1) - z_start
    z_striatum = z_start + round(STRIATAL_AXIAL_CONTENT_FRACTION * span_len)
    z_cortex = z_start + round(0.75 * span_len)
    stria_r2 = (xx - cy) ** 2 + (yy - cx) ** 2 <= 3 ** 2
    for z in range(max(z_start, z_striatum - 1), min(z_end, z_striatum + 2)):
        arr[:, :, z][stria_r2] = bg_val + 200.0 if bg_val < 0 else 200.0
    cort_r2 = (xx - cy) ** 2 + (yy - cx) ** 2 <= 2 ** 2
    for z in range(max(z_start, z_cortex - 1), min(z_end, z_cortex + 2)):
        arr[:, :, z][cort_r2] = bg_val + 120.0 if bg_val < 0 else 120.0
    return arr


def _make_3d_pair(shape=(24, 28, 32), spacing=(1.0, 1.0, 1.5)):
    fi = _physical_gaussian_blob(
        shape=shape,
        spacing=spacing,
        origin=(12.0, 14.0, -20.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
        center_lps=(0.0, 0.0, 4.0),
    )
    mi = _physical_gaussian_blob(
        shape=shape,
        spacing=spacing,
        origin=(12.0, 14.0, -20.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
        center_lps=(-2.0, 2.0, 4.0),
    )
    wa = _physical_gaussian_blob(
        shape=shape,
        spacing=spacing,
        origin=(12.0, 14.0, -20.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
        center_lps=(0.0, 0.0, 4.0),
    )
    warp_arr = np.zeros((*shape, 3), dtype=np.float32)
    warp_arr[..., 0] = 0.4
    warp = ants.from_numpy(
        warp_arr,
        origin=fi.origin,
        spacing=fi.spacing,
        direction=fi.direction,
        has_components=True,
    )
    detj = fi.new_image_like(np.full(shape, 1.05, dtype=np.float32))
    err = fi.new_image_like(np.linspace(0.0, 1.6, int(np.prod(shape)), dtype=np.float32).reshape(shape))
    return fi, mi, wa, warp, detj, err


def test_exports_and_all():
    assert hasattr(syntx, "render_registration_qc")
    assert hasattr(viz, "render_registration_qc")
    assert syntx.render_registration_qc is viz.render_registration_qc
    assert "render_registration_qc" in syntx.__all__
    assert "render_registration_qc" in viz.__all__
    for suffix in ("4panel", "qc", "6panel", "8panel"):
        removed = f"render_standard_{suffix}"
        assert not hasattr(syntx, removed)
        assert not hasattr(viz, removed)
        assert removed not in syntx.__all__
        assert removed not in viz.__all__
    for sym in (
        "STRIATAL_AXIAL_CONTENT_FRACTION",
        "LpsSliceResolution",
        "foreground_content_span",
        "select_content_slice_index",
        "resolve_lps_slice_pair",
    ):
        assert hasattr(viz, sym)
        assert sym in viz.__all__


def test_render_registration_qc_6panel_default_light_theme_layout_and_lazy_ice(tmp_path):
    fi, mi, wa, warp, detj, err = _make_3d_pair()
    lazy_calls = {"count": 0}

    def _loader():
        lazy_calls["count"] += 1
        return err

    reg = {
        "warpedmovout": wa,
        "fwdtransforms": [],
        "disp_tensor_zyx": torch.zeros((1, 3, *reversed(fi.shape)), dtype=torch.float32),
        "inverse_identity_errors": {
            "composed": _LazyKeyDict(
                {"max_error": 1.25, "mean_error": 0.45},
                lazy_loaders={"error_map": _loader},
            )
        },
    }

    out_png = tmp_path / "nested" / "qc_6panel.png"
    # Default n_panels=6 without passing n_panels explicitly
    fig = render_registration_qc(
        fixed=fi,
        warped=reg,
        detJ=detj,
        moving=mi,
        output_path=str(out_png),
    )
    assert lazy_calls["count"] == 0, "6-panel must not materialize lazy ICE error_map"
    assert out_png.exists() and out_png.stat().st_size > 1000
    assert mcolors.to_hex(fig.get_facecolor()) == "#ffffff"

    img_axes = [ax for ax in fig.axes if ax.images]
    assert len(img_axes) == 6
    for ax in img_axes:
        assert mcolors.to_hex(ax.get_facecolor()) == "#ffffff"

    # Sort visually top-to-bottom (-y0), left-to-right (x0)
    ordered = sorted(
        img_axes,
        key=lambda ax: (-round(ax.get_position().y0, 2), round(ax.get_position().x0, 2)),
    )
    # Top row shares higher y0 than bottom row; each row has 3 ascending x0 columns
    assert ordered[0].get_position().y0 > ordered[3].get_position().y0 + 0.1
    assert ordered[0].get_position().x0 < ordered[1].get_position().x0 < ordered[2].get_position().x0
    assert ordered[3].get_position().x0 < ordered[4].get_position().x0 < ordered[5].get_position().x0
    titles = [ax.get_title() for ax in ordered]
    assert titles[0].startswith("Fixed")
    assert "Fixed with Deformation Grid" in titles[1]
    assert "Jacobian (det(J))" in titles[2]
    assert titles[3].startswith("Moving")
    assert "Moving Deformed to Fixed (Warped)" in titles[4]
    assert titles[5].startswith("Edge Overlay")
    plt.close(fig)

    # Dark theme switch and moving=None
    fig_dark = render_registration_qc(fixed=fi, warped=wa, warp=warp, detJ=detj, moving=None, n_panels=6, theme="dark")
    assert mcolors.to_hex(fig_dark.get_facecolor()) != "#ffffff"
    assert len([ax for ax in fig_dark.axes if ax.images]) == 6
    assert any("Moving (Unavailable)" in ax.get_title() for ax in fig_dark.axes if ax.images)
    plt.close(fig_dark)


def test_render_registration_qc_6panel_2d_and_numpy_torch_inputs(tmp_path):
    arr_f = np.zeros((32, 32), dtype=np.float32)
    arr_f[8:24, 8:24] = 1.0
    arr_w = np.zeros((32, 32), dtype=np.float32)
    arr_w[9:25, 9:25] = 1.0
    warp_2d = np.zeros((32, 32, 2), dtype=np.float32)
    detj_2d = np.ones((32, 32), dtype=np.float32)

    out_2d = tmp_path / "qc6_2d.png"
    fig_2d = render_registration_qc(
        fixed=arr_f,
        warped=torch.from_numpy(arr_w),
        warp=warp_2d,
        detJ=detj_2d,
        moving=arr_f,
        n_panels=6,
        filename=str(out_2d),
    )
    assert out_2d.exists() and out_2d.stat().st_size > 1000
    assert len([ax for ax in fig_2d.axes if ax.images]) == 6
    plt.close(fig_2d)

    # 3D torch / numpy without detJ or warp -> renders N/A without crashing or fabricating 1.0
    vol = np.zeros((16, 18, 20), dtype=np.float32)
    vol[4:12, 4:14, 4:16] = 1.0
    fig_3d_na = render_registration_qc(fixed=torch.from_numpy(vol), warped=vol, moving=vol, n_panels=6)
    img_axes = [ax for ax in fig_3d_na.axes if ax.images]
    assert len(img_axes) == 6
    assert any("Unavailable (N/A)" in ax.get_title() for ax in img_axes)
    plt.close(fig_3d_na)


def test_render_registration_qc_8panel_ice_composed_lazy_p95_and_checkerboard(tmp_path):
    fi, mi, wa, warp, detj, _ = _make_3d_pair()
    # Create known 3D ICE volume where p95 over positive voxels differs cleanly from 0.0 and slice p95
    err_arr = np.zeros(fi.shape, dtype=np.float32)
    err_arr[:, :, 4:28] = 0.25
    err_arr[:, :, 26] = 2.40  # top positive slice (>5% of positive voxels -> ~2.40 p95)
    err_img = fi.new_image_like(err_arr)
    pos_vals = err_arr[err_arr > 0]
    expected_p95 = float(np.percentile(pos_vals, 95))

    lazy_calls = {"count": 0}

    def _loader():
        lazy_calls["count"] += 1
        return err_img

    reg = {
        "warpedmovout": wa,
        "fwdtransforms": [warp],
        "inverse_identity_errors": {
            "composed": _LazyKeyDict(
                {"max_error": 2.40, "mean_error": float(np.mean(pos_vals))},
                lazy_loaders={"error_map": _loader},
            )
        },
    }

    out_png = tmp_path / "qc_8panel.png"
    fig = render_registration_qc(
        fixed=fi,
        warped=reg,
        detJ=detj,
        moving=mi,
        n_panels=8,
        slice_idx=10,
        output_path=str(out_png),
    )
    assert lazy_calls["count"] == 1
    assert out_png.exists() and out_png.stat().st_size > 1000
    assert mcolors.to_hex(fig.get_facecolor()) == "#ffffff"

    img_axes = [ax for ax in fig.axes if ax.images]
    assert len(img_axes) == 8
    ordered = sorted(
        img_axes,
        key=lambda ax: (-round(ax.get_position().y0, 2), round(ax.get_position().x0, 2)),
    )
    assert ordered[0].get_position().y0 > ordered[4].get_position().y0 + 0.1
    assert ordered[0].get_position().x0 < ordered[1].get_position().x0 < ordered[2].get_position().x0 < ordered[3].get_position().x0
    assert ordered[4].get_position().x0 < ordered[5].get_position().x0 < ordered[6].get_position().x0 < ordered[7].get_position().x0
    titles = [ax.get_title() for ax in ordered]
    assert titles[0].startswith("Fixed")
    assert "Fixed with Deformation Grid" in titles[1]
    assert "Jacobian (det(J))" in titles[2]
    assert "Checkerboard (Fixed / Warped)" in titles[3]
    assert titles[4].startswith("Moving")
    assert "Moving Deformed to Fixed (Warped)" in titles[5]
    assert titles[6].startswith("Edge Overlay")
    ice_title = titles[7]
    assert "Inverse Identity Error (ICE, mm)" in ice_title
    assert "2.4000mm" in ice_title
    assert f"p95: {expected_p95:.4f}mm" in ice_title
    plt.close(fig)

    # Also test phi_2 fallback and missing inv_err_map rendering N/A (never synthetic 0.0mm)
    reg_phi2 = {
        "warpedmovout": wa,
        "fwdtransforms": [warp],
        "inverse_identity_errors": {
            "phi_2": {"error_map": err_img, "max_error": 2.40, "mean_error": 0.35, "p95_error": 1.95}
        },
    }
    fig_phi2 = render_registration_qc(fixed=fi, warped=reg_phi2, detJ=detj, moving=mi, n_panels=8)
    assert any("1.9500mm" in ax.get_title() for ax in fig_phi2.axes if ax.images)
    plt.close(fig_phi2)

    fig_no_ice = render_registration_qc(fixed=fi, warped=wa, warp=warp, detJ=detj, moving=mi, n_panels=8)
    ice_axes = [ax for ax in fig_no_ice.axes if ax.images and "Inverse Identity Error" in ax.get_title()]
    assert len(ice_axes) == 1
    assert "Unavailable (N/A)" in ice_axes[0].get_title()
    assert "0.0000mm" not in ice_axes[0].get_title()
    plt.close(fig_no_ice)

    # Verify checkerboard tile alternation on 2D horizontal vs vertical stripe patterns
    h_stripes = np.zeros((32, 32), dtype=np.float32)
    h_stripes[::2, :] = 1.0
    v_stripes = np.zeros((32, 32), dtype=np.float32)
    v_stripes[:, ::2] = 1.0
    fig_cb = render_registration_qc(
        fixed=h_stripes,
        warped=v_stripes,
        panels=["checkerboard"],
        checkerboard_pattern_size=8,
    )
    cb_ax = [ax for ax in fig_cb.axes if ax.images][0]
    cb_img = np.asarray(cb_ax.images[0].get_array())
    # Tile (0, 0) uses fixed (horizontal stripes); tile (0, 1) uses warped (vertical stripes)
    assert np.allclose(cb_img[0:8, 0:8], h_stripes[0:8, 0:8])
    assert np.allclose(cb_img[0:8, 8:16], v_stripes[0:8, 8:16])
    plt.close(fig_cb)


def test_render_registration_qc_4panel_preset_compatibility():
    fi, mi, wa, warp, detj, err = _make_3d_pair()
    with pytest.raises(ValueError, match="inv_err_map"):
        render_registration_qc(fixed=fi, warped=wa, warp=warp, detJ=detj, n_panels=4)

    reg_composed = {
        "warpedmovout": wa,
        "fwdtransforms": [warp],
        "inverse_identity_errors": {
            "composed": {"error_map": err, "max_error": 1.60, "mean_error": 0.50, "p95_error": 1.44}
        },
    }
    fig_4 = render_registration_qc(
        fixed=fi, warped=reg_composed, detJ=detj, moving=mi, n_panels=4, show_header=True
    )
    assert mcolors.to_hex(fig_4.get_facecolor()) == "#0d1117"
    titles = [ax.get_title() for ax in fig_4.axes if ax.get_title()]
    assert "Fixed Target Image" in titles
    assert "Moving Image (Before)" in titles
    assert "Warped Moving Image (After)" in titles
    assert any("1.4400mm" in t for t in titles)
    plt.close(fig_4)


def test_render_registration_qc_custom_panels_and_validation():
    fi, mi, wa, warp, detj, err = _make_3d_pair()
    lazy_calls = {"count": 0}

    def _loader():
        lazy_calls["count"] += 1
        return err

    reg = {
        "warpedmovout": wa,
        "fwdtransforms": [warp],
        "inverse_identity_errors": {
            "composed": _LazyKeyDict({"max_error": 1.0, "mean_error": 0.3}, lazy_loaders={"error_map": _loader})
        },
    }

    fig_1d = render_registration_qc(fixed=fi, warped=reg, panels=("fixed", "warped", "checkerboard"))
    assert lazy_calls["count"] == 0
    assert mcolors.to_hex(fig_1d.get_facecolor()) == "#ffffff"
    assert len([ax for ax in fig_1d.axes if ax.images]) == 3
    plt.close(fig_1d)

    fig_2d_grid = render_registration_qc(
        fixed=fi,
        warped=reg,
        panels=[["fixed", "moving"], ["warped", "ice"]],
        moving=mi,
    )
    assert lazy_calls["count"] == 1
    assert len([ax for ax in fig_2d_grid.axes if ax.images]) == 4
    plt.close(fig_2d_grid)

    with pytest.raises(ValueError, match="n_panels"):
        render_registration_qc(fixed=fi, warped=wa, n_panels=5)
    with pytest.raises(ValueError, match="Unknown panel key"):
        render_registration_qc(fixed=fi, warped=wa, panels=["fixed", "bogus_panel"])
    with pytest.raises(ValueError, match="empty"):
        render_registration_qc(fixed=fi, warped=wa, panels=[])


def test_striatal_axial_slice_padding_cropping_and_background_invariance():
    base_arr = _make_striatal_phantom(shape=(24, 24, 120), z_start=20, z_end=100, bg_val=0.0)
    inf_pad_arr = _make_striatal_phantom(shape=(24, 24, 180), z_start=80, z_end=160, bg_val=0.0)
    sup_pad_arr = _make_striatal_phantom(shape=(24, 24, 180), z_start=20, z_end=100, bg_val=0.0)
    crop_arr = _make_striatal_phantom(shape=(24, 24, 80), z_start=0, z_end=80, bg_val=0.0)

    # Physical origins match the brain start so physical z_lps of the brain is identical across all 4 FOVs
    variants = [
        ants.from_numpy(base_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=np.diag([-1.0, -1.0, 1.0])),
        ants.from_numpy(inf_pad_arr, origin=(0.0, 0.0, -60.0), spacing=(1.0, 1.0, 1.0), direction=np.diag([-1.0, -1.0, 1.0])),
        ants.from_numpy(sup_pad_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=np.diag([-1.0, -1.0, 1.0])),
        ants.from_numpy(crop_arr, origin=(0.0, 0.0, 20.0), spacing=(1.0, 1.0, 1.0), direction=np.diag([-1.0, -1.0, 1.0])),
    ]

    lps_z_values = []
    for img in variants:
        res = resolve_lps_slice_pair(img, plane="axial")
        assert isinstance(res, LpsSliceResolution)
        z_idx = res.fixed_slice_idx
        sl = img.numpy()[:, :, z_idx]
        assert float(np.max(sl)) >= 195.0, f"Selected axial slice {z_idx} missed striatal ring"
        lps_z_values.append(res.lps_coord_mm)

    assert max(lps_z_values) - min(lps_z_values) <= 1.0

    # CT-like negative background (-1000 HU) and z-scored MRI (-1.5 background)
    ct_arr = _make_striatal_phantom(shape=(24, 24, 140), z_start=30, z_end=110, bg_val=-1000.0)
    ct_span = foreground_content_span(ct_arr, axis=2)
    assert ct_span == (30, 109, True)
    z_ct = select_content_slice_index(ct_arr, plane="axial")
    assert float(np.max(ct_arr[:, :, z_ct])) >= 195.0 - 1000.0

    zscore_arr = _make_striatal_phantom(shape=(24, 24, 140), z_start=30, z_end=110, bg_val=-1.5)
    assert foreground_content_span(zscore_arr, axis=2) == (30, 109, True)

    # Sagittal and coronal center at 0.50 of foreground span
    assert select_content_slice_index(base_arr, plane="sagittal") == 12
    assert select_content_slice_index(base_arr, plane="coronal") == 12

    # Optional on-disk blast24 + CIT168 striatal calibration check
    tpl_path = "/Users/stnava/data/blast_cohorts/tbm_blast24_gpu/template/blast24_template_t1_brain.nii.gz"
    cit_path = "/Users/stnava/data/blast_cohorts/tbm_blast24_gpu/template/blast24_template_cit168.nii.gz"
    if os.path.exists(tpl_path) and os.path.exists(cit_path):
        tpl = ants.image_read(tpl_path)
        cit = ants.image_read(cit_path)
        z_tpl = select_content_slice_index(tpl.numpy(), plane="axial")
        cit_sl = cit.numpy()[:, :, z_tpl]
        # Putamen (1, 17) and Caudate (2, 18) must both be strongly visible
        assert np.sum((cit_sl == 1) | (cit_sl == 17)) > 200
        assert np.sum((cit_sl == 2) | (cit_sl == 18)) > 100


def test_lps_slice_pair_oblique_mismatched_and_fallbacks():
    # Fixed: oblique 15 deg, anisotropic, shifted origin
    fi_dir = _oblique_direction(15.0)
    fi_probe = _physical_gaussian_blob(
        shape=(28, 32, 36),
        spacing=(1.2, 1.0, 1.5),
        origin=(30.0, 45.0, -25.0),
        direction=fi_dir,
        center_lps=(15.0, 30.0, 5.0),
    )
    res_self = resolve_lps_slice_pair(fi_probe, plane="axial")
    target_lps_z = res_self.lps_coord_mm
    assert target_lps_z is not None

    # Moving: different oblique angle, anisotropic spacing, and origin such that nz // 2 is far from target_lps_z
    mi_dir = _oblique_direction(-10.0)
    # Place physical Gaussian landmark at fixed's resolved LPS point
    raw_c_vox = np.array([14.0, 16.0, float(res_self.fixed_slice_idx)], dtype=np.float64)
    target_pt = vox_to_physical(fi_probe, raw_c_vox)[0]
    fi = _physical_gaussian_blob(
        shape=(28, 32, 36),
        spacing=(1.2, 1.0, 1.5),
        origin=(30.0, 45.0, -25.0),
        direction=fi_dir,
        center_lps=tuple(target_pt),
        sigma_mm=3.0,
    )
    mi = _physical_gaussian_blob(
        shape=(30, 28, 44),
        spacing=(1.0, 1.4, 2.0),
        origin=(28.0, 46.0, -10.0),
        direction=mi_dir,
        center_lps=tuple(target_pt),
        sigma_mm=3.0,
    )

    res = resolve_lps_slice_pair(fi, moving=mi, plane="axial")
    assert res.moving_mapped_from_lps is True
    assert res.moving_slice_idx != mi.shape[2] // 2

    fig = render_registration_qc(fixed=fi, warped=fi, moving=mi, n_panels=6)
    img_axes = [ax for ax in fig.axes if ax.images]
    fixed_disp = img_axes[0].images[0].get_array()
    moving_disp = img_axes[3].images[0].get_array()
    assert float(np.max(fixed_disp)) >= 60.0
    assert float(np.max(moving_disp)) >= 60.0
    plt.close(fig)

    # Disjoint +500 mm LPS fallback
    mi_disjoint = _physical_gaussian_blob(
        shape=(24, 24, 24),
        spacing=(1.0, 1.0, 1.0),
        origin=(15.0, 30.0, 500.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
        center_lps=(5.0, 20.0, 512.0),
    )
    res_disjoint = resolve_lps_slice_pair(fi, moving=mi_disjoint, plane="axial")
    assert res_disjoint.moving_mapped_from_lps is False
    assert 0 <= res_disjoint.moving_slice_idx < mi_disjoint.shape[2]

    # Out-of-foreground scanner-origin fallback:
    # fi's target_pt[2] falls INSIDE mi_oof's voxel domain [0, 200] (e.g. near z=180),
    # while mi_oof's brain foreground sits at z in [20, 90].
    mi_oof_arr = _make_striatal_phantom(shape=(24, 24, 200), z_start=20, z_end=90)
    mi_oof = ants.from_numpy(
        mi_oof_arr,
        origin=(0.0, 0.0, float(target_pt[2] - 160.0)),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )
    res_oof = resolve_lps_slice_pair(fi, moving=mi_oof, plane="axial")
    assert res_oof.moving_mapped_from_lps is False
    assert 20 <= res_oof.moving_slice_idx < 90

    # Gridless NumPy array moving fallback
    mi_np = _make_striatal_phantom(shape=(24, 24, 50), z_start=10, z_end=40)
    res_np = resolve_lps_slice_pair(fi, moving=mi_np, plane="axial")
    assert res_np.moving_mapped_from_lps is False

    # 2D image fallback
    fi_2d = ants.from_numpy(np.ones((20, 24), dtype=np.float32))
    mi_2d = ants.from_numpy(np.ones((18, 22), dtype=np.float32))
    res_2d = resolve_lps_slice_pair(fi_2d, moving=mi_2d, plane="axial")
    assert res_2d == LpsSliceResolution(0, 0, None, False)


def test_regression_same_grid_shifted_fg_and_matching_shape_gridless_numpy():
    """Defect 1: same-grid ANTsImage & matching-shape gridless NumPy must not render blank air."""
    fi = ants.from_numpy(
        np.zeros((24, 24, 80), dtype=np.float32),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )
    arr_f = fi.numpy().copy()
    arr_f[6:18, 6:18, 10:30] = 100.0
    fi = fi.new_image_like(arr_f)

    mi_np = np.zeros((80, 24, 24), dtype=np.float32)
    mi_np[55:75, 6:18, 6:18] = 100.0
    mi_ants = fi.new_image_like(np.transpose(mi_np, (2, 1, 0)))

    r_np = resolve_lps_slice_pair(fi, moving=mi_np, plane="axial")
    assert 55 <= r_np.moving_slice_idx <= 74
    assert r_np.moving_mapped_from_lps is False

    r_sg = resolve_lps_slice_pair(fi, moving=mi_ants, plane="axial")
    assert 55 <= r_sg.moving_slice_idx <= 74
    assert r_sg.moving_mapped_from_lps is False

    for mov in (mi_np, mi_ants):
        fig = render_registration_qc(fixed=fi, warped=fi, moving=mov, n_panels=6)
        img_axes = [ax for ax in fig.axes if ax.images]
        mov_ax = [ax for ax in img_axes if ax.get_title().startswith("Moving (")][0]
        assert "LPS" not in mov_ax.get_title()
        assert float(np.max(mov_ax.images[0].get_array())) > 50.0
        plt.close(fig)


def test_regression_reorient_false_orthogonal_planes_fallback():
    """Defect 2: reorient=False must not pair orthogonal anatomical planes (PIR vs LPI) under LPS."""
    pir_arr = np.zeros((30, 40, 50), dtype=np.float32)
    pir_arr[5:25, 10:30, 15:35] = 100.0
    # PIR direction: axis 0 = P (+y), axis 1 = I (-z), axis 2 = R (-x)
    pir_dir = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    pir_img = ants.from_numpy(pir_arr, origin=(10.0, -20.0, 30.0), spacing=(1.0, 2.0, 3.0), direction=pir_dir)

    lpi_arr = np.zeros((40, 40, 40), dtype=np.float32)
    lpi_arr[10:30, 10:30, 22:36] = 100.0
    lpi_img = ants.from_numpy(
        lpi_arr,
        origin=(0.0, 0.0, -80.0),
        spacing=(1.5, 1.5, 1.5),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )

    res = resolve_lps_slice_pair(pir_img, moving=lpi_img, plane="axial", reorient=False)
    assert res.lps_coord_mm is None
    assert res.moving_mapped_from_lps is False
    assert 22 <= res.moving_slice_idx <= 35


def test_regression_mismatched_gridless_numpy_moving_and_warped_none(tmp_path):
    """Defect 3 & Defect 6: mismatched gridless NumPy moving keeps content-span slice & no LPS;
    warped=None and str fixed path work with registration_result.
    """
    fi = ants.from_numpy(
        np.zeros((24, 24, 40), dtype=np.float32),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )
    arr_f = fi.numpy().copy()
    arr_f[6:18, 6:18, 5:36] = 100.0
    fi = fi.new_image_like(arr_f)

    mi_np = np.zeros((60, 24, 24), dtype=np.float32)
    mi_np[15:59, 6:18, 6:18] = 100.0

    expected = resolve_lps_slice_pair(fi, moving=mi_np, plane="axial")
    assert expected.moving_slice_idx == 38
    assert expected.moving_mapped_from_lps is False

    fig = render_registration_qc(fi, fi, moving=mi_np, n_panels=6)
    mov_titles = [ax.get_title() for ax in fig.axes if ax.images and ax.get_title().startswith("Moving (")]
    assert mov_titles == ["Moving (Z=38)"]
    plt.close(fig)

    # Defect 6a: warped=None with registration_result
    reg = {"warpedmovout": fi, "fwdtransforms": []}
    fig_none_w = render_registration_qc(fi, None, registration_result=reg, n_panels=6)
    assert len([ax for ax in fig_none_w.axes if ax.images]) == 6
    plt.close(fig_none_w)

    # Defect 6b: str fixed path with registration_result and detJ=None auto-computes detJ
    fi_path = str(tmp_path / "fixed.nii.gz")
    ants.image_write(fi, fi_path)
    warp = ants.from_numpy(
        np.zeros((*fi.shape, 3), dtype=np.float32),
        origin=fi.origin,
        spacing=fi.spacing,
        direction=fi.direction,
        has_components=True,
    )
    reg_warp = {"warpedmovout": fi, "fwdtransforms": [warp]}
    fig_str = render_registration_qc(fi_path, None, registration_result=reg_warp, detJ=None, n_panels=6)
    jac_titles = [ax.get_title() for ax in fig_str.axes if ax.images and "Jacobian" in ax.get_title()]
    assert len(jac_titles) == 1
    assert "Unavailable" not in jac_titles[0]
    plt.close(fig_str)


def test_regression_reg_none_warp_ice_and_channels_first_tensor_ice():
    """Defect 4 & Defect 5: reg=None with warp+inv_err_map computes 3D ICE; channels-first 2D/3D tensor ICE norms."""
    # Defect 4: reg=None, detJ=None, warp=warp, inv_err_map=err_img
    fi = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))
    warp = ants.from_numpy(np.zeros((16, 16, 16, 3), dtype=np.float32), has_components=True)
    err_arr = np.zeros((16, 16, 16), dtype=np.float32)
    err_arr[4:12, 4:12, 8] = 0.5
    err_arr[4:12, 4:12, 14] = 4.0
    err_img = fi.new_image_like(err_arr)

    fig = render_registration_qc(
        fixed=fi, warped=fi, warp=warp, detJ=None, inv_err_map=err_img, n_panels=8, slice_idx=8
    )
    ice_title = [ax.get_title() for ax in fig.axes if "Inverse Identity Error" in ax.get_title()][0]
    assert "Max: 4.0000mm | Mean: 2.2500mm | p95: 4.0000mm" in ice_title
    plt.close(fig)

    # Defect 5a: 2D channels-first (2, 32, 32)
    f_2d = np.ones((32, 32), dtype=np.float32)
    err_2d = np.zeros((2, 32, 32), dtype=np.float32)
    err_2d[0, 8:24, 8:24] = 0.6
    err_2d[1, 8:24, 8:24] = 0.8
    fig_2d = render_registration_qc(fixed=f_2d, warped=f_2d, inv_err_map=err_2d, n_panels=8)
    t_2d = [ax.get_title() for ax in fig_2d.axes if "Inverse Identity Error" in ax.get_title()][0]
    assert "Max: 1.0000mm | Mean: 1.0000mm | p95: 1.0000mm" in t_2d
    plt.close(fig_2d)

    # Defect 5b: 3D channels-first tensor (1, 3, 12, 16, 20)
    t_f = torch.ones((1, 1, 12, 16, 20), dtype=torch.float32)
    t_err = torch.zeros((1, 3, 12, 16, 20), dtype=torch.float32)
    t_err[0, 0, 2:10, 4:12, 5:15] = 0.6
    t_err[0, 1, 2:10, 4:12, 5:15] = 0.8
    fig_3d = render_registration_qc(
        fixed=t_f,
        warped={"warpedmovout": t_f, "inverse_identity_errors": {"composed": {"error_map": t_err}}},
        n_panels=8,
    )
    t_3d = [ax.get_title() for ax in fig_3d.axes if "Inverse Identity Error" in ax.get_title()][0]
    assert "Max: 1.0000mm | Mean: 1.0000mm | p95: 1.0000mm" in t_3d
    plt.close(fig_3d)


def test_unregistered_shifted_scanner_z_moving_slices_striatum_not_vertex():
    """Unregistered moving with shifted scanner Z (breacher__sub-0509 mode) slices own striatum."""
    fi_arr = _make_striatal_phantom(shape=(24, 24, 120), z_start=20, z_end=100)
    fi = ants.from_numpy(
        fi_arr,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )
    # Shift moving scanner origin so fixed's striatal z_LPS (+63mm) maps to z=130 (parietal vertex)
    mi_arr = _make_striatal_phantom(shape=(24, 24, 160), z_start=40, z_end=140)
    mi = ants.from_numpy(
        mi_arr,
        origin=(0.0, 0.0, -67.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )

    # Default render_registration_qc (match_moving_lps=False) uses moving's own 0.54 striatal span
    res_unreg = resolve_lps_slice_pair(fi, moving=mi, plane="axial", match_physical_lps=False)
    assert res_unreg.fixed_slice_idx == 63
    assert res_unreg.moving_slice_idx == 93
    assert res_unreg.moving_mapped_from_lps is False

    # Explicit match_physical_lps=True still maps fixed's z_LPS into moving's shifted vertex (Z=130)
    res_lps = resolve_lps_slice_pair(fi, moving=mi, plane="axial", match_physical_lps=True)
    assert res_lps.moving_slice_idx == 130
    assert res_lps.moving_mapped_from_lps is True

    fig_default = render_registration_qc(fixed=fi, warped=fi, moving=mi, n_panels=6)
    img_axes = [ax for ax in fig_default.axes if ax.images]
    mov_ax = [ax for ax in img_axes if ax.get_title().startswith("Moving (")][0]
    assert mov_ax.get_title() == "Moving (Z=93)"
    assert "LPS" not in mov_ax.get_title()
    assert float(np.max(mov_ax.images[0].get_array())) >= 195.0
    plt.close(fig_default)

    fig_lps = render_registration_qc(fixed=fi, warped=fi, moving=mi, n_panels=6, match_moving_lps=True)
    mov_ax_lps = [ax for ax in fig_lps.axes if ax.images and ax.get_title().startswith("Moving (")][0]
    assert mov_ax_lps.get_title() == "Moving (Z=130, LPS +63.0mm)"
    assert float(np.max(mov_ax_lps.images[0].get_array())) < 150.0
    plt.close(fig_lps)


def test_unregistered_explicit_slice_idx_across_planes_and_clamping():
    """Explicit slice_idx maps proportionally via content-span across axial/coronal/sagittal and clamps."""
    fi_arr = np.zeros((30, 36, 50), dtype=np.float32)
    fi_arr[5:25, 6:30, 10:40] = 100.0
    fi = ants.from_numpy(
        fi_arr,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )

    mi_arr = np.zeros((40, 48, 80), dtype=np.float32)
    mi_arr[8:36, 10:42, 20:70] = 100.0
    mi = ants.from_numpy(
        mi_arr,
        origin=(20.0, -30.0, -100.0),
        spacing=(1.2, 1.1, 1.5),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )

    specs = [
        ("sagittal", 0, "X", (5, 24), (8, 35)),
        ("coronal", 1, "Y", (6, 29), (10, 41)),
        ("axial", 2, "Z", (10, 39), (20, 69)),
    ]
    for plane_name, axis, label, (f_min, f_max), (m_min, m_max) in specs:
        # Interior endpoints & fractional interpolation
        r_min = resolve_lps_slice_pair(fi, moving=mi, plane=plane_name, slice_idx=f_min, match_physical_lps=False)
        assert r_min.moving_slice_idx == m_min
        assert r_min.moving_mapped_from_lps is False

        r_max = resolve_lps_slice_pair(fi, moving=mi, plane=plane_name, slice_idx=f_max, match_physical_lps=False)
        assert r_max.moving_slice_idx == m_max
        assert r_max.moving_mapped_from_lps is False

        f_quarter = f_min + round(0.25 * (f_max - f_min))
        frac = (f_quarter - f_min) / float(f_max - f_min)
        expected_mov = m_min + round(frac * (m_max - m_min))
        r_q = resolve_lps_slice_pair(fi, moving=mi, plane=plane_name, slice_idx=f_quarter, match_physical_lps=False)
        assert r_q.moving_slice_idx == expected_mov
        assert r_q.moving_mapped_from_lps is False

        # Out-of-foreground clamping (low and high)
        r_low = resolve_lps_slice_pair(fi, moving=mi, plane=plane_name, slice_idx=0, match_physical_lps=False)
        assert r_low.moving_slice_idx == m_min
        r_high = resolve_lps_slice_pair(
            fi, moving=mi, plane=plane_name, slice_idx=fi.shape[axis] - 1, match_physical_lps=False
        )
        assert r_high.moving_slice_idx == m_max

        fig = render_registration_qc(fixed=fi, warped=fi, moving=mi, slice_axis=axis, slice_idx=f_quarter, n_panels=6)
        mov_ax = [ax for ax in fig.axes if ax.images and ax.get_title().startswith("Moving (")][0]
        assert mov_ax.get_title() == f"Moving ({label}={expected_mov})"
        assert "LPS" not in mov_ax.get_title()
        plt.close(fig)

    # Single-slice foreground in fixed -> maps to 0.50 of moving's foreground span
    fi_single_arr = np.zeros((20, 20, 40), dtype=np.float32)
    fi_single_arr[5:15, 5:15, 18] = 100.0
    fi_single = ants.from_numpy(fi_single_arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    r_single = resolve_lps_slice_pair(
        fi_single, moving=mi, plane="axial", slice_idx=18, match_physical_lps=False
    )
    assert r_single.moving_slice_idx == 20 + round(0.50 * (69 - 20))


@pytest.mark.parametrize("n_panels", [8, 4])
@pytest.mark.parametrize(
    ("fill_val", "explicit_max", "expected_vmax"),
    [
        (1.25, None, 1.25),
        (0.20, None, 0.5),
        (0.0, None, 0.5),
        (float("nan"), None, 0.5),
        (0.20, float("nan"), 0.5),
        (float("inf"), None, 0.5),
        (0.20, float("inf"), 0.5),
        (4.0, None, 4.0),
    ],
)
def test_ice_vmax_autoscaling_8panel_and_4panel(n_panels, fill_val, explicit_max, expected_vmax):
    """ICE panel colormap auto-scales vmax to max(0.5, max_err) with finite/zero guards."""
    fi = ants.from_numpy(
        np.ones((16, 16, 16), dtype=np.float32),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag([-1.0, -1.0, 1.0]),
    )
    warp = ants.from_numpy(
        np.zeros((16, 16, 16, 3), dtype=np.float32),
        origin=fi.origin,
        spacing=fi.spacing,
        direction=fi.direction,
        has_components=True,
    )
    err_arr = np.zeros((16, 16, 16), dtype=np.float32)
    err_arr[4:12, 4:12, 6:10] = fill_val
    err_img = fi.new_image_like(err_arr)

    fig = render_registration_qc(
        fixed=fi,
        warped=fi,
        warp=warp,
        detJ=fi,
        inv_err_map=err_img,
        inv_err_max=explicit_max,
        n_panels=n_panels,
        slice_idx=8,
    )
    ice_ax = [ax for ax in fig.axes if ax.images and "Inverse Identity Error" in ax.get_title()][0]
    im_err = ice_ax.images[-1]
    assert im_err.cmap.name == "inferno"
    assert im_err.get_clim() == pytest.approx((0.0, expected_vmax))
    plt.close(fig)
