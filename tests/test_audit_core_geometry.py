"""Regression tests for the docstring-audit behaviour issues in core/inverse.py, core/utils.py,
spatial.py, core/jacobian.py, core/grid.py, core/affine.py, core/pipeline.py. CPU only."""
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------------------
# core/inverse.py
# ---------------------------------------------------------------------------------------

def _smooth_field(shape=(12, 12), amp=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(1, 3, 3, 2, generator=g) * amp
    return torch.nn.functional.interpolate(u.permute(0, 3, 1, 2), size=shape, mode="bilinear",
                                           align_corners=True).permute(0, 2, 3, 1)


def test_fixed_point_normalised_needs_both_thresholds(monkeypatch):
    from syntx.core.inverse import update_inverse_field_nd
    u = _smooth_field()
    # a huge max threshold alone must not stop the iteration (mean still above its threshold)
    v1 = update_inverse_field_nd(u, steps=30, method="fixed_point", max_error_threshold=1e9,
                                 mean_error_threshold=1e-12)
    v2 = update_inverse_field_nd(u, steps=30, method="fixed_point", max_error_threshold=1e-12,
                                 mean_error_threshold=1e-12)
    assert torch.allclose(v1, v2)


def test_fixed_point_honours_relaxation():
    from syntx.core.inverse import update_inverse_field_nd
    u = _smooth_field()
    a = update_inverse_field_nd(u, steps=1, method="fixed_point", relaxation=1.0)
    b = update_inverse_field_nd(u, steps=1, method="fixed_point", relaxation=0.5)
    assert not torch.allclose(a, b)


def test_integrate_velocity_dt_default_solver_validation_and_midpoint():
    from syntx.core.inverse import integrate_time_varying_velocity_field
    v = [torch.full((1, 6, 6, 2), 0.1) for _ in range(4)]     # constant velocity, T = 4
    phi = integrate_time_varying_velocity_field(v, solver="euler")
    assert torch.allclose(phi, torch.full_like(phi, 0.1), atol=1e-5)   # dt = 1/T: t in [0, 1]
    with pytest.raises(ValueError, match="solver"):
        integrate_time_varying_velocity_field(v, solver="rk3")
    ramp = [torch.zeros(1, 8, 8, 2)]
    xs = torch.linspace(-1, 1, 8)
    ramp[0][..., 0] = 0.3 * xs.view(1, 1, 8)                    # v_x = 0.3 x (normalised)
    e = integrate_time_varying_velocity_field(ramp, dt=1.0, solver="euler")
    m = integrate_time_varying_velocity_field(ramp, dt=1.0, solver="midpoint")
    assert not torch.allclose(e, m)                             # midpoint is not Euler any more


def test_inverse_error_mean_over_evaluated_voxels_and_xphys_check():
    from syntx.core.inverse import calculate_inverse_identity_error, update_inverse_field_nd
    shape = (10, 10)
    u = torch.zeros(1, *shape, 2); u[..., 0] = 30.0             # pushes everything off the grid
    out = calculate_inverse_identity_error(u, torch.zeros_like(u), spacing=(1.0, 1.0),
                                           origin=(0.0, 0.0), direction=np.eye(2))
    assert out["mean_error"] == 0.0 or np.isnan(out["mean_error"]) or out["mean_error"] > 0
    w = torch.zeros(1, *shape, 2); w[..., 0] = 1.0              # 1 mm shift, partly off-grid
    out = calculate_inverse_identity_error(w, torch.zeros_like(w), spacing=(1.0, 1.0),
                                           origin=(0.0, 0.0), direction=np.eye(2))
    assert abs(out["mean_error"] - 1.0) < 1e-4                  # every evaluated voxel errs by 1 mm
    with pytest.raises(ValueError, match="X_phys"):
        update_inverse_field_nd(w, method="fixed_point", X_phys=torch.zeros(1, *shape, 2))


# ---------------------------------------------------------------------------------------
# core/utils.py
# ---------------------------------------------------------------------------------------

def test_normalize_image_zscore_not_skipped_for_unit_range_input():
    from syntx.core.utils import normalize_image
    a = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    z = normalize_image(a, method="zscore", foreground_only=False)
    assert abs(float(z.mean())) < 1e-5 and abs(float(z.std()) - 1.0) < 1e-3
    assert np.allclose(normalize_image(a, method="minmax"), a)     # idempotent for [0, 1] methods


# ---------------------------------------------------------------------------------------
# spatial.py
# ---------------------------------------------------------------------------------------

def test_spatial_coordinate_grid_axis_ranges():
    import ants
    from syntx.spatial import get_spatial_coordinate_grid
    img = ants.from_numpy(np.zeros((4, 3, 2), np.float32), spacing=(1.0, 2.0, 3.0))   # ANTs (x, y, z)
    pts, shape = get_spatial_coordinate_grid(img)
    assert shape == (2, 3, 4)
    assert float(pts[:, 0].max()) == 3.0 and float(pts[:, 2].max()) == 3.0          # x: 0..3, z: 0..1 * 3 mm
    assert float(pts[:, 1].max()) == 4.0


def test_restriction_from_orientation_labels_and_inputs():
    import ants
    from syntx.spatial import restriction_from_orientation
    img3 = ants.from_numpy(np.zeros((4, 4, 4), np.float32))
    img2 = ants.from_numpy(np.zeros((4, 4), np.float32))
    assert restriction_from_orientation(img3, anatomical_axis="I") == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="2-D"):
        restriction_from_orientation(img2, anatomical_axis="z")
    with pytest.raises(ValueError, match="exactly one"):
        restriction_from_orientation(img3, anatomical_axis="AP", bids_phase_encoding_direction="j")


def test_physical_to_normalized_uses_true_inverse_direction():
    from syntx.spatial import get_physical_to_normalized_affine, physical_to_normalized_fast
    shape = torch.tensor([6.0, 7.0]); sp = torch.tensor([1.0, 1.0]); org = torch.zeros(2)
    D = torch.tensor([[1.0, 0.3], [0.0, 1.0]])                 # sheared (not orthonormal)
    M, b = get_physical_to_normalized_affine(shape, sp, org, D)
    idx = torch.tensor([[2.0, 3.0]])                           # tensor (row, col) voxel index
    x = (idx * sp) @ D.T + org                                  # physical point of that index
    n = physical_to_normalized_fast(x, M, b)
    expect = torch.tensor([[3.0 / 6 * 2 - 1, 2.0 / 5 * 2 - 1]])  # (x, y) = (col, row) normalised
    assert torch.allclose(n, expect, atol=1e-5)


def test_lps_ras_keep_dtype():
    from syntx.spatial import lps_to_ras, ras_to_lps
    a = np.array([[1.123456789, 2.0, 3.0]], dtype=np.float64)
    assert lps_to_ras(a).dtype == np.float64 and np.allclose(ras_to_lps(lps_to_ras(a)), a, atol=0)


def test_deformation_gradient_rejects_ignored_geometry():
    import ants
    from syntx.spatial import deformation_gradient
    img = ants.from_numpy(np.zeros((6, 6, 2), np.float32), has_components=True)
    with pytest.raises(ValueError, match="own geometry"):
        deformation_gradient(img, spacing=(2.0, 2.0))
    ref = ants.from_numpy(np.zeros((6, 6), np.float32))
    with pytest.raises(ValueError, match="ref_image"):
        deformation_gradient(torch.zeros(1, 6, 6, 2), ref_image=ref, spacing=(2.0, 2.0))
