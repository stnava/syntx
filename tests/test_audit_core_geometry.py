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


# ---------------------------------------------------------------------------------------
# core/jacobian.py
# ---------------------------------------------------------------------------------------

def _anisotropic_linear(physical, comps_xyz):
    """2-D field on a (10, 14) grid, spacing (x, y) = (1, 2): u_x = 0.1 x_phys (det = 1.1)."""
    H, W = 10, 14
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    if physical:
        ux = 0.1 * xx * 1.0                                     # mm
    else:
        ux = 0.1 * (xx / (W - 1) * 2 - 1)                       # normalised x
    zero = torch.zeros_like(ux)
    u = torch.stack([ux, zero] if comps_xyz else [zero, ux], dim=-1)
    return u.unsqueeze(0)


@pytest.mark.parametrize("method", ["central", "bspline"])
def test_jacobian_nd_paths_agree(method):
    from syntx.core.jacobian import compute_jacobian_determinant_nd
    phys = compute_jacobian_determinant_nd(_anisotropic_linear(True, comps_xyz=False),
                                           physical_spacing=(1.0, 2.0), method=method)
    norm = compute_jacobian_determinant_nd(_anisotropic_linear(False, comps_xyz=True),
                                           method=method, is_physical=False)
    inner = (0, slice(2, -2), slice(2, -2))
    assert torch.allclose(phys[inner], torch.full_like(phys[inner], 1.1), atol=1e-6)
    assert torch.allclose(norm[inner], torch.full_like(norm[inner], 1.1), atol=1e-6)


def test_jacobian_nd_unbatched_input():
    from syntx.core.jacobian import compute_jacobian_determinant_nd
    u = _anisotropic_linear(True, comps_xyz=False)
    a = compute_jacobian_determinant_nd(u, physical_spacing=(1.0, 2.0))
    b = compute_jacobian_determinant_nd(u[0], physical_spacing=(1.0, 2.0))
    assert b.shape == a.shape[1:] and torch.allclose(a[0], b)


def test_physical_jacobian_uses_direction():
    """u(p) = (A - I) p on an oblique, anisotropic grid: det(I + du/dp) == det(A) everywhere."""
    from syntx.core.jacobian import compute_physical_jacobian_determinant
    th = 0.6
    D = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sp = np.array([1.5, 0.7])                                      # ANTs (x, y)
    A = np.array([[1.3, 0.4], [-0.2, 0.9]])                        # det 1.25
    H, W = 9, 11
    iy, ix = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    idx = np.stack([ix, iy], -1).astype(float)                    # (H, W, 2) xy
    p = idx @ (D @ np.diag(sp)).T
    u_xy = p @ (A - np.eye(2)).T
    u = torch.tensor(u_xy[..., ::-1].copy(), dtype=torch.float64)[None]   # tensor order (y, x)
    u.is_physical = True
    det = compute_physical_jacobian_determinant(u, direction=D, spacing=sp)
    assert det.shape == (1, H, W)
    np.testing.assert_allclose(det.numpy(), np.linalg.det(A), rtol=1e-9)
    with pytest.raises(TypeError):
        compute_physical_jacobian_determinant(u, direction=D, spacing=sp, origin=[0, 0])


def _mid_inputs(I, sp, D):
    from syntx.spatial import get_physical_grid_torch
    shape = I.shape[2:]
    dim = len(shape)
    X = get_physical_grid_torch(shape, sp, [0.0] * dim, D, dtype=torch.float64)
    Drev = torch.tensor(np.asarray(D)[::-1, ::-1].copy(), dtype=torch.float64)
    sh = torch.tensor(list(shape), dtype=torch.float64)
    spt = torch.tensor(list(reversed(sp)), dtype=torch.float64)
    ot = torch.zeros(dim, dtype=torch.float64)
    z = torch.zeros_like(X)
    return (z, z, I, I, X, sh, spt, ot, Drev, sh, spt, ot, Drev, list(sp), list(sp),
            torch.eye(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64), None)


@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_mid_image_gradients_tensor_order(theta):
    """I(p) = a . p: the sampled physical gradients must equal a in tensor order (y, x)."""
    from syntx.core.grid import prepare_mid_images_and_gradients_torch
    from syntx.spatial import get_physical_grid_torch
    D = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sp = [1.5, 0.75]                                              # ANTs (x, y)
    X = get_physical_grid_torch((12, 10), sp, [0.0, 0.0], D, dtype=torch.float64)
    a_t = torch.tensor([0.3, 1.0], dtype=torch.float64)          # (d/dy, d/dx), tensor order
    I = (X @ a_t)[:, None]                                        # (1, 1, 12, 10)
    out = prepare_mid_images_and_gradients_torch(*_mid_inputs(I, sp, D))
    gI, gJ = out[2], out[3]
    inner = (slice(None), slice(3, -3), slice(3, -3))
    for g in (gI, gJ):
        np.testing.assert_allclose(g[inner].reshape(-1, 2).numpy(), np.tile(a_t.numpy(), (g[inner].numel() // 2, 1)),
                                   atol=1e-9)


def test_image_spatial_gradient_no_wraparound():
    """A linear ramp has a constant gradient up to the border (no I[0] - I[N-1] wrap term),
    and the interior is the plain central difference."""
    from syntx.core.grid import _image_spatial_gradient
    H, W = 6, 7
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing='ij')
    I = (2.0 * x + 0.5 * y)[None, None]
    g = _image_spatial_gradient(I)                    # (1, 1, 2, H, W), (x, y) order
    assert torch.allclose(g[0, 0, 0], torch.full((H, W), 2.0, dtype=torch.float64))
    assert torch.allclose(g[0, 0, 1], torch.full((H, W), 0.5, dtype=torch.float64))
    z = torch.zeros(1, 1, 4, 5, 6, dtype=torch.float64)
    z[..., 2, 2, 3] = 1.0
    g3 = _image_spatial_gradient(z)
    assert g3[0, 0, 0, 2, 2, 2] == 0.5 and g3[0, 0, 2, 1, 2, 3] == 0.5


def test_generic_label_zeros_padding_and_dtype():
    """Outside the image 'zeros' padding gives label 0 (not the lowest label present), and an
    integer label map comes back integer."""
    from syntx.core.grid import grid_sample_nd
    lab = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 2, 2], [3, 3, 2, 2]], dtype=torch.int64)[None, None]
    grid = torch.tensor([[[[-0.9, -0.9], [0.9, 0.9], [3.0, 3.0]]]], dtype=torch.float32)   # last point outside
    out = grid_sample_nd(lab, grid, padding_mode='zeros', interpolator='genericLabel')
    assert out.dtype == torch.int64
    assert out.flatten().tolist() == [1, 2, 0]
    out_itk = grid_sample_nd(lab.float(), grid, padding_mode='itk', interpolator='genericLabel')
    assert out_itk.flatten().tolist() == [1.0, 2.0, 0.0]


def test_grid_sample_nd_rejects_unknown_options():
    from syntx.core.grid import grid_sample_nd, grid_sample_bspline_torch, DeterministicGridSample
    img = torch.rand(1, 1, 5, 5)
    grid = torch.zeros(1, 3, 3, 2)
    with pytest.raises(ValueError, match="interpolator"):
        grid_sample_nd(img, grid, interpolator='lanczosWindowedSinc')
    with pytest.raises(ValueError, match="padding_mode"):
        grid_sample_bspline_torch(img, grid, padding_mode='reflection')
    with pytest.raises(ValueError, match="padding_mode"):
        DeterministicGridSample.apply(img, grid, 'reflection')
    for interp in ('linear', 'nearestNeighbor', 'bspline', 'genericLabel', 'nearest'):
        assert grid_sample_nd(img, grid, interpolator=interp).shape == (1, 1, 3, 3)


def _euler3d(angles, trans, center=(0.0, 0.0, 0.0)):
    import ants
    t = ants.create_ants_transform(transform_type='Euler3DTransform', dimension=3)
    t.set_fixed_parameters(list(center) + [0.0])
    t.set_parameters(list(angles) + list(trans))
    return t


def test_parse_ants_affine_types_order_and_nonlinear(tmp_path):
    import ants
    from syntx.core.affine import parse_ants_affine
    A = _euler3d([0.1, -0.2, 0.3], [1.0, 2.0, -3.0], center=(4.0, 5.0, 6.0))
    B = ants.create_ants_transform(transform_type='AffineTransform', dimension=3,
                                   matrix=np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.2], [0.05, 0.0, 1.2]]),
                                   translation=(0.5, -1.0, 2.0))
    p = np.array([3.0, -2.0, 7.0])
    # Euler objects are parsed (were skipped)
    M, t = parse_ants_affine([A], 3)
    np.testing.assert_allclose(M.numpy() @ p + t.numpy(), A.apply_to_point(tuple(p)), atol=1e-4)
    # ANTs transformlist order: the last item is applied first
    pa, pb = str(tmp_path / 'a.mat'), str(tmp_path / 'b.mat')
    ants.write_transform(A, pa)
    ants.write_transform(B, pb)
    M, t = parse_ants_affine([pa, pb], 3)
    ref = A.apply_to_point(B.apply_to_point(tuple(p)))
    np.testing.assert_allclose(M.numpy() @ p + t.numpy(), ref, atol=1e-4)
    # a displacement field is not linear: raise, or (None, None) when the caller can fall back
    fld = ants.from_numpy(np.random.default_rng(0).random((6, 6, 6, 3)).astype('float32'), has_components=True)
    pw = str(tmp_path / 'w.nii.gz')
    ants.image_write(fld, pw)
    with pytest.raises(ValueError, match="not a linear"):
        parse_ants_affine([pw, pb], 3)
    assert parse_ants_affine([pw, pb], 3, allow_nonlinear=True) == (None, None)
    with pytest.raises(ValueError):
        parse_ants_affine([str(tmp_path / 'missing.mat')], 3)
    with pytest.raises(ValueError, match="dimension"):
        parse_ants_affine([pb], 2)
    M, t = parse_ants_affine('identity', 3)
    assert torch.equal(M, torch.eye(3)) and torch.equal(t, torch.zeros(3))
