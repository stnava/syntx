"""
syntx.liouville_determinant: the per-method Jacobian determinant used as the folding measure.
Where the map is smooth the finite-difference Jacobian of the exported warp is reliable, so the
two must agree there; greedy is defined as that finite difference.
"""

import numpy as np
import pytest
import ants

import syntx


@pytest.fixture(scope="module")
def pair():
    fi = ants.image_read(ants.get_ants_data("r16")).resample_image((2, 2), 0, 0)
    mi = ants.image_read(ants.get_ants_data("r64")).resample_image((2, 2), 0, 0)
    return fi, mi


def _run(method, fi, mi):
    fn = getattr(syntx, method)
    kw = dict(fixed=fi, moving=mi, device="cpu", verbose=False)
    kw["initial_transform"] = False if method == "greedy" else "identity"
    kw["reg_iterations"] = [20, 10, 5, 0] if method == "syngs" else [20, 10, 5]
    return fn(**kw)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("tvf", marks=pytest.mark.slow),
        pytest.param("syngs", marks=pytest.mark.slow),
        pytest.param("syn", marks=pytest.mark.slow),
        "greedy",
    ],
)
def test_determinant_agrees_with_finite_differences_where_smooth(method, pair):
    fi, mi = pair
    r = _run(method, fi, mi)
    det, info = syntx.liouville_determinant(r, fi, return_details=True)
    assert info["method"] == method
    warp = next(x for x in r["fwdtransforms"] if x.endswith(".nii.gz"))
    fd = ants.create_jacobian_determinant_image(fi, warp).numpy()
    d = det.numpy()
    assert d.shape == fd.shape
    if method == "greedy":
        assert np.allclose(d, fd)
        return
    ok = (fd > 0.5) & (fd < 2.0) & (ants.get_mask(fi).numpy() > 0)
    assert ok.sum() > 100
    assert abs(np.median(d[ok] / fd[ok]) - 1.0) < 0.02, (method, np.median(d[ok] / fd[ok]))


def test_method_is_detected_and_validated(pair):
    fi, mi = pair
    r = syntx.tvf(fixed=fi, moving=mi, initial_transform="identity", reg_iterations=[1], n_time_steps=2, device="cpu", verbose=False)
    assert syntx.liouville_determinant(r, fi, return_details=True)[1]["method"] == "tvf"
    with pytest.raises(ValueError, match="method must be one of"):
        syntx.liouville_determinant(r, fi, method="ants")
    s = syntx.determinant_summary(syntx.liouville_determinant(r, fi), fi)
    assert set(s) == {"min", "max", "mean", "std", "folding_pct"} and s["min"] > 0


def test_step_log_det_explicit_3d_matches_linalg_det_and_mps_large_grid():
    import torch
    from syntx.liouville import step_log_det

    torch.manual_seed(0)
    grad_v = 0.25 * torch.randn(2, 8, 8, 8, 3, 3, dtype=torch.float32)
    dt = 0.125
    ld, neg = step_log_det(grad_v, dt)

    ref_d = torch.linalg.det(torch.eye(3) + dt * grad_v)
    ref_ld = torch.log(ref_d.abs().clamp(min=1e-30))
    ref_neg = (ref_d <= 0).to(grad_v.dtype)
    assert torch.allclose(ld, ref_ld, atol=1e-5, rtol=1e-5)
    assert torch.equal(neg, ref_neg)

    if torch.backends.mps.is_available():
        # Full Mindboggle 3-D volume (12.58M 3x3 Jacobians) previously aborted MPS
        # in torch.linalg.det with IOGPUDeviceShmem allocation failure.
        g_mps = torch.zeros((1, 256, 256, 192, 3, 3), device="mps", dtype=torch.float32)
        ld_mps, neg_mps = step_log_det(g_mps, 0.125)
        torch.mps.synchronize()
        assert float(ld_mps.abs().max().item()) < 1e-6
        assert int(neg_mps.sum().item()) == 0


def test_syn_liouville_boundary_out_of_fov_pullback_no_false_zero_folds():
    """Out-of-FOV boundary pullbacks (y = x + w_l2r_inv outside box) must not emit -0.0 folds."""
    import glob
    import os
    import tempfile
    import torch

    shape_ants = (24, 26, 20)
    spacing = (1.2, 1.2, 2.0)
    origin = (-10.0, 5.0, 2.0)
    fi = ants.from_numpy(np.ones(shape_ants, dtype=np.float32), origin=origin, spacing=spacing)

    sp_t = tuple(reversed(shape_ants))  # (20, 26, 24)
    # Strictly diffeomorphic constant shift (+3.5 mm in Z, +2.0 mm in X): det = 1.0 everywhere,
    # while boundary slices pull back outside the midpoint bounding box.
    w_l2r = torch.zeros((1, *sp_t, 3), dtype=torch.float32)
    w_l2r[..., 0] = -3.5
    w_l2r[..., 2] = -2.0
    w_l2r_inv = -w_l2r.clone()
    w_r2l = torch.zeros_like(w_l2r)

    class _MockSyN:
        pass

    m = _MockSyN()
    m.__class__.__name__ = "SyNTo"
    m.spacing = spacing
    m.origin = origin
    m.direction = np.eye(3)
    m.midpoint_warp_l2r = w_l2r
    m.midpoint_warp_r2l = w_r2l
    m.midpoint_warp_l2r_inv = w_l2r_inv

    tmp_dir = tempfile.gettempdir()
    before_leaks = set(glob.glob(os.path.join(tmp_dir, "*_l2r_inv_Warp.nii.gz")))
    det_img = syntx.liouville_determinant({"model": m, "provenance": {"algorithm": "syntx.syn"}}, fi)
    after_leaks = set(glob.glob(os.path.join(tmp_dir, "*_l2r_inv_Warp.nii.gz")))
    assert after_leaks == before_leaks

    summary = syntx.determinant_summary(det_img, fi)
    assert summary["folding_pct"] == 0.0
    assert abs(summary["min"] - 1.0) < 1e-4
    assert abs(summary["max"] - 1.0) < 1e-4


def test_liouville_mismatched_native_fixed_vs_working_model_grid():
    """Calling liouville_determinant with a finer native fixed image must use model geometry first."""
    from syntx.pyramid import cap_ants_image

    # Build a synthetic 3-D smooth Gaussian anatomy at 0.8 mm spacing and cap to 1.6 mm working grid
    z, y, x = np.mgrid[0:32, 0:48, 0:48].astype(np.float32)
    arr = np.exp(-((x - 24.0) ** 2 + (y - 24.0) ** 2 + (z - 16.0) ** 2) / 120.0).astype(np.float32)
    fi_native = ants.from_numpy(arr, origin=(100.0, 80.0, -20.0), spacing=(0.8, 0.8, 1.5))
    mi_native = ants.from_numpy(np.roll(arr, 2, axis=1), origin=(100.0, 80.0, -20.0), spacing=(0.8, 0.8, 1.5))

    fi_wrk = cap_ants_image(fi_native, max_resolution_mm=1.6, policy="deformable")
    mi_wrk = cap_ants_image(mi_native, max_resolution_mm=1.6, policy="deformable")
    assert fi_wrk.shape != fi_native.shape

    for method in ("syn", "tvf"):
        fn = getattr(syntx, method)
        kw = dict(fixed=fi_wrk, moving=mi_wrk, initial_transform="identity", reg_iterations=[4, 2], device="cpu", verbose=False)
        if method == "tvf":
            kw["n_time_steps"] = 2
        res = fn(**kw)

        s_wrk = syntx.determinant_summary(syntx.liouville_determinant(res, fi_wrk), fi_wrk)
        det_nat = syntx.liouville_determinant(res, fi_native)
        assert det_nat.shape == fi_native.shape
        s_nat = syntx.determinant_summary(det_nat, fi_native)
        assert s_wrk["folding_pct"] == 0.0
        assert s_nat["folding_pct"] == 0.0
        assert abs(s_nat["mean"] - s_wrk["mean"]) < 0.05


def test_syn_liouville_genuine_half_fold_detected_and_finite_near_zero():
    """Genuine half-field folds must remain <= 0 and finite (never -inf or inverted ratio spikes)."""
    import torch

    shape_ants = (28, 28, 28)
    fi = ants.from_numpy(np.ones(shape_ants, dtype=np.float32), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    z, y, x = torch.meshgrid(
        torch.arange(28, dtype=torch.float32),
        torch.arange(28, dtype=torch.float32),
        torch.arange(28, dtype=torch.float32),
        indexing="ij",
    )
    # Localised fold in X around x=14: du_x/dx = -1.6 * exp(...) * (1 - 2*(x-14)^2/9) -> min det ~ -0.6,
    # passing continuously through det_l2r == 0 on the ring around x=14.
    r2 = ((x - 14.0) ** 2 + (y - 14.0) ** 2 + (z - 14.0) ** 2) / 9.0
    ux = -1.6 * (x - 14.0) * torch.exp(-r2)
    w_l2r = torch.zeros((1, 28, 28, 28, 3), dtype=torch.float32)
    w_l2r[0, ..., 2] = ux  # tensor component 2 = dx
    w_l2r_inv = -w_l2r.clone()
    w_r2l = torch.zeros_like(w_l2r)

    class _MockSyN:
        pass

    m = _MockSyN()
    m.spacing = (1.0, 1.0, 1.0)
    m.origin = (0.0, 0.0, 0.0)
    m.direction = np.eye(3)
    m.midpoint_warp_l2r = w_l2r
    m.midpoint_warp_r2l = w_r2l
    m.midpoint_warp_l2r_inv = w_l2r_inv

    det_img = syntx.liouville_determinant({"model": m, "provenance": {"algorithm": "syntx.syn"}}, fi)
    d = det_img.numpy()
    assert np.isfinite(d).all()
    summary = syntx.determinant_summary(det_img, fi)
    assert summary["folding_pct"] > 0.0
    assert -2.5 < summary["min"] < -0.1


def test_syn_jax_liouville_backend_parity(pair):
    """syntx.syn(backend='jax') returns SyNJAX with PhysicalWarpArray half-fields; liouville must support it."""
    fi, mi = pair
    res = syntx.syn(
        fixed=fi,
        moving=mi,
        backend="jax",
        initial_transform="identity",
        reg_iterations=[6, 3],
        verbose=False,
    )
    det_img, info = syntx.liouville_determinant(res, fi, return_details=True)
    assert info["method"] == "syn"
    s = syntx.determinant_summary(det_img, fi)
    assert s["folding_pct"] == 0.0
    assert s["min"] > 0.0


def test_step_log_det_at_norm_matches_velocity_gradient_2d_and_3d_oblique():
    """Verify fused step_log_det_at_norm matches velocity_gradient_at + step_log_det in 2-D and 3-D
    across oblique, anisotropic physical coordinate frames (including folding voxels)."""
    import torch
    import torch.nn.functional as F
    from syntx.liouville import step_log_det, step_log_det_at_norm, velocity_gradient_at
    from syntx.spatial import get_physical_grid_torch, reverse_metadata

    torch.manual_seed(42)

    # 3-D oblique anisotropic grid
    shape_3d = (10, 12, 14)
    spacing_3d = [1.8, 1.1, 0.9]
    origin_3d = [-12.0, 15.0, 4.0]
    rx, ry, rz = np.deg2rad([14.0, -9.0, 18.0])
    d3 = (
        np.array([[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]])
        @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]])
    )
    sp_rev, or_rev, dir_rev = reverse_metadata(spacing_3d, origin_3d, d3)
    pts_3d = get_physical_grid_torch(shape_3d, spacing_3d, origin_3d, d3, device=torch.zeros(1).device, dtype=torch.float32)
    pts_3d = pts_3d + 0.4 * torch.randn_like(pts_3d)

    shape_t = torch.tensor(list(shape_3d), dtype=torch.float32)
    spacing_t = torch.tensor(sp_rev, dtype=torch.float32)
    origin_t = torch.tensor(or_rev, dtype=torch.float32)
    direction_t = torch.tensor(dir_rev, dtype=torch.float32)
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    m_mat = direction_t * scale_t.unsqueeze(0)
    b_vec = -(origin_t @ m_mat) - 1.0
    m_norm, b_norm = torch.flip(m_mat, dims=[1]), torch.flip(b_vec, dims=[0])

    # Include strong gradients so some voxels fold (det <= 0) as well as stay positive
    v_cf_3d = 3.5 * torch.randn((1, 3, *shape_3d), dtype=torch.float32)
    h_3d = 0.5 * float(spacing_t.min())
    dt = 0.2

    def _sample_3d(p: torch.Tensor) -> torch.Tensor:
        q = (p.reshape(-1, 3) @ m_norm + b_norm).view(p.shape)
        return torch.movedim(
            F.grid_sample(v_cf_3d, q, mode="bilinear", padding_mode="border", align_corners=True), 1, -1
        )

    ref_ld_3d, ref_neg_3d = step_log_det(velocity_gradient_at(_sample_3d, pts_3d, h_3d), dt)
    phi_norm_3d = (pts_3d.reshape(-1, 3) @ m_norm + b_norm).view(pts_3d.shape)
    act_ld_3d, act_neg_3d = step_log_det_at_norm(v_cf_3d, phi_norm_3d, m_norm, h_3d, dt)

    assert int(ref_neg_3d.sum().item()) > 0  # ensures folding branch is also tested
    assert torch.equal(act_neg_3d, ref_neg_3d)
    ref_det_3d = torch.where(ref_neg_3d > 0, -torch.exp(ref_ld_3d), torch.exp(ref_ld_3d))
    act_det_3d = torch.where(act_neg_3d > 0, -torch.exp(act_ld_3d), torch.exp(act_ld_3d))
    assert torch.allclose(act_det_3d, ref_det_3d, atol=2.5e-5, rtol=2.5e-5)
    nonsingular_3d = ref_det_3d.abs() >= 0.1
    assert torch.allclose(act_ld_3d[nonsingular_3d], ref_ld_3d[nonsingular_3d], atol=5e-5, rtol=5e-5)

    # 2-D oblique anisotropic grid
    shape_2d = (14, 16)
    spacing_2d = [1.4, 0.85]
    origin_2d = [-5.0, 7.0]
    th = np.deg2rad(22.0)
    d2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sp2_rev, or2_rev, dir2_rev = reverse_metadata(spacing_2d, origin_2d, d2)
    pts_2d = get_physical_grid_torch(shape_2d, spacing_2d, origin_2d, d2, device=torch.zeros(1).device, dtype=torch.float32)
    shape2_t = torch.tensor(list(shape_2d), dtype=torch.float32)
    spacing2_t = torch.tensor(sp2_rev, dtype=torch.float32)
    origin2_t = torch.tensor(or2_rev, dtype=torch.float32)
    dir2_t = torch.tensor(dir2_rev, dtype=torch.float32)
    m2 = dir2_t * (2.0 / (spacing2_t * (shape2_t - 1.0))).unsqueeze(0)
    b2 = -(origin2_t @ m2) - 1.0
    m2_norm, b2_norm = torch.flip(m2, dims=[1]), torch.flip(b2, dims=[0])

    v_cf_2d = 2.5 * torch.randn((1, 2, *shape_2d), dtype=torch.float32)
    h_2d = 0.5 * float(spacing2_t.min())

    def _sample_2d(p: torch.Tensor) -> torch.Tensor:
        q = (p.reshape(-1, 2) @ m2_norm + b2_norm).view(p.shape)
        return torch.movedim(
            F.grid_sample(v_cf_2d, q, mode="bilinear", padding_mode="border", align_corners=True), 1, -1
        )

    ref_ld_2d, ref_neg_2d = step_log_det(velocity_gradient_at(_sample_2d, pts_2d, h_2d), dt)
    phi_norm_2d = (pts_2d.reshape(-1, 2) @ m2_norm + b2_norm).view(pts_2d.shape)
    act_ld_2d, act_neg_2d = step_log_det_at_norm(v_cf_2d, phi_norm_2d, m2_norm, h_2d, dt)

    assert torch.equal(act_neg_2d, ref_neg_2d)
    ref_det_2d = torch.where(ref_neg_2d > 0, -torch.exp(ref_ld_2d), torch.exp(ref_ld_2d))
    act_det_2d = torch.where(act_neg_2d > 0, -torch.exp(act_ld_2d), torch.exp(act_ld_2d))
    assert torch.allclose(act_det_2d, ref_det_2d, atol=2.5e-5, rtol=2.5e-5)
    nonsingular_2d = ref_det_2d.abs() >= 0.1
    assert torch.allclose(act_ld_2d[nonsingular_2d], ref_ld_2d[nonsingular_2d], atol=5e-5, rtol=5e-5)



