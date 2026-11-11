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

